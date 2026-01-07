from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

import numpy as np
import pandas as pd

try:
    import optuna
except Exception:
    optuna = None

try:
    from sklearn.model_selection import BaseCrossValidator
except Exception as e:
    raise ImportError("scikit-learn error.") from e


@dataclass(frozen=True)
class NestedCVConfig:
    horizon: int
    cost_bps: float
    outer_params: dict[str, Any]
    inner_params: dict[str, Any]
    seeds: list[int]

    task: Literal["gate_classifier", "ranker"] = "ranker"

    feature_selection: Literal["importance", "corr"] = "importance"

    optuna_n_trials: int = 25

    min_rank_train_groups: int = 10
    min_rank_train_winners: int = 25
    min_rank_train_winner_dates: int = 5

    top_n_features: int = 25
    min_feature_folds: int = 2

    inner_threshold_grid_p: list[float] | None = None
    inner_threshold_grid_u_quantiles: list[float] | None = None


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    indices: dict[str, np.ndarray]
    chosen_features: list[str]
    chosen_params: dict[str, Any]
    thresholds: dict[str, float]


@dataclass(frozen=True)
class SelectionResult:
    chosen_features: list[str]
    chosen_params: dict[str, Any]
    thresholds: dict[str, float]


def _as_numpy_index(idx: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(idx), dtype=int)
    return arr


def _mask_valid_labels(y: pd.DataFrame) -> np.ndarray:
    if "label_excess" not in y.columns:
        raise ValueError("y must contain label_excess")
    return y["label_excess"].notna().to_numpy()


def _mask_valid_target(y: pd.DataFrame, *, task: str) -> np.ndarray:
    if task == "gate_classifier":
        if "y_gate" not in y.columns:
            raise ValueError("y must contain y_gate for gate_classifier")
        return y["y_gate"].notna().to_numpy()
    if task == "ranker":
        if "rel_rank" in y.columns:
            mask = y["rel_rank"].notna()
            if "label_excess" in y.columns:
                mask = mask & y["label_excess"].notna()
            return mask.to_numpy()

        if "label_excess" not in y.columns:
            raise ValueError("y must contain rel_rank or label_excess for ranker")
        return y["label_excess"].notna().to_numpy()
    raise ValueError(f"Unknown task: {task}")


def _extract_feature_importance(model: Any, feature_names: list[str]) -> pd.Series:
    n = int(len(feature_names))
    if n == 0:
        return pd.Series(dtype=float)

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(getattr(model, "feature_importances_"))
        if imp.shape[0] == n:
            return pd.Series(imp.astype(float), index=feature_names)

    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"))
        coef = np.ravel(coef)
        if coef.shape[0] == n:
            return pd.Series(np.abs(coef).astype(float), index=feature_names)

    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            out = pd.Series(0.0, index=feature_names, dtype=float)
            for k, v in score.items():
                if k in out.index:
                    out.loc[k] = float(v)
            return out
        except Exception:
            pass

    return pd.Series(0.0, index=feature_names, dtype=float)


def _split_fixed_and_search_params(inner_params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed: dict[str, Any] = {}
    space: dict[str, Any] = {}
    for k, v in (inner_params or {}).items():
        if isinstance(v, (list, tuple, dict)):
            space[k] = v
        else:
            fixed[k] = v
    return fixed, space


def _suggest_optuna_param(trial: Any, name: str, spec: Any) -> Any:
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)

    if isinstance(spec, tuple):
        if len(spec) == 2:
            low, high = spec
            if isinstance(low, int) and isinstance(high, int):
                return trial.suggest_int(name, int(low), int(high))
            return trial.suggest_float(name, float(low), float(high))

        if len(spec) == 3:
            low, high, mode = spec
            mode_s = str(mode).lower()
            if mode_s in {"log", "loguniform"}:
                return trial.suggest_float(name, float(low), float(high), log=True)
            if mode_s in {"int", "int_linear"}:
                return trial.suggest_int(name, int(low), int(high))
            if mode_s in {"int_log"}:
                return trial.suggest_int(name, int(low), int(high), log=True)
            return trial.suggest_float(name, float(low), float(high))

    if isinstance(spec, dict):
        low = spec.get("low")
        high = spec.get("high")
        kind = str(spec.get("type", spec.get("kind", "float"))).lower()
        log = bool(spec.get("log", False))
        choices = spec.get("choices")
        if choices is not None:
            return trial.suggest_categorical(name, list(choices))
        if kind.startswith("int"):
            return trial.suggest_int(name, int(low), int(high), log=log)
        return trial.suggest_float(name, float(low), float(high), log=log)

    raise ValueError(f"Unsupported Optuna spec for {name}: {spec!r}")


def _rank_spearman_by_group(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0

    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "s", "g"])
    if df.empty:
        return 0.0

    vals: list[float] = []
    for _, sub in df.groupby("g", sort=False):
        if len(sub) < 2:
            continue
        yv = sub["y"].to_numpy(dtype=float)
        sv = sub["s"].to_numpy(dtype=float)
        if np.nanstd(yv) == 0.0 or np.nanstd(sv) == 0.0:
            continue
        ry = pd.Series(yv).rank(method="average").to_numpy(dtype=float)
        rs = pd.Series(sv).rank(method="average").to_numpy(dtype=float)
        c = float(np.corrcoef(ry, rs)[0, 1])
        if not np.isfinite(c):
            continue
        vals.append(c)

    if not vals:
        return 0.0
    return float(np.mean(vals))


def tune_hyperparameters(
    *,
    config: NestedCVConfig,
    X_tr: pd.DataFrame,
    y_tr: pd.DataFrame,
    g_tr: np.ndarray,
    chosen_features: list[str],
    inner_cv: BaseCrossValidator,
    model_factory: Callable[[int, dict[str, Any]], Any],
    fit_fn: Callable[..., Any],
    predict_fn: Callable[..., np.ndarray],
) -> dict[str, Any]:
    fixed, space = _split_fixed_and_search_params(config.inner_params)
    n_trials = int(getattr(config, "optuna_n_trials", 0) or 0)
    if n_trials <= 0 or not space:
        return dict(fixed)

    if optuna is None:
        raise ImportError("Optuna is not installed but optuna_n_trials > 0 and a search space was provided.")

    seed0 = int(config.seeds[0]) if config.seeds else 0
    sampler = optuna.samplers.TPESampler(seed=seed0)
    direction = "maximize"
    study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial: Any) -> float:
        trial_params = {k: _suggest_optuna_param(trial, k, spec) for k, spec in space.items()}
        params = {**fixed, **trial_params}

        scores: list[float] = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_tr, y_tr, groups=g_tr):
            tr_idx = _as_numpy_index(inner_train_idx)
            va_idx = _as_numpy_index(inner_val_idx)

            X_i_tr = X_tr.iloc[tr_idx][chosen_features]
            y_i_tr = y_tr.iloc[tr_idx]
            g_i_tr = np.asarray(g_tr)[tr_idx]

            X_i_va = X_tr.iloc[va_idx][chosen_features]
            y_i_va = y_tr.iloc[va_idx]
            g_i_va = np.asarray(g_tr)[va_idx]

            m_tr = _mask_valid_target(y_i_tr, task=config.task)
            m_va = _mask_valid_target(y_i_va, task=config.task)

            X_i_tr = X_i_tr.iloc[m_tr]
            y_i_tr = y_i_tr.iloc[m_tr]
            g_i_tr = g_i_tr[m_tr]

            X_i_va = X_i_va.iloc[m_va]
            y_i_va = y_i_va.iloc[m_va]
            g_i_va = g_i_va[m_va]

            if X_i_tr.empty or X_i_va.empty:
                continue

            model = model_factory(seed0, dict(params))
            if config.task == "gate_classifier":
                model = fit_fn(model, X_i_tr, y_i_tr["y_gate"], groups=g_i_tr)
                pred = np.asarray(predict_fn(model, X_i_va), dtype=float)
                yt = pd.to_numeric(y_i_va["y_gate"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(pred) & np.isfinite(yt)
                if not ok.any():
                    continue
                yb = (yt[ok] >= 0.5).astype(int)
                pb = (pred[ok] >= 0.5).astype(int)
                tp = float(np.sum((pb == 1) & (yb == 1)))
                fp = float(np.sum((pb == 1) & (yb == 0)))
                fn = float(np.sum((pb == 0) & (yb == 1)))
                denom = (2 * tp + fp + fn)
                f1 = 0.0 if denom <= 0 else (2 * tp) / denom
                scores.append(float(f1))

            elif config.task == "ranker":
                Xs_tr, ys_tr, _gs_tr, group_sizes_tr = _rank_order_and_group_sizes(X=X_i_tr, y=y_i_tr, groups=g_i_tr)
                model = fit_fn(model, Xs_tr, ys_tr["rel_rank"], group_sizes=group_sizes_tr)

                Xs_va, ys_va, gs_va, _ = _rank_order_and_group_sizes(X=X_i_va, y=y_i_va, groups=g_i_va)
                pred = np.asarray(predict_fn(model, Xs_va), dtype=float)
                yt = pd.to_numeric(ys_va["rel_rank"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(pred) & np.isfinite(yt)
                if not ok.any():
                    continue
                s = _rank_spearman_by_group(yt[ok], pred[ok], np.asarray(gs_va)[ok])
                scores.append(float(s))
            else:
                raise ValueError(f"Unknown task: {config.task}")

        if not scores:
            return -1e9

        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(getattr(study, "best_params", {}) or {})
    return {**fixed, **best}


def _compute_rank_thresholds_from_train(
    *,
    label_excess_train: pd.Series,
    cost: float,
    q60: float = 0.60,
    q80: float = 0.80,
) -> tuple[float, float]:
    valid = pd.to_numeric(label_excess_train, errors="coerce").dropna()
    winners = valid[valid > float(cost)]
    if winners.empty:
        q60_v = float(cost)
        q80_v = float(cost)
    else:
        q60_v = float(winners.quantile(float(q60)))
        q80_v = float(winners.quantile(float(q80)))

    q60_v = max(float(q60_v), float(cost))
    q80_v = max(float(q80_v), float(q60_v))
    return float(q60_v), float(q80_v)


def _make_rel_rank_from_excess(
    *,
    label_excess: pd.Series,
    cost: float,
    q60: float,
    q80: float,
) -> pd.Series:
    ex = pd.to_numeric(label_excess, errors="coerce")
    out = pd.Series(pd.NA, index=ex.index, dtype="Int8")
    mask = ex.notna()
    if not mask.any():
        return out

    out.loc[mask & (ex <= cost)] = 0
    out.loc[mask & (ex > cost) & (ex <= q60)] = 1
    out.loc[mask & (ex > q60) & (ex <= q80)] = 2
    out.loc[mask & (ex > q80)] = 3
    return out


def _drop_small_rank_groups(
    *,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
    min_items_per_group: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if len(X) == 0:
        return X, y, groups
    g = pd.Series(groups)
    sizes = g.groupby(g).transform("size")
    keep = sizes >= int(min_items_per_group)
    return X.loc[keep.to_numpy()], y.loc[keep.to_numpy()], np.asarray(groups)[keep.to_numpy()]


def get_xgboost_group_sizes_from_preordered(
    df: pd.DataFrame,
    *,
    group_col: str = "Date",
) -> np.ndarray:
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col={group_col!r} in df")

    grp = df[group_col]
    if grp.isna().any():
        raise ValueError(f"group_col={group_col!r} contains NaNs")

    seen: set[object] = set()
    prev = object()
    for v in grp.to_numpy():
        if v != prev:
            if v in seen:
                raise ValueError(
                    f"Rows are not contiguous by {group_col!r}; do not compute group sizes without aligning order. "
                    f"Sort X/y/keys first (e.g. by {group_col!r} and any tie-breakers), then call this helper."
                )
            seen.add(v)
            prev = v

    sizes = grp.groupby(grp, sort=False).size().to_numpy(dtype=np.int32)

    if int(sizes.sum()) != len(df):
        raise RuntimeError("Group sizes do not sum to number of rows")

    return sizes


def _rank_order_and_group_sizes(
    *,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if len(X) == 0:
        return X, y, np.asarray([], dtype=object), np.asarray([], dtype=int)

    df_keys = pd.DataFrame({"_g": groups}, index=X.index)
    if "Date" in y.columns:
        df_keys["Date"] = y["Date"].to_numpy()
    if "Sector" in y.columns:
        df_keys["Sector"] = y["Sector"].astype(str).to_numpy()

    sort_cols = [c for c in ("Date", "_g", "Sector") if c in df_keys.columns]
    if not sort_cols:
        sort_cols = ["_g"]

    order = df_keys.sort_values(sort_cols, kind="mergesort").index
    Xs = X.loc[order]
    ys = y.loc[order]
    groups_ser = pd.Series(groups, index=X.index)
    gs = groups_ser.loc[order].to_numpy()

    if len(gs) == 0:
        group_sizes = np.asarray([], dtype=int)
    else:
        change_idx = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        group_sizes = np.diff(np.r_[change_idx, len(gs)]).astype(int)
    return Xs, ys, gs, group_sizes


def _corr_stable_feature_selection(
    *,
    X: pd.DataFrame,
    y_signal: pd.Series,
    inner_cv: BaseCrossValidator,
    groups: np.ndarray,
    top_n: int,
    min_folds: int,
) -> list[str]:
    features = list(X.columns)
    if not features:
        return []

    counts = pd.Series(0, index=features, dtype=int)

    for inner_train_idx, _inner_val_idx in inner_cv.split(X, y_signal, groups=groups):
        Xi = X.iloc[_as_numpy_index(inner_train_idx)]
        yi = y_signal.iloc[_as_numpy_index(inner_train_idx)]

        ok = yi.notna()
        if int(ok.sum()) < 2:
            continue

        scores = {}
        yv = pd.to_numeric(yi[ok], errors="coerce")
        for c in features:
            xv = pd.to_numeric(Xi.loc[ok, c], errors="coerce")
            s = xv.corr(yv)
            scores[c] = 0.0 if pd.isna(s) else float(abs(s))

        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: int(top_n)]
        for c, _ in top:
            counts.loc[c] += 1

    selected = counts[counts >= int(min_folds)].sort_values(ascending=False)
    if selected.empty:
        selected = counts.sort_values(ascending=False)

    out = selected.index.tolist()[: int(top_n)]
    return out


def _importance_stability_selection(
    *,
    config: NestedCVConfig,
    base_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
    inner_cv: BaseCrossValidator,
    model_factory: Callable[[int, dict[str, Any]], Any],
    fit_fn: Callable[..., Any],
    top_n: int,
    min_folds: int,
) -> list[str]:
    features = list(X.columns)
    if not features:
        return []

    counts = pd.Series(0, index=features, dtype=int)
    mean_imp = pd.Series(0.0, index=features, dtype=float)
    n_used = 0

    seed0 = int(config.seeds[0]) if config.seeds else 0
    for inner_train_idx, _inner_val_idx in inner_cv.split(X, y, groups=groups):
        tr_idx = _as_numpy_index(inner_train_idx)
        X_i_tr = X.iloc[tr_idx]
        y_i_tr = y.iloc[tr_idx]
        g_i_tr = np.asarray(groups)[tr_idx]

        m_tr = _mask_valid_target(y_i_tr, task=config.task)
        X_i_tr = X_i_tr.iloc[m_tr]
        y_i_tr = y_i_tr.iloc[m_tr]
        g_i_tr = g_i_tr[m_tr]

        if X_i_tr.empty:
            continue

        model = model_factory(seed0, dict(base_params))
        if config.task == "gate_classifier":
            model = fit_fn(model, X_i_tr, y_i_tr["y_gate"], groups=g_i_tr)
        elif config.task == "ranker":
            Xs, ys, _gs, group_sizes = _rank_order_and_group_sizes(X=X_i_tr, y=y_i_tr, groups=g_i_tr)
            model = fit_fn(model, Xs, ys["rel_rank"], group_sizes=group_sizes)
        else:
            raise ValueError(f"Unknown task: {config.task}")

        imp = _extract_feature_importance(model, features)
        imp = imp.fillna(0.0)
        if float(imp.abs().sum()) > 0:
            imp = imp.abs() / float(imp.abs().sum())

        top = imp.sort_values(ascending=False)
        top = top[top > 0.0]
        if top.empty:
            continue

        chosen = top.index.tolist()[: int(top_n)]
        counts.loc[chosen] += 1
        mean_imp.loc[chosen] += top.loc[chosen].to_numpy(dtype=float)
        n_used += 1

    if n_used > 0:
        mean_imp = mean_imp / float(n_used)

    selected = counts[counts >= int(min_folds)]
    if selected.empty:
        selected = counts

    tmp = pd.DataFrame({"count": selected, "mean_imp": mean_imp.loc[selected.index]})
    tmp["feature"] = tmp.index.astype(str)
    order = tmp.sort_values(["count", "mean_imp", "feature"], ascending=[False, False, True]).index.tolist()
    return order[: int(top_n)]


def run_inner_search(
    *,
    config: NestedCVConfig,
    X_tr: pd.DataFrame,
    y_tr: pd.DataFrame,
    g_tr: np.ndarray,
    inner_cv_factory: Callable[[NestedCVConfig, np.ndarray], BaseCrossValidator],
    model_factory: Callable[[int, dict[str, Any]], Any],
    fit_fn: Callable[..., Any],
    predict_fn: Callable[..., np.ndarray],
) -> SelectionResult:
    inner_cv = inner_cv_factory(config, g_tr)

    base_params, _space = _split_fixed_and_search_params(config.inner_params)

    if config.feature_selection == "corr":
        if config.task == "gate_classifier":
            y_signal = y_tr["y_gate"]
        elif config.task == "ranker":
            if "label_excess" in y_tr.columns:
                y_signal = y_tr["label_excess"]
            elif "rel_rank" in y_tr.columns:
                y_signal = pd.to_numeric(y_tr["rel_rank"], errors="coerce")
            else:
                raise ValueError("y_tr must contain label_excess or rel_rank for ranker corr feature selection")
        else:
            raise ValueError(f"Unknown task: {config.task}")
        chosen_features = _corr_stable_feature_selection(
            X=X_tr,
            y_signal=y_signal,
            inner_cv=inner_cv,
            groups=g_tr,
            top_n=config.top_n_features,
            min_folds=config.min_feature_folds,
        )
    else:
        chosen_features = _importance_stability_selection(
            config=config,
            base_params=base_params,
            X=X_tr,
            y=y_tr,
            groups=g_tr,
            inner_cv=inner_cv,
            model_factory=model_factory,
            fit_fn=fit_fn,
            top_n=config.top_n_features,
            min_folds=config.min_feature_folds,
        )

    chosen_params = tune_hyperparameters(
        config=config,
        X_tr=X_tr,
        y_tr=y_tr,
        g_tr=g_tr,
        chosen_features=chosen_features,
        inner_cv=inner_cv,
        model_factory=model_factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
    )
    thresholds: dict[str, float] = {}

    if config.task == "gate_classifier" and "y_gate" in y_tr.columns:
        p_grid = config.inner_threshold_grid_p
        if p_grid is None:
            p_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        u_q = config.inner_threshold_grid_u_quantiles
        if u_q is None:
            u_q = [0.50, 0.75, 0.90, 0.95, 1.00]

        rows: list[pd.DataFrame] = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_tr, y_tr, groups=g_tr):
            tr_idx = _as_numpy_index(inner_train_idx)
            va_idx = _as_numpy_index(inner_val_idx)

            X_i_tr = X_tr.iloc[tr_idx][chosen_features]
            y_i_tr = y_tr.iloc[tr_idx]
            g_i_tr = np.asarray(g_tr)[tr_idx]

            X_i_va = X_tr.iloc[va_idx][chosen_features]
            y_i_va = y_tr.iloc[va_idx]
            g_i_va = np.asarray(g_tr)[va_idx]

            m_tr = _mask_valid_target(y_i_tr, task="gate_classifier")
            m_va = _mask_valid_target(y_i_va, task="gate_classifier")

            X_i_tr = X_i_tr.iloc[m_tr]
            y_i_tr = y_i_tr.iloc[m_tr]
            g_i_tr = g_i_tr[m_tr]

            X_i_va = X_i_va.iloc[m_va]
            y_i_va = y_i_va.iloc[m_va]
            g_i_va = g_i_va[m_va]

            if X_i_va.empty or X_i_tr.empty:
                continue

            seed_preds: list[np.ndarray] = []
            for seed in config.seeds:
                model = model_factory(int(seed), dict(chosen_params))
                model = fit_fn(model, X_i_tr, y_i_tr["y_gate"], groups=g_i_tr)
                pred = np.asarray(predict_fn(model, X_i_va))
                seed_preds.append(pred)

            pred_mat = np.column_stack(seed_preds)
            pred_mean = np.mean(pred_mat, axis=1)
            pred_std = np.std(pred_mat, axis=1)

            rows.append(
                pd.DataFrame(
                    {
                        "pred_mean": pred_mean,
                        "pred_std": pred_std,
                        "y_true": pd.to_numeric(y_i_va["y_gate"], errors="coerce").to_numpy(),
                    }
                )
            )

        if rows:
            inner_oof = pd.concat(rows, ignore_index=True)
            inner_oof = inner_oof.dropna(subset=["y_true"])

            if not inner_oof.empty:
                std_vals = inner_oof["pred_std"].to_numpy()
                u_grid = [float(np.quantile(std_vals, q)) for q in u_q]

                best = (-np.inf, None, None)
                y_true = inner_oof["y_true"].to_numpy()
                for p in p_grid:
                    for u in u_grid:
                        pred_pos = (inner_oof["pred_mean"].to_numpy() >= float(p)) & (
                            inner_oof["pred_std"].to_numpy() <= float(u)
                        )
                        tp = float(np.sum((pred_pos == 1) & (y_true == 1)))
                        fp = float(np.sum((pred_pos == 1) & (y_true == 0)))
                        fn = float(np.sum((pred_pos == 0) & (y_true == 1)))
                        denom = (2 * tp + fp + fn)
                        f1 = 0.0 if denom <= 0 else (2 * tp) / denom
                        cand = (f1, float(p), float(u))
                        if cand[0] > best[0]:
                            best = cand

                if best[1] is not None and best[2] is not None:
                    thresholds = {"p_star": float(best[1]), "u_star": float(best[2])}

    return SelectionResult(
        chosen_features=chosen_features,
        chosen_params=chosen_params,
        thresholds=thresholds,
    )


def run_outer_folds(
    *,
    config: NestedCVConfig,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
    outer_cv: BaseCrossValidator,
    inner_cv_factory: Callable[[NestedCVConfig, np.ndarray], BaseCrossValidator],
    model_factory: Callable[[int, dict[str, Any]], Any],
    fit_fn: Callable[..., Any],
    predict_fn: Callable[..., np.ndarray],
    fold_result_sink: Callable[[FoldResult], None] | None = None,
) -> pd.DataFrame:
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, groups must have the same length")

    oof_rows: list[pd.DataFrame] = []

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        outer_train_idx = _as_numpy_index(outer_train_idx)
        outer_test_idx = _as_numpy_index(outer_test_idx)

        X_tr = X.iloc[outer_train_idx]
        y_tr = y.iloc[outer_train_idx]
        g_tr = np.asarray(groups)[outer_train_idx]

        X_te = X.iloc[outer_test_idx]
        y_te = y.iloc[outer_test_idx]
        g_te = np.asarray(groups)[outer_test_idx]

        tr_mask = _mask_valid_target(y_tr, task=config.task)
        te_mask = _mask_valid_target(y_te, task=config.task)

        X_tr = X_tr.iloc[tr_mask]
        y_tr = y_tr.iloc[tr_mask]
        g_tr = g_tr[tr_mask]

        X_te = X_te.iloc[te_mask]
        y_te = y_te.iloc[te_mask]
        g_te = g_te[te_mask]

        thresholds_fold: dict[str, float] = {}
        if config.task == "ranker":
            if "label_excess" not in y_tr.columns:
                raise ValueError("ranker task requires label_excess in y for fold-specific thresholds")
            cost = float(config.cost_bps) / 10_000.0
            q60_v, q80_v = _compute_rank_thresholds_from_train(label_excess_train=y_tr["label_excess"], cost=cost)
            y_tr = y_tr.copy()
            y_te = y_te.copy()
            y_tr["rel_rank"] = _make_rel_rank_from_excess(label_excess=y_tr["label_excess"], cost=cost, q60=q60_v, q80=q80_v)
            y_te["rel_rank"] = _make_rel_rank_from_excess(label_excess=y_te["label_excess"], cost=cost, q60=q60_v, q80=q80_v)
            thresholds_fold = {"q60_train": float(q60_v), "q80_train": float(q80_v), "cost": float(cost)}

        if config.task == "ranker":
            X_tr, y_tr, g_tr = _drop_small_rank_groups(X=X_tr, y=y_tr, groups=g_tr, min_items_per_group=2)
            X_te, y_te, g_te = _drop_small_rank_groups(X=X_te, y=y_te, groups=g_te, min_items_per_group=2)

            if X_tr.empty or X_te.empty:
                continue

            n_train_groups = int(pd.Series(g_tr).nunique())
            if n_train_groups < int(config.min_rank_train_groups):
                continue

            winners = pd.to_numeric(y_tr["label_excess"], errors="coerce") > float(cost)
            n_winners = int(winners.sum())
            if n_winners < int(config.min_rank_train_winners):
                continue

            if "Date" in y_tr.columns:
                win_dates = int(y_tr.loc[winners, "Date"].nunique())
                if win_dates < int(config.min_rank_train_winner_dates):
                    continue

        if X_te.empty:
            continue

        selection = run_inner_search(
            config=config,
            X_tr=X_tr,
            y_tr=y_tr,
            g_tr=g_tr,
            inner_cv_factory=inner_cv_factory,
            model_factory=model_factory,
            fit_fn=fit_fn,
            predict_fn=predict_fn,
        )

        feats = selection.chosen_features
        chosen_params = selection.chosen_params
        thresholds = selection.thresholds
        if config.task == "ranker":
            thresholds = {**thresholds_fold, **dict(thresholds)}

        seed_preds: list[np.ndarray] = []
        for seed in config.seeds:
            model = model_factory(int(seed), dict(chosen_params))

            if config.task == "gate_classifier":
                model = fit_fn(model, X_tr[feats], y_tr["y_gate"], groups=g_tr)
            elif config.task == "ranker":
                Xs, ys, _gs, group_sizes = _rank_order_and_group_sizes(X=X_tr[feats], y=y_tr, groups=g_tr)
                model = fit_fn(model, Xs, ys["rel_rank"], group_sizes=group_sizes)
            else:
                raise ValueError(f"Unknown task: {config.task}")

            if config.task == "ranker":
                Xt, yst, gst, _ = _rank_order_and_group_sizes(X=X_te[feats], y=y_te, groups=g_te)
                pred = np.asarray(predict_fn(model, Xt))
            else:
                pred = np.asarray(predict_fn(model, X_te[feats]))

            seed_preds.append(pred)

        pred_mat = np.column_stack(seed_preds)
        pred_mean = np.mean(pred_mat, axis=1)
        pred_std = np.std(pred_mat, axis=1)

        if config.task == "ranker":
            keys = {}
            for k in ("Date", "Sector"):
                if k in yst.columns:
                    keys[k] = yst[k].to_numpy()
            out = pd.DataFrame(
                {
                    **keys,
                    "fold_id": int(fold_id),
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                    "label_excess": (
                        yst["label_excess"].to_numpy() if "label_excess" in yst.columns else np.full(len(pred_mean), np.nan)
                    ),
                    "group": gst,
                }
            )
        else:
            keys = {}
            for k in ("Date", "Sector"):
                if k in y_te.columns:
                    keys[k] = y_te[k].to_numpy()
            out = pd.DataFrame(
                {
                    **keys,
                    "fold_id": int(fold_id),
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                    "label_excess": (
                        y_te["label_excess"].to_numpy() if "label_excess" in y_te.columns else np.full(len(pred_mean), np.nan)
                    ),
                    "group": g_te,
                }
            )

        if fold_result_sink is not None:
            fold_result_sink(
                FoldResult(
                    fold_id=int(fold_id),
                    indices={
                        "outer_train_idx": outer_train_idx,
                        "outer_test_idx": outer_test_idx,
                    },
                    chosen_features=list(feats),
                    chosen_params=dict(chosen_params),
                    thresholds=dict(thresholds),
                )
            )

        oof_rows.append(out)

    if not oof_rows:
        return pd.DataFrame()

    oof_df = pd.concat(oof_rows, ignore_index=True)

    sort_cols = [c for c in ("Date", "Sector", "fold_id") if c in oof_df.columns]
    if sort_cols:
        oof_df = oof_df.sort_values(sort_cols).reset_index(drop=True)

    return oof_df
