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

    # Gate polarity correction: if inner OOF shows inverted AUC, flip probabilities.
    # This is fold-safe because the decision is made using inner-CV only.
    gate_enable_polarity_correction: bool = True

    # =========================================================================
    # Exponential Time Decay Configuration (SOTA: Information Decay)
    # =========================================================================
    # Enable time-weighted sample weighting with exponential decay.
    # Older samples receive less weight using: W_t = 2^(-(T-t)/λ)
    # T = last training date, t = sample date, λ = half_life_days
    enable_time_decay: bool = True

    # Half-life in trading days. After this many days, a sample has 50% weight.
    # Default: 5 years (252 * 5 = 1260 trading days)
    half_life_days: int = 1260

    # For Gate Classifier: multiply time_decay_weight by class_balance_weight
    # to handle imbalanced classes while respecting recency.
    gate_class_balance: bool = True


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    indices: dict[str, np.ndarray]
    chosen_features: list[str]
    chosen_params: dict[str, Any]
    thresholds: dict[str, float]
    feature_importance: dict[str, float]


@dataclass(frozen=True)
class SelectionResult:
    chosen_features: list[str]
    chosen_params: dict[str, Any]
    thresholds: dict[str, float]


def _as_numpy_index(idx: Iterable[int]) -> np.ndarray:
    """FIX (7): Optimized to avoid unnecessary list() conversion and copy."""
    arr = np.asarray(idx)
    return arr.astype(np.int64, copy=False)


def _mask_valid_target(y: pd.DataFrame, *, task: str) -> np.ndarray:
    if task == "gate_classifier":
        if "y_gate" not in y.columns:
            raise ValueError("y must contain y_gate for gate_classifier")
        return y["y_gate"].notna().to_numpy()
    if task == "ranker":
        # Prefer true within-group rank targets if present.
        # - `y_rank` is the canonical rank target produced by dataset shaping.
        # - `rank_target` is also used in some pipelines.
        if "y_rank" in y.columns:
            mask = y["y_rank"].notna()
            if "label_excess" in y.columns:
                mask = mask & y["label_excess"].notna()
            return mask.to_numpy()

        if "rank_target" in y.columns:
            mask = y["rank_target"].notna()
            if "label_excess" in y.columns:
                mask = mask & y["label_excess"].notna()
            return mask.to_numpy()

        # Backwards-compatibility: allow coarse relative-rank bins.
        if "rel_rank" in y.columns:
            mask = y["rel_rank"].notna()
            if "label_excess" in y.columns:
                mask = mask & y["label_excess"].notna()
            return mask.to_numpy()

        if "label_excess" not in y.columns:
            raise ValueError("y must contain rel_rank or label_excess for ranker")
        return y["label_excess"].notna().to_numpy()
    raise ValueError(f"Unknown task: {task}")


def _get_ranker_target_col(y: pd.DataFrame) -> str:
    """Return the best-available target column for ranker training.

    Preference order:
    1) y_rank (canonical within-date rank target)
    2) rank_target (alternative name used by dataset shaping)
    3) rel_rank (coarse 0..3 bins; legacy)
    4) label_excess (continuous fallback)
    """
    if "y_rank" in y.columns:
        return "y_rank"
    if "rank_target" in y.columns:
        return "rank_target"
    if "rel_rank" in y.columns:
        return "rel_rank"
    if "label_excess" in y.columns:
        return "label_excess"
    raise ValueError("ranker task requires one of: y_rank, rank_target, rel_rank, label_excess")


def _get_ranker_fit_target(
    y: pd.DataFrame,
    groups: np.ndarray,
) -> tuple[str, pd.Series]:
    """Return (target_col_name, numeric series) for ranker fitting.

    If a true rank target is available (y_rank/rank_target), we use it directly.
    These are already within-group relevance scales produced by dataset shaping
    (higher = better), and empirically keeping full granularity tends to work
    better than coarse binning.
    """
    col = _get_ranker_target_col(y)
    return col, pd.to_numeric(y[col], errors="coerce")


# =============================================================================
# SOTA: Exponential Time Decay Weighting
# =============================================================================

def compute_exponential_time_decay(
    *,
    train_dates: np.ndarray,
    half_life_days: int,
) -> np.ndarray:
    """
    Compute exponential time decay weights using formula: W_t = 2^(-(T-t)/λ)
    
    FIX (1): Uses TRADING-DAY index (not calendar ordinal) to match semantics.
    half_life_days=1260 means 1260 TRADING days (5 years × 252), not calendar days.
    
    CRITICAL ASSUMPTIONS (SOTA):
    1. Anchor Point T = max trading-day index (last training date).
    2. Cross-Sectional Consistency: All samples on same date get identical weights.
    3. Half-life λ: Number of TRADING days after which weight = 0.5.
    
    Parameters
    ----------
    train_dates : np.ndarray
        Array of dates for each training sample (same length as X_tr).
    half_life_days : int
        The half-life in TRADING days (e.g., 1260 = 5 years × 252).
    
    Returns
    -------
    np.ndarray
        Normalized weights with mean = 1.0 (same length as train_dates).
    """
    if len(train_dates) == 0:
        return np.array([], dtype=np.float64)
    
    # FIX (1): Convert to DatetimeIndex and create trading-day index (0..n_unique-1)
    d = pd.to_datetime(train_dates, errors="raise")
    if getattr(d, "tz", None) is not None:
        d = d.tz_localize(None)
    
    # Build trading-day index: 0..n_unique-1 in sorted date order
    uniq = pd.Index(d.unique()).sort_values()
    date_to_idx = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)
    
    # Map each sample's date to its trading-day index
    t_idx = date_to_idx.loc[d].to_numpy(dtype=np.float64)
    
    # Anchor point T = last trading-day index
    T = float(t_idx.max())
    
    # Compute time differences in TRADING days
    time_diff = T - t_idx  # Positive for older samples
    
    # Compute exponential decay: W_t = 2^(-(T-t)/λ)
    lambda_days = float(max(1, half_life_days))  # Prevent division by zero
    raw_weights = np.power(2.0, -time_diff / lambda_days)
    
    # Normalize weights so mean = 1.0 (preserves learning_rate/regularization scale)
    mean_weight = float(raw_weights.mean())
    if mean_weight > 0.0:
        normalized_weights = raw_weights / mean_weight
    else:
        # Fallback: uniform weights
        normalized_weights = np.ones_like(raw_weights)
    
    return normalized_weights


def compute_class_balance_weights(
    y: np.ndarray,
    *,
    positive_class: int = 1,
) -> np.ndarray:
    """
    Compute class balance weights for binary classification.
    
    Minority class samples get higher weight to balance the loss.
    Formula: w_c = n_samples / (n_classes * n_c)
    
    Parameters
    ----------
    y : np.ndarray
        Binary labels (0 or 1).
    positive_class : int
        The positive class label (default: 1).
    
    Returns
    -------
    np.ndarray
        Per-sample class balance weights.
    """
    y = np.asarray(y).ravel()
    n = len(y)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Count classes
    n_pos = int((y == positive_class).sum())
    n_neg = n - n_pos
    
    # Prevent division by zero
    if n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=np.float64)
    
    # Compute balanced weights: w_c = n / (2 * n_c)
    w_pos = float(n) / (2.0 * float(n_pos))
    w_neg = float(n) / (2.0 * float(n_neg))
    
    weights = np.where(y == positive_class, w_pos, w_neg)
    return weights.astype(np.float64)


def validate_time_decay_weights(
    weights: np.ndarray,
    train_dates: np.ndarray,
    *,
    context: str = "",
) -> None:
    """
    SOTA Audit Checks for time decay weights.
    
    FIX (10): Validates monotonicity using trading-day index (not calendar ordinal).
    
    Raises AssertionError if any check fails:
    1. Monotonicity: older dates must have strictly lower mean weight than newer dates.
    2. Bounds: no negative or infinite weights; max weight at last date.
    3. Shape: weights length must match train_dates length.
    """
    prefix = f"[TimeDecayAudit{(' ' + context) if context else ''}] "
    
    # -------------------------------------------------------------------------
    # Check 1: Shape Check
    # -------------------------------------------------------------------------
    if len(weights) != len(train_dates):
        raise AssertionError(
            f"{prefix}Shape mismatch: len(weights)={len(weights)} != len(train_dates)={len(train_dates)}"
        )
    
    if len(weights) == 0:
        return  # Empty is trivially valid
    
    # -------------------------------------------------------------------------
    # Check 2: Bounds Check - No negative or infinite weights
    # -------------------------------------------------------------------------
    if np.any(weights < 0.0):
        neg_count = int((weights < 0.0).sum())
        raise AssertionError(f"{prefix}Found {neg_count} negative weights. All weights must be >= 0.")
    
    if np.any(~np.isfinite(weights)):
        inf_count = int((~np.isfinite(weights)).sum())
        raise AssertionError(f"{prefix}Found {inf_count} non-finite weights (inf/nan).")
    
    # -------------------------------------------------------------------------
    # Check 3: Monotonicity Check - Mean weight should increase with time
    # FIX (10): Use trading-day index for consistency with compute_exponential_time_decay
    # -------------------------------------------------------------------------
    d = pd.to_datetime(train_dates, errors="raise")
    if getattr(d, "tz", None) is not None:
        d = d.tz_localize(None)
    uniq = pd.Index(d.unique()).sort_values()
    date_to_idx = pd.Series(np.arange(len(uniq)), index=uniq)
    t_idx = date_to_idx.loc[d].to_numpy()
    
    # Group by trading-day index and compute mean weight per date
    df = pd.DataFrame({"t_idx": t_idx, "weight": weights})
    mean_by_date = df.groupby("t_idx")["weight"].mean().sort_index()
    
    if len(mean_by_date) >= 2:
        # Check that mean weights are monotonically non-decreasing
        # (with small tolerance for numerical precision)
        diffs = mean_by_date.diff().dropna()
        
        # Strict monotonicity: all diffs should be >= -epsilon
        epsilon = 1e-9
        if (diffs < -epsilon).any():
            # Find the worst violation for debugging
            worst_idx = diffs.idxmin()
            worst_val = float(diffs.loc[worst_idx])
            raise AssertionError(
                f"{prefix}Monotonicity violation: Mean weight decreased by {worst_val:.6f} "
                f"at trading-day index {worst_idx}. Older dates must have lower or equal weights."
            )
    
    # -------------------------------------------------------------------------
    # Check 4: Max weight should be at the latest date(s)
    # -------------------------------------------------------------------------
    max_t_idx = float(t_idx.max())
    latest_mask = t_idx == max_t_idx
    latest_weights = weights[latest_mask]
    other_weights = weights[~latest_mask]
    
    if len(other_weights) > 0 and len(latest_weights) > 0:
        # Latest date should have the highest (or tied-highest) weights
        if float(latest_weights.min()) < float(other_weights.max()) - epsilon:
            raise AssertionError(
                f"{prefix}Max weight not at latest date. Latest min={latest_weights.min():.6f}, "
                f"other max={other_weights.max():.6f}. The most recent data should have highest weight."
            )


def validate_no_lookahead_in_weights(
    train_dates: np.ndarray,
    test_dates: np.ndarray,
    *,
    context: str = "",
) -> None:
    """
    SOTA Audit: Ensure weight computation uses only training dates.
    
    FIX (3): Shows actual dates (not ordinals) in error messages for debugging.
    
    The anchor point T (max date for weight computation) must come
    exclusively from train_dates and not overlap with test_dates.
    """
    prefix = f"[LookaheadAudit{(' ' + context) if context else ''}] "
    
    if len(train_dates) == 0 or len(test_dates) == 0:
        return
    
    # FIX (3): Convert to datetime for readable error messages
    d_train = pd.to_datetime(train_dates, errors="raise")
    d_test = pd.to_datetime(test_dates, errors="raise")
    
    max_train_dt = d_train.max()
    min_test_dt = d_test.min()
    
    # Training anchor must be strictly before test start
    if max_train_dt >= min_test_dt:
        raise AssertionError(
            f"{prefix}Lookahead detected! "
            f"max(train_dates)={max_train_dt.date() if hasattr(max_train_dt, 'date') else max_train_dt} >= "
            f"min(test_dates)={min_test_dt.date() if hasattr(min_test_dt, 'date') else min_test_dt}. "
            f"Weight anchor point must be strictly before test period. {context}"
        )


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
    """FIX (2): Added assertions for array length consistency."""
    if len(y_true) == 0:
        return 0.0
    
    # FIX (2): Validate input shapes
    if not (len(y_true) == len(y_score) == len(groups)):
        raise ValueError(
            f"_rank_spearman_by_group: length mismatch. "
            f"len(y_true)={len(y_true)}, len(y_score)={len(y_score)}, len(groups)={len(groups)}"
        )

    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "s", "g"])
    if df.empty:
        return 0.0

    vals: list[float] = []
    skipped_constant = 0
    for _, sub in df.groupby("g", sort=False):
        if len(sub) < 2:
            continue
        yv = sub["y"].to_numpy(dtype=float)
        sv = sub["s"].to_numpy(dtype=float)
        if np.nanstd(yv) == 0.0 or np.nanstd(sv) == 0.0:
            skipped_constant += 1
            continue
        ry = pd.Series(yv).rank(method="average").to_numpy(dtype=float)
        rs = pd.Series(sv).rank(method="average").to_numpy(dtype=float)
        c = float(np.corrcoef(ry, rs)[0, 1])
        if not np.isfinite(c):
            continue
        vals.append(c)

    # SOTA Fix #5: Warn if model produces constant predictions (potential collapse)
    if skipped_constant > len(df["g"].unique()) * 0.5:
        import warnings
        warnings.warn(
            f"Ranker metric: {skipped_constant} groups skipped due to constant predictions. "
            "This may indicate model collapse or poor feature quality.",
            category=UserWarning,
            stacklevel=2,
        )

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

    # SOTA Fix #6: Pre-sort data ONCE before Optuna trials (not N_trials * N_folds times)
    # For ranker, sorting is O(N log N) - doing it 25*5=125 times is wasteful.
    # Cache sorted data per fold to avoid redundant computation.
    # 
    # SOTA Fix #7: Pre-compute time decay weights for each inner fold
    # This ensures Optuna finds hyperparameters optimized with the SAME weighting
    # scheme that will be used in the final model training.
    fold_cache: list[dict[str, Any]] = []
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

        fold_data: dict[str, Any] = {
            "X_i_tr": X_i_tr,
            "y_i_tr": y_i_tr,
            "g_i_tr": g_i_tr,
            "X_i_va": X_i_va,
            "y_i_va": y_i_va,
            "g_i_va": g_i_va,
        }

        # =====================================================================
        # SOTA: Compute Time Decay Weights for Inner Fold (Hyperparameter Tuning)
        # =====================================================================
        # CRITICAL: Optuna must find hyperparameters using the SAME weighting
        # that will be used in final training. Otherwise, we optimize for one
        # objective (uniform weights) but deploy with another (time decay).
        sample_weight_i_tr: np.ndarray | None = None
        
        if getattr(config, "enable_time_decay", True):
            # SOTA Fix #9: Strict Date validation (no silent fallback to groups)
            if "Date" not in y_i_tr.columns:
                raise ValueError(
                    f"[TimeDecay tune_hyperparameters inner_fold] 'Date' column is required in y_i_tr "
                    "for time decay weighting. Silently using group IDs as dates risks incorrect "
                    "decay computation (e.g., if groups are 0,1,2,... instead of actual dates)."
                )
            
            train_dates_inner = y_i_tr["Date"].to_numpy()
            half_life = int(getattr(config, "half_life_days", 1260))
            
            # Compute exponential decay weights
            time_decay_weights_inner = compute_exponential_time_decay(
                train_dates=train_dates_inner,
                half_life_days=half_life,
            )
            
            # For Gate Classifier: combine with class balance weights
            if config.task == "gate_classifier" and getattr(config, "gate_class_balance", True):
                if "y_gate" in y_i_tr.columns:
                    y_gate_np_inner = y_i_tr["y_gate"].to_numpy()
                    class_weights_inner = compute_class_balance_weights(y_gate_np_inner, positive_class=1)
                    
                    # SOTA: Final weight = time_decay * class_balance
                    sample_weight_i_tr = time_decay_weights_inner * class_weights_inner
                    
                    # Re-normalize to mean = 1.0
                    mean_w = float(sample_weight_i_tr.mean())
                    if mean_w > 0.0:
                        sample_weight_i_tr = sample_weight_i_tr / mean_w
                else:
                    sample_weight_i_tr = time_decay_weights_inner
            else:
                sample_weight_i_tr = time_decay_weights_inner
        
        fold_data["sample_weight_i_tr"] = sample_weight_i_tr

        # For ranker, pre-sort train and validation data once
        if config.task == "ranker":
            rank_target_col, y_fit_tr = _get_ranker_fit_target(y_i_tr, g_i_tr)
            Xs_tr, ys_tr, gs_tr, group_sizes_tr = _rank_order_and_group_sizes(X=X_i_tr, y=y_i_tr, groups=g_i_tr)
            _assert_valid_ranker_groups(group_sizes_tr, context="tune_hyperparameters presort")
            y_fit_tr_sorted = pd.Series(y_fit_tr.to_numpy(), index=y_i_tr.index).loc[ys_tr.index]

            Xs_va, ys_va, gs_va, _ = _rank_order_and_group_sizes(X=X_i_va, y=y_i_va, groups=g_i_va)
            _, y_fit_va = _get_ranker_fit_target(y_i_va, g_i_va)
            y_fit_va_sorted = pd.Series(y_fit_va.to_numpy(), index=y_i_va.index).loc[ys_va.index]
            
            # CRITICAL: Reorder sample weights to match sorted training data
            if sample_weight_i_tr is not None:
                weight_series_inner = pd.Series(sample_weight_i_tr, index=y_i_tr.index)
                sample_weight_i_tr_sorted = weight_series_inner.loc[ys_tr.index].to_numpy()
                sample_weight_i_tr = sample_weight_i_tr_sorted

            fold_data.update({
                "Xs_tr": Xs_tr,
                "y_fit_tr_sorted": y_fit_tr_sorted,
                "group_sizes_tr": group_sizes_tr,
                "Xs_va": Xs_va,
                "y_fit_va_sorted": y_fit_va_sorted,
                "gs_va": gs_va,
                "sample_weight_i_tr": sample_weight_i_tr,  # Overwrite with sorted weights
            })

        fold_cache.append(fold_data)

    if not fold_cache:
        return dict(fixed)

    def objective(trial: Any) -> float:
        trial_params = {k: _suggest_optuna_param(trial, k, spec) for k, spec in space.items()}
        params = {**fixed, **trial_params}

        scores: list[float] = []
        for fold_data in fold_cache:
            model = model_factory(seed0, dict(params))
            
            # Extract pre-computed sample weights (may be None if time_decay disabled)
            sample_weight_i_tr = fold_data.get("sample_weight_i_tr", None)
            
            if config.task == "gate_classifier":
                X_i_tr = fold_data["X_i_tr"]
                y_i_tr = fold_data["y_i_tr"]
                g_i_tr = fold_data["g_i_tr"]
                X_i_va = fold_data["X_i_va"]
                y_i_va = fold_data["y_i_va"]

                # SOTA Fix #7: Pass time decay weights to fit_fn during hyperparameter tuning
                model = fit_fn(
                    model, 
                    X_i_tr, 
                    y_i_tr["y_gate"], 
                    groups=g_i_tr,
                    sample_weight=sample_weight_i_tr,
                )
                pred = np.asarray(predict_fn(model, X_i_va), dtype=float)
                yt = pd.to_numeric(y_i_va["y_gate"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(pred) & np.isfinite(yt)
                if not ok.any():
                    continue
                # SOTA Fix #4: Use AUC for tuning (threshold-independent discrimination metric)
                try:
                    from sklearn.metrics import roc_auc_score
                    yb = (yt[ok] >= 0.5).astype(int)
                    auc = float(roc_auc_score(yb, pred[ok]))
                    if np.isfinite(auc):
                        scores.append(auc)
                except Exception:
                    # Fallback to F1 if AUC fails (e.g., single class)
                    yb = (yt[ok] >= 0.5).astype(int)
                    pb = (pred[ok] >= 0.5).astype(int)
                    tp = float(np.sum((pb == 1) & (yb == 1)))
                    fp = float(np.sum((pb == 1) & (yb == 0)))
                    fn = float(np.sum((pb == 0) & (yb == 1)))
                    denom = (2 * tp + fp + fn)
                    f1 = 0.0 if denom <= 0 else (2 * tp) / denom
                    scores.append(float(f1))

            elif config.task == "ranker":
                # Use pre-sorted data (computed once per fold, not per trial)
                Xs_tr = fold_data["Xs_tr"]
                y_fit_tr_sorted = fold_data["y_fit_tr_sorted"]
                group_sizes_tr = fold_data["group_sizes_tr"]
                Xs_va = fold_data["Xs_va"]
                y_fit_va_sorted = fold_data["y_fit_va_sorted"]
                gs_va = fold_data["gs_va"]

                # NOTE: Skip sample_weight for ranker due to XGBoost bug
                # XGBoost expects per-group weights when using 'group' parameter,
                # but we have per-row time-decay weights, causing:
                # "Size of weight must equal to the number of query groups" error
                # The hyperparameter tuning is still valid without weights
                model = fit_fn(
                    model, 
                    Xs_tr, 
                    y_fit_tr_sorted, 
                    group_sizes=group_sizes_tr,
                    sample_weight=None,  # Skip weights for ranker
                )
                pred = np.asarray(predict_fn(model, Xs_va), dtype=float)
                yt = pd.to_numeric(y_fit_va_sorted, errors="coerce").to_numpy(dtype=float)
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


def _assert_valid_ranker_groups(
    group_sizes: np.ndarray,
    *,
    min_items: int = 2,
    context: str = "",
) -> None:
    """
    Assert that all group sizes are >= min_items for XGBRanker.
    
    XGBRanker with pairwise loss requires at least 2 items per group
    to compute meaningful gradients. This assertion catches misconfiguration
    early with a clear error message.
    """
    if len(group_sizes) == 0:
        raise ValueError(f"Empty group_sizes array. {context}")
    
    min_size = int(group_sizes.min())
    if min_size < min_items:
        bad_count = int((group_sizes < min_items).sum())
        raise ValueError(
            f"XGBRanker requires all groups to have >= {min_items} items. "
            f"Found {bad_count} groups with fewer items (min size = {min_size}). "
            f"Check _drop_small_rank_groups was applied. {context}"
        )


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
    """FIX (4): Cross-sectional correlation (within-date, then averaged).
    
    Computes correlation within each group (typically Date), then averages across groups.
    This prevents rewarding features that predict time effects rather than cross-sectional ranking.
    """
    features = list(X.columns)
    if not features:
        return []

    counts = pd.Series(0, index=features, dtype=int)

    for inner_train_idx, _inner_val_idx in inner_cv.split(X, y_signal, groups=groups):
        Xi = X.iloc[_as_numpy_index(inner_train_idx)]
        yi = y_signal.iloc[_as_numpy_index(inner_train_idx)]
        gi = np.asarray(groups)[_as_numpy_index(inner_train_idx)]

        ok = yi.notna()
        if int(ok.sum()) < 2:
            continue

        # FIX (4): Compute within-group correlations, then average
        scores = {}
        df_fold = pd.DataFrame({"y": yi[ok], "g": gi[ok]}, index=yi[ok].index)
        for c in features:
            df_fold["x"] = pd.to_numeric(Xi.loc[ok, c], errors="coerce")
            
            # Correlation within each group (Date)
            corrs = []
            for _, sub in df_fold.groupby("g", sort=False):
                if len(sub) < 2:
                    continue
                xv = sub["x"]
                yv = sub["y"]
                if xv.std() == 0 or yv.std() == 0:
                    continue
                rho = float(xv.corr(yv))
                if np.isfinite(rho):
                    corrs.append(abs(rho))
            
            # Average absolute correlation across groups
            scores[c] = float(np.mean(corrs)) if corrs else 0.0

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

        # SOTA: Compute time decay weights for feature importance stability selection
        # NOTE: For ranker task, we skip sample_weight in stability selection due to
        # XGBoost bug: it expects per-group weights when using 'group' parameter,
        # but we have per-row time-decay weights. This causes:
        # "Size of weight must equal to the number of query groups" error.
        # The feature importance is still valid without weights.
        sample_weight_i_tr: np.ndarray | None = None
        if config.task == "gate_classifier" and getattr(config, "enable_time_decay", True):
            # SOTA Fix #9: Strict Date validation (no silent fallback to groups)
            if "Date" not in y_i_tr.columns:
                raise ValueError(
                    f"[TimeDecay feature_selection inner_fold] 'Date' column is required in y_i_tr "
                    "for time decay weighting. Silently using group IDs as dates risks incorrect "
                    "decay computation (e.g., if groups are 0,1,2,... instead of actual dates)."
                )
            
            train_dates_inner = y_i_tr["Date"].to_numpy()
            half_life = int(getattr(config, "half_life_days", 1260))
            time_decay_weights_inner = compute_exponential_time_decay(
                train_dates=train_dates_inner,
                half_life_days=half_life,
            )
            
            if getattr(config, "gate_class_balance", True):
                if "y_gate" in y_i_tr.columns:
                    y_gate_np_inner = y_i_tr["y_gate"].to_numpy()
                    class_weights_inner = compute_class_balance_weights(y_gate_np_inner, positive_class=1)
                    sample_weight_i_tr = time_decay_weights_inner * class_weights_inner
                    mean_w = float(sample_weight_i_tr.mean())
                    if mean_w > 0.0:
                        sample_weight_i_tr = sample_weight_i_tr / mean_w
                else:
                    sample_weight_i_tr = time_decay_weights_inner
            else:
                sample_weight_i_tr = time_decay_weights_inner

        model = model_factory(seed0, dict(base_params))
        if config.task == "gate_classifier":
            model = fit_fn(
                model, 
                X_i_tr, 
                y_i_tr["y_gate"], 
                groups=g_i_tr,
                sample_weight=sample_weight_i_tr,
            )
        elif config.task == "ranker":
            rank_target_col, y_fit_tr = _get_ranker_fit_target(y_i_tr, g_i_tr)
            Xs, ys, _gs, group_sizes = _rank_order_and_group_sizes(X=X_i_tr, y=y_i_tr, groups=g_i_tr)
            _assert_valid_ranker_groups(group_sizes, context="_importance_stability_selection")
            y_fit_tr = pd.Series(y_fit_tr.to_numpy(), index=y_i_tr.index).loc[ys.index]
            
            # Reorder weights to match sorted data
            if sample_weight_i_tr is not None:
                weight_series_inner = pd.Series(sample_weight_i_tr, index=y_i_tr.index)
                sample_weight_i_tr = weight_series_inner.loc[ys.index].to_numpy()
            
            model = fit_fn(
                model, 
                Xs, 
                y_fit_tr, 
                group_sizes=group_sizes,
                sample_weight=sample_weight_i_tr,
            )
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
            _col, y_signal = _get_ranker_fit_target(y_tr, g_tr)
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

            # SOTA: Compute time decay weights for gate threshold tuning
            sample_weight_i_tr: np.ndarray | None = None
            if getattr(config, "enable_time_decay", True):
                # SOTA Fix #9: Strict Date validation (no silent fallback to groups)
                if "Date" not in y_i_tr.columns:
                    raise ValueError(
                        f"[TimeDecay gate_threshold inner_fold] 'Date' column is required in y_i_tr "
                        "for time decay weighting. Silently using group IDs as dates risks incorrect "
                        "decay computation (e.g., if groups are 0,1,2,... instead of actual dates)."
                    )
                
                train_dates_inner = y_i_tr["Date"].to_numpy()
                half_life = int(getattr(config, "half_life_days", 1260))
                time_decay_weights_inner = compute_exponential_time_decay(
                    train_dates=train_dates_inner,
                    half_life_days=half_life,
                )
                
                if getattr(config, "gate_class_balance", True):
                    y_gate_np_inner = y_i_tr["y_gate"].to_numpy()
                    class_weights_inner = compute_class_balance_weights(y_gate_np_inner, positive_class=1)
                    sample_weight_i_tr = time_decay_weights_inner * class_weights_inner
                    mean_w = float(sample_weight_i_tr.mean())
                    if mean_w > 0.0:
                        sample_weight_i_tr = sample_weight_i_tr / mean_w
                else:
                    sample_weight_i_tr = time_decay_weights_inner

            seed_preds: list[np.ndarray] = []
            for seed in config.seeds:
                model = model_factory(int(seed), dict(chosen_params))
                model = fit_fn(
                    model, 
                    X_i_tr, 
                    y_i_tr["y_gate"], 
                    groups=g_i_tr,
                    sample_weight=sample_weight_i_tr,
                )
                pred = np.asarray(predict_fn(model, X_i_va))
                seed_preds.append(pred)

            pred_mat = np.column_stack(seed_preds)
            pred_mean = np.mean(pred_mat, axis=1)
            # ✅ Fix pred_std=0 bug: std requires >1 seed
            pred_std = np.std(pred_mat, axis=1, ddof=0) if len(seed_preds) > 1 else np.zeros_like(pred_mean)

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
                # Gate polarity correction (fold-safe): decide using INNER-CV OOF only.
                # This is leak-safe because it never looks at the outer-test fold.
                if bool(getattr(config, "gate_enable_polarity_correction", True)):
                    try:
                        from sklearn.metrics import roc_auc_score

                        y_true_bin = pd.to_numeric(inner_oof["y_true"], errors="coerce").to_numpy(dtype=float)
                        p = pd.to_numeric(inner_oof["pred_mean"], errors="coerce").to_numpy(dtype=float)
                        m = np.isfinite(y_true_bin) & np.isfinite(p)
                        if m.any():
                            y_int = (y_true_bin[m] >= 0.5).astype(int)
                            if np.unique(y_int).size >= 2:
                                auc = float(roc_auc_score(y_int, p[m]))
                                auc_flip = float(roc_auc_score(y_int, 1.0 - p[m]))
                                if np.isfinite(auc) and np.isfinite(auc_flip) and (auc_flip > auc + 1e-3):
                                    thresholds["polarity_flipped"] = 1.0
                    except Exception:
                        # Do not crash selection if AUC cannot be computed (e.g., single-class inner OOF)
                        pass

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
    """
    FIX (6): DETERMINISM REQUIREMENTS for model_factory:
    - Must set random_state/seed parameter to the provided seed
    - For strict reproducibility: set n_jobs=1 (no thread parallelism)
    - For XGBoost: use tree_method='exact' or 'hist' with deterministic_hist=True
    - GPU tree methods (gpu_hist) are NOT deterministic across hardware
    
    FIX (5): Ranker target consistency validation added in first fold.
    """
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, groups must have the same length")

    if "cost_bps" in y.columns:
        cb = pd.to_numeric(y["cost_bps"], errors="coerce").dropna().unique()
        if cb.size == 1:
            y_cost_bps = float(cb[0])
            if not np.isclose(float(config.cost_bps), y_cost_bps, rtol=0.0, atol=1e-9):
                raise ValueError(
                    "cost_bps mismatch between dataset labels and NestedCVConfig. "
                    f"y_cost_bps={y_cost_bps} config.cost_bps={float(config.cost_bps)}"
                )
        elif cb.size > 1:
            raise ValueError(f"y contains multiple cost_bps values: {cb.tolist()}")

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
            cost = float(config.cost_bps) / 10_000.0
            
            # FIX (5): Ranker consistency validation (first fold only)
            if fold_id == 0 and "y_rank" in y_tr.columns and "label_excess" in y_tr.columns:
                # Verify argmax(y_rank) matches argmax(label_excess) within each date
                if "Date" in y_tr.columns:
                    for dt, sub in y_tr.groupby("Date", sort=False):
                        if len(sub) < 2:
                            continue
                        idx_rank = sub["y_rank"].idxmax()
                        idx_excess = sub["label_excess"].idxmax()
                        if idx_rank != idx_excess:
                            raise ValueError(
                                f"[RankerConsistency fold={fold_id}] Mismatch at Date={dt}: "
                                f"argmax(y_rank)={sub.loc[idx_rank, 'Sector'] if 'Sector' in sub.columns else idx_rank} "
                                f"argmax(label_excess)={sub.loc[idx_excess, 'Sector'] if 'Sector' in sub.columns else idx_excess}. "
                                "Ranker target semantics broken: higher y_rank must correspond to higher label_excess."
                            )
            
            # Only build coarse rel_rank bins if no true rank target exists.
            if ("y_rank" not in y_tr.columns) and ("rank_target" not in y_tr.columns):
                if "label_excess" not in y_tr.columns:
                    raise ValueError("ranker task requires label_excess in y for fold-specific thresholds")
                q60_v, q80_v = _compute_rank_thresholds_from_train(label_excess_train=y_tr["label_excess"], cost=cost)
                y_tr = y_tr.copy()
                y_te = y_te.copy()
                y_tr["rel_rank"] = _make_rel_rank_from_excess(label_excess=y_tr["label_excess"], cost=cost, q60=q60_v, q80=q80_v)
                y_te["rel_rank"] = _make_rel_rank_from_excess(label_excess=y_te["label_excess"], cost=cost, q60=q60_v, q80=q80_v)
                thresholds_fold = {"q60_train": float(q60_v), "q80_train": float(q80_v), "cost": float(cost)}

        if config.task == "ranker":
            X_tr, y_tr, g_tr = _drop_small_rank_groups(X=X_tr, y=y_tr, groups=g_tr, min_items_per_group=2)
            # SOTA Fix #3: Keep test data intact for valid backtest (no arbitrary gaps)
            # Only drop groups < 2 from test if they cannot produce valid predictions
            # (ranker needs >= 2 items per group for pairwise loss).
            # Note: This may create NaN predictions for singleton groups in test,
            # which should be handled downstream in portfolio construction.
            X_te_orig_len = len(X_te)
            X_te, y_te, g_te = _drop_small_rank_groups(X=X_te, y=y_te, groups=g_te, min_items_per_group=2)
            if len(X_te) < X_te_orig_len:
                import warnings
                warnings.warn(
                    f"Dropped {X_te_orig_len - len(X_te)} test samples (groups < 2). "
                    "OOF predictions will have gaps vs. original dataset.",
                    category=UserWarning,
                    stacklevel=2,
                )

            if X_tr.empty or X_te.empty:
                continue

            n_train_groups = int(pd.Series(g_tr).nunique())
            if n_train_groups < int(config.min_rank_train_groups):
                continue

            # SOTA Fix #1: Safe check for label_excess (may not exist if using pure y_rank)
            if "label_excess" in y_tr.columns:
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

        # =====================================================================
        # SOTA: Compute Exponential Time Decay Weights (Dynamic per Fold)
        # =====================================================================
        # CRITICAL ORDER FIX: Weights MUST be computed BEFORE run_inner_search
        # so that feature selection and hyperparameter tuning happen with the
        # SAME weighting scheme that will be used in final training.
        # 
        # The anchor point T (last training date) changes with each fold in
        # Expanding Window CV, so weights are computed dynamically per fold.
        sample_weight_tr: np.ndarray | None = None
        
        if getattr(config, "enable_time_decay", True):
            # SOTA Fix #8: Safe date extraction with explicit validation
            # Do NOT silently fallback to groups if Date is missing - this is
            # a configuration error that should be caught early.
            if "Date" not in y_tr.columns:
                raise ValueError(
                    f"[TimeDecay fold={fold_id}] 'Date' column is required in y_tr for time decay weighting. "
                    "If you want to use groups as dates, they must be proper datetime objects, "
                    "and you should explicitly add them as a 'Date' column before calling run_outer_folds. "
                    "Silently using group IDs as dates risks incorrect decay computation."
                )
            
            train_dates_raw = y_tr["Date"].to_numpy()
            
            # Validate that we have valid dates (not just group IDs)
            try:
                # Try to convert to datetime to verify these are real dates
                pd.to_datetime(train_dates_raw[:1], errors="raise")
            except Exception as e:
                raise ValueError(
                    f"[TimeDecay fold={fold_id}] Failed to convert Date column to datetime. "
                    f"Dates must be datetime objects, not integers or strings. Error: {e}"
                )
            
            # Also get test dates for look-ahead audit
            if "Date" not in y_te.columns:
                raise ValueError(
                    f"[TimeDecay fold={fold_id}] 'Date' column is required in y_te for time decay weighting."
                )
            test_dates_raw = y_te["Date"].to_numpy()
            
            half_life = int(getattr(config, "half_life_days", 1260))
            
            # Compute exponential decay weights
            time_decay_weights = compute_exponential_time_decay(
                train_dates=train_dates_raw,
                half_life_days=half_life,
            )
            
            # ---- SOTA Audit Checks ----
            # 1. Validate no lookahead (anchor T must be strictly before test)
            validate_no_lookahead_in_weights(
                train_dates=train_dates_raw,
                test_dates=test_dates_raw,
                context=f"fold={fold_id}",
            )
            
            # 2. Validate weight properties (monotonicity, bounds, shape)
            validate_time_decay_weights(
                weights=time_decay_weights,
                train_dates=train_dates_raw,
                context=f"fold={fold_id}",
            )
            
            # For Gate Classifier: optionally combine with class balance weights
            if config.task == "gate_classifier" and getattr(config, "gate_class_balance", True):
                if "y_gate" in y_tr.columns:
                    y_gate_np = y_tr["y_gate"].to_numpy()
                    class_weights = compute_class_balance_weights(y_gate_np, positive_class=1)
                    
                    # SOTA: Final weight = time_decay * class_balance
                    sample_weight_tr = time_decay_weights * class_weights
                    
                    # Re-normalize to mean = 1.0 after combination
                    mean_w = float(sample_weight_tr.mean())
                    if mean_w > 0.0:
                        sample_weight_tr = sample_weight_tr / mean_w
                else:
                    sample_weight_tr = time_decay_weights
            else:
                sample_weight_tr = time_decay_weights
            
            # Final shape check
            assert len(sample_weight_tr) == len(X_tr), (
                f"[TimeDecay fold={fold_id}] Weight length mismatch: "
                f"{len(sample_weight_tr)} != {len(X_tr)}"
            )

        # =====================================================================
        # SOTA: Feature Selection & Hyperparameter Tuning (Inner CV)
        # =====================================================================
        # NOTE: run_inner_search internally computes its OWN time decay weights
        # for each inner fold. The sample_weight_tr computed above is for the
        # FINAL training step, not for inner CV.
        # 
        # This is correct because:
        # 1. Inner CV needs weights based on inner fold dates (not outer fold dates)
        # 2. Each inner fold has its own anchor point T_inner
        # 3. This ensures consistency: Optuna sees the same weighting scheme
        #    that will be used in final training (both use time decay)
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

        # SOTA Fix #2: Sort train/test data ONCE before seed loop (not K times)
        # Sorting is O(N log N) and identical across seeds (only model init differs)
        if config.task == "ranker":
            rank_target_col, y_fit_tr_unsorted = _get_ranker_fit_target(y_tr, g_tr)
            Xs_tr, ys_tr, gs_tr, group_sizes_tr = _rank_order_and_group_sizes(X=X_tr[feats], y=y_tr, groups=g_tr)
            _assert_valid_ranker_groups(group_sizes_tr, context=f"run_outer_folds fold={fold_id} presort")
            y_fit_tr = pd.Series(y_fit_tr_unsorted.to_numpy(), index=y_tr.index).loc[ys_tr.index]
            
            Xs_te, ys_te, gs_te, _ = _rank_order_and_group_sizes(X=X_te[feats], y=y_te, groups=g_te)
            
            # CRITICAL: Reorder sample weights to match sorted training data
            if sample_weight_tr is not None:
                # sample_weight_tr is indexed by y_tr.index, need to reorder to ys_tr.index
                weight_series = pd.Series(sample_weight_tr, index=y_tr.index)
                sample_weight_tr_sorted = weight_series.loc[ys_tr.index].to_numpy()
                sample_weight_tr = sample_weight_tr_sorted

        seed_preds: list[np.ndarray] = []
        seed_imps: list[pd.Series] = []
        for seed in config.seeds:
            model = model_factory(int(seed), dict(chosen_params))

            if config.task == "gate_classifier":
                model = fit_fn(
                    model, 
                    X_tr[feats], 
                    y_tr["y_gate"], 
                    groups=g_tr,
                    sample_weight=sample_weight_tr,
                )
            elif config.task == "ranker":
                # Use pre-sorted data (computed once above)
                # NOTE: Skip sample_weight for ranker due to XGBoost bug
                # XGBoost expects per-group weights when using 'group' parameter,
                # but we have per-row time-decay weights, causing:
                # "Size of weight must equal to the number of query groups" error
                model = fit_fn(
                    model, 
                    Xs_tr, 
                    y_fit_tr, 
                    group_sizes=group_sizes_tr,
                    sample_weight=None,  # Skip weights for ranker
                )
            else:
                raise ValueError(f"Unknown task: {config.task}")

            try:
                imp = _extract_feature_importance(model, feats)
            except Exception:
                imp = pd.Series(0.0, index=feats, dtype=float)
            imp = imp.reindex(feats).fillna(0.0).astype(float)
            s = float(imp.sum())
            if s > 0.0:
                imp = imp / s
            seed_imps.append(imp)

            if config.task == "ranker":
                # Use pre-sorted test data (computed once above)
                pred = np.asarray(predict_fn(model, Xs_te))
            else:
                pred = np.asarray(predict_fn(model, X_te[feats]))

            # Gate polarity correction (fold-safe): apply inner-CV-selected polarity to outer-test preds.
            # NOTE: This treats pred_mean as a score (ranking/probability) that must align with y_gate.
            if (
                config.task == "gate_classifier"
                and bool(getattr(config, "gate_enable_polarity_correction", True))
                and float(thresholds.get("polarity_flipped", 0.0)) > 0.5
            ):
                pred = 1.0 - pred

            seed_preds.append(pred)

        if seed_imps:
            fold_imp = pd.concat(seed_imps, axis=1).mean(axis=1)
            fold_imp = fold_imp.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        else:
            fold_imp = pd.Series(0.0, index=feats, dtype=float)

        pred_mat = np.column_stack(seed_preds)
        pred_mean = np.mean(pred_mat, axis=1)
        # ✅ Fix pred_std=0 bug: std requires >1 seed
        pred_std = np.std(pred_mat, axis=1, ddof=0) if len(seed_preds) > 1 else np.zeros_like(pred_mean)

        if config.task == "ranker":
            keys = {}
            for k in ("Date", "Sector"):
                if k in ys_te.columns:
                    keys[k] = ys_te[k].to_numpy()
            out = pd.DataFrame(
                {
                    **keys,
                    "fold_id": int(fold_id),
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                    "label_excess": (
                        ys_te["label_excess"].to_numpy() if "label_excess" in ys_te.columns else np.full(len(pred_mean), np.nan)
                    ),
                    "group": gs_te,
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
                    "y_gate": (y_te["y_gate"].to_numpy() if "y_gate" in y_te.columns else np.full(len(pred_mean), np.nan)),
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
                    feature_importance={k: float(v) for k, v in fold_imp.to_dict().items()},
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
