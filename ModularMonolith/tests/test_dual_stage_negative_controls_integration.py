from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ModularMonolith.data.dataset_shaping import EXPECTED_SECTORS, ShapingResult
from ModularMonolith.src.reporting.metrics_rank import RANK_SCORE_LOWER_IS_BETTER
from ModularMonolith.src.cv.run_nested_cv import FoldResult


def _make_synthetic_shaping_result(*, horizon: int = 21, n_dates: int = 420, seed: int = 0) -> ShapingResult:
    rng = np.random.default_rng(int(seed))

    sectors = list(EXPECTED_SECTORS)
    n_sectors = len(sectors)

    # Use business-day calendar; must be datetime-like for PurgedWalkForwardCV.
    dates = pd.bdate_range("2014-01-01", periods=int(n_dates), freq="B")

    # Latent drivers
    t = np.arange(len(dates), dtype=float)
    macro = np.sin(2.0 * np.pi * t / 60.0) + 0.25 * np.sin(2.0 * np.pi * t / 17.0)
    sector_alpha = np.linspace(-1.0, 1.0, n_sectors)

    # Build full_df (pre-ordered Date×Sector)
    full_df = pd.DataFrame(
        {
            "Date": np.repeat(dates.to_numpy(), n_sectors),
            "Sector": np.tile(np.asarray(sectors, dtype=object), len(dates)),
        }
    )

    # Features
    # NOTE: these integration tests need the gate to be learnable without leakage.
    # So we make label_excess a *direct* function of a couple of cross-sectional
    # features. Avoid any hidden date->winner mapping that the model can't see.
    n_cs_feats = 16  # cross-sectional features (kept by ranker)
    n_macro_feats = 6  # date-constant (dropped by ranker feature hygiene)

    X = pd.DataFrame(index=full_df.index)

    sec_component = np.tile(sector_alpha.astype(np.float32), len(dates))
    macro_component = np.repeat(macro.astype(np.float32), n_sectors)

    # Make a couple of strong, low-noise signal features.
    base_noise = rng.normal(0.0, 1.0, size=len(full_df)).astype(np.float32)
    X["cs_00"] = (sec_component + 0.05 * base_noise).astype(np.float32)
    X["cs_01"] = ((sec_component * macro_component) + 0.05 * base_noise).astype(np.float32)

    # Remaining cross-sectional features: mostly noise (some weak correlation).
    for k in range(2, n_cs_feats):
        w1 = float(rng.normal())
        w2 = float(rng.normal())
        noise = rng.normal(0.0, 1.0, size=len(full_df)).astype(np.float32)
        X[f"cs_{k:02d}"] = (
            (0.10 * w1 * sec_component)
            + (0.10 * w2 * (sec_component * macro_component))
            + (0.90 * noise)
        ).astype(np.float32)

    # Macro features: constant across sectors within each date
    for k in range(n_macro_feats):
        w = float(rng.normal())
        X[f"macro_{k:02d}"] = (w * macro_component).astype(np.float32)

    feature_cols = list(X.columns)

    # Targets
    # Build label_excess from the signal features so both gate+ranker are learnable.
    eps_y = rng.normal(0.0, 0.02, size=len(full_df)).astype(np.float32)
    label_excess = (
        (1.50 * X["cs_00"].to_numpy(dtype=np.float32))
        + (2.25 * X["cs_01"].to_numpy(dtype=np.float32))
        + (0.10 * macro_component)
        + eps_y
    ).astype(np.float32)

    # Deterministic, tiny tie-break by sector index.
    sec_idx = np.tile(np.arange(n_sectors, dtype=np.float32), len(dates))
    label_excess = (label_excess + 1e-6 * (sec_idx + 1.0)).astype(np.float32)
    full_df["label_excess"] = label_excess

    # Rank target: must satisfy argmax(y_rank) == argmax(label_excess) per Date.
    y_rank = label_excess

    # Gate target (not strictly used by DualModelTrainer when gate_label_mode != "excess_gt_cost",
    # but included for completeness / run metadata).
    # Use top-3 winners per date -> prevalence ~ 1/3.
    tmp = full_df[["Date", "label_excess"]].copy()
    tmp["_rank"] = tmp.groupby("Date", sort=False)["label_excess"].rank(method="first", ascending=False)
    y_gate = (tmp["_rank"].to_numpy(dtype=float) <= 3.0).astype(np.int8)

    group_sizes = np.full(len(dates), n_sectors, dtype=np.int32)

    # Minimal paths (not used in these tests)
    p = Path(".")

    return ShapingResult(
        horizon=int(horizon),
        target_col="label_excess",
        features_path=p / "synthetic_features.parquet",
        labels_path=p / "synthetic_labels.parquet",
        correlations_path=p / "synthetic_corr.parquet",
        full_df=full_df,
        X=X,
        y_gate=y_gate,
        y_rank=y_rank,
        group_sizes=group_sizes,
        feature_cols=feature_cols,
        dropped_rows_nan_target=0,
        dropped_dates_nan_target=0,
        dropped_dates_universe_policy=0,
        cost_threshold=0.0,
    )


def _fake_get_cv_config_for_sr(sr: ShapingResult) -> dict:
    unique_dates = pd.Index(pd.to_datetime(sr.full_df["Date"], errors="raise").unique()).sort_values()

    # Make sure we have enough room for:
    # min_train_size + purge_gap + test_size * n_splits
    # Keep this feasible for BOTH outer CV and the inner CV created by
    # `inner_cv_factory` (which currently hard-codes inner_n_splits=2 and
    # does NOT pass test_start). That means the inner CV must satisfy:
    #   n_groups >= 2*test_size + purge_gap + min_train_size
    n_splits = 4
    test_size = 10
    purge_gap = 21
    min_train_size = 40

    # IMPORTANT: inner_cv_factory (used inside nested-CV) currently creates an
    # inner splitter with test_start=None. That means the inner CV requires
    # enough unique training dates in the FIRST outer fold.
    #
    # So we intentionally place `test_start` far enough into the series so
    # that early outer folds have plenty of train history.
    first_test_start = max(min_train_size + purge_gap + 5, 220)
    if first_test_start >= len(unique_dates):
        raise AssertionError("Synthetic dataset too small for configured CV")

    test_start = unique_dates[first_test_start]

    return {
        "n_splits": int(n_splits),
        "test_size": int(test_size),
        "step_size": int(test_size),
        "purge_gap": int(purge_gap),
        "embargo": 0,
        "min_train_size": int(min_train_size),
        "max_train_size": None,
        "warmup_days": int(min_train_size + purge_gap),
        "test_start": test_start,
        "contract": {"synthetic": True},
        "include_remainder_in_last_fold": False,
    }


def _selected_topk_by_date(df: pd.DataFrame, *, k: int = 3) -> pd.DataFrame:
    if not RANK_SCORE_LOWER_IS_BETTER:
        raise AssertionError("These tests assume lower-is-better rank convention")

    d = df[["Date", "Sector", "rank_mean"]].copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="raise")
    d["Sector"] = d["Sector"].astype(str)

    # Deterministic tie-breaker: rank_mean then Sector
    d = d.sort_values(["Date", "rank_mean", "Sector"], kind="mergesort")
    return d.groupby("Date", sort=False, as_index=False).head(int(k))[["Date", "Sector"]]


def _uplift_vs_full_universe(
    merged_oof: pd.DataFrame,
    *,
    gate_fold_results: list[FoldResult],
    k: int = 3,
) -> float:
    """Compute end-to-end uplift vs the FULL universe.

    For each Date:
    - Gate: keep rows passing that row's fold-specific p_star/u_star thresholds
    - Rank: pick top-k among survivors using the global convention (lower score = better)
    - Score: (mean label_excess of selected) - (mean label_excess of ALL sectors that date)

    Returns the mean uplift across dates with at least one survivor.
    """
    if merged_oof.empty:
        return 0.0

    required = {"Date", "label_excess", "rank_mean", "p_gate_mean", "fold_id"}
    missing = required - set(merged_oof.columns)
    if missing:
        raise AssertionError(f"merged_oof missing required columns: {sorted(missing)}")

    fold_thresholds: dict[int, dict[str, float]] = {
        int(fr.fold_id): dict(fr.thresholds or {}) for fr in gate_fold_results
    }

    d = merged_oof[["Date", "Sector", "label_excess", "rank_mean", "p_gate_mean", "fold_id"]].copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="raise")

    def _passed_row(r: pd.Series) -> bool:
        t = fold_thresholds.get(int(r["fold_id"]), {})
        p_star = float(t.get("p_star", 0.5))
        u_star = float(t.get("u_star", float("inf")))
        ok = float(r["p_gate_mean"]) >= p_star
        # If we don't have p_gate_std in merged_oof, ignore uncertainty threshold.
        if np.isfinite(u_star) and ("p_gate_std" in merged_oof.columns):
            ok = ok and (float(r.get("p_gate_std", float("inf"))) <= u_star)
        return bool(ok)

    passed = d.apply(_passed_row, axis=1)
    d_pass = d.loc[passed].copy()
    if d_pass.empty:
        return 0.0

    uplifts: list[float] = []
    for dt, sub_all in d.groupby("Date", sort=False):
        sub_pass = d_pass.loc[d_pass["Date"] == dt]
        if sub_pass.empty:
            continue

        mean_all = float(pd.to_numeric(sub_all["label_excess"], errors="coerce").mean())
        sub_pass = sub_pass.dropna(subset=["label_excess", "rank_mean"])
        if sub_pass.empty:
            continue

        # Lower score = better globally.
        sub_pass = sub_pass.sort_values(["rank_mean", "Sector"], ascending=[True, True], kind="mergesort")
        top = sub_pass.head(int(min(int(k), len(sub_pass))))
        mean_sel = float(pd.to_numeric(top["label_excess"], errors="coerce").mean())
        if np.isfinite(mean_all) and np.isfinite(mean_sel):
            uplifts.append(mean_sel - mean_all)

    return float(np.mean(uplifts)) if uplifts else 0.0


def test_phase4_baseline_integration_synthetic_shapingresult(monkeypatch, tmp_path):
    from ModularMonolith.src.models import train_dual_system as tds

    sr = _make_synthetic_shaping_result(horizon=21, n_dates=420, seed=0)

    monkeypatch.setattr(tds.DualModelTrainer, "_load_data", lambda self: sr)
    monkeypatch.setattr(tds, "get_cv_config", lambda horizon: _fake_get_cv_config_for_sr(sr))

    # Record stage calls (extra integration check)
    calls: list[str] = []
    real_run_outer_folds = tds.run_outer_folds

    def _wrapped_run_outer_folds(*, config, **kwargs):
        calls.append(str(getattr(config, "task", "")))
        return real_run_outer_folds(config=config, **kwargs)

    monkeypatch.setattr(tds, "run_outer_folds", _wrapped_run_outer_folds)

    trainer = tds.DualModelTrainer(
        21,
        seeds=[0],
        gate_params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.10},
        ranker_params={"n_estimators": 40, "max_depth": 3, "learning_rate": 0.10},
        feature_selection="corr",
        top_n_features=15,
        optuna_n_trials=0,
        use_gpu=False,
        verbose=False,
        base_results_dir=str(tmp_path),
        auto_generate_plots=False,
        auto_write_printable=False,
        auto_run_backtest=False,
        auto_save_models=False,
        enable_rolling_zscore=False,
        gate_label_mode="topk",
        gate_topk=3,
        deterministic_mode=True,
    )

    result = trainer.run()

    assert calls[:2] == ["gate_classifier", "ranker"], f"Unexpected stage call sequence: {calls}"

    assert float(result.gate_metrics.get("auc", 0.0)) > 0.60

    # Synthetic should be strongly learnable under the global convention:
    # lower score = better => rank_ic should be strongly NEGATIVE.
    assert float(result.rank_metrics.rank_ic) < -0.80

    # End-to-end (gate -> rank) should improve mean return vs full universe.
    uplift_full = _uplift_vs_full_universe(result.merged_oof, gate_fold_results=trainer._gate_fold_results, k=3)
    assert uplift_full > 0.10


def test_phase4_permute_gate_train_only(monkeypatch, tmp_path):
    from ModularMonolith.src.models import train_dual_system as tds

    sr = _make_synthetic_shaping_result(horizon=21, n_dates=420, seed=1)

    monkeypatch.setattr(tds.DualModelTrainer, "_load_data", lambda self: sr)
    monkeypatch.setattr(tds, "get_cv_config", lambda horizon: _fake_get_cv_config_for_sr(sr))

    real_fit_classifier = tds.fit_classifier

    def _fit_classifier_permute_y(model, X, y, *, groups=None, sample_weight=None):
        # Permute labels WITHIN each Date group to preserve per-date prevalence
        # while destroying cross-sectional learnability.
        y_np = np.asarray(y).ravel().copy()
        g = np.asarray(groups).ravel() if groups is not None else np.arange(len(y_np))
        out = y_np.copy()
        # IMPORTANT: do NOT seed by dt itself, otherwise the permutation becomes
        # a deterministic function of Date and can become learnable via macro/time features.
        rng = np.random.default_rng(12345)
        for dt in pd.Index(g).unique():
            mask = g == dt
            if int(mask.sum()) <= 1:
                continue
            vals = out[mask].copy()
            rng.shuffle(vals)
            out[mask] = vals
        return real_fit_classifier(model, X, out, groups=groups, sample_weight=sample_weight)

    monkeypatch.setattr(tds, "fit_classifier", _fit_classifier_permute_y)

    trainer = tds.DualModelTrainer(
        21,
        seeds=[0],
        gate_params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.10},
        ranker_params={"n_estimators": 40, "max_depth": 3, "learning_rate": 0.10},
        feature_selection="corr",
        top_n_features=15,
        optuna_n_trials=0,
        use_gpu=False,
        verbose=False,
        base_results_dir=str(tmp_path),
        auto_generate_plots=False,
        auto_write_printable=False,
        auto_run_backtest=False,
        auto_save_models=False,
        enable_rolling_zscore=False,
        gate_label_mode="topk",
        gate_topk=3,
        deterministic_mode=True,
    )

    result = trainer.run()

    # Gate collapses to chance.
    assert abs(float(result.gate_metrics.get("auc", 0.5)) - 0.5) < 0.08

    # End-to-end uplift vs the full universe should collapse.
    uplift_full = _uplift_vs_full_universe(result.merged_oof, gate_fold_results=trainer._gate_fold_results, k=3)
    assert uplift_full < 0.0


def test_phase4_permute_ranker_train_only(monkeypatch, tmp_path):
    from ModularMonolith.src.models import train_dual_system as tds
    from ModularMonolith.src.cv import run_nested_cv as rnc

    sr = _make_synthetic_shaping_result(horizon=21, n_dates=420, seed=2)

    monkeypatch.setattr(tds.DualModelTrainer, "_load_data", lambda self: sr)
    monkeypatch.setattr(tds, "get_cv_config", lambda horizon: _fake_get_cv_config_for_sr(sr))

    real_get_ranker_fit_target = rnc._get_ranker_fit_target

    def _permute_within_date(series: pd.Series, dates: np.ndarray, *, fold_seed: int) -> pd.Series:
        out = series.copy()
        # Seed by fold and date value to avoid repetition.
        for dt in pd.Index(dates).unique():
            mask = dates == dt
            if int(mask.sum()) <= 1:
                continue
            # Derive a stable int seed from dt + fold_seed.
            dt64 = np.datetime64(pd.Timestamp(dt).to_datetime64())
            dt_int = int(dt64.astype("datetime64[ns]").astype(np.int64) % (2**32))
            rng = np.random.default_rng((fold_seed + dt_int) % (2**32))
            vals = out.loc[mask].to_numpy(copy=True)
            rng.shuffle(vals)
            # Add tiny noise to avoid ties.
            vals = vals + rng.normal(0.0, 1e-6, size=len(vals))
            out.loc[mask] = vals.astype(out.dtype, copy=False)
        return out

    def _get_ranker_fit_target_permuted(y: pd.DataFrame, groups: np.ndarray):
        col, s = real_get_ranker_fit_target(y, groups)
        dates = pd.to_datetime(y["Date"], errors="raise").to_numpy()
        # Fold-dependent seed: based on train date span.
        fold_seed = int(pd.to_datetime(dates).max().value % (2**31 - 1))
        s_perm = _permute_within_date(pd.Series(s.to_numpy(), index=y.index), dates, fold_seed=fold_seed)
        return col, s_perm

    monkeypatch.setattr(rnc, "_get_ranker_fit_target", _get_ranker_fit_target_permuted)

    trainer = tds.DualModelTrainer(
        21,
        seeds=[0],
        gate_params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.10},
        ranker_params={"n_estimators": 40, "max_depth": 3, "learning_rate": 0.10},
        feature_selection="corr",
        top_n_features=15,
        optuna_n_trials=0,
        use_gpu=False,
        verbose=False,
        base_results_dir=str(tmp_path),
        auto_generate_plots=False,
        auto_write_printable=False,
        auto_run_backtest=False,
        auto_save_models=False,
        enable_rolling_zscore=False,
        gate_label_mode="topk",
        gate_topk=3,
        deterministic_mode=True,
    )

    result = trainer.run()

    # Gate should remain learnable.
    assert float(result.gate_metrics.get("auc", 0.0)) > 0.60

    # Ranker collapses.
    assert abs(float(result.rank_metrics.rank_ic)) < 0.25
    assert abs(float(result.rank_metrics.lift_at_3)) < 0.15

    # Gated rank metrics collapse too.
    if result.rank_metrics_gated is not None:
        assert abs(float(result.rank_metrics_gated.rank_ic)) < 0.30
        assert abs(float(result.rank_metrics_gated.lift_at_3)) < 0.15


def test_phase4_future_perturbation_invariance(monkeypatch, tmp_path):
    from ModularMonolith.src.models import train_dual_system as tds

    sr_base = _make_synthetic_shaping_result(horizon=21, n_dates=420, seed=3)

    # Mutate features only after cutoff.
    unique_dates = pd.Index(pd.to_datetime(sr_base.full_df["Date"], errors="raise").unique()).sort_values()
    # Keep the cutoff inside the OOS test region so this is non-trivial.
    cutoff = unique_dates[245]

    X2 = sr_base.X.copy()
    after = pd.to_datetime(sr_base.full_df["Date"], errors="raise") > cutoff
    # Large perturbation to ensure the check is meaningful
    X2.loc[after, :] = X2.loc[after, :] + 1_000.0

    sr_mut = replace(sr_base, X=X2)

    def _run(sr: ShapingResult):
        monkeypatch.setattr(tds.DualModelTrainer, "_load_data", lambda self: sr)
        monkeypatch.setattr(tds, "get_cv_config", lambda horizon: _fake_get_cv_config_for_sr(sr))

        trainer = tds.DualModelTrainer(
            21,
            seeds=[0],
            gate_params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.10},
            ranker_params={"n_estimators": 40, "max_depth": 3, "learning_rate": 0.10},
            feature_selection="corr",
            top_n_features=15,
            optuna_n_trials=0,
            use_gpu=False,
            verbose=False,
            base_results_dir=str(tmp_path),
            auto_generate_plots=False,
            auto_write_printable=False,
            auto_run_backtest=False,
            auto_save_models=False,
            enable_rolling_zscore=False,
            gate_label_mode="topk",
            gate_topk=3,
            deterministic_mode=True,
        )
        return trainer.run()

    res_a = _run(sr_base)
    res_b = _run(sr_mut)

    a = res_a.merged_oof.copy()
    b = res_b.merged_oof.copy()

    a["Date"] = pd.to_datetime(a["Date"], errors="raise")
    b["Date"] = pd.to_datetime(b["Date"], errors="raise")

    a_pre = a.loc[a["Date"] <= cutoff].sort_values(["Date", "Sector", "fold_id"], kind="mergesort")
    b_pre = b.loc[b["Date"] <= cutoff].sort_values(["Date", "Sector", "fold_id"], kind="mergesort")

    assert len(a_pre) == len(b_pre) and len(a_pre) > 0

    # For dates <= cutoff: gate preds and rank scores must be identical.
    np.testing.assert_allclose(
        a_pre["p_gate_mean"].to_numpy(dtype=float),
        b_pre["p_gate_mean"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        a_pre["rank_mean"].to_numpy(dtype=float),
        b_pre["rank_mean"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )

    # Selection set must match too.
    sel_a = _selected_topk_by_date(a_pre, k=3)
    sel_b = _selected_topk_by_date(b_pre, k=3)

    key_a = list(map(tuple, sel_a.to_numpy()))
    key_b = list(map(tuple, sel_b.to_numpy()))
    assert key_a == key_b
