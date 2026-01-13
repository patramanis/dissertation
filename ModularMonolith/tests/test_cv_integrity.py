from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ModularMonolith.src.cv.purged_walk_forward_cv import PurgedWalkForwardCV
from ModularMonolith.src.cv.run_nested_cv import (
    NestedCVConfig,
    compute_exponential_time_decay,
    run_outer_folds,
    validate_no_lookahead_in_weights,
    validate_time_decay_weights,
)


def _make_panel(*, n_dates: int = 260, n_sectors: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    dates = pd.date_range("2000-01-03", periods=int(n_dates), freq="B")
    sectors = [f"S{i}" for i in range(int(n_sectors))]
    idx = pd.MultiIndex.from_product([dates, sectors], names=["Date", "Sector"])
    df = idx.to_frame(index=False)
    groups = df["Date"].to_numpy(dtype="datetime64[ns]")
    return df, groups


def _date_to_pos(dates: np.ndarray) -> dict[np.datetime64, int]:
    uniq = np.unique(dates)
    uniq = np.sort(uniq)
    return {np.datetime64(d): int(i) for i, d in enumerate(uniq)}


def _effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float).ravel()
    w = w[np.isfinite(w)]
    if w.size == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return 0.0 if s2 <= 0.0 else (s1 * s1) / s2


def test_purged_walk_forward_invariants_no_overlap_and_time_order() -> None:
    df, groups = _make_panel(n_dates=300, n_sectors=3)
    X = np.zeros((len(df), 1), dtype=float)

    uniq_dates = np.sort(df["Date"].unique())

    cv = PurgedWalkForwardCV(
        n_splits=3,
        test_size=20,
        purge_gap=21,
        min_train_size=60,
        embargo=0,
        # IMPORTANT: test_start must be specified in *date space*, not row space.
        test_start=uniq_dates[120],
    )

    prev_train_end = None
    for train_idx, test_idx, info in cv.split_with_info(X, groups=groups):
        assert train_idx.size > 0
        assert test_idx.size > 0
        assert np.intersect1d(train_idx, test_idx).size == 0

        train_dates = groups[train_idx]
        test_dates = groups[test_idx]

        assert np.max(train_dates) < np.min(test_dates)

        # Expanding: train start fixed, train end non-decreasing.
        if prev_train_end is not None:
            assert info.train_end >= prev_train_end
        prev_train_end = info.train_end

        assert info.train_end < info.test_start


def test_purged_walk_forward_purge_gap_is_horizon_safe() -> None:
    # Horizon-aware purge contract: if labels use a window ending at t+H,
    # then for test starting at t0, we must have (train_date + H) < t0.
    horizon = 21

    df, groups = _make_panel(n_dates=320, n_sectors=4)
    X = np.zeros((len(df), 1), dtype=float)

    uniq_dates = np.sort(df["Date"].unique())

    cv = PurgedWalkForwardCV(
        n_splits=2,
        test_size=30,
        purge_gap=horizon,
        min_train_size=80,
        embargo=0,
        test_start=uniq_dates[140],
    )

    pos = _date_to_pos(groups)

    for train_idx, test_idx, info in cv.split_with_info(X, groups=groups):
        test_start_pos = pos[np.datetime64(info.test_start)]
        train_dates = groups[train_idx]
        for d in np.unique(train_dates):
            dpos = pos[np.datetime64(d)]
            label_end_pos = dpos + int(horizon)
            assert label_end_pos < test_start_pos


def test_embargo_rejected_for_expanding_cv() -> None:
    # This CV class explicitly rejects embargo>0 to avoid a false sense of safety.
    with pytest.raises(ValueError):
        PurgedWalkForwardCV(
            n_splits=2,
            test_size=10,
            purge_gap=5,
            min_train_size=30,
            embargo=2,
            enforce_expanding_only=True,
        )


def test_fold_ledger_csv_written_and_consistent(tmp_path: Path) -> None:
    df, groups = _make_panel(n_dates=280, n_sectors=2)
    X = np.zeros((len(df), 1), dtype=float)

    uniq_dates = np.sort(df["Date"].unique())

    purge_gap = 10
    cv = PurgedWalkForwardCV(
        n_splits=2,
        test_size=25,
        purge_gap=purge_gap,
        min_train_size=60,
        embargo=0,
        test_start=uniq_dates[160],
    )

    uniq = np.sort(np.unique(groups))
    pos = {np.datetime64(d): int(i) for i, d in enumerate(uniq)}

    for fold_id, (train_idx, test_idx, info) in enumerate(cv.split_with_info(X, groups=groups), start=1):
        train_dates = np.unique(groups[train_idx])
        test_dates = np.unique(groups[test_idx])

        test_start_pos = pos[np.datetime64(info.test_start)]
        train_end_pos = test_start_pos - int(purge_gap)
        purged_dates = set(uniq[train_end_pos:test_start_pos].tolist())

        ledger = pd.DataFrame({"Date": uniq})
        ledger["is_train"] = ledger["Date"].isin(train_dates)
        ledger["is_test"] = ledger["Date"].isin(test_dates)
        ledger["is_purged"] = ledger["Date"].isin(list(purged_dates))
        ledger["is_embargoed"] = False
        ledger["is_used"] = ledger["is_train"]

        assert not bool((ledger["is_train"] & ledger["is_test"]).any())
        assert not bool((ledger["is_used"] & ledger["is_purged"]).any())
        assert not bool((ledger["is_test"] & ledger["is_purged"]).any())

        out = tmp_path / f"fold_ledger_{fold_id}.csv"
        ledger.to_csv(out, index=False)
        assert out.exists()


def test_time_decay_weights_monotonic_and_normalized() -> None:
    df, groups = _make_panel(n_dates=200, n_sectors=3)
    train_dates = df["Date"].to_numpy(dtype="datetime64[ns]")

    w = compute_exponential_time_decay(train_dates=train_dates, half_life_days=50)
    validate_time_decay_weights(w, train_dates, context="unit")

    assert len(w) == len(train_dates)
    assert np.isfinite(w).all()
    assert float(np.mean(w)) == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_time_decay_weights_detect_misalignment() -> None:
    df, _ = _make_panel(n_dates=200, n_sectors=1)
    # Use strictly increasing dates to maximize monotonicity sensitivity.
    train_dates = df["Date"].drop_duplicates().to_numpy(dtype="datetime64[ns]")

    w = compute_exponential_time_decay(train_dates=train_dates, half_life_days=25)

    # Break alignment: permute weights but keep dates ordered.
    rng = np.random.default_rng(0)
    w_bad = rng.permutation(w)

    with pytest.raises(AssertionError):
        validate_time_decay_weights(w_bad, train_dates, context="misaligned")


def test_time_decay_weights_underflow_and_ess_diagnostic() -> None:
    # Very long history + tiny half-life => severe underflow for older samples.
    dates = pd.date_range("1990-01-01", periods=6000, freq="B").to_numpy(dtype="datetime64[ns]")
    w = compute_exponential_time_decay(train_dates=dates, half_life_days=1)

    assert len(w) == len(dates)
    assert float(np.mean(w)) == pytest.approx(1.0, rel=0.0, abs=1e-12)

    ess = _effective_sample_size(w)
    # ESS should collapse dramatically (practically ultra-short window)
    assert ess < 200.0


def test_time_decay_weights_equivalence_large_halflife_near_uniform() -> None:
    dates = pd.date_range("2000-01-03", periods=500, freq="B").to_numpy(dtype="datetime64[ns]")
    w = compute_exponential_time_decay(train_dates=dates, half_life_days=1_000_000)

    assert float(np.mean(w)) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert float(np.max(w) - np.min(w)) < 1e-3

    ess = _effective_sample_size(w)
    assert ess > 0.99 * len(w)


def test_validate_no_lookahead_in_weights_catches_overlap() -> None:
    train_dates = pd.date_range("2020-01-01", periods=10, freq="B").to_numpy(dtype="datetime64[ns]")
    # Test starts at an earlier date than max(train) => illegal for time-decay anchor.
    test_dates = pd.date_range("2020-01-10", periods=5, freq="B").to_numpy(dtype="datetime64[ns]")

    with pytest.raises(AssertionError):
        validate_no_lookahead_in_weights(train_dates, test_dates, context="unit")


def _make_synthetic_gate_dataset(*, n_dates: int = 260, n_sectors: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    df, groups = _make_panel(n_dates=n_dates, n_sectors=n_sectors)

    # Deterministic signal: sector effect + slow regime drift.
    t = (pd.to_datetime(df["Date"]).astype("int64") // 86_400_000_000_000).to_numpy(dtype=float)
    s = df["Sector"].astype("category").cat.codes.to_numpy(dtype=float)

    f_signal = 0.2 * np.sin(t / 11.0) + 0.3 * (s - s.mean())
    f_noise = 0.1 * np.cos(t / 7.0)

    X = pd.DataFrame({"f_signal": f_signal, "f_noise": f_noise})

    # Binary target with a clear relationship to f_signal.
    y_latent = f_signal + 0.05 * np.sin(t / 3.0)
    y_gate = (y_latent > np.median(y_latent)).astype(np.int8)

    y = pd.DataFrame(
        {
            "y_gate": y_gate,
            "Date": df["Date"].to_numpy(dtype="datetime64[ns]"),
            "Sector": df["Sector"].astype("string"),
        }
    )

    return X, y, groups


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel().astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _resolve_oof_y_true(oof: pd.DataFrame) -> pd.Series:
    # run_outer_folds uses task-specific columns; for gate it carries y_gate.
    if "y_true" in oof.columns:
        return pd.to_numeric(oof["y_true"], errors="coerce")
    if "y_gate" in oof.columns:
        return pd.to_numeric(oof["y_gate"], errors="coerce")
    raise KeyError(f"Cannot find y_true in oof columns: {list(oof.columns)}")


def test_run_outer_folds_label_permutation_negative_control() -> None:
    # If we permute labels ONLY in train, out-of-sample AUC should collapse to ~0.5.
    X, y, groups = _make_synthetic_gate_dataset(n_dates=260, n_sectors=5)

    outer_cv = PurgedWalkForwardCV(
        n_splits=3,
        test_size=20,
        purge_gap=10,
        min_train_size=80,
        test_start=np.unique(groups)[180],
    )

    def inner_cv_factory(config: NestedCVConfig, groups_inner: np.ndarray):
        return PurgedWalkForwardCV(
            n_splits=2,
            test_size=15,
            purge_gap=10,
            min_train_size=40,
        )

    from sklearn.linear_model import LogisticRegression

    def model_factory(seed: int, params: dict):
        # liblinear is deterministic for binary problems.
        return LogisticRegression(
            solver="liblinear",
            random_state=int(seed),
            max_iter=200,
        )

    def fit_fn(model, X_tr, y_tr, *, groups=None, sample_weight=None, **kwargs):
        # Correct path: fit on true labels.
        model.fit(X_tr.to_numpy(), np.asarray(y_tr).ravel(), sample_weight=sample_weight)
        return model

    def fit_fn_permuted(model, X_tr, y_tr, *, groups=None, sample_weight=None, **kwargs):
        rng = np.random.default_rng(0)
        y_perm = rng.permutation(np.asarray(y_tr).ravel())
        model.fit(X_tr.to_numpy(), y_perm, sample_weight=sample_weight)
        return model

    def predict_fn(model, X_te):
        return model.predict_proba(X_te.to_numpy())[:, 1]

    base_cfg = NestedCVConfig(
        horizon=21,
        cost_bps=50.0,
        outer_params={"test_size": 20, "purge_gap": 10, "embargo": 0, "min_train_size": 80, "test_start": np.unique(groups)[180]},
        inner_params={"test_size": 15, "purge_gap": 10, "embargo": 0, "min_train_size": 40},
        seeds=[1],
        task="gate_classifier",
        feature_selection="importance",
        top_n_features=2,
        optuna_n_trials=0,
        enable_time_decay=True,
        half_life_days=126,
        gate_class_balance=False,
    )

    oof_good = run_outer_folds(
        config=base_cfg,
        X=X,
        y=y,
        groups=groups,
        outer_cv=outer_cv,
        inner_cv_factory=inner_cv_factory,
        model_factory=model_factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
    )
    auc_good = _auc(_resolve_oof_y_true(oof_good), oof_good["pred_mean"])
    assert auc_good > 0.60

    oof_bad = run_outer_folds(
        config=base_cfg,
        X=X,
        y=y,
        groups=groups,
        outer_cv=outer_cv,
        inner_cv_factory=inner_cv_factory,
        model_factory=model_factory,
        fit_fn=fit_fn_permuted,
        predict_fn=predict_fn,
    )
    auc_bad = _auc(_resolve_oof_y_true(oof_bad), oof_bad["pred_mean"])
    assert auc_bad < 0.58


def test_run_outer_folds_future_perturbation_invariance() -> None:
    # Changing future features must not affect OOF predictions for earlier folds.
    X, y, groups = _make_synthetic_gate_dataset(n_dates=260, n_sectors=4)
    cutoff = np.unique(groups)[210]

    outer_cv = PurgedWalkForwardCV(
        n_splits=2,
        test_size=20,
        purge_gap=10,
        min_train_size=80,
        test_start=np.unique(groups)[180],
    )

    def inner_cv_factory(config: NestedCVConfig, groups_inner: np.ndarray):
        return PurgedWalkForwardCV(
            n_splits=2,
            test_size=15,
            purge_gap=10,
            min_train_size=40,
        )

    from sklearn.linear_model import LogisticRegression

    def model_factory(seed: int, params: dict):
        return LogisticRegression(
            solver="liblinear",
            random_state=int(seed),
            max_iter=200,
        )

    def fit_fn(model, X_tr, y_tr, *, groups=None, sample_weight=None, **kwargs):
        model.fit(X_tr.to_numpy(), np.asarray(y_tr).ravel(), sample_weight=sample_weight)
        return model

    def predict_fn(model, X_te):
        return model.predict_proba(X_te.to_numpy())[:, 1]

    cfg = NestedCVConfig(
        horizon=21,
        cost_bps=50.0,
        outer_params={"test_size": 20, "purge_gap": 10, "embargo": 0, "min_train_size": 80, "test_start": np.unique(groups)[180]},
        inner_params={"test_size": 15, "purge_gap": 10, "embargo": 0, "min_train_size": 40},
        seeds=[1],
        task="gate_classifier",
        feature_selection="importance",
        top_n_features=2,
        optuna_n_trials=0,
        enable_time_decay=True,
        half_life_days=126,
        gate_class_balance=False,
    )

    oof1 = run_outer_folds(
        config=cfg,
        X=X,
        y=y,
        groups=groups,
        outer_cv=outer_cv,
        inner_cv_factory=inner_cv_factory,
        model_factory=model_factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
    )

    X2 = X.copy()
    future_mask = pd.to_datetime(y["Date"]) > pd.Timestamp(cutoff)
    X2.loc[future_mask.to_numpy(), "f_signal"] = X2.loc[future_mask.to_numpy(), "f_signal"] + 1000.0

    oof2 = run_outer_folds(
        config=cfg,
        X=X2,
        y=y,
        groups=groups,
        outer_cv=outer_cv,
        inner_cv_factory=inner_cv_factory,
        model_factory=model_factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
    )

    a = oof1.loc[pd.to_datetime(oof1["Date"]) <= pd.Timestamp(cutoff)].sort_values(["Date", "Sector"]).reset_index(drop=True)
    b = oof2.loc[pd.to_datetime(oof2["Date"]) <= pd.Timestamp(cutoff)].sort_values(["Date", "Sector"]).reset_index(drop=True)

    np.testing.assert_allclose(a["pred_mean"].to_numpy(), b["pred_mean"].to_numpy(), rtol=0.0, atol=1e-12)


def test_run_outer_folds_shifted_labels_collapse_signal() -> None:
    # If X and y are misaligned by +1 date, performance should collapse.
    X, y, groups = _make_synthetic_gate_dataset(n_dates=260, n_sectors=5)

    outer_cv = PurgedWalkForwardCV(
        n_splits=3,
        test_size=20,
        purge_gap=10,
        min_train_size=80,
        test_start=np.unique(groups)[180],
    )

    def inner_cv_factory(config: NestedCVConfig, groups_inner: np.ndarray):
        return PurgedWalkForwardCV(
            n_splits=2,
            test_size=15,
            purge_gap=10,
            min_train_size=40,
        )

    from sklearn.linear_model import LogisticRegression

    def model_factory(seed: int, params: dict):
        return LogisticRegression(
            solver="liblinear",
            random_state=int(seed),
            max_iter=200,
        )

    def fit_fn(model, X_tr, y_tr, *, groups=None, sample_weight=None, **kwargs):
        model.fit(X_tr.to_numpy(), np.asarray(y_tr).ravel(), sample_weight=sample_weight)
        return model

    def predict_fn(model, X_te):
        return model.predict_proba(X_te.to_numpy())[:, 1]

    cfg = NestedCVConfig(
        horizon=21,
        cost_bps=50.0,
        outer_params={"test_size": 20, "purge_gap": 10, "embargo": 0, "min_train_size": 80, "test_start": np.unique(groups)[180]},
        inner_params={"test_size": 15, "purge_gap": 10, "embargo": 0, "min_train_size": 40},
        seeds=[1],
        task="gate_classifier",
        feature_selection="importance",
        top_n_features=2,
        optuna_n_trials=0,
        enable_time_decay=True,
        half_life_days=126,
        gate_class_balance=False,
    )

    oof_good = run_outer_folds(
        config=cfg,
        X=X,
        y=y,
        groups=groups,
        outer_cv=outer_cv,
        inner_cv_factory=inner_cv_factory,
        model_factory=model_factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
    )
    auc_good = _auc(_resolve_oof_y_true(oof_good), oof_good["pred_mean"])

    # Shift y_gate forward by one business day within each sector.
    y_shift = y.copy()
    y_shift["Date"] = pd.to_datetime(y_shift["Date"])
    y_shift["y_gate"] = (
        y_shift.sort_values(["Sector", "Date"], kind="mergesort")
        .groupby("Sector", sort=False)["y_gate"]
        .shift(-1)
        .fillna(0)
        .astype(np.int8)
        .to_numpy()
    )

    oof_shift = run_outer_folds(
        config=cfg,
        X=X,
        y=y_shift,
        groups=groups,
        outer_cv=outer_cv,
        inner_cv_factory=inner_cv_factory,
        model_factory=model_factory,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
    )
    auc_shift = _auc(_resolve_oof_y_true(oof_shift), oof_shift["pred_mean"])

    assert auc_good > 0.60
    assert auc_shift < 0.58
