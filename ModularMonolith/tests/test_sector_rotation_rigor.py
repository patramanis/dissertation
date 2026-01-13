from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from ModularMonolith.data.dataset_shaping import ShapingResult
from ModularMonolith.src.backtesting.vectorized_backtester import VectorizedBacktester
from ModularMonolith.src.cv.run_nested_cv import _extract_feature_importance
from ModularMonolith.src.models.sector_rotation_trainer import SectorRotationTrainer


def _make_synthetic_shaping_result(*, cost_threshold: float = 0.001) -> ShapingResult:
    # Enough dates for horizon=21 CV defaults: min_train_size=252 and test_start=2016-01-01.
    # Need at least (min_train_size + purge_gap) unique dates BEFORE test_start.
    # Also extend beyond 2017 for the no-lookahead cutoff test.
    dates = pd.date_range("2014-01-01", periods=900, freq="B")
    sectors = ["XLB", "XLE", "XLF"]

    idx = pd.MultiIndex.from_product([dates, sectors], names=["Date", "Sector"])
    df = idx.to_frame(index=False)

    # Deterministic, time-varying label_excess with a regime change.
    # Force early period to be mostly <= cost_threshold so some folds may be single-class.
    t = (df["Date"].astype("int64") // 86_400_000_000_000).to_numpy(dtype=float)
    s = df["Sector"].astype("category").cat.codes.to_numpy(dtype=float)

    base = 0.0005 * np.sin(t / 7.0) + 0.0002 * (s - s.mean())
    bump = np.where(df["Date"] >= pd.Timestamp("2016-06-01"), 0.0030 * (1.0 + 0.1 * s), 0.0)
    label_excess = base + bump

    # Clamp early period to guarantee mostly-negatives.
    early_mask = df["Date"] < pd.Timestamp("2016-02-01")
    label_excess = np.where(early_mask, np.minimum(label_excess, cost_threshold * 0.5), label_excess)

    df["label_excess"] = label_excess

    # Simple deterministic features (no RNG).
    df["f1"] = np.sin(t / 11.0) + 0.05 * s
    df["f2"] = np.cos(t / 13.0) - 0.03 * s
    X = df[["f1", "f2"]].copy()

    y_gate = (df["label_excess"].to_numpy(dtype=float) > float(cost_threshold)).astype(np.int8)

    return ShapingResult(
        horizon=21,
        target_col="label_excess",
        features_path=Path("."),
        labels_path=Path("."),
        correlations_path=Path("."),
        full_df=df[["Date", "Sector", "label_excess"]].copy(),
        X=X,
        y_gate=y_gate,
        y_rank=np.zeros(len(df), dtype=float),
        group_sizes=np.ones(len(dates), dtype=np.int64),
        feature_cols=list(X.columns),
        dropped_rows_nan_target=0,
        dropped_dates_nan_target=0,
        dropped_dates_universe_policy=0,
        cost_threshold=float(cost_threshold),
    )


def test_extract_feature_importance_safe_for_unknown_models() -> None:
    class Dummy:
        pass

    feats = ["a", "b", "c"]
    imp = _extract_feature_importance(Dummy(), feats)
    assert list(imp.index) == feats
    assert np.isfinite(imp.to_numpy()).all()
    assert float(imp.sum()) == 0.0


def test_predict_classifier_respects_classes_order() -> None:
    # Regression test: if a classifier exposes classes_ in a non-standard order,
    # `predict_classifier` must still return P(class==1).
    from ModularMonolith.src.models.sector_rotation_trainer import predict_classifier

    class Dummy:
        def __init__(self):
            # Deliberately reversed order.
            self.classes_ = np.asarray([1, 0], dtype=np.int32)

        def predict_proba(self, X):
            n = len(X)
            # Column 0 corresponds to class==1, column 1 to class==0.
            p1 = np.full(n, 0.8, dtype=float)
            p0 = 1.0 - p1
            return np.column_stack([p1, p0])

    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    out = predict_classifier(Dummy(), X)
    assert out.shape == (len(X),)
    assert np.allclose(out, 0.8)


def test_determinism_deterministic_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _make_synthetic_shaping_result(cost_threshold=0.001)

    # Patch the loader inside the trainer module.
    import ModularMonolith.src.models.sector_rotation_trainer as srt

    monkeypatch.setattr(srt, "load_dual_model_dataset", lambda *args, **kwargs: data)

    def run_once() -> pd.DataFrame:
        trainer = SectorRotationTrainer(
            horizon=21,
            cost_threshold=float(data.cost_threshold),
            seeds=[42],
            optuna_n_trials=0,
            verbose=False,
            base_results_dir=str(tmp_path),
            deterministic_mode=True,
            use_gpu=False,
            drop_cs_constant_features=False,
            top_n_features=2,
            model_params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
        )
        res = trainer.run()
        return res.proba_oof[["Date", "Sector", "fold_id", "pred_mean", "pred_std", "y_gate", "label_excess"]].copy()

    oof1 = run_once().sort_values(["Date", "Sector", "fold_id"]).reset_index(drop=True)
    oof2 = run_once().sort_values(["Date", "Sector", "fold_id"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(oof1, oof2, check_exact=True)


def test_no_lookahead_negative_control(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = _make_synthetic_shaping_result(cost_threshold=0.001)

    cutoff = pd.Timestamp("2017-01-03")

    # First run (baseline)
    import ModularMonolith.src.models.sector_rotation_trainer as srt

    monkeypatch.setattr(srt, "load_dual_model_dataset", lambda *args, **kwargs: base)

    trainer1 = SectorRotationTrainer(
        horizon=21,
        cost_threshold=float(base.cost_threshold),
        seeds=[42],
        optuna_n_trials=0,
        verbose=False,
        base_results_dir=str(tmp_path / "r1"),
        deterministic_mode=True,
        use_gpu=False,
        drop_cs_constant_features=False,
        top_n_features=2,
        model_params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
    )
    res1 = trainer1.run()
    oof1 = res1.proba_oof.copy()

    # Second run: perturb ONLY data after cutoff.
    pert0 = _make_synthetic_shaping_result(cost_threshold=0.001)
    full_df_pert = pert0.full_df.copy()
    m = full_df_pert["Date"] > cutoff
    full_df_pert.loc[m, "label_excess"] = full_df_pert.loc[m, "label_excess"] + 0.05  # huge change
    y_gate_pert = (full_df_pert["label_excess"].to_numpy(dtype=float) > float(pert0.cost_threshold)).astype(np.int8)
    pert = replace(pert0, full_df=full_df_pert, y_gate=y_gate_pert)

    monkeypatch.setattr(srt, "load_dual_model_dataset", lambda *args, **kwargs: pert)

    trainer2 = SectorRotationTrainer(
        horizon=21,
        cost_threshold=float(pert.cost_threshold),
        seeds=[42],
        optuna_n_trials=0,
        verbose=False,
        base_results_dir=str(tmp_path / "r2"),
        deterministic_mode=True,
        use_gpu=False,
        drop_cs_constant_features=False,
        top_n_features=2,
        model_params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
    )
    res2 = trainer2.run()
    oof2 = res2.proba_oof.copy()

    # Compare only the portion up to cutoff.
    a = oof1.loc[pd.to_datetime(oof1["Date"]) <= cutoff].sort_values(["Date", "Sector", "fold_id"]).reset_index(drop=True)
    b = oof2.loc[pd.to_datetime(oof2["Date"]) <= cutoff].sort_values(["Date", "Sector", "fold_id"]).reset_index(drop=True)

    # If any future modification changes past OOF, we have leakage.
    pd.testing.assert_frame_equal(
        a[["Date", "Sector", "fold_id", "pred_mean", "pred_std"]],
        b[["Date", "Sector", "fold_id", "pred_mean", "pred_std"]],
        check_exact=True,
    )


def test_backtest_engine_internal_parity() -> None:
    # Minimal parity check: Backtester output equals recomputation from returned weights.
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    tickers = ["A", "B", "C"]

    returns = pd.DataFrame(
        {
            "A": np.linspace(0.001, 0.002, len(dates)),
            "B": np.linspace(-0.0005, 0.001, len(dates)),
            "C": np.linspace(0.0002, -0.0001, len(dates)),
        },
        index=dates,
    )

    preds = pd.DataFrame(
        {
            "A": np.linspace(0.0, 1.0, len(dates)),
            "B": np.linspace(1.0, 0.0, len(dates)),
            "C": 0.5,
        },
        index=dates,
    )

    bt = VectorizedBacktester(returns_df=returns)
    out = bt.run(predictions=preds, top_k=2, holding_period=1, cost_bps=0.0)

    weights = out.weights
    exec_w = weights.shift(1)

    strat = (exec_w * returns).sum(axis=1, min_count=1)
    strat = strat.dropna()

    pd.testing.assert_series_equal(out.strategy_returns, strat, check_names=False, check_exact=True)


def test_date_purity_across_folds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _make_synthetic_shaping_result(cost_threshold=0.001)

    import ModularMonolith.src.models.sector_rotation_trainer as srt

    monkeypatch.setattr(srt, "load_dual_model_dataset", lambda *args, **kwargs: data)

    trainer = SectorRotationTrainer(
        horizon=21,
        cost_threshold=float(data.cost_threshold),
        seeds=[42],
        optuna_n_trials=0,
        verbose=False,
        base_results_dir=str(tmp_path),
        deterministic_mode=True,
        use_gpu=False,
        drop_cs_constant_features=False,
        top_n_features=2,
        model_params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
    )
    res = trainer.run()
    oof = res.proba_oof
    date_fold_counts = oof.groupby("Date")["fold_id"].nunique()
    assert int(date_fold_counts.max()) == 1


def test_multi_seed_ensemble_produces_nonzero_pred_std(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _make_synthetic_shaping_result(cost_threshold=0.001)

    import ModularMonolith.src.models.sector_rotation_trainer as srt

    monkeypatch.setattr(srt, "load_dual_model_dataset", lambda *args, **kwargs: data)

    trainer = SectorRotationTrainer(
        horizon=21,
        cost_threshold=float(data.cost_threshold),
        seeds=[1, 2, 3],
        optuna_n_trials=0,
        verbose=False,
        base_results_dir=str(tmp_path),
        deterministic_mode=True,
        use_gpu=False,
        drop_cs_constant_features=False,
        top_n_features=2,
        model_params={
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    )
    res = trainer.run()
    oof = res.proba_oof
    assert float(pd.to_numeric(oof["pred_std"], errors="coerce").max()) > 0.0
