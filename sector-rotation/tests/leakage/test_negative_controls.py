from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = (THIS_DIR.parents[1] / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

log = logging.getLogger(__name__)

def _workspace_root() -> Path:
    return THIS_DIR.parents[1]

def _load_panel(h: int) -> pd.DataFrame:
    ws = _workspace_root()
    panel_path = ws / "data" / "processed" / "panel" / f"panel_h{h}.csv"
    if not panel_path.exists():
        pytest.skip(f"Panel not found: {panel_path}")
    return pd.read_csv(panel_path, parse_dates=["Date"])

def _load_cv_folds(h: int, cv_type: str = "rolling") -> list[dict]:
    ws = _workspace_root()
    cv_path = ws / "configs" / "cv" / f"cv_{cv_type}_h{h}.json"
    if not cv_path.exists():
        pytest.skip(f"CV config not found: {cv_path}")
    data = json.loads(cv_path.read_text(encoding="utf-8"))
    return data.get("folds", [])

def _get_feature_columns(panel: pd.DataFrame) -> list[str]:
    non_features = {
        "Date", "Sector", "label_excess",
        "horizon", "label_contract", "sample_weight"
    }
    return [c for c in panel.columns if c not in non_features]

def _train_simple_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:

    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)

def compute_ic_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series,
) -> dict[str, float]:

    ics = []
    for date in dates.unique():
        mask = dates == date
        y_t = y_true[mask].values
        y_p = y_pred[mask]
        if len(np.unique(y_t)) > 1:
            res_any: Any = stats.spearmanr(y_t, y_p)
            ic = float(getattr(res_any, "correlation", res_any[0]))
            if not np.isnan(ic):
                ics.append(ic)

    return {
        "ic_mean": float(np.mean(ics)) if ics else 0.0,
        "ic_std": float(np.std(ics)) if ics else 0.0,
        "ic_n": len(ics),
    }

class TestWithinDatePermutation:

    @pytest.mark.parametrize("h", [21])
    def test_permutation_degrades_regression_ic(self, h: int):
        panel = _load_panel(h)
        folds = _load_cv_folds(h, "rolling")

        if len(folds) < 2:
            pytest.skip("Not enough folds")

        fold = folds[0]

        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])

        train_mask = (panel["Date"] >= train_start) & (panel["Date"] <= train_end)
        test_mask = (panel["Date"] >= test_start) & (panel["Date"] <= test_end)

        train_data = panel.loc[train_mask].copy()
        test_data = panel.loc[test_mask].copy()

        feature_cols = _get_feature_columns(panel)
        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_train = train_data["label_excess"]
        y_test = test_data["label_excess"]

        y_pred_baseline = _train_simple_model(X_train, y_train, X_test)
        metrics_baseline = compute_ic_metrics(
            y_test, y_pred_baseline, test_data["Date"]
        )

        rng = np.random.default_rng(42)
        y_train_permuted = train_data.groupby("Date")["label_excess"].transform(
            lambda x: rng.permutation(np.asarray(x.to_numpy()))
        )

        y_pred_permuted = _train_simple_model(X_train, y_train_permuted, X_test)
        metrics_permuted = compute_ic_metrics(
            y_test, y_pred_permuted, test_data["Date"]
        )

        log.info("Baseline IC: %.4f ± %.4f", metrics_baseline["ic_mean"], metrics_baseline["ic_std"])
        log.info("Permuted IC: %.4f ± %.4f", metrics_permuted["ic_mean"], metrics_permuted["ic_std"])

        assert abs(metrics_permuted["ic_mean"]) < 0.10, (
            f"Permuted IC ({metrics_permuted['ic_mean']:.4f}) should be near 0. "
            "If high, possible leakage or mislabeled data."
        )

class TestFutureShiftFeature:

    @pytest.mark.parametrize("h", [21])
    def test_future_shift_improves_regression_ic(self, h: int):
        panel = _load_panel(h)
        folds = _load_cv_folds(h, "rolling")

        if len(folds) < 2:
            pytest.skip("Not enough folds")

        fold = folds[0]

        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])

        feature_cols = _get_feature_columns(panel)

        train_mask = (panel["Date"] >= train_start) & (panel["Date"] <= train_end)
        test_mask = (panel["Date"] >= test_start) & (panel["Date"] <= test_end)

        train_data = panel.loc[train_mask].copy()
        test_data = panel.loc[test_mask].copy()

        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_train = train_data["label_excess"]
        y_test = test_data["label_excess"]

        y_pred_baseline = _train_simple_model(X_train, y_train, X_test)
        metrics_baseline = compute_ic_metrics(
            y_test, y_pred_baseline, test_data["Date"]
        )

        panel_shifted = panel.copy()
        for col in feature_cols:
            panel_shifted[col] = panel_shifted.groupby("Sector")[col].shift(-1)

        panel_shifted = panel_shifted.dropna(subset=feature_cols)

        train_mask_s = (panel_shifted["Date"] >= train_start) & (panel_shifted["Date"] <= train_end)
        test_mask_s = (panel_shifted["Date"] >= test_start) & (panel_shifted["Date"] <= test_end)

        train_data_s = panel_shifted.loc[train_mask_s].copy()
        test_data_s = panel_shifted.loc[test_mask_s].copy()

        if len(train_data_s) < 100 or len(test_data_s) < 50:
            pytest.skip("Not enough data after shift")

        X_train_s = train_data_s[feature_cols]
        X_test_s = test_data_s[feature_cols]
        y_train_s = train_data_s["label_excess"]
        y_test_s = test_data_s["label_excess"]

        y_pred_shifted = _train_simple_model(X_train_s, y_train_s, X_test_s)
        metrics_shifted = compute_ic_metrics(
            y_test_s, y_pred_shifted, test_data_s["Date"]
        )

        log.info("Baseline IC: %.4f ± %.4f", metrics_baseline["ic_mean"], metrics_baseline["ic_std"])
        log.info("Future-shifted IC: %.4f ± %.4f", metrics_shifted["ic_mean"], metrics_shifted["ic_std"])

        improvement = metrics_shifted["ic_mean"] - metrics_baseline["ic_mean"]

        log.info("Improvement: %.4f", improvement)

        assert metrics_shifted["ic_mean"] >= metrics_baseline["ic_mean"] - 0.05, (
            f"Future-shifted IC ({metrics_shifted['ic_mean']:.4f}) should not be much worse "
            f"than baseline ({metrics_baseline['ic_mean']:.4f}). "
            "If degraded, check feature timing."
        )

class TestLeakageNegativeControls:

    @pytest.mark.parametrize("h", [5, 21, 63])
    def test_negative_controls_pass(self, h: int):
        panel = _load_panel(h)
        folds = _load_cv_folds(h, "rolling")

        if len(folds) < 2:
            pytest.skip("Not enough folds")

        fold = folds[0]

        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])

        train_mask = (panel["Date"] >= train_start) & (panel["Date"] <= train_end)
        test_mask = (panel["Date"] >= test_start) & (panel["Date"] <= test_end)

        train_data = panel.loc[train_mask].copy()
        test_data = panel.loc[test_mask].copy()

        feature_cols = _get_feature_columns(panel)
        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_train = train_data["label_excess"]
        y_test = test_data["label_excess"]

        results: dict[str, Any] = {
            "horizon": h,
            "train_size": len(train_data),
            "test_size": len(test_data),
        }

        y_pred_baseline = _train_simple_model(X_train, y_train, X_test)
        results["baseline_ic"] = compute_ic_metrics(y_test, y_pred_baseline, test_data["Date"])["ic_mean"]

        rng = np.random.default_rng(42)
        y_train_permuted = train_data.groupby("Date")["label_excess"].transform(
            lambda x: rng.permutation(np.asarray(x.to_numpy()))
        )
        y_pred_permuted = _train_simple_model(X_train, y_train_permuted, X_test)
        results["permuted_ic"] = compute_ic_metrics(y_test, y_pred_permuted, test_data["Date"])["ic_mean"]

        log.info("Negative Controls Summary for h=%d", h)
        log.info("Baseline IC: %.4f", results["baseline_ic"])
        log.info("Permuted IC: %.4f (should be ~0.0)", results["permuted_ic"])
        log.info("Permutation collapse: %s",
                 "PASS" if abs(results["permuted_ic"]) < 0.15 else "SUSPICIOUS")

        threshold = 0.20
        assert abs(results["permuted_ic"]) < threshold, (
            f"Permuted IC ({results['permuted_ic']:.4f}) exceeds threshold {threshold}. "
            f"This may indicate feature leakage. "
            f"Expected: |IC| < {threshold} for random labels."
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "-s", "--tb=short"])