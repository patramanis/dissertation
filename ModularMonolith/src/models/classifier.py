"""
Single Classifier Training System for Sector Rotation.

Implements pseudo-ranking strategy using a single XGBoostClassifier:
- Train ONE classifier to predict binary target (excess_return > cost_threshold)
- Use predict_proba(:, 1) as ranking score
- Sort sectors by probability score (descending) for allocation

Key design principles:
- Simplified architecture (no dual-stage gate+ranker)
- Nested CV for all tuning (purged walk-forward)
- Multi-seed ensemble for uncertainty quantification
- Fold-safe thresholds (computed from train only)

Usage:
    from ModularMonolith.src.models.sector_rotation_trainer import SectorRotationTrainer
    
    trainer = SectorRotationTrainer(horizon=21)
    results = trainer.run()
    results.save("Results/")
"""
from __future__ import annotations

import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

# Local imports
from ModularMonolith.data.dataset_shaping import (
    HORIZONS,
    ShapingResult,
    load_dual_model_dataset,
)
from ModularMonolith.src.config.cv_config import get_cv_config
from ModularMonolith.src.cv.purged_walk_forward_cv import PurgedWalkForwardCV
from ModularMonolith.src.cv.run_nested_cv import (
    FoldResult,
    NestedCVConfig,
    run_outer_folds,
)
from ModularMonolith.src.reporting.drift import DriftDetector
from ModularMonolith.src.reporting.metrics_rank import (
    RankMetrics,
    gate_classification_metrics,
)
from ModularMonolith.src.results.run_manager import RunManager


# Suppress XGBoost verbosity
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def _as_datetime_index(values: Any) -> pd.DatetimeIndex:
    """Normalize to tz-naive DatetimeIndex (UTC-stripped) for stable time ordering."""
    dti = pd.to_datetime(values, errors="raise")
    if getattr(dti, "tz", None) is not None:
        dti = dti.tz_localize(None)
    return pd.DatetimeIndex(dti)


def _validate_shaping_result(sr: ShapingResult, X: pd.DataFrame, *, cost_threshold: float) -> None:
    """Validate dataset shaping output contract and basic hygiene.

    This is intentionally strict and fail-fast: downstream CV/metrics assume these invariants.
    """
    if sr is None:
        raise ValueError("ShapingResult is None")
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame; got {type(X)}")
    if sr.full_df is None or not isinstance(sr.full_df, pd.DataFrame):
        raise TypeError("sr.full_df must be a pandas DataFrame")

    # Required columns in full_df
    required_cols = {"Date", "label_excess"}
    missing = sorted(required_cols - set(sr.full_df.columns))
    if missing:
        raise ValueError(f"sr.full_df missing required columns: {missing}")

    # Contract: lengths must match
    n_full = int(len(sr.full_df))
    n_x = int(len(sr.X))
    n_gate = int(np.asarray(sr.y_gate).shape[0])
    if not (n_full == n_x == n_gate):
        raise ValueError(
            "ShapingResult length mismatch: "
            f"len(full_df)={n_full}, len(X)={n_x}, len(y_gate)={n_gate}"
        )

    # Date: must parse to tz-naive datetime64[ns]
    try:
        dti = _as_datetime_index(sr.full_df["Date"]).to_numpy(dtype="datetime64[ns]")
    except Exception as e:
        raise ValueError(f"Failed to coerce Date to datetime64[ns]: {e}") from e
    if dti.size != n_full:
        raise ValueError(f"Date length mismatch: len(Date)={int(dti.size)} len(full_df)={n_full}")

    # label_excess: numeric float with no NaN/inf
    le = pd.to_numeric(sr.full_df["label_excess"], errors="coerce").astype(float)
    bad = ~np.isfinite(le.to_numpy(dtype=float))
    if bool(np.any(bad)):
        bad_rows = sr.full_df.loc[pd.Series(bad), ["Date", "label_excess"]].head(25)
        raise ValueError(
            "label_excess contains NaN/inf after numeric coercion; sample rows: "
            f"{bad_rows.to_dict(orient='records')}"
        )

    # y_gate: must be exactly {0,1}. Allow bool -> int8.
    y = np.asarray(sr.y_gate).ravel()
    if y.dtype == bool:
        yv = y.astype(np.int8, copy=False)
    else:
        if y.size == 0:
            raise ValueError("y_gate is empty")
        if np.any(~np.isfinite(y.astype(float, copy=False))):
            raise ValueError("y_gate contains NaN/inf")
        yv = y

    uniq = np.unique(yv)
    ok = bool(np.all((uniq == 0) | (uniq == 1)))
    if not ok:
        sample = uniq[:20]
        raise ValueError(
            "y_gate must be exactly {0,1} (bool allowed); "
            f"got unique sample={sample.tolist()}, dtype={str(y.dtype)}"
        )

    # Duplicate key check for (Date, Sector) if Sector present
    if "Sector" in sr.full_df.columns:
        sector = sr.full_df["Sector"].astype("string")
        keys = pd.DataFrame({"Date": pd.Series(dti), "Sector": sector})
        dup_mask = keys.duplicated(subset=["Date", "Sector"], keep=False)
        if bool(dup_mask.any()):
            sample = sr.full_df.loc[dup_mask, ["Date", "Sector", "label_excess"]].head(25)
            raise ValueError(
                "Duplicate (Date,Sector) rows detected; this breaks ranking/CV invariants. "
                f"Sample: {sample.to_dict(orient='records')}"
            )

    # X hygiene: replace +/-inf with NaN (XGBoost handles NaN), but never allow inf.
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if np.isinf(X.to_numpy(dtype=float, copy=False)).any():
        raise ValueError("X contains +/-inf after replacement")

    # Strict numeric-only features
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        dtypes = {str(c): str(X[c].dtype) for c in non_numeric[:200]}
        raise TypeError(
            "Non-numeric feature columns detected (strict mode). "
            f"First columns/dtypes: {dtypes}"
        )

    # cost_threshold contract
    if not np.isclose(float(sr.cost_threshold), float(cost_threshold), rtol=0.0, atol=1e-12):
        raise ValueError(
            "cost_threshold mismatch between trainer and dataset. "
            f"trainer.cost_threshold={float(cost_threshold)!r} dataset.cost_threshold={float(sr.cost_threshold)!r}"
        )


def _input_schema_snapshot(*, sr: ShapingResult, X: pd.DataFrame) -> dict[str, Any]:
    """Create a compact schema snapshot suitable for diagnostics."""
    full_df = sr.full_df
    out: dict[str, Any] = {
        "n_rows": int(len(full_df)),
        "n_features": int(X.shape[1]),
        "full_df_columns": list(map(str, full_df.columns)),
        "full_df_dtypes": {str(c): str(full_df[c].dtype) for c in full_df.columns},
        "X_columns": list(map(str, X.columns)),
        "X_dtypes": {str(c): str(X[c].dtype) for c in X.columns},
        "has_sector": bool("Sector" in full_df.columns),
    }
    try:
        dti = _as_datetime_index(full_df["Date"]).to_numpy(dtype="datetime64[ns]")
        out["date_min"] = pd.Timestamp(dti.min()).isoformat() if dti.size else None
        out["date_max"] = pd.Timestamp(dti.max()).isoformat() if dti.size else None
        out["n_unique_dates"] = int(pd.Index(dti).nunique())
    except Exception:
        out["date_min"] = None
        out["date_max"] = None
        out["n_unique_dates"] = None

    if "Sector" in full_df.columns:
        try:
            sector = full_df["Sector"].astype("string")
            out["n_unique_sectors"] = int(sector.nunique(dropna=False))
        except Exception:
            out["n_unique_sectors"] = None

        try:
            keys = pd.DataFrame({"Date": pd.Series(_as_datetime_index(full_df["Date"]).to_numpy(dtype="datetime64[ns]")), "Sector": sector})
            out["has_duplicates_date_sector"] = bool(keys.duplicated(["Date", "Sector"], keep=False).any())
        except Exception:
            out["has_duplicates_date_sector"] = None

    # Feature hygiene
    try:
        out["x_has_inf"] = bool(np.isinf(X.to_numpy(dtype=float, copy=False)).any())
    except Exception:
        out["x_has_inf"] = None
    out["x_has_nan"] = bool(X.isna().any().any())
    out["non_numeric_feature_cols"] = [str(c) for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return out


def _save_df_parquet_or_csv(
    *,
    run_manager: RunManager,
    df: pd.DataFrame,
    name: str,
    subfolder: str = "diagnostic",
) -> None:
    """Best-effort parquet save (preferred), with guaranteed CSV fallback."""
    if run_manager.run_dir is None:
        raise RuntimeError("RunManager has no run_dir")

    # Guaranteed CSV fallback: write directly to filesystem (do not depend on RunManager).
    out_dir = (run_manager.run_dir / subfolder)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_name = Path(name).name.replace(".parquet", "") + ".csv"
    csv_path = out_dir / csv_name
    tmp_csv_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    df.to_csv(tmp_csv_path, index=False)
    tmp_csv_path.replace(csv_path)

    # Attempt parquet (optional).
    try:
        out_dir = (run_manager.run_dir / subfolder)
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_name = Path(name).name
        if not parquet_name.endswith(".parquet"):
            parquet_name = parquet_name + ".parquet"
        parquet_path = out_dir / parquet_name
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # Non-fatal: parquet engine may be missing.
        return


class _ConstantProbaModel:
    """Dummy classifier for degenerate folds (single-class train)."""

    def __init__(self, proba_pos: float) -> None:
        self.proba_pos = float(np.clip(proba_pos, 0.0, 1.0))
        # Mirror sklearn classifiers: columns of predict_proba align with classes_.
        self.classes_ = np.asarray([0, 1], dtype=np.int32)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        p1 = np.full(n, self.proba_pos, dtype=np.float32)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    # Minimal sklearn-like API for downstream tooling.
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"proba_pos": self.proba_pos}

    def set_params(self, **params: Any) -> "_ConstantProbaModel":
        if "proba_pos" in params:
            self.proba_pos = float(np.clip(float(params["proba_pos"]), 0.0, 1.0))
        return self


# =============================================================================
# GPU Detection and XGBoost Configuration
# =============================================================================

from functools import lru_cache

@lru_cache(maxsize=2)
def _detect_gpu() -> dict[str, Any]:
    """Detect GPU availability and return XGBoost device config."""
    try:
        import xgboost as xgb
        
        # Try CUDA first
        try:
            # XGBoost 2.0+ syntax
            test_params = {"device": "cuda", "tree_method": "hist"}
            xgb.XGBClassifier(**test_params, n_estimators=1).fit(
                np.array([[1, 2], [3, 4]]),
                np.array([0, 1]),
            )
            return {"device": "cuda", "tree_method": "hist"}
        except Exception:
            pass
        
        # Fallback to gpu_hist (older XGBoost)
        try:
            test_params = {"tree_method": "gpu_hist"}
            xgb.XGBClassifier(**test_params, n_estimators=1).fit(
                np.array([[1, 2], [3, 4]]),
                np.array([0, 1]),
            )
            return {"tree_method": "gpu_hist"}
        except Exception:
            pass
        
    except ImportError:
        pass
    
    # CPU fallback (no `device` to support older XGBoost versions)
    return {"tree_method": "hist"}


@lru_cache(maxsize=2)
def _get_device_config(use_gpu: bool) -> dict[str, Any]:
    """Cached GPU/CPU device configuration."""
    if not use_gpu:
        # CPU fallback (no `device` to support older XGBoost versions)
        return {"tree_method": "hist"}
    return _detect_gpu()

def _get_xgb_base_params(*, use_gpu: bool = True, deterministic: bool = False) -> dict[str, Any]:
    """Get baseline XGBoost parameters with device config."""
    device_cfg = _get_device_config(use_gpu)
    
    params = {
        **device_cfg,
        "random_state": 42,
        "n_jobs": 1 if deterministic else -1,
    }
    
    return params


# =============================================================================
# Default Model Parameters
# =============================================================================

DEFAULT_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
}


# =============================================================================
# Model Fitting Functions
# =============================================================================

def fit_classifier(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    **fit_kwargs,
) -> Any:
    """
    Fit XGBClassifier.
    
    Args:
        model: XGBClassifier instance
        X: Feature matrix
        y: Binary target
        fit_kwargs: Additional fit arguments (sample_weight, eval_set, etc.)
                    Note: 'groups' is ignored (XGBClassifier doesn't use it)
    
    Returns:
        Fitted model
    """
    X_np = X.to_numpy(dtype=np.float32, copy=False) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)

    y_arr = y.to_numpy(copy=False) if isinstance(y, pd.Series) else np.asarray(y)
    y_raw = np.asarray(y_arr).ravel()
    if y_raw.size == 0:
        raise ValueError("y is empty")
    if np.any(~np.isfinite(y_raw)):
        raise ValueError("y contains NaN/inf")

    # CRITICAL: validate exact {0,1} BEFORE casting.
    # Otherwise e.g. 0.2/0.7/-1 become 0 and silently corrupt labels.
    ok = (y_raw == 0) | (y_raw == 1)
    if not bool(np.all(ok)):
        bad = y_raw[~ok]
        sample = np.unique(bad)[:10]
        raise ValueError(
            "y must be exactly 0/1 before casting; "
            f"got min={float(np.min(y_raw))}, max={float(np.max(y_raw))}, bad_sample={sample.tolist()}"
        )

    y_int = y_raw.astype(np.int32, copy=False)
    uniq = np.unique(y_int)

    # Degenerate fold safety: if only one class is present, return constant predictor.
    if uniq.size < 2:
        # Base-rate probability for class 1 in train.
        p1 = float(np.mean(y_int == 1))
        return _ConstantProbaModel(proba_pos=p1)

    # Remove None values and 'groups' (not used by XGBClassifier)
    fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None and k != "groups"}

    # Normalize sample_weight if present
    if "sample_weight" in fit_kwargs and fit_kwargs["sample_weight"] is not None:
        fit_kwargs["sample_weight"] = np.asarray(fit_kwargs["sample_weight"]).ravel().astype(np.float32)

    model.fit(X_np, y_int, **fit_kwargs)
    return model


def predict_classifier(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Get probabilities from classifier.
    
    Args:
        model: Fitted XGBClassifier
        X: Feature matrix
    
    Returns:
        Probability scores (predict_proba[:,  1])
    """
    X_np = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
    proba = np.asarray(model.predict_proba(X_np))

    # Robustly select the probability column for the positive class (label==1).
    # Do not assume column order is [0, 1]; use model.classes_ when available.
    if proba.ndim == 2 and proba.shape[1] >= 2 and hasattr(model, "classes_"):
        classes = np.asarray(getattr(model, "classes_"))
        try:
            pos_idx = int(np.where(classes == 1)[0][0])
            return proba[:, pos_idx]
        except Exception:
            # Fall back to sklearn convention if classes_ is missing/malformed.
            return proba[:, 1]

    # Fallbacks: assume second column is positive class; or already a 1D score.
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return np.asarray(proba).ravel()


# =============================================================================
# Results Dataclass
# =============================================================================

@dataclass
class SectorRotationResult:
    """Container for training results."""
    
    # Metadata
    horizon: int
    seeds: list[int]
    n_dates: int
    n_observations: int
    
    # OOF predictions
    proba_oof: pd.DataFrame  # Date, Sector, pred_mean, pred_std, y_true, label_excess
    
    # Metrics
    metrics: RankMetrics
    metrics_by_regime: dict[str, RankMetrics] | None
    
    # Feature importance
    feature_importance: pd.Series
    
    # Drift report
    drift_report: dict[str, Any] | None
    
    # Fold results
    fold_results: list[dict[str, Any]]
    
    # Timing
    training_time_seconds: float

    # Gate diagnostics
    gate_metrics: dict[str, float] | None = None
    gate_polarity_flipped: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "horizon": self.horizon,
            "seeds": self.seeds,
            "n_dates": self.n_dates,
            "n_observations": self.n_observations,
            "metrics": {
                "rank_ic": self.metrics.rank_ic,
                "rank_ic_std": self.metrics.rank_ic_std,
                "rank_ic_ir": self.metrics.rank_ic_ir,
                "precision_at_1": self.metrics.precision_at_1,
                "precision_at_3": self.metrics.precision_at_3,
                "lift_at_1": self.metrics.lift_at_1,
                "lift_at_3": self.metrics.lift_at_3,
                "uplift_at_1": self.metrics.uplift_at_1,
                "uplift_at_3": self.metrics.uplift_at_3,
                "hit_rate_top1": self.metrics.hit_rate_top1,
                "hit_rate_top3": self.metrics.hit_rate_top3,
            },
            "gate_metrics": dict(self.gate_metrics) if isinstance(self.gate_metrics, dict) else None,
            "gate_polarity_flipped": bool(self.gate_polarity_flipped),
            "training_time_seconds": self.training_time_seconds,
        }


def _rank_metrics_high_is_better(
    *,
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    cost_threshold: float,
) -> RankMetrics:
    """Compute rank metrics assuming higher score = better."""
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true).ravel(),
            "s": np.asarray(y_score).ravel(),
            "g": np.asarray(groups).ravel(),
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "s", "g"])
    if df.empty:
        return RankMetrics(
            rank_ic=0.0,
            rank_ic_std=0.0,
            rank_ic_ir=0.0,
            hit_rate_top1=0.0,
            hit_rate_top3=0.0,
            precision_at_1=0.0,
            precision_at_3=0.0,
            lift_at_1=0.0,
            lift_at_3=0.0,
            uplift_at_1=0.0,
            uplift_at_3=0.0,
            n_dates=0,
            n_observations=0,
        )

    ics: list[float] = []
    prec1: list[float] = []
    prec3: list[float] = []
    hr1 = 0
    hr3 = 0
    u1: list[float] = []
    u3: list[float] = []

    base_rate = float(np.mean(df["y"].to_numpy(dtype=float) > float(cost_threshold)))
    base_rate = max(base_rate, 1e-12)

    n_dates = 0
    for _, sub in df.groupby("g", sort=False):
        n_dates += 1
        if len(sub) < 2:
            continue

        yv = sub["y"].to_numpy(dtype=float)
        sv = sub["s"].to_numpy(dtype=float)
        if np.nanstd(yv) > 0.0 and np.nanstd(sv) > 0.0:
            rho = pd.Series(yv).corr(pd.Series(sv), method="spearman")
            if rho is not None and np.isfinite(float(rho)):
                ics.append(float(rho))

        # Top-k selection: highest score
        sub_sorted = sub.sort_values("s", ascending=False, kind="mergesort")
        y_sorted = sub_sorted["y"].to_numpy(dtype=float)

        # k=1
        top1 = y_sorted[:1]
        prec1.append(float(np.mean(top1 > float(cost_threshold))))
        u1.append(float(np.mean(top1) - float(np.mean(y_sorted))))

        # k=3
        k3 = min(3, len(y_sorted))
        top3 = y_sorted[:k3]
        prec3.append(float(np.mean(top3 > float(cost_threshold))))
        u3.append(float(np.mean(top3) - float(np.mean(y_sorted))))

        # Hit-rate@k: overlap with actual top-k by y
        actual_sorted = sub.sort_values("y", ascending=False, kind="mergesort")
        actual_top1_idx = set(actual_sorted.index[:1])
        actual_top3_idx = set(actual_sorted.index[:k3])
        pred_top1_idx = set(sub_sorted.index[:1])
        pred_top3_idx = set(sub_sorted.index[:k3])
        hr1 += int(len(pred_top1_idx & actual_top1_idx) > 0)
        hr3 += int(len(pred_top3_idx & actual_top3_idx) > 0)

    mean_ic = float(np.mean(ics)) if ics else 0.0
    std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    ic_ir = float(mean_ic / std_ic) if std_ic > 0 else 0.0

    p1 = float(np.mean(prec1)) if prec1 else 0.0
    p3 = float(np.mean(prec3)) if prec3 else 0.0
    l1 = float((p1 / base_rate) - 1.0)
    l3 = float((p3 / base_rate) - 1.0)

    return RankMetrics(
        rank_ic=mean_ic,
        rank_ic_std=std_ic,
        rank_ic_ir=ic_ir,
        hit_rate_top1=float(hr1) / float(max(1, n_dates)),
        hit_rate_top3=float(hr3) / float(max(1, n_dates)),
        precision_at_1=p1,
        precision_at_3=p3,
        lift_at_1=l1,
        lift_at_3=l3,
        uplift_at_1=float(np.mean(u1)) if u1 else 0.0,
        uplift_at_3=float(np.mean(u3)) if u3 else 0.0,
        n_dates=int(n_dates),
        n_observations=int(len(df)),
    )


def _rank_metrics_by_regime_high_is_better(
    oof: pd.DataFrame,
    *,
    regime_col: str,
    cost_threshold: float,
    y_col: str = "label_excess",
    score_col: str = "pred_mean",
    group_col: str = "Date",
) -> dict[str, RankMetrics]:
    out: dict[str, RankMetrics] = {}
    if regime_col not in oof.columns:
        return out
    for reg, sub in oof.groupby(regime_col, sort=False):
        out[str(reg)] = _rank_metrics_high_is_better(
            y_true=sub[y_col],
            y_score=sub[score_col],
            groups=sub[group_col],
            cost_threshold=cost_threshold,
        )
    return out


# =============================================================================
# Main Trainer Class
# =============================================================================

class SectorRotationTrainer:
    """
    Single Classifier Training System for Sector Rotation.
    
    Implements pseudo-ranking with one XGBoostClassifier.
    Uses predict_proba as ranking score for sector selection.
    """
    
    def __init__(
        self,
        horizon: int,
        *,
        seeds: list[int] | None = None,
        model_params: dict[str, Any] | None = None,
        feature_selection: Literal["importance", "corr"] = "importance",
        top_n_features: int | None = None,
        optuna_n_trials: int = 25,
        use_gpu: bool = True,
        verbose: bool = True,
        base_results_dir: str = "Results",
        auto_generate_plots: bool = True,
        auto_write_printable: bool = True,
        auto_run_backtest: bool = True,
        auto_save_models: bool = True,
        backtest_top_k: int = 3,
        backtest_holding_period: int = 1,
        backtest_cost_bps: float = 0.0,
        enable_rolling_zscore: bool = False,
        rolling_zscore_window: int = 252,
        rolling_zscore_min_periods: int = 60,
        cost_threshold: float | None = None,
        drop_cs_constant_features: bool = True,
        cs_constant_eps: float = 1e-6,
        ts_constant_eps: float = 1e-8,
        min_features_after_cs_filter: int = 10,
        deterministic_mode: bool = False,
        rank_eval_target: Literal["label_excess", "y_gate"] = "label_excess",
        enforce_date_purity: bool = True,
        enforce_ensemble_diversity: bool = True,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            horizon: Prediction horizon (5, 21, or 63 days)
            seeds: Random seeds for ensemble (default: 10 seeds)
            model_params: XGBClassifier parameters
            feature_selection: "importance" or "corr" based selection
            top_n_features: Number of features to select
            optuna_n_trials: Number of Optuna trials (0 to disable)
            use_gpu: Whether to use GPU acceleration
            verbose: Print progress
            base_results_dir: Base directory for results
            cost_threshold: Cost threshold for binary labels (default: 0.001 = 10bps)
            deterministic_mode: Force CPU, n_jobs=1 for bitwise reproducibility
        """
        if horizon not in HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}. Expected {HORIZONS}")
        
        self.horizon = horizon
        self.seeds = seeds or list(range(10))
        self.model_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
        self.feature_selection = feature_selection

        # Default feature budget by horizon
        default_top_n_by_horizon = {5: 10, 21: 25, 63: 25}
        if top_n_features is None:
            top_n_features = int(default_top_n_by_horizon.get(int(horizon), 25))
        self.top_n_features = int(top_n_features)
        if self.top_n_features <= 0:
            raise ValueError("top_n_features must be positive")
        
        self.optuna_n_trials = optuna_n_trials
        self.deterministic_mode = bool(deterministic_mode)
        if self.deterministic_mode:
            self.use_gpu = False
            if verbose:
                print("Deterministic mode: forcing CPU, n_jobs=1")
        else:
            self.use_gpu = use_gpu
        
        self.verbose = verbose
        self.base_results_dir = base_results_dir
        self.rank_eval_target = str(rank_eval_target)
        if self.rank_eval_target not in {"label_excess", "y_gate"}:
            raise ValueError("rank_eval_target must be one of: 'label_excess', 'y_gate'")
        self.enforce_date_purity = bool(enforce_date_purity)
        self.enforce_ensemble_diversity = bool(enforce_ensemble_diversity)
        self.auto_generate_plots = bool(auto_generate_plots)
        self.auto_write_printable = bool(auto_write_printable)
        self.auto_run_backtest = bool(auto_run_backtest)
        self.auto_save_models = bool(auto_save_models)
        self.backtest_top_k = int(backtest_top_k)
        self.backtest_holding_period = int(backtest_holding_period)
        self.backtest_cost_bps = float(backtest_cost_bps)

        # Rolling normalization
        self.enable_rolling_zscore = bool(enable_rolling_zscore)
        self.rolling_zscore_window = int(rolling_zscore_window)
        self.rolling_zscore_min_periods = int(rolling_zscore_min_periods)

        # Cost threshold (default 10bps)
        self.cost_threshold = float(cost_threshold) if cost_threshold is not None else 0.001

        # Cross-sectional hygiene: drop features with ~0 within-date variance.
        self.drop_cs_constant_features = bool(drop_cs_constant_features)
        self.cs_constant_eps = float(cs_constant_eps)
        self.ts_constant_eps = float(ts_constant_eps)
        self.min_features_after_cs_filter = int(min_features_after_cs_filter)

        self.data: ShapingResult | None = None
        self.run_manager: RunManager | None = None
        self._inner_cv_fallback_events: list[dict[str, Any]] = []

    def _log(self, msg: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            ts = datetime.now(timezone.utc).isoformat()
            print(f"[SectorRotationTrainer] {msg}")

    def run(self) -> SectorRotationResult:
        """
        Execute full training pipeline.
        
        Returns:
            SectorRotationResult with OOF predictions, metrics, and artifacts
        """
        start_time = datetime.now(timezone.utc)
        self._log(f"Starting training at {start_time.isoformat()}")

        # Keep references for fail-fast diagnostics.
        X_work: pd.DataFrame | None = None
        full_df_work: pd.DataFrame | None = None
        y_gate_work: np.ndarray | None = None
        y_rank_work: np.ndarray | None = None
        X_final: pd.DataFrame | None = None
        full_df_ord: pd.DataFrame | None = None
        y_gate_ord: np.ndarray | None = None
        y_rank_ord: np.ndarray | None = None
        proba_oof: pd.DataFrame | None = None
        fold_results: list[FoldResult] = []
        cv_config: dict[str, Any] | None = None
        cs_filter_report: dict[str, Any] | None = None
        resolved_device_cfg: dict[str, Any] | None = None

        self.run_manager = RunManager(base_dir=self.base_results_dir)

        try:
            # Linear pipeline: validate -> CS-filter -> deterministic ordering -> create_run -> train -> metrics/diagnostics -> save
            self._log(f"Loading dataset for horizon={self.horizon}...")
            self.data = load_dual_model_dataset(
                self.horizon,
                cost_threshold=float(self.cost_threshold),
                verbose=self.verbose,
            )
            if self.data is None:
                raise RuntimeError("Failed to load dataset")

            cv_config = get_cv_config(self.horizon)

            # Copies + strict validation
            X_work = self.data.X.copy()
            full_df_work = self.data.full_df.copy()
            y_gate_work = np.asarray(self.data.y_gate)
            y_rank_work = None
            if hasattr(self.data, "y_rank") and self.data.y_rank is not None:
                y_rank_work = np.asarray(self.data.y_rank)

            _validate_shaping_result(self.data, X_work, cost_threshold=float(self.cost_threshold))

            # Canonical types
            full_df_work["Date"] = _as_datetime_index(full_df_work["Date"]).to_numpy(dtype="datetime64[ns]")
            full_df_work["label_excess"] = pd.to_numeric(full_df_work["label_excess"], errors="raise").astype(float)
            if "Sector" in full_df_work.columns:
                full_df_work["Sector"] = full_df_work["Sector"].astype("string")
            if y_gate_work.dtype == bool:
                y_gate_work = y_gate_work.astype(np.int8, copy=False)
            else:
                y_gate_work = y_gate_work.astype(np.int8, copy=False)

            input_schema = _input_schema_snapshot(sr=self.data, X=X_work)

            # C) CS-constant filter (leakage-safe) + report
            X_tmp = X_work
            if self.drop_cs_constant_features:
                test_start_raw = cv_config.get("test_start") if cv_config else None
                if not test_start_raw:
                    raise ValueError(
                        "cv_config must define test_start for leakage-safe pre-train CS-constant hygiene. "
                        "Set MM_CV_TEST_START or disable drop_cs_constant_features."
                    )

                dates_all = _as_datetime_index(full_df_work["Date"]).to_series(index=X_tmp.index)
                test_start = pd.Timestamp(test_start_raw)
                pretrain_mask = dates_all < test_start

                X_pre = X_tmp.loc[pretrain_mask]
                d_pre = dates_all.loc[pretrain_mask]
                if X_pre.empty or int(d_pre.nunique()) < 2:
                    raise ValueError(
                        "Pre-train window has insufficient data for CS-constant hygiene. "
                        f"test_start={str(test_start_raw)!r}, pretrain_rows={int(X_pre.shape[0])}, pretrain_dates={int(d_pre.nunique())}."
                    )

                sizes = d_pre.value_counts(dropna=False)
                ok_dates = sizes[sizes >= 2].index
                X_cs = X_pre.loc[d_pre.isin(ok_dates)]
                d_cs = d_pre.loc[d_pre.isin(ok_dates)]
                if X_cs.empty or int(d_cs.nunique()) < 2:
                    raise ValueError(
                        "Pre-train window has insufficient cross-sectional coverage for CS-constant hygiene. "
                        f"eligible_rows={int(X_cs.shape[0])}, eligible_dates={int(d_cs.nunique())}."
                    )

                cs_std = X_cs.groupby(d_cs, sort=False).std(ddof=0)
                cs_std = cs_std.replace([np.inf, -np.inf], np.nan)
                finite_frac = cs_std.notna().mean(axis=0)
                mean_cs_std = cs_std.mean(axis=0, skipna=True)

                # Time variation proxy: per-date mean (panel collapse) std over time.
                ts_mean = X_cs.groupby(d_cs, sort=False).mean(numeric_only=True)
                ts_mean = ts_mean.replace([np.inf, -np.inf], np.nan)
                ts_std = ts_mean.std(axis=0, ddof=0, skipna=True)

                if float(finite_frac.max()) < 0.8 or bool(mean_cs_std.isna().all()):
                    raise ValueError("CS std degenerate; check data alignment")

                cs_ok = mean_cs_std > float(self.cs_constant_eps)
                ts_ok = ts_std > float(self.ts_constant_eps)
                keep_mask = (finite_frac >= 0.8) & (cs_ok | ts_ok)
                keep = mean_cs_std.index[keep_mask].tolist()
                dropped_feats = [c for c in X_tmp.columns if c not in set(keep)]

                cs_filter_report = {
                    "test_start": str(test_start_raw),
                    "pretrain_rows": int(X_pre.shape[0]),
                    "pretrain_dates": int(d_pre.nunique()),
                    "eligible_rows": int(X_cs.shape[0]),
                    "eligible_dates": int(d_cs.nunique()),
                    "cs_constant_eps": float(self.cs_constant_eps),
                    "ts_constant_eps": float(self.ts_constant_eps),
                    "finite_frac_summary": {
                        "min": float(np.nanmin(finite_frac.to_numpy())),
                        "p50": float(np.nanmedian(finite_frac.to_numpy())),
                        "mean": float(np.nanmean(finite_frac.to_numpy())),
                        "max": float(np.nanmax(finite_frac.to_numpy())),
                    },
                    "mean_cs_std_summary": {
                        "min": float(np.nanmin(mean_cs_std.to_numpy())),
                        "p50": float(np.nanmedian(mean_cs_std.to_numpy())),
                        "mean": float(np.nanmean(mean_cs_std.to_numpy())),
                        "max": float(np.nanmax(mean_cs_std.to_numpy())),
                    },
                    "ts_std_summary": {
                        "min": float(np.nanmin(ts_std.to_numpy())),
                        "p50": float(np.nanmedian(ts_std.to_numpy())),
                        "mean": float(np.nanmean(ts_std.to_numpy())),
                        "max": float(np.nanmax(ts_std.to_numpy())),
                    },
                    "kept_by": {
                        "cs_nonconstant": int(cs_ok[keep_mask].sum()),
                        "ts_nonconstant": int(ts_ok[keep_mask].sum()),
                    },
                    "kept": list(map(str, keep)),
                    "dropped": list(map(str, dropped_feats)),
                }

                if len(keep) < int(self.min_features_after_cs_filter):
                    raise ValueError(
                        "Too few features after cross-sectional constant filter: "
                        f"kept={len(keep)} < min={int(self.min_features_after_cs_filter)}."
                    )
                X_tmp = X_tmp[keep]

            # B) Deterministic ordering without pandas index alignment (critical)
            date_arr = _as_datetime_index(full_df_work["Date"]).to_numpy(dtype="datetime64[ns]")

            if "Sector" in full_df_work.columns:
                sector_arr = (
                    full_df_work["Sector"].astype("string").fillna("<<NA>>").to_numpy()
                )
            else:
                sector_arr = np.arange(len(X_tmp), dtype=np.int64).astype(str)

            order = (
                pd.DataFrame({"Date": date_arr, "Sector": sector_arr})
                .sort_values(["Date", "Sector"], kind="mergesort")
                .index.to_numpy(dtype=np.int64)
            )

            X_final = X_tmp.iloc[order].reset_index(drop=True)
            y_gate_ord = np.asarray(y_gate_work, dtype=np.int8)[order]
            full_df_ord = full_df_work.iloc[order].reset_index(drop=True)
            full_df_ord["Date"] = _as_datetime_index(full_df_ord["Date"]).to_numpy(dtype="datetime64[ns]")

            if y_rank_work is not None:
                try:
                    yr = np.asarray(y_rank_work)
                    y_rank_ord = yr[order] if yr.shape[0] == y_gate_ord.shape[0] else None
                except Exception:
                    y_rank_ord = None

            # E) Resolve device config and persist it
            resolved_device_cfg = _get_device_config(bool(self.use_gpu))
            if self.deterministic_mode:
                resolved_device_cfg = {"tree_method": "hist"}

            # Create run directory now that X is final
            self.run_manager.create_run(
                horizon=int(self.horizon),
                seeds=list(self.seeds),
                cost_bps=float(self.cost_threshold) * 10_000.0,
                cv_config=dict(cv_config or {}),
                gate_params=dict(self.model_params),
                ranker_params={},
                X=X_final,
                y_gate=y_gate_ord,
                y_rank=y_rank_ord,
                group_sizes=getattr(self.data, "group_sizes", None),
                feature_selection_method=str(self.feature_selection),
                top_n_features=int(self.top_n_features),
                optuna_n_trials=int(self.optuna_n_trials),
                notes="sector_rotation_trainer: single-classifier pseudo-ranking (auto_* flags are no-ops)",
            )

            # H) Put trainer params into metadata (without changing metadata schema)
            if self.run_manager.metadata is not None:
                self.run_manager.metadata.cv_config = dict(self.run_manager.metadata.cv_config)
                self.run_manager.metadata.cv_config["trainer_params"] = {
                    "top_n_features": int(self.top_n_features),
                    "feature_selection": str(self.feature_selection),
                    "optuna_n_trials": int(self.optuna_n_trials),
                    "deterministic_mode": bool(self.deterministic_mode),
                    "drop_cs_constant_features": bool(self.drop_cs_constant_features),
                    "cs_constant_eps": float(self.cs_constant_eps),
                    "min_features_after_cs_filter": int(self.min_features_after_cs_filter),
                    "auto_flags": {
                        "auto_generate_plots": bool(self.auto_generate_plots),
                        "auto_write_printable": bool(self.auto_write_printable),
                        "auto_run_backtest": bool(self.auto_run_backtest),
                        "auto_save_models": bool(self.auto_save_models),
                    },
                    "flags_effective": False,
                }
            self.run_manager.save_metadata()

            # G) Never-silent diagnostics (always save)
            self.run_manager.save_artifact(input_schema, "input_schema.json", "diagnostic")
            self.run_manager.save_artifact(dict(cv_config or {}), "cv_config.json", "diagnostic")
            self.run_manager.save_artifact(dict(resolved_device_cfg or {}), "resolved_device_config.json", "diagnostic")
            if cs_filter_report is not None:
                self.run_manager.save_artifact(cs_filter_report, "cs_constant_filter_report.json", "diagnostic")

            # Train
            self._log("Training Classifier...")
            proba_oof, fold_results = self._train_classifier(X_final, y_gate_ord, full_df_ord)
            if proba_oof is None or proba_oof.empty:
                raise RuntimeError("OOF is empty: CV produced no test folds")

            # F) Canonical OOF normalization + checks
            proba_oof = proba_oof.copy()
            proba_oof["Date"] = _as_datetime_index(proba_oof["Date"]).to_numpy(dtype="datetime64[ns]")
            if "Sector" in proba_oof.columns:
                proba_oof["Sector"] = proba_oof["Sector"].astype("string")

            required_cols = {"fold_id", "pred_mean", "pred_std", "y_gate", "label_excess", "Date"}
            missing = sorted(required_cols - set(proba_oof.columns))
            if missing:
                raise AssertionError(f"OOF missing required columns: {missing}")

            pred = pd.to_numeric(proba_oof["pred_mean"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(pred).all():
                _save_df_parquet_or_csv(
                    run_manager=self.run_manager,
                    df=proba_oof,
                    name="oof_nan_debug.parquet",
                    subfolder="diagnostic",
                )
                raise ValueError("OOF pred_mean contains NaN/inf")

            # A) Ensemble invariant: if multiple seeds are requested, pred_std must reflect it.
            if bool(self.enforce_ensemble_diversity) and int(len(self.seeds)) > 1:
                std = pd.to_numeric(proba_oof["pred_std"], errors="coerce").to_numpy(dtype=float)
                std = std[np.isfinite(std)]
                max_std = float(np.max(std)) if std.size else 0.0
                if max_std <= 1e-12:
                    self.run_manager.save_artifact(
                        {
                            "seeds": list(self.seeds),
                            "max_pred_std": float(max_std),
                            "note": "Multiple seeds requested but predictions are identical across seeds. If pred_std is used as uncertainty, enable stochasticity (e.g., subsample/colsample_bytree < 1) or disable enforce_ensemble_diversity.",
                        },
                        "ensemble_invariant_failed.json",
                        "diagnostic",
                    )
                    raise RuntimeError(
                        "Multi-seed ensemble produced pred_std≈0 everywhere. "
                        "Either only one seed ran, seed dimension was swallowed, or the model is fully deterministic across seeds."
                    )

            # G) Save OOF + fold_results immediately
            _save_df_parquet_or_csv(
                run_manager=self.run_manager,
                df=proba_oof,
                name="proba_oof.parquet",
                subfolder="diagnostic",
            )
            self.run_manager.save_artifact(
                {
                    "fold_results": [
                        {
                            "fold_id": int(fr.fold_id),
                            "chosen_features": list(fr.chosen_features),
                            "chosen_params": dict(fr.chosen_params),
                            "thresholds": dict(fr.thresholds),
                        }
                        for fr in fold_results
                    ]
                },
                "fold_results.json",
                "diagnostic",
            )

            # I) Negative controls
            fold_min = proba_oof.groupby("fold_id", sort=True)["Date"].min()
            if not bool(fold_min.is_monotonic_increasing):
                self.run_manager.save_artifact(
                    {"fold_min_dates": {str(k): str(pd.Timestamp(v).isoformat()) for k, v in fold_min.items()}},
                    "fold_date_monotonicity.json",
                    "diagnostic",
                )
                raise RuntimeError("Fold min(Date) is not monotonic increasing; possible leakage/alignment issue")

            # D) CV splitting invariant (Date purity): each Date must belong to exactly one fold.
            if bool(self.enforce_date_purity) and "Date" in proba_oof.columns and "fold_id" in proba_oof.columns:
                date_fold_counts = proba_oof.groupby("Date", sort=True)["fold_id"].nunique()
                max_folds_per_date = int(date_fold_counts.max()) if len(date_fold_counts) else 1
                if max_folds_per_date > 1:
                    offenders = date_fold_counts[date_fold_counts > 1]
                    self.run_manager.save_artifact(
                        {
                            "max_folds_per_date": int(max_folds_per_date),
                            "n_offender_dates": int(len(offenders)),
                            "offender_dates": [str(pd.Timestamp(d).isoformat()) for d in offenders.index[:200]],
                        },
                        "cv_date_purity_violation.json",
                        "diagnostic",
                    )
                    raise RuntimeError(
                        f"Leakage/CV bug: same Date appears in multiple folds (max_folds_per_date={max_folds_per_date})."
                    )

            # I) Negative control: strict parity between OOF labels and source labels by key (Date,Sector)
            # This protects against index misalignment/leakage even if label definitions evolve.
            if full_df_ord is not None and "Sector" in full_df_ord.columns and "Sector" in proba_oof.columns:
                truth_df = full_df_ord[["Date", "Sector"]].copy()
                truth_df["label_excess_true"] = pd.to_numeric(full_df_ord["label_excess"], errors="coerce").to_numpy(dtype=float)
                truth_df["y_gate_true"] = np.asarray(y_gate_ord, dtype=np.int8)
                truth_df["Sector"] = truth_df["Sector"].astype("string")

                merged = proba_oof.merge(
                    truth_df,
                    on=["Date", "Sector"],
                    how="left",
                    validate="many_to_one",
                )
                if int(merged["y_gate_true"].isna().sum()) > 0:
                    _save_df_parquet_or_csv(
                        run_manager=self.run_manager,
                        df=merged,
                        name="oof_key_merge_debug.parquet",
                        subfolder="diagnostic",
                    )
                    raise RuntimeError("OOF (Date,Sector) keys did not match source dataset; alignment bug")

                y_oof = pd.to_numeric(merged["y_gate"], errors="coerce").to_numpy(dtype=float)
                y_src = pd.to_numeric(merged["y_gate_true"], errors="coerce").to_numpy(dtype=float)
                oky = np.isfinite(y_oof) & np.isfinite(y_src)
                if oky.any() and not bool(np.all((y_oof[oky] >= 0.5).astype(np.int8) == (y_src[oky] >= 0.5).astype(np.int8))):
                    _save_df_parquet_or_csv(
                        run_manager=self.run_manager,
                        df=merged,
                        name="oof_y_gate_parity_mismatch.parquet",
                        subfolder="diagnostic",
                    )
                    raise RuntimeError("OOF y_gate does not match source y_gate by (Date,Sector); alignment bug")

                le_oof = pd.to_numeric(merged["label_excess"], errors="coerce").to_numpy(dtype=float)
                le_src = pd.to_numeric(merged["label_excess_true"], errors="coerce").to_numpy(dtype=float)
                okle = np.isfinite(le_oof) & np.isfinite(le_src)
                if okle.any() and not bool(np.all(np.isclose(le_oof[okle], le_src[okle], rtol=1e-9, atol=1e-12))):
                    _save_df_parquet_or_csv(
                        run_manager=self.run_manager,
                        df=merged,
                        name="oof_label_excess_parity_mismatch.parquet",
                        subfolder="diagnostic",
                    )
                    raise RuntimeError("OOF label_excess does not match source label_excess by (Date,Sector); alignment bug")
            else:
                warnings.warn(
                    "Skipping OOF/source parity check: Sector column missing.",
                    category=UserWarning,
                    stacklevel=2,
                )

            # I) Negative control: y_gate vs (label_excess > cost_threshold)
            # Some datasets may define y_gate using a different transform than label_excess.
            # Treat as diagnostic-only (warn + persist), not a hard failure.
            le_np = pd.to_numeric(proba_oof["label_excess"], errors="coerce").to_numpy(dtype=float)
            yg_np = pd.to_numeric(proba_oof["y_gate"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(le_np) & np.isfinite(yg_np)
            implied = (le_np > float(self.cost_threshold)).astype(np.int8)
            observed = (yg_np >= 0.5).astype(np.int8)
            match = float(np.mean((observed[ok] == implied[ok]).astype(float))) if ok.any() else 1.0
            if match < 0.999:
                mismatch_df = proba_oof.loc[ok].copy()
                mismatch_df["y_gate_implied"] = implied[ok]
                mismatch_df["y_gate_observed"] = observed[ok]
                _save_df_parquet_or_csv(
                    run_manager=self.run_manager,
                    df=mismatch_df,
                    name="gate_label_mismatch.parquet",
                    subfolder="diagnostic",
                )
                self.run_manager.save_artifact(
                    {
                        "match_rate": float(match),
                        "pos_rate_observed": float(np.mean(observed[ok])) if ok.any() else None,
                        "pos_rate_implied": float(np.mean(implied[ok])) if ok.any() else None,
                        "cost_threshold": float(self.cost_threshold),
                        "note": "Diagnostic-only: dataset y_gate is not exactly (label_excess > cost_threshold).",
                    },
                    "gate_label_contract_mismatch.json",
                    "diagnostic",
                )
                warnings.warn(
                    "Dataset label contract mismatch: y_gate != (label_excess > cost_threshold); continuing. "
                    f"match_rate={match:.6f}. See diagnostic/gate_label_contract_mismatch.json",
                    category=UserWarning,
                    stacklevel=2,
                )

            # Gate metrics + polarity diagnostic (no auto-flip)
            p_gate_np = proba_oof["pred_mean"].to_numpy(dtype=float)
            y_gate_bin = proba_oof["y_gate"].to_numpy(dtype=float)
            gate_metrics = gate_classification_metrics(y_gate_bin, p_gate_np, threshold=0.5)
            try:
                gate_metrics_flip = gate_classification_metrics(y_gate_bin, 1.0 - p_gate_np, threshold=0.5)
            except Exception:
                gate_metrics_flip = None

            polarity_flipped = any(
                float(fr.thresholds.get("polarity_flipped", 0.0)) > 0.5 for fr in (fold_results or [])
            )
            eps_auc = 1e-3
            if gate_metrics_flip and (
                float(gate_metrics_flip.get("auc", 0.0)) > float(gate_metrics.get("auc", 0.0)) + eps_auc
            ):
                diag: dict[str, Any] = {
                    "auc": float(gate_metrics.get("auc", float("nan"))),
                    "auc_flip": float(gate_metrics_flip.get("auc", float("nan"))),
                    "mean_y_gate": float(np.mean(y_gate_bin)) if len(y_gate_bin) else float("nan"),
                    "mean_pred": float(np.mean(p_gate_np)) if len(p_gate_np) else float("nan"),
                }
                try:
                    from sklearn.metrics import roc_auc_score

                    rows = []
                    for fold_id, g in proba_oof.groupby("fold_id", sort=True):
                        yb = pd.to_numeric(g["y_gate"], errors="coerce").to_numpy(dtype=float)
                        pb = pd.to_numeric(g["pred_mean"], errors="coerce").to_numpy(dtype=float)
                        m = np.isfinite(yb) & np.isfinite(pb)
                        if not m.any():
                            continue
                        y_int = (yb[m] >= 0.5).astype(int)
                        if np.unique(y_int).size < 2:
                            auc_f = float("nan")
                            auc_f_flip = float("nan")
                        else:
                            auc_f = float(roc_auc_score(y_int, pb[m]))
                            auc_f_flip = float(roc_auc_score(y_int, 1.0 - pb[m]))
                        rows.append({"fold_id": int(fold_id), "n": int(m.sum()), "auc": auc_f, "auc_flip": auc_f_flip})
                    if rows:
                        diag["per_fold"] = rows
                        finite = [r for r in rows if np.isfinite(r["auc"]) and np.isfinite(r["auc_flip"])]
                        if finite:
                            improved = [r for r in finite if float(r["auc_flip"]) > float(r["auc"]) + eps_auc]
                            diag["per_fold_improve_frac"] = float(len(improved)) / float(len(finite))
                except Exception as ee:
                    diag["per_fold_error"] = str(ee)

                self.run_manager.save_artifact(diag, "gate_polarity_diagnostic.json", "diagnostic")
                improve_frac = float(diag.get("per_fold_improve_frac", float("nan")))
                if np.isfinite(improve_frac) and improve_frac >= 0.8:
                    raise RuntimeError(
                        "Gate polarity appears consistently inverted across folds (>=80% folds improved by flip). "
                        f"auc={float(gate_metrics.get('auc', 0.0)):.6f} auc_flip={float(gate_metrics_flip.get('auc', 0.0)):.6f}."
                    )
                warnings.warn(
                    "Gate global AUC improves if flipped, but fold-level polarity is mixed; continuing without any flip. "
                    f"auc={float(gate_metrics.get('auc', 0.0)):.6f} auc_flip={float(gate_metrics_flip.get('auc', 0.0)):.6f}",
                    category=UserWarning,
                    stacklevel=2,
                )

            # Metrics
            self._log("Computing metrics...")
            metrics_label_excess = _rank_metrics_high_is_better(
                y_true=proba_oof["label_excess"],
                y_score=proba_oof["pred_mean"],
                groups=proba_oof["Date"],
                cost_threshold=float(self.cost_threshold),
            )
            metrics_y_gate = _rank_metrics_high_is_better(
                y_true=proba_oof["y_gate"],
                y_score=proba_oof["pred_mean"],
                groups=proba_oof["Date"],
                cost_threshold=0.5,
            )
            self.run_manager.save_artifact(
                {
                    "rank_eval_target": str(self.rank_eval_target),
                    "label_excess": metrics_label_excess.__dict__,
                    "y_gate": metrics_y_gate.__dict__,
                    "note": "If training target is y_gate, prefer y_gate metrics unless y_gate strictly equals (label_excess > cost_threshold).",
                },
                "rank_metrics_targets.json",
                "diagnostic",
            )
            metrics = metrics_label_excess if self.rank_eval_target == "label_excess" else metrics_y_gate

            metrics_by_regime = None
            if self.data is not None and "regime" in self.data.full_df.columns and "regime" in proba_oof.columns:
                metrics_by_regime = _rank_metrics_by_regime_high_is_better(
                    proba_oof,
                    regime_col="regime",
                    cost_threshold=float(self.cost_threshold),
                )

            feature_importance = pd.Series(0.0, index=X_final.columns, dtype=float)
            if fold_results:
                for fr in fold_results:
                    try:
                        s = pd.Series(fr.feature_importance, dtype=float)
                        feature_importance = feature_importance.add(s, fill_value=0.0)
                    except Exception:
                        continue
                feature_importance = feature_importance.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                total = float(feature_importance.sum())
                if total > 0.0:
                    feature_importance = feature_importance / total

            drift_report = self._compute_drift_report(X_final, full_df_ord["Date"])

            end_time = datetime.now(timezone.utc)
            training_time = (end_time - start_time).total_seconds()
            self._log(f"Completed in {training_time:.1f} seconds")

            result = SectorRotationResult(
                horizon=self.horizon,
                seeds=self.seeds,
                n_dates=int(pd.Index(proba_oof["Date"]).nunique()),
                n_observations=int(len(proba_oof)),
                proba_oof=proba_oof,
                metrics=metrics,
                metrics_by_regime=metrics_by_regime,
                gate_metrics=gate_metrics,
                gate_polarity_flipped=polarity_flipped,
                feature_importance=feature_importance,
                drift_report=drift_report,
                fold_results=[
                    {
                        "fold_id": int(fr.fold_id),
                        "chosen_features": list(fr.chosen_features),
                        "chosen_params": dict(fr.chosen_params),
                        "thresholds": dict(fr.thresholds),
                    }
                    for fr in fold_results
                ],
                training_time_seconds=float(training_time),
            )

            self._save_results(result)
            return result

        except Exception as e:
            # G) Save diagnostics even on fail-fast (ALWAYS). Do NOT create_run() here.
            try:
                if self.run_manager and self.run_manager.run_dir is not None:
                    if self.data is not None and X_work is not None:
                        self.run_manager.save_artifact(
                            _input_schema_snapshot(sr=self.data, X=X_work),
                            "input_schema.json",
                            "diagnostic",
                        )
                    if cv_config is not None:
                        self.run_manager.save_artifact(dict(cv_config), "cv_config.json", "diagnostic")
                    if resolved_device_cfg is not None:
                        self.run_manager.save_artifact(dict(resolved_device_cfg), "resolved_device_config.json", "diagnostic")
                    if cs_filter_report is not None:
                        self.run_manager.save_artifact(cs_filter_report, "cs_constant_filter_report.json", "diagnostic")
                    if isinstance(proba_oof, pd.DataFrame) and not proba_oof.empty:
                        _save_df_parquet_or_csv(
                            run_manager=self.run_manager,
                            df=proba_oof,
                            name="proba_oof.parquet",
                            subfolder="diagnostic",
                        )
                    if fold_results:
                        self.run_manager.save_artifact(
                            {
                                "fold_results": [
                                    {
                                        "fold_id": int(fr.fold_id),
                                        "chosen_features": list(fr.chosen_features),
                                        "chosen_params": dict(fr.chosen_params),
                                        "thresholds": dict(fr.thresholds),
                                    }
                                    for fr in fold_results
                                ]
                            },
                            "fold_results.json",
                            "diagnostic",
                        )
                    self.run_manager.save_artifact(
                        {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        },
                        "exception.json",
                        "diagnostic",
                    )
                    try:
                        self.run_manager.save_metadata()
                    except Exception:
                        pass
                else:
                    fail_dir = Path(self.base_results_dir) / (
                        "FAILED_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    )
                    diag_dir = fail_dir / "diagnostic"
                    diag_dir.mkdir(parents=True, exist_ok=True)
                    (diag_dir / "exception.txt").write_text(
                        traceback.format_exc(),
                        encoding="utf-8",
                    )
            finally:
                raise

    def _train_classifier(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        full_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[FoldResult]]:
        """
        Train classifier via nested CV.
        
        Returns:
            DataFrame with OOF predictions
        """
        # Build CV config
        cv_config = get_cv_config(self.horizon)
        outer_cv = PurgedWalkForwardCV(
            n_splits=cv_config.get("n_splits"),
            test_size=cv_config["test_size"],
            purge_gap=cv_config["purge_gap"],
            embargo=cv_config["embargo"],
            min_train_size=cv_config["min_train_size"],
            test_start=cv_config.get("test_start"),
        )
        
        # IMPORTANT: X/y/full_df are expected to be already ordered deterministically
        # (Date, Sector) by run(). Do not reorder here.
        if len(X) != len(full_df):
            raise ValueError(f"X/full_df length mismatch: len(X)={len(X)} len(full_df)={len(full_df)}")
        if int(np.asarray(y).shape[0]) != int(len(X)):
            raise ValueError(f"X/y length mismatch: len(X)={len(X)} len(y)={int(np.asarray(y).shape[0])}")

        date_idx = _as_datetime_index(full_df["Date"]).to_numpy(dtype="datetime64[ns]")
        if not np.issubdtype(date_idx.dtype, np.datetime64):
            raise TypeError(f"Expected datetime64 groups; got dtype={date_idx.dtype}")
        groups = date_idx
        
        # Prepare y DataFrame (run_outer_folds expects DataFrame)
        y_payload: dict[str, Any] = {
            "y_gate": np.asarray(y).astype(np.int8, copy=False),
            "label_excess": pd.to_numeric(full_df["label_excess"], errors="raise").to_numpy(dtype=float),
            "Date": date_idx,
        }
        if "Sector" in full_df.columns:
            y_payload["Sector"] = full_df["Sector"].astype("string").to_numpy()
        if "regime" in full_df.columns:
            y_payload["regime"] = full_df["regime"].to_numpy()

        y_df = pd.DataFrame(y_payload, index=X.index)
        
        # Inner CV factory
        def inner_cv_factory_fn(config_inner: Any, groups_inner: np.ndarray) -> Any:
            # D) Do not ignore config_inner. Runners may adapt inner CV params by fold.
            try:
                if hasattr(config_inner, "get"):
                    n_splits = config_inner.get("n_splits", 2)
                    test_size = config_inner.get("test_size", cv_config["test_size"])
                    purge_gap = config_inner.get("purge_gap", cv_config["purge_gap"])
                    embargo = config_inner.get("embargo", cv_config["embargo"])
                    min_train_size = config_inner.get("min_train_size", cv_config["min_train_size"])
                else:
                    n_splits = getattr(config_inner, "n_splits", 2)
                    test_size = getattr(config_inner, "test_size", cv_config["test_size"])
                    purge_gap = getattr(config_inner, "purge_gap", cv_config["purge_gap"])
                    embargo = getattr(config_inner, "embargo", cv_config["embargo"])
                    min_train_size = getattr(config_inner, "min_train_size", cv_config["min_train_size"])

                n_splits = int(n_splits) if n_splits is not None else 2
                return PurgedWalkForwardCV(
                    n_splits=n_splits,
                    test_size=int(test_size),
                    purge_gap=int(purge_gap),
                    embargo=int(embargo),
                    min_train_size=int(min_train_size),
                )
            except Exception as e:
                # Fallback for small folds (do not crash)
                ctx = {
                    "error": str(e),
                    "config_inner": repr(config_inner),
                    "n_samples": int(np.asarray(groups_inner).shape[0]) if groups_inner is not None else None,
                    "n_unique_groups": int(np.unique(np.asarray(groups_inner)).shape[0]) if groups_inner is not None else None,
                }
                self._inner_cv_fallback_events.append(ctx)
                if self.run_manager:
                    try:
                        self.run_manager.save_artifact(self._inner_cv_fallback_events, "inner_cv_fallback.json", "diagnostic")
                    except Exception:
                        pass
                self._log(f"WARNING: inner CV fallback engaged: {e}")
                ts = int(max(1, min(int(test_size) if 'test_size' in locals() else 20, 20)))
                return PurgedWalkForwardCV(
                    n_splits=1,
                    test_size=max(1, ts),
                    purge_gap=int(purge_gap) if 'purge_gap' in locals() else int(cv_config["purge_gap"]),
                    embargo=int(embargo) if 'embargo' in locals() else int(cv_config["embargo"]),
                    min_train_size=0,
                )
        
        # Model factory
        def model_factory_fn(seed: int, params: dict[str, Any]) -> Any:
            import xgboost as xgb
            xgb_base = _get_xgb_base_params(
                use_gpu=self.use_gpu,
                deterministic=self.deterministic_mode,
            )
            full_params = {
                **xgb_base,
                **self.model_params,
                **params,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "missing": np.nan,
                "verbosity": 0,
                "random_state": seed,
            }

            # E) Determinism: always CPU hist + n_jobs=1
            if self.deterministic_mode:
                full_params["n_jobs"] = 1
                full_params["tree_method"] = "hist"
                full_params.pop("device", None)
            return xgb.XGBClassifier(**full_params)
        
        # Build NestedCVConfig
        inner_params = {**cv_config, "n_splits": 2}
        nested_config = NestedCVConfig(
            horizon=self.horizon,
            cost_bps=float(self.cost_threshold) * 10000.0,
            outer_params=cv_config,
            inner_params=inner_params,
            seeds=self.seeds,
            task="gate_classifier",
            feature_selection=self.feature_selection,
            top_n_features=self.top_n_features,
            optuna_n_trials=self.optuna_n_trials,
            enable_time_decay=True,
            half_life_days=252,
            gate_class_balance=True,
        )
        
        # Run nested CV
        fold_results: list[FoldResult] = []
        oof = run_outer_folds(
            config=nested_config,
            X=X,
            y=y_df,
            groups=groups,
            outer_cv=outer_cv,
            inner_cv_factory=inner_cv_factory_fn,
            model_factory=model_factory_fn,
            fit_fn=fit_classifier,
            predict_fn=predict_classifier,
            fold_result_sink=fold_results.append,
        )

        # Hardening: run_outer_folds returns a new, sorted DataFrame with reset index.
        # Do NOT attempt index-based joins back to y_df; validate by columns/keys instead.
        if not oof.empty:
            required_cols = {"fold_id", "pred_mean", "pred_std", "y_gate", "label_excess", "Date"}
            missing = sorted(required_cols - set(oof.columns))
            if missing:
                raise AssertionError(f"OOF missing required columns: {missing}")

            if oof[["y_gate", "label_excess", "Date"]].isna().any().any():
                raise AssertionError("OOF contains NaNs in required label metadata")

            # Groups/date consistency: runner typically carries both; validate if present.
            if "group" in oof.columns:
                if oof["group"].isna().any():
                    raise AssertionError("OOF contains NaNs in 'group' column")
                d1 = _as_datetime_index(oof["Date"])
                d2 = _as_datetime_index(oof["group"])
                if not np.array_equal(d1.to_numpy(), d2.to_numpy()):
                    raise AssertionError("OOF Date != group; potential groups/y_df Date mismatch")

        return oof, fold_results

    def _compute_drift_report(self, X: pd.DataFrame, dates: pd.Series) -> dict[str, Any]:
        """Compute drift statistics."""
        try:
            # Hardening: align dates "by position" with X (not by label semantics)
            dti = _as_datetime_index(dates)
            if len(dti) != len(X):
                raise ValueError(f"drift: dates length mismatch: len(dates)={len(dti)} len(X)={len(X)}")
            d = pd.Series(dti.to_numpy(), index=X.index)
            order = np.argsort(d.to_numpy())
            Xs = X.iloc[order].reset_index(drop=True)
            ds = d.iloc[order].reset_index(drop=True)

            unique_dates = pd.Index(ds.unique()).sort_values()
            if len(unique_dates) < 2:
                return None
            mid_date = unique_dates[len(unique_dates) // 2]

            ref = Xs.loc[ds <= mid_date]
            cur = Xs.loc[ds > mid_date]
            if ref.empty or cur.empty:
                return None

            # Backwards/forwards compatibility: some DriftDetector versions may
            # accept `feature_cols`, others infer it from reference_df.
            try:
                detector = DriftDetector(reference_df=ref, feature_cols=list(X.columns))
            except TypeError:
                detector = DriftDetector(reference_df=ref)
            current_df = cur
            report = detector.compute_drift(current_df)
            
            return {
                "snapshot": {
                    "n_features": report.n_features,
                    "n_features_drifted": report.n_features_drifted,
                    "mean_psi": report.mean_psi,
                    "max_psi": report.max_psi,
                    "features_with_drift": report.features_with_drift[:50],
                    "psi_by_feature": {k: v for k, v in list(report.psi_by_feature.items())[:50]},
                },
            }
        except Exception as e:
            self._log(f"WARNING: Drift computation failed: {e}")
            return None

    def _save_results(self, result: SectorRotationResult) -> None:
        """Save all artifacts."""
        if not self.run_manager:
            return
        
        # Save metrics
        self.run_manager.save_artifact(
            result.to_dict(),
            "metrics.json",
            "diagnostic",
        )
        
        # Save drift
        if result.drift_report:
            self.run_manager.save_artifact(
                result.drift_report,
                "drift_report.json",
                "diagnostic",
            )
        
        self._log(f"Results saved to {self.run_manager.run_dir}")


# =============================================================================
# Convenience Function
# =============================================================================

def train_sector_rotation(
    horizon: int = 21,
    *,
    optuna_n_trials: int = 0,
    seeds: list[int] | None = None,
    use_gpu: bool = True,
    verbose: bool = True,
) -> SectorRotationResult:
    """
    Convenience function for training.
    
    Args:
        horizon: Prediction horizon
        optuna_n_trials: Number of Optuna trials (0 = no tuning)
        seeds: Random seeds for ensemble
        use_gpu: Use GPU acceleration
        verbose: Print progress
    
    Returns:
        SectorRotationResult
    """
    trainer = SectorRotationTrainer(
        horizon=horizon,
        optuna_n_trials=optuna_n_trials,
        seeds=seeds or [42],
        use_gpu=use_gpu,
        verbose=verbose,
    )
    
    return trainer.run()