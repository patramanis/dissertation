"""
Dual Model Training System for Sector Rotation.

Implements a two-stage prediction pipeline:
1. Gate Classifier (XGBClassifier): Filters "actionable" sectors
2. Ranker (XGBRanker): Ranks surviving sectors for allocation

Key design principles:
- Filter-then-Sort hierarchy (Gate filters, Ranker sorts survivors)
- No rank score as gate feature (prevents leakage)
- Nested CV for all tuning (purged walk-forward)
- Multi-seed ensemble for uncertainty quantification
- Fold-safe thresholds (computed from train only)

Usage:
    from ModularMonolith.src.models.train_dual_system import DualModelTrainer
    
    trainer = DualModelTrainer(horizon=21)
    results = trainer.run()
    results.save("Results/")
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

# Local imports
from ModularMonolith.data.dataset_shaping import (
    EXPECTED_SECTORS,
    HORIZONS,
    ShapingResult,
    load_dual_model_dataset,
    load_dual_model_dataset_cached,
)
from ModularMonolith.src.config.cv_config import get_cv_config
from ModularMonolith.src.cv.purged_walk_forward_cv import PurgedWalkForwardCV
from ModularMonolith.src.cv.run_nested_cv import (
    FoldResult,
    NestedCVConfig,
    run_outer_folds,
)
from ModularMonolith.src.reporting.drift import (
    DriftDetector,
    compute_drift_over_time,
    compute_psi_by_feature_over_time,
    format_drift_summary,
)
from ModularMonolith.src.reporting.metrics_rank import (
    RankMetrics,
    compute_ic_over_time,
    compute_precision_over_time,
    compute_rank_metrics,
    compute_rank_metrics_by_regime,
    format_metrics_summary,
    gate_classification_metrics,
)
from ModularMonolith.src.results.run_manager import RunManager


# Suppress XGBoost verbosity
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


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
    
    # CPU fallback
    return {"tree_method": "hist", "device": "cpu"}


@lru_cache(maxsize=2)
def _get_device_config(use_gpu: bool) -> dict[str, Any]:
    """Cached GPU/CPU device configuration."""
    if not use_gpu:
        return {"tree_method": "hist", "device": "cpu"}
    return _detect_gpu()

def _get_xgb_base_params(*, use_gpu: bool = True, deterministic: bool = False) -> dict[str, Any]:
    """Get base XGBoost parameters with GPU/CPU detection.
    
    FIX #6: deterministic=True forces n_jobs=1 for bitwise reproducibility.
    """
    device_config = _get_device_config(use_gpu)
    base = {
        **device_config,
        "verbosity": 0,
        "n_jobs": 1 if deterministic else -1,
    }
    return base


# =============================================================================
# Model Factories
# =============================================================================

def gate_classifier_factory(seed: int, params: dict[str, Any], *, use_gpu: bool = True, deterministic: bool = False) -> Any:
    """Create XGBClassifier for gate model."""
    import xgboost as xgb
    
    # Merge with GPU config
    full_params = {
        **_get_xgb_base_params(use_gpu=use_gpu, deterministic=deterministic),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": seed,
        **params,
    }
    
    return xgb.XGBClassifier(**full_params)


def ranker_factory(seed: int, params: dict[str, Any], *, use_gpu: bool = True, deterministic: bool = False) -> Any:
    """Create XGBRanker for ranking model."""
    import xgboost as xgb
    
    full_params = {
        **_get_xgb_base_params(use_gpu=use_gpu, deterministic=deterministic),
        "objective": "rank:pairwise",
        "random_state": seed,
        **params,
    }
    
    return xgb.XGBRanker(**full_params)


# =============================================================================
# Fit and Predict Functions
# =============================================================================

def fit_classifier(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    groups: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """
    Fit XGBClassifier with optional sample weights.
    
    SOTA: sample_weight combines time decay and class balance weights
    when enabled. This allows the model to focus on recent market regimes
    while still handling class imbalance.
    """
    # FIX #9: copy=False for performance (no mutation downstream)
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    # FIX #7: Binary classifier expects int labels, not float
    y_np = np.asarray(y).ravel().astype(np.int32)
    
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight).ravel().astype(np.float32)
    
    model.fit(X_np, y_np, **fit_kwargs)
    return model


def fit_ranker(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    group_sizes: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """
    Fit XGBRanker with group sizes and optional per-GROUP sample weights.
    
    CRITICAL FIX: XGBoost ranker with 'group' parameter expects:
    - group: array of group sizes (len = n_groups, sum = n_rows)
    - sample_weight: array of GROUP weights (len = n_groups), NOT per-row
    
    If you pass per-row weights (len = n_rows), you'll get:
    "Size of weight must equal to the number of query groups"
    """
    if group_sizes is None:
        raise ValueError("group_sizes required for ranker")
    
    # FIX #9: copy=False for performance
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    y_np = np.asarray(y).ravel().astype(np.float32)
    
    fit_kwargs: dict[str, Any] = {"group": np.asarray(group_sizes, dtype=np.int32)}
    
    # CRITICAL: sample_weight must be per-GROUP (len = n_groups)
    if sample_weight is not None:
        sw = np.asarray(sample_weight).ravel().astype(np.float32)
        n_groups = len(group_sizes)
        
        # If passed per-row weights, aggregate to per-group (mean per group)
        if len(sw) == len(X_np):
            # Convert per-row to per-group by averaging within each group
            group_weights = np.zeros(n_groups, dtype=np.float32)
            idx = 0
            for i, size in enumerate(group_sizes):
                group_weights[i] = sw[idx:idx+size].mean()
                idx += size
            sw = group_weights
        
        # Validate: sample_weight must match n_groups
        if len(sw) != n_groups:
            raise ValueError(
                f"Ranker sample_weight must be per-GROUP: "
                f"len(sw)={len(sw)} != n_groups={n_groups}. "
                f"For time-decay, aggregate row weights to group-level means."
            )
        
        fit_kwargs["sample_weight"] = sw
    
    model.fit(X_np, y_np, **fit_kwargs)
    return model


def predict_classifier(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Get probability predictions from classifier."""
    # FIX #9: copy=False for performance
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    return model.predict_proba(X_np)[:, 1]


def predict_ranker(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Get score predictions from ranker.

    Ranking convention (system-wide): LOWER score = better.

    XGBoost rankers naturally output HIGHER score = better (more relevant).
    We negate the raw prediction so that downstream selection + metrics
    (which use nsmallest) are consistent.
    """
    # FIX #9: copy=False for performance
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    return -model.predict(X_np)


# =============================================================================
# Inner CV Factory
# =============================================================================

def inner_cv_factory(config: NestedCVConfig, groups: np.ndarray) -> Any:
    """
    Create inner CV splitter for nested cross-validation.
    
    The inner CV operates on subsets of the outer fold's training data,
    so it needs fewer splits than the outer CV to ensure feasibility.
    We use a conservative n_splits=2 to ensure the inner CV always has
    enough data, even in early outer folds with small training sets.
    
    Args:
        config: Nested CV configuration
        groups: Group labels for the outer fold's training set
    
    Returns:
        PurgedWalkForwardCV splitter for inner CV
    """
    cv_params = get_cv_config(config.horizon)

    # Use conservative n_splits=2 for inner CV to ensure feasibility
    # Even with expanding window, early outer folds have limited data
    # Example: min_train=252, test_size=84, purge=21, n_splits=2
    # Requires: 2*84 + 21 + 252 = 441 dates (feasible for most outer folds)
    inner_n_splits = 2

    return PurgedWalkForwardCV(
        n_splits=inner_n_splits,
        test_size=cv_params["test_size"],
        purge_gap=cv_params["purge_gap"],
        embargo=cv_params["embargo"],
        min_train_size=cv_params["min_train_size"],
    )


# =============================================================================
# Default Parameters
# =============================================================================

DEFAULT_GATE_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
}

DEFAULT_RANKER_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# Optuna search spaces (conservative for production safety)
GATE_SEARCH_SPACE: dict[str, Any] = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": (0.01, 0.1, "log"),
    "min_child_weight": [5, 10, 20],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}

RANKER_SEARCH_SPACE: dict[str, Any] = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": (0.01, 0.1, "log"),
    "min_child_weight": [3, 5, 10],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class DualModelResult:
    """Container for dual model training results."""
    horizon: int
    run_dir: Path
    
    # OOF DataFrames
    gate_oof: pd.DataFrame
    ranker_oof: pd.DataFrame
    merged_oof: pd.DataFrame
    
    # Metrics
    gate_metrics: dict[str, float]
    rank_metrics: RankMetrics
    regime_metrics: dict[Any, RankMetrics]
    
    # Fold-level info
    gate_fold_results: list[FoldResult]
    ranker_fold_results: list[FoldResult]
    
    # Feature stability
    feature_stability: pd.DataFrame
    
    # Drift report
    drift_report: dict[str, Any]
    
    # Thresholds
    thresholds: dict[str, float]

    # Optional diagnostics
    rank_metrics_gated: RankMetrics | None = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def save(self, run_manager: RunManager) -> None:
        """Save all results using RunManager."""
        # OOF predictions
        # Prefer CSV for maximum portability (avoid Parquet/Arrow dtype issues in Results/Run #K)
        run_manager.save_artifact(self.gate_oof, "gate_oof.csv", "diagnostic")
        run_manager.save_artifact(self.ranker_oof, "ranker_oof.csv", "diagnostic")
        run_manager.save_artifact(self.merged_oof, "merged_oof.csv", "diagnostic")
        
        # Metrics (save to diagnostic folder - root folder not allowed by RunManager)
        run_manager.save_artifact(
            {
                "gate_metrics": self.gate_metrics,
                "rank_metrics": self.rank_metrics.to_dict(),
                "rank_metrics_gated": (self.rank_metrics_gated.to_dict() if self.rank_metrics_gated is not None else None),
                "thresholds": self.thresholds,
            },
            "metrics.json",
            "diagnostic",
        )
        
        # Feature stability
        run_manager.save_artifact(self.feature_stability, "feature_stability.csv", "diagnostic")
        
        # Fold results
        fold_data = {
            "gate_folds": [
                {
                    "fold_id": f.fold_id,
                    "chosen_features": f.chosen_features,
                    "chosen_params": f.chosen_params,
                    "thresholds": f.thresholds,
                }
                for f in self.gate_fold_results
            ],
            "ranker_folds": [
                {
                    "fold_id": f.fold_id,
                    "chosen_features": f.chosen_features,
                    "chosen_params": f.chosen_params,
                    "thresholds": f.thresholds,
                }
                for f in self.ranker_fold_results
            ],
        }
        run_manager.save_artifact(fold_data, "fold_results.json", "diagnostic")
        
        # Drift
        run_manager.save_artifact(self.drift_report, "drift_report.json", "diagnostic")
        
        # Save metadata
        run_manager.save_metadata()
        
        print(f"[DualModelResult] Saved to {run_manager.run_dir}")


# =============================================================================
# Main Trainer Class
# =============================================================================

class DualModelTrainer:
    """
    Dual Model Training System.
    
    Implements filter-then-sort strategy:
    1. Gate (classifier) → filters actionable winners
    2. Ranker → sorts survivors for allocation
    
    All tuning happens via nested CV with purging/embargo.
    """
    
    def __init__(
        self,
        horizon: int,
        *,
        seeds: list[int] | None = None,
        gate_params: dict[str, Any] | None = None,
        ranker_params: dict[str, Any] | None = None,
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
        enable_rolling_zscore: bool = True,  # FIX: Enabled by default to prevent drift
        rolling_zscore_window: int = 252,
        rolling_zscore_min_periods: int = 60,
        gate_label_mode: Literal["excess_gt_cost", "topk", "top_pct"] = "excess_gt_cost",
        gate_topk: int = 3,
        gate_top_pct: float = 0.30,
        gate_cost_threshold: float | None = None,
        backtest_reject_all_if_gate_pass_rate_below: float | None = 0.20,
        ranker_drop_date_constant_features: bool = True,
        ranker_constant_std_eps: float = 1e-12,
        ranker_constant_frac_threshold: float = 0.99,
        deterministic_mode: bool = False,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            horizon: Prediction horizon (5, 21, or 63 days)
            seeds: Random seeds for ensemble (default: 10 seeds)
            gate_params: XGBClassifier parameters
            ranker_params: XGBRanker parameters
            feature_selection: "importance" or "corr" based selection
            top_n_features: Number of features to select
            optuna_n_trials: Number of Optuna trials (0 to disable)
            use_gpu: Whether to use GPU acceleration
            verbose: Print progress
            base_results_dir: Base directory for results
            deterministic_mode: FIX #6: Force CPU, n_jobs=1 for bitwise reproducibility
        """
        if horizon not in HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}. Expected {HORIZONS}")
        
        self.horizon = horizon
        self.seeds = seeds or list(range(10))
        self.gate_params = {**DEFAULT_GATE_PARAMS, **(gate_params or {})}
        self.ranker_params = {**DEFAULT_RANKER_PARAMS, **(ranker_params or {})}
        self.feature_selection = feature_selection

        # Set early: used by deterministic-mode branch below.
        self.verbose = verbose

        # Quant research simplification heuristic: default feature budget by horizon
        # (keep this as a default only; user can override explicitly)
        default_top_n_by_horizon = {5: 10, 21: 25, 63: 25}
        if top_n_features is None:
            top_n_features = int(default_top_n_by_horizon.get(int(horizon), 25))
        self.top_n_features = int(top_n_features)
        if self.top_n_features <= 0:
            raise ValueError("top_n_features must be positive")
        self.optuna_n_trials = optuna_n_trials
        # FIX #6: Deterministic mode forces CPU + n_jobs=1
        self.deterministic_mode = bool(deterministic_mode)
        if self.deterministic_mode:
            self.use_gpu = False
            if self.verbose:
                self._log("Deterministic mode: forcing CPU, n_jobs=1")
        else:
            self.use_gpu = use_gpu
        self.base_results_dir = base_results_dir

        self.auto_generate_plots = bool(auto_generate_plots)
        self.auto_write_printable = bool(auto_write_printable)
        self.auto_run_backtest = bool(auto_run_backtest)
        self.auto_save_models = bool(auto_save_models)
        self.backtest_top_k = int(backtest_top_k)
        self.backtest_holding_period = int(backtest_holding_period)
        self.backtest_cost_bps = float(backtest_cost_bps)

        # Optional drift mitigation: point-in-time rolling normalization
        self.enable_rolling_zscore = bool(enable_rolling_zscore)
        self.rolling_zscore_window = int(rolling_zscore_window)
        self.rolling_zscore_min_periods = int(rolling_zscore_min_periods)

        # Gate target definition
        self.gate_label_mode = gate_label_mode
        self.gate_topk = int(gate_topk)
        self.gate_top_pct = float(gate_top_pct)
        self.gate_cost_threshold = (None if gate_cost_threshold is None else float(gate_cost_threshold))

        # Backtest hygiene: if the gate passes too few names on a date, abstain (avoid forced actionability)
        self.backtest_reject_all_if_gate_pass_rate_below = (
            None
            if backtest_reject_all_if_gate_pass_rate_below is None
            else float(backtest_reject_all_if_gate_pass_rate_below)
        )

        # Ranker hygiene: optionally drop features that are constant across sectors within a Date.
        self.ranker_drop_date_constant_features = bool(ranker_drop_date_constant_features)
        self.ranker_constant_std_eps = float(ranker_constant_std_eps)
        self.ranker_constant_frac_threshold = float(ranker_constant_frac_threshold)
        
        # Will be set during run
        self.data: ShapingResult | None = None
        self.run_manager: RunManager | None = None
        
        # Fold result collectors
        self._gate_fold_results: list[FoldResult] = []
        self._ranker_fold_results: list[FoldResult] = []
    
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DualModelTrainer] {msg}")
    
    def _load_data(self) -> ShapingResult:
        """Load and validate dataset."""
        self._log(f"Loading dataset for horizon={self.horizon}...")

        # Prefer the final cached dataset in ModularMonolith/data/dataset/h{horizon}
        # for reproducibility + speed. Fall back to rebuilding via dataset_shaping
        # if the cache is missing or incomplete.
        try:
            data = load_dual_model_dataset_cached(
                self.horizon,
                enforce_full_universe=True,
                verbose=self.verbose,
            )
        except FileNotFoundError:
            data = load_dual_model_dataset(
                self.horizon,
                enforce_full_universe=True,
                verbose=self.verbose,
            )
        
        # Validate
        assert data.X.shape[0] == len(data.y_gate) == len(data.y_rank)
        assert data.group_sizes.sum() == len(data.y_gate)
        assert np.all(data.group_sizes == len(EXPECTED_SECTORS))
        
        self._log(f"Loaded: {data.X.shape[0]} rows, {data.X.shape[1]} features, {len(data.group_sizes)} dates")
        
        return data
    
    def _build_y_dataframe(self, data: ShapingResult) -> pd.DataFrame:
        """Build y DataFrame with all required columns for CV."""
        y = data.full_df[["Date", "Sector"]].copy()
        # Gate target can be defined in different ways.
        # - excess_gt_cost: label_excess > cost_threshold (default)
        # - topk: top-K sectors by label_excess within each Date
        # - top_pct: top-PCT sectors by label_excess within each Date (quantile labeling)
        if self.gate_label_mode == "excess_gt_cost":
            thr = float(data.cost_threshold) if self.gate_cost_threshold is None else float(self.gate_cost_threshold)
            y["y_gate"] = (data.full_df["label_excess"].to_numpy() > thr).astype(np.int8)
        elif self.gate_label_mode == "topk":
            k = max(1, min(int(self.gate_topk), len(EXPECTED_SECTORS)))
            tmp = data.full_df[["Date", "label_excess"]].copy()
            tmp["_rank"] = tmp.groupby("Date", sort=False)["label_excess"].rank(method="first", ascending=False)
            y["y_gate"] = (tmp["_rank"].to_numpy() <= float(k)).astype(np.int8)
        elif self.gate_label_mode == "top_pct":
            # Quantile-based labeling per Date (e.g. top 30% as positive)
            pct = float(self.gate_top_pct)
            if not np.isfinite(pct) or pct <= 0.0 or pct >= 1.0:
                raise ValueError("gate_top_pct must be in (0,1)")

            tmp = data.full_df[["Date", "label_excess"]].copy()
            tmp["_rank"] = tmp.groupby("Date", sort=False)["label_excess"].rank(method="first", ascending=False)
            tmp["_n"] = tmp.groupby("Date", sort=False)["label_excess"].transform("size")
            # select top ceil(pct*n) per date, but at least 1
            k_eff = np.ceil(tmp["_n"].to_numpy(dtype=float) * pct)
            k_eff = np.maximum(1.0, k_eff)
            y["y_gate"] = (tmp["_rank"].to_numpy(dtype=float) <= k_eff).astype(np.int8)
        else:
            raise ValueError(f"Unknown gate_label_mode={self.gate_label_mode}")
        y["label_excess"] = data.full_df["label_excess"].to_numpy()
        y["y_rank"] = np.asarray(data.y_rank).astype(np.float32)
        y["cost_bps"] = float(data.cost_threshold) * 10_000.0
        
        return y

    def _apply_point_in_time_rolling_zscore(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply point-in-time rolling z-score normalization to drift-prone scale features.

        This uses only past information for each (Sector, feature) time series:
        mean/std are computed on shifted values (t-1 backward), avoiding lookahead.
        
        FIX #5: Use stable row id to prevent drop/duplicate rows on restore.
        """
        if not self.enable_rolling_zscore:
            return X

        if self.data is None:
            return X

        keys = self.data.full_df[["Date", "Sector"]].reset_index(drop=True).copy()
        keys["_row"] = np.arange(len(keys), dtype=np.int64)
        if len(keys) != len(X):
            return X

        # Heuristic: normalize only scale-like columns; avoid cross-sectional ranks.
        def should_norm(col: str) -> bool:
            c = str(col).lower()
            if c.startswith("rank_"):
                return False
            return ("idio_vol" in c) or c.startswith("vol") or ("vix" in c)

        cols = [c for c in X.columns if should_norm(str(c))]
        if not cols:
            return X

        tmp = pd.concat([keys.reset_index(drop=True), X[cols].reset_index(drop=True)], axis=1)
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="raise")

        # Sort for deterministic rolling per Sector
        tmp = tmp.sort_values(["Sector", "Date"], kind="mergesort").reset_index(drop=True)

        w = max(5, int(self.rolling_zscore_window))
        mp = max(5, int(self.rolling_zscore_min_periods))

        out_vals = tmp[cols].copy()
        for sector, idx in tmp.groupby("Sector", sort=False).groups.items():
            sub = tmp.loc[idx, cols]
            sub = sub.apply(pd.to_numeric, errors="coerce")
            # past-only rolling stats
            mean = sub.shift(1).rolling(window=w, min_periods=mp).mean()
            std = sub.shift(1).rolling(window=w, min_periods=mp).std(ddof=0)
            std = std.replace(0.0, np.nan)
            z = (sub - mean) / std
            out_vals.loc[idx, cols] = z

        tmp[cols] = out_vals[cols]

        # FIX #5: Restore by stable row id (already in tmp from keys)
        tmp = tmp.sort_values("_row", kind="mergesort")

        X2 = X.copy()
        X2[cols] = tmp[cols].to_numpy()
        return X2

    def _select_ranker_feature_subset(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Drop features that are (almost) constant across sectors within a Date.

        This focuses the ranker on cross-sectional discriminators. Macro *level*
        features typically have ~zero cross-sectional variance and can be better used
        as conditioning (e.g., for the gate).
        """
        if not self.ranker_drop_date_constant_features:
            return X
        if X.empty:
            return X
        if ("Date" not in y.columns) or ("Sector" not in y.columns):
            return X

        tmp = pd.concat([y[["Date", "Sector"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="raise")
        tmp = tmp.sort_values(["Date", "Sector"], kind="mergesort").reset_index(drop=True)

        sizes = tmp.groupby("Date", sort=False).size().to_numpy(dtype=int)
        if len(sizes) == 0:
            return X
        if int(sizes.min()) != int(sizes.max()):
            return X
        g = int(sizes[0])
        if g < 2:
            return X

        cols = list(X.columns)
        vals = tmp[cols].to_numpy(dtype=np.float32, copy=False)
        n_dates = int(len(sizes))
        if vals.shape[0] != n_dates * g:
            return X

        vals3 = vals.reshape(n_dates, g, len(cols))
        cs_std = np.nanstd(vals3, axis=1)
        
        # ✅ Stricter threshold: feature is "constant" if std < 1e-6 (not 1e-12)
        # Many features with std ~ 1e-9 still cause model degeneracy
        constant = (~np.isfinite(cs_std)) | (cs_std <= 1e-6)
        frac_constant = constant.mean(axis=0)

        keep_cols = [c for c, fc in zip(cols, frac_constant) if float(fc) < float(self.ranker_constant_frac_threshold)]
        
        # ✅ Fail-fast: if too few cross-sectional features remain, raise error
        if len(keep_cols) < 10:
            raise ValueError(
                f"Cross-sectional degeneracy: only {len(keep_cols)}/{len(cols)} features "
                f"have within-date variance. Need at least 10 for ranking."
            )
        
        if self.verbose:
            self._log(
                f"Ranker feature split: kept={len(keep_cols)}/{len(cols)} "
                f"(dropped_constant={len(cols) - len(keep_cols)})"
            )

        if not keep_cols:
            return X
        
        # DIAGNOSTIC: Report feature dispersion stats for final kept features
        if keep_cols and self.verbose:
            # Sample 100 random dates to avoid overhead
            sample_dates = tmp["Date"].drop_duplicates().sample(min(100, n_dates), random_state=42).tolist()
            tmp_sample = tmp[tmp["Date"].isin(sample_dates)]
            
            feature_cs_std = []
            for col in keep_cols[:10]:  # Show top 10 for brevity
                by_date = tmp_sample.groupby("Date")[col].std()
                mean_cs_std = by_date.mean()
                feature_cs_std.append((col, mean_cs_std))
            
            feature_cs_std.sort(key=lambda x: x[1], reverse=True)
            self._log("Top cross-sectional features (by mean within-date std):")
            for col, std in feature_cs_std[:5]:
                self._log(f"  {col}: {std:.6f}")
        
        return X[keep_cols]

    def _write_printable(self, result: DualModelResult) -> None:
        if self.run_manager is None or self.run_manager.printable_dir is None:
            return

        from ModularMonolith.src.reporting.metrics_rank import format_metrics_summary

        lines: list[str] = []
        lines.append(f"Run: {self.run_manager.run_dir}")
        lines.append(f"Horizon: {self.horizon}")
        lines.append(f"Seeds: {self.seeds}")
        lines.append(f"Gate label mode: {self.gate_label_mode} (topk={self.gate_topk})")
        lines.append("")
        lines.append(format_metrics_summary(result.gate_metrics, result.rank_metrics))
        lines.append("")

        drift = result.drift_report or {}
        drift_snapshot = drift
        if isinstance(drift, dict) and "snapshot" in drift and isinstance(drift.get("snapshot"), dict):
            drift_snapshot = drift.get("snapshot") or {}

        if isinstance(drift_snapshot, dict) and drift_snapshot:
            lines.append("=" * 60)
            lines.append("DRIFT (from drift_report.json)")
            lines.append("=" * 60)
            lines.append(
                f"n_features={drift_snapshot.get('n_features')}  "
                f"n_drifted={drift_snapshot.get('n_features_drifted')}  "
                f"mean_psi={drift_snapshot.get('mean_psi')}  max_psi={drift_snapshot.get('max_psi')}"
            )
            feats = drift_snapshot.get("features_with_drift") or []
            if feats:
                lines.append("Top drifted features (up to 10):")
                psi_by_feat = drift_snapshot.get("psi_by_feature") or {}
                for feat in list(feats)[:10]:
                    lines.append(f"  {feat}: PSI={psi_by_feat.get(feat)}")
            lines.append("")

        self.run_manager.save_artifact("\n".join(lines), "summary.md", "printable")

    def _run_oof_backtest(self, result: DualModelResult) -> None:
        if self.run_manager is None or self.run_manager.backtest_dir is None:
            return

        from ModularMonolith.src.backtesting.vectorized_backtester import VectorizedBacktester

        merged = result.merged_oof.copy()
        if not {"Date", "Sector", "rank_mean", "label_excess"}.issubset(merged.columns):
            return

        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
        merged = merged.dropna(subset=["Date", "Sector"]).copy()

        returns_wide = merged.pivot_table(index="Date", columns="Sector", values="label_excess", aggfunc="mean")
        # Backtester assumes HIGHER score = better; rank_mean convention is LOWER = better.
        preds_rank = -merged.pivot_table(index="Date", columns="Sector", values="rank_mean", aggfunc="mean")

        bt = VectorizedBacktester(returns_df=returns_wide)

        res_rank = bt.run(
            preds_rank,
            top_k=self.backtest_top_k,
            holding_period=self.backtest_holding_period,
            cost_bps=self.backtest_cost_bps,
        )

        self.run_manager.save_artifact(res_rank.strategy_returns, "strategy_returns_rank_only.csv", "backtest")
        self.run_manager.save_artifact(res_rank.equity_curve, "equity_curve_rank_only.csv", "backtest")
        self.run_manager.save_artifact(res_rank.weights, "weights_rank_only.csv", "backtest")
        self.run_manager.save_artifact(res_rank.metrics, "backtest_metrics_rank_only.json", "backtest")

        if "p_gate_mean" in merged.columns:
            preds_gated_long = merged[["Date", "Sector", "rank_mean", "p_gate_mean"] + (["p_gate_std"] if "p_gate_std" in merged.columns else []) + (["fold_id"] if "fold_id" in merged.columns else [])].copy()

            # Prefer fold-safe thresholds from gate fold results (avoid global snooping)
            fold_thresholds: dict[int, dict[str, float]] = {}
            try:
                for fr in (result.gate_fold_results or []):
                    fold_thresholds[int(fr.fold_id)] = dict(fr.thresholds or {})
            except Exception:
                fold_thresholds = {}

            if fold_thresholds and ("fold_id" in preds_gated_long.columns):
                def _row_pass(r: pd.Series) -> bool:
                    t = fold_thresholds.get(int(r.get("fold_id", -1)), {})
                    p_star = float(t.get("p_star", 0.5))
                    u_star = float(t.get("u_star", float("inf")))
                    ok = float(r.get("p_gate_mean", 0.0)) >= p_star
                    if ("p_gate_std" in preds_gated_long.columns) and np.isfinite(u_star):
                        ok = ok and (float(r.get("p_gate_std", float("inf"))) <= u_star)
                    return bool(ok)

                passed = preds_gated_long.apply(_row_pass, axis=1)
                preds_gated_long.loc[~passed, "rank_mean"] = np.nan
            else:
                # Fallback: global p_star (legacy)
                p_star = float((result.thresholds or {}).get("p_star", 0.5))
                preds_gated_long.loc[preds_gated_long["p_gate_mean"] < p_star, "rank_mean"] = np.nan

            # Reject-all policy if gate passes too few names on a date
            thr = self.backtest_reject_all_if_gate_pass_rate_below
            if thr is not None:
                thr = float(thr)
                if np.isfinite(thr) and 0.0 < thr < 1.0:
                    pass_rate = preds_gated_long.groupby("Date", sort=False)["rank_mean"].apply(lambda s: float(np.mean(np.isfinite(s.to_numpy(dtype=float)))))
                    bad_dates = pass_rate[pass_rate < thr].index
                    if len(bad_dates):
                        preds_gated_long.loc[preds_gated_long["Date"].isin(bad_dates), "rank_mean"] = np.nan
            preds_gated = preds_gated_long.pivot_table(index="Date", columns="Sector", values="rank_mean", aggfunc="mean")
            # Backtester assumes HIGHER score = better; rank_mean convention is LOWER = better.
            preds_gated = -preds_gated

            res_gated = bt.run(
                preds_gated,
                top_k=self.backtest_top_k,
                holding_period=self.backtest_holding_period,
                cost_bps=self.backtest_cost_bps,
            )

            self.run_manager.save_artifact(res_gated.strategy_returns, "strategy_returns_gated.csv", "backtest")
            self.run_manager.save_artifact(res_gated.equity_curve, "equity_curve_gated.csv", "backtest")
            self.run_manager.save_artifact(res_gated.weights, "weights_gated.csv", "backtest")
            self.run_manager.save_artifact(res_gated.metrics, "backtest_metrics_gated.json", "backtest")

    def _save_final_models(self, X: pd.DataFrame, y: pd.DataFrame, group_sizes: np.ndarray) -> None:
        """Save final models trained on all data with most stable features from CV.
        
        FIX #3: Make final models consistent with CV training:
        - Use best params from folds (not just self.gate_params/self.ranker_params)
        - Apply time-decay sample weights (anchored on last training date)
        - Use stable feature set from CV (frequency-based selection)
        """
        if self.run_manager is None or self.run_manager.models_dir is None:
            return

        def _top_features(folds: list[FoldResult], available_cols: list[str]) -> list[str]:
            """Select top-K features by selection frequency across folds."""
            counts: dict[str, int] = {}
            for fr in folds:
                for feat in (fr.chosen_features or []):
                    if feat in available_cols:
                        counts[feat] = counts.get(feat, 0) + 1
            
            if not counts:
                # Fallback: use first top_n_features from available columns
                return available_cols[: self.top_n_features]
            
            feats_sorted = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            selected = [f for f, _ in feats_sorted[: self.top_n_features]]
            
            # Ensure we have at least some features
            if not selected:
                selected = available_cols[: self.top_n_features]
            
            return selected
        
        def _best_params(folds: list[FoldResult], default: dict[str, Any]) -> dict[str, Any]:
            """Extract most common params across folds (mode for each key)."""
            from collections import defaultdict
            param_counts: dict[str, dict[Any, int]] = defaultdict(lambda: {})
            
            for fr in folds:
                for k, v in (fr.chosen_params or {}).items():
                    if isinstance(v, (int, float, str, bool)):
                        param_counts[k][v] = param_counts[k].get(v, 0) + 1
            
            best = dict(default)
            for k, v_counts in param_counts.items():
                if v_counts:
                    best[k] = max(v_counts.items(), key=lambda x: x[1])[0]
            
            return best

        # FIX #3: Compute time-decay weights (anchored on last training date)
        from ModularMonolith.src.cv.run_nested_cv import compute_exponential_time_decay
        train_dates = y["Date"].to_numpy()
        # Use horizon-specific half-life (5 years = 1260 trading days)
        half_life_days = 1260
        time_weights = compute_exponential_time_decay(
            train_dates=train_dates,
            half_life_days=half_life_days
        )

        # Gate features and params
        gate_feats = _top_features(self._gate_fold_results, list(X.columns))
        gate_params_best = _best_params(self._gate_fold_results, self.gate_params)
        
        # Ranker features and params: apply same preprocessing as in _train_ranker
        X_rank = self._select_ranker_feature_subset(X, y)
        rank_feats = _top_features(self._ranker_fold_results, list(X_rank.columns))
        rank_params_best = _best_params(self._ranker_fold_results, self.ranker_params)

        # Save model specifications (features + best hyperparams from CV)
        self.run_manager.save_artifact(
            {"features": gate_feats, "params": gate_params_best, "from_cv": True},
            "gate_model_spec.json",
            "models",
        )
        self.run_manager.save_artifact(
            {"features": rank_feats, "params": rank_params_best, "from_cv": True},
            "ranker_model_spec.json",
            "models",
        )

        try:
            # FIX #3: Train gate with best params and time-decay weights
            gate_model = gate_classifier_factory(
                0, gate_params_best, use_gpu=self.use_gpu, deterministic=self.deterministic_mode
            )
            fit_classifier(
                gate_model,
                X[gate_feats],
                y["y_gate"],
                groups=None,
                sample_weight=time_weights,
            )
            gate_model.get_booster().save_model(str(self.run_manager.models_dir / "gate_model.json"))
            self._log(f"Saved gate model with {len(gate_feats)} features (time-weighted)")

            # FIX #3: Train ranker with best params and time-decay weights (per-group)
            # Aggregate per-row time_weights to per-group (mean per date)
            group_weights = np.zeros(len(group_sizes), dtype=np.float32)
            idx = 0
            for i, size in enumerate(group_sizes):
                group_weights[i] = time_weights[idx:idx+size].mean()
                idx += size
            
            rank_model = ranker_factory(
                0, rank_params_best, use_gpu=self.use_gpu, deterministic=self.deterministic_mode
            )
            fit_ranker(
                rank_model,
                X_rank[rank_feats],
                y["y_rank"],
                group_sizes=group_sizes,
                sample_weight=group_weights,  # Per-group weights (len = n_dates)
            )
            rank_model.get_booster().save_model(str(self.run_manager.models_dir / "ranker_model.json"))
            self._log(f"Saved ranker model with {len(rank_feats)} features (per-group time-decay weights)")
        except Exception as e:
            self._log(f"Error saving final models: {e}")
            self.run_manager.save_artifact({"error": str(e)}, "model_save_error.json", "models")
    
    def _get_outer_cv(self) -> PurgedWalkForwardCV:
        """Get outer CV splitter."""
        cv_params = get_cv_config(self.horizon)
        return PurgedWalkForwardCV(
            n_splits=cv_params.get("n_splits"),
            test_size=cv_params["test_size"],
            purge_gap=cv_params["purge_gap"],
            embargo=cv_params["embargo"],
            min_train_size=cv_params["min_train_size"],
            test_start=cv_params.get("test_start"),
        )
    
    def _build_gate_config(self) -> NestedCVConfig:
        """Build config for gate classifier."""
        cv_params = get_cv_config(self.horizon)
        
        inner_params = dict(self.gate_params)
        if self.optuna_n_trials > 0:
            inner_params.update(GATE_SEARCH_SPACE)
        
        return NestedCVConfig(
            horizon=self.horizon,
            cost_bps=float(self.data.cost_threshold) * 10_000.0,
            outer_params=cv_params,
            inner_params=inner_params,
            seeds=self.seeds,
            task="gate_classifier",
            feature_selection=self.feature_selection,
            optuna_n_trials=self.optuna_n_trials,
            top_n_features=self.top_n_features,
            gate_enable_polarity_correction=True,
        )
    
    def _build_ranker_config(self) -> NestedCVConfig:
        """Build config for ranker."""
        cv_params = get_cv_config(self.horizon)
        
        inner_params = dict(self.ranker_params)
        if self.optuna_n_trials > 0:
            inner_params.update(RANKER_SEARCH_SPACE)
        
        return NestedCVConfig(
            horizon=self.horizon,
            cost_bps=float(self.data.cost_threshold) * 10_000.0,
            outer_params=cv_params,
            inner_params=inner_params,
            seeds=self.seeds,
            task="ranker",
            feature_selection=self.feature_selection,
            optuna_n_trials=self.optuna_n_trials,
            top_n_features=self.top_n_features,
        )
    
    def _train_gate(self, X: pd.DataFrame, y: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
        """Train gate classifier via nested CV."""
        self._log("Training Gate Classifier...")
        
        config = self._build_gate_config()
        outer_cv = self._get_outer_cv()
        
        def fold_sink(fr: FoldResult) -> None:
            self._gate_fold_results.append(fr)
        
        oof = run_outer_folds(
            config=config,
            X=X,
            y=y,
            groups=groups,
            outer_cv=outer_cv,
            inner_cv_factory=inner_cv_factory,
            model_factory=(lambda seed, params: gate_classifier_factory(
                seed, params, use_gpu=self.use_gpu, deterministic=self.deterministic_mode
            )),
            fit_fn=fit_classifier,
            predict_fn=predict_classifier,
            fold_result_sink=fold_sink,
        )
        
        self._log(f"Gate OOF: {len(oof)} predictions from {len(self._gate_fold_results)} folds")
        return oof
    
    def _train_ranker(self, X: pd.DataFrame, y: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
        """Train ranker via nested CV.
        
        FIX #8: Validates that groups match Date structure with expected sector count.
        """
        self._log("Training Ranker...")

        X_rank = self._select_ranker_feature_subset(X, y)
        
        # FIX #8: Ranker group validation (each group = single Date with len(EXPECTED_SECTORS) rows)
        if "Date" in y.columns:
            from ModularMonolith.data.dataset_shaping import EXPECTED_SECTORS
            dates_unique = y.groupby(y["Date"])["Date"].count()
            expected_size = len(EXPECTED_SECTORS)
            if not (dates_unique == expected_size).all():
                mismatched = dates_unique[dates_unique != expected_size]
                self._log(
                    f"WARNING: Ranker group size mismatch. Expected {expected_size} sectors per date, "
                    f"but found {len(mismatched)} dates with different counts (first 5): "
                    f"{mismatched.head().to_dict()}"
                )
        
        config = self._build_ranker_config()
        outer_cv = self._get_outer_cv()
        
        def fold_sink(fr: FoldResult) -> None:
            self._ranker_fold_results.append(fr)
        
        oof = run_outer_folds(
            config=config,
            X=X_rank,
            y=y,
            groups=groups,
            outer_cv=outer_cv,
            inner_cv_factory=inner_cv_factory,
            model_factory=(lambda seed, params: ranker_factory(
                seed, params, use_gpu=self.use_gpu, deterministic=self.deterministic_mode
            )),
            fit_fn=fit_ranker,
            predict_fn=predict_ranker,
            fold_result_sink=fold_sink,
        )
        
        self._log(f"Ranker OOF: {len(oof)} predictions from {len(self._ranker_fold_results)} folds")
        
        # DIAGNOSTIC: Check for constant predictions per date (causes ConstantInputWarning)
        if "Date" in oof.columns and "pred_mean" in oof.columns:
            pred_std_by_date = oof.groupby("Date")["pred_mean"].std()
            n_constant = (pred_std_by_date <= 1e-12).sum()
            pct_constant = 100.0 * n_constant / len(pred_std_by_date)
            if pct_constant > 5.0:
                self._log(
                    f"WARNING: {n_constant}/{len(pred_std_by_date)} dates ({pct_constant:.1f}%) "
                    f"have near-constant predictions (std <= 1e-12). "
                    f"This causes ConstantInputWarning and breaks rank IC. "
                    f"Check that ranker features have cross-sectional variance."
                )
        
        return oof
    
    def _merge_oof(self, gate_oof: pd.DataFrame, ranker_oof: pd.DataFrame) -> pd.DataFrame:
        """
        Merge gate and ranker OOF predictions.
        
        Implements filter-then-sort:
        - p_gate_mean: Gate probability
        - p_gate_std: Gate uncertainty (epistemic)
        - rank_mean: Ranker score
        - rank_std: Ranker uncertainty
        - actionable: Boolean, whether gate passes threshold
        """
        # Rename columns to avoid conflict
        gate = gate_oof.rename(columns={
            "pred_mean": "p_gate_mean",
            "pred_std": "p_gate_std",
        })
        
        ranker = ranker_oof.rename(columns={
            "pred_mean": "rank_mean",
            "pred_std": "rank_std",
        })
        
        # Merge on Date, Sector
        merge_cols = ["Date", "Sector"]
        if "fold_id" in gate.columns and "fold_id" in ranker.columns:
            merge_cols.append("fold_id")
        
        merged = gate[merge_cols + ["p_gate_mean", "p_gate_std", "label_excess"]].merge(
            ranker[merge_cols + ["rank_mean", "rank_std"]],
            on=merge_cols,
            how="inner",
        )
        
        # Compute uncertainty metrics
        # Aleatoric proxy: p(1-p) for classifier (Bernoulli variance)
        merged["aleatoric_gate"] = merged["p_gate_mean"] * (1 - merged["p_gate_mean"])
        
        # Rank margin: distance from median rank score per date
        merged["rank_margin"] = merged.groupby("Date")["rank_mean"].transform(
            lambda x: x - x.median()
        )
        
        return merged
    
    def _compute_and_log_diagnostics(
        self,
        merged_oof: pd.DataFrame,
        gate_oof: pd.DataFrame,
    ) -> None:
        """
        Compute and log comprehensive diagnostics for ranking metrics.
        
        Implements:
        A) Lift validation with base_rate checks
        B) Gate target prevalence analysis + polarity check
        C) Constant predictions analysis
        D) Negative controls (random ranker, oracle ranker)
        """
        from ModularMonolith.src.reporting.metrics_diagnostics import (
            compute_lift_with_diagnostics,
            compute_gate_prevalence_diagnostics,
            random_ranker_control,
            oracle_ranker_control,
            gate_oracle_control,
            gate_polarity_check,
            analyze_constant_predictions,
            print_diagnostics_report,
        )
        
        print("\n" + "=" * 80)
        print("RUNNING METRICS DIAGNOSTICS")
        print("=" * 80)
        
        # A) Lift@K diagnostics with base_rate validation
        try:
            lift_diag_k1 = compute_lift_with_diagnostics(
                merged_oof["label_excess"],
                merged_oof["rank_mean"],
                merged_oof["Date"],
                k=1,
                cost_threshold=self.data.cost_threshold,
            )
            
            lift_diag_k3 = compute_lift_with_diagnostics(
                merged_oof["label_excess"],
                merged_oof["rank_mean"],
                merged_oof["Date"],
                k=3,
                cost_threshold=self.data.cost_threshold,
            )
            
            # Save diagnostics to run artifacts
            (self.run_manager.run_dir / "diagnostic").mkdir(parents=True, exist_ok=True)
            self.run_manager.save_artifact(
                {"lift_k1": lift_diag_k1.to_dict(), "lift_k3": lift_diag_k3.to_dict()},
                "lift_diagnostics.json",
                "diagnostic"
            )
        except Exception as e:
            import traceback
            print(f"WARNING: Lift diagnostics failed: {e}")
            print(traceback.format_exc())
            lift_diag_k1 = None
            lift_diag_k3 = None
        
        # B) Gate prevalence diagnostics
        gate_diag = None
        try:
            if "y_gate" in gate_oof.columns and "Date" in gate_oof.columns:
                gate_diag = compute_gate_prevalence_diagnostics(
                    gate_oof["y_gate"],
                    gate_oof["Date"],
                )
                
                # Save gate diagnostics
                self.run_manager.save_artifact(
                    gate_diag.to_dict(),
                    "gate_prevalence_diagnostics.json",
                    "diagnostic"
                )
        except Exception as e:
            import traceback
            print(f"WARNING: Gate prevalence diagnostics failed: {e}")
            print(traceback.format_exc())
        
        # B2) Gate polarity check (if AUC < 0.5, check for inversion)
        gate_polarity = None
        try:
            if "y_gate" in gate_oof.columns and "pred_mean" in gate_oof.columns:
                gate_polarity = gate_polarity_check(
                    gate_oof["y_gate"],
                    gate_oof["pred_mean"],
                )
                
                self.run_manager.save_artifact(
                    gate_polarity,
                    "gate_polarity_check.json",
                    "diagnostic"
                )
        except Exception as e:
            import traceback
            print(f"WARNING: Gate polarity check failed: {e}")
            print(traceback.format_exc())
        
        # B3) Constant predictions analysis
        constant_pred_analysis = None
        try:
            # ✅ Φτιάξε y_keys με Date/Sector ώστε _select_ranker_feature_subset να δουλέψει
            keys = self.data.full_df[["Date", "Sector"]].reset_index(drop=True)
            y_keys = self.data.full_df[["Date", "Sector", "label_excess"]].reset_index(drop=True)
            
            X_rank_feats = self._select_ranker_feature_subset(
                self.data.X.reset_index(drop=True),
                y_keys,
            )
            
            # Add Date/Sector keys for merge
            X_rank_with_keys = pd.concat([keys, X_rank_feats], axis=1)
            
            constant_pred_analysis = analyze_constant_predictions(
                merged_oof,
                X_rank_with_keys,
                cost_threshold=self.data.cost_threshold,
            )
            
            self.run_manager.save_artifact(
                constant_pred_analysis,
                "constant_predictions_analysis.json",
                "diagnostic"
            )
        except Exception as e:
            import traceback
            print(f"WARNING: Constant predictions analysis failed: {e}")
            print(traceback.format_exc())
        
        # ✅ Invariant checks πριν metrics (πιάνει merge bugs)
        try:
            tmp = merged_oof[["Date", "Sector"]].copy()
            tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
            
            # Check universe size per date
            per_date = tmp.groupby("Date", sort=False)["Sector"].nunique()
            bad = per_date[per_date != len(EXPECTED_SECTORS)]
            if len(bad):
                raise ValueError(
                    f"merged_oof has wrong universe size on {len(bad)} dates. "
                    f"Expected {len(EXPECTED_SECTORS)} sectors per date. "
                    f"Example: {bad.head().to_dict()}"
                )
            
            # Check for duplicate (Date, Sector) rows
            dup = merged_oof.duplicated(subset=["Date", "Sector"]).sum()
            if dup:
                raise ValueError(
                    f"merged_oof has {dup} duplicated (Date, Sector) rows. "
                    f"Merge contract broken."
                )
        except Exception as e:
            import traceback
            print(f"WARNING: merged_oof invariant check failed: {e}")
            print(traceback.format_exc())
        
        # C) Negative controls: Random ranker
        random_control = None
        try:
            random_control = random_ranker_control(
                merged_oof["label_excess"],
                merged_oof["Date"],
                seed=42,
            )
            
            self.run_manager.save_artifact(
                random_control,
                "random_ranker_control.json",
                "diagnostic"
            )
        except Exception as e:
            import traceback
            print(f"WARNING: Random ranker control failed: {e}")
            print(traceback.format_exc())
        
        # D) Positive controls: Oracle ranker
        oracle_control = None
        try:
            oracle_control = oracle_ranker_control(
                merged_oof["label_excess"],
                merged_oof["Date"],
            )
            
            self.run_manager.save_artifact(
                oracle_control,
                "oracle_ranker_control.json",
                "diagnostic"
            )
        except Exception as e:
            import traceback
            print(f"WARNING: Oracle ranker control failed: {e}")
            print(traceback.format_exc())
        
        # E) Gate oracle control
        try:
            if "y_gate" in gate_oof.columns:
                gate_oracle = gate_oracle_control(gate_oof["y_gate"])
                
                self.run_manager.save_artifact(
                    gate_oracle,
                    "gate_oracle_control.json",
                    "diagnostic"
                )
        except Exception as e:
            import traceback
            print(f"WARNING: Gate oracle control failed: {e}")
            print(traceback.format_exc())
        
        # Print consolidated report
        try:
            if lift_diag_k1 and lift_diag_k3:
                print_diagnostics_report(
                    lift_diag_k1,
                    lift_diag_k3,
                    gate_diag,
                    random_control,
                    oracle_control,
                    gate_polarity,
                    constant_pred_analysis,
                )
            else:
                print("WARNING: Lift diagnostics failed, skipping consolidated report")
        except Exception as e:
            import traceback
            print(f"WARNING: print_diagnostics_report failed: {e}")
            print(traceback.format_exc())
            print(traceback.format_exc())
        
        print("=" * 80 + "\n")
    
    def _compute_metrics(
        self,
        merged_oof: pd.DataFrame,
        gate_oof: pd.DataFrame,
    ) -> tuple[dict[str, float], RankMetrics, RankMetrics | None, dict[Any, RankMetrics]]:
        """
        Compute all metrics from OOF predictions.
        
        Uses per-fold thresholds to avoid data snooping:
        - For each fold, applies that fold's p_star/u_star thresholds
        - Aggregates across folds for final metrics
        
        If per-fold thresholds are available (from run_inner_search),
        they are used. Otherwise falls back to threshold=0.5.
        """
        self._log("Computing metrics...")
        
        # Build per-fold threshold map from gate fold results
        fold_thresholds: dict[int, dict[str, float]] = {}
        for fr in self._gate_fold_results:
            fold_thresholds[fr.fold_id] = fr.thresholds
        
        # Gate metrics with per-fold thresholds
        if fold_thresholds and "fold_id" in gate_oof.columns:
            # Apply per-fold thresholds: no global snooping
            y_true_list = []
            y_pred_binary_list = []
            
            for fold_id, fold_df in gate_oof.groupby("fold_id"):
                thresh = fold_thresholds.get(int(fold_id), {})
                p_star = thresh.get("p_star", 0.5)
                u_star = thresh.get("u_star", float("inf"))

                if "y_gate" in fold_df.columns:
                    y_true = pd.to_numeric(fold_df["y_gate"], errors="coerce").fillna(0).astype(int)
                else:
                    y_true = (fold_df["label_excess"] > self.data.cost_threshold).astype(int)
                
                # Apply both p_star and u_star (uncertainty) thresholds
                if "pred_std" in fold_df.columns and u_star < float("inf"):
                    y_pred_binary = (
                        (fold_df["pred_mean"] >= p_star) &
                        (fold_df["pred_std"] <= u_star)
                    ).astype(int)
                else:
                    y_pred_binary = (fold_df["pred_mean"] >= p_star).astype(int)
                
                y_true_list.extend(y_true.tolist())
                y_pred_binary_list.extend(y_pred_binary.tolist())
            
            y_true_all = np.array(y_true_list)
            y_pred_all = np.array(y_pred_binary_list)
            
            # Compute metrics from per-fold thresholded predictions
            tp = float(((y_pred_all == 1) & (y_true_all == 1)).sum())
            fp = float(((y_pred_all == 1) & (y_true_all == 0)).sum())
            fn = float(((y_pred_all == 0) & (y_true_all == 1)).sum())
            tn = float(((y_pred_all == 0) & (y_true_all == 0)).sum())
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            # Also compute AUC on raw probabilities (threshold-independent)
            from sklearn.metrics import roc_auc_score
            try:
                if "y_gate" in gate_oof.columns:
                    y_auc = pd.to_numeric(gate_oof["y_gate"], errors="coerce").fillna(0).astype(int)
                else:
                    y_auc = (gate_oof["label_excess"] > self.data.cost_threshold).astype(int)
                auc = roc_auc_score(y_auc, gate_oof["pred_mean"])
            except Exception:
                auc = 0.5
            
            gate_metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "auc": float(auc),
                "pred_positive_rate": float(np.mean(y_pred_all)) if len(y_pred_all) else 0.0,
                "n": int(len(y_pred_all)),
                "using_per_fold_thresholds": True,
            }
            self._log(f"Gate metrics computed with per-fold thresholds (p_star/u_star)")
        else:
            # Fallback: use fixed threshold=0.5
            gate_metrics = gate_classification_metrics(
                (pd.to_numeric(gate_oof["y_gate"], errors="coerce").fillna(0).astype(int) if "y_gate" in gate_oof.columns else (gate_oof["label_excess"] > self.data.cost_threshold)),
                gate_oof["pred_mean"],
                threshold=0.5,
            )
            gate_metrics["using_per_fold_thresholds"] = False

            try:
                gate_metrics["pred_positive_rate"] = float(np.mean((gate_oof["pred_mean"] >= 0.5).astype(int)))
            except Exception:
                gate_metrics["pred_positive_rate"] = 0.0
            gate_metrics["n"] = int(len(gate_oof))

        # Gate top-k policy diagnostics (precision/recall at fixed K per date)
        # Helps detect if the gate is producing too many false positives.
        try:
            k = int(self.gate_topk)
            if k > 0 and ("Date" in gate_oof.columns) and ("pred_mean" in gate_oof.columns):
                df = gate_oof[["Date", "pred_mean"]].copy()
                if "y_gate" in gate_oof.columns:
                    df["y_true"] = pd.to_numeric(gate_oof["y_gate"], errors="coerce").fillna(0).astype(int)
                else:
                    df["y_true"] = (gate_oof["label_excess"] > self.data.cost_threshold).astype(int)

                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).reset_index(drop=True)
                if len(df):
                    df["pred_rank"] = df.groupby("Date", sort=False)["pred_mean"].rank(ascending=False, method="first")
                    df["y_pred_topk"] = (df["pred_rank"] <= k).astype(int)

                    y_true = df["y_true"].to_numpy(dtype=int)
                    y_pred = df["y_pred_topk"].to_numpy(dtype=int)

                    tp = float(((y_pred == 1) & (y_true == 1)).sum())
                    fp = float(((y_pred == 1) & (y_true == 0)).sum())
                    fn = float(((y_pred == 0) & (y_true == 1)).sum())
                    tn = float(((y_pred == 0) & (y_true == 0)).sum())

                    precision_k = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall_k = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
                    accuracy_k = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

                    gate_metrics.update(
                        {
                            "topk_policy_k": int(k),
                            "topk_policy_precision": float(precision_k),
                            "topk_policy_recall": float(recall_k),
                            "topk_policy_f1": float(f1_k),
                            "topk_policy_accuracy": float(accuracy_k),
                            "topk_policy_pred_rate": float(np.mean(y_pred)) if len(y_pred) else 0.0,
                        }
                    )
        except Exception:
            # Never fail the run due to diagnostics
            pass
        
        # Ranker metrics
        rank_metrics = compute_rank_metrics(
            merged_oof["label_excess"],
            merged_oof["rank_mean"],
            merged_oof["Date"],
            cost_threshold=self.data.cost_threshold,
        )
        
        # DIAGNOSTIC: Compute lift diagnostics with validation
        self._compute_and_log_diagnostics(merged_oof, gate_oof)

        # Ranker metrics after applying gate policy (fold-safe when fold_id + per-fold thresholds exist)
        rank_metrics_gated: RankMetrics | None = None
        try:
            if ("p_gate_mean" in merged_oof.columns) and ("fold_id" in merged_oof.columns) and fold_thresholds:
                mm = merged_oof.copy()
                p_map = {int(k): float(v.get("p_star", 0.5)) for k, v in fold_thresholds.items()}
                u_map = {int(k): float(v.get("u_star", float("inf"))) for k, v in fold_thresholds.items()}
                mm["_p_star"] = mm["fold_id"].map(p_map).fillna(0.5).astype(float)
                mm["_u_star"] = mm["fold_id"].map(u_map).fillna(float("inf")).astype(float)
                passed = mm["p_gate_mean"].astype(float) >= mm["_p_star"]
                if "p_gate_std" in mm.columns:
                    passed = passed & (mm["p_gate_std"].astype(float) <= mm["_u_star"])
                mm = mm.loc[passed].copy()
                if len(mm) >= 10:
                    rank_metrics_gated = compute_rank_metrics(
                        mm["label_excess"],
                        mm["rank_mean"],
                        mm["Date"],
                        cost_threshold=self.data.cost_threshold,
                    )
                    gate_metrics["rank_gated_pass_rate"] = float(len(mm)) / float(len(merged_oof)) if len(merged_oof) else 0.0
        except Exception:
            rank_metrics_gated = None
        
        regime_metrics: dict[Any, RankMetrics] = {}
        
        return gate_metrics, rank_metrics, rank_metrics_gated, regime_metrics
    
    def _compute_feature_stability(self) -> pd.DataFrame:
        """Compute feature selection stability across folds."""
        all_features: dict[str, list[int]] = {}
        all_importances: dict[str, list[float]] = {}
        
        for fr in self._gate_fold_results + self._ranker_fold_results:
            for feat in fr.chosen_features:
                all_features.setdefault(feat, []).append(fr.fold_id)
                try:
                    imp = float((fr.feature_importance or {}).get(feat, 0.0))
                except Exception:
                    imp = 0.0
                if np.isfinite(imp):
                    all_importances.setdefault(feat, []).append(imp)
        
        rows = []
        for feat, folds in all_features.items():
            rows.append({
                "feature": feat,
                "selection_count": len(folds),
                "folds": sorted(set(folds)),
                "mean_importance": float(np.mean(all_importances.get(feat, [0.0]))),
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("selection_count", ascending=False).reset_index(drop=True)
        return df
    
    def _compute_drift(self) -> dict[str, Any]:
        """Compute feature drift between train/test splits."""
        if self.data is None:
            return {}

        psi_threshold = 0.20

        # Use union of selected features across folds (more actionable + faster)
        selected: list[str] = []
        try:
            s = set()
            for fr in (self._gate_fold_results + self._ranker_fold_results):
                for f in (fr.chosen_features or []):
                    s.add(str(f))
            selected = [c for c in self.data.X.columns if c in s]
        except Exception:
            selected = []

        feature_cols = selected or list(self.data.X.columns)

        # Proxy snapshot drift: first vs last window (diagnostic only)
        n = int(len(self.data.X))
        n_window = max(100, n // 5)
        reference_df = self.data.X.iloc[:n_window][feature_cols]
        current_df = self.data.X.iloc[-n_window:][feature_cols]

        detector = DriftDetector(reference_df)
        report = detector.compute_drift(current_df, psi_threshold=psi_threshold)
        alerts = detector.get_alerts(report, psi_threshold=psi_threshold)

        # Rolling, point-in-time safe monitoring over dates
        dates = self.data.full_df["Date"].reset_index(drop=True)
        feats = self.data.X[feature_cols].reset_index(drop=True)
        drift_df = pd.concat([dates.rename("Date"), feats], axis=1)

        ref_window = 252
        step = max(1, int(self.horizon))
        drift_over_time = compute_drift_over_time(
            drift_df,
            date_col="Date",
            feature_cols=feature_cols,
            reference_window=ref_window,
            step=step,
            n_bins=10,
        )
        psi_by_feature_over_time = compute_psi_by_feature_over_time(
            drift_df,
            date_col="Date",
            feature_cols=feature_cols,
            reference_window=ref_window,
            step=step,
            n_bins=10,
        )

        # Persist detailed artifacts (do not influence training)
        artifact_paths: dict[str, str] = {}
        if self.run_manager is not None:
            try:
                p1 = self.run_manager.save_artifact(drift_over_time, "drift_over_time.csv", "diagnostic")
                artifact_paths["drift_over_time"] = str(p1)
            except Exception:
                pass
            try:
                p2 = self.run_manager.save_artifact(
                    psi_by_feature_over_time,
                    "psi_by_feature_over_time.csv",
                    "diagnostic",
                )
                artifact_paths["psi_by_feature_over_time"] = str(p2)
            except Exception:
                pass

        self._log(f"Drift snapshot: {report.n_features_drifted} features with PSI > {psi_threshold}")

        return {
            "psi_threshold": float(psi_threshold),
            "snapshot": report.to_dict(),
            "snapshot_alerts": alerts,
            "rolling": {
                "reference_window": int(ref_window),
                "step": int(step),
                "n_features": int(len(feature_cols)),
                "mean_psi_over_time": float(drift_over_time["mean_psi"].mean()) if not drift_over_time.empty else 0.0,
                "max_psi_over_time": float(drift_over_time["max_psi"].max()) if not drift_over_time.empty else 0.0,
            },
            "artifacts": artifact_paths,
        }
    
    def _compute_thresholds(
        self,
        gate_oof: pd.DataFrame,
        ranker_oof: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute optimal thresholds from OOF predictions.
        
        FIX #4: DIAGNOSTIC ONLY - global OOF threshold search is snooping.
        Per-fold thresholds (from run_outer_folds) should be used for decisions.
        This global threshold is provided for comparison only, never for deployment.
        """
        # Gate threshold: maximize F1
        best_f1 = 0.0
        best_p = 0.5
        
        if "y_gate" in gate_oof.columns:
            y_true = pd.to_numeric(gate_oof["y_gate"], errors="coerce").fillna(0).astype(int)
        else:
            y_true = (gate_oof["label_excess"] > self.data.cost_threshold).astype(int)
        
        for p in np.arange(0.3, 0.9, 0.05):
            y_pred = (gate_oof["pred_mean"] >= p).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_p = float(p)
        
        # Uncertainty threshold: 75th percentile of std
        u_star = float(gate_oof["pred_std"].quantile(0.75))
        
        return {
            "p_star": best_p,
            "u_star": u_star,
            "cost_threshold": float(self.data.cost_threshold),
            "best_gate_f1": best_f1,
        }
    
    def run(self) -> DualModelResult:
        """
        Run complete dual model training pipeline.
        
        Returns:
            DualModelResult with all predictions and metrics
        """
        start_time = datetime.now(timezone.utc)
        self._log(f"Starting dual model training at {start_time.isoformat()}")
        
        # 1. Load data
        self.data = self._load_data()
        
        # 2. Initialize RunManager
        self.run_manager = RunManager(base_dir=self.base_results_dir)
        cv_config = get_cv_config(self.horizon)
        
        run_dir = self.run_manager.create_run(
            horizon=self.horizon,
            seeds=self.seeds,
            cost_bps=float(self.data.cost_threshold) * 10_000.0,
            cv_config=cv_config,
            gate_params=self.gate_params,
            ranker_params=self.ranker_params,
            X=self.data.X,
            y_gate=self.data.y_gate,
            y_rank=self.data.y_rank,
            group_sizes=self.data.group_sizes,
            feature_selection_method=self.feature_selection,
            top_n_features=self.top_n_features,
            optuna_n_trials=self.optuna_n_trials,
        )
        
        # 3. Prepare data
        X = self.data.X.copy()
        y = self._build_y_dataframe(self.data)
        groups = self.data.full_df["Date"].to_numpy()

        # Optional drift mitigation: point-in-time rolling normalization (then cast for GPU)
        X = self._apply_point_in_time_rolling_zscore(X)

        # SOTA Fix #7: Cast only numeric columns to float32 (avoid crashing on Date/Sector)
        # The X from dataset_shaping should already be pure features, but this is defensive.
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            X[numeric_cols] = X[numeric_cols].astype(np.float32)
        else:
            # Fallback: all columns are already numeric (standard case from dataset_shaping)
            X = X.astype(np.float32)
        
        # 4. Train Gate Classifier
        gate_oof = self._train_gate(X, y, groups)
        
        # 5. Train Ranker
        ranker_oof = self._train_ranker(X, y, groups)
        
        # 6. Merge OOF predictions
        merged_oof = self._merge_oof(gate_oof, ranker_oof)
        
        # 7. Compute metrics
        gate_metrics, rank_metrics, rank_metrics_gated, regime_metrics = self._compute_metrics(merged_oof, gate_oof)
        
        # 8. Feature stability
        feature_stability = self._compute_feature_stability()
        
        # 9. Drift detection
        drift_report = self._compute_drift()
        
        # 10. Compute thresholds
        thresholds = self._compute_thresholds(gate_oof, ranker_oof)
        
        # 11. Build result
        result = DualModelResult(
            horizon=self.horizon,
            run_dir=run_dir,
            gate_oof=gate_oof,
            ranker_oof=ranker_oof,
            merged_oof=merged_oof,
            gate_metrics=gate_metrics,
            rank_metrics=rank_metrics,
            rank_metrics_gated=rank_metrics_gated,
            regime_metrics=regime_metrics,
            gate_fold_results=self._gate_fold_results,
            ranker_fold_results=self._ranker_fold_results,
            feature_stability=feature_stability,
            drift_report=drift_report,
            thresholds=thresholds,
            metadata={
                "start_time": start_time.isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        # 12. Save results
        result.save(self.run_manager)

        # 12b. Populate subfolders (best-effort)
        if self.run_manager is not None:
            if self.auto_generate_plots:
                try:
                    self.generate_plots(result)
                except Exception as e:
                    self._log(f"Plot generation skipped: {e}")

            if self.auto_write_printable:
                try:
                    self._write_printable(result)
                except Exception as e:
                    self._log(f"Printable summary skipped: {e}")

            if self.auto_run_backtest:
                try:
                    self._run_oof_backtest(result)
                except Exception as e:
                    self._log(f"Backtest skipped: {e}")

            if self.auto_save_models:
                try:
                    self._save_final_models(X, y, self.data.group_sizes)
                except Exception as e:
                    self._log(f"Model saving skipped: {e}")
        
        # 13. Print summary
        self._log("\n" + format_metrics_summary(gate_metrics, rank_metrics))
        
        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - start_time).total_seconds()
        self._log(f"Completed in {elapsed:.1f} seconds")
        
        return result
    
    def generate_plots(self, result: DualModelResult) -> None:
        """Generate all plots for the run."""
        from ModularMonolith.src.reporting.plots import (
            plot_drift_heatmap,
            plot_feature_stability,
            plot_ic_over_time,
            plot_precision_over_time,
        )
        
        if self.run_manager is None:
            return
        
        graphs_dir = self.run_manager.graphs_dir
        
        # IC over time - merged_oof has 'rank_mean' not 'pred_mean'
        ic_df = compute_ic_over_time(result.merged_oof, pred_col="rank_mean")
        if not ic_df.empty:
            plot_ic_over_time(
                ic_df,
                title=f"Rank IC Over Time (h={self.horizon})",
                save_path=graphs_dir / "ic_over_time.png",
            )
        
        # Precision over time - merged_oof has 'rank_mean' not 'pred_mean'
        prec_df = compute_precision_over_time(
            result.merged_oof,
            k=3,
            pred_col="rank_mean",
            cost_threshold=self.data.cost_threshold,
        )
        if not prec_df.empty:
            plot_precision_over_time(
                prec_df,
                k=3,
                save_path=graphs_dir / "precision_at_3_over_time.png",
            )
        
        # Feature stability
        if not result.feature_stability.empty:
            plot_feature_stability(
                result.feature_stability,
                save_path=graphs_dir / "feature_stability.png",
            )
        
        # Drift heatmap
        drift = result.drift_report or {}
        snap = drift
        if isinstance(drift, dict) and isinstance(drift.get("snapshot"), dict):
            snap = drift.get("snapshot") or {}

        drift_df = pd.DataFrame([
            {"feature": k, "psi": v}
            for k, v in (snap.get("psi_by_feature", {}) if isinstance(snap, dict) else {}).items()
        ])
        if not drift_df.empty:
            plot_drift_heatmap(
                drift_df,
                save_path=graphs_dir / "drift_psi_heatmap.png",
            )
        
        self._log(f"Plots saved to {graphs_dir}")


# =============================================================================
# Convenience Function
# =============================================================================

def train_dual_model(
    horizon: int,
    *,
    seeds: list[int] | None = None,
    optuna_n_trials: int = 25,
    use_gpu: bool = True,
    verbose: bool = True,
) -> DualModelResult:
    """
    Convenience function to train dual model.
    
    Example:
        result = train_dual_model(21, optuna_n_trials=0)
        print(result.rank_metrics.rank_ic)
    """
    trainer = DualModelTrainer(
        horizon=horizon,
        seeds=seeds,
        optuna_n_trials=optuna_n_trials,
        use_gpu=use_gpu,
        verbose=verbose,
    )
    
    result = trainer.run()
    trainer.generate_plots(result)
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Dual Model for Sector Rotation")
    parser.add_argument("--horizon", type=int, default=21, choices=[5, 21, 63])
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds for ensemble")
    
    args = parser.parse_args()
    
    result = train_dual_model(
        horizon=args.horizon,
        seeds=list(range(args.seeds)),
        optuna_n_trials=args.optuna_trials,
        use_gpu=not args.no_gpu,
    )
    
    print(f"\nResults saved to: {result.run_dir}")
