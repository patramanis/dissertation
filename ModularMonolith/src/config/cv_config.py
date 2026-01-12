from __future__ import annotations

import os
from typing import Any


def get_cv_config(horizon: int) -> dict[str, Any]:
    """
    SOTA CV Configuration for Expanding Window Walk-Forward Cross-Validation.
    
    CRITICAL CONTRACT (must match data pipeline):
    - Features[T]: Use information through Close(T-1) (raw2 shifted semantics)
    - Labels[T]: Return from Close(T-1) to Close(T-1+h) (aligned with features)
    - Horizon unit: Trading days (not calendar days)
    
    Purge/Embargo Rules:
    1. purge_gap = h: Exclude train dates t where label window [T-1, T-1+h] overlaps test
    2. embargo: Configurable (default conservative h//4) for OOF tuning safety
       - Set embargo=0 ONLY if:
         * Using pure expanding window (no rolling)
         * No threshold tuning or feature selection on OOF predictions
         * No calibration procedures that see test indirectly
    3. test_size: Scales with horizon to maintain ~4 non-overlapping label periods
    4. min_train_size: 252 trading days (1 year minimum)
    
    Time Decay Weighting (configured in NestedCVConfig):
    - enable_time_decay: bool = True (apply exponential decay to sample weights)
    - half_life_days: int = 1260 (5 years; sample weight = 50% after 5 years)
    - gate_class_balance: bool = True (combine time decay with class balance for Gate)
    
    Formula: W_t = 2^(-(T-t)/λ) where T = last training date, λ = half_life_days
    
    WARNING: Remainder dates are DROPPED (not included in last fold) for metric consistency.
    If you need recent regime data, reduce test_size or increase step_size overlap.
    """
    h = int(horizon)
    
    # ===== TEMPORAL CONTRACT (must match pipeline) =====
    contract = {
        "label_contract": f"Return(Close(T-1) -> Close(T-1+{h}))",
        "feature_asof": "Close(T-1)",
        "horizon_unit": "trading_days",
        "horizon_value": h,
    }
    
    # ===== PURGE GAP (based on label overlap) =====
    # Train date t is purged if its label window [t-1, t-1+h] overlaps test start S:
    # Condition: t-1+h >= S  =>  t >= S-h+1
    # Therefore: purge last (h) train dates before test start
    # FIX: Was h+1 (overly conservative without justification)
    purge_gap = h
    
    # ===== EMBARGO (must be 0 for expanding-only CV) =====
    # FIX: purged_walk_forward_cv enforces expanding-only, so embargo MUST be 0.
    # Expanding window means train is ALWAYS chronologically before test,
    # so embargo (which excludes dates immediately after test from train) has no effect.
    # The purge_gap already handles label overlap prevention.
    embargo = 0
    
    # ===== TEST SIZE (scales with horizon for ~4 independent periods) =====
    # Goal: Each test fold has ~4 non-overlapping label windows
    # Formula: test_size = 4*h, capped at [50, 252]
    # FIX: Was fixed 50/105 (arbitrary, not scaled to effective sample size)
    test_size_raw = 4 * h
    test_size = max(50, min(test_size_raw, 252))  # Clamp to [50, 252]
    
    # ===== STEP SIZE (how far test window moves each fold) =====
    # Default: step_size = test_size (non-overlapping test windows)
    # For more folds with overlapping tests: set step_size < test_size (requires larger embargo)
    step_size = test_size
    
    # ===== MIN TRAIN SIZE =====
    # Minimum 1 year of training data (252 trading days)
    min_train_size = 252
    
    # ===== MAX TRAIN SIZE (None = expanding) =====
    # None: Expanding window (train grows indefinitely)
    # N: Rolling window (train capped at N days)
    max_train_size = None
    
    # ===== TEST START (fixed for scientific reproducibility) =====
    # For dissertation/research: Use FIXED test_start date for reproducibility
    # This ensures backtest period is deterministic and documented
    # Warmup period (252 + h + purge_gap days) should end before test_start
    warmup_days = 252 + h + purge_gap
    test_start = "2016-01-01"  # Fixed date where OOS testing begins
    
    # n_splits computed automatically by splitter based on:
    # (total_dates - test_start_index) / test_size
    # This gives many folds (~3 per year with test_size=84 for h=21)
    n_splits = None  # Auto-computed from test_start and test_size

    # ===== OPTIONAL OVERRIDES (for quick runs / debugging) =====
    # These keep defaults unchanged unless explicitly set.
    # Examples (PowerShell):
    #   $env:MM_CV_N_SPLITS="3"; $env:MM_CV_TEST_START="2022-01-01"
    #   $env:MM_CV_TEST_SIZE="84"; $env:MM_CV_STEP_SIZE="84"
    v = str(os.environ.get("MM_CV_TEST_START", "")).strip()
    if v:
        test_start = v

    v = str(os.environ.get("MM_CV_N_SPLITS", "")).strip()
    if v:
        try:
            n_splits = int(v)
        except Exception as e:
            raise ValueError(f"Invalid MM_CV_N_SPLITS={v!r}") from e

    v = str(os.environ.get("MM_CV_TEST_SIZE", "")).strip()
    if v:
        try:
            test_size = int(v)
        except Exception as e:
            raise ValueError(f"Invalid MM_CV_TEST_SIZE={v!r}") from e

    v = str(os.environ.get("MM_CV_STEP_SIZE", "")).strip()
    if v:
        try:
            step_size = int(v)
        except Exception as e:
            raise ValueError(f"Invalid MM_CV_STEP_SIZE={v!r}") from e
    
    return {
        # Core CV parameters
        "n_splits": n_splits,
        "test_size": test_size,
        "step_size": step_size,
        "purge_gap": purge_gap,
        "embargo": embargo,
        "min_train_size": min_train_size,
        "max_train_size": max_train_size,
        
        # Derived/metadata (for splitter and validation)
        "warmup_days": warmup_days,
        "test_start": test_start,  # Fixed date for OOS testing (scientific reproducibility)
        "contract": contract,
        
        # Remainder handling
        "include_remainder_in_last_fold": False,  # SOTA: False for consistency
    }
