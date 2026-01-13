"""
Diagnostic utilities for ranking metrics validation and negative controls.

Implements:
A) Lift validation with base_rate checks
B) Gate target confirmation (prevalence analysis)
C) Hit-Rate vs Precision consistency checks  
D) Negative controls (random ranker, oracle ranker)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiftDiagnostics:
    """Diagnostic information for Lift@K calculation."""
    k: int
    lift: float
    mean_base_rate: float
    min_base_rate: float
    max_base_rate: float
    n_dates_valid: int
    n_dates_skipped_zero_mean: int
    n_dates_skipped_negative_mean: int
    precision_at_k: float
    
    def validate(self) -> list[str]:
        """Return list of validation errors (empty if all OK)."""
        errors: list[str] = []
        
        # Check 1: Lift must be >= -1 when base_rate > 0
        if self.mean_base_rate > 0 and self.lift < -1.0:
            errors.append(
                f"INVALID LIFT: Lift={self.lift:.4f} < -1 with base_rate={self.mean_base_rate:.6f}. "
                f"Formula: Lift = precision/base - 1 implies Lift >= -1 when base > 0."
            )
        
        # Check 2: Base rate should be reasonable (0.01 to 0.99)
        if self.mean_base_rate < 0.001 or self.mean_base_rate > 0.99:
            errors.append(
                f"SUSPICIOUS BASE_RATE: {self.mean_base_rate:.6f}. "
                f"Expected range [0.01, 0.99]. Check if cost_threshold or winner definition is correct."
            )
        
        # Check 3: Many skipped dates suggests problem
        total_dates = self.n_dates_valid + self.n_dates_skipped_zero_mean + self.n_dates_skipped_negative_mean
        if total_dates > 0:
            skip_rate = (self.n_dates_skipped_zero_mean + self.n_dates_skipped_negative_mean) / total_dates
            if skip_rate > 0.5:
                errors.append(
                    f"HIGH SKIP RATE: {skip_rate:.1%} dates skipped "
                    f"(zero_mean={self.n_dates_skipped_zero_mean}, neg_mean={self.n_dates_skipped_negative_mean}). "
                    f"This suggests label_excess has many zero/negative mean dates."
                )
        
        return errors
    
    def to_dict(self) -> dict[str, Any]:
        return {
            f"lift@{self.k}": self.lift,
            f"precision@{self.k}": self.precision_at_k,
            f"mean_base_rate@{self.k}": self.mean_base_rate,
            f"min_base_rate@{self.k}": self.min_base_rate,
            f"max_base_rate@{self.k}": self.max_base_rate,
            f"n_dates_valid@{self.k}": self.n_dates_valid,
            f"n_dates_skipped_zero@{self.k}": self.n_dates_skipped_zero_mean,
            f"n_dates_skipped_negative@{self.k}": self.n_dates_skipped_negative_mean,
        }


@dataclass(frozen=True)
class GatePrevalenceDiagnostics:
    """Diagnostic information for gate target y_gate."""
    overall_prevalence: float
    prevalence_by_date_mean: float
    prevalence_by_date_std: float
    prevalence_by_date_min: float
    prevalence_by_date_max: float
    n_dates: int
    n_total_samples: int
    n_positive_samples: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_prevalence_overall": self.overall_prevalence,
            "gate_prevalence_by_date_mean": self.prevalence_by_date_mean,
            "gate_prevalence_by_date_std": self.prevalence_by_date_std,
            "gate_prevalence_by_date_min": self.prevalence_by_date_min,
            "gate_prevalence_by_date_max": self.prevalence_by_date_max,
            "gate_n_dates": self.n_dates,
            "gate_n_total": self.n_total_samples,
            "gate_n_positive": self.n_positive_samples,
        }


def compute_lift_with_diagnostics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    k: int = 3,
    *,
    cost_threshold: float = 0.0,
) -> LiftDiagnostics:
    """
    Compute Classification Lift@K with comprehensive diagnostics (BINARY target).
    
    Uses prevalence-based lift: Lift = (Precision / BaseRate) - 1
    Returns detailed breakdown including base_rate per date.
    """
    df = pd.DataFrame({
        "y_cont": np.asarray(y_true).ravel(),
        "pred": np.asarray(y_pred).ravel(),
        "g": np.asarray(groups).ravel(),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.empty:
        return LiftDiagnostics(
            k=k,
            lift=0.0,
            mean_base_rate=0.0,
            min_base_rate=0.0,
            max_base_rate=0.0,
            n_dates_valid=0,
            n_dates_skipped_zero_mean=0,
            n_dates_skipped_negative_mean=0,
            precision_at_k=0.0,
        )
    
    # Binary target: 1 if winner, 0 otherwise
    df["y_bin"] = (df["y_cont"] > cost_threshold).astype(float)
    
    lifts: list[float] = []
    base_rates: list[float] = []
    precision_scores: list[float] = []
    n_skipped_zero = 0
    n_skipped_negative = 0  # Not used for binary target, but kept for compatibility
    
    for _, sub in df.groupby("g", sort=False):
        if len(sub) < k:
            continue
        
        base_rate = sub["y_bin"].mean()
        
        # Skip if base_rate is zero (no winners in this date)
        if base_rate == 0:
            n_skipped_zero += 1
            continue
        
        # Get top-K by prediction
        # IMPORTANT: align Lift@K direction with the global ranking convention.
        # The project-wide convention is "LOWER score = better".
        # If diagnostics use the wrong tail (nlargest), Lift/Precision will look inverted
        # even when RankIC and other metrics are correct.
        from ModularMonolith.src.reporting.metrics_rank import RANK_SCORE_LOWER_IS_BETTER

        if RANK_SCORE_LOWER_IS_BETTER:
            top_k_idx = sub["pred"].nsmallest(k).index
        else:
            top_k_idx = sub["pred"].nlargest(k).index
        precision_k = sub.loc[top_k_idx, "y_bin"].mean()
        
        # Classification Lift formula: (Precision / BaseRate) - 1
        lift = (precision_k / base_rate) - 1.0
        
        # Sanity check
        assert lift >= -1.0 or np.isclose(lift, -1.0, atol=1e-6), (
            f"Invalid lift={lift:.4f} with base_rate={base_rate:.4f}, precision={precision_k:.4f}"
        )
        
        lifts.append(float(lift))
        base_rates.append(float(base_rate))
        precision_scores.append(float(precision_k))
    
    if not lifts:
        return LiftDiagnostics(
            k=k,
            lift=0.0,
            mean_base_rate=0.0,
            min_base_rate=0.0,
            max_base_rate=0.0,
            n_dates_valid=0,
            n_dates_skipped_zero_mean=n_skipped_zero,
            n_dates_skipped_negative_mean=n_skipped_negative,
            precision_at_k=0.0,
        )
    
    return LiftDiagnostics(
        k=k,
        lift=float(np.mean(lifts)),
        mean_base_rate=float(np.mean(base_rates)),
        min_base_rate=float(np.min(base_rates)),
        max_base_rate=float(np.max(base_rates)),
        n_dates_valid=len(lifts),
        n_dates_skipped_zero_mean=n_skipped_zero,
        n_dates_skipped_negative_mean=n_skipped_negative,
        precision_at_k=float(np.mean(precision_scores)),
    )


def compute_gate_prevalence_diagnostics(
    y_gate: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
) -> GatePrevalenceDiagnostics:
    """
    Analyze gate target prevalence overall and by date.
    
    Args:
        y_gate: Binary gate target (0 or 1)
        groups: Date groups
    
    Returns:
        GatePrevalenceDiagnostics with prevalence statistics
    """
    df = pd.DataFrame({
        "y_gate": np.asarray(y_gate).ravel(),
        "g": np.asarray(groups).ravel(),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.empty:
        return GatePrevalenceDiagnostics(
            overall_prevalence=0.0,
            prevalence_by_date_mean=0.0,
            prevalence_by_date_std=0.0,
            prevalence_by_date_min=0.0,
            prevalence_by_date_max=0.0,
            n_dates=0,
            n_total_samples=0,
            n_positive_samples=0,
        )
    
    # Overall prevalence
    overall_prev = df["y_gate"].mean()
    n_total = len(df)
    n_positive = int(df["y_gate"].sum())
    
    # Per-date prevalence
    date_prev = df.groupby("g")["y_gate"].mean()
    
    return GatePrevalenceDiagnostics(
        overall_prevalence=float(overall_prev),
        prevalence_by_date_mean=float(date_prev.mean()),
        prevalence_by_date_std=float(date_prev.std()),
        prevalence_by_date_min=float(date_prev.min()),
        prevalence_by_date_max=float(date_prev.max()),
        n_dates=int(date_prev.count()),
        n_total_samples=n_total,
        n_positive_samples=n_positive,
    )


def random_ranker_control(
    y_true: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    *,
    seed: int = 42,
) -> dict[str, float]:
    """
    Negative control: Random ranker (random scores within each date).
    
    Expected: IC ≈ 0, Precision ≈ base_rate, Lift ≈ 0
    
    Returns:
        dict with rank_ic, precision@3, lift@3
    """
    from ModularMonolith.src.reporting.metrics_rank import compute_rank_metrics
    
    df = pd.DataFrame({
        "y": np.asarray(y_true).ravel(),
        "g": np.asarray(groups).ravel(),
    })
    df = df.dropna()
    
    np.random.seed(seed)
    
    # IMPORTANT:
    # The ranking metric implementation treats LOWER scores as better.
    # If we used a permutation of y as the score, the top-k selection would be
    # biased towards low y (because it picks nsmallest), producing an
    # artificially bad (non-zero) control.
    #
    # Instead generate random scores independent of y to simulate chance.
    random_pred = np.zeros(len(df), dtype=float)
    for _g_val, sub_df in df.groupby("g", sort=False):
        idx = sub_df.index.to_numpy()
        random_pred[idx] = np.random.standard_normal(size=len(idx))
    
    metrics = compute_rank_metrics(df["y"].to_numpy(), random_pred, df["g"].to_numpy())
    
    return {
        "random_rank_ic": metrics.rank_ic,
        "random_precision@3": metrics.precision_at_3,
        "random_lift@3": metrics.lift_at_3,
    }


def oracle_ranker_control(
    y_true: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Positive control: Oracle ranker.

    NOTE: The rank-metric implementation treats LOWER scores as better.
    So we use score = -y_true so that "best" items have the smallest scores.

    Expected: Precision@3 ≈ 1.0, Lift@3 near the maximum possible.
    
    Returns:
        dict with rank_ic, precision@3, lift@3
    """
    from ModularMonolith.src.reporting.metrics_rank import compute_rank_metrics
    
    y_arr = np.asarray(y_true).ravel().astype(float)
    # Score convention: lower is better.
    y_pred = -y_arr
    metrics = compute_rank_metrics(y_arr, y_pred, groups)
    
    return {
        # IC sign depends on score direction; report absolute value as a
        # magnitude-only positive control.
        "oracle_rank_ic": float(abs(metrics.rank_ic)),
        "oracle_rank_ic_signed": float(metrics.rank_ic),
        "oracle_precision@3": metrics.precision_at_3,
        "oracle_lift@3": metrics.lift_at_3,
    }


def gate_oracle_control(
    y_gate: np.ndarray | pd.Series,
    *,
    groups: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """
    Positive control for gate classifier: Perfect predictions.
    
    Expected: AUC = 1.0, F1 = 1.0, Accuracy = 1.0
    
    Returns:
        dict with auc, f1, accuracy
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    
    y_gate_arr = np.asarray(y_gate).ravel()
    y_gate_arr = y_gate_arr[~np.isnan(y_gate_arr)]
    
    if len(y_gate_arr) == 0:
        return {"oracle_auc": 0.0, "oracle_f1": 0.0, "oracle_accuracy": 0.0}
    
    # Perfect predictions (p = y)
    p_oracle = y_gate_arr.astype(float)
    y_int = y_gate_arr.astype(int)
    
    auc = roc_auc_score(y_int, p_oracle) if len(np.unique(y_int)) > 1 else 1.0
    f1 = f1_score(y_int, (p_oracle > 0.5).astype(int), zero_division=1.0)
    acc = accuracy_score(y_int, (p_oracle > 0.5).astype(int))
    
    return {
        "oracle_gate_auc": float(auc),
        "oracle_gate_f1": float(f1),
        "oracle_gate_accuracy": float(acc),
    }


def gate_polarity_check(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Check for gate polarity inversion: if AUC < 0.5, compute AUC(1-y, p).
    
    If AUC(1-y, p) > 0.5, labels are inverted relative to model interpretation.
    
    Returns:
        dict with auc_normal, auc_flipped, likely_inverted
    """
    from sklearn.metrics import roc_auc_score
    
    y_arr = np.asarray(y_true).ravel()
    p_arr = np.asarray(y_pred).ravel()
    
    # Remove NaN
    mask = ~(np.isnan(y_arr) | np.isnan(p_arr))
    y_arr = y_arr[mask]
    p_arr = p_arr[mask]
    
    if len(y_arr) == 0 or len(np.unique(y_arr)) < 2:
        return {"auc_normal": 0.5, "auc_flipped": 0.5, "likely_inverted": False}
    
    y_int = y_arr.astype(int)
    
    auc_normal = float(roc_auc_score(y_int, p_arr))
    auc_flipped = float(roc_auc_score(1 - y_int, p_arr))
    
    # If flipped AUC > 0.5 and normal < 0.5, polarity is inverted
    likely_inverted = (auc_normal < 0.5) and (auc_flipped > 0.5)
    
    return {
        "auc_normal": auc_normal,
        "auc_flipped": auc_flipped,
        "likely_inverted": bool(likely_inverted),
    }


def analyze_constant_predictions(
    merged_oof: pd.DataFrame,
    X_ranker: pd.DataFrame,
    cost_threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Analyze dates with near-constant predictions (pred std ≈ 0).
    
    For flagged dates, compute:
    - % rows with NaN in features
    - Count of zero-variance features
    - Unique row count (if 1, guaranteed constant pred)
    
    Returns:
        dict with analysis results
    """
    # Identify dates with constant predictions
    pred_std_by_date = merged_oof.groupby("Date")["rank_mean"].std()
    constant_dates = pred_std_by_date[pred_std_by_date <= 1e-12].index.tolist()
    
    if len(constant_dates) == 0:
        return {
            "n_constant_dates": 0,
            "pct_constant_dates": 0.0,
            "constant_dates_sample": [],
        }
    
    # ✅ Merge με keys (χωρίς index magic)
    # X_ranker έχει ήδη Date/Sector columns από τον caller
    merged_with_feats = merged_oof.merge(
        X_ranker,
        on=["Date", "Sector"],
        how="left"
    )
    
    # Feature columns = όλα εκτός από Date, Sector, και merged_oof columns
    feature_cols = [c for c in X_ranker.columns if c not in ["Date", "Sector"]]
    
    # Analyze each constant date
    analysis = []
    for date in constant_dates[:10]:  # Limit to first 10 for performance
        date_data = merged_with_feats[merged_with_feats["Date"] == date]
        
        # % NaN (μόνο στα features)
        nan_pct = date_data[feature_cols].isna().mean().mean()
        
        # Zero-variance features (numeric_only=True αγνοεί strings)
        feat_stds = date_data[feature_cols].std(numeric_only=True)
        n_zero_var = int((feat_stds <= 1e-12).sum())
        
        # Unique rows (μόνο features, όχι Date/Sector keys)
        n_unique_rows = date_data[feature_cols].drop_duplicates().shape[0]
        
        analysis.append({
            "date": str(date),
            "nan_pct": float(nan_pct),
            "n_zero_var_features": int(n_zero_var),
            "n_unique_rows": int(n_unique_rows),
            "n_total_rows": int(len(date_data)),
        })
    
    return {
        "n_constant_dates": len(constant_dates),
        "pct_constant_dates": float(len(constant_dates) / len(pred_std_by_date) * 100),
        "constant_dates_sample": analysis,
    }


def print_diagnostics_report(
    lift_diag_k1: LiftDiagnostics,
    lift_diag_k3: LiftDiagnostics,
    gate_diag: GatePrevalenceDiagnostics | None = None,
    random_control: dict[str, float] | None = None,
    oracle_control: dict[str, float] | None = None,
    gate_polarity: dict[str, Any] | None = None,
    constant_pred_analysis: dict[str, Any] | None = None,
) -> None:
    """Print comprehensive diagnostics report."""
    print("\n" + "=" * 80)
    print("RANKING METRICS DIAGNOSTICS REPORT")
    print("=" * 80)
    
    print("\n[A] LIFT@K VALIDATION")
    print("-" * 80)
    for diag in [lift_diag_k1, lift_diag_k3]:
        print(f"\nLift@{diag.k}:")
        print(f"  Lift value:        {diag.lift:+.4f}")
        print(f"  Precision@{diag.k}:      {diag.precision_at_k:.4f}")
        print(f"  Mean base rate:    {diag.mean_base_rate:.4f} (min={diag.min_base_rate:.4f}, max={diag.max_base_rate:.4f})")
        print(f"  Valid dates:       {diag.n_dates_valid}")
        print(f"  Skipped (zero):    {diag.n_dates_skipped_zero_mean}")
        print(f"  Skipped (negative): {diag.n_dates_skipped_negative_mean}")
        
        errors = diag.validate()
        if errors:
            print(f"\n  ⚠️  VALIDATION ERRORS:")
            for err in errors:
                print(f"      - {err}")
        else:
            print(f"  ✓ Validation: PASS")
    
    if gate_diag:
        print("\n[B] GATE TARGET PREVALENCE")
        print("-" * 80)
        print(f"  Overall prevalence:     {gate_diag.overall_prevalence:.4f}")
        print(f"  By-date prevalence:     {gate_diag.prevalence_by_date_mean:.4f} ± {gate_diag.prevalence_by_date_std:.4f}")
        print(f"  By-date range:          [{gate_diag.prevalence_by_date_min:.4f}, {gate_diag.prevalence_by_date_max:.4f}]")
        print(f"  Total samples:          {gate_diag.n_total_samples} ({gate_diag.n_positive_samples} positive)")
        print(f"  Number of dates:        {gate_diag.n_dates}")
    
    if gate_polarity:
        print("\n[B2] GATE POLARITY CHECK")
        print("-" * 80)
        print(f"  AUC (normal):       {gate_polarity.get('auc_normal', 0):.4f}")
        print(f"  AUC (flipped):      {gate_polarity.get('auc_flipped', 0):.4f}")
        if gate_polarity.get('likely_inverted', False):
            print(f"  ⚠️  LIKELY POLARITY INVERSION: AUC(1-y, p) = {gate_polarity['auc_flipped']:.4f} > 0.5")
            print(f"      Labels may be inverted relative to model interpretation.")
        else:
            print(f"  ✓ No polarity inversion detected")
    
    if constant_pred_analysis:
        print("\n[B3] CONSTANT PREDICTIONS ANALYSIS")
        print("-" * 80)
        n_const = constant_pred_analysis.get('n_constant_dates', 0)
        pct_const = constant_pred_analysis.get('pct_constant_dates', 0)
        print(f"  Dates with pred std ≤ 1e-12: {n_const} ({pct_const:.1f}%)")
        if n_const > 0:
            print(f"\n  Sample of constant-prediction dates:")
            for item in constant_pred_analysis.get('constant_dates_sample', [])[:5]:
                print(f"    Date {item['date']}:")
                print(f"      NaN %: {item['nan_pct']*100:.1f}%")
                print(f"      Zero-var features: {item['n_zero_var_features']}")
                print(f"      Unique rows: {item['n_unique_rows']}/{item['n_total_rows']}")
    
    if random_control:
        print("\n[C] RANDOM RANKER CONTROL (Negative)")
        print("-" * 80)
        print(f"  Random IC:          {random_control.get('random_rank_ic', 0):.4f} (expect ≈ 0)")
        print(f"  Random Precision@3: {random_control.get('random_precision@3', 0):.4f} (expect ≈ base_rate)")
        print(f"  Random Lift@3:      {random_control.get('random_lift@3', 0):+.4f} (expect ≈ 0)")
    
    if oracle_control:
        print("\n[D] ORACLE RANKER CONTROL (Positive)")
        print("-" * 80)
        print(f"  Oracle IC:          {oracle_control.get('oracle_rank_ic', 0):.4f} (expect = 1.0)")
        print(f"  Oracle Precision@3: {oracle_control.get('oracle_precision@3', 0):.4f} (expect = 1.0)")
        print(f"  Oracle Lift@3:      {oracle_control.get('oracle_lift@3', 0):+.4f} (expect > 0)")
    
    print("\n" + "=" * 80)
