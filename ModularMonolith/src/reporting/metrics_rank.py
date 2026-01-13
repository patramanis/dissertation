"""
Ranking metrics for cross-sectional sector rotation strategies.

Implements:
- Rank IC (Spearman correlation) per fold and overall
- Precision@K, Hit-Rate@K, Lift@K
- Optional regime-conditional versions (by a provided regime label)

Based on Poh et al. (2021) ranking metrics methodology for
cross-sectional equity strategies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Ranking convention (CRITICAL)
# =============================================================================
# End-to-end convention for any "pick top-k" by predicted score:
#   - LOWER score = better (use nsmallest)
# This must match diagnostics and portfolio selection logic.
RANK_SCORE_LOWER_IS_BETTER: bool = True


def _k_eff(k: int, group_size: int) -> int:
    """Effective k for a group.

    Contract: k_eff = min(k, group_size) and we do NOT skip groups
    just because group_size < k (unless group_size == 0).
    """
    if group_size <= 0:
        return 0
    kk = int(k)
    if kk <= 0:
        raise ValueError("k must be positive")
    return min(kk, int(group_size))


def _pred_top_k_idx(sub: pd.DataFrame, *, pred_col: str, k: int) -> pd.Index:
    """Return indices of the predicted top-k within a group under the global convention."""
    k_eff = _k_eff(k, len(sub))
    if k_eff <= 0:
        return sub.index[:0]
    if RANK_SCORE_LOWER_IS_BETTER:
        return sub[pred_col].nsmallest(k_eff).index
    return sub[pred_col].nlargest(k_eff).index


@dataclass(frozen=True)
class RankMetrics:
    """Container for ranking metrics."""
    rank_ic: float
    rank_ic_std: float
    rank_ic_ir: float  # IC / std(IC) - Information Ratio
    hit_rate_top1: float
    hit_rate_top3: float
    precision_at_1: float
    precision_at_3: float
    lift_at_1: float  # Classification lift: (precision/base_rate) - 1, using binary target
    lift_at_3: float
    uplift_at_1: float  # Return uplift: mean(top-k) - mean(all), using continuous target
    uplift_at_3: float
    n_dates: int
    n_observations: int
    
    def to_dict(self) -> dict[str, float | int]:
        return {
            "rank_ic": self.rank_ic,
            "rank_ic_std": self.rank_ic_std,
            "rank_ic_ir": self.rank_ic_ir,
            "hit_rate_top1": self.hit_rate_top1,
            "hit_rate_top3": self.hit_rate_top3,
            "precision_at_1": self.precision_at_1,
            "precision_at_3": self.precision_at_3,
            "lift_at_1": self.lift_at_1,
            "lift_at_3": self.lift_at_3,
            "uplift_at_1": self.uplift_at_1,
            "uplift_at_3": self.uplift_at_3,
            "n_dates": self.n_dates,
            "n_observations": self.n_observations,
        }


def spearman_ic_by_group(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
) -> tuple[float, float, list[float]]:
    """
    Compute Spearman rank correlation (IC) per group and aggregate.
    
    Returns:
        (mean_ic, std_ic, per_group_ics)
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true).ravel(),
        "pred": np.asarray(y_pred).ravel(),
        "g": np.asarray(groups).ravel(),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.empty:
        return 0.0, 0.0, []
    
    ics: list[float] = []
    for _, sub in df.groupby("g", sort=False):
        if len(sub) < 2:
            continue
        y = sub["y"].to_numpy(dtype=float)
        p = sub["pred"].to_numpy(dtype=float)
        
        if np.nanstd(y) == 0 or np.nanstd(p) == 0:
            continue
        
        rho, _ = stats.spearmanr(y, p, nan_policy="omit")
        if np.isfinite(rho):
            ics.append(float(rho))
    
    if not ics:
        return 0.0, 0.0, []
    
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    
    return mean_ic, std_ic, ics


def precision_at_k(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    k: int = 3,
    *,
    threshold: float | None = None,
) -> float:
    """
    Precision@K: fraction of top-K predictions that are "winners".
    
    Winner is defined as:
    - If threshold is None: top-K by actual value
    - If threshold is given: actual value > threshold
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true).ravel(),
        "pred": np.asarray(y_pred).ravel(),
        "g": np.asarray(groups).ravel(),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.empty:
        return 0.0
    
    precisions: list[float] = []
    for _, sub in df.groupby("g", sort=False):
        k_eff = _k_eff(k, len(sub))
        if k_eff <= 0:
            continue
        
        top_k_idx = _pred_top_k_idx(sub, pred_col="pred", k=k_eff)
        top_k_actual = sub.loc[top_k_idx, "y"]
        
        if threshold is None:
            # Top-K by actual (baseline)
            actual_top_k = set(sub["y"].nlargest(k_eff).index)
            hits = sum(1 for idx in top_k_idx if idx in actual_top_k)
        else:
            # Winners by threshold
            hits = sum(1 for v in top_k_actual if v > threshold)
        
        precisions.append(float(hits) / float(k_eff))
    
    if not precisions:
        return 0.0
    
    return float(np.mean(precisions))


def hit_rate_at_k(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    k: int = 3,
) -> float:
    """
    Hit-Rate@K: fraction of dates where at least one of top-K predictions
    is in the actual top-K.
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true).ravel(),
        "pred": np.asarray(y_pred).ravel(),
        "g": np.asarray(groups).ravel(),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.empty:
        return 0.0
    
    hits = 0
    total = 0
    
    for _, sub in df.groupby("g", sort=False):
        k_eff = _k_eff(k, len(sub))
        if k_eff <= 0:
            continue

        pred_top_k = set(_pred_top_k_idx(sub, pred_col="pred", k=k_eff))
        actual_top_k = set(sub["y"].nlargest(k_eff).index)
        
        if pred_top_k & actual_top_k:  # Any overlap
            hits += 1
        total += 1
    
    return float(hits) / float(total) if total > 0 else 0.0


def lift_at_k(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    k: int = 3,
    cost_threshold: float = 0.0,
) -> float:
    """
    Classification Lift@K: prevalence-based lift using BINARY target.
    
    Lift = (Precision@K / BaseRate) - 1
    
    Where:
    - BaseRate = prevalence of winners (y > cost_threshold) in full universe
    - Precision@K = prevalence of winners in top-K selections
    - Lift must be >= -1 when BaseRate > 0 (mathematical constraint)
    
    This is distinct from return uplift (see uplift_at_k).
    """
    df = pd.DataFrame({
        "y_cont": np.asarray(y_true).ravel(),
        "pred": np.asarray(y_pred).ravel(),
        "g": np.asarray(groups).ravel(),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.empty:
        return 0.0
    
    # Binary target: 1 if winner (y > threshold), 0 otherwise
    df["y_bin"] = (df["y_cont"] > cost_threshold).astype(float)
    
    lifts: list[float] = []
    for _, sub in df.groupby("g", sort=False):
        k_eff = _k_eff(k, len(sub))
        if k_eff <= 0:
            continue
        
        base_rate = sub["y_bin"].mean()
        if base_rate == 0:  # No winners in this date
            continue
        
        top_k_idx = _pred_top_k_idx(sub, pred_col="pred", k=k_eff)
        precision_k = sub.loc[top_k_idx, "y_bin"].mean()
        
        lift = (precision_k / base_rate) - 1.0
        
        # Sanity check: Lift must be >= -1 when base_rate > 0
        assert lift >= -1.0 or np.isclose(lift, -1.0, atol=1e-6), (
            f"Invalid lift={lift:.4f} with base_rate={base_rate:.4f}, precision={precision_k:.4f}"
        )
        
        lifts.append(float(lift))
    
    if not lifts:
        return 0.0
    
    return float(np.mean(lifts))


def uplift_at_k(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    k: int = 3,
) -> float:
    """
    Return Uplift@K: mean excess return of top-K vs universe, using CONTINUOUS target.
    
    Uplift@K = mean(label_excess[top-K]) - mean(label_excess[all])
    
    Uses DIFFERENCE (not ratio) to avoid explosions when universe mean is near 0
    and sign flips when universe mean is negative.
    
    This is distinct from classification lift (see lift_at_k).
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true).ravel(),
        "pred": np.asarray(y_pred).ravel(),
        "g": np.asarray(groups).ravel(),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.empty:
        return 0.0
    
    uplifts: list[float] = []
    for _, sub in df.groupby("g", sort=False):
        k_eff = _k_eff(k, len(sub))
        if k_eff <= 0:
            continue
        
        mean_all = sub["y"].mean()
        top_k_idx = _pred_top_k_idx(sub, pred_col="pred", k=k_eff)
        mean_selected = sub.loc[top_k_idx, "y"].mean()
        
        # Difference, not ratio
        uplift = mean_selected - mean_all
        uplifts.append(float(uplift))
    
    if not uplifts:
        return 0.0
    
    return float(np.mean(uplifts))


def compute_rank_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    *,
    cost_threshold: float = 0.0,
) -> RankMetrics:
    """
    Compute comprehensive ranking metrics.
    
    Args:
        y_true: Actual excess returns (label_excess)
        y_pred: Predicted scores (rank mean from ensemble)
        groups: Date groups for cross-sectional evaluation
        cost_threshold: Threshold for "winner" definition
    
    Returns:
        RankMetrics with all computed values
    """
    mean_ic, std_ic, ics = spearman_ic_by_group(y_true, y_pred, groups)
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
    
    hr1 = hit_rate_at_k(y_true, y_pred, groups, k=1)
    hr3 = hit_rate_at_k(y_true, y_pred, groups, k=3)
    
    p1 = precision_at_k(y_true, y_pred, groups, k=1, threshold=cost_threshold)
    p3 = precision_at_k(y_true, y_pred, groups, k=3, threshold=cost_threshold)
    
    # Classification lift (binary target: winner prevalence)
    l1 = lift_at_k(y_true, y_pred, groups, k=1, cost_threshold=cost_threshold)
    l3 = lift_at_k(y_true, y_pred, groups, k=3, cost_threshold=cost_threshold)
    
    # Return uplift (continuous target: excess return difference)
    u1 = uplift_at_k(y_true, y_pred, groups, k=1)
    u3 = uplift_at_k(y_true, y_pred, groups, k=3)
    
    df = pd.DataFrame({"g": np.asarray(groups).ravel()})
    n_dates = df["g"].nunique()
    n_obs = len(df)
    
    return RankMetrics(
        rank_ic=mean_ic,
        rank_ic_std=std_ic,
        rank_ic_ir=ic_ir,
        hit_rate_top1=hr1,
        hit_rate_top3=hr3,
        precision_at_1=p1,
        precision_at_3=p3,
        lift_at_1=l1,
        lift_at_3=l3,
        uplift_at_1=u1,
        uplift_at_3=u3,
        n_dates=n_dates,
        n_observations=n_obs,
    )


def compute_rank_metrics_by_regime(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    regimes: np.ndarray | pd.Series,
    *,
    cost_threshold: float = 0.0,
) -> dict[Any, RankMetrics]:
    """
    Compute ranking metrics conditioned on regime states.
    
    Args:
        regimes: Regime label per observation
    
    Returns:
        Dict mapping regime -> RankMetrics
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true).ravel(),
        "pred": np.asarray(y_pred).ravel(),
        "g": np.asarray(groups).ravel(),
        "regime": np.asarray(regimes).ravel(),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    results: dict[Any, RankMetrics] = {}
    for regime, sub in df.groupby("regime", sort=True):
        if len(sub) < 10:  # Minimum observations for meaningful stats
            continue
        metrics = compute_rank_metrics(
            sub["y"].to_numpy(),
            sub["pred"].to_numpy(),
            sub["g"].to_numpy(),
            cost_threshold=cost_threshold,
        )
        results[regime] = metrics
    
    return results


def compute_ic_over_time(
    oof_df: pd.DataFrame,
    *,
    date_col: str = "Date",
    pred_col: str = "pred_mean",
    target_col: str = "label_excess",
    rolling_window: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling Rank IC over time for visualization.
    
    Returns DataFrame with columns: Date, ic, ic_rolling
    """
    df = oof_df[[date_col, pred_col, target_col]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    ics = []
    for dt, sub in df.groupby(date_col, sort=True):
        if len(sub) < 2:
            continue
        rho, _ = stats.spearmanr(sub[target_col], sub[pred_col], nan_policy="omit")
        if np.isfinite(rho):
            ics.append({"Date": dt, "ic": float(rho)})
    
    if not ics:
        return pd.DataFrame(columns=["Date", "ic", "ic_rolling"])
    
    ic_df = pd.DataFrame(ics)
    ic_df = ic_df.sort_values("Date").reset_index(drop=True)
    ic_df["ic_rolling"] = ic_df["ic"].rolling(window=rolling_window, min_periods=1).mean()
    
    return ic_df


def compute_precision_over_time(
    oof_df: pd.DataFrame,
    *,
    k: int = 3,
    date_col: str = "Date",
    pred_col: str = "pred_mean",
    target_col: str = "label_excess",
    cost_threshold: float = 0.0,
    rolling_window: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling Precision@K over time for visualization.
    """
    df = oof_df[[date_col, pred_col, target_col]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    precs = []
    for dt, sub in df.groupby(date_col, sort=True):
        k_eff = _k_eff(k, len(sub))
        if k_eff <= 0:
            continue

        top_k_idx = _pred_top_k_idx(sub, pred_col=pred_col, k=k_eff)
        hits = sum(1 for idx in top_k_idx if sub.loc[idx, target_col] > cost_threshold)
        precs.append({"Date": dt, "precision": float(hits) / float(k_eff)})
    
    if not precs:
        return pd.DataFrame(columns=["Date", "precision", "precision_rolling"])
    
    prec_df = pd.DataFrame(precs)
    prec_df = prec_df.sort_values("Date").reset_index(drop=True)
    prec_df["precision_rolling"] = prec_df["precision"].rolling(
        window=rolling_window, min_periods=1
    ).mean()
    
    return prec_df


def gate_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute classification metrics for gate model.
    
    Returns:
        Dict with accuracy, precision, recall, F1, AUC
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    
    y = np.asarray(y_true).ravel()
    proba = np.asarray(y_pred_proba).ravel()
    
    # Remove NaNs
    mask = np.isfinite(y) & np.isfinite(proba)
    y = y[mask]
    proba = proba[mask]
    
    if len(y) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}
    
    y_pred = (proba >= threshold).astype(int)
    y_int = (y >= 0.5).astype(int)
    
    try:
        auc = float(roc_auc_score(y_int, proba))
    except Exception:
        auc = 0.0
    
    return {
        "accuracy": float(accuracy_score(y_int, y_pred)),
        "precision": float(precision_score(y_int, y_pred, zero_division=0)),
        "recall": float(recall_score(y_int, y_pred, zero_division=0)),
        "f1": float(f1_score(y_int, y_pred, zero_division=0)),
        "auc": auc,
    }


def format_metrics_summary(
    gate_metrics: dict[str, float],
    rank_metrics: RankMetrics,
) -> str:
    """Format metrics as a printable summary."""
    lines = [
        "=" * 60,
        "DUAL MODEL METRICS SUMMARY",
        "=" * 60,
        "",
        "GATE CLASSIFIER:",
        f"  Accuracy:  {gate_metrics.get('accuracy', 0):.4f}",
        f"  Precision: {gate_metrics.get('precision', 0):.4f}",
        f"  Recall:    {gate_metrics.get('recall', 0):.4f}",
        f"  F1:        {gate_metrics.get('f1', 0):.4f}",
        f"  AUC:       {gate_metrics.get('auc', 0):.4f}",
        "",
        "RANKER:",
        f"  Rank IC:        {rank_metrics.rank_ic:.4f} ± {rank_metrics.rank_ic_std:.4f}",
        f"  IC IR:          {rank_metrics.rank_ic_ir:.4f}",
        f"  Hit-Rate@1:     {rank_metrics.hit_rate_top1:.4f}",
        f"  Hit-Rate@3:     {rank_metrics.hit_rate_top3:.4f}",
        f"  Precision@1:    {rank_metrics.precision_at_1:.4f}",
        f"  Precision@3:    {rank_metrics.precision_at_3:.4f}",
        "",
        "  Classification Lift (binary target, prevalence-based):",
        f"    Lift@1:       {rank_metrics.lift_at_1:+.4f}",
        f"    Lift@3:       {rank_metrics.lift_at_3:+.4f}",
        "",
        "  Return Uplift (continuous target, mean difference):",
        f"    Uplift@1:     {rank_metrics.uplift_at_1:+.6f}",
        f"    Uplift@3:     {rank_metrics.uplift_at_3:+.6f}",
        "",
        f"  N Dates:        {rank_metrics.n_dates}",
        f"  N Observations: {rank_metrics.n_observations}",
        "=" * 60,
    ]
    return "\n".join(lines)
