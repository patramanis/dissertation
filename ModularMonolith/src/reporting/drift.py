"""
Drift detection and monitoring for feature distribution shifts.

Implements:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) test
- Per-feature and per-family aggregation
- Threshold alerts for production safety

Usage:
    detector = DriftDetector(reference_df=train_features)
    report = detector.compute_drift(current_df=test_features)
    alerts = detector.get_alerts(report, psi_threshold=0.25)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DriftReport:
    """Container for drift detection results."""
    psi_by_feature: dict[str, float]
    ks_by_feature: dict[str, tuple[float, float]]  # (statistic, p-value)
    psi_by_family: dict[str, float]
    mean_psi: float
    max_psi: float
    features_with_drift: list[str]
    n_features: int
    n_features_drifted: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "psi_by_feature": self.psi_by_feature,
            "ks_by_feature": {k: {"statistic": v[0], "pvalue": v[1]} for k, v in self.ks_by_feature.items()},
            "psi_by_family": self.psi_by_family,
            "mean_psi": self.mean_psi,
            "max_psi": self.max_psi,
            "features_with_drift": self.features_with_drift,
            "n_features": self.n_features,
            "n_features_drifted": self.n_features_drifted,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy viewing."""
        rows = []
        for feat, psi in self.psi_by_feature.items():
            ks_stat, ks_pval = self.ks_by_feature.get(feat, (np.nan, np.nan))
            rows.append({
                "feature": feat,
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "drifted": feat in self.features_with_drift,
            })
        return pd.DataFrame(rows).sort_values("psi", ascending=False)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-8,
    *,
    use_quantile_bins: bool = True,
) -> float:
    """
    Compute Population Stability Index (PSI).
    
    PSI interpretation:
    - < 0.1: No significant change
    - 0.1 - 0.25: Moderate change, monitor
    - > 0.25: Significant change, investigate
    
    Args:
        reference: Reference (training) distribution
        current: Current (test) distribution
        n_bins: Number of bins for discretization
        eps: Small value to avoid log(0)
        use_quantile_bins: If True, use quantile-based bins (more stable for
            skewed distributions). If False, use equal-width bins.
    
    Returns:
        PSI value
    """
    ref = np.asarray(reference).ravel()
    cur = np.asarray(current).ravel()
    
    # Remove NaNs and infs
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    
    if len(ref) < n_bins or len(cur) < n_bins:
        return 0.0
    
    # ✅ Winsorize extreme outliers to prevent PSI explosion
    ref_min, ref_max = np.percentile(ref, [1, 99])
    ref = np.clip(ref, ref_min, ref_max)
    cur = np.clip(cur, ref_min, ref_max)
    
    # Create bins from reference distribution
    if use_quantile_bins:
        # Quantile-based bins: more robust for skewed features
        # Each bin has ~equal number of reference observations
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(ref, quantiles)
        # Ensure unique bin edges (handle ties)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            # Fall back to histogram if too few unique quantiles
            _, bin_edges = np.histogram(ref, bins=n_bins)
    else:
        # Equal-width bins (original behavior)
        _, bin_edges = np.histogram(ref, bins=n_bins)
    
    # Ensure edges cover both distributions
    bin_edges = np.asarray(bin_edges, dtype=float).copy()
    bin_edges[0] = min(bin_edges[0], cur.min() - eps)
    bin_edges[-1] = max(bin_edges[-1], cur.max() + eps)
    
    # Compute histograms
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)
    
    actual_bins = len(ref_counts)
    
    # ✅ Convert to proportions with stronger epsilon smoothing
    # eps = 1e-8 is too small; use 1e-4 to avoid log(tiny) explosion
    eps_smooth = max(eps, 1e-4)
    ref_pct = (ref_counts + eps_smooth) / (len(ref) + eps_smooth * actual_bins)
    cur_pct = (cur_counts + eps_smooth) / (len(cur) + eps_smooth * actual_bins)
    
    # PSI formula: Σ(p_current - p_ref) * ln(p_current / p_ref)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    
    # ✅ Clip PSI to reasonable range (anything > 10 is likely calculation bug)
    psi = np.clip(psi, -10.0, 10.0)
    
    return float(psi)


def compute_ks_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov test statistic.
    
    Returns:
        (ks_statistic, p_value)
    """
    ref = np.asarray(reference).ravel()
    cur = np.asarray(current).ravel()
    
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    
    if len(ref) < 2 or len(cur) < 2:
        return 0.0, 1.0
    
    stat, pval = stats.ks_2samp(ref, cur)
    return float(stat), float(pval)


def get_feature_family(feature_name: str) -> str:
    """
    Extract feature family from feature name.
    
    Examples:
        'XLK_logret21' -> 'logret'
        'corr_vix_21' -> 'corr'
        'beta21' -> 'beta'
    """
    name = str(feature_name).lower()
    
    # Common feature families
    families = [
        "logret", "mom", "z_", "diff", "sum", "beta", "corr",
        "vol", "idiovol", "regime", "psi", "rsi",
        "macd", "ema", "sma", "atr", "vix", "spread",
    ]
    
    for fam in families:
        if fam in name:
            return fam
    
    # Default to first word or prefix
    for sep in ["_", "."]:
        if sep in name:
            return name.split(sep)[0]
    
    return "other"


class DriftDetector:
    """
    Detect feature drift between reference and current distributions.
    
    Usage:
        detector = DriftDetector(reference_df=train_X)
        report = detector.compute_drift(current_df=test_X)
        alerts = detector.get_alerts(report, psi_threshold=0.25)
    """
    
    def __init__(
        self,
        reference_df: pd.DataFrame,
        *,
        n_bins: int = 10,
        use_quantile_bins: bool = True,
    ) -> None:
        """
        Initialize with reference (training) data.
        
        Args:
            reference_df: Training features DataFrame
            n_bins: Number of bins for PSI computation
            use_quantile_bins: If True, use quantile-based bins (more stable
                for skewed feature distributions like volatility).
        """
        self.reference_df = reference_df.copy()
        self.n_bins = n_bins
        self.use_quantile_bins = use_quantile_bins
        self.feature_cols = list(reference_df.columns)
    
    def compute_drift(
        self,
        current_df: pd.DataFrame,
        *,
        psi_threshold: float = 0.25,
    ) -> DriftReport:
        """
        Compute drift metrics for all features.
        
        Args:
            current_df: Current (test) features DataFrame
            psi_threshold: PSI threshold for flagging drift
        
        Returns:
            DriftReport with all metrics
        """
        common_cols = [c for c in self.feature_cols if c in current_df.columns]
        
        psi_by_feature: dict[str, float] = {}
        ks_by_feature: dict[str, tuple[float, float]] = {}
        
        for col in common_cols:
            ref = self.reference_df[col].to_numpy()
            cur = current_df[col].to_numpy()
            
            psi_by_feature[col] = compute_psi(
                ref, cur,
                n_bins=self.n_bins,
                use_quantile_bins=self.use_quantile_bins,
            )
            ks_by_feature[col] = compute_ks_statistic(ref, cur)
        
        # Aggregate by family
        family_psis: dict[str, list[float]] = {}
        for col, psi in psi_by_feature.items():
            family = get_feature_family(col)
            family_psis.setdefault(family, []).append(psi)
        
        psi_by_family = {fam: float(np.mean(vals)) for fam, vals in family_psis.items()}
        
        # Summary stats
        psi_values = list(psi_by_feature.values())
        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
        max_psi = float(np.max(psi_values)) if psi_values else 0.0
        
        # Flag drifted features
        features_with_drift = [col for col, psi in psi_by_feature.items() if psi > psi_threshold]
        
        return DriftReport(
            psi_by_feature=psi_by_feature,
            ks_by_feature=ks_by_feature,
            psi_by_family=psi_by_family,
            mean_psi=mean_psi,
            max_psi=max_psi,
            features_with_drift=features_with_drift,
            n_features=len(common_cols),
            n_features_drifted=len(features_with_drift),
        )
    
    def get_alerts(
        self,
        report: DriftReport,
        *,
        psi_threshold: float = 0.25,
        ks_pvalue_threshold: float = 0.01,
    ) -> list[dict[str, Any]]:
        """
        Generate alerts for drifted features.
        
        Returns:
            List of alert dicts with feature, metric, value, severity
        """
        alerts: list[dict[str, Any]] = []
        
        for feat, psi in report.psi_by_feature.items():
            if psi > psi_threshold:
                severity = "critical" if psi > 0.5 else "warning"
                alerts.append({
                    "feature": feat,
                    "metric": "PSI",
                    "value": psi,
                    "threshold": psi_threshold,
                    "severity": severity,
                    "message": f"PSI={psi:.3f} exceeds threshold {psi_threshold}",
                })
        
        for feat, (ks_stat, ks_pval) in report.ks_by_feature.items():
            if ks_pval < ks_pvalue_threshold:
                alerts.append({
                    "feature": feat,
                    "metric": "KS",
                    "value": ks_stat,
                    "pvalue": ks_pval,
                    "threshold": ks_pvalue_threshold,
                    "severity": "warning",
                    "message": f"KS p-value={ks_pval:.4f} below threshold {ks_pvalue_threshold}",
                })
        
        return alerts


def compute_drift_over_time(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    feature_cols: list[str],
    reference_window: int = 252,
    step: int = 21,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling PSI over time for monitoring.
    
    Uses a sliding window approach:
    - Reference: previous `reference_window` days
    - Current: next `step` days
    
    Returns DataFrame with Date and mean_psi columns.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    dates = df[date_col].drop_duplicates().sort_values().tolist()
    
    results = []
    for i in range(reference_window, len(dates) - step, step):
        ref_dates = dates[i - reference_window:i]
        cur_dates = dates[i:i + step]
        
        ref_df = df[df[date_col].isin(ref_dates)][feature_cols]
        cur_df = df[df[date_col].isin(cur_dates)][feature_cols]
        
        if ref_df.empty or cur_df.empty:
            continue
        
        psis = []
        for col in feature_cols:
            if col in ref_df.columns and col in cur_df.columns:
                psi = compute_psi(
                    ref_df[col].to_numpy(),
                    cur_df[col].to_numpy(),
                    n_bins=n_bins,
                )
                psis.append(psi)
        
        if psis:
            results.append({
                "Date": cur_dates[0],
                "mean_psi": float(np.mean(psis)),
                "max_psi": float(np.max(psis)),
                "n_features": len(psis),
            })
    
    return pd.DataFrame(results)


def compute_psi_by_feature_over_time(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    feature_cols: list[str],
    reference_window: int = 252,
    step: int = 21,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling per-feature PSI over time for monitoring.

    This is a point-in-time safe diagnostic:
    - Bins are derived from the reference (past) window only.
    - The PSI for each step compares the reference window vs the next `step` window.

    Returns long DataFrame with columns:
        Date, feature, psi, ref_start, ref_end, cur_end
    """
    if reference_window <= 0:
        raise ValueError("reference_window must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    dates = df[date_col].drop_duplicates().sort_values().tolist()
    if len(dates) < (reference_window + step + 1):
        return pd.DataFrame(columns=["Date", "feature", "psi", "ref_start", "ref_end", "cur_end"])

    rows: list[dict[str, Any]] = []
    for i in range(reference_window, len(dates) - step, step):
        ref_dates = dates[i - reference_window : i]
        cur_dates = dates[i : i + step]

        ref_df = df[df[date_col].isin(ref_dates)][feature_cols]
        cur_df = df[df[date_col].isin(cur_dates)][feature_cols]

        if ref_df.empty or cur_df.empty:
            continue

        ref_start = ref_dates[0]
        ref_end = ref_dates[-1]
        cur_end = cur_dates[-1]

        for col in feature_cols:
            if col not in ref_df.columns or col not in cur_df.columns:
                continue
            psi = compute_psi(
                ref_df[col].to_numpy(),
                cur_df[col].to_numpy(),
                n_bins=n_bins,
            )
            rows.append(
                {
                    "Date": cur_dates[0],
                    "feature": col,
                    "psi": float(psi),
                    "ref_start": ref_start,
                    "ref_end": ref_end,
                    "cur_end": cur_end,
                }
            )

    return pd.DataFrame(rows)


def format_drift_summary(report: DriftReport) -> str:
    """Format drift report as printable summary."""
    lines = [
        "=" * 60,
        "DRIFT DETECTION SUMMARY",
        "=" * 60,
        "",
        f"Features analyzed: {report.n_features}",
        f"Features with drift (PSI > 0.25): {report.n_features_drifted}",
        f"Mean PSI: {report.mean_psi:.4f}",
        f"Max PSI: {report.max_psi:.4f}",
        "",
    ]
    
    if report.features_with_drift:
        lines.append("DRIFTED FEATURES:")
        for feat in report.features_with_drift[:10]:
            psi = report.psi_by_feature.get(feat, 0)
            lines.append(f"  {feat}: PSI={psi:.4f}")
        if len(report.features_with_drift) > 10:
            lines.append(f"  ... and {len(report.features_with_drift) - 10} more")
    
    lines.append("")
    lines.append("PSI BY FAMILY:")
    for family, psi in sorted(report.psi_by_family.items(), key=lambda x: -x[1]):
        status = "⚠️" if psi > 0.25 else "✓"
        lines.append(f"  {family}: {psi:.4f} {status}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
