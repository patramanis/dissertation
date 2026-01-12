"""
Plotting utilities for dual model training reports.

Provides:
- IC over time plots
- Precision@K over time
- Drift heatmaps (PSI/KS)
- SHAP summary plots
- Equity curves
- Feature stability charts

All functions return matplotlib Figure or save directly to file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _ensure_matplotlib():
    """Lazy import matplotlib to avoid startup overhead."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    return plt


def plot_ic_over_time(
    ic_df: pd.DataFrame,
    *,
    date_col: str = "Date",
    ic_col: str = "ic",
    rolling_col: str = "ic_rolling",
    title: str = "Rank IC Over Time",
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> Any:
    """
    Plot Rank IC over time with rolling average.
    
    Args:
        ic_df: DataFrame with Date, ic, ic_rolling columns
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure
    """
    plt = _ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dates = pd.to_datetime(ic_df[date_col])
    ic = ic_df[ic_col].to_numpy()
    ic_roll = ic_df[rolling_col].to_numpy()
    
    # Plot daily IC as scatter
    colors = np.where(ic >= 0, "green", "red")
    ax.scatter(dates, ic, c=colors, alpha=0.3, s=10, label="Daily IC")
    
    # Plot rolling average
    ax.plot(dates, ic_roll, color="blue", linewidth=2, label="Rolling IC (21d)")
    
    # Zero line
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    
    # Styling
    ax.set_xlabel("Date")
    ax.set_ylabel("Rank IC (Spearman)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # Add mean IC annotation
    mean_ic = float(np.nanmean(ic))
    ax.text(
        0.02, 0.98, f"Mean IC: {mean_ic:.4f}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_precision_over_time(
    prec_df: pd.DataFrame,
    *,
    date_col: str = "Date",
    prec_col: str = "precision",
    rolling_col: str = "precision_rolling",
    k: int = 3,
    title: str | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> Any:
    """
    Plot Precision@K over time.
    """
    plt = _ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dates = pd.to_datetime(prec_df[date_col])
    prec = prec_df[prec_col].to_numpy()
    prec_roll = prec_df[rolling_col].to_numpy()
    
    # Plot daily precision as bars
    colors = np.where(prec >= 0.5, "green", "orange")
    ax.bar(dates, prec, color=colors, alpha=0.4, width=1, label=f"Daily Precision@{k}")
    
    # Plot rolling average
    ax.plot(dates, prec_roll, color="blue", linewidth=2, label=f"Rolling Precision@{k}")
    
    # Random baseline
    baseline = float(k) / 9.0  # 9 sectors
    ax.axhline(baseline, color="red", linestyle="--", alpha=0.7, label=f"Random ({baseline:.2f})")
    
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Precision@{k}")
    ax.set_title(title or f"Precision@{k} Over Time")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_drift_heatmap(
    drift_df: pd.DataFrame,
    *,
    feature_col: str = "feature",
    psi_col: str = "psi",
    top_n: int = 30,
    title: str = "Feature Drift (PSI)",
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (10, 12),
) -> Any:
    """
    Plot PSI heatmap for top drifted features.
    """
    plt = _ensure_matplotlib()
    
    df = drift_df.nlargest(top_n, psi_col).copy()
    df = df.sort_values(psi_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    features = df[feature_col].tolist()
    psis = df[psi_col].to_numpy()
    
    # Color by severity
    colors = []
    for psi in psis:
        if psi > 0.5:
            colors.append("red")
        elif psi > 0.25:
            colors.append("orange")
        elif psi > 0.1:
            colors.append("yellow")
        else:
            colors.append("green")
    
    bars = ax.barh(features, psis, color=colors, alpha=0.7)
    
    # Add threshold lines
    ax.axvline(0.1, color="yellow", linestyle="--", alpha=0.7, label="Monitor (0.1)")
    ax.axvline(0.25, color="orange", linestyle="--", alpha=0.7, label="Warning (0.25)")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Critical (0.5)")
    
    ax.set_xlabel("PSI")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_feature_stability(
    stability_df: pd.DataFrame,
    *,
    feature_col: str = "feature",
    count_col: str = "selection_count",
    importance_col: str = "mean_importance",
    top_n: int = 25,
    title: str = "Feature Selection Stability",
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> Any:
    """
    Plot feature selection frequency and importance across folds.
    """
    plt = _ensure_matplotlib()
    
    df = stability_df.nlargest(top_n, count_col).copy()
    df = df.sort_values(count_col, ascending=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    features = df[feature_col].tolist()
    counts = df[count_col].to_numpy()
    importances = df[importance_col].to_numpy() if importance_col in df.columns else np.zeros_like(counts)
    
    # Selection count
    ax1.barh(features, counts, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Times Selected")
    ax1.set_title("Selection Frequency")
    ax1.grid(True, alpha=0.3, axis="x")
    
    # Mean importance
    ax2.barh(features, importances, color="coral", alpha=0.7)
    ax2.set_xlabel("Mean Importance")
    ax2.set_title("Mean Feature Importance")
    ax2.grid(True, alpha=0.3, axis="x")
    
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_equity_curves(
    equity_dict: dict[str, pd.Series],
    *,
    title: str = "Strategy Equity Curves",
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (14, 6),
) -> Any:
    """
    Plot multiple equity curves for comparison.
    
    Args:
        equity_dict: Dict mapping strategy name to equity Series
    """
    plt = _ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, equity in equity_dict.items():
        dates = equity.index
        values = equity.values
        ax.plot(dates, values, label=name, linewidth=1.5)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_regime_performance(
    metrics_by_regime: dict[Any, dict[str, float]],
    *,
    metric_key: str = "rank_ic",
    title: str = "Performance by Regime",
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> Any:
    """
    Plot performance metrics broken down by regime.
    """
    plt = _ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    regimes = list(metrics_by_regime.keys())
    values = [metrics_by_regime[r].get(metric_key, 0) for r in regimes]
    
    colors = ["green" if v > 0 else "red" for v in values]
    
    ax.bar(range(len(regimes)), values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([str(r) for r in regimes])
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    
    ax.set_xlabel("Regime")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    X: pd.DataFrame | np.ndarray,
    *,
    max_display: int = 20,
    title: str = "SHAP Feature Importance",
    save_path: Path | str | None = None,
) -> Any:
    """
    Plot SHAP summary plot for model explainability.
    
    Requires: shap library installed.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Skipping SHAP plot.")
        return None
    
    plt = _ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    return fig


def plot_gate_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    n_bins: int = 10,
    title: str = "Gate Model Calibration",
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> Any:
    """
    Plot calibration curve for gate classifier.
    """
    plt = _ensure_matplotlib()
    
    # Remove NaNs
    mask = np.isfinite(y_true) & np.isfinite(y_pred_proba)
    y = np.asarray(y_true)[mask]
    proba = np.asarray(y_pred_proba)[mask]
    
    # Compute calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    actual_probs = []
    predicted_probs = []
    counts = []
    
    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i + 1])
        if mask.sum() > 0:
            actual_probs.append(y[mask].mean())
            predicted_probs.append(proba[mask].mean())
            counts.append(mask.sum())
        else:
            actual_probs.append(np.nan)
            predicted_probs.append(bin_centers[i])
            counts.append(0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax1.scatter(predicted_probs, actual_probs, s=50, alpha=0.7)
    ax1.plot(predicted_probs, actual_probs, "b-", alpha=0.5)
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(title)
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.bar(bin_centers, counts, width=0.08, alpha=0.7, color="steelblue")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def create_summary_page(
    gate_metrics: dict[str, float],
    rank_metrics: dict[str, float],
    drift_summary: dict[str, Any],
    *,
    title: str = "Dual Model Summary",
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (12, 10),
) -> Any:
    """
    Create a summary page with key metrics in a grid layout.
    """
    plt = _ensure_matplotlib()
    
    fig = plt.figure(figsize=figsize)
    
    # Title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    # Create text block
    text_content = []
    text_content.append("=" * 50)
    text_content.append("GATE CLASSIFIER")
    text_content.append("=" * 50)
    for k, v in gate_metrics.items():
        text_content.append(f"{k:15s}: {v:.4f}")
    
    text_content.append("")
    text_content.append("=" * 50)
    text_content.append("RANKER")
    text_content.append("=" * 50)
    for k, v in rank_metrics.items():
        if isinstance(v, float):
            text_content.append(f"{k:15s}: {v:.4f}")
        else:
            text_content.append(f"{k:15s}: {v}")
    
    text_content.append("")
    text_content.append("=" * 50)
    text_content.append("DRIFT")
    text_content.append("=" * 50)
    for k, v in drift_summary.items():
        if isinstance(v, float):
            text_content.append(f"{k:15s}: {v:.4f}")
        elif isinstance(v, list):
            text_content.append(f"{k:15s}: {len(v)} items")
        else:
            text_content.append(f"{k:15s}: {v}")
    
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0.05, 0.95, "\n".join(text_content),
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig
