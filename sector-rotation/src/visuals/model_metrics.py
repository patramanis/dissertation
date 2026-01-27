from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

from .style import (
    apply_paper_style,
    get_figure_size,
    get_mode_color,
    get_horizon_color,
    format_percent_axis,
    format_date_axis,
)


log = logging.getLogger(__name__)

def plot_ic_time_series(
    ic_series: pd.Series | pd.DataFrame,
    title: str = "Information Coefficient Over Time",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    rolling_window: int = 21,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("ic_series")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(ic_series, pd.DataFrame):
        for col in ic_series.columns:
            ax.plot(ic_series.index, ic_series[col], 
                    label=col, linewidth=0.8, alpha=0.7)
            
            rolling = ic_series[col].rolling(rolling_window).mean()
            ax.plot(rolling.index, rolling, linewidth=1.5)
    else:
        ax.plot(ic_series.index, ic_series, color="#CCCCCC", 
                linewidth=0.5, alpha=0.5, label="Daily IC")
        
        rolling = ic_series.rolling(rolling_window).mean()
        ax.plot(rolling.index, rolling, color="#1f77b4", 
                linewidth=1.5, label=f"{rolling_window}d Rolling Mean")
    
    ax.axhline(y=0, color="#888888", linestyle="--", alpha=0.7)
    
    mean_ic = ic_series.mean() if isinstance(ic_series, pd.Series) else ic_series.mean().mean()
    ax.axhline(y=mean_ic, color="#2ca02c", linestyle=":", alpha=0.7)
    ax.text(
        ic_series.index[10], mean_ic,
        f"Mean IC: {mean_ic:.3f}",
        fontsize=9,
        color="#2ca02c",
    )
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Information Coefficient")
    ax.legend(loc="upper right")
    
    format_date_axis(ax)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig


def plot_ic_distribution(
    ic_series: pd.Series,
    title: str = "IC Distribution",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("single_golden")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(
        ic_series.dropna(),
        bins=50,
        density=True,
        alpha=0.7,
        color="#1f77b4",
        edgecolor="white",
    )
    
    mean_ic = ic_series.mean()
    median_ic = ic_series.median()
    
    ax.axvline(mean_ic, color="#d62728", linestyle="-", linewidth=1.5,
               label=f"Mean: {mean_ic:.3f}")
    ax.axvline(median_ic, color="#2ca02c", linestyle="--", linewidth=1.5,
               label=f"Median: {median_ic:.3f}")
    ax.axvline(0, color="#888888", linestyle=":", linewidth=1)
    std_ic = ic_series.std()
    ir = mean_ic / std_ic if std_ic > 0 else 0
    
    ax.text(
        0.95, 0.95,
        f"IC IR: {ir:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    ax.set_title(title)
    ax.set_xlabel("Information Coefficient")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_ic_by_horizon(
    ic_by_horizon: dict[int, pd.Series],
    title: str = "IC by Prediction Horizon",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("single_golden")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    horizons = sorted(ic_by_horizon.keys())
    
    stats = []
    for h in horizons:
        ic = ic_by_horizon[h]
        stats.append({
            "Horizon": h,
            "Mean IC": ic.mean(),
            "Std IC": ic.std(),
            "IC IR": ic.mean() / ic.std() if ic.std() > 0 else 0,
        })
    
    stats_df = pd.DataFrame(stats)
    colors = [get_horizon_color(h) for h in horizons]
    
    ax.bar(
        range(len(horizons)),
        stats_df["Mean IC"],
        yerr=stats_df["Std IC"],
        capsize=5,
        color=colors,
        edgecolor="white",
        alpha=0.8,
    )
    
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"H={h}" for h in horizons])
    ax.axhline(y=0, color="#888888", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Prediction Horizon (days)")
    ax.set_ylabel("Mean IC")
    
    for idx, (i, row) in enumerate(stats_df.iterrows()):
        ax.text(
            float(idx), row["Mean IC"] + row["Std IC"] + 0.01,
            f"IR={row['IC IR']:.2f}",
            ha="center",
            fontsize=8,
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_sharpe_comparison(
    sharpe_values: dict[str, float],
    title: str = "Sharpe Ratio Comparison",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    sort: bool = True,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("double_wide")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sharpe = pd.Series(sharpe_values)
    
    if sort:
        sharpe = sharpe.sort_values(ascending=True)
    
    colors = [get_mode_color(s.split("_h")[0]) for s in sharpe.index]
    bars = ax.barh(range(len(sharpe)), np.array(sharpe.values, dtype=float), color=colors)
    ax.set_yticks(range(len(sharpe)))
    ax.set_yticklabels(sharpe.index)
    ax.axvline(x=0, color="#888888", linestyle="--", alpha=0.5)
    ax.axvline(x=1, color="#2ca02c", linestyle=":", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Sharpe Ratio")
    
    for i, (bar, val) in enumerate(zip(bars, sharpe.values)):
        ax.text(
            val + 0.05 if val >= 0 else val - 0.1,
            i,
            f"{val:.2f}",
            va="center",
            fontsize=8,
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig


def plot_metrics_heatmap(
    metrics: pd.DataFrame,
    title: str = "Strategy Metrics Comparison",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("heatmap")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    normalized = (metrics - metrics.min()) / (metrics.max() - metrics.min())
    
    inverse_cols = ["volatility", "max_drawdown", "max_dd"]
    for col in metrics.columns:
        if any(inv in col.lower() for inv in inverse_cols):
            normalized[col] = 1 - normalized[col]
    
    sns.heatmap(
        normalized,
        annot=metrics,
        fmt=".2f",
        cmap="RdYlGn",
        ax=ax,
        cbar_kws={"label": "Relative Performance"},
    )
    
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Strategy")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_rmse_comparison(
    rmse_values: dict[str, float],
    title: str = "RMSE Comparison",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("double_wide")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    rmse = pd.Series(rmse_values).sort_values(ascending=True)
    
    colors = [get_mode_color(s.split("_h")[0]) for s in rmse.index]
    
    ax.barh(range(len(rmse)), np.array(rmse.values, dtype=float), color=colors)
    
    ax.set_yticks(range(len(rmse)))
    ax.set_yticklabels(rmse.index)
    
    ax.set_title(title)
    ax.set_xlabel("RMSE")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig


def plot_prediction_vs_actual(
    predictions: pd.Series,
    actuals: pd.Series,
    title: str = "Predicted vs Actual Returns",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("single_square")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    common = predictions.index.intersection(actuals.index)
    pred = predictions.loc[common]
    actual = actuals.loc[common]
    
    ax.scatter(pred, actual, alpha=0.3, s=5, color="#1f77b4")
    
    lims = [
        min(pred.min(), actual.min()),
        max(pred.max(), actual.max()),
    ]
    ax.plot(lims, lims, color="#d62728", linestyle="--", linewidth=1)
    
    from scipy.stats import linregress
    reg_result = linregress(pred.values, actual.values)
    slope_: float = float(reg_result.slope)  # type: ignore[attr-defined]
    intercept_: float = float(reg_result.intercept)  # type: ignore[attr-defined]
    r_value_: float = float(reg_result.rvalue)  # type: ignore[attr-defined]
    
    x_line = np.linspace(float(pred.min()), float(pred.max()), 100)
    y_line = slope_ * x_line + intercept_
    ax.plot(x_line, y_line, color="#2ca02c", linestyle="-", linewidth=1,
            label=f"R²={r_value_**2:.3f}")
    
    ax.set_title(title)
    ax.set_xlabel("Predicted Return")
    ax.set_ylabel("Actual Return")
    ax.legend(loc="upper left")
    
    ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_feature_importance(
    importance: pd.Series | dict[str, float],
    title: str = "Feature Importance",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    top_n: int = 20,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = (get_figure_size("single_golden")[0], 
                   min(10, top_n * 0.3 + 1))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(importance, dict):
        importance = pd.Series(importance)
    
    top = importance.nlargest(top_n).sort_values(ascending=True)
    
    ax.barh(range(len(top)), np.array(top.values, dtype=float), color="#1f77b4")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_quantile_returns(
    quantile_returns: pd.DataFrame,
    title: str = "Returns by Factor Quantile",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("single_golden")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    quantile_returns.plot(kind="bar", ax=ax)
    
    ax.axhline(y=0, color="#888888", linestyle="--", alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel("Factor Quantile")
    ax.set_ylabel("Mean Return")
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig