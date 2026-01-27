from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns

from .style import (
    apply_paper_style,
    get_figure_size,
    get_sector_color,
    SECTOR_COLORS,
    format_percent_axis,
    format_date_axis,
    add_recession_shading,
)


log = logging.getLogger(__name__)

def plot_sector_returns(
    prices: pd.DataFrame,
    title: str = "Sector ETF Cumulative Returns",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    normalize: bool = True,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("equity_curve")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        prices = prices / prices.iloc[0]
    
    for col in prices.columns:
        color = get_sector_color(col)
        ax.plot(prices.index, prices[col], label=col, color=color, linewidth=1.0)
    
    add_recession_shading(ax)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    
    format_date_axis(ax)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig


def plot_sector_correlations(
    returns: pd.DataFrame,
    title: str = "Sector Return Correlations",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("single_square")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    corr = returns.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig


def plot_sector_rolling_correlations(
    returns: pd.DataFrame,
    base_sector: str = "XLK",
    window: int = 63,
    title: str | None = None,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("ic_series")
    
    if title is None:
        title = f"Rolling {window}d Correlation with {base_sector}"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    base = returns[base_sector]
    
    for col in returns.columns:
        if col == base_sector:
            continue
        
        color = get_sector_color(col)
        rolling_corr = base.rolling(window).corr(returns[col])
        
        ax.plot(rolling_corr.index, rolling_corr, label=col, 
                color=color, linewidth=0.8, alpha=0.8)
    
    ax.axhline(y=0, color="#888888", linestyle="--", alpha=0.5)
    ax.axhline(y=0.5, color="#888888", linestyle=":", alpha=0.3)
    ax.axhline(y=-0.5, color="#888888", linestyle=":", alpha=0.3)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_ylim(-1, 1)
    
    format_date_axis(ax)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_feature_correlation_matrix(
    features: pd.DataFrame,
    title: str = "Feature Correlations",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    cluster: bool = True,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("heatmap")
    
    corr = features.corr()
    
    if cluster:
        g = sns.clustermap(
            corr,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            figsize=figsize,
            dendrogram_ratio=0.1,
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
        )
        g.fig.suptitle(title, y=1.02)
        fig = g.fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            corr,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            xticklabels=True,
            yticklabels=True,
        )
        
        ax.set_title(title)
        plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig


def plot_feature_distributions(
    features: pd.DataFrame,
    top_n: int = 12,
    title: str = "Feature Distributions",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = (get_figure_size("double_golden")[0], 8)
    
    variances = features.var().nlargest(top_n)
    selected = features[variances.index]
    
    n_rows = (top_n + 3) // 4
    n_cols = min(4, top_n)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(selected.columns):
        if i >= len(axes):
            break
        
        axes[i].hist(
            selected[col].dropna(),
            bins=30,
            density=True,
            alpha=0.7,
            color="#1f77b4",
            edgecolor="white",
        )
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=7)
    
    for i in range(len(selected.columns), len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_feature_time_series(
    features: pd.DataFrame,
    feature_names: list[str],
    title: str = "Feature Time Series",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    n_features = len(feature_names)
    
    if figsize is None:
        figsize = (get_figure_size("equity_curve")[0], 2 * n_features)
    
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_features))
    
    for i, (feature, ax) in enumerate(zip(feature_names, axes)):
        if feature not in features.columns:
            continue
        
        ax.plot(features.index, features[feature], color=colors[i], linewidth=0.8)
        ax.set_ylabel(feature, fontsize=8)
        ax.axhline(y=0, color="#888888", linestyle="--", alpha=0.3)
        mean_val = features[feature].mean()
        ax.axhline(y=mean_val, color="#2ca02c", linestyle=":", alpha=0.5)
    
    axes[-1].set_xlabel("Date")
    fig.suptitle(title)
    
    format_date_axis(axes[-1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_missing_data(
    data: pd.DataFrame,
    title: str = "Missing Data Pattern",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("heatmap")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    missing = data.isnull().sum() / len(data) * 100
    missing = missing[missing > 0].sort_values(ascending=True)
    
    if len(missing) == 0:
        ax.text(0.5, 0.5, "No missing data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
    else:
        ax.barh(range(len(missing)), np.array(missing.values, dtype=float), color="#d62728")
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index)
        ax.set_xlabel("Missing (%)")
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_label_distribution(
    labels: pd.Series,
    by_sector: bool = False,
    sector_col: str = "Sector",
    title: str = "Label Distribution",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("double_golden")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if by_sector and isinstance(labels, pd.DataFrame) and sector_col in labels.columns:
        sectors = sorted(labels[sector_col].unique())
        
        data = [labels[labels[sector_col] == s].drop(columns=[sector_col]).values.flatten()
                for s in sectors]
        
        colors = [get_sector_color(s) for s in sectors]
        
        bp = ax.boxplot(data, patch_artist=True)
        
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticklabels(sectors, rotation=45)
    else:
        if hasattr(labels, "to_numpy"):
            label_vals = labels.to_numpy().flatten()
        elif hasattr(labels, "values"):
            label_vals = np.array(labels.values).flatten()
        else:
            label_vals = np.array(labels).flatten()
        
        ax.hist(
            label_vals,
            bins=50,
            density=True,
            alpha=0.7,
            color="#1f77b4",
            edgecolor="white",
        )
        
        mean_val = float(np.nanmean(labels))
        ax.axvline(mean_val, color="#d62728", linestyle="-", linewidth=1.5,
                   label=f"Mean: {mean_val:.4f}")
        ax.axvline(0, color="#888888", linestyle="--", alpha=0.7)
        
        ax.legend(loc="upper right")
    
    ax.set_title(title)
    ax.set_xlabel("Forward Return")
    ax.set_ylabel("Density" if not by_sector else "Return")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_summary_statistics(
    data: pd.DataFrame,
    title: str = "Summary Statistics",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("double_wide")
    
    stats = data.describe().T
    stats["skew"] = data.skew()
    stats["kurtosis"] = data.kurtosis()
    stats = stats.round(4)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(
        cellText=[[str(v) for v in row] for row in stats.values],
        colLabels=list(stats.columns),
        rowLabels=list(stats.index),
        cellLoc="center",
        loc="center",
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    ax.set_title(title, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig