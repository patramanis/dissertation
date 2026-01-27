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
    REGIME_COLORS,
    format_percent_axis,
    format_date_axis,
    add_regime_zones,
)


log = logging.getLogger(__name__)

VIX_ZONES = {
    "low_vol": (0, 12),
    "normal": (12, 20),
    "high_vol": (20, 30),
    "crisis": (30, float("inf")),
}
def classify_vix_regime(vix: float) -> str:
    for regime, (low, high) in VIX_ZONES.items():
        if low <= vix < high:
            return regime
    return "crisis"
def build_regime_series(vix_series: pd.Series) -> pd.Series:
    return vix_series.apply(classify_vix_regime)
def plot_vix_with_zones(
    vix_series: pd.Series,
    equity_curve: pd.Series | None = None,
    title: str = "VIX with Regime Zones",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    if figsize is None:
        figsize = get_figure_size("regime_zones")
    
    if equity_curve is not None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
        ax_vix = axes[0]
        ax_equity = axes[1]
    else:
        fig, ax_vix = plt.subplots(figsize=figsize)
        ax_equity = None
    
    for regime, (low, high) in VIX_ZONES.items():
        color = REGIME_COLORS.get(regime, "#CCCCCC")
        
        if high == float("inf"):
            high = vix_series.max() + 5
        
        ax_vix.axhspan(low, high, color=color, alpha=0.3, zorder=0)
        
        mid = (low + min(high, 50)) / 2
        ax_vix.text(
            vix_series.index[10], mid,
            regime.replace("_", " ").title(),
            fontsize=8,
            color="#333333",
            alpha=0.7,
        )
    
    ax_vix.plot(vix_series.index, vix_series, color="#1f77b4", linewidth=1.0)
    ax_vix.set_ylabel("VIX")
    ax_vix.set_title(title)
    
    for threshold in [12, 20, 30]:
        ax_vix.axhline(threshold, color="#333333", linestyle=":", 
                       linewidth=0.5, alpha=0.5)
    
    if ax_equity is not None and equity_curve is not None:
        equity = equity_curve / equity_curve.iloc[0]
        ax_equity.plot(equity.index, equity, color="#2ca02c", linewidth=1.2)
        ax_equity.set_ylabel("Portfolio Value")
        ax_equity.set_xlabel("Date")
        
        regime_series = build_regime_series(vix_series)
        add_regime_zones(ax_equity, regime_series, alpha=0.15)
        
        format_date_axis(ax_equity)
    else:
        format_date_axis(ax_vix)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
        log.info("Saved VIX zones plot to %s", output_path)
    
    return fig

def plot_regime_timeline(
    regime_series: pd.Series,
    title: str = "Regime Timeline",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = (get_figure_size("equity_curve")[0], 2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    regime_map = {"low_vol": 0, "normal": 1, "high_vol": 2, "crisis": 3}
    numeric = regime_series.map(regime_map).fillna(1)
    
    colors = [REGIME_COLORS.get(r, "#CCCCCC") for r in regime_series]
    
    ax.scatter(
        regime_series.index,
        [1] * len(regime_series),
        c=colors,
        s=5,
        marker="|",
    )
    
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("Date")
    
    format_date_axis(ax)
    
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=REGIME_COLORS[r], label=r.replace("_", " ").title())
        for r in ["low_vol", "normal", "high_vol", "crisis"]
    ]
    ax.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_performance_by_regime(
    returns: pd.Series | pd.DataFrame,
    regime_series: pd.Series,
    metric: str = "sharpe",
    title: str = "Performance by Regime",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("single_golden")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(returns, pd.Series):
        returns = returns.to_frame("Strategy")
    
    common = returns.index.intersection(regime_series.index)
    returns = returns.loc[common]
    regimes = regime_series.loc[common]
    
    regimes_order = ["low_vol", "normal", "high_vol", "crisis"]
    
    data = []
    
    for regime in regimes_order:
        mask = regimes == regime
        
        if mask.sum() < 10:
            continue
        
        regime_returns = returns[mask]
        
        for col in returns.columns:
            r = regime_returns[col]
            
            if metric == "sharpe":
                val = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
            elif metric == "return":
                val = r.mean() * 252 * 100
            elif metric == "volatility":
                val = r.std() * np.sqrt(252) * 100
            elif metric == "win_rate":
                val = (r > 0).mean() * 100
            else:
                val = r.mean() * 252 * 100
            
            data.append({
                "Regime": regime.replace("_", " ").title(),
                "Strategy": col,
                "Value": val,
            })
    
    df = pd.DataFrame(data)
    
    regimes_labels = [r.replace("_", " ").title() for r in regimes_order]
    
    colors = [REGIME_COLORS[r] for r in regimes_order if r.replace("_", " ").title() in df["Regime"].values]
    
    if len(returns.columns) == 1:
        pivot = df.pivot(index="Regime", columns="Strategy", values="Value")
        pivot.reindex(regimes_labels).plot(kind="bar", ax=ax, color=colors, legend=False)
    else:
        pivot = df.pivot(index="Regime", columns="Strategy", values="Value")
        pivot.reindex(regimes_labels).plot(kind="bar", ax=ax, legend=True)
    
    ax.set_title(title)
    ax.set_xlabel("Regime")
    
    metric_labels = {
        "sharpe": "Sharpe Ratio",
        "return": "Annualized Return (%)",
        "volatility": "Annualized Volatility (%)",
        "win_rate": "Win Rate (%)",
    }
    ax.set_ylabel(metric_labels.get(metric, metric))
    
    ax.axhline(y=0, color="#888888", linestyle="--", alpha=0.5)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig


def plot_regime_heatmap(
    returns: pd.DataFrame,
    regime_series: pd.Series,
    metric: str = "sharpe",
    title: str = "Strategy-Regime Heatmap",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:

    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("heatmap")
    
    common = returns.index.intersection(regime_series.index)
    returns = returns.loc[common]
    regimes = regime_series.loc[common]
    
    regimes_order = ["low_vol", "normal", "high_vol", "crisis"]
    
    matrix = pd.DataFrame(
        index=[r.replace("_", " ").title() for r in regimes_order],
        columns=returns.columns,
        dtype=float,
    )
    
    for regime in regimes_order:
        mask = regimes == regime
        
        if mask.sum() < 10:
            continue
        
        regime_returns = returns[mask]
        regime_label = regime.replace("_", " ").title()
        
        for col in returns.columns:
            r = regime_returns[col]
            
            if metric == "sharpe":
                val = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
            elif metric == "return":
                val = r.mean() * 252 * 100
            elif metric == "volatility":
                val = r.std() * np.sqrt(252) * 100
            else:
                val = r.mean() * 252 * 100
            
            matrix.loc[regime_label, col] = val
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = "RdYlGn" if metric in ("sharpe", "return") else "YlOrRd"
    center = 0 if metric in ("sharpe", "return") else None
    
    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=center,
        ax=ax,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )
    
    ax.set_title(title)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Regime")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_regime_transitions(
    regime_series: pd.Series,
    title: str = "Regime Transition Matrix",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("single_square")
    
    regimes = ["low_vol", "normal", "high_vol", "crisis"]
    
    transitions = pd.DataFrame(0, index=regimes, columns=regimes, dtype=float)
    
    for i in range(len(regime_series) - 1):
        current = regime_series.iloc[i]
        next_regime = regime_series.iloc[i + 1]
        
        if current in regimes and next_regime in regimes:
            transitions.loc[current, next_regime] += 1
    
    row_sums = transitions.sum(axis=1)
    trans_prob = transitions.div(row_sums, axis=0).fillna(0)
    
    trans_prob.index = [r.replace("_", " ").title() for r in trans_prob.index]
    trans_prob.columns = [r.replace("_", " ").title() for r in trans_prob.columns]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        trans_prob,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Probability"},
    )
    
    ax.set_title(title)
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_expectation_decomposition(
    predictions: pd.DataFrame,
    title: str = r"Expectation Decomposition $E_t$",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("regime_zones")
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    
    pred_cols = [c for c in predictions.columns if "pred" in c.lower()]
    
    if not pred_cols:
        log.warning("No prediction column found")
        return fig
    
    pred_col = pred_cols[0]
    
    if "Date" in predictions.columns:
        pivot = predictions.pivot(index="Date", columns="Sector", values=pred_col)
    else:
        pivot = predictions
    
    for col in pivot.columns:
        from .style import get_sector_color
        color = get_sector_color(col)
        axes[0].plot(pivot.index, pivot[col], label=col, color=color, 
                     linewidth=0.8, alpha=0.7)
    
    axes[0].set_ylabel(r"$E_t[r_{i,t+h}]$")
    axes[0].set_title(title)
    axes[0].axhline(y=0, color="#888888", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", ncol=3, fontsize=7)
    
    dispersion = pivot.std(axis=1)
    axes[1].fill_between(dispersion.index, 0, dispersion, 
                         color="#1f77b4", alpha=0.5)
    axes[1].plot(dispersion.index, dispersion, color="#1f77b4", linewidth=0.8)
    axes[1].set_ylabel("Cross-Sectional Std")
    axes[1].set_xlabel("Date")
    
    format_date_axis(axes[1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig