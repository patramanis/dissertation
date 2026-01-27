from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .style import (
    apply_paper_style,
    get_figure_size,
    get_mode_color,
    get_horizon_color,
    format_percent_axis,
    format_date_axis,
    add_recession_shading,
    annotate_max_drawdown,
    annotate_sharpe,
)

log = logging.getLogger(__name__)

def plot_equity_curves(
    equity_curves: dict[str, pd.Series] | pd.DataFrame,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curves",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    log_scale: bool = False,
    show_legend: bool = True,
    normalize: bool = True,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("equity_curve")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(equity_curves, dict):
        df = pd.DataFrame(equity_curves)
    else:
        df = equity_curves.copy()
    
    if normalize:
        df = df / df.iloc[0]
        if benchmark is not None:
            benchmark = benchmark / benchmark.iloc[0]
    
    for col in df.columns:
        mode = col.split("_h")[0] if "_h" in col else col
        color = get_mode_color(mode)
        
        ax.plot(df.index, df[col], label=col, color=color, linewidth=1.5)
    
    if benchmark is not None:
        ax.plot(
            benchmark.index, benchmark,
            label="Benchmark",
            color="#333333",
            linestyle="--",
            linewidth=1.5,
        )
    
    add_recession_shading(ax)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    
    if log_scale:
        ax.set_yscale("log")
    
    format_date_axis(ax)
    
    if show_legend:
        ax.legend(
            loc="upper left",
            ncol=min(3, len(df.columns)),
            fontsize=8,
        )
    
    ax.axhline(y=1.0, color="#888888", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
        log.info("Saved equity curves to %s", output_path)
    
    return fig


def plot_equity_by_horizon(
    equity_curves: dict[str, pd.Series],
    mode: str,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("equity_curve")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mode_curves = {k: v for k, v in equity_curves.items() if k.startswith(mode)}
    
    for name, curve in mode_curves.items():
        horizon = int(name.split("_h")[-1]) if "_h" in name else 21
        color = get_horizon_color(horizon)
        
        curve = curve / curve.iloc[0]
        
        ax.plot(curve.index, curve, label=f"H={horizon}", color=color, linewidth=1.5)
    
    ax.set_title(f"{mode.replace('_', ' ').title()} - Horizon Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    
    format_date_axis(ax)
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_drawdowns(
    drawdown_curves: dict[str, pd.Series] | pd.DataFrame,
    title: str = "Drawdowns",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    top_n: int | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("equity_curve")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(drawdown_curves, dict):
        df = pd.DataFrame(drawdown_curves)
    else:
        df = drawdown_curves.copy()
    
    if top_n is not None:
        max_dd = df.min().nlargest(top_n)
        df = df[max_dd.index]
    
    for col in df.columns:
        mode = col.split("_h")[0] if "_h" in col else col
        color = get_mode_color(mode)
        
        ax.fill_between(
            df.index, 
            0, 
            df[col] * 100,
            alpha=0.3,
            color=color,
            label=col,
        )
        ax.plot(df.index, df[col] * 100, color=color, linewidth=0.5)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    format_date_axis(ax)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(top=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig


def plot_underwater(
    equity_curve: pd.Series,
    title: str = "Underwater Chart",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = (get_figure_size("equity_curve")[0], 6)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                              gridspec_kw={"height_ratios": [3, 1]})
    
    equity = equity_curve / equity_curve.iloc[0]
    
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    
    axes[0].plot(equity.index, equity, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("Portfolio Value")
    axes[0].set_title(title)
    add_recession_shading(axes[0])
    
    axes[1].fill_between(
        drawdown.index, 0, drawdown,
        color="#d62728", alpha=0.5,
    )
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylim(top=0)
    
    format_date_axis(axes[1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_rolling_sharpe(
    returns: dict[str, pd.Series] | pd.DataFrame,
    window: int = 252,
    title: str = "Rolling Sharpe Ratio (1Y)",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    risk_free_rate: float = 0.025,
) -> Figure:
  
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("ic_series")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(returns, dict):
        df = pd.DataFrame(returns)
    else:
        df = returns.copy()
    
    daily_rf = risk_free_rate / 252
    
    for col in df.columns:
        mode = col.split("_h")[0] if "_h" in col else col
        color = get_mode_color(mode)
        
        excess = df[col] - daily_rf
        rolling_mean = excess.rolling(window).mean()
        rolling_std = excess.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        ax.plot(
            rolling_sharpe.index, 
            rolling_sharpe,
            label=col,
            color=color,
            linewidth=1.2,
        )
    
    ax.axhline(y=0, color="#888888", linestyle="--", alpha=0.7)
    ax.axhline(y=1, color="#2ca02c", linestyle=":", alpha=0.5)
    ax.axhline(y=-1, color="#d62728", linestyle=":", alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    
    format_date_axis(ax)
    ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig


def plot_rolling_volatility(
    returns: dict[str, pd.Series] | pd.DataFrame,
    window: int = 63,
    title: str = "Rolling Volatility (3M)",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("ic_series")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(returns, dict):
        df = pd.DataFrame(returns)
    else:
        df = returns.copy()
    
    for col in df.columns:
        mode = col.split("_h")[0] if "_h" in col else col
        color = get_mode_color(mode)
        
        rolling_vol = df[col].rolling(window).std() * np.sqrt(252) * 100
        
        ax.plot(
            rolling_vol.index,
            rolling_vol,
            label=col,
            color=color,
            linewidth=1.2,
        )
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (%)")
    
    format_date_axis(ax)
    ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_relative_performance(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
    title: str = "Relative Performance",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:

    apply_paper_style()
    
    if figsize is None:
        figsize = (get_figure_size("equity_curve")[0], 5)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    
    strategy = strategy_equity / strategy_equity.iloc[0]
    benchmark = benchmark_equity / benchmark_equity.iloc[0]
    
    common = strategy.index.intersection(benchmark.index)
    strategy = strategy.loc[common]
    benchmark = benchmark.loc[common]
    
    axes[0].plot(strategy.index, strategy, label=strategy_name, 
                 color="#1f77b4", linewidth=1.5)
    axes[0].plot(benchmark.index, benchmark, label=benchmark_name,
                 color="#333333", linestyle="--", linewidth=1.5)
    axes[0].set_ylabel("Portfolio Value")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left")
    add_recession_shading(axes[0])
    
    relative = (strategy / benchmark) - 1
    
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in relative]
    axes[1].fill_between(
        relative.index, 0, relative * 100,
        color="#1f77b4", alpha=0.5,
    )
    axes[1].axhline(y=0, color="#888888", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel(f"vs {benchmark_name} (%)")
    axes[1].set_xlabel("Date")
    
    format_date_axis(axes[1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig

def plot_return_distribution(
    returns: pd.Series,
    title: str = "Return Distribution",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:

    apply_paper_style()
    
    if figsize is None:
        figsize = get_figure_size("single_golden")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(
        returns * 100,
        bins=50,
        density=True,
        alpha=0.7,
        color="#1f77b4",
        edgecolor="white",
    )
    
    from scipy import stats
    
    x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
    mean = returns.mean() * 100
    std = returns.std() * 100
    
    normal = stats.norm.pdf(x, mean, std)
    ax.plot(x, normal, color="#d62728", linestyle="--", linewidth=1.5, 
            label="Normal")
    
    var_95 = float(np.percentile(returns, 5) * 100)
    ax.axvline(var_95, color="#9467bd", linestyle=":", linewidth=1.5,
               label=f"VaR 5%: {var_95:.2f}%")
    
    ax.set_title(title)
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    
    return fig