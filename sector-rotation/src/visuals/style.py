from __future__ import annotations
from typing import Any, TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

MODE_COLORS = {
    "unfiltered": "#1f77b4",
    "relative": "#ff7f0e",
    "absolute": "#2ca02c",
    "dual": "#d62728",
    "pure_momentum": "#9467bd",
    "regime": "#8c564b",
    "regime_invol": "#e377c2",
    "regime_dynamic": "#bcbd22",
    "regime_scalar": "#17becf",
    "equal_weight": "#333333",
    "spy": "#000000",
    "momentum": "#666666",
}

SECTOR_COLORS = {
    "XLB": "#8B4513",
    "XLE": "#228B22",
    "XLF": "#4169E1",
    "XLI": "#FFD700",
    "XLK": "#9400D3",
    "XLP": "#FF6347",
    "XLU": "#20B2AA",
    "XLV": "#DC143C",
    "XLY": "#FF8C00",
}

HORIZON_COLORS = {
    5: "#1f77b4",
    21: "#ff7f0e",
    63: "#2ca02c",
}

REGIME_COLORS = {
    "low_vol": "#90EE90",
    "normal": "#FFFACD",
    "high_vol": "#FFB6C1",
    "crisis": "#FF6B6B",
}

FONT_CONFIG = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "mathtext.fontset": "stix",
}

SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.0
FULL_PAGE_WIDTH = 7.5

GOLDEN_RATIO = 1.618
SQUARE = 1.0
WIDE = 2.0
EXTRA_WIDE = 3.0

FIGURE_SIZES = {
    "single_square": (SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH),
    "single_golden": (SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / GOLDEN_RATIO),
    "single_wide": (SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH / WIDE),
    "double_square": (DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH / 2),
    "double_golden": (DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH / GOLDEN_RATIO),
    "double_wide": (DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH / WIDE),
    "full_page": (FULL_PAGE_WIDTH, 10),
    "equity_curve": (DOUBLE_COLUMN_WIDTH, 4),
    "heatmap": (DOUBLE_COLUMN_WIDTH, 5),
    "comparison_table": (DOUBLE_COLUMN_WIDTH, 3),
    "ic_series": (DOUBLE_COLUMN_WIDTH, 3.5),
    "regime_zones": (DOUBLE_COLUMN_WIDTH, 4.5),
}
def apply_paper_style() -> None:
    mpl.rcdefaults()
    
    plt.rcParams.update(FONT_CONFIG)
    
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.autolayout": True,
        
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.format": "pdf",
    })


def get_figure_size(name: str) -> tuple[float, float]:
    return FIGURE_SIZES.get(name, FIGURE_SIZES["double_golden"])
def get_mode_color(mode: str) -> str:
    return MODE_COLORS.get(mode, "#888888")
def get_sector_color(sector: str) -> str:
    return SECTOR_COLORS.get(sector, "#888888")
def get_horizon_color(horizon: int) -> str:
    return HORIZON_COLORS.get(horizon, "#888888")
def format_percent_axis(ax: "Axes", axis: str = "y") -> None:

    from matplotlib.ticker import FuncFormatter
    
    formatter = FuncFormatter(lambda x, _: f"{x*100:.0f}%")
    
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
def format_date_axis(ax: "Axes", rotation: int = 45) -> None:
    import matplotlib.dates as mdates
    
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha="right")
def add_recession_shading(
    ax: "Axes",
    recessions: list[tuple[str, str]] | None = None,
    color: str = "#CCCCCC",
    alpha: float = 0.3,
) -> None:
    import matplotlib.dates as mdates
    import pandas as pd
    
    if recessions is None:
        recessions = [
            ("2008-01-01", "2009-06-01"),
            ("2020-02-01", "2020-04-30"),
        ]
    
    for start, end in recessions:
        ax.axvspan(
            float(mdates.date2num(pd.Timestamp(start))),
            float(mdates.date2num(pd.Timestamp(end))),
            color=color,
            alpha=alpha,
            zorder=0,
        )
def add_regime_zones(
    ax: "Axes",
    regime_series: "pd.Series",
    alpha: float = 0.2,
) -> None:
    import pandas as pd
    
    regimes = regime_series.unique()
    
    for regime in regimes:
        if regime not in REGIME_COLORS:
            continue
        
        color = REGIME_COLORS[regime]
        
        mask = regime_series == regime
        changes = mask.astype(int).diff().fillna(0)
        
        starts = changes[changes == 1].index
        ends = changes[changes == -1].index
        
        if mask.iloc[0]:
            starts = mask.index[:1].append(starts)
        if mask.iloc[-1]:
            ends = ends.append(mask.index[-1:])
        
        for start, end in zip(starts, ends):
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=0)
def create_mode_legend_handles() -> list[Any]:
    from matplotlib.lines import Line2D
    
    handles = []
    for mode, color in MODE_COLORS.items():
        if mode in ("equal_weight", "spy", "momentum"):
            linestyle = "--"
        else:
            linestyle = "-"
        
        handles.append(
            Line2D([0], [0], color=color, linestyle=linestyle, 
                   label=mode.replace("_", " ").title())
        )
    
    return handles
def create_sector_legend_handles() -> list[Any]:
    from matplotlib.lines import Line2D
    
    handles = []
    for sector, color in SECTOR_COLORS.items():
        handles.append(
            Line2D([0], [0], color=color, label=sector)
        )
    
    return handles
def annotate_max_drawdown(
    ax: "Axes",
    equity_curve: "pd.Series",
    color: str = "#d62728",
) -> None:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.loc[max_dd_idx]
    
    peak_idx = running_max.loc[:max_dd_idx].idxmax()
    
    import matplotlib.dates as mdates
    if hasattr(max_dd_idx, 'timestamp'):
        x_pos = float(mdates.date2num(max_dd_idx))
    else:
        loc = equity_curve.index.get_loc(max_dd_idx)
        x_pos = float(int(loc)) if isinstance(loc, (int, np.integer)) else float(loc)  # type: ignore[arg-type]
    
    ax.axvline(x_pos, color=color, linestyle=":", alpha=0.7)
    
    ax.annotate(
        f"Max DD: {max_dd_val*100:.1f}%",
        xy=(x_pos, float(equity_curve.loc[max_dd_idx])),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=8,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
    )
def annotate_sharpe(
    ax: "Axes",
    sharpe: float,
    position: str = "upper left",
) -> None:
    loc_map = {
        "upper left": (0.02, 0.98),
        "upper right": (0.98, 0.98),
        "lower left": (0.02, 0.02),
        "lower right": (0.98, 0.02),
    }
    
    x, y = loc_map.get(position, (0.02, 0.98))
    ha = "left" if "left" in position else "right"
    va = "top" if "upper" in position else "bottom"
    
    ax.text(
        x, y,
        f"Sharpe: {sharpe:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                  edgecolor="#CCCCCC", alpha=0.9),
    )