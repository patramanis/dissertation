from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

def run_quantstats(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    output_dir: Path | None = None,
    title: str = "Strategy Performance",
    rf: float = 0.0,
) -> dict[str, Any]:
    try:
        import quantstats as qs  # type: ignore[import-untyped]
    except ImportError:
        log.warning("QuantStats not installed. Run: pip install quantstats")
        return {"error": "quantstats not installed"}
    
    if isinstance(returns, pd.DataFrame):
        if "net_ret" in returns.columns:
            returns = returns["net_ret"]
        else:
            first_col = returns.columns[0]
            returns = returns[first_col]
    
    returns = returns.dropna()
    
    if len(returns) < 10:
        log.warning("Not enough returns for QuantStats analysis")
        return {"error": "insufficient_data"}
    metrics = {}
    
    try:
        metrics["total_return"] = float(qs.stats.comp(returns))  # type: ignore[arg-type]
        metrics["cagr"] = float(qs.stats.cagr(returns))  # type: ignore[arg-type]
        metrics["sharpe"] = float(qs.stats.sharpe(returns, rf=rf))  # type: ignore[arg-type]
        metrics["sortino"] = float(qs.stats.sortino(returns, rf=rf))  # type: ignore[arg-type]
        metrics["max_drawdown"] = float(qs.stats.max_drawdown(returns))  # type: ignore[arg-type]
        metrics["calmar"] = float(qs.stats.calmar(returns))  # type: ignore[arg-type]
        metrics["volatility"] = float(qs.stats.volatility(returns))  # type: ignore[arg-type]
        metrics["avg_return"] = float(returns.mean())
        metrics["win_rate"] = float(qs.stats.win_rate(returns))  # type: ignore[arg-type]
        metrics["best_day"] = float(returns.max())
        metrics["worst_day"] = float(returns.min())
        metrics["skew"] = float(qs.stats.skew(returns))  # type: ignore[arg-type]
        metrics["kurtosis"] = float(qs.stats.kurtosis(returns))  # type: ignore[arg-type]
        
        dd = qs.stats.to_drawdown_series(returns)
        metrics["avg_drawdown"] = float(dd.mean())
        
    except Exception as e:
        log.warning("Error computing some QuantStats metrics: %s", e)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        html_path = output_dir / "quantstats.html"
        
        try:
            qs.reports.html(
                returns,
                benchmark=benchmark,
                output=str(html_path),
                title=title,
                rf=rf,
            )
            log.info("QuantStats HTML report saved to %s", html_path)
            metrics["html_path"] = str(html_path)
        except Exception as e:
            log.error("Failed to generate QuantStats HTML: %s", e)
            metrics["html_error"] = str(e)
        
        metrics_path = output_dir / "quantstats_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, default=str),
            encoding="utf-8",
        )
        log.info("QuantStats metrics saved to %s", metrics_path)
    
    return metrics


def run_quantstats_comparison(
    strategies: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    try:
        import quantstats as qs  # type: ignore[import-untyped]
    except ImportError:
        log.warning("QuantStats not installed")
        return pd.DataFrame()
    
    rows = []
    
    for name, returns in strategies.items():
        returns = returns.dropna()
        
        if len(returns) < 10:
            continue
        
        row = {
            "Strategy": name,
            "Total Return": qs.stats.comp(returns),
            "CAGR": qs.stats.cagr(returns),
            "Sharpe": qs.stats.sharpe(returns),
            "Sortino": qs.stats.sortino(returns),
            "Max DD": qs.stats.max_drawdown(returns),
            "Calmar": qs.stats.calmar(returns),
            "Volatility": qs.stats.volatility(returns),
            "Win Rate": qs.stats.win_rate(returns),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / "strategy_comparison.csv"
        df.to_csv(csv_path, index=False)
        log.info("Strategy comparison saved to %s", csv_path)
    
    return df

def prepare_returns_for_quantstats(
    returns_df: pd.DataFrame,
    return_col: str = "net_ret",
) -> pd.Series:
    if "Date" in returns_df.columns:
        returns_df = returns_df.set_index("Date")
    
    returns = returns_df[return_col].copy()
    returns.index = pd.to_datetime(returns.index)
    returns.index.name = None
    
    returns = returns.asfreq("D", fill_value=0)
    
    return returns


def load_spy_benchmark() -> pd.Series:
    from .contracts import get_data_paths
    
    paths = get_data_paths()
    
    if not paths["spy_prices"].exists():
        log.warning("SPY prices not found")
        return pd.Series(dtype=float)
    
    spy = pd.read_csv(paths["spy_prices"], parse_dates=["Date"])
    spy = spy.set_index("Date")
    
    if "Price" in spy.columns:
        prices = spy["Price"]
    elif "Close" in spy.columns:
        prices = spy["Close"]
    else:
        price_cols = [c for c in spy.columns if c not in ["Volume"]]
        prices = spy[price_cols[0]] if price_cols else pd.Series(dtype=float)
    
    returns = prices.pct_change().dropna()
    
    return returns