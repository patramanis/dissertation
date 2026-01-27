from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.025

@dataclass
class VectorBTResult:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_dd_duration: int = 0
    
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    
    start_date: str = ""
    end_date: str = ""
    n_days: int = 0
    mode: str = ""
    horizon: int = 0


def run_vectorbt(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
    risk_free_rate: float = RISK_FREE_RATE,
    output_dir: Path | None = None,
    mode: str = "",
    horizon: int = 0,
    rebalance_frequency: int | None = None,
) -> VectorBTResult:
    try:
        import vectorbt as vbt  # type: ignore[import-untyped]
        from vectorbt.portfolio.enums import Direction  # type: ignore[import-untyped]
    except ImportError:
        log.warning("VectorBT not installed.")
        return _run_vectorbt_fallback(
            weights, prices, transaction_cost_bps, 
            risk_free_rate, output_dir, mode, horizon, rebalance_frequency
        )
    
    rebal_freq = rebalance_frequency if rebalance_frequency is not None else (horizon if horizon > 0 else 1)
    
    if rebal_freq > 1:
        log.info("Horizon-matched rebalancing: every %d days (horizon=%d)", rebal_freq, horizon)
    
    common_cols = [c for c in weights.columns if c in prices.columns and c != "cash"]
    
    if not common_cols:
        log.error("No common sector columns between weights and prices")
        return VectorBTResult()
    
    prices = prices.sort_index()
    weights = weights.sort_index()
    
    common_dates = weights.index.intersection(prices.index)
    
    last_price_date = prices.index[-1]
    valid_weight_dates = [d for d in common_dates 
                          if d in prices.index and d < last_price_date]
    valid_weight_dates = pd.DatetimeIndex(valid_weight_dates)
    
    if len(valid_weight_dates) < 10:
        log.error("Insufficient data for backtest: %d valid dates", len(valid_weight_dates))
        return VectorBTResult()
    
    weights_aligned = weights.loc[valid_weight_dates, common_cols].copy()

    if rebal_freq > 1:
        rebal_indices = list(range(0, len(weights_aligned), rebal_freq))
        rebal_dates = list(weights_aligned.index[rebal_indices])
        
        weights_rebal = weights_aligned.copy()
        weights_rebal.loc[:] = np.nan
        weights_rebal.loc[rebal_dates, :] = weights_aligned.loc[rebal_dates, :].values
        weights_rebal = weights_rebal.ffill()
        
        log.info(
            "Applied horizon-matched rebalancing: %d rebal dates, %d total dates",
            len(rebal_dates), len(weights_aligned)
        )
        weights_aligned = weights_rebal
    
    forward_returns = prices[common_cols].pct_change().shift(-1)
    returns_aligned = forward_returns.loc[valid_weight_dates].copy()
    
    valid_mask = ~returns_aligned.isna().any(axis=1)
    weights_aligned = weights_aligned.loc[valid_mask]
    returns_aligned = returns_aligned.loc[valid_mask]
    
    log.info("PIT Aligned: %d dates, %d sectors", len(weights_aligned), len(common_cols))
    
    weight_changes = weights_aligned.diff().abs().fillna(0)
    costs = weight_changes * (transaction_cost_bps / 10000)
    
    portfolio_returns = (weights_aligned * returns_aligned).sum(axis=1)
    
    cash_weight = 1 - weights_aligned.sum(axis=1)
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    cash_return = cash_weight * rf_daily
    
    gross_returns = portfolio_returns + cash_return
    net_returns = gross_returns - costs.sum(axis=1)
    
    equity_curve = (1 + net_returns).cumprod()
    equity_curve = equity_curve / equity_curve.iloc[0]
    equity_curve.index.name = "Date"
    
    running_max = equity_curve.cummax()
    drawdown_curve = (equity_curve - running_max) / running_max
    drawdown_curve.index.name = "Date"
    
    result = VectorBTResult(
        mode=mode,
        horizon=horizon,
        start_date=str(weights_aligned.index[0].date()),
        end_date=str(weights_aligned.index[-1].date()),
        n_days=len(weights_aligned),
    )
    
    result.total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    
    years = result.n_days / TRADING_DAYS
    result.cagr = float((1 + result.total_return) ** (1/years) - 1) if years > 0 else 0
    
    result.volatility = float(net_returns.std() * np.sqrt(TRADING_DAYS))
    
    excess_return = result.cagr - risk_free_rate
    result.sharpe = float(excess_return / result.volatility) if result.volatility > 0 else 0
    
    downside = net_returns[net_returns < 0]
    result.downside_deviation = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 0 else 0
    result.sortino = float(excess_return / result.downside_deviation) if result.downside_deviation > 0 else 0
    
    result.max_drawdown = float(drawdown_curve.min())
    
    result.calmar = float(result.cagr / abs(result.max_drawdown)) if result.max_drawdown < 0 else 0
    
    dd_periods = (drawdown_curve != 0).astype(int)
    dd_groups = (dd_periods != dd_periods.shift()).cumsum()
    dd_durations = dd_periods.groupby(dd_groups).sum()
    result.max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0
    
    result.var_95 = float(np.percentile(net_returns, 5))
    result.cvar_95 = float(net_returns[net_returns <= result.var_95].mean())
    
    significant_trades = (weight_changes > 0.01).sum(axis=1)
    result.total_trades = int(significant_trades.sum())
    
    winning_days = (net_returns > 0).sum()
    result.win_rate = float(winning_days / len(net_returns))
    
    result.avg_trade_return = float(net_returns.mean())
    
    gains = net_returns[net_returns > 0].sum()
    losses = abs(net_returns[net_returns < 0].sum())
    result.profit_factor = float(gains / losses) if losses > 0 else float("inf")
    
    result.equity_curve = equity_curve
    result.drawdown_curve = drawdown_curve
    result.returns = net_returns
    
    if output_dir is not None:
        _save_vectorbt_results(result, output_dir)
    
    log.info(
        "VectorBT backtest complete: Return=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
        result.total_return * 100,
        result.sharpe,
        result.max_drawdown * 100,
    )
    
    return result

def _run_vectorbt_fallback(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    transaction_cost_bps: float,
    risk_free_rate: float,
    output_dir: Path | None,
    mode: str,
    horizon: int,
    rebalance_frequency: int | None = None,
) -> VectorBTResult:
    log.info("Using fallback vectorized backtest (VectorBT not available)")
    rebal_freq = rebalance_frequency if rebalance_frequency is not None else (horizon if horizon > 0 else 1)
    
    common_cols = [c for c in weights.columns if c in prices.columns and c != "cash"]
    
    if not common_cols:
        return VectorBTResult()
    
    prices = prices.sort_index()
    weights = weights.sort_index()
    
    common_dates = weights.index.intersection(prices.index)
    last_price_date = prices.index[-1]
    valid_weight_dates = [d for d in common_dates 
                          if d in prices.index and d < last_price_date]
    valid_weight_dates = pd.DatetimeIndex(valid_weight_dates)
    
    if len(valid_weight_dates) < 10:
        return VectorBTResult()
    
    if len(valid_weight_dates) < 10:
        return VectorBTResult()
    
    weights_aligned = weights.loc[valid_weight_dates, common_cols].copy()
    
    if rebal_freq > 1:
        rebal_indices = list(range(0, len(weights_aligned), rebal_freq))
        rebal_dates = list(weights_aligned.index[rebal_indices])
        weights_rebal = weights_aligned.copy()
        weights_rebal.loc[:] = np.nan
        weights_rebal.loc[rebal_dates, :] = weights_aligned.loc[rebal_dates, :].values
        weights_rebal = weights_rebal.ffill()
        weights_aligned = weights_rebal
    
    forward_returns = prices[common_cols].pct_change().shift(-1)
    returns_aligned = forward_returns.loc[valid_weight_dates].copy()
    valid_mask = ~returns_aligned.isna().any(axis=1)
    w = weights_aligned.loc[valid_mask]
    returns = returns_aligned.loc[valid_mask]
    weight_changes = w.diff().abs().fillna(0)
    costs = weight_changes * (transaction_cost_bps / 10000)

    portfolio_returns = (w * returns).sum(axis=1)
    
    cash_weight = 1 - w.sum(axis=1)
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    cash_return = cash_weight * rf_daily
    
    gross_returns = portfolio_returns + cash_return
    net_returns = gross_returns - costs.sum(axis=1)
    
    equity_curve = (1 + net_returns).cumprod()
    
    running_max = equity_curve.cummax()
    drawdown_curve = (equity_curve - running_max) / running_max
    
    result = VectorBTResult(
        mode=mode,
        horizon=horizon,
        start_date=str(common_dates[0].date()),
        end_date=str(common_dates[-1].date()),
        n_days=len(common_dates),
    )
    
    result.total_return = float(equity_curve.iloc[-1] - 1)
    years = result.n_days / TRADING_DAYS
    result.cagr = float((1 + result.total_return) ** (1/years) - 1) if years > 0 else 0
    result.volatility = float(net_returns.std() * np.sqrt(TRADING_DAYS))
    result.sharpe = float((result.cagr - risk_free_rate) / result.volatility) if result.volatility > 0 else 0
    result.max_drawdown = float(drawdown_curve.min())
    result.calmar = float(result.cagr / abs(result.max_drawdown)) if result.max_drawdown < 0 else 0
    result.var_95 = float(np.percentile(net_returns, 5))
    result.cvar_95 = float(net_returns[net_returns <= result.var_95].mean()) if any(net_returns <= result.var_95) else result.var_95
    result.equity_curve = equity_curve
    result.drawdown_curve = drawdown_curve
    result.returns = net_returns
    
    if output_dir:
        _save_vectorbt_results(result, output_dir)
    
    return result


def _save_vectorbt_results(result: VectorBTResult, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    equity_path = output_dir / "equity_curve.csv"
    equity_df = pd.DataFrame({"value": result.equity_curve.values}, index=result.equity_curve.index)
    equity_df.index.name = "Date"
    equity_df.to_csv(equity_path)
    
    dd_path = output_dir / "drawdowns.csv"
    dd_df = pd.DataFrame({"drawdown": result.drawdown_curve.values}, index=result.drawdown_curve.index)
    dd_df.index.name = "Date"
    dd_df.to_csv(dd_path)
    
    returns_path = output_dir / "returns.csv"
    returns_df = pd.DataFrame({"return": result.returns.values}, index=result.returns.index)
    returns_df.index.name = "Date"
    returns_df.to_csv(returns_path)
    
    metrics = {
        "total_return": result.total_return,
        "cagr": result.cagr,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "calmar": result.calmar,
        "max_drawdown": result.max_drawdown,
        "max_dd_duration": result.max_dd_duration,
        "volatility": result.volatility,
        "var_95": result.var_95,
        "cvar_95": result.cvar_95,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "total_trades": result.total_trades,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "n_days": result.n_days,
        "mode": result.mode,
        "horizon": result.horizon,
    }
    
    metrics_path = output_dir / "vectorbt_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    log.info("VectorBT results saved to %s", output_dir)


def compare_strategies(
    results: dict[str, VectorBTResult],
    benchmark: VectorBTResult | None = None,
) -> pd.DataFrame:
    rows = []
    
    for name, result in results.items():
        row = {
            "Strategy": name,
            "Total Return (%)": result.total_return * 100,
            "CAGR (%)": result.cagr * 100,
            "Volatility (%)": result.volatility * 100,
            "Sharpe": result.sharpe,
            "Sortino": result.sortino,
            "Max DD (%)": result.max_drawdown * 100,
            "Calmar": result.calmar,
            "Win Rate (%)": result.win_rate * 100,
        }
        
        if benchmark is not None:
            row["Alpha (%)"] = (result.cagr - benchmark.cagr) * 100
            row["Beta"] = _compute_beta(result.returns, benchmark.returns)
        
        rows.append(row)
    
    return pd.DataFrame(rows).set_index("Strategy")


def _compute_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    common = strategy_returns.index.intersection(benchmark_returns.index)
    
    if len(common) < 10:
        return 0.0
    
    s = strategy_returns.loc[common]
    b = benchmark_returns.loc[common]
    
    cov_val = s.cov(b)
    var_val = b.var()
    
    cov = float(cov_val) if cov_val is not None else 0.0  # type: ignore[arg-type]
    var = float(var_val) if var_val is not None else 0.0  # type: ignore[arg-type]
    
    return cov / var if var > 0 else 0.0


def build_equity_comparison_df(
    results: dict[str, VectorBTResult],
) -> pd.DataFrame:
    curves = {}
    
    for name, result in results.items():
        curves[name] = result.equity_curve
    
    df = pd.DataFrame(curves)
    df = df / df.iloc[0]
    
    return df


def build_drawdown_comparison_df(
    results: dict[str, VectorBTResult],
) -> pd.DataFrame:
    curves = {}
    
    for name, result in results.items():
        curves[name] = result.drawdown_curve
    
    return pd.DataFrame(curves)


def build_equal_weight_benchmark(
    prices: pd.DataFrame,
    sectors: list[str] | None = None,
    transaction_cost_bps: float = 10.0,
) -> VectorBTResult:
    from .contracts import SECTORS
    
    if sectors is None:
        sectors = SECTORS
    
    available = [s for s in sectors if s in prices.columns]
    prices = prices[available]
    
    n_sectors = len(available)
    weights = pd.DataFrame(
        1.0 / n_sectors,
        index=prices.index,
        columns=available,
    )
    
    return run_vectorbt(
        weights=weights,
        prices=prices,
        transaction_cost_bps=transaction_cost_bps,
        mode="equal_weight",
        horizon=0,
    )


def build_momentum_benchmark(
    prices: pd.DataFrame,
    lookback: int = 21,
    top_n: int = 3,
    transaction_cost_bps: float = 10.0,
) -> VectorBTResult:
    momentum = prices.pct_change(lookback)
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for date in momentum.index[lookback:]:
        row = momentum.loc[date]
        top = row.nlargest(top_n).index
        weights.loc[date, top] = 1.0 / top_n
    
    return run_vectorbt(
        weights=weights.iloc[lookback:],
        prices=prices.iloc[lookback:],
        transaction_cost_bps=transaction_cost_bps,
        mode=f"momentum_{lookback}d",
        horizon=0,
    )