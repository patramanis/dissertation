from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.025
DEFAULT_COST_BPS = 10.0

HORIZONS = [5, 21, 63]

SECTORS = [
    "XLB", "XLE", "XLF", "XLI", "XLK", 
    "XLP", "XLU", "XLV", "XLY"
]

@dataclass
class VectorBTMetrics:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    volatility: float = 0.0
    
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    avg_exposure: float = 0.0
    time_in_market: float = 0.0
    
    best_day: float = 0.0
    worst_day: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0


@dataclass
class AlphalensMetrics:
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    ic_t_stat: float = 0.0
    ic_skew: float = 0.0
    
    quantile_spread_mean: float = 0.0
    quantile_spread_std: float = 0.0
    
    factor_return_mean: float = 0.0
    factor_return_sharpe: float = 0.0
    
    turnover: float = 0.0
    autocorr_1d: float = 0.0
    autocorr_5d: float = 0.0


@dataclass
class QuantStatsMetrics:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    
    volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    
    omega: float = 0.0
    gain_to_pain: float = 0.0
    payoff_ratio: float = 0.0
    
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    alpha: float = 0.0
    beta: float = 0.0
    correlation_to_spy: float = 0.0


@dataclass
class UnifiedBacktestResult:
    run_id: str = ""
    mode: str = ""
    horizon: int = 0
    category: str = ""
    timestamp: str = ""
    
    start_date: str = ""
    end_date: str = ""
    n_trading_days: int = 0
    
    n_rankings: int = 0
    n_weights: int = 0
    pct_data_loss: float = 0.0
    
    vectorbt: VectorBTMetrics = field(default_factory=VectorBTMetrics)
    alphalens: AlphalensMetrics = field(default_factory=AlphalensMetrics)
    quantstats: QuantStatsMetrics = field(default_factory=QuantStatsMetrics)
    
    sharpe_vectorbt_vs_quantstats_diff: float = 0.0
    return_vectorbt_vs_quantstats_diff: float = 0.0
    
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    output_dir: str = ""
    vectorbt_equity_path: str = ""
    quantstats_html_path: str = ""
    alphalens_tearsheet_path: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def load_prices(
    workspace_root: Path | None = None,
) -> pd.DataFrame:
    if workspace_root is None:
        workspace_root = Path(__file__).parent.parent.parent
    
    candidates = [
        workspace_root / "data" / "raw" / "SPDR.csv",
        workspace_root / "data" / "interim" / "aligned_pit" / "SPDR.csv",
        workspace_root / "data" / "processed" / "SPDR.csv",
    ]
    
    for path in candidates:
        if path.exists():
            prices = pd.read_csv(path, index_col=0, parse_dates=True)
            available = [s for s in SECTORS if s in prices.columns]
            
            if "SPY" in prices.columns:
                available.append("SPY")
            
            log.info("Loaded prices from %s: %d dates, %d assets", 
                     path, len(prices), len(available))
            return prices[available]
    
    raise FileNotFoundError(f"No price data found in: {candidates}")


def load_spy_returns(
    prices: pd.DataFrame | None = None,
    workspace_root: Path | None = None,
) -> pd.Series:
    if prices is not None and "SPY" in prices.columns:
        spy_returns = prices["SPY"].pct_change().dropna()
        spy_returns.name = "SPY"
        return spy_returns
    
    if workspace_root is None:
        workspace_root = Path(__file__).parents[2]
    
    spy_path = workspace_root / "data" / "raw" / "SPY.csv"
    if not spy_path.exists():
        spy_path = workspace_root / "data" / "interim" / "aligned_pit" / "SPY.csv"
    
    if spy_path.exists():
        spy_df = pd.read_csv(spy_path, parse_dates=["Date"])
        spy_df = spy_df.set_index("Date")
        
        if "SPY" in spy_df.columns:
            prices_col = spy_df["SPY"]
        elif "Close" in spy_df.columns:
            prices_col = spy_df["Close"]
        elif "Price" in spy_df.columns:
            prices_col = spy_df["Price"]
        else:
            prices_col = spy_df.select_dtypes(include=[np.number]).iloc[:, 0]
        
        spy_returns = prices_col.pct_change().dropna()
        spy_returns.name = "SPY"
        log.info("Loaded SPY benchmark from %s (%d days)", spy_path.name, len(spy_returns))
        return spy_returns
    
    log.warning("SPY not found in prices or dedicated file, benchmark comparison disabled")
    return pd.Series(dtype=float)

def load_fold_rankings(
    run_dir: Path,
) -> pd.DataFrame:

    run_dir = Path(run_dir)
    all_dfs = []
    
    fold_dirs = sorted(
        run_dir.glob("fold*"), 
        key=lambda x: int(x.name.replace("fold", ""))
    )
    
    for fold_dir in fold_dirs:
        rankings_path = fold_dir / "rankings.csv"
        
        if not rankings_path.exists():
            log.warning("Rankings not found: %s", rankings_path)
            continue
        
        try:
            df = pd.read_csv(rankings_path, parse_dates=["Date"])
            fold_num = int(fold_dir.name.replace("fold", ""))
            df["fold"] = fold_num
            all_dfs.append(df)
            log.debug("Loaded fold %d: %d rows", fold_num, len(df))
        except Exception as e:
            log.error("Failed to load %s: %s", rankings_path, e)
    
    if not all_dfs:
        root_rankings = run_dir / "rankings.csv"
        if root_rankings.exists():
            log.info("Using root-level rankings.csv (single-fold mode)")
            try:
                df = pd.read_csv(root_rankings, parse_dates=["Date"])
                df["fold"] = 0
                all_dfs.append(df)
            except Exception as e:
                log.error("Failed to load root rankings: %s", e)
    
    if not all_dfs:
        raise FileNotFoundError(f"No rankings found in {run_dir}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    combined = combined.sort_values(["Date", "Sector", "fold"])
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["Date", "Sector"], keep="first")
    n_after = len(combined)
    
    if n_before > n_after:
        log.info("Removed %d overlapping rows (kept first fold)", n_before - n_after)
    
    log.info("Loaded %d rankings from %d folds, %d unique dates", 
             len(combined), len(fold_dirs), combined["Date"].nunique())
    
    return combined

def build_weights_from_rankings(
    rankings: pd.DataFrame,
    apply_actionability_shift: bool = True,
) -> pd.DataFrame:

    weights_wide = rankings.pivot(
        index="Date",
        columns="Sector",
        values="portfolio_weight"
    )
    
    for sector in SECTORS:
        if sector not in weights_wide.columns:
            weights_wide[sector] = 0.0
    
    weights_wide = weights_wide[SECTORS].copy()
    weights_wide = weights_wide.fillna(0.0)
    weights_wide["cash"] = 1.0 - weights_wide[SECTORS].sum(axis=1)
    weights_wide = weights_wide.clip(lower=0.0)
    
    row_sums = weights_wide.sum(axis=1)
    weights_wide = weights_wide.div(row_sums, axis=0)
    weights_wide = weights_wide.sort_index()
    
    if apply_actionability_shift:
        weights_wide = weights_wide.shift(1)
        weights_wide = weights_wide.dropna()
        log.info("Applied actionability shift: signal@t → weight@t+1")
    
    return weights_wide

def run_vectorbt_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
    risk_free_rate: float = RISK_FREE_RATE,
    horizon: int = 21,
    output_dir: Path | None = None,
) -> tuple[VectorBTMetrics, pd.Series, pd.Series]:
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import vectorbt as vbt  # type: ignore[import-untyped]
    except ImportError:
        log.error("VectorBT not installed. Run: pip install vectorbt")
        return VectorBTMetrics(), pd.Series(dtype=float), pd.Series(dtype=float)
    
    metrics = VectorBTMetrics()
    
    sector_cols = [c for c in weights.columns if c in SECTORS and c in prices.columns]
    
    if not sector_cols:
        log.error("No common sectors between weights and prices")
        return metrics, pd.Series(dtype=float), pd.Series(dtype=float)
    
    common_dates = weights.index.intersection(prices.index)
    
    if len(common_dates) < 10:
        log.error("Insufficient overlapping dates: %d", len(common_dates))
        return metrics, pd.Series(dtype=float), pd.Series(dtype=float)
    
    weights_aligned = weights.loc[common_dates, sector_cols].copy()
    prices_aligned = prices.loc[common_dates, sector_cols].copy()
    
    try:
        returns = prices_aligned.pct_change().shift(-1)
        port_returns = (weights_aligned * returns).sum(axis=1)
        
        cash_weight = 1 - weights_aligned[sector_cols].sum(axis=1)
        rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
        cash_return = cash_weight * rf_daily
        gross_returns = port_returns + cash_return
        
        weight_changes = weights_aligned.diff().abs().sum(axis=1)
        costs = weight_changes * (cost_bps / 10000)
        net_returns = gross_returns - costs
        
        net_returns = net_returns.iloc[:-1].dropna()
        
        if len(net_returns) < 10:
            log.error("Insufficient returns after processing: %d", len(net_returns))
            return metrics, pd.Series(dtype=float), pd.Series(dtype=float)
        
        equity = (1 + net_returns).cumprod()
        metrics.total_return = float(equity.iloc[-1] - 1)
        
        n_years = len(net_returns) / TRADING_DAYS
        if n_years > 0 and equity.iloc[-1] > 0:
            metrics.cagr = float((equity.iloc[-1]) ** (1 / n_years) - 1)
        
        metrics.volatility = float(net_returns.std() * np.sqrt(TRADING_DAYS))
        
        if metrics.volatility > 0:
            excess_return = net_returns.mean() * TRADING_DAYS - risk_free_rate
            metrics.sharpe = float(excess_return / metrics.volatility)
        
        downside_returns = net_returns[net_returns < 0]
        if len(downside_returns) > 0:
            downside_std = float(downside_returns.std() * np.sqrt(TRADING_DAYS))
            if downside_std > 0:
                excess_return = net_returns.mean() * TRADING_DAYS - risk_free_rate
                metrics.sortino = float(excess_return / downside_std)
        
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        metrics.max_drawdown = float(drawdown.min())
        
        if metrics.max_drawdown != 0:
            metrics.calmar = float(metrics.cagr / abs(metrics.max_drawdown))
        
        cash_weights = 1 - weights_aligned[sector_cols].sum(axis=1)
        metrics.avg_exposure = float(1 - cash_weights.mean())
        metrics.time_in_market = float((cash_weights < 0.99).mean())
        metrics.best_day = float(net_returns.max())
        metrics.worst_day = float(net_returns.min())
        metrics.win_rate = float((net_returns > 0).mean())
        winning = net_returns[net_returns > 0]
        losing = net_returns[net_returns < 0]
        if len(winning) > 0:
            metrics.avg_win = float(winning.mean())
        if len(losing) > 0:
            metrics.avg_loss = float(losing.mean())
        
        metrics.total_trades = int((weight_changes > 0.01).sum())
        
        if output_dir is not None:
            equity_path = output_dir / "equity_curve.csv"
            equity.to_csv(equity_path)
            
            returns_path = output_dir / "daily_returns.csv"
            net_returns.to_csv(returns_path)
            
            drawdown_path = output_dir / "drawdown.csv"
            drawdown.to_csv(drawdown_path)
            
            log.info("VectorBT outputs saved to %s", output_dir)
        
        return metrics, equity, net_returns
        
    except Exception as e:
        log.error("VectorBT backtest failed: %s", e)
        import traceback
        log.error("Traceback:\n%s", traceback.format_exc())
        return VectorBTMetrics(), pd.Series(dtype=float), pd.Series(dtype=float)

def run_alphalens_analysis(
    rankings: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int = 21,
    quantiles: int = 5,
    output_dir: Path | None = None,
) -> AlphalensMetrics:
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import alphalens  # type: ignore[import-untyped]
        from alphalens.utils import get_clean_factor_and_forward_returns  # type: ignore[import-untyped]
        from alphalens.performance import (  # type: ignore[import-untyped]
            factor_information_coefficient,
            mean_return_by_quantile,
            factor_returns,
            factor_alpha_beta,
        )
    except ImportError:
        log.error("Alphalens not installed")
        return AlphalensMetrics()
    
    metrics = AlphalensMetrics()
    
    pred_cols = [c for c in rankings.columns if "pred" in c.lower()]
    if not pred_cols:
        log.warning("No prediction column found for Alphalens")
        return metrics
    
    pred_col = pred_cols[0]
    
    try:
        factor_df = rankings[["Date", "Sector", pred_col]].copy()
        factor_df = factor_df.drop_duplicates(subset=["Date", "Sector"], keep="last")
        factor_df = factor_df.set_index(["Date", "Sector"])[pred_col]
        
        unique_dates = pd.to_datetime(factor_df.index.get_level_values(0).unique())
        if isinstance(factor_df.index, pd.MultiIndex):
            factor_df.index = factor_df.index.set_levels(unique_dates, level=0)  # type: ignore[union-attr]
            factor_df.index = factor_df.index.rename(["date", "asset"])
        
        if factor_df.index.duplicated().any():
            factor_df = factor_df[~factor_df.index.duplicated(keep="last")]
        
        prices_for_al = prices.copy()
        prices_for_al.index = pd.to_datetime(prices_for_al.index)
        
        factor_data = get_clean_factor_and_forward_returns(
            factor_df,
            prices_for_al,
            periods=(horizon,),
            quantiles=quantiles,
            max_loss=0.35,
        )
        
        try:
            ic = factor_information_coefficient(factor_data)
        except TypeError as te:
            if "Timestamp" in str(te):
                log.warning("pandas compatibility")
                
                factor_col = "factor"
                fwd_ret_cols = [c for c in factor_data.columns if c.endswith("D") and c[:-1].lstrip("-").isdigit()]
                if not fwd_ret_cols:
                    fwd_ret_cols = [c for c in factor_data.columns if c != factor_col]
                fwd_ret_col = fwd_ret_cols[0] if fwd_ret_cols else factor_data.columns[1]
                
                def calc_ic_for_date(group: pd.DataFrame) -> float:
                    return group[factor_col].corr(group[fwd_ret_col], method="spearman")
                
                fd_reset = factor_data.reset_index()
                date_candidates = [c for c in fd_reset.columns if "date" in str(c).lower() or c == "level_0"]
                date_col = date_candidates[0] if date_candidates else fd_reset.columns[0]
                
                ic_series = fd_reset.groupby(date_col).apply(calc_ic_for_date, include_groups=False)
                ic = pd.DataFrame({fwd_ret_col: ic_series})
            else:
                raise
        
        ic_col = ic.columns[0]
        
        metrics.ic_mean = float(ic[ic_col].mean())
        metrics.ic_std = float(ic[ic_col].std())
        if metrics.ic_std > 0:
            metrics.ic_ir = float(metrics.ic_mean / metrics.ic_std)
            metrics.ic_t_stat = float(metrics.ic_ir * np.sqrt(len(ic)))
        metrics.ic_skew = float(ic[ic_col].skew())  # type: ignore[arg-type]
        
        try:
            qret = mean_return_by_quantile(factor_data)
            if len(qret) > 0:
                q_returns = qret[0]
                if len(q_returns) >= 2:
                    top_q = q_returns.iloc[-1].values[0]
                    bottom_q = q_returns.iloc[0].values[0]
                    metrics.quantile_spread_mean = float(top_q - bottom_q)
        except Exception as e:
            log.warning("Quantile return calculation failed: %s", e)
        
        try:
            fret = factor_returns(factor_data)
            fret_col = fret.columns[0]
            metrics.factor_return_mean = float(fret[fret_col].mean())
            if fret[fret_col].std() > 0:
                metrics.factor_return_sharpe = float(
                    fret[fret_col].mean() / fret[fret_col].std() * np.sqrt(TRADING_DAYS / horizon)
                )
        except Exception as e:
            log.warning("Factor return calculation failed: %s", e)
        
        try:
            autocorr = factor_df.unstack().T.corrwith(
                factor_df.unstack().T.shift(1)
            )
            metrics.autocorr_1d = float(autocorr.mean()) if len(autocorr) > 0 else 0.0
        except:
            pass
        
        if output_dir is not None:
            ic_path = output_dir / "ic_timeseries.csv"
            ic.to_csv(ic_path)
            
            factor_path = output_dir / "factor_data.csv"
            factor_df.to_csv(factor_path)
            
            log.info("Alphalens outputs saved to %s", output_dir)
        
        log.info("Alphalens: IC=%.4f, IR=%.2f, t-stat=%.2f", 
                 metrics.ic_mean, metrics.ic_ir, metrics.ic_t_stat)
        
    except Exception as e:
        log.error("Alphalens analysis failed: %s", e)
        import traceback
        log.error("Traceback:\n%s", traceback.format_exc())
    
    return metrics

def run_quantstats_analysis(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
    output_dir: Path | None = None,
    title: str = "Strategy",
) -> QuantStatsMetrics:
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import quantstats as qs  # type: ignore[import-untyped]
    except ImportError:
        log.error("QuantStats not installed. Run: pip install quantstats")
        return QuantStatsMetrics()
    
    metrics = QuantStatsMetrics()
    returns = returns.dropna()
    
    if len(returns) < 10:
        log.warning("Insufficient returns for QuantStats: %d", len(returns))
        return metrics
    
    try:
        metrics.total_return = float(qs.stats.comp(returns))  # type: ignore[arg-type]
        metrics.cagr = float(qs.stats.cagr(returns))  # type: ignore[arg-type]
        metrics.sharpe = float(qs.stats.sharpe(returns, rf=risk_free_rate))  # type: ignore[arg-type]
        metrics.sortino = float(qs.stats.sortino(returns, rf=risk_free_rate))  # type: ignore[arg-type]
        metrics.max_drawdown = float(qs.stats.max_drawdown(returns))  # type: ignore[arg-type]
        metrics.calmar = float(qs.stats.calmar(returns))  # type: ignore[arg-type]
        metrics.volatility = float(qs.stats.volatility(returns))  # type: ignore[arg-type]
        
        try:
            metrics.var_95 = float(qs.stats.var(returns))  # type: ignore[arg-type]
            metrics.cvar_95 = float(qs.stats.cvar(returns))  # type: ignore[arg-type]
        except:
            pass
        
        metrics.skew = float(qs.stats.skew(returns))  # type: ignore[arg-type]
        metrics.kurtosis = float(qs.stats.kurtosis(returns))  # type: ignore[arg-type]
        
        try:
            metrics.omega = float(qs.stats.omega(returns))  # type: ignore[arg-type]
            metrics.gain_to_pain = float(qs.stats.gain_to_pain_ratio(returns))  # type: ignore[arg-type]
            metrics.payoff_ratio = float(qs.stats.payoff_ratio(returns))  # type: ignore[arg-type]
        except:
            pass
        
        metrics.win_rate = float(qs.stats.win_rate(returns))  # type: ignore[arg-type]
        winning = returns[returns > 0]
        losing = returns[returns < 0]
        if len(winning) > 0:
            metrics.avg_win = float(winning.mean())
        if len(losing) > 0:
            metrics.avg_loss = float(losing.mean())
        
        if benchmark is not None and len(benchmark) > 0:
            try:
                common_idx = returns.index.intersection(benchmark.index)
                if len(common_idx) > 10:
                    ret_aligned = returns.loc[common_idx]
                    bench_aligned = benchmark.loc[common_idx]
                    
                    metrics.correlation_to_spy = float(ret_aligned.corr(bench_aligned))
                    
                    cov = np.cov(ret_aligned, bench_aligned)
                    if cov[1, 1] > 0:
                        metrics.beta = float(cov[0, 1] / cov[1, 1])
                        metrics.alpha = float(
                            ret_aligned.mean() * TRADING_DAYS - 
                            metrics.beta * bench_aligned.mean() * TRADING_DAYS
                        )
            except Exception as e:
                log.warning("Benchmark comparison failed: %s", e)
        
        if output_dir is not None:
            html_path = output_dir / "tearsheet.html"
            
            try:
                qs.reports.html(
                    returns,
                    benchmark=benchmark,
                    output=str(html_path),
                    title=title,
                    rf=risk_free_rate,
                )
                log.info("QuantStats HTML saved to %s", html_path)
            except Exception as e:
                log.warning("HTML generation failed: %s", e)
        
        log.info("QuantStats: Return=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
                 metrics.total_return * 100, metrics.sharpe, metrics.max_drawdown * 100)
        
    except Exception as e:
        log.error("QuantStats analysis failed: %s", e)
        import traceback
        log.error("Traceback:\n%s", traceback.format_exc())
    
    return metrics

def _generate_report_md(result: UnifiedBacktestResult, output_dir: Path) -> None:
    report = []
    report.append(f"Experiment Report: {result.run_id}")
    report.append(f"\nGenerated: {result.timestamp}")
    report.append(f"\nMode: {result.mode}")
    report.append(f"\nHorizon: {result.horizon} days")
    report.append(f"\nDate Range: {result.start_date} to {result.end_date}")
    report.append(f"\nTrading Days: {result.n_trading_days}")
    report.append("\nMetrics VectorBT")
    report.append("\nMetric | Value")
    report.append("---------|--------")
    report.append(f"Total Return | {result.vectorbt.total_return:.2%}")
    report.append(f"CAGR | {result.vectorbt.cagr:.2%}")
    report.append(f"Sharpe Ratio | {result.vectorbt.sharpe:.2f}")
    report.append(f"Sortino Ratio | {result.vectorbt.sortino:.2f}")
    report.append(f"Max Drawdown | {result.vectorbt.max_drawdown:.2%}")
    report.append(f"Calmar Ratio | {result.vectorbt.calmar:.2f}")
    report.append(f"Volatility | {result.vectorbt.volatility:.2%}")
    report.append(f"Win Rate | {result.vectorbt.win_rate:.2%}")
    report.append(f"Avg Exposure |{result.vectorbt.avg_exposure:.2%}")
    report.append("\n\nFactor Analysis (Alphalens)")
    report.append("\nMetric | Value | Interpretation")
    report.append("--------|-------|----------------")
    
    ic = result.alphalens.ic_mean
    ic_interp = "Strong" if abs(ic) > 0.05 else "Moderate" if abs(ic) > 0.02 else "Weak"
    report.append(f"IC Mean | {ic:.4f} | {ic_interp}")
    
    ir = result.alphalens.ic_ir
    ir_interp = "Good" if ir > 0.5 else "Moderate" if ir > 0.2 else "Poor"
    report.append(f"IC IR | {ir:.2f} | {ir_interp}")
    
    t_stat = result.alphalens.ic_t_stat
    t_interp = "Significant" if abs(t_stat) > 2 else "Marginal" if abs(t_stat) > 1.5 else "Not Significant"
    report.append(f"IC t-stat | {t_stat:.2f} | {t_interp}")
    
    spread = result.alphalens.quantile_spread_mean
    report.append(f"Q5-Q1 Spread | {spread:.4f} | -")
    
    report.append("\n\nRisk Analysis (QuantStats)")
    report.append("\nMetric | Value")
    report.append("--------|-------")
    report.append(f"Alpha | {result.quantstats.alpha:.2%}")
    report.append(f"Beta | {result.quantstats.beta:.2f}")
    report.append(f"Omega | {result.quantstats.omega:.2f}")
    report.append(f"VaR (95%) | {result.quantstats.var_95:.2%}")
    report.append(f"CVaR (95%) | {result.quantstats.cvar_95:.2%}")
    report.append(f"Skew | {result.quantstats.skew:.2f}")
    report.append(f"Kurtosis | {result.quantstats.kurtosis:.2f}")
    
    report.append("\n\nPerformance Assessment")
    
    if result.vectorbt.total_return > 0:
        report.append("\nProfitable: Strategy generated positive returns")
    else:
        report.append("\nUnprofitable: Strategy generated negative returns")
    
    if result.vectorbt.sharpe > 0.5:
        report.append("\nGood Sharpe: Risk-adjusted returns are solid")
    elif result.vectorbt.sharpe > 0:
        report.append("\nWeak Sharpe: Risk-adjusted returns are marginal")
    else:
        report.append("\nPoor Sharpe: Risk-adjusted returns are weak")
    
    if abs(result.alphalens.ic_mean) > 0.02:
        report.append("\nPredictive: Model has meaningful IC")
    elif abs(result.alphalens.ic_mean) > 0.01:
        report.append("\nMarginal IC: Model has weak predictive power")
    else:
        report.append("\nNo IC: Model predictions do not correlate with returns")
    
    if result.vectorbt.max_drawdown > -0.2:
        report.append("\nControlled Drawdown: Max drawdown below 20%")
    elif result.vectorbt.max_drawdown > -0.3:
        report.append("\nModerate Drawdown: Max drawdown between 20-30%")
    else:
        report.append("\nSevere Drawdown: Max drawdown exceeds 30%")
    
    if result.warnings:
        report.append("\n\nWarnings")
        for w in result.warnings:
            report.append(f"\n- {w}")
    
    if result.errors:
        report.append("\n\nErrors")
        for e in result.errors:
            report.append(f"\n- {e}")
    
    report_content = "\n".join(report)
    report_path = output_dir / "report.md"
    report_path.write_text(report_content, encoding="utf-8")
    
    log.info("Report saved to %s", report_path)

def run_unified_backtest(
    run_dir: Path | str,
    output_dir: Path | str | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    risk_free_rate: float = RISK_FREE_RATE,
    prices: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
) -> UnifiedBacktestResult:
    run_dir = Path(run_dir)
    
    if output_dir is None:
        output_dir = run_dir / "backtest_unified"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = UnifiedBacktestResult()
    result.run_id = run_dir.name
    result.timestamp = datetime.now().isoformat()
    result.output_dir = str(output_dir)
    
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        result.mode = config.get("filter_type", "unknown")
        result.horizon = config.get("horizon", 21)
        result.category = config.get("category", "unknown")
    
    log.info("\n\nUNIFIED BACKTEST: %s (h=%d)", result.run_id, result.horizon)
    
    try:
        if prices is None:
            prices = load_prices()
        
        if benchmark is None:
            benchmark = load_spy_returns(prices)
        
        rankings = load_fold_rankings(run_dir)
        result.n_rankings = len(rankings)
        
        weights = build_weights_from_rankings(rankings)
        result.n_weights = len(weights)
        if len(benchmark) > 0:
            benchmark_forward = benchmark.shift(-1).dropna()
            benchmark_forward.name = "SPY"
        else:
            benchmark_forward = benchmark
        
        result.start_date = str(weights.index[0].date())
        result.end_date = str(weights.index[-1].date())
        result.n_trading_days = len(weights)
        
    except Exception as e:
        result.errors.append(f"Data loading failed: {e}")
        log.error("Data loading failed: %s", e)
        return result
    
    log.info("Running VectorBT backtest...")
    
    vbt_dir = output_dir / "vectorbt"
    result.vectorbt, equity, returns = run_vectorbt_backtest(
        weights=weights,
        prices=prices,
        cost_bps=cost_bps,
        risk_free_rate=risk_free_rate,
        horizon=result.horizon,
        output_dir=vbt_dir,
    )
    result.vectorbt_equity_path = str(vbt_dir / "equity_curve.csv")
    
    log.info("Running Alphalens analysis...")
    
    al_dir = output_dir / "alphalens"
    result.alphalens = run_alphalens_analysis(
        rankings=rankings,
        prices=prices,
        horizon=result.horizon,
        output_dir=al_dir,
    )
    result.alphalens_tearsheet_path = str(al_dir / "factor_data.csv")
    
    log.info("Running QuantStats analysis...")
    
    qs_dir = output_dir / "quantstats"
    if len(returns) > 0:
        result.quantstats = run_quantstats_analysis(
            returns=returns,
            benchmark=benchmark_forward,
            risk_free_rate=risk_free_rate,
            output_dir=qs_dir,
            title=f"{result.mode} h={result.horizon}",
        )
        result.quantstats_html_path = str(qs_dir / "tearsheet.html")
    
    result.sharpe_vectorbt_vs_quantstats_diff = abs(
        result.vectorbt.sharpe - result.quantstats.sharpe
    )
    result.return_vectorbt_vs_quantstats_diff = abs(
        result.vectorbt.total_return - result.quantstats.total_return
    )
    
    if result.sharpe_vectorbt_vs_quantstats_diff > 0.1:
        result.warnings.append(
            f"Sharpe mismatch: VBT={result.vectorbt.sharpe:.2f} vs QS={result.quantstats.sharpe:.2f}"
        )
    
    result_path = output_dir / "unified_result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    
    _generate_report_md(result, output_dir)
    

    log.info("\n\nUNIFIED BACKTEST COMPLETE")
    log.info("VectorBT:  Return=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
             result.vectorbt.total_return * 100,
             result.vectorbt.sharpe,
             result.vectorbt.max_drawdown * 100)
    log.info("Alphalens: IC=%.4f, IR=%.2f",
             result.alphalens.ic_mean,
             result.alphalens.ic_ir)
    log.info("QuantStats: Sharpe=%.2f, Calmar=%.2f, Beta=%.2f",
             result.quantstats.sharpe,
             result.quantstats.calmar,
             result.quantstats.beta)
    log.info("Output: %s", output_dir)
    
    return result


def run_unified_backtest_all_horizons(
    base_run_pattern: str,
    category: str = "normal",
    output_base: Path | str | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict[int, UnifiedBacktestResult]:
    workspace_root = Path(__file__).parent.parent.parent
    runs_dir = workspace_root / "runs" / category
    
    horizon_runs = {}
    for h in HORIZONS:
        pattern = f"run_{base_run_pattern}_*"
        matching = list(runs_dir.glob(pattern))
        
        for run_dir in matching:
            config_path = run_dir / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text())
                if config.get("horizon") == h:
                    horizon_runs[h] = run_dir
                    break
    
    if not horizon_runs:
        log.error("No runs found matching pattern: %s", base_run_pattern)
        return {}
    
    prices = load_prices(workspace_root)
    benchmark = load_spy_returns(prices)
    
    if output_base is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = runs_dir / f"unified_backtest_{timestamp}"
    output_base = Path(output_base)
    
    results = {}
    
    for h, run_dir in sorted(horizon_runs.items()):
        log.info("\nProcessing horizon h=%d: %s\n", h, run_dir.name)
        
        output_dir = output_base / f"h{h}"
        
        try:
            result = run_unified_backtest(
                run_dir=run_dir,
                output_dir=output_dir,
                cost_bps=cost_bps,
                prices=prices,
                benchmark=benchmark,
            )
            results[h] = result
        except Exception as e:
            log.error("Horizon h=%d failed: %s", h, e)
    
    if results:
        summary = generate_horizon_comparison_summary(results, output_base)
        summary_path = output_base / "horizon_comparison.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        log.info("Horizon comparison saved to %s", summary_path)
    
    return results


def generate_horizon_comparison_summary(
    results: dict[int, UnifiedBacktestResult],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    summary = {
        "generated_at": datetime.now().isoformat(),
        "horizons": {},
        "best_by_metric": {},
    }
    
    for h, result in sorted(results.items()):
        summary["horizons"][h] = {
            "mode": result.mode,
            "n_days": result.n_trading_days,
            "vectorbt": {
                "return": result.vectorbt.total_return,
                "sharpe": result.vectorbt.sharpe,
                "sortino": result.vectorbt.sortino,
                "max_dd": result.vectorbt.max_drawdown,
                "calmar": result.vectorbt.calmar,
                "exposure": result.vectorbt.avg_exposure,
            },
            "alphalens": {
                "ic": result.alphalens.ic_mean,
                "ir": result.alphalens.ic_ir,
                "t_stat": result.alphalens.ic_t_stat,
                "spread": result.alphalens.quantile_spread_mean,
            },
            "quantstats": {
                "alpha": result.quantstats.alpha,
                "beta": result.quantstats.beta,
                "omega": result.quantstats.omega,
            },
        }
    
    if results:
        metrics = ["sharpe", "ic", "calmar", "return"]
        for metric in metrics:
            best_h = None
            best_val = -float("inf")
            
            for h, result in results.items():
                if metric == "sharpe":
                    val = result.vectorbt.sharpe
                elif metric == "ic":
                    val = result.alphalens.ic_mean
                elif metric == "calmar":
                    val = result.vectorbt.calmar
                elif metric == "return":
                    val = result.vectorbt.total_return
                else:
                    continue
                
                if val > best_val:
                    best_val = val
                    best_h = h
            
            summary["best_by_metric"][metric] = {"horizon": best_h, "value": best_val}
    
    return summary

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(description="Run unified backtest")
    parser.add_argument("run_dir", type=Path, help="Run directory")
    parser.add_argument("output_dir", type=Path, default=None)
    parser.add_argument("cost_bps", type=float, default=DEFAULT_COST_BPS)
    
    args = parser.parse_args()
    
    result = run_unified_backtest(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        cost_bps=args.cost_bps,
    )
    
    print(f"\nCompleted: {result.run_id}")
    print(f"Sharpe: {result.vectorbt.sharpe:.2f}")
    print(f"IC: {result.alphalens.ic_mean:.4f}")