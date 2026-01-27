from __future__ import annotations
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtests.unified_backtest_runner import (
    run_unified_backtest,
    run_unified_backtest_all_horizons,
    load_prices,
    load_spy_returns,
    UnifiedBacktestResult,
    VectorBTMetrics,
    AlphalensMetrics,
    QuantStatsMetrics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_post_experiment_backtest(
    run_dir: Path | str,
    cost_bps: float = 10.0,
) -> UnifiedBacktestResult:
    run_dir = Path(run_dir)
    log.info("\n\nPOST-EXPERIMENT BACKTEST: %s", run_dir.name)

    
    result = run_unified_backtest(
        run_dir=run_dir,
        cost_bps=cost_bps,
    )
    
    update_run_summary_with_external_metrics(run_dir, result)
    
    return result


def update_run_summary_with_external_metrics(
    run_dir: Path,
    result: UnifiedBacktestResult,
) -> None:
    summary_path = run_dir / "summary.json"
    
    if not summary_path.exists():
        log.warning("Summary not found: %s", summary_path)
        return
    
    summary = json.loads(summary_path.read_text())
    
    summary["external_validation"] = {
        "timestamp": result.timestamp,
        "n_trading_days": result.n_trading_days,
        "date_range": [result.start_date, result.end_date],
        
        "vectorbt": {
            "total_return": result.vectorbt.total_return,
            "cagr": result.vectorbt.cagr,
            "sharpe": result.vectorbt.sharpe,
            "sortino": result.vectorbt.sortino,
            "max_drawdown": result.vectorbt.max_drawdown,
            "calmar": result.vectorbt.calmar,
            "volatility": result.vectorbt.volatility,
            "win_rate": result.vectorbt.win_rate,
            "avg_exposure": result.vectorbt.avg_exposure,
        },
        
        "alphalens": {
            "ic_mean": result.alphalens.ic_mean,
            "ic_std": result.alphalens.ic_std,
            "ic_ir": result.alphalens.ic_ir,
            "ic_t_stat": result.alphalens.ic_t_stat,
            "quantile_spread": result.alphalens.quantile_spread_mean,
            "factor_sharpe": result.alphalens.factor_return_sharpe,
        },
        
        "quantstats": {
            "total_return": result.quantstats.total_return,
            "sharpe": result.quantstats.sharpe,
            "sortino": result.quantstats.sortino,
            "max_drawdown": result.quantstats.max_drawdown,
            "alpha": result.quantstats.alpha,
            "beta": result.quantstats.beta,
            "omega": result.quantstats.omega,
            "var_95": result.quantstats.var_95,
        },
        
        "discrepancies": {
            "sharpe_diff_vbt_vs_qs": result.sharpe_vectorbt_vs_quantstats_diff,
            "return_diff_vbt_vs_qs": result.return_vectorbt_vs_quantstats_diff,
        },
        
        "report_paths": {
            "vectorbt_equity": result.vectorbt_equity_path,
            "quantstats_html": result.quantstats_html_path,
            "alphalens_data": result.alphalens_tearsheet_path,
        },
        
        "warnings": result.warnings,
        "errors": result.errors,
    }
    
    internal_sharpe = summary.get("sharpe_ratio_mean", 0)
    external_sharpe = result.vectorbt.sharpe
    
    if abs(internal_sharpe - external_sharpe) > 0.3:
        summary["external_validation"]["DISCREPANCY_WARNING"] = (
            f"Internal Sharpe ({internal_sharpe:.2f}) differs significantly from "
            f"VectorBT Sharpe ({external_sharpe:.2f}). VectorBT is ground truth."
        )
    
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    
    log.info("Updated summary with external validation: %s", summary_path)

def generate_experiment_report(
    run_dir: Path | str,
    result: UnifiedBacktestResult,
) -> str:

    report = []
    report.append(f"Experiment Report: {result.run_id}")
    report.append(f"\nGenerated: {result.timestamp}")
    report.append(f"\nMode: {result.mode}")
    report.append(f"\nHorizon: {result.horizon} days")
    report.append(f"\nDate Range: {result.start_date} to {result.end_date}")
    report.append(f"\nTrading Days: {result.n_trading_days}")
    
    report.append("\n\nMetrics VectorBT")
    report.append("\nMetric | Value")
    report.append("--------|-------")
    report.append(f"Total Return | {result.vectorbt.total_return:.2%}")
    report.append(f"CAGR | {result.vectorbt.cagr:.2%}")
    report.append(f"Sharpe Ratio | {result.vectorbt.sharpe:.2f}")
    report.append(f"Sortino Ratio | {result.vectorbt.sortino:.2f}")
    report.append(f"Max Drawdown | {result.vectorbt.max_drawdown:.2%}")
    report.append(f"Calmar Ratio | {result.vectorbt.calmar:.2f}")
    report.append(f"Volatility | {result.vectorbt.volatility:.2%}")
    report.append(f"Win Rate | {result.vectorbt.win_rate:.2%}")
    report.append(f"Avg Exposure | {result.vectorbt.avg_exposure:.2%}")
    
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
    report.append(f"Q5-Q1 Spread | {spread:.4f}")
    
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
    
    report.append("\n\n## Performance Assessment")
    
    if result.vectorbt.total_return > 0:
        report.append("\nProfitable: Strategy generated positive returns")
    else:
        report.append("\nUnprofitable: Strategy generated negative returns")
    
    if result.vectorbt.sharpe > 1.0:
        report.append("\nGood Sharpe: Risk-adjusted returns are strong")
    elif result.vectorbt.sharpe > 0.5:
        report.append("\nModerate Sharpe: Risk-adjusted returns are acceptable")
    else:
        report.append("\nPoor Sharpe: Risk-adjusted returns are weak")
    
    if result.alphalens.ic_mean > 0.03 and result.alphalens.ic_t_stat > 2:
        report.append("\nSignificant IC: Model predictions have predictive power")
    elif result.alphalens.ic_mean > 0.01:
        report.append("\nWeak IC: Model has some predictive power but limited")
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
    
    report.append("\n\n")
    report.append(f"\n*Report generated by unified_backtest_runner")
    
    report_content = "\n".join(report)
    
    run_dir = Path(run_dir)
    report_path = run_dir / "backtest_unified" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    log.info("Report saved to %s", report_path)
    return report_content


def run_all_pending_backtests(
    category: str = "normal",
    filter_type: str | None = None,
    force: bool = False,
) -> dict[str, UnifiedBacktestResult]:
    workspace_root = PROJECT_ROOT
    runs_dir = workspace_root / "runs" / category
    
    if not runs_dir.exists():
        log.error("Runs directory not found: %s", runs_dir)
        return {}
    
    run_dirs = [d for d in runs_dir.iterdir() 
                if d.is_dir() and d.name.startswith("run_")]
    
    if filter_type:
        run_dirs = [d for d in run_dirs if filter_type in d.name]
    
    results = {}
    
    for run_dir in sorted(run_dirs):
        summary_path = run_dir / "summary.json"
        
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text())
            if "external_validation" in summary:
                log.info("Skipping %s (already has external validation)", run_dir.name)
                continue
        
        fold_dirs = list(run_dir.glob("fold*"))
        if not fold_dirs:
            log.warning("Skipping %s (no fold results)", run_dir.name)
            continue
        
        try:
            result = run_post_experiment_backtest(run_dir)
            generate_experiment_report(run_dir, result)
            results[run_dir.name] = result
        except Exception as e:
            log.error("Failed for %s: %s", run_dir.name, e)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run post-experiment backtest with VectorBT, Alphalens, QuantStats"
    )
    
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Run directory to backtest",
    )
    
    parser.add_argument(
        "all",
        action="store_true",
        help="Run backtests for all pending runs",
    )
    
    parser.add_argument(
        "category",
        type=str,
        default="normal",
        choices=["test", "normal", "maximum"],
        help="Run category",
    )
    
    parser.add_argument(
        "filter-type",
        type=str,
        default=None,
        help="Filter to specific filter type (e.g., regime_scalar)",
    )
    
    parser.add_argument(
        "force",
        action="store_true",
        help="Force re-run even if external validation exists",
    )
    
    parser.add_argument(
        "cost-bps",
        type=float,
        default=10.0,
        help="Transaction cost in basis points",
    )
    
    args = parser.parse_args()
    
    if args.all:
        results = run_all_pending_backtests(
            category=args.category,
            filter_type=args.filter_type,
            force=args.force,
        )
        print(f"\nCompleted {len(results)} backtests")
    elif args.run_dir:
        result = run_post_experiment_backtest(
            run_dir=args.run_dir,
            cost_bps=args.cost_bps,
        )
        generate_experiment_report(args.run_dir, result)
        print(f"\nCompleted: {result.run_id}")
        print(f"Sharpe (VectorBT): {result.vectorbt.sharpe:.2f}")
        print(f"IC (Alphalens): {result.alphalens.ic_mean:.4f}")
        print(f"Max DD: {result.vectorbt.max_drawdown:.2%}")
    else:
        parser.print_help()