from __future__ import annotations
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))

def clear_pycache():
    pycache_dirs = [
        WORKSPACE_ROOT / "src" / "backtests" / "__pycache__",
        WORKSPACE_ROOT / "src" / "training" / "__pycache__",
        WORKSPACE_ROOT / "src" / "momentum" / "__pycache__",
        WORKSPACE_ROOT / "scripts" / "__pycache__",
    ]
    for cache_dir in pycache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared: {cache_dir}")

clear_pycache()

from run_common import run_all_horizons, FILTER_TO_ARG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


MAXIMUM_STRATEGIES = [
    "unfiltered",
    "relative",
    "absolute", 
    "dual",
    "regime",
    "regime_invol",
    "regime_dynamic",
    "regime_scalar",
    "regime_streak_decay",
    "layered_adaptive",
    "pure_momentum",
    "vix_adaptive",
    "trend_vix",
    "vix_zscore",
]

def run_all_maximum_strategies():
    results = {}
    failed = []
    start_time = time.time()
    log.info("AGGREGATE MAXIMUM RUNNER - Starting %d strategies", len(MAXIMUM_STRATEGIES))
    
    for i, strategy in enumerate(MAXIMUM_STRATEGIES, 1):
        log.info("STRATEGY %d/%d: %s", i, len(MAXIMUM_STRATEGIES), strategy.upper())
  
        strategy_start = time.time()
        
        try:
            result = run_all_horizons(
                filter_type=strategy,
                category="maximum",
                num_folds=39,
                config="autogluon_extreme",
            )
            results[strategy] = result
            
            strategy_time = time.time() - strategy_start
            log.info("%s completed in %.1f hours", strategy.upper(), strategy_time / 3600)
            
        except Exception as e:
            log.error("%s FAILED: %s", strategy.upper(), e)
            failed.append(strategy)
            results[strategy] = {"error": str(e)}
            
            _save_intermediate_results(results, failed, start_time)
    
    total_time = time.time() - start_time
    
    summary = {
        "category": "maximum",
        "timestamp": datetime.now().isoformat(),
        "total_strategies": len(MAXIMUM_STRATEGIES),
        "successful": len(MAXIMUM_STRATEGIES) - len(failed),
        "failed": failed,
        "total_runtime_hours": total_time / 3600,
        "model_config": "autogluon_extreme",
        "num_folds_per_horizon": 39,
        "horizons": [5, 21, 63],
        "results": {},
    }
    
    for strategy, result in results.items():
        if "error" not in result:
            strategy_summary = {}
            for h, metrics in result.items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    ext = metrics.get("external_validation", {})
                    strategy_summary[f"h{h}"] = {
                        "vectorbt_sharpe": ext.get("vectorbt_sharpe"),
                        "vectorbt_return": ext.get("vectorbt_return"),
                        "vectorbt_max_dd": ext.get("vectorbt_max_dd"),
                        "vectorbt_sortino": ext.get("vectorbt_sortino"),
                        "vectorbt_calmar": ext.get("vectorbt_calmar"),
                        "alphalens_ic": ext.get("alphalens_ic"),
                        "alphalens_ir": ext.get("alphalens_ir"),
                        "quantstats_alpha": ext.get("quantstats_alpha"),
                        "quantstats_beta": ext.get("quantstats_beta"),
                        "internal_sharpe": metrics.get("sharpe_ratio_mean"),
                        "internal_ic": metrics.get("ic_mean_mean"),
                    }
            summary["results"][strategy] = strategy_summary
        else:
            summary["results"][strategy] = {"error": result["error"]}
    
    summary_path = WORKSPACE_ROOT / "runs" / "maximum" / "aggregate_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    
    _save_comparison_csv(summary, "maximum")
    
    log.info("AGGREGATE MAXIMUM RUNNER - COMPLETE")
    log.info("Total runtime: %.1f hours", total_time / 3600)
    log.info("Successful: %d/%d", len(MAXIMUM_STRATEGIES) - len(failed), len(MAXIMUM_STRATEGIES))
    if failed:
        log.info("Failed: %s", failed)
    log.info("Summary saved to: %s", summary_path)
    log.info("Comparison CSV: %s", WORKSPACE_ROOT / "runs" / "maximum" / "comparison_table.csv")
    log.info("LaTeX table: %s", WORKSPACE_ROOT / "runs" / "maximum" / "results_table.tex")
    
    _print_comparison_table(summary, MAXIMUM_STRATEGIES)
    _print_sharpe_ranking(summary)
    
    return summary


def _save_intermediate_results(results: dict, failed: list, start_time: float):
    intermediate = {
        "timestamp": datetime.now().isoformat(),
        "completed": list(results.keys()),
        "failed": failed,
        "runtime_hours": (time.time() - start_time) / 3600,
    }
    path = WORKSPACE_ROOT / "runs" / "maximum" / "intermediate_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(intermediate, indent=2, default=str), encoding="utf-8")
    log.info("Intermediate results saved to: %s", path)


def _save_comparison_csv(summary: dict, category: str):
    import pandas as pd
    
    rows = []
    for strategy, data in summary.get("results", {}).items():
        if isinstance(data, dict) and "error" not in data:
            for horizon_key, metrics in data.items():
                if isinstance(metrics, dict):
                    row = {
                        "strategy": strategy,
                        "horizon": horizon_key,
                        **metrics
                    }
                    rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        csv_path = WORKSPACE_ROOT / "runs" / category / "comparison_table.csv"
        df.to_csv(csv_path, index=False)
        log.info("Comparison CSV saved to: %s", csv_path)

def _print_comparison_table(summary: dict, strategies: list):
    log.info("COMPARISON TABLE (VectorBT)")
    log.info("%-20s | %8s | %10s | %10s | %10s | %10s | %10s", 
             "Strategy", "Horizon", "Sharpe", "Return%", "MaxDD%", "IC", "Alpha")
    
    for strategy in strategies:
        if strategy in summary["results"]:
            data = summary["results"][strategy]
            if "error" not in data:
                for h in ["h5", "h21", "h63"]:
                    metrics = data.get(h, {})
                    if metrics:
                        sharpe = metrics.get("vectorbt_sharpe")
                        ret = metrics.get("vectorbt_return")
                        dd = metrics.get("vectorbt_max_dd")
                        ic = metrics.get("alphalens_ic")
                        alpha = metrics.get("quantstats_alpha")
                        
                        log.info("%-20s | %8s | %10s | %10s | %10s | %10s | %10s",
                                 strategy if h == "h5" else "",
                                 h,
                                 f"{sharpe:.2f}" if sharpe else "N/A",
                                 f"{ret*100:.1f}" if ret else "N/A",
                                 f"{dd*100:.1f}" if dd else "N/A",
                                 f"{ic:.4f}" if ic else "N/A",
                                 f"{alpha:.4f}" if alpha else "N/A")
            else:
                log.info("%-20s | %8s", strategy, "ERROR")

def _print_sharpe_ranking(summary: dict):
    log.info("SHARPE RANKING (h=21 horizon, VectorBT)")
    
    rankings = []
    for strategy, data in summary.get("results", {}).items():
        if isinstance(data, dict) and "error" not in data:
            h21 = data.get("h21", {})
            if h21:
                sharpe = h21.get("vectorbt_sharpe")
                if sharpe is not None:
                    rankings.append((strategy, sharpe))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (strategy, sharpe) in enumerate(rankings, 1):
        medal = "1ST" if i == 1 else "2ND" if i == 2 else "3RD" if i == 3 else "  "
        log.info("%s %2d. %-25s Sharpe = %.2f", medal, i, strategy, sharpe)
    
    if rankings:
        best = rankings[0]
        log.info("BEST STRATEGY: %s with Sharpe = %.2f", best[0], best[1])

if __name__ == "__main__":
    run_all_maximum_strategies()