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

TEST_STRATEGIES = [
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


def run_all_test_strategies():    
    results = {}
    failed = []
    start_time = time.time()
    
    log.info("AGGREGATE TEST RUNNER - Starting %d strategies", len(TEST_STRATEGIES))
    
    for i, strategy in enumerate(TEST_STRATEGIES, 1):
        log.info("STRATEGY %d/%d: %s", i, len(TEST_STRATEGIES), strategy.upper())
        
        strategy_start = time.time()
        
        try:
            result = run_all_horizons(
                filter_type=strategy,
                category="test",
                num_folds=5,
                config="autogluon_medium",
            )
            results[strategy] = result
            
            strategy_time = time.time() - strategy_start
            log.info("%s completed in %.1f minutes", strategy.upper(), strategy_time / 60)
            
        except Exception as e:
            log.error("%s FAILED: %s", strategy.upper(), e)
            failed.append(strategy)
            results[strategy] = {"error": str(e)}
    
    total_time = time.time() - start_time
    
    summary = {
        "category": "test",
        "timestamp": datetime.now().isoformat(),
        "total_strategies": len(TEST_STRATEGIES),
        "successful": len(TEST_STRATEGIES) - len(failed),
        "failed": failed,
        "total_runtime_minutes": total_time / 60,
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
                        "alphalens_ic": ext.get("alphalens_ic"),
                        "alphalens_ir": ext.get("alphalens_ir"),
                    }
            summary["results"][strategy] = strategy_summary
        else:
            summary["results"][strategy] = {"error": result["error"]}
    
    summary_path = WORKSPACE_ROOT / "runs" / "test" / "aggregate_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    
    log.info("AGGREGATE TEST RUNNER - COMPLETE")
    log.info("Total runtime: %.1f minutes", total_time / 60)
    log.info("Successful: %d/%d", len(TEST_STRATEGIES) - len(failed), len(TEST_STRATEGIES))
    if failed:
        log.info("Failed: %s", failed)
    log.info("Summary saved to: %s", summary_path)
    
    log.info("COMPARISON TABLE (h=21 horizon, VectorBT)")
    log.info("%-20s | %10s | %10s | %10s | %10s", 
             "Strategy", "Sharpe", "Return%", "MaxDD%", "IC")
    
    for strategy in TEST_STRATEGIES:
        if strategy in summary["results"]:
            h21 = summary["results"][strategy].get("h21", {})
            if h21 and "error" not in h21:
                sharpe = h21.get("vectorbt_sharpe")
                ret = h21.get("vectorbt_return")
                dd = h21.get("vectorbt_max_dd")
                ic = h21.get("alphalens_ic")
                
                log.info("%-20s | %10s | %10s | %10s | %10s",
                         strategy,
                         f"{sharpe:.2f}" if sharpe else "N/A",
                         f"{ret*100:.1f}" if ret else "N/A",
                         f"{dd*100:.1f}" if dd else "N/A",
                         f"{ic:.4f}" if ic else "N/A")
            else:
                log.info("%-20s | %10s", strategy, "ERROR")
    
    return summary

if __name__ == "__main__":
    run_all_test_strategies()