from __future__ import annotations
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtests.run_backtest_suite import BacktestOrchestrator
from src.backtests.contracts import SECTORS
from src.backtests.vectorbt_runner import run_vectorbt
from src.backtests.quantstats_runner import run_quantstats
from src.backtests.alphalens_runner import run_alphalens, prepare_factor_for_alphalens
from src.backtests.weights_builder import build_weights_from_rankings
from src.backtests.cost_model import compute_portfolio_returns


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_prices() -> pd.DataFrame:
    candidates = [
        Path("data/raw/SPDR.csv"),
        Path("data/interim/aligned_pit/SPDR.csv"),
        Path("data/processed/SPDR.csv"),
    ]
    
    for path in candidates:
        if path.exists():
            prices = pd.read_csv(path, index_col=0, parse_dates=True)
            available = [s for s in SECTORS if s in prices.columns]
            return prices[available]
    
    raise FileNotFoundError("No price data found")


def load_all_fold_rankings(run_dir: Path) -> pd.DataFrame:
    all_dfs = []
    
    fold_dirs = sorted(run_dir.glob("fold*"), key=lambda x: int(x.name.replace("fold", "")))
    
    for fold_dir in fold_dirs:
        rankings_path = fold_dir / "rankings.csv"
        
        if not rankings_path.exists():
            continue
        
        try:
            df = pd.read_csv(rankings_path, parse_dates=["Date"])
            fold_num = int(fold_dir.name.replace("fold", ""))
            df["fold"] = fold_num
            all_dfs.append(df)
        except Exception as e:
            log.warning("Failed to load %s: %s", rankings_path, e)
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    combined = combined.sort_values(["Date", "Sector", "fold"])
    combined = combined.drop_duplicates(subset=["Date", "Sector"], keep="first")
    
    return combined


def run_backtest_for_run(
    run_dir: Path,
    prices: pd.DataFrame,
    output_dir: Path,
    cost_bps: float = 10.0,
    rf_rate: float = 0.025,
) -> dict:    
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    horizon = config.get("horizon", 21)
    mode = config.get("filter_type", "unknown")
    
    log.info("Running backtest for %s (h=%d)", run_dir.name, horizon)
    
    rankings = load_all_fold_rankings(run_dir)
    if rankings.empty:
        log.error("No rankings found")
        return {"error": "no_rankings"}
    
    log.info("Loaded %d rankings for %d dates", len(rankings), rankings["Date"].nunique())
    
    weights = build_weights_from_rankings(rankings)
    log.info("Built weights: %d dates", len(weights))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "run_dir": str(run_dir),
        "mode": mode,
        "horizon": horizon,
        "n_dates": len(weights),
        "date_range": [str(weights.index[0].date()), str(weights.index[-1].date())],
    }
    
    log.info("Running VectorBT analysis...")
    
    vbt_dir = output_dir / "vectorbt"
    vbt_dir.mkdir(exist_ok=True)
    
    try:
        vbt_result = run_vectorbt(
            weights=weights,
            prices=prices,
            transaction_cost_bps=cost_bps,
            risk_free_rate=rf_rate,
            output_dir=vbt_dir,
            mode=mode,
            horizon=horizon,
        )
        
        results["vectorbt"] = {
            "total_return": vbt_result.total_return,
            "cagr": vbt_result.cagr,
            "sharpe": vbt_result.sharpe,
            "sortino": vbt_result.sortino,
            "max_drawdown": vbt_result.max_drawdown,
            "calmar": vbt_result.calmar,
        }
        
        log.info("VectorBT: Return=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
                vbt_result.total_return * 100,
                vbt_result.sharpe,
                vbt_result.max_drawdown * 100)
        
        log.info("Running QuantStats analysis...")
        
        qs_dir = output_dir / "quantstats"
        qs_dir.mkdir(exist_ok=True)
        
        if hasattr(vbt_result, 'returns') and len(vbt_result.returns) > 0:
            run_quantstats(
                returns=vbt_result.returns,
                output_dir=qs_dir,
                benchmark=None,
                title=f"{mode} H{horizon}",
            )
            results["quantstats"] = {"status": "completed", "output_dir": str(qs_dir)}
            log.info("QuantStats report generated (using VectorBT returns)")
        else:
            results["quantstats"] = {"error": "no_vbt_returns"}
                
    except Exception as e:
        log.error("VectorBT failed: %s", e)
        results["vectorbt"] = {"error": str(e)}
        results["quantstats"] = {"error": "vbt_failed"}

    log.info("Running Alphalens analysis...")
    
    al_dir = output_dir / "alphalens"
    al_dir.mkdir(exist_ok=True)
    
    try:
        pred_cols = [c for c in rankings.columns if "pred" in c.lower()]
        
        if pred_cols:
            factor = prepare_factor_for_alphalens(
                rankings,
                factor_col=pred_cols[0],
            )
            
            al_result = run_alphalens(
                factor=factor,
                prices=prices,
                periods=(horizon,),
                output_dir=al_dir,
            )
            
            results["alphalens"] = al_result
            log.info("Alphalens analysis completed")
        else:
            log.warning("No prediction columns found for Alphalens")
            results["alphalens"] = {"error": "no_pred_columns"}
            
    except Exception as e:
        log.error("Alphalens failed: %s", e)
        results["alphalens"] = {"error": str(e)}
    
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    log.info("Results saved to %s", results_path)
    
    return results


def main():   
    runs = [
        ("h=5", Path("runs/normal/run_regime_scalar_5")),
        ("h=21", Path("runs/normal/run_regime_scalar_6")),
        ("h=63", Path("runs/normal/run_regime_scalar_7")),
    ]
    
    log.info("Loading prices...")
    prices = load_prices()
    log.info("Loaded prices: %d dates, %d sectors", len(prices), len(prices.columns))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(f"runs/normal/backtest_suite_{timestamp}")
    base_output.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for name, run_dir in runs:
        if not run_dir.exists():
            log.warning("Run directory not found: %s", run_dir)
            continue
        
        output_dir = base_output / run_dir.name
        
        try:
            result = run_backtest_for_run(
                run_dir=run_dir,
                prices=prices,
                output_dir=output_dir,
            )
            all_results[name] = result
        except Exception as e:
            log.error("Failed for %s: %s", name, e)
            all_results[name] = {"error": str(e)}
    
    log.info("\n\nBACKTEST SUITE COMPLETE")

    for name, result in all_results.items():
        if "error" in result:
            log.info("%s: ERROR - %s", name, result["error"])
        elif "vectorbt" in result and "error" not in result["vectorbt"]:
            vbt = result["vectorbt"]
            log.info("%s: Return=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
                    name,
                    vbt.get("total_return", 0) * 100,
                    vbt.get("sharpe", 0),
                    vbt.get("max_drawdown", 0) * 100)
    
    log.info("Results saved to: %s", base_output)


if __name__ == "__main__":
    main()