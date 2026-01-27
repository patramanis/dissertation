from __future__ import annotations
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RUN_TRACKER_PATH = WORKSPACE_ROOT / "runs" / "run_tracker.json"

MOMENTUM_FILTERS = [
    "unfiltered", "dual", "relative", "absolute",
    "pure_momentum",
    "regime", "regime_invol",
    "regime_dynamic", "regime_scalar", "regime_streak_decay",
    "layered_adaptive",
    "vix_adaptive", "trend_vix", "vix_zscore",
]

FILTER_TO_ARG = {
    "unfiltered": "none",
    "dual": "dual",
    "relative": "relative",
    "absolute": "absolute",
    "regime": "regime",
    "regime_invol": "regime",
    "regime_dynamic": "regime_dynamic",
    "regime_scalar": "regime_scalar",
    "regime_streak_decay": "regime_streak_decay",
    "layered_adaptive": "layered_adaptive",
    "pure_momentum": "pure_momentum",
    "vix_adaptive": "vix_adaptive",
    "trend_vix": "trend_vix",
    "vix_zscore": "vix_zscore",
}

FILTER_TO_WEIGHT_METHOD = {
    "unfiltered": "equal",
    "dual": "equal",
    "relative": "equal",
    "absolute": "equal",
    "regime": "equal",
    "regime_invol": "inverse_vol",
    "regime_dynamic": "inverse_vol",
    "regime_scalar": "inverse_vol",
    "regime_streak_decay": "inverse_vol",
    "layered_adaptive": "inverse_vol",
    "pure_momentum": "equal",
    "vix_adaptive": "equal",
    "trend_vix": "equal",
    "vix_zscore": "equal",
}

FILTER_TO_RISK_OFF_ALLOCATION = {
    "unfiltered": 0.0,
    "dual": 0.0,
    "relative": 0.0,
    "absolute": 0.0,
    "regime": 0.0,
    "regime_invol": 0.0,
    "regime_dynamic": 0.0,
    "regime_scalar": 0.0,
    "regime_streak_decay": 0.0,
    "layered_adaptive": 0.0,
    "pure_momentum": 0.0,
    "vix_adaptive": 0.0,
    "trend_vix": 0.0,
    "vix_zscore": 0.0,
}

def load_run_tracker() -> dict:
    if not RUN_TRACKER_PATH.exists():
        raise FileNotFoundError(f"Run tracker not found: {RUN_TRACKER_PATH}")
    return json.loads(RUN_TRACKER_PATH.read_text(encoding="utf-8"))


def save_run_tracker(tracker: dict) -> None:
    RUN_TRACKER_PATH.write_text(
        json.dumps(tracker, indent=2, default=str), 
        encoding="utf-8"
    )

def get_next_run_number(category: str, filter_type: str) -> int:
    category_dir = WORKSPACE_ROOT / "runs" / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"run_{filter_type}_"
    existing_numbers = []
    
    for folder in category_dir.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            try:
                num_str = folder.name[len(prefix):]
                num = int(num_str)
                existing_numbers.append(num)
            except ValueError:
                continue
    
    if not existing_numbers:
        next_num = 1
    else:
        existing_set = set(existing_numbers)
        next_num = 1
        while next_num in existing_set:
            next_num += 1
    
    try:
        tracker = load_run_tracker()
        tracker["categories"][category]["filters"][filter_type] = max(existing_numbers) if existing_numbers else 0
        save_run_tracker(tracker)
    except Exception:
        pass
    
    log.info("Dynamic run numbering: category=%s, filter=%s, existing=%s, next=%d",
             category, filter_type, sorted(existing_numbers), next_num)
    
    return next_num


def get_output_dir(category: str, filter_type: str, run_number: int) -> Path:
    return WORKSPACE_ROOT / "runs" / category / f"run_{filter_type}_{run_number}"


def log_run_start(category: str, filter_type: str, run_number: int, num_folds: int) -> dict:
    tracker = load_run_tracker()
    
    run_entry = {
        "category": category,
        "filter_type": filter_type,
        "run_number": run_number,
        "num_folds": num_folds,
        "timestamp": datetime.now().isoformat(),
        "status": "started",
    }
    
    tracker["runs"].append(run_entry)
    save_run_tracker(tracker)
    
    return run_entry

def log_run_complete(category: str, filter_type: str, run_number: int, 
                     status: str = "completed", metrics: Optional[dict] = None) -> None:
    tracker = load_run_tracker()
    
    for run in tracker["runs"]:
        if (run["category"] == category and 
            run["filter_type"] == filter_type and 
            run["run_number"] == run_number):
            run["status"] = status
            run["completed_at"] = datetime.now().isoformat()
            if metrics:
                run["summary_metrics"] = metrics
            break
    
    save_run_tracker(tracker)

def run_fold(
    horizon: int,
    fold: int,
    filter_type: str,
    output_dir: Path,
    config: str = "autogluon_medium",
    log_level: str = "INFO",
    weight_method: str = "equal",
    smoothing_window: int = 1,
    risk_off_allocation: float = 0.0,
) -> dict:
    script_path = WORKSPACE_ROOT / "scripts" / "10_run_autogluon_example.py"
    filter_arg = FILTER_TO_ARG[filter_type]
    
    cmd = [
        sys.executable,
        str(script_path),
        "horizon", str(horizon),
        "cv-type", "rolling",
        "fold", str(fold),
        "config", config,
        "momentum-filter", filter_arg,
        "output-dir", str(output_dir),
        "weight-method", weight_method,
        "smoothing-window", str(smoothing_window),
        "risk-off-allocation", str(risk_off_allocation),
        "log-level", log_level,
    ]
    
    log.info("Running fold %d with filter=%s, weights=%s, smooth=%d, risk_off=%.2f", 
             fold, filter_type, weight_method, smoothing_window, risk_off_allocation)
    
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=WORKSPACE_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Fold {fold} failed with code {result.returncode}")
    
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    
    return {"error": "No metrics file found"}

def aggregate_metrics(output_dir: Path, num_folds: int) -> dict:
    import numpy as np
    
    all_metrics = []
    for fold in range(num_folds):
        fold_dir = output_dir / f"fold{fold}"
        metrics_path = fold_dir / "metrics.json"
        if metrics_path.exists():
            all_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    
    if not all_metrics:
        return {"error": "No metrics found"}
    
    keys = ["ic_mean", "sharpe_ratio", "hit_rate", "avg_positions", "cash_pct",
            "avg_excess_net", "avg_excess_gross"]
    
    agg: dict[str, int | float | str | list] = {
        "num_folds": len(all_metrics),
    }
    
    for key in keys:
        values = [m.get(key) for m in all_metrics if m.get(key) is not None]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    
    return agg

def run_experiment(
    category: str,
    filter_type: str,
    num_folds: int,
    horizon: int = 21,
    config: str = "autogluon_medium",
    log_level: str = "INFO",
    skip_data: bool = True,
    weight_method: str = "equal",
    smoothing_window: int = 1,
    risk_off_allocation: float = 0.0,
) -> dict:
    run_number = get_next_run_number(category, filter_type)
    
    base_output_dir = get_output_dir(category, filter_type, run_number)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Starting %s run #%d for filter=%s", category.upper(), run_number, filter_type)
    log.info("Output: %s", base_output_dir)
    log.info("Folds: %d, Horizon: %d, Config: %s", num_folds, horizon, config)
    
    log_run_start(category, filter_type, run_number, num_folds)
    
    run_config = {
        "category": category,
        "filter_type": filter_type,
        "run_number": run_number,
        "num_folds": num_folds,
        "horizon": horizon,
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "weight_method": weight_method,
        "smoothing_window": smoothing_window,
        "risk_off_allocation": risk_off_allocation,
    }
    (base_output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    
    fold_metrics = []
    failed_folds = []
    
    for fold in range(num_folds):
        fold_output_dir = base_output_dir / f"fold{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            metrics = run_fold(
                horizon=horizon,
                fold=fold,
                filter_type=filter_type,
                output_dir=fold_output_dir,
                config=config,
                log_level=log_level,
                weight_method=weight_method,
                smoothing_window=smoothing_window,
                risk_off_allocation=risk_off_allocation,
            )
            fold_metrics.append(metrics)
            log.info("Fold %d completed: Sharpe=%.2f, IC=%.4f", 
                     fold, metrics.get("sharpe_ratio", 0), metrics.get("ic_mean", 0))
        except Exception as e:
            log.error("Fold %d failed: %s", fold, e)
            failed_folds.append(fold)
    
    agg_metrics = aggregate_metrics(base_output_dir, num_folds)
    agg_metrics["failed_folds"] = failed_folds
    agg_metrics["successful_folds"] = num_folds - len(failed_folds)
    
    (base_output_dir / "summary.json").write_text(
        json.dumps(agg_metrics, indent=2), encoding="utf-8"
    )
    
    status = "completed" if not failed_folds else f"completed_with_errors ({len(failed_folds)} failed)"
    log_run_complete(category, filter_type, run_number, status, agg_metrics)
    
    log.info("%s run #%d COMPLETE for filter=%s", category.upper(), run_number, filter_type)
    log.info("Successful folds: %d/%d", agg_metrics["successful_folds"], num_folds)
    if "sharpe_ratio_mean" in agg_metrics:
        log.info("Mean Sharpe: %.2f ± %.2f", 
                 agg_metrics["sharpe_ratio_mean"], agg_metrics.get("sharpe_ratio_std", 0))
    if "ic_mean_mean" in agg_metrics:
        log.info("Mean IC: %.4f ± %.4f",
                 agg_metrics["ic_mean_mean"], agg_metrics.get("ic_mean_std", 0))
    
    if agg_metrics.get("successful_folds", 0) > 0:
        log.info("Running post-experiment external validation...")
        try:
            from run_post_backtest import run_post_experiment_backtest, generate_experiment_report
            
            result = run_post_experiment_backtest(base_output_dir)
            generate_experiment_report(base_output_dir, result)
            
            agg_metrics["external_validation"] = {
                "vectorbt_sharpe": result.vectorbt.sharpe,
                "vectorbt_return": result.vectorbt.total_return,
                "vectorbt_max_dd": result.vectorbt.max_drawdown,
                "alphalens_ic": result.alphalens.ic_mean,
                "alphalens_ir": result.alphalens.ic_ir,
                "quantstats_alpha": result.quantstats.alpha,
                "quantstats_beta": result.quantstats.beta,
            }
            
            log.info("EXTERNAL VALIDATION RESULTS:")
            log.info("VectorBT Sharpe: %.2f", result.vectorbt.sharpe)
            log.info("VectorBT Return: %.2f%%", result.vectorbt.total_return * 100)
            log.info("VectorBT MaxDD: %.2f%%", result.vectorbt.max_drawdown * 100)
            log.info("Alphalens IC: %.4f", result.alphalens.ic_mean)
            log.info("Alphalens IR: %.2f", result.alphalens.ic_ir)
            
        except ImportError:
            log.warning("Post-backtest module not available. Skipping external validation.")
        except Exception as e:
            log.error("External validation failed: %s", e)
    
    return agg_metrics

ALL_HORIZONS = [5, 21, 63]

FOLDS_PER_HORIZON = {
    5: 39,
    21: 39,
    63: 39,
}

SMOOTHING_PER_HORIZON = {
    5: 2,
    21: 5,  
    63: 15,
}

def run_all_horizons(
    category: str,
    filter_type: str,
    num_folds: int | None = None,
    horizons: list[int] | None = None,
    config: str = "autogluon_medium",
    log_level: str = "INFO",
    weight_method: str | None = None,
    smoothing_window: int | None = None,
    risk_off_allocation: float | None = None,
) -> dict[int, dict]:
    if horizons is None:
        horizons = ALL_HORIZONS
    if weight_method is None:
        weight_method = FILTER_TO_WEIGHT_METHOD.get(filter_type, "equal")
    if risk_off_allocation is None:
        risk_off_allocation = FILTER_TO_RISK_OFF_ALLOCATION.get(filter_type, 0.0)
    
    log.info("RUNNING ALL HORIZONS for %s - %s", category.upper(), filter_type)
    log.info("Horizons: %s", horizons)
    log.info("Weight method: %s", weight_method)
    if risk_off_allocation > 0:
        log.info("Risk-OFF allocation: %.0f%% (SOFT GATING)", risk_off_allocation * 100)
    else:
        log.info("Risk-OFF allocation: 0%% (BINARY)")
    if smoothing_window is None:
        log.info("Smoothing: horizon-specific (h5=%d, h21=%d, h63=%d)", 
                 SMOOTHING_PER_HORIZON[5], SMOOTHING_PER_HORIZON[21], SMOOTHING_PER_HORIZON[63])
    else:
        log.info("Smoothing: %d (fixed)", smoothing_window)

    results = {}
    
    for h in horizons:
        folds = num_folds if num_folds is not None else FOLDS_PER_HORIZON.get(h, 39)
        
        smooth = smoothing_window if smoothing_window is not None else SMOOTHING_PER_HORIZON.get(h, 1)
        log.info("\nStarting horizon h=%d with %d folds, smoothing=%d <<<\n", h, folds, smooth)
        
        try:
            metrics = run_experiment(
                category=category,
                filter_type=filter_type,
                num_folds=folds,
                horizon=h,
                config=config,
                log_level=log_level,
                weight_method=weight_method,
                smoothing_window=smooth,
                risk_off_allocation=risk_off_allocation,
            )
            results[h] = metrics
        except Exception as e:
            log.error("Horizon h=%d FAILED: %s", h, e)
            results[h] = {"error": str(e)}
    
    log.info("SUMMARY - ALL HORIZONS for %s - %s", category.upper(), filter_type)
    log.info("\nNOTE: Internal Sharpe uses overlapping returns (DIAGNOSTIC ONLY)")
    log.info("\nVectorBT Sharpe is GROUND TRUTH (daily mark-to-market)")

    for h, metrics in results.items():
        if "error" in metrics:
            log.info("  h=%d: FAILED - %s", h, metrics["error"])
        else:
            ext_val = metrics.get("external_validation", {})
            vbt_sharpe = ext_val.get("vectorbt_sharpe")
            al_ic = ext_val.get("alphalens_ic")
            
            if vbt_sharpe is not None and al_ic is not None:
                log.info("  h=%d: VectorBT Sharpe=%.2f, Alphalens IC=%.4f (GROUND TRUTH)", h, vbt_sharpe, al_ic)
            else:
                sharpe = metrics.get("sharpe_ratio_mean", 0)
                ic = metrics.get("ic_mean_mean", 0)
                log.info("  h=%d: Internal Sharpe=%.2f, IC=%.4f (OVERLAPPING - UNRELIABLE)", h, sharpe, ic)
    log.info("=" * 80)
    
    return results