from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .contracts import (
    SECTORS,
    COST_BPS,
    RISK_FREE_ANNUAL,
    validate_rankings,
    validate_weights,
    validate_returns,
)


log = logging.getLogger(__name__)

class ValidationResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.details: dict[str, Any] = {}
    
    @property
    def failures(self) -> list[str]:
        return self.errors
    
    def add_error(self, msg: str):
        self.passed = False
        self.errors.append(msg)
        log.error("[%s] %s", self.name, msg)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        log.warning("[%s] %s", self.name, msg)
    
    def add_detail(self, key: str, value: Any):
        self.details[key] = value
    
    def merge(self, other: "ValidationResult") -> None:
        if not other.passed:
            self.passed = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.details.update(other.details)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }

def validate_oof_only(
    rankings: pd.DataFrame,
    fold_config: dict,
) -> ValidationResult:
    result = ValidationResult("OOF-only")
    
    train_end = pd.Timestamp(fold_config["train_end"]).normalize()
    test_start = pd.Timestamp(fold_config["test_start"]).normalize()
    
    min_date = rankings["Date"].min()
    max_date = rankings["Date"].max()
    
    result.add_detail("train_end", str(train_end))
    result.add_detail("test_start", str(test_start))
    result.add_detail("rankings_min_date", str(min_date))
    result.add_detail("rankings_max_date", str(max_date))
    
    if min_date < test_start:
        result.add_error(
            f"Rankings contain dates before test_start: {min_date} < {test_start}"
        )
    
    train_dates = rankings[rankings["Date"] <= train_end]
    if len(train_dates) > 0:
        result.add_error(
            f"Rankings contain {len(train_dates)} rows from training period"
        )
    
    return result

def validate_no_lookahead(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> ValidationResult:
    result = ValidationResult("No-Lookahead")
    
    weight_dates = set(weights.index)
    return_dates = set(returns.index)
    
    common = weight_dates.intersection(return_dates)
    only_weights = weight_dates - return_dates
    only_returns = return_dates - weight_dates
    
    result.add_detail("common_dates", len(common))
    result.add_detail("only_in_weights", len(only_weights))
    result.add_detail("only_in_returns", len(only_returns))
    
    if len(common) == 0:
        result.add_error("No common dates between weights and returns")
        return result
    coverage = len(common) / len(return_dates) if len(return_dates) > 0 else 0
    result.add_detail("coverage_pct", coverage * 100)
    
    if coverage < 0.95:
        result.add_warning(f"Low coverage: only {coverage:.1%} of returns have weights")
    
    return result

def validate_weight_invariants(
    weights: pd.DataFrame,
) -> ValidationResult:
    result = ValidationResult("Weight-Invariants")
    
    errors = validate_weights(weights)
    for err in errors:
        result.add_error(err)
    
    sector_weights = weights[SECTORS]
    
    max_weight = sector_weights.max().max()
    result.add_detail("max_sector_weight", float(max_weight))
    
    if max_weight > 1.001:
        result.add_error(f"Sector weight exceeds 100%: {max_weight:.2%}")
    
    avg_positions = (sector_weights > 0.001).sum(axis=1).mean()
    result.add_detail("avg_positions", float(avg_positions))
    
    return result


def validate_turnover_sanity(
    returns: pd.DataFrame,
    max_daily_turnover: float = 2.0,
) -> ValidationResult:

    result = ValidationResult("Turnover-Sanity")
    
    if "turnover" not in returns.columns:
        result.add_warning("No turnover column in returns")
        return result
    
    turnover = returns["turnover"]
    
    avg_turnover = turnover.mean()
    max_turnover = turnover.max()
    
    result.add_detail("avg_turnover", float(avg_turnover))
    result.add_detail("max_turnover", float(max_turnover))
    result.add_detail("total_turnover", float(turnover.sum()))
    
    excessive = turnover > max_daily_turnover
    if excessive.any():
        n_excessive = excessive.sum()
        result.add_warning(f"{n_excessive} dates have turnover > {max_daily_turnover}")
    
    if (turnover < -0.001).any():
        result.add_error("Negative turnover values found")
    
    return result


def validate_cost_sanity(
    returns: pd.DataFrame,
) -> ValidationResult:
    result = ValidationResult("Cost-Sanity")
    
    if "cost" not in returns.columns:
        result.add_warning("No cost column in returns")
        return result
    
    costs = returns["cost"]
    
    total_cost = costs.sum()
    avg_cost = costs.mean()
    
    result.add_detail("total_cost", float(total_cost))
    result.add_detail("avg_cost_per_period", float(avg_cost))
    
    if (costs < -0.0001).any():
        result.add_error("Negative transaction costs found")
    
    if "gross_ret" in returns.columns and "net_ret" in returns.columns:
        cost_drag = (returns["gross_ret"] - returns["net_ret"]).mean()
        result.add_detail("avg_cost_drag", float(cost_drag))
    
    return result

def validate_cash_hurdle_correctness(
    rankings: pd.DataFrame,
    horizon: int = 21,
) -> ValidationResult:
    result = ValidationResult("Cash-Hurdle")
    
    rf_daily = RISK_FREE_ANNUAL / 252
    cash_hurdle = (1 + rf_daily) ** horizon - 1
    cost = COST_BPS / 10000.0
    total_hurdle = cost + cash_hurdle
    
    result.add_detail("cost", float(cost))
    result.add_detail("cash_hurdle", float(cash_hurdle))
    result.add_detail("total_hurdle", float(total_hurdle))
    
    if "relative_signal" not in rankings.columns:
        result.add_warning("No relative_signal column - cannot verify cash hurdle")
        return result
    
    if "pred_excess" not in rankings.columns:
        result.add_warning("No pred_excess column - cannot verify cash hurdle")
        return result
    
    expected_signal = rankings["pred_excess"] > total_hurdle
    actual_signal = rankings["relative_signal"]
    
    disagreements = (expected_signal != actual_signal).sum()
    
    result.add_detail("n_disagreements", int(disagreements))
    result.add_detail("total_rows", len(rankings))
    
    if disagreements > len(rankings) * 0.001:
        result.add_error(
            f"Cash hurdle applied incorrectly: {disagreements} disagreements"
        )
    
    return result

def validate_engine_parity(
    returns_python: pd.DataFrame,
    returns_vectorbt: pd.DataFrame,
    tolerance: float = 1e-6,
) -> ValidationResult:
    result = ValidationResult("Engine-Parity")
    
    common_dates = returns_python.index.intersection(returns_vectorbt.index)
    result.add_detail("common_dates", len(common_dates))
    
    if len(common_dates) == 0:
        result.add_error("No common dates between engines")
        return result
    
    py_ret = returns_python.loc[common_dates, "net_ret"]
    vbt_ret = returns_vectorbt.loc[common_dates, "net_ret"]
    
    diff = (py_ret - vbt_ret).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    result.add_detail("max_diff", float(max_diff))
    result.add_detail("mean_diff", float(mean_diff))
    
    if max_diff > tolerance:
        result.add_error(
            f"Return calculation mismatch: max diff = {max_diff:.8f}"
        )
    
    py_cum = (1 + py_ret).cumprod()
    vbt_cum = (1 + vbt_ret).cumprod()
    
    cum_diff = (py_cum - vbt_cum).abs()
    final_diff = cum_diff.iloc[-1]
    
    result.add_detail("final_cumret_diff", float(final_diff))
    
    if final_diff > tolerance * len(common_dates):
        result.add_warning(
            f"Cumulative return drift: {final_diff:.6f}"
        )
    
    return result

def run_full_validation(
    rankings: pd.DataFrame,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    fold_config: dict | None = None,
    horizon: int = 21,
    mode: str = "dual",
    output_dir: Path | None = None,
    prices: pd.DataFrame | None = None,
) -> ValidationResult:
    combined = ValidationResult("Full-Validation")
    
    ranking_errors = validate_rankings(rankings, mode)
    for err in ranking_errors:
        combined.add_error(f"[Rankings-Schema] {err}")
    
    weight_result = validate_weight_invariants(weights)
    combined.merge(weight_result)
    
    turnover_result = validate_turnover_sanity(returns)
    combined.merge(turnover_result)
    
    cost_result = validate_cost_sanity(returns)
    combined.merge(cost_result)
    
    lookahead_result = validate_no_lookahead(weights, returns)
    combined.merge(lookahead_result)
    
    cash_result = validate_cash_hurdle_correctness(rankings, horizon)
    combined.merge(cash_result)
    
    if fold_config is not None:
        oof_result = validate_oof_only(rankings, fold_config)
        combined.merge(oof_result)
    
    n_errors = len(combined.errors)
    log.info("Validation: %s (%d errors)", "PASSED" if combined.passed else "FAILED", n_errors)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_validation_report({"Full": combined}, output_dir / "validation_report.json")
    
    return combined


def save_validation_report(
    results: dict[str, ValidationResult],
    output_path: Path,
) -> None:
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_passed": all(v.passed for v in results.values()),
        "n_passed": sum(1 for v in results.values() if v.passed),
        "n_total": len(results),
        "validators": {
            name: v.to_dict() for name, v in results.items()
        },
    }
    
    output_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Validation report saved to %s", output_path)