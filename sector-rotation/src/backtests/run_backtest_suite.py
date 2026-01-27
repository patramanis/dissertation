from __future__ import annotations
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

from .contracts import (
    ALL_MODES,
    HORIZONS,
    SECTORS,
    RunManifest,
    get_run_dir,
    load_rankings,
    load_weights,
    load_returns,
    save_manifest,
)
from .cost_model import (
    compute_turnover,
    compute_net_returns,
    compute_portfolio_returns,
)
from .weights_builder import (
    build_weights_from_rankings,
    build_trades_from_weights,
)
from .validators import (
    ValidationResult,
    run_full_validation,
)
from .quantstats_runner import (
    run_quantstats,
    run_quantstats_comparison,
)
from .vectorbt_runner import (
    run_vectorbt,
    VectorBTResult,
    compare_strategies,
    build_equity_comparison_df,
    build_equal_weight_benchmark,
)
from .alphalens_runner import (
    run_alphalens,
    prepare_factor_for_alphalens,
    prepare_prices_for_alphalens,
)

log = logging.getLogger(__name__)

class BacktestOrchestrator:

    def __init__(
        self,
        base_dir: Path | str,
        output_dir: Path | str | None = None,
        modes: list[str] | None = None,
        horizons: list[int] | None = None,
        transaction_cost_bps: float = 10.0,
        risk_free_rate: float = 0.025,
    ):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir) if output_dir else self.base_dir / "backtest_results"
        self.modes = modes or ALL_MODES
        self.horizons = horizons or HORIZONS
        self.transaction_cost_bps = transaction_cost_bps
        self.risk_free_rate = risk_free_rate

        self.results: dict[str, dict[int, Any]] = {}
        self.validation_results: dict[str, dict[int, ValidationResult]] = {}
        self.vectorbt_results: dict[str, VectorBTResult] = {}
        self.alphalens_results: dict[str, dict] = {}

        self._prices: pd.DataFrame | None = None

    def load_prices(self, prices_path: Path | str) -> None:
        prices_path = Path(prices_path)

        if prices_path.suffix == ".csv":
            self._prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        elif prices_path.suffix == ".parquet":
            self._prices = pd.read_parquet(prices_path)
            if "Date" in self._prices.columns:
                self._prices = self._prices.set_index("Date")

        if self._prices is not None:
            available = [s for s in SECTORS if s in self._prices.columns]
            self._prices = self._prices[available]

            log.info("Loaded prices: %d dates, %d sectors",
                     len(self._prices), len(available))

    @property
    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            raise ValueError("Prices not loaded. Call load_prices() first.")
        return self._prices

    def run_single(
        self,
        mode: str,
        horizon: int,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        if run_dir is None:
            run_dir = get_run_dir(self.base_dir, mode, horizon)

        run_dir = Path(run_dir)

        if not run_dir.exists():
            log.warning("Run directory not found: %s", run_dir)
            return {"error": "run_not_found"}

        log.info("Processing %s/h%d from %s", mode, horizon, run_dir)

        results = {
            "mode": mode,
            "horizon": horizon,
            "run_dir": str(run_dir),
        }

        try:
            rankings = load_rankings(run_dir)
            results["n_rankings"] = len(rankings)
        except FileNotFoundError:
            log.warning("Rankings not found in %s", run_dir)
            return {"error": "rankings_not_found", **results}

        weights = build_weights_from_rankings(rankings)
        results["n_weights"] = len(weights)

        trades = build_trades_from_weights(weights)

        returns = compute_portfolio_returns(
            weights=weights,
            prices=self.prices,
            cost_bps=self.transaction_cost_bps,
        )
        results["n_returns"] = len(returns)

        turnover = compute_turnover(weights)
        results["avg_turnover"] = float(turnover.mean())
        results["total_turnover"] = float(turnover.sum())

        output_subdir = self.output_dir / f"{mode}_h{horizon}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        validation = run_full_validation(
            rankings=rankings,
            weights=weights,
            returns=returns,
            prices=self.prices,
            output_dir=output_subdir,
        )

        results["validation"] = validation.to_dict()
        self.validation_results.setdefault(mode, {})[horizon] = validation

        if not validation.passed:
            log.warning("Validation failed for %s/h%d: %s",
                       mode, horizon, validation.failures)

        vbt_result = run_vectorbt(
            weights=weights,
            prices=self.prices,
            transaction_cost_bps=self.transaction_cost_bps,
            risk_free_rate=self.risk_free_rate,
            output_dir=output_subdir / "vectorbt",
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

        key = f"{mode}_h{horizon}"
        self.vectorbt_results[key] = vbt_result

        if hasattr(vbt_result, 'returns') and len(vbt_result.returns) > 0:
            run_quantstats(
                returns=vbt_result.returns,
                output_dir=output_subdir / "quantstats",
                benchmark=None,
                title=f"{mode} H{horizon}",
            )

        pred_cols = [c for c in rankings.columns if "pred" in c.lower()]
        if pred_cols:
            factor = prepare_factor_for_alphalens(
                rankings,
                factor_col=pred_cols[0],
            )

            al_result = run_alphalens(
                factor=factor,
                prices=self.prices,
                periods=(horizon,),
                output_dir=output_subdir / "alphalens",
            )

            results["alphalens"] = al_result
            self.alphalens_results[key] = al_result

        self.results.setdefault(mode, {})[horizon] = results

        results_path = output_subdir / "results.json"
        results_path.write_text(
            json.dumps(results, indent=2, default=str),
            encoding="utf-8",
        )

        return results

    def run_all(self) -> dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log.info("Running backtest suite for %d modes x %d horizons",
                 len(self.modes), len(self.horizons))

        all_results = {}

        for mode in self.modes:
            for horizon in self.horizons:
                try:
                    result = self.run_single(mode, horizon)
                    all_results[f"{mode}_h{horizon}"] = result
                except Exception as e:
                    log.error("Failed %s/h%d: %s", mode, horizon, e)
                    all_results[f"{mode}_h{horizon}"] = {"error": str(e)}

        self._generate_comparisons()

        master_path = self.output_dir / f"master_results_{timestamp}.json"
        master_path.write_text(
            json.dumps(all_results, indent=2, default=str),
            encoding="utf-8",
        )

        log.info("Backtest suite complete. Results saved to %s", self.output_dir)

        return all_results

    def _generate_comparisons(self) -> None:
        if not self.vectorbt_results:
            return

        comparison = compare_strategies(self.vectorbt_results)

        comparison_path = self.output_dir / "strategy_comparison.csv"
        comparison.to_csv(comparison_path)

        latex_path = self.output_dir / "strategy_comparison.tex"
        latex_path.write_text(
            comparison.to_latex(
                float_format="%.2f",
                caption="Strategy Comparison",
                label="tab:strategy_comparison",
            ),
            encoding="utf-8",
        )

        equity_df = build_equity_comparison_df(self.vectorbt_results)
        equity_path = self.output_dir / "equity_curves_comparison.csv"
        equity_df.to_csv(equity_path)

        log.info("Generated comparison outputs")

    def run_validation_only(self) -> dict[str, ValidationResult]:
        results = {}

        for mode in self.modes:
            for horizon in self.horizons:
                run_dir = get_run_dir(self.base_dir, mode, horizon)

                if not run_dir.exists():
                    continue

                try:
                    rankings = load_rankings(run_dir)
                    weights = build_weights_from_rankings(rankings)
                    returns = compute_portfolio_returns(
                        weights, self.prices, cost_bps=self.transaction_cost_bps
                    )

                    validation = run_full_validation(
                        rankings=rankings,
                        weights=weights,
                        returns=returns,
                    )

                    key = f"{mode}_h{horizon}"
                    results[key] = validation

                except Exception as e:
                    log.error("Validation failed for %s/h%d: %s", mode, horizon, e)

        passed = sum(1 for v in results.values() if v.passed)
        log.info("Validation: %d/%d passed", passed, len(results))

        return results

def main():
    parser = argparse.ArgumentParser(
        description="Run backtest suite for paper-ready outputs"
    )

    parser.add_argument(
        "base_dir",
        type=Path,
        help="Base directory containing run results",
    )

    parser.add_argument(
        "output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: base_dir/backtest_results)",
    )

    parser.add_argument(
        "prices",
        type=Path,
        required=True,
        help="Path to prices CSV/parquet",
    )

    parser.add_argument(
        "modes", "-m",
        nargs="+",
        default=None,
        help="Modes to process (default: all)",
    )

    parser.add_argument(
        "horizons", "-H",
        nargs="+",
        type=int,
        default=None,
        help="Horizons to process (default: 5, 21, 63)",
    )

    parser.add_argument(
        "cost-bps",
        type=float,
        default=10.0,
        help="Transaction cost in basis points",
    )

    parser.add_argument(
        "validation-only",
        action="store_true",
        help="Run validation only, no full backtest",
    )

    parser.add_argument(
        "verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    orchestrator = BacktestOrchestrator(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        modes=args.modes,
        horizons=args.horizons,
        transaction_cost_bps=args.cost_bps,
    )

    orchestrator.load_prices(args.prices)

    if args.validation_only:
        results = orchestrator.run_validation_only()

        for key, val in results.items():
            status = "PASS" if val.passed else "FAIL"
            print(f"{key}: {status}")
            if not val.passed:
                for fail in val.failures:
                    print(f"  - {fail}")
    else:
        orchestrator.run_all()

if __name__ == "__main__":
    main()