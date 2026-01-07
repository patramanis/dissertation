from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    strategy_returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    metrics: dict[str, float]


class VectorizedBacktester:
    def __init__(self, returns_df: pd.DataFrame, prices_df: pd.DataFrame | None = None) -> None:
        if not isinstance(returns_df, pd.DataFrame):
            raise TypeError("returns_df must be a pandas DataFrame")
        if returns_df.index.nlevels != 1:
            raise ValueError("returns_df must have a 1-level Datetime-like index")
        if returns_df.columns.duplicated().any():
            raise ValueError("returns_df has duplicate columns")

        self.returns_df = returns_df.copy()
        self.prices_df = prices_df.copy() if prices_df is not None else None

    def run(
        self,
        predictions: pd.DataFrame,
        *,
        top_k: int = 3,
        holding_period: int = 1,
        cost_bps: float = 0.0,
    ) -> BacktestResult:
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("predictions must be a pandas DataFrame")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")
        if cost_bps < 0:
            raise ValueError("cost_bps must be >= 0")

        common_cols = [c for c in predictions.columns if c in self.returns_df.columns]
        if not common_cols:
            raise ValueError("No overlapping tickers between predictions and returns_df")

        preds = predictions.loc[:, common_cols].copy()
        rets = self.returns_df.loc[:, common_cols].copy()

        common_idx = preds.index.intersection(rets.index)
        preds = preds.loc[common_idx]
        rets = rets.loc[common_idx]

        preds = preds.sort_index(axis=1)
        rets = rets.reindex(columns=preds.columns)

        n_assets = preds.shape[1]
        k = min(int(top_k), int(n_assets))

        scores = preds.to_numpy(dtype=float, copy=True)
        scores_nan = np.isnan(scores)
        scores[scores_nan] = -np.inf

        weights = np.zeros_like(scores, dtype=float)
        # Deterministic top-k selection: stable sort on -scores,
        # with implicit tie-break by column order (after preds.sort_index(axis=1)).
        top_idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        row_idx = np.arange(scores.shape[0])[:, None]

        selected_is_finite = np.isfinite(scores[row_idx, top_idx])
        selected_counts = selected_is_finite.sum(axis=1).astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            per_row_w = np.where(selected_counts > 0, 1.0 / selected_counts, 0.0)

        weights[row_idx, top_idx] = selected_is_finite * per_row_w[:, None]
        weights_df = pd.DataFrame(weights, index=preds.index, columns=preds.columns)
        exec_w = weights_df.shift(holding_period)

        # Gross returns (before costs)
        strat = (exec_w * rets).sum(axis=1, min_count=1)

        # Simple turnover-based transaction cost model.
        # Note: this assumes rebalancing each period in `preds` frequency.
        turnover = exec_w.diff().abs().sum(axis=1).fillna(0.0) / 2.0
        cost = turnover * (float(cost_bps) / 10_000.0)

        strat_net = strat - cost
        strat_clean = strat_net.dropna()
        equity = (1.0 + strat_clean).cumprod()
        metrics = self.calculate_metrics(strat_clean)

        if len(strat_clean) > 0:
            metrics = dict(metrics)
            metrics["Avg Turnover"] = float(pd.to_numeric(turnover.loc[strat_clean.index], errors="coerce").mean())
            metrics["Cost (bps)"] = float(cost_bps)

        return BacktestResult(
            strategy_returns=strat_clean,
            equity_curve=equity,
            weights=weights_df,
            metrics=metrics,
        )

    @staticmethod
    def calculate_metrics(strategy_returns: pd.Series) -> dict[str, float]:
        if not isinstance(strategy_returns, pd.Series):
            strategy_returns = pd.Series(strategy_returns)

        r = pd.to_numeric(strategy_returns, errors="coerce").dropna()
        if r.empty:
            return {
                "Total Return": float("nan"),
                "Annualized Return": float("nan"),
                "Annualized Volatility": float("nan"),
                "Sharpe Ratio": float("nan"),
                "Max Drawdown": float("nan"),
            }

        equity = (1.0 + r).cumprod()

        total_return = float(equity.iloc[-1] - 1.0)

        n_days = int(r.shape[0])
        ann_return = float((equity.iloc[-1]) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0) if n_days > 0 else float("nan")

        ann_vol = float(r.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")

        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_dd = float(drawdown.min())

        return {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
        }
