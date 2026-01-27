from __future__ import annotations
import logging
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from .contracts import SECTORS, get_data_paths

log = logging.getLogger(__name__)

MAX_POSITIONS = 5

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
    
    weights_wide = weights_wide[SECTORS]
    weights_wide = weights_wide.fillna(0.0)
    weights_wide["cash"] = 1.0 - weights_wide[SECTORS].sum(axis=1)
    weights_wide = weights_wide.clip(lower=0.0)
    row_sums = weights_wide.sum(axis=1)
    weights_wide = weights_wide.div(row_sums, axis=0)
    weights_wide = weights_wide.sort_index()
    
    if apply_actionability_shift:
        weights_wide = weights_wide.shift(1)
        weights_wide = weights_wide.dropna()
        log.info("Applied actionability shift: weights at t from signal at t-1")
    
    return weights_wide


def build_trades_from_weights(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    trades = []
    
    weights = weights.sort_index()
    prev_weights = None
    
    for date in weights.index:
        current = weights.loc[date]
        
        if prev_weights is None:
            for sector in SECTORS:
                w = current[sector]
                if w > 0.001:
                    trades.append({
                        "Date": date,
                        "Sector": sector,
                        "direction": "buy",
                        "weight_before": 0.0,
                        "weight_after": w,
                        "delta_weight": w,
                        "abs_delta": w,
                    })
        else:
            for sector in SECTORS:
                w_before = prev_weights[sector]
                w_after = current[sector]
                delta = w_after - w_before
                
                if abs(delta) > 0.001:
                    trades.append({
                        "Date": date,
                        "Sector": sector,
                        "direction": "buy" if delta > 0 else "sell",
                        "weight_before": w_before,
                        "weight_after": w_after,
                        "delta_weight": delta,
                        "abs_delta": abs(delta),
                    })
        
        prev_weights = current
    
    return pd.DataFrame(trades)


def build_regime_series(
    rankings: pd.DataFrame,
) -> pd.DataFrame | None:
    regime_cols = [
        "regime_signal", "allocation", "zone", "vix",
        "s_trend", "s_vix", "risk_scalar",
        "bullish_votes", "in_cooldown", "in_bear_trend"
    ]
    
    available_cols = [c for c in regime_cols if c in rankings.columns]
    
    if not available_cols:
        return None
    
    regime_df = rankings.groupby("Date")[available_cols].first().reset_index()
    
    return regime_df

def build_weights_from_predictions(
    predictions: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    mode: str = "dual",
    max_positions: int = MAX_POSITIONS,
    weight_method: Literal["equal", "inverse_vol"] = "equal",
    cost_threshold: float = 0.0020,
) -> pd.DataFrame:
    df = predictions.copy()
    
    df["relative_signal"] = df["pred_excess"] > cost_threshold
    
    if "abs_momentum" in df.columns:
        df["absolute_signal"] = df["abs_momentum"] > 0
    else:
        df["absolute_signal"] = True
    
    if mode == "dual":
        df["passed_gate"] = df["relative_signal"] & df["absolute_signal"]
    elif mode == "relative":
        df["passed_gate"] = df["relative_signal"]
    elif mode == "absolute":
        df["passed_gate"] = df["absolute_signal"]
    else:
        df["passed_gate"] = df["relative_signal"]
    
    df["rank"] = df.groupby("Date")["pred_excess"].rank(ascending=False, method="first")
    
    def select_top(grp):
        passed = grp[grp["passed_gate"]].sort_values("rank")
        top_ranks = passed["rank"].head(max_positions).tolist()
        return grp["rank"].isin(top_ranks) & grp["passed_gate"]
    
    df["selected"] = df.groupby("Date", group_keys=False).apply(
        select_top
    ).reset_index(level=0, drop=True)
    
    if weight_method == "equal":
        def compute_weights(grp):
            n_sel = grp["selected"].sum()
            if n_sel == 0:
                return pd.Series(0.0, index=grp.index)
            return grp["selected"].astype(float) / n_sel
        
        df["weight"] = df.groupby("Date", group_keys=False).apply(
            compute_weights
        ).reset_index(level=0, drop=True)
    else:
        df["weight"] = df.groupby("Date", group_keys=False).apply(
            lambda g: g["selected"].astype(float) / max(1, g["selected"].sum())
        ).reset_index(level=0, drop=True)
    
    weights_wide = df.pivot(index="Date", columns="Sector", values="weight")
    
    for sector in SECTORS:
        if sector not in weights_wide.columns:
            weights_wide[sector] = 0.0
    
    weights_wide = weights_wide[SECTORS].fillna(0.0)
    weights_wide["cash"] = 1.0 - weights_wide[SECTORS].sum(axis=1)
    weights_wide = weights_wide.shift(1).dropna()
    
    return weights_wide

def load_rankings_from_run(
    run_dir: Path,
    fold: int | None = None,
) -> pd.DataFrame:
    if fold is not None:
        path = run_dir / f"fold{fold}" / "rankings.csv"
    else:
        path = run_dir / "rankings.csv"
        if not path.exists():
            path = run_dir / "fold0" / "rankings.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Rankings not found at {path}")
    
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    
    return df


def load_sector_prices() -> pd.DataFrame:
    paths = get_data_paths()

    if not paths["spdr_prices"].exists():
        raise FileNotFoundError(f"SPDR prices not found at {paths['spdr_prices']}")
    
    df = pd.read_csv(paths["spdr_prices"], parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.set_index("Date")
    
    return df