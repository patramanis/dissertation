from __future__ import annotations
import numpy as np
import pandas as pd
from .contracts import COST_BPS, RISK_FREE_ANNUAL, SECTORS

COST_DECIMAL = COST_BPS / 10000.0
RISK_FREE_DAILY = RISK_FREE_ANNUAL / 252

def compute_turnover(
    weights: pd.DataFrame,
    sectors: list[str] | None = None,
) -> pd.Series:
    if sectors is None:
        sectors = SECTORS
    
    weights = weights.sort_index()
    
    sector_weights = weights[sectors]
    weight_changes = sector_weights.diff().abs()
    
    turnover = weight_changes.sum(axis=1) * 0.5
    
    turnover.iloc[0] = 0.0
    
    return turnover

def compute_entry_exit_costs(
    weights: pd.DataFrame,
    cost_bps: float = COST_BPS,
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    if sectors is None:
        sectors = SECTORS
    
    weights = weights.sort_index()
    sector_weights = weights[sectors]
    
    weight_changes = sector_weights.diff()
    
    entries = weight_changes.clip(lower=0)
    entry_amounts = entries.sum(axis=1)
    
    exits = (-weight_changes).clip(lower=0)
    exit_amounts = exits.sum(axis=1)
    
    cost_decimal = cost_bps / 10000.0
    
    result = pd.DataFrame({
        "entry_cost": entry_amounts * cost_decimal,
        "exit_cost": exit_amounts * cost_decimal,
        "total_cost": (entry_amounts + exit_amounts) * cost_decimal,
    }, index=weights.index)
    
    result.iloc[0, 0] = sector_weights.iloc[0].sum() * cost_decimal
    result.iloc[0, 1] = 0.0
    result.iloc[0, 2] = result.iloc[0, 0]
    
    return result

def compute_gross_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    sectors: list[str] | None = None,
) -> pd.Series:
    if sectors is None:
        sectors = SECTORS

    sector_prices = prices[sectors]
    forward_returns = sector_prices.pct_change().shift(-1)
    
    common_dates = weights.index.intersection(forward_returns.index)
    
    valid_dates = common_dates[~forward_returns.loc[common_dates].isna().any(axis=1)]
    
    w = weights.loc[valid_dates, sectors]
    r = forward_returns.loc[valid_dates, sectors]
    
    gross_returns = (w * r).sum(axis=1)
    
    return gross_returns

def compute_net_returns(
    gross_returns: pd.Series,
    costs: pd.Series,
    cash_weights: pd.Series | None = None,
) -> pd.Series:
    net = gross_returns - costs
    
    if cash_weights is not None:
        rf_contribution = cash_weights * RISK_FREE_DAILY
        net = net + rf_contribution
    
    return net

def compute_portfolio_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = COST_BPS,
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    if sectors is None:
        sectors = SECTORS
    
    if "cash" not in weights.columns:
        sector_sum = weights[sectors].sum(axis=1)
        weights = weights.copy()
        weights["cash"] = 1.0 - sector_sum
    
    turnover = compute_turnover(weights, sectors)
    costs_df = compute_entry_exit_costs(weights, cost_bps, sectors)
    gross_ret = compute_gross_returns(weights, prices, sectors)
    
    cash_weight = weights["cash"]
    exposure = 1.0 - cash_weight
    
    net_ret = compute_net_returns(
        gross_ret, 
        costs_df["total_cost"],
        cash_weight,
    )
    
    rf_contribution = cash_weight * RISK_FREE_DAILY
    n_positions = (weights[sectors] > 0.001).sum(axis=1)
    
    result = pd.DataFrame({
        "Date": weights.index,
        "gross_ret": gross_ret.values,
        "net_ret": net_ret.values,
        "cash_weight": cash_weight.values,
        "exposure": exposure.values,
        "turnover": turnover.values,
        "cost": costs_df["total_cost"].values,
        "n_positions": n_positions.values,
        "rf_contribution": rf_contribution.values,
    })
    
    result = result.set_index("Date")
    
    return result

def estimate_slippage(
    trade_amounts: pd.Series,
    volatility: pd.Series | None = None,
    market_impact_coef: float = 0.1,
) -> pd.Series:
    if volatility is None:
        volatility = pd.Series(0.02, index=trade_amounts.index)
    
    slippage = market_impact_coef * np.sqrt(trade_amounts.abs()) * volatility
    
    return pd.Series(slippage, index=trade_amounts.index)