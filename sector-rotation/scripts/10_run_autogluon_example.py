from __future__ import annotations
import argparse
import json
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = (THIS_DIR.parent / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COST_BPS = 10
COST = COST_BPS / 10000.0

RISK_FREE_ANNUAL = 0.025
RISK_FREE_DAILY = RISK_FREE_ANNUAL / 252

ABSOLUTE_MOMENTUM_WINDOWS = {
    5: 63,
    21: 200, 
    63: 256,
}
DEFAULT_MOMENTUM_WINDOW = 200
WEIGHT_METHODS = ["equal", "inverse_vol"]

MAX_POSITION_WEIGHT = 0.40

REGIME_SMA_WINDOWS = {
    "short": 21,
    "medium": 63,
    "long": 252,
}
REGIME_COOLDOWN_DAYS = 10
REGIME_RISK_ON_THRESHOLD = 2

DEFAULT_RISK_OFF_ALLOCATION = 0.0

VIX_THRESHOLD_LOW = 20.0
VIX_THRESHOLD_HIGH = 35.0
VIX_PANIC_THRESHOLD = 40.0

TREND_SMA_WINDOW = 50

HYSTERESIS_BUFFER = 2.0
MIN_COOLDOWN_DAYS = 3
MAX_COOLDOWN_DAYS = 20

YELLOW_ZONE_ALLOC_MIN = 0.25
YELLOW_ZONE_ALLOC_MAX = 0.75

BEAR_TREND_ALLOC_CAP = 0.50

CRISIS_FILTER_SMA = 200
BOLD_SCALAR_MIN_FLOOR = 0.50
BOLD_SCALAR_MAX = 1.0
VIX_RECOVERY_THRESHOLD = 0.30
VIX_PEAK_LOOKBACK = 21


def _workspace_root() -> Path:
    return THIS_DIR.parent


def _load_spy_prices() -> pd.DataFrame:
    ws = _workspace_root()
    spy_path = ws / "data" / "interim" / "aligned_pit" / "SPY.csv"
    if not spy_path.exists():
        log.warning("SPY.csv not found, regime filter will be disabled")
        return pd.DataFrame()
    
    df = pd.read_csv(spy_path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    
    if "SPY" in df.columns and "Price" not in df.columns:
        df["Price"] = df["SPY"]
    
    return df


def _load_vix_prices() -> pd.DataFrame:
    ws = _workspace_root()
    futures_path = ws / "data" / "interim" / "aligned_pit" / "Futures.csv"
    if not futures_path.exists():
        log.warning("Futures.csv not found, VIX-based regime filter disabled")
        return pd.DataFrame()
    
    df = pd.read_csv(futures_path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    
    if "^VIX" not in df.columns:
        log.warning("^VIX column not found in Futures.csv")
        return pd.DataFrame()
    
    result = df[["Date", "^VIX"]].copy()
    result = result.rename(columns={"^VIX": "VIX"})
    result = result.dropna(subset=["VIX"])
    return result

def compute_regime_signal(
    dates: pd.Series,
    cooldown_days: int = REGIME_COOLDOWN_DAYS,
    risk_on_threshold: int = REGIME_RISK_ON_THRESHOLD,
) -> pd.DataFrame:
    spy_df = _load_spy_prices()
    
    if spy_df.empty:
        unique_dates = dates.drop_duplicates().sort_values()
        return pd.DataFrame({
            "Date": unique_dates,
            "regime_signal": True,
            "bullish_votes": 3,
            "in_cooldown": False,
        })
    
    if "Price" not in spy_df.columns and "Close" in spy_df.columns:
        spy_df["Price"] = spy_df["Close"]
    elif "Price" not in spy_df.columns:
        price_cols = [c for c in spy_df.columns if c not in ["Date", "Volume"]]
        if price_cols:
            spy_df["Price"] = spy_df[price_cols[0]]
        else:
            log.warning("No price column found in SPY.csv")
            unique_dates = dates.drop_duplicates().sort_values()
            return pd.DataFrame({
                "Date": unique_dates,
                "regime_signal": True,
                "bullish_votes": 3,
                "in_cooldown": False,
            })
    
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    for name, window in REGIME_SMA_WINDOWS.items():
        spy_df[f"sma_{name}"] = spy_df["Price"].rolling(window=window, min_periods=window).mean()
    spy_df["vote_short"] = (spy_df["Price"] > spy_df["sma_short"]).astype(int)
    spy_df["vote_medium"] = (spy_df["Price"] > spy_df["sma_medium"]).astype(int)
    spy_df["vote_long"] = (spy_df["Price"] > spy_df["sma_long"]).astype(int)
    spy_df["bullish_votes"] = spy_df["vote_short"] + spy_df["vote_medium"] + spy_df["vote_long"]
    spy_df["raw_risk_on"] = spy_df["bullish_votes"] >= risk_on_threshold
    spy_df["regime_signal"] = True
    spy_df["in_cooldown"] = False
    
    cooldown_counter = 0
    for i in range(len(spy_df)):
        if cooldown_counter > 0:
            spy_df.loc[spy_df.index[i], "regime_signal"] = False
            spy_df.loc[spy_df.index[i], "in_cooldown"] = True
            cooldown_counter -= 1
        elif not spy_df.loc[spy_df.index[i], "raw_risk_on"]:
            spy_df.loc[spy_df.index[i], "regime_signal"] = False
            cooldown_counter = cooldown_days
        else:
            spy_df.loc[spy_df.index[i], "regime_signal"] = True
    
    result = spy_df[["Date", "regime_signal", "bullish_votes", "in_cooldown"]].copy()
    
    log.info("Regime filter computed: %d/%d days Risk-ON (%.1f%%), cooldown=%d days",
             result["regime_signal"].sum(), len(result),
             100 * result["regime_signal"].mean(), cooldown_days)
    
    return result

def compute_dynamic_regime_signal(
    dates: pd.Series,
    vix_low: float = VIX_THRESHOLD_LOW,
    vix_high: float = VIX_THRESHOLD_HIGH,
    trend_sma: int = TREND_SMA_WINDOW,
    hysteresis_buffer: float = HYSTERESIS_BUFFER,
    min_cooldown: int = MIN_COOLDOWN_DAYS,
    max_cooldown: int = MAX_COOLDOWN_DAYS,
    bear_cap: float = BEAR_TREND_ALLOC_CAP,
) -> pd.DataFrame:
    spy_df = _load_spy_prices()
    vix_df = _load_vix_prices()
    unique_dates = dates.drop_duplicates().sort_values()
    
    if spy_df.empty or vix_df.empty:
        log.warning("Missing SPY or VIX data - using full allocation fallback")
        return pd.DataFrame({
            "Date": unique_dates,
            "zone": "green",
            "allocation": 1.0,
            "vix": np.nan,
            "in_bear_trend": False,
            "days_in_red": 0,
            "regime_signal": True,
        })
    
    if "Price" not in spy_df.columns and "Close" in spy_df.columns:
        spy_df["Price"] = spy_df["Close"]
    elif "Price" not in spy_df.columns:
        price_cols = [c for c in spy_df.columns if c not in ["Date", "Volume"]]
        if price_cols:
            spy_df["Price"] = spy_df[price_cols[0]]
        else:
            log.warning("No price column in SPY.csv")
            return pd.DataFrame({
                "Date": unique_dates,
                "zone": "green",
                "allocation": 1.0,
                "vix": np.nan,
                "in_bear_trend": False,
                "days_in_red": 0,
                "regime_signal": True,
            })
    
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)
    merged = spy_df.merge(vix_df, on="Date", how="left")
    merged["VIX"] = merged["VIX"].ffill()
    merged["sma_trend"] = merged["Price"].rolling(window=trend_sma, min_periods=trend_sma).mean()
    merged["in_bear_trend"] = merged["Price"] < merged["sma_trend"]
    
    def _classify_zone_and_alloc(row):
        vix = row["VIX"]
        
        if pd.isna(vix):
            return "green", 1.0
        if vix <= vix_low:
            return "green", 1.0
        elif vix >= vix_high:
            return "red", 0.0
        else:
            raw_alloc = (vix_high - vix) / (vix_high - vix_low)
            clamped_alloc = YELLOW_ZONE_ALLOC_MIN + raw_alloc * (YELLOW_ZONE_ALLOC_MAX - YELLOW_ZONE_ALLOC_MIN)
            return "yellow", clamped_alloc
    
    zone_alloc = merged.apply(_classify_zone_and_alloc, axis=1, result_type="expand")
    merged["zone"] = zone_alloc[0]
    merged["base_allocation"] = zone_alloc[1]
    merged["allocation"] = merged["base_allocation"]
    bear_mask = merged["in_bear_trend"].fillna(False)
    merged.loc[bear_mask, "allocation"] = merged.loc[bear_mask, "allocation"].clip(upper=bear_cap)

    merged["days_in_red"] = 0
    days_in_red = 0
    was_in_red = False
    cooldown_active = False
    cooldown_remaining = 0
    
    exit_threshold = vix_high - hysteresis_buffer
    
    for i in range(len(merged)):
        current_zone = str(merged["zone"].iloc[i])
        current_vix = float(merged["VIX"].iloc[i])
        current_alloc = float(merged["allocation"].iloc[i])
        
        if current_zone == "red":
            days_in_red += 1
            merged.loc[merged.index[i], "days_in_red"] = days_in_red
            was_in_red = True
            cooldown_remaining = 0
            
        elif was_in_red:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                progress = 1.0 - (cooldown_remaining / max(min_cooldown, 1))
                merged.loc[merged.index[i], "allocation"] = current_alloc * progress
                merged.loc[merged.index[i], "days_in_red"] = -cooldown_remaining
                
            elif current_vix > exit_threshold and days_in_red >= min_cooldown:
                cooldown_active = True
                cooldown_remaining = min(days_in_red, max_cooldown)
                cooldown_remaining -= 1
                progress = 1.0 - (cooldown_remaining / max(min_cooldown, 1))
                merged.loc[merged.index[i], "allocation"] = current_alloc * max(0.25, progress)
                merged.loc[merged.index[i], "days_in_red"] = -cooldown_remaining
                
            else:
                was_in_red = False
                days_in_red = 0
                cooldown_active = False
                merged.loc[merged.index[i], "days_in_red"] = 0
                log.debug("V-shape override triggered at index %d, VIX=%.1f", i, current_vix)
                
        else:
            days_in_red = 0
            merged.loc[merged.index[i], "days_in_red"] = 0
    
    merged["regime_signal"] = merged["allocation"] > 0
    
    result = merged[["Date", "zone", "allocation", "VIX", "in_bear_trend", "days_in_red", "regime_signal"]].copy()
    result = result.rename(columns={"VIX": "vix"})
    
    n_green = (result["zone"] == "green").sum()
    n_yellow = (result["zone"] == "yellow").sum()
    n_red = (result["zone"] == "red").sum()
    avg_alloc = result["allocation"].mean()
    
    log.info("Dynamic regime computed: GREEN=%d (%.1f%%), YELLOW=%d (%.1f%%), RED=%d (%.1f%%)",
             n_green, 100 * n_green / len(result),
             n_yellow, 100 * n_yellow / len(result),
             n_red, 100 * n_red / len(result))
    log.info("  Average allocation: %.1f%%, Bear trend days: %d",
             100 * avg_alloc, result["in_bear_trend"].sum())
    
    return result


def compute_composite_risk_scalar(
    dates: pd.Series,
    vix_low: float = VIX_THRESHOLD_LOW,
    vix_high: float = VIX_THRESHOLD_HIGH,
    trend_sma: int = TREND_SMA_WINDOW,
    use_hybrid_bold: bool = True,
    crisis_sma: int = CRISIS_FILTER_SMA,
    min_floor: float = BOLD_SCALAR_MIN_FLOOR,
) -> pd.DataFrame:
    spy_df = _load_spy_prices()
    vix_df = _load_vix_prices()
    
    unique_dates = dates.drop_duplicates().sort_values()
    
    if spy_df.empty or vix_df.empty:
        log.warning("Missing SPY or VIX data - using full allocation fallback")
        return pd.DataFrame({
            "Date": unique_dates,
            "risk_scalar": 1.0,
            "s_trend": 1.0,
            "s_vix": 1.0,
            "in_crisis": False,
            "base_scalar": 1.0,
        })
    
    if "Price" not in spy_df.columns and "Close" in spy_df.columns:
        spy_df["Price"] = spy_df["Close"]
    elif "Price" not in spy_df.columns:
        price_cols = [c for c in spy_df.columns if c not in ["Date", "Volume"]]
        if price_cols:
            spy_df["Price"] = spy_df[price_cols[0]]
        else:
            return pd.DataFrame({
                "Date": unique_dates,
                "risk_scalar": 1.0,
                "s_trend": 1.0,
                "s_vix": 1.0,
                "in_crisis": False,
                "base_scalar": 1.0,
            })
    
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)
    merged = spy_df.merge(vix_df, on="Date", how="left")
    merged["VIX"] = merged["VIX"].ffill()
    merged["sma"] = merged["Price"].rolling(window=trend_sma, min_periods=trend_sma).mean()
    merged["sma200"] = merged["Price"].rolling(window=crisis_sma, min_periods=crisis_sma).mean()
    above_sma50 = merged["Price"] >= merged["sma"]
    above_sma200 = merged["Price"] >= merged["sma200"]

    merged["regime"] = np.where(
        above_sma200,
        "green",
        np.where(
            above_sma50,
            "yellow",
            "red"
        )
    )
    
    merged["in_crisis"] = merged["regime"] == "red"
    trend_sensitivity = 10.0
    merged["price_ratio"] = merged["Price"] / merged["sma"]
    merged["s_trend"] = ((merged["price_ratio"] - 1) * trend_sensitivity + 0.5).clip(0, 1)
    merged["s_vix"] = ((vix_high - merged["VIX"]) / (vix_high - vix_low)).clip(0, 1)
    merged["base_scalar"] = merged["s_trend"] * merged["s_vix"]

    if use_hybrid_bold:
        yellow_scalar = merged["base_scalar"] * 0.5
        
        merged["risk_scalar"] = np.select(
            [
                merged["regime"] == "red",
                merged["regime"] == "yellow",
                merged["regime"] == "green",
            ],
            [
                0.0,
                yellow_scalar,
                merged["base_scalar"],
            ],
            default=1.0
        )
        
        vix_panic_mask = merged["VIX"] > VIX_PANIC_THRESHOLD
        merged.loc[vix_panic_mask, "risk_scalar"] = 0.0
        merged.loc[vix_panic_mask, "regime"] = "panic"
        
        n_panic_days = vix_panic_mask.sum()
        if n_panic_days > 0:
            log.info("VIX PANIC TRIGGER: %d days with VIX > %.0f -> forced 0%% allocation",
                     n_panic_days, VIX_PANIC_THRESHOLD)
    else:
        merged["risk_scalar"] = merged["base_scalar"]
    
    merged["risk_scalar"] = merged["risk_scalar"].fillna(1.0)
    merged["s_trend"] = merged["s_trend"].fillna(1.0)
    merged["s_vix"] = merged["s_vix"].fillna(1.0)
    merged["base_scalar"] = merged["base_scalar"].fillna(1.0)
    merged["in_crisis"] = merged["in_crisis"].fillna(False)
    merged["regime"] = merged["regime"].fillna("green")
    merged["sma200"] = merged["sma200"].fillna(merged["Price"])
    
    result = merged[merged["Date"].isin(unique_dates)].copy()
    result = result[["Date", "risk_scalar", "s_trend", "s_vix", "VIX", "Price", 
                      "sma", "sma200", "regime", "in_crisis", "base_scalar"]].copy()
    result = result.rename(columns={"VIX": "vix", "Price": "price"})
    
    avg_scalar = result["risk_scalar"].mean()
    avg_trend = result["s_trend"].mean()
    avg_vix = result["s_vix"].mean()
    
    red_days = (result["regime"] == "red").sum()
    yellow_days = (result["regime"] == "yellow").sum()
    green_days = (result["regime"] == "green").sum()
    total_days = len(result)
    
    red_pct = 100 * red_days / total_days if total_days > 0 else 0
    yellow_pct = 100 * yellow_days / total_days if total_days > 0 else 0
    green_pct = 100 * green_days / total_days if total_days > 0 else 0
    
    mode_str = "THREE-REGIME TRAFFIC LIGHT" if use_hybrid_bold else "LEGACY"
    log.info("Composite Risk Scalar (%s): E_t avg=%.2f, S_trend avg=%.2f, S_vix avg=%.2f",
             mode_str, avg_scalar, avg_trend, avg_vix)
    log.info("E_t range: [%.2f, %.2f], Median: %.2f",
             result["risk_scalar"].min(), result["risk_scalar"].max(), 
             result["risk_scalar"].median())
    log.info("[RED] crisis: %4d days (%5.1f%%) -> 0%% allocation",
             red_days, red_pct)
    log.info("[YELLOW] caution: %4d days (%5.1f%%) -> 0-50%% allocation (linear)",
             yellow_days, yellow_pct)
    log.info("[GREEN] bull: %4d days (%5.1f%%) -> 0-100%% allocation",
             green_days, green_pct)
    
    return result


def compute_streak_decay_scalar(
    dates: pd.Series,
    decay_factor: float = 0.8,
    vix_multiplier: bool = True,
    vix_threshold: float = 25.0,
    reset_on_up_day: bool = True,
) -> pd.DataFrame:
    spy_df = _load_spy_prices()
    vix_df = _load_vix_prices()
    
    unique_dates = dates.drop_duplicates().sort_values()

    if spy_df.empty or vix_df.empty:
        log.warning("Missing SPY or VIX data - using full allocation fallback")
        return pd.DataFrame({
            "Date": unique_dates,
            "risk_scalar": 1.0,
            "consecutive_down": 0,
            "effective_decay": decay_factor,
            "regime": "normal",
        })
    
    merged = spy_df.merge(vix_df, on="Date", how="outer").sort_values("Date")
    merged = merged.ffill().dropna()
    merged["daily_return"] = merged["Price"].pct_change()
    
    consecutive_down = []
    current_streak = 0
    
    for ret in merged["daily_return"].fillna(0):
        if ret < 0:
            current_streak += 1
        else:
            if reset_on_up_day:
                current_streak = 0
            else:
                current_streak = max(0, current_streak - 1)
        consecutive_down.append(current_streak)
    
    merged["consecutive_down"] = consecutive_down
    
    if vix_multiplier:
        vix_penalty = ((merged["VIX"] - vix_threshold) / 50).clip(lower=0, upper=0.5)
        merged["effective_decay"] = decay_factor * (1 - vix_penalty)
    else:
        merged["effective_decay"] = decay_factor
    
    merged["risk_scalar"] = merged["effective_decay"] ** merged["consecutive_down"]
    
    conditions = [
        merged["consecutive_down"] <= 2,
        merged["consecutive_down"] <= 5,
        merged["consecutive_down"] <= 9,
        merged["consecutive_down"] >= 10,
    ]
    choices = ["normal", "caution", "stress", "panic"]
    merged["regime"] = np.select(conditions, choices, default="normal")
    merged["risk_scalar"] = merged["risk_scalar"].fillna(1.0)
    merged["consecutive_down"] = merged["consecutive_down"].fillna(0)
    merged["effective_decay"] = merged["effective_decay"].fillna(decay_factor)
    
    result = merged[merged["Date"].isin(unique_dates)].copy()
    result = result[["Date", "risk_scalar", "consecutive_down", "VIX", "Price",
                      "daily_return", "effective_decay", "regime"]].copy()
    result = result.rename(columns={"VIX": "vix", "Price": "price"})
    
    avg_scalar = result["risk_scalar"].mean()
    max_streak = result["consecutive_down"].max()
    normal_days = (result["regime"] == "normal").sum()
    caution_days = (result["regime"] == "caution").sum()
    stress_days = (result["regime"] == "stress").sum()
    panic_days = (result["regime"] == "panic").sum()
    total_days = len(result)
    
    log.info("VIX-Weighted Streak Decay: E_t avg=%.2f, Max streak=%d days",
             avg_scalar, int(max_streak))
    log.info("Decay factor: %.2f, VIX multiplier: %s (threshold=%.0f)",
             decay_factor, "ON" if vix_multiplier else "OFF", vix_threshold)
    log.info("[NORMAL] streak<=2:  %4d days (%5.1f%%) -> 64-100%% allocation",
             normal_days, 100 * normal_days / total_days if total_days > 0 else 0)
    log.info("[CAUTION] streak 3-5: %4d days (%5.1f%%) -> 33-51%% allocation",
             caution_days, 100 * caution_days / total_days if total_days > 0 else 0)
    log.info("[STRESS] streak 6-9: %4d days (%5.1f%%) -> 17-26%% allocation",
             stress_days, 100 * stress_days / total_days if total_days > 0 else 0)
    log.info("[PANIC] streak>=10:  %4d days (%5.1f%%) -> <17%% allocation",
             panic_days, 100 * panic_days / total_days if total_days > 0 else 0)
    
    return result

VIX_ADAPTIVE_THRESHOLDS = {
    5: 28,
    21: 30,
    63: 30,
}

VIX_ADAPTIVE_ROC_WINDOW = 5

def compute_vix_adaptive_signal(
    dates: pd.Series,
    horizon: int = 21,
) -> pd.DataFrame:
    vix_df = _load_vix_prices()
    
    if vix_df.empty:
        log.warning("VIX data not found - VIX adaptive filter disabled")
        unique_dates = dates.drop_duplicates().sort_values()
        return pd.DataFrame({
            "Date": unique_dates,
            "vix": 20.0,
            "vix_roc": 0.0,
            "vix_signal": True,
            "exposure": 1.0,
        })
    
    vix_threshold = VIX_ADAPTIVE_THRESHOLDS.get(horizon, 30)
    vix_df = vix_df.sort_values("Date").reset_index(drop=True)
    vix_df["vix_roc"] = vix_df["VIX"].pct_change(VIX_ADAPTIVE_ROC_WINDOW) * 100
    vix_below_threshold = vix_df["VIX"] < vix_threshold
    vix_falling = vix_df["vix_roc"] < 0
    
    vix_df["vix_signal"] = vix_below_threshold | (vix_df["VIX"] >= vix_threshold) & vix_falling
    vix_df["exposure"] = vix_df["vix_signal"].astype(float)
    vix_df = vix_df.rename(columns={"VIX": "vix"})
    
    n_dates = len(vix_df)
    n_invest = vix_df["vix_signal"].sum()
    pct_invest = 100 * n_invest / n_dates if n_dates > 0 else 0
    avg_vix = vix_df["vix"].mean()
    
    n_below_thresh = vix_below_threshold.sum()
    n_above_falling = ((vix_df["vix"] >= vix_threshold) & vix_falling).sum()
    n_above_rising = ((vix_df["vix"] >= vix_threshold) & ~vix_falling).sum()
    
    log.info("VIX ADAPTIVE FILTER (Horizon=%d, Threshold=%d)", horizon, vix_threshold)
    log.info("VIX stats: avg=%.1f, max=%.1f", avg_vix, vix_df["vix"].max())
    log.info("Days VIX < %d (normal):           %4d (%.1f%%) → INVEST",
             vix_threshold, n_below_thresh, 100 * n_below_thresh / n_dates if n_dates > 0 else 0)
    log.info("Days VIX >= %d & falling (rally): %4d (%.1f%%) → INVEST",
             vix_threshold, n_above_falling, 100 * n_above_falling / n_dates if n_dates > 0 else 0)
    log.info("Days VIX >= %d & rising (crisis): %4d (%.1f%%) → CASH",
             vix_threshold, n_above_rising, 100 * n_above_rising / n_dates if n_dates > 0 else 0)
    log.info("RESULT: %.1f%% of days invested, %.1f%% in cash",
             pct_invest, 100 - pct_invest)
    
    return vix_df[["Date", "vix", "vix_roc", "vix_signal", "exposure"]]

TREND_VIX_CONFIG = {
    5: {
        'ma_short': 30,
        'ma_long': 200,
        'vix_panic': 35,
        'use_vix': True,
    },
    21: {
        'ma_short': 50,
        'ma_long': 200,
        'vix_panic': 30,
        'use_vix': True,
    },
    63: {
        'ma_short': 50,
        'ma_long': 200,
        'vix_panic': None,
        'use_vix': False,
    },
}


def compute_trend_vix_signal(
    dates: pd.Series,
    horizon: int = 21,
) -> pd.DataFrame:
    config = TREND_VIX_CONFIG.get(horizon, TREND_VIX_CONFIG[21])
    
    spy_path = Path(__file__).parent.parent / "data" / "raw" / "SPY.csv"
    if not spy_path.exists():
        log.warning("SPY data not found - Trend+VIX filter disabled")
        unique_dates = dates.drop_duplicates().sort_values()
        return pd.DataFrame({
            "Date": unique_dates,
            "invest_signal": True,
            "exposure": 1.0,
        })
    
    spy_df = pd.read_csv(spy_path, parse_dates=["Date"])
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)
    spy_df = spy_df.rename(columns={"SPY": "spy"})
    spy_df["ma_short"] = spy_df["spy"].rolling(config['ma_short']).mean()
    spy_df["ma_long"] = spy_df["spy"].rolling(config['ma_long']).mean()
    spy_df["trend_bullish"] = spy_df["ma_short"] > spy_df["ma_long"]
    
    if config['use_vix'] and config['vix_panic'] is not None:
        vix_df = _load_vix_prices()
        if not vix_df.empty:
            vix_df = vix_df.sort_values("Date").reset_index(drop=True)
            spy_df = spy_df.merge(
                vix_df[["Date", "VIX"]].rename(columns={"VIX": "vix"}),
                on="Date",
                how="left"
            )
            spy_df["vix"] = spy_df["vix"].ffill()
            spy_df["vix_safe"] = spy_df["vix"] < config['vix_panic']
        else:
            spy_df["vix"] = 20.0
            spy_df["vix_safe"] = True
    else:
        spy_df["vix"] = 20.0
        spy_df["vix_safe"] = True
    
    spy_df["invest_signal"] = spy_df["trend_bullish"] & spy_df["vix_safe"]
    spy_df["exposure"] = spy_df["invest_signal"].astype(float)
    
    n_dates = len(spy_df)
    n_trend_bull = spy_df["trend_bullish"].sum()
    n_trend_bear = n_dates - n_trend_bull
    n_vix_panic = (~spy_df["vix_safe"]).sum() if config['use_vix'] else 0
    n_invest = spy_df["invest_signal"].sum()
    pct_invest = 100 * n_invest / n_dates if n_dates > 0 else 0
    
    log.info("TREND + VIX FILTER (Horizon=%d)", horizon)
    log.info("Config: MA%d/MA%d, VIX panic=%s", 
             config['ma_short'], config['ma_long'],
             str(config['vix_panic']) if config['use_vix'] else "disabled")
    log.info("Trend Analysis:")
    log.info("  Golden Cross (bullish): %4d days (%.1f%%)", 
             n_trend_bull, 100 * n_trend_bull / n_dates if n_dates > 0 else 0)
    log.info("  Death Cross (bearish):  %4d days (%.1f%%)", 
             n_trend_bear, 100 * n_trend_bear / n_dates if n_dates > 0 else 0)
    if config['use_vix']:
        log.info("VIX Panic (>%d):          %4d days (%.1f%%)", 
                 config['vix_panic'], n_vix_panic, 100 * n_vix_panic / n_dates if n_dates > 0 else 0)
    log.info("RESULT: %.1f%% invested, %.1f%% in cash", pct_invest, 100 - pct_invest)
    
    return spy_df[["Date", "spy", "ma_short", "ma_long", "vix", 
                   "trend_bullish", "vix_safe", "invest_signal", "exposure"]]

VIX_ZSCORE_CONFIG = {
    5: {
        'lookback': 60,
        'z_exit': 2.0,
        'z_enter': 1.0,
        'roc_window': 10,
        'require_trend_confirm': True,
    },
    21: {
        'lookback': 60,
        'z_exit': 2.0,
        'z_enter': 1.0,
        'roc_window': 21,
        'require_trend_confirm': True,
    },
    63: {
        'lookback': 60,
        'z_exit': 1.5,
        'z_enter': 0.5,
        'roc_window': 42,
        'require_trend_confirm': True,
    },
}

def compute_vix_zscore_signal(
    dates: pd.Series,
    horizon: int = 21,
) -> pd.DataFrame:
    config = VIX_ZSCORE_CONFIG.get(horizon, VIX_ZSCORE_CONFIG[21])
    
    vix_df = _load_vix_prices()
    if vix_df.empty:
        log.warning("VIX data not found - VIX Z-Score filter disabled")
        unique_dates = dates.drop_duplicates().sort_values()
        return pd.DataFrame({
            "Date": unique_dates,
            "invest_signal": True,
            "exposure": 1.0,
        })
    
    vix_df = vix_df.sort_values("Date").reset_index(drop=True)
    
    lookback = config['lookback']
    vix_df["vix_sma"] = vix_df["VIX"].rolling(lookback).mean()
    vix_df["vix_std"] = vix_df["VIX"].rolling(lookback).std()
    
    vix_df["z_score"] = (vix_df["VIX"] - vix_df["vix_sma"]) / vix_df["vix_std"]
    vix_df["z_score"] = vix_df["z_score"].fillna(0)
    
    spy_path = Path(__file__).parent.parent / "data" / "raw" / "SPY.csv"
    if spy_path.exists():
        spy_df = pd.read_csv(spy_path, parse_dates=["Date"])
        spy_df = spy_df.sort_values("Date").reset_index(drop=True)
        
        roc_window = config['roc_window']
        spy_df["spy_roc"] = spy_df["SPY"].pct_change(roc_window)
        
        vix_df = vix_df.merge(
            spy_df[["Date", "SPY", "spy_roc"]],
            on="Date",
            how="left"
        )
        vix_df["spy_roc"] = vix_df["spy_roc"].ffill()
    else:
        vix_df["SPY"] = 100.0
        vix_df["spy_roc"] = 0.0
    
    z_exit = config['z_exit']
    z_enter = config['z_enter']
    require_trend = config['require_trend_confirm']
    roc_window = config['roc_window']
    
    if require_trend:
        vix_df["panic_condition"] = (vix_df["z_score"] > z_exit) & (vix_df["spy_roc"] < 0)
    else:
        vix_df["panic_condition"] = vix_df["z_score"] > z_exit
    
    vix_df["calm_condition"] = vix_df["z_score"] < z_enter
    
    invest_signal = []
    current_state = True
    
    for idx, row in vix_df.iterrows():
        if pd.isna(row["z_score"]):
            invest_signal.append(current_state)
            continue
            
        if current_state:
            
            if row["panic_condition"]:
                current_state = False
        else:

            if row["calm_condition"]:
                current_state = True
        
        invest_signal.append(current_state)
    
    vix_df["invest_signal"] = invest_signal
    vix_df["exposure"] = vix_df["invest_signal"].astype(float)
    
    n_dates = len(vix_df)
    n_panic = vix_df["panic_condition"].sum()
    n_calm = vix_df["calm_condition"].sum()
    n_invest = vix_df["invest_signal"].sum()
    pct_invest = 100 * n_invest / n_dates if n_dates > 0 else 0
    avg_z = vix_df["z_score"].mean()
    max_z = vix_df["z_score"].max()
    
    log.info("VIX Z-SCORE ADAPTIVE FILTER (Horizon=%d)", horizon)

    log.info("Config: Lookback=%d days, Z_exit=%.1f, Z_enter=%.1f", 
             lookback, z_exit, z_enter)
    log.info("Trend Confirm: %s (ROC window=%d)", 
             "Yes" if require_trend else "No", roc_window)
    log.info("Z-Score Stats:")
    log.info("  Average Z-Score: %.2f", avg_z)
    log.info("  Max Z-Score:     %.2f", max_z)
    log.info("  Panic days (Z>%.1f & ROC<0): %d (%.1f%%)", 
             z_exit, n_panic, 100 * n_panic / n_dates if n_dates > 0 else 0)
    log.info("  Calm days (Z<%.1f):          %d (%.1f%%)", 
             z_enter, n_calm, 100 * n_calm / n_dates if n_dates > 0 else 0)
    log.info("RESULT: %.1f%% invested, %.1f%% in cash", pct_invest, 100 - pct_invest)

    
    return vix_df[["Date", "VIX", "vix_sma", "vix_std", "z_score", 
                   "spy_roc", "panic_condition", "calm_condition",
                   "invest_signal", "exposure"]]

def compute_layered_exposure_scalar(
    dates: pd.Series,
    sma_long: int = 200,
    bear_market_cap: float = 0.50,
    decay_factor: float = 0.80,
    vix_threshold: float = 25.0,
    vix_panic_threshold: float = 35.0,
    sma_short: int = 20,
    vshape_exposure: float = 0.50,
    min_exposure: float = 0.20,
    soft_gate_vix_cap: float = 40.0,
) -> pd.DataFrame:
    spy_df = _load_spy_prices()
    vix_df = _load_vix_prices()
    
    unique_dates = dates.drop_duplicates().sort_values()
    
    if spy_df.empty or vix_df.empty:
        log.warning("Missing SPY or VIX data - using full allocation fallback")
        return pd.DataFrame({
            "Date": unique_dates,
            "exposure": 1.0,
            "trend_regime": "bull",
            "streak_scalar": 1.0,
            "consecutive_down": 0,
            "vix": 20.0,
            "vshape_active": False,
            "soft_gate_active": False,
        })
    
    merged = spy_df.merge(vix_df, on="Date", how="outer").sort_values("Date")
    merged = merged.ffill().dropna()
    merged["sma_long"] = merged["Price"].rolling(window=sma_long, min_periods=50).mean()
    merged["sma_short"] = merged["Price"].rolling(window=sma_short, min_periods=5).mean()
    
    merged["trend_regime"] = np.where(
        merged["Price"] >= merged["sma_long"], 
        "bull", 
        "bear"
    )
    
    merged["trend_cap"] = np.where(
        merged["trend_regime"] == "bull",
        1.0,
        bear_market_cap
    )
    
    merged["daily_return"] = merged["Price"].pct_change()
    
    vix_penalty = np.where(
        merged["VIX"] < vix_threshold,
        0.0,
        np.where(
            merged["VIX"] < vix_panic_threshold,
            (merged["VIX"] - vix_threshold) / 50,
            0.2 + (merged["VIX"] - vix_panic_threshold) / 30
        )
    )
    vix_penalty = np.clip(vix_penalty, 0, 0.5)
    merged["effective_decay"] = decay_factor * (1 - vix_penalty)
    consecutive_down = []
    current_streak = 0
    
    for ret in merged["daily_return"].fillna(0):
        if ret < 0:
            current_streak += 1
        else:
            current_streak = max(0, current_streak - 1)
        consecutive_down.append(current_streak)
    
    merged["consecutive_down"] = consecutive_down
    merged["streak_scalar"] = merged["effective_decay"] ** merged["consecutive_down"]
    merged["streak_scalar"] = merged["streak_scalar"].clip(lower=0.05, upper=1.0)
    merged["vshape_active"] = (
        (merged["trend_regime"] == "bear") &
        (merged["Price"] > merged["sma_short"]) &
        (merged["consecutive_down"] <= 1)
    )
    
    merged["soft_gate_active"] = (
        (merged["Price"] > merged["sma_short"]) &
        (merged["VIX"] < soft_gate_vix_cap) &
        (merged["consecutive_down"] <= 2)
    )
    
    merged["base_exposure"] = np.minimum(
        merged["trend_cap"],
        merged["streak_scalar"]
    )
    
    merged["exposure"] = np.where(
        merged["vshape_active"],
        np.maximum(merged["base_exposure"], vshape_exposure),
        merged["base_exposure"]
    )
    
    merged["exposure"] = np.where(
        merged["soft_gate_active"],
        np.maximum(merged["exposure"], min_exposure),
        merged["exposure"]
    )
    
    merged["exposure"] = merged["exposure"].clip(lower=0.0, upper=1.0)
    
    merged["exposure"] = merged["exposure"].fillna(1.0)
    merged["streak_scalar"] = merged["streak_scalar"].fillna(1.0)
    merged["consecutive_down"] = merged["consecutive_down"].fillna(0)
    merged["vix"] = merged["VIX"].fillna(20.0)
    
    result = merged[merged["Date"].isin(unique_dates)].copy()
    result = result[["Date", "exposure", "trend_regime", "streak_scalar",
                     "consecutive_down", "vix", "vshape_active", 
                     "soft_gate_active", "effective_decay"]].copy()
    
    avg_exposure = result["exposure"].mean()
    bull_pct = (result["trend_regime"] == "bull").mean() * 100
    bear_pct = (result["trend_regime"] == "bear").mean() * 100
    vshape_pct = result["vshape_active"].mean() * 100
    softgate_pct = result["soft_gate_active"].mean() * 100
    max_streak = result["consecutive_down"].max()
    avg_vix = result["vix"].mean()
    
    log.info("LAYERED ADAPTIVE EXPOSURE SYSTEM")
    log.info("Layer 3A (Trend):   Bull=%.1f%%, Bear=%.1f%% | SMA%d",
             bull_pct, bear_pct, sma_long)
    log.info("Layer 3B (Shock):   Decay=%.2f, Max streak=%d | VIX avg=%.1f",
             decay_factor, int(max_streak), avg_vix)
    log.info("Layer 3C (V-Shape): Active %.1f%% of days", vshape_pct)
    log.info("Layer 4 (Soft):     Active %.1f%% of days | Floor=%.0f%%",
             softgate_pct, min_exposure * 100)
    log.info("FINAL EXPOSURE: avg=%.2f (%.0f%% of full allocation)",
             avg_exposure, avg_exposure * 100)
    
    return result


def _load_sector_prices() -> pd.DataFrame:
    ws = _workspace_root()
    spdr_path = ws / "data" / "interim" / "aligned_pit" / "SPDR.csv"
    if not spdr_path.exists():
        log.warning("SPDR.csv not found, absolute momentum will be disabled")
        return pd.DataFrame()
    
    df = pd.read_csv(spdr_path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df


def load_panel(h: int) -> pd.DataFrame:
    ws = _workspace_root()
    path = ws / "data" / "processed" / "panel" / f"panel_h{h}.csv"
    log.info("Loading panel from %s", path)
    panel = pd.read_csv(path, parse_dates=["Date"])
    return panel


def load_cv_folds(h: int, cv_type: str) -> list[dict]:
    ws = _workspace_root()
    path = ws / "configs" / "cv" / f"cv_{cv_type}_h{h}.json"
    log.info("Loading CV folds from %s", path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("folds", [])

def load_model_config(config_name: str = "autogluon_quick") -> dict:
    ws = _workspace_root()
    raw = Path(config_name)

    if raw.is_absolute():
        path = raw
    else:
        candidate = (ws / raw)
        if candidate.suffix == ".json" and candidate.exists():
            path = candidate
        else:
            path = ws / "configs" / "models" / f"{config_name}.json"
    
    log.info("Loading model config from %s", path)
    return json.loads(path.read_text(encoding="utf-8"))

def get_feature_columns(panel: pd.DataFrame, config: dict) -> list[str]:
    exclude = set(config.get("feature_metadata", {}).get("exclude_from_features", []))
    drop_features = set(config.get("drop_features", []))
    exclude = exclude | drop_features
    
    feature_cols = [c for c in panel.columns if c not in exclude]
    return feature_cols

def _validate_feature_columns(feature_cols: list[str], *, context: str) -> None:
    forbidden = {
        "Date", "label_excess", "horizon",
        "y_gate", "rel_rank", "rank_target", "rank_target_best0",
        "cost_bps", "label_contract", "sample_weight",
    }
    bad = sorted([c for c in feature_cols if c in forbidden])
    if bad:
        raise ValueError(f"Forbidden columns leaked into features ({context}): {bad}")

def compute_sample_weights(
    panel: pd.DataFrame,
    train_start: str,
    train_end: str,
    decay_len: int | None = None,
    min_weight: float = 0.05,
) -> pd.Series:
    from cv.weights import compute_linear_decay_weights

    return compute_linear_decay_weights(
        panel,
        train_start=train_start,
        train_end=train_end,
        decay_len=decay_len,
        min_weight=min_weight,
    )

def train_regressor(
    train_data: pd.DataFrame,
    tuning_data: pd.DataFrame | None,
    feature_cols: list[str],
    config: dict,
    output_dir: Path,
    sample_weights: pd.Series | None = None,
    num_gpus: int | None = None,
    seed: int | None = None,
) -> Any:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        log.error("AutoGluon not installed.")
        return None
    
    reg_config = config.get("regressor", {})
    label = "label_excess"
    
    train_df = train_data[feature_cols + [label]].copy()
    
    weight_col = "sample_weight" if sample_weights is not None else None
    if weight_col and sample_weights is not None:
        train_df[weight_col] = sample_weights.loc[train_data.index].values
    
    tuning_df = None
    if tuning_data is not None and len(tuning_data) > 0:
        tuning_df = tuning_data[feature_cols + [label]].copy()
        if weight_col and sample_weights is not None:
            tuning_df[weight_col] = sample_weights.loc[tuning_data.index].values
    
    log.info("Training regressor for label_excess...")
    log.info("Train samples: %d", len(train_df))
    log.info("Tuning samples: %d", len(tuning_df) if tuning_df is not None else 0)
    log.info("Features: %d", len(feature_cols))
    
    predictor_kwargs: dict[str, Any] = {
        "label": label,
        "problem_type": "regression",
        "eval_metric": reg_config.get("eval_metric", "root_mean_squared_error"),
        "path": str(output_dir / "regressor"),
        "verbosity": reg_config.get("verbosity", 2),
    }
    if weight_col and sample_weights is not None:
        predictor_kwargs["sample_weight"] = weight_col

    predictor = TabularPredictor(**predictor_kwargs)
    
    fit_kwargs: dict[str, Any] = {
        "train_data": train_df,
        "time_limit": reg_config.get("time_limit", 300),
        "presets": reg_config.get("presets", "medium_quality"),
        "hyperparameters": reg_config.get("hyperparameters"),
        "num_bag_folds": 0,
        "num_bag_sets": 1,
        "auto_stack": False,
        "excluded_model_types": reg_config.get("excluded_model_types"),
    }

    if seed is not None:
        fit_kwargs.setdefault("ag_args_fit", {})
        fit_kwargs["ag_args_fit"]["random_seed"] = int(seed)
        
        hyperparameters = fit_kwargs.get("hyperparameters")
        if isinstance(hyperparameters, dict):
            hyperparameters = json.loads(json.dumps(hyperparameters))
            
            def inject_seed(hp_dict: dict, key: str, param_name: str, seed_val: int):
                if key in hp_dict:
                    val = hp_dict[key]
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, dict):
                                item[param_name] = seed_val
                    elif isinstance(val, dict):
                        val[param_name] = seed_val
            
            inject_seed(hyperparameters, "GBM", "seed", seed)
            inject_seed(hyperparameters, "XGB", "random_state", seed)
            inject_seed(hyperparameters, "CAT", "random_seed", seed)
            inject_seed(hyperparameters, "RF", "random_state", seed)
            
            fit_kwargs["hyperparameters"] = hyperparameters

    if tuning_df is not None:
        fit_kwargs["tuning_data"] = tuning_df

    predictor.fit(**fit_kwargs)
    
    return predictor

MOMENTUM_FILTER_TYPES = ["none", "dual", "relative", "absolute", "regime", "regime_dynamic", "regime_scalar", "regime_streak_decay", "layered_adaptive", "pure_momentum", "vix_adaptive", "trend_vix", "vix_zscore"]

def generate_rankings(
    test_data: pd.DataFrame,
    regressor,
    feature_cols: list[str],
    cost: float = COST,
    demean_predictions: bool = False,
    max_positions: int = 5,
    momentum_filter: str = "dual",
    abs_momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    weight_method: str = "equal",
    smoothing_window: int = 1,
    regime_cooldown: int = REGIME_COOLDOWN_DAYS,
    risk_off_allocation: float = DEFAULT_RISK_OFF_ALLOCATION,
    horizon: int = 21,
) -> pd.DataFrame:
    if momentum_filter not in MOMENTUM_FILTER_TYPES:
        raise ValueError(f"Invalid momentum_filter '{momentum_filter}'. Must be one of {MOMENTUM_FILTER_TYPES}")
    
    if weight_method not in WEIGHT_METHODS:
        raise ValueError(f"Invalid weight_method '{weight_method}'. Must be one of {WEIGHT_METHODS}")
    
    if momentum_filter == "pure_momentum":
        result = test_data[["Date", "Sector", "label_excess"]].copy()
        result["pred_excess_raw"] = 0.0
        result["pred_excess"] = 0.0
        log.info("Pure Momentum mode: SKIPPING ML predictions (using momentum only)")
    else:
        if regressor is None:
            raise ValueError("Regressor required for non-pure_momentum filters")
        X_test = test_data[feature_cols]
        pred_excess = regressor.predict(X_test).values
        
        result = test_data[["Date", "Sector", "label_excess"]].copy()
        result["pred_excess_raw"] = pred_excess
        
        if demean_predictions:
            result["pred_excess"] = result.groupby("Date")["pred_excess_raw"].transform(
                lambda x: x - x.mean()
            )
            log.info("Applied cross-sectional demeaning to predictions")
        else:
            result["pred_excess"] = result["pred_excess_raw"]
    
    if "vol" in test_data.columns:
        result["vol"] = test_data["vol"].values
    
    if smoothing_window > 1 and momentum_filter != "pure_momentum":
        result = result.sort_values(["Sector", "Date"]).reset_index(drop=True)
        
        result["pred_excess"] = result.groupby("Sector")["pred_excess"].transform(
            lambda x: x.ewm(span=smoothing_window, min_periods=1).mean()
        )
        
        result = result.sort_values(["Date", "Sector"]).reset_index(drop=True)
        log.info("Applied prediction smoothing with EMA span=%d", smoothing_window)
    
    sectors_per_date = result.groupby("Date")["Sector"].nunique()
    bad_dates = sectors_per_date[sectors_per_date != 9]
    if len(bad_dates) > 0:
        log.warning("INTEGRITY: %d dates don't have exactly 9 sectors", len(bad_dates))
    
    dup_check = result.groupby(["Date", "Sector"]).size()
    dups = dup_check[dup_check > 1]
    if len(dups) > 0:
        raise ValueError("INTEGRITY ERROR: Duplicate (Date,Sector) pairs found")
    
    need_absolute = momentum_filter in ["dual", "absolute", "pure_momentum"]
    
    if need_absolute:
        prices_df = _load_sector_prices()
        
        if len(prices_df) > 0:
            sectors = result["Sector"].unique()
            
            abs_mom_data = []
            for sec in sectors:
                if sec in prices_df.columns:
                    sec_prices = prices_df[["Date", sec]].copy()
                    sec_prices = sec_prices.sort_values("Date")
                    sec_prices[f"abs_mom_{sec}"] = np.log(
                        sec_prices[sec] / sec_prices[sec].shift(abs_momentum_window)
                    )
                    abs_mom_data.append(sec_prices[["Date", f"abs_mom_{sec}"]])
            
            if abs_mom_data:
                abs_mom_df = abs_mom_data[0]
                for df in abs_mom_data[1:]:
                    abs_mom_df = abs_mom_df.merge(df, on="Date", how="outer")
                
                abs_mom_long = abs_mom_df.melt(
                    id_vars=["Date"],
                    var_name="abs_mom_col",
                    value_name="abs_momentum"
                )
                abs_mom_long["Sector"] = abs_mom_long["abs_mom_col"].str.replace("abs_mom_", "")
                abs_mom_long = abs_mom_long[["Date", "Sector", "abs_momentum"]]
                
                result = result.merge(abs_mom_long, on=["Date", "Sector"], how="left")
                
                log.info("Momentum Filter: Using %d-day absolute momentum (%s mode)", 
                         abs_momentum_window, momentum_filter)
            else:
                log.warning("No absolute momentum data available")
                result["abs_momentum"] = np.nan
        else:
            log.warning("No price data for absolute momentum")
            result["abs_momentum"] = np.nan
    else:
        result["abs_momentum"] = np.nan
    
    cash_hurdle = (1 + RISK_FREE_DAILY) ** horizon - 1
    total_hurdle = cost + cash_hurdle
    log.info("Cash Hurdle: cost=%.4f + rf_hurdle=%.4f = total_hurdle=%.4f (horizon=%d days)",
             cost, cash_hurdle, total_hurdle, horizon)
    
    result["relative_signal"] = result["pred_excess"] > total_hurdle
    
    if "abs_momentum" in result.columns and need_absolute:
        result["absolute_signal"] = result["abs_momentum"].fillna(-np.inf) > 0
    else:
        result["absolute_signal"] = True
    
    if momentum_filter == "none":
        result["passed_gate"] = True
        result["relative_signal"] = True
        log.info("Filter mode: NONE (TRUE UNFILTERED - always invest in top sectors)")
    elif momentum_filter == "dual":
        result["passed_gate"] = result["relative_signal"] & result["absolute_signal"]
        log.info("Filter mode: DUAL (relative AND absolute)")
    elif momentum_filter == "relative":
        result["passed_gate"] = result["relative_signal"]
        log.info("Filter mode: RELATIVE (model prediction only)")
    elif momentum_filter == "absolute":
        result["passed_gate"] = result["absolute_signal"]
        log.info("Filter mode: ABSOLUTE (sector uptrend only)")
    elif momentum_filter == "pure_momentum":
        if "abs_momentum" in result.columns:
            momentum_values = result["abs_momentum"].fillna(-np.inf)
            
            result["momentum_relative_signal"] = momentum_values > total_hurdle
            result["absolute_signal"] = momentum_values > 0
            result["passed_gate"] = result["momentum_relative_signal"] & result["absolute_signal"]
            result["momentum_for_rank"] = momentum_values.replace(-np.inf, -999.0)
            result["rank"] = result.groupby("Date")["momentum_for_rank"].rank(
                ascending=False, method="first"
            )
            
            n_rel_pass = result.groupby("Date")["momentum_relative_signal"].sum().mean()
            n_abs_pass = result.groupby("Date")["absolute_signal"].sum().mean()
            n_gate_pass = result.groupby("Date")["passed_gate"].sum().mean()
            
            log.info("Filter mode: TRADITIONAL DUAL MOMENTUM (No ML - Fair Academic Baseline)")
            log.info("Gate: (momentum > %.4f) AND (momentum > 0)", total_hurdle)
            log.info("Avg sectors passing: relative=%.1f, absolute=%.1f, gate=%.1f",
                     n_rel_pass, n_abs_pass, n_gate_pass)
            log.info("Ranking: by %d-day momentum strength (not ML predictions)", 
                     abs_momentum_window)
        else:
            log.warning("No momentum data available for pure_momentum filter - using fallback")
            result["passed_gate"] = result["absolute_signal"]
            result["rank"] = result.groupby("Date")["pred_excess"].rank(
                ascending=False, method="first"
            )
            log.info("Filter mode: PURE_MOMENTUM (fallback to model predictions)")
    elif momentum_filter == "regime":
        regime_df = compute_regime_signal(
            dates=result["Date"],
            cooldown_days=regime_cooldown,
        )
        
        result = result.merge(regime_df, on="Date", how="left")
        result["regime_signal"] = result["regime_signal"].fillna(False)
    
        if risk_off_allocation > 0:
            result["passed_gate"] = True
            result["regime_weight_mult"] = result["regime_signal"].map(
                {True: 1.0, False: risk_off_allocation}
            ).fillna(risk_off_allocation)
            log.info("Filter mode: REGIME SOFT GATING (Risk-OFF allocation: %.0f%%)", 
                     risk_off_allocation * 100)
        else:
            result["passed_gate"] = result["regime_signal"]
            result["regime_weight_mult"] = 1.0
            log.info("Filter mode: REGIME BINARY (3-SMA Voting + %d-day Cooldown)", regime_cooldown)
        
        n_risk_on = result.groupby("Date")["regime_signal"].first().sum()
        n_cooldown = result.groupby("Date")["in_cooldown"].first().sum()
        n_dates = result["Date"].nunique()
        log.info("Risk-ON days: %d/%d (%.1f%%), Cooldown days: %d",
                 n_risk_on, n_dates, 100 * n_risk_on / n_dates, n_cooldown)
    
    elif momentum_filter == "regime_dynamic":
        dynamic_regime_df = compute_dynamic_regime_signal(
            dates=result["Date"],
            vix_low=VIX_THRESHOLD_LOW,
            vix_high=VIX_THRESHOLD_HIGH,
            trend_sma=TREND_SMA_WINDOW,
            hysteresis_buffer=HYSTERESIS_BUFFER,
            min_cooldown=MIN_COOLDOWN_DAYS,
            max_cooldown=MAX_COOLDOWN_DAYS,
            bear_cap=BEAR_TREND_ALLOC_CAP,
        )
        
        result = result.merge(dynamic_regime_df, on="Date", how="left")
        result["allocation"] = result["allocation"].fillna(0.0)
        result["zone"] = result["zone"].fillna("red")
        result["regime_signal"] = result["regime_signal"].fillna(False)
        result["passed_gate"] = True
        result["regime_weight_mult"] = result["allocation"]
        
        zone_counts = result.groupby("Date")["zone"].first().value_counts()
        n_dates = result["Date"].nunique()
        n_green = zone_counts.get("green", 0)
        n_yellow = zone_counts.get("yellow", 0)
        n_red = zone_counts.get("red", 0)
        
        avg_alloc = result.groupby("Date")["allocation"].first().mean()
        bear_days = result.groupby("Date")["in_bear_trend"].first().sum()
        
        log.info("Filter mode: REGIME DYNAMIC (3-Zone VIX + Trend Confirmation)")
        log.info("Zone distribution: GREEN=%d (%.1f%%), YELLOW=%d (%.1f%%), RED=%d (%.1f%%)",
                 n_green, 100 * n_green / n_dates,
                 n_yellow, 100 * n_yellow / n_dates,
                 n_red, 100 * n_red / n_dates)
        log.info("Average allocation: %.1f%%, Bear trend days: %d (%.1f%%)",
                 100 * avg_alloc, bear_days, 100 * bear_days / n_dates)
    
    elif momentum_filter == "regime_scalar":
        scalar_df = compute_composite_risk_scalar(
            dates=result["Date"],
            vix_low=VIX_THRESHOLD_LOW,
            vix_high=VIX_THRESHOLD_HIGH,
            trend_sma=TREND_SMA_WINDOW,
        )
        
        result = result.merge(scalar_df, on="Date", how="left")
        result["risk_scalar"] = result["risk_scalar"].fillna(1.0)
        result["s_trend"] = result["s_trend"].fillna(1.0)
        result["s_vix"] = result["s_vix"].fillna(1.0)
        result["passed_gate"] = True
        result["regime_weight_mult"] = result["risk_scalar"]
        result["regime_signal"] = result["risk_scalar"] > 0.5
        result["allocation"] = result["risk_scalar"]
        
        n_dates = result["Date"].nunique()
        avg_scalar = result.groupby("Date")["risk_scalar"].first().mean()
        avg_trend = result.groupby("Date")["s_trend"].first().mean()
        avg_vix = result.groupby("Date")["s_vix"].first().mean()
        
        scalar_bins = pd.cut(result.groupby("Date")["risk_scalar"].first(), 
                           bins=[0, 0.25, 0.5, 0.75, 1.0], 
                           labels=["0-25%", "25-50%", "50-75%", "75-100%"])
        bin_counts = scalar_bins.value_counts().sort_index()
        
        log.info("Filter mode: REGIME SCALAR (Composite Risk E_t = S_trend * S_vix)")
        log.info("E_t avg=%.2f, S_trend avg=%.2f, S_vix avg=%.2f", avg_scalar, avg_trend, avg_vix)
        log.info("E_t distribution: %s", 
                 ", ".join([f"{k}:{v}" for k, v in bin_counts.items()]))
    
    elif momentum_filter == "regime_streak_decay":

        streak_df = compute_streak_decay_scalar(
            dates=result["Date"],
            decay_factor=0.8,
            vix_multiplier=True,
            vix_threshold=25.0,
            reset_on_up_day=False,
        )
        
        result = result.merge(streak_df, on="Date", how="left")
        
        result["risk_scalar"] = result["risk_scalar"].fillna(1.0)
        result["consecutive_down"] = result["consecutive_down"].fillna(0)
        result["effective_decay"] = result["effective_decay"].fillna(0.8)
        result["passed_gate"] = True
        result["regime_weight_mult"] = result["risk_scalar"]
        
        result["regime_signal"] = result["risk_scalar"] > 0.5
        result["allocation"] = result["risk_scalar"]
        
        if "regime" not in result.columns:
            result["regime"] = "normal"
        
        n_dates = result["Date"].nunique()
        avg_scalar = result.groupby("Date")["risk_scalar"].first().mean()
        max_streak = result.groupby("Date")["consecutive_down"].first().max()
        
        regime_counts = result.groupby("Date")["regime"].first().value_counts()
        n_normal = regime_counts.get("normal", 0)
        n_caution = regime_counts.get("caution", 0)
        n_stress = regime_counts.get("stress", 0)
        n_panic = regime_counts.get("panic", 0)
        
        log.info("Filter mode: VIX-WEIGHTED STREAK DECAY (Fast Crash Detection)")
        log.info("E_t avg=%.2f, Max streak=%d days", avg_scalar, int(max_streak))
        log.info("Regime distribution: NORMAL=%d (%.1f%%), CAUTION=%d (%.1f%%), STRESS=%d (%.1f%%), PANIC=%d (%.1f%%)",
                 n_normal, 100 * n_normal / n_dates if n_dates > 0 else 0,
                 n_caution, 100 * n_caution / n_dates if n_dates > 0 else 0,
                 n_stress, 100 * n_stress / n_dates if n_dates > 0 else 0,
                 n_panic, 100 * n_panic / n_dates if n_dates > 0 else 0)
    
    elif momentum_filter == "layered_adaptive":
        
        layered_df = compute_layered_exposure_scalar(
            dates=result["Date"],
            sma_long=200,
            bear_market_cap=0.50,
            decay_factor=0.80,
            vix_threshold=25.0,
            vix_panic_threshold=35.0,
            sma_short=20,
            vshape_exposure=0.50,
            min_exposure=0.20,
            soft_gate_vix_cap=40.0,
        )
        
        result = result.merge(layered_df, on="Date", how="left")
        result["exposure"] = result["exposure"].fillna(1.0)
        result["streak_scalar"] = result["streak_scalar"].fillna(1.0)
        result["consecutive_down"] = result["consecutive_down"].fillna(0)
        result["trend_regime"] = result["trend_regime"].fillna("bull")
        result["vshape_active"] = result["vshape_active"].fillna(False)
        result["soft_gate_active"] = result["soft_gate_active"].fillna(False)
        result["passed_gate"] = True
        result["regime_weight_mult"] = result["exposure"]
        result["regime_signal"] = result["exposure"] > 0.5
        result["allocation"] = result["exposure"]
        
        def classify_layered_regime(row):
            if row["consecutive_down"] >= 6:
                return "panic"
            elif row["consecutive_down"] >= 3:
                return "caution"
            elif row["trend_regime"] == "bear":
                return "bear"
            else:
                return "bull"
        
        result["regime"] = result.apply(classify_layered_regime, axis=1)
        result["risk_scalar"] = result["exposure"]
        n_dates = result["Date"].nunique()
        avg_exposure = result.groupby("Date")["exposure"].first().mean()
        trend_counts = result.groupby("Date")["trend_regime"].first().value_counts()
        n_bull = trend_counts.get("bull", 0)
        n_bear = trend_counts.get("bear", 0)
        vshape_days = result.groupby("Date")["vshape_active"].first().sum()
        softgate_days = result.groupby("Date")["soft_gate_active"].first().sum()
        max_streak = result.groupby("Date")["consecutive_down"].first().max()
        avg_vix = result.groupby("Date")["vix"].first().mean()
        
        log.info("\nFilter mode: LAYERED ADAPTIVE (Complete Quant Architecture)")

        log.info("Layer 1 (Selection): ML ranking → Top-%d sectors", max_positions)
        log.info("Layer 2 (Sizing): Inverse Volatility (applied at weight stage)")
        log.info("Layer 3A (Trend): BULL=%d days (%.1f%%), BEAR=%d days (%.1f%%)",
                 n_bull, 100 * n_bull / n_dates if n_dates > 0 else 0,
                 n_bear, 100 * n_bear / n_dates if n_dates > 0 else 0)
        log.info("Layer 3B (Shock): Max streak=%d days, Avg VIX=%.1f",
                 int(max_streak), avg_vix)
        log.info("Layer 3C (V-Shape): Active %d days (%.1f%%) - catching bottoms",
                 int(vshape_days), 100 * vshape_days / n_dates if n_dates > 0 else 0)
        log.info("Layer 4 (Soft): Active %d days (%.1f%%) - anti cash-drag",
                 int(softgate_days), 100 * softgate_days / n_dates if n_dates > 0 else 0)
        log.info("FINAL EXPOSURE: avg=%.2f (%.0f%% of full allocation)",
                 avg_exposure, avg_exposure * 100)
    
    elif momentum_filter == "vix_adaptive":
        horizon = int(test_data["horizon"].iloc[0]) if "horizon" in test_data.columns else 21
        vix_df = compute_vix_adaptive_signal(
            dates=result["Date"],
            horizon=horizon,
        )
        
        result = result.merge(vix_df, on="Date", how="left")
        result["vix"] = result["vix"].fillna(20.0)
        result["vix_roc"] = result["vix_roc"].fillna(0.0)
        result["vix_signal"] = result["vix_signal"].fillna(True)
        result["exposure"] = result["exposure"].fillna(1.0)
        result["passed_gate"] = result["vix_signal"]
        result["regime_weight_mult"] = result["exposure"]
        result["regime_signal"] = result["vix_signal"]
        result["allocation"] = result["exposure"]
        result["risk_scalar"] = result["exposure"]
        
        def classify_vix_regime(row):
            if row["vix"] >= VIX_ADAPTIVE_THRESHOLDS.get(horizon, 30):
                if row["vix_roc"] >= 0:
                    return "crisis"
                else:
                    return "recovery"
            else:
                return "normal"
        
        result["regime"] = result.apply(classify_vix_regime, axis=1)
        
        log.info("Filter mode: VIX ADAPTIVE (Simple threshold + ROC reentry)")
    
    elif momentum_filter == "trend_vix":
        horizon = int(test_data["horizon"].iloc[0]) if "horizon" in test_data.columns else 21
        
        trend_vix_df = compute_trend_vix_signal(
            dates=result["Date"],
            horizon=horizon,
        )
        
        result = result.merge(
            trend_vix_df[["Date", "invest_signal", "exposure", "trend_bullish", "vix_safe"]].drop_duplicates("Date"),
            on="Date", 
            how="left"
        )
        
        result["invest_signal"] = result["invest_signal"].fillna(True)
        result["exposure"] = result["exposure"].fillna(1.0)
        result["trend_bullish"] = result["trend_bullish"].fillna(True)
        result["vix_safe"] = result["vix_safe"].fillna(True)
        result["passed_gate"] = result["invest_signal"]
        result["regime_weight_mult"] = result["exposure"]
        result["regime_signal"] = result["invest_signal"]
        result["allocation"] = result["exposure"]
        result["risk_scalar"] = result["exposure"]
        
        def classify_trend_vix_regime(row):
            if not row["trend_bullish"]:
                return "bear"
            elif not row["vix_safe"]:
                return "panic"
            else:
                return "bull"
        
        result["regime"] = result.apply(classify_trend_vix_regime, axis=1)
        
        n_sectors = result["Sector"].nunique()
        n_dates = len(result["Date"].unique())
        n_bull = (result["regime"] == "bull").sum() // n_sectors if n_sectors > 0 else 0
        n_bear = (result["regime"] == "bear").sum() // n_sectors if n_sectors > 0 else 0
        n_panic = (result["regime"] == "panic").sum() // n_sectors if n_sectors > 0 else 0
        
        log.info("Filter mode: TREND + VIX (Research-optimal)")
        log.info("  Regime breakdown: BULL=%d, BEAR=%d, PANIC=%d days", n_bull, n_bear, n_panic)
    
    elif momentum_filter == "vix_zscore":
        horizon = int(test_data["horizon"].iloc[0]) if "horizon" in test_data.columns else 21
        
        zscore_df = compute_vix_zscore_signal(
            dates=result["Date"],
            horizon=horizon,
        )
        
        result = result.merge(
            zscore_df[["Date", "invest_signal", "exposure", "z_score", 
                       "panic_condition", "calm_condition"]].drop_duplicates("Date"),
            on="Date", 
            how="left"
        )
        
        result["invest_signal"] = result["invest_signal"].fillna(True)
        result["exposure"] = result["exposure"].fillna(1.0)
        result["z_score"] = result["z_score"].fillna(0.0)
        result["panic_condition"] = result["panic_condition"].fillna(False)
        result["calm_condition"] = result["calm_condition"].fillna(True)
        result["passed_gate"] = result["invest_signal"]
        result["regime_weight_mult"] = result["exposure"]
        result["regime_signal"] = result["invest_signal"]
        result["allocation"] = result["exposure"]
        result["risk_scalar"] = result["exposure"]
        
        def classify_zscore_regime(row):
            if row["panic_condition"]:
                return "panic"
            elif row["z_score"] > 1.0:
                return "elevated"
            else:
                return "normal"
        
        result["regime"] = result.apply(classify_zscore_regime, axis=1)
        
        n_sectors = result["Sector"].nunique()
        n_dates = len(result["Date"].unique())
        n_normal = (result["regime"] == "normal").sum() // n_sectors if n_sectors > 0 else 0
        n_elevated = (result["regime"] == "elevated").sum() // n_sectors if n_sectors > 0 else 0
        n_panic = (result["regime"] == "panic").sum() // n_sectors if n_sectors > 0 else 0
        avg_z = result["z_score"].mean()
        
        log.info("Filter mode: VIX Z-SCORE (Adaptive Bollinger)")
        log.info("  Avg Z-Score: %.2f", avg_z)
        log.info("  Regime breakdown: NORMAL=%d, ELEVATED=%d, PANIC=%d days", 
                 n_normal, n_elevated, n_panic)
    
    result = result.sort_values(["Date", "Sector"]).reset_index(drop=True)
    
    if "rank" not in result.columns:
        result["rank"] = result.groupby("Date")["pred_excess"].rank(ascending=False, method="first")
    
    def _select_portfolio(grp):
        passed = grp[grp["passed_gate"]].sort_values("rank")
        n_selected = min(len(passed), max_positions)
        selected_mask = grp["rank"].isin(passed["rank"].head(n_selected).values) & grp["passed_gate"]
        return selected_mask
    
    result["selected"] = result.groupby("Date", group_keys=False).apply(
        _select_portfolio, include_groups=False  # type: ignore[call-overload]
    ).reset_index(level=0, drop=True)
    
    if weight_method == "equal":
        def _compute_weights_equal(grp):
            n_selected = grp["selected"].sum()
            if n_selected == 0:
                return pd.Series(0.0, index=grp.index)
            weight = 1.0 / n_selected
            return grp["selected"].astype(float) * weight
        
        result["portfolio_weight"] = result.groupby("Date", group_keys=False).apply(
            _compute_weights_equal, include_groups=False  # type: ignore[call-overload]
        ).reset_index(level=0, drop=True)
        
    elif weight_method == "inverse_vol":        
        def _compute_weights_inverse_vol(grp):
            selected = grp[grp["selected"]]
            n_selected = len(selected)
            
            if n_selected == 0:
                return pd.Series(0.0, index=grp.index)
            
            if "vol" not in grp.columns or selected["vol"].isna().all():
                weight = 1.0 / n_selected
                return grp["selected"].astype(float) * weight
            
            vols = selected["vol"].fillna(selected["vol"].median())
            vols = vols.clip(lower=0.001)
            
            inv_vols = 1.0 / vols
            weight_sum = inv_vols.sum()
            
            if weight_sum <= 0:
                weight = 1.0 / n_selected
                return grp["selected"].astype(float) * weight
            
            raw_weights = inv_vols / weight_sum
            capped_weights = raw_weights.clip(upper=MAX_POSITION_WEIGHT)
            final_weights = capped_weights / capped_weights.sum()
            
            output = pd.Series(0.0, index=grp.index)
            output.loc[selected.index] = final_weights.values
            return output
        
        result["portfolio_weight"] = result.groupby("Date", group_keys=False).apply(
            _compute_weights_inverse_vol, include_groups=False  # type: ignore[call-overload]
        ).reset_index(level=0, drop=True)
        
        log.info("Weight method: INVERSE_VOL (Risk Parity Lite, max weight=%.0f%%)", 
                 MAX_POSITION_WEIGHT * 100)
    
    if "regime_weight_mult" in result.columns:
        result["portfolio_weight"] = result["portfolio_weight"] * result["regime_weight_mult"]
        
        n_risk_off_dates = (~result.groupby("Date")["regime_signal"].first()).sum()
        if n_risk_off_dates > 0 and risk_off_allocation > 0:
            avg_risk_off_weight = result.loc[~result["regime_signal"], "portfolio_weight"].sum() / n_risk_off_dates
            log.info("  Soft gating: %.0f Risk-OFF dates with avg %.1f%% allocation",
                     n_risk_off_dates, avg_risk_off_weight * 100)
    
    n_selected_per_date = result.groupby("Date")["selected"].sum()
    avg_positions = n_selected_per_date.mean()
    cash_dates = (n_selected_per_date == 0).sum()
    
    n_relative_passed = result.groupby("Date")["relative_signal"].sum().mean()
    n_absolute_passed = result.groupby("Date")["absolute_signal"].sum().mean()
    n_both_passed = result.groupby("Date")["passed_gate"].sum().mean()
    
    log.info("Portfolio stats: avg %.1f positions, %d dates in cash (%.1f%%)",
             avg_positions, cash_dates, 100 * cash_dates / len(n_selected_per_date))
    log.info("Momentum filter [%s]: avg %.1f relative OK, %.1f absolute OK, %.1f gate passed",
             momentum_filter, n_relative_passed, n_absolute_passed, n_both_passed)
    
    return result


def compute_metrics(
    rankings: pd.DataFrame,
    cost: float = COST,
    horizon: int = 21,
) -> dict[str, Any]:
    from scipy import stats
    
    rf_per_period = (1 + RISK_FREE_DAILY) ** horizon - 1
    
    ics = []
    dates_with_ic = []
    for date in rankings["Date"].unique():
        mask = rankings["Date"] == date
        subset = rankings.loc[mask]
        pred = subset["pred_excess"].to_numpy()
        truth = subset["label_excess"].to_numpy()
        
        if np.isnan(truth).any() or np.isnan(pred).any():
            continue
            
        if len(np.unique(truth)) > 1:
            res: Any = stats.spearmanr(pred, truth)
            ic = float(getattr(res, "correlation", res[0]))
            if not np.isnan(ic):
                ics.append(ic)
                dates_with_ic.append(date)
    
    ic_mean = float(np.mean(ics)) if ics else 0.0
    ic_std = float(np.std(ics)) if ics else 0.0
    n_ic = len(ics)
    
    ic_tstat_naive = float(ic_mean / (ic_std / np.sqrt(n_ic))) if ic_std > 0 and n_ic > 1 else 0.0
    n_effective = max(1, n_ic / horizon)
    ic_tstat_adjusted = float(ic_mean / (ic_std / np.sqrt(n_effective))) if ic_std > 0 and n_effective > 1 else 0.0
    
    dates_sorted = sorted(rankings["Date"].unique())
    
    portfolio_returns = []
    portfolio_gross_returns = []
    n_positions_list = []
    
    prev_positions: set[str] = set()
    prev_weights: dict[str, float] = {}
    
    cash_returns_earned = 0.0
    
    for date in dates_sorted:
        mask = rankings["Date"] == date
        subset = rankings.loc[mask]
        
        selected = subset[subset["selected"]]
        curr_positions = set(selected["Sector"].tolist())
        curr_weights = dict(zip(selected["Sector"], selected["portfolio_weight"]))
        n_pos = len(curr_positions)
        n_positions_list.append(n_pos)
        
        total_invested_weight = sum(curr_weights.values()) if curr_weights else 0.0
        cash_weight = 1.0 - total_invested_weight
        
        if n_pos > 0:
            port_ret_gross = float((selected["portfolio_weight"] * selected["label_excess"]).sum())
            port_ret_gross += cash_weight * rf_per_period
        else:
            port_ret_gross = rf_per_period
            cash_returns_earned += rf_per_period
        
        portfolio_gross_returns.append(port_ret_gross)
        
        entries = curr_positions - prev_positions
        exits = prev_positions - curr_positions
        
        entry_cost = sum(curr_weights.get(s, 0) for s in entries) * cost
        exit_cost = sum(prev_weights.get(s, 0) for s in exits) * cost
        total_cost = entry_cost + exit_cost
        
        port_ret_net = port_ret_gross - total_cost
        portfolio_returns.append(port_ret_net)
        
        prev_positions = curr_positions
        prev_weights = curr_weights
    
    portfolio_returns = np.array(portfolio_returns)
    portfolio_gross_returns = np.array(portfolio_gross_returns)
    n_positions_arr = np.array(n_positions_list)
    
    avg_positions = float(np.mean(n_positions_arr))
    cash_dates = int((n_positions_arr == 0).sum())
    n_dates = len(dates_sorted)
    cash_pct = float(cash_dates / n_dates) if n_dates > 0 else 0.0
    
    pos_dist = {i: int((n_positions_arr == i).sum()) for i in range(6)}
    
    selected = rankings[rankings["selected"]]
    n_selections = len(selected)
    
    if n_selections > 0:
        hit_rate = float((selected["label_excess"] > 0).mean())
        avg_excess_per_selection = float(selected["label_excess"].mean())
    else:
        hit_rate = 0.0
        avg_excess_per_selection = 0.0
    
    avg_gross_return = float(portfolio_gross_returns.mean()) if len(portfolio_gross_returns) > 0 else 0.0
    avg_net_return = float(portfolio_returns.mean()) if len(portfolio_returns) > 0 else 0.0
    
    if len(portfolio_returns) > 1 and portfolio_returns.std() > 0:
        n_observations = len(portfolio_returns)
        n_effective = max(1, n_observations / horizon)
        periods_per_year = 252 / horizon
        overlap_factor = min(horizon, n_observations)
        adjusted_std = portfolio_returns.std() * np.sqrt(overlap_factor)
        sharpe = float(portfolio_returns.mean() / adjusted_std * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0
    
    total_turnover = n_selections
    
    passed_gate_sectors = rankings[rankings["passed_gate"]]
    avg_passed = float(passed_gate_sectors.groupby("Date").size().mean()) if len(passed_gate_sectors) > 0 else 0.0
    
    if "relative_signal" in rankings.columns and "absolute_signal" in rankings.columns:
        avg_relative_passed = float(rankings.groupby("Date")["relative_signal"].sum().mean())
        avg_absolute_passed = float(rankings.groupby("Date")["absolute_signal"].sum().mean())
        avg_both_passed = float(rankings.groupby("Date")["passed_gate"].sum().mean())
        
        n_dates_abs_blocked = int((rankings.groupby("Date")["absolute_signal"].sum() == 0).sum())
    else:
        avg_relative_passed = avg_passed
        avg_absolute_passed = 9.0
        avg_both_passed = avg_passed
        n_dates_abs_blocked = 0
    
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_n": n_ic,
        "ic_n_effective": float(n_effective),
        "ic_tstat_naive": ic_tstat_naive,
        "ic_tstat_adjusted": ic_tstat_adjusted,
        "ic_tstat": ic_tstat_adjusted,
        "hit_rate": hit_rate,
        "avg_excess_per_selection": avg_excess_per_selection,
        "n_selections": n_selections,
        "avg_excess_gross": avg_gross_return,
        "avg_excess_net": avg_net_return,
        "sharpe_ratio": sharpe,
        "avg_positions": avg_positions,
        "cash_pct": cash_pct,
        "cash_dates": cash_dates,
        "position_distribution": pos_dist,
        "avg_relative_passed": avg_relative_passed,
        "avg_absolute_passed": avg_absolute_passed,
        "avg_both_passed": avg_both_passed,
        "n_dates_abs_blocked": n_dates_abs_blocked,
        "avg_sectors_passed_gate": avg_passed,
        "total_turnover": total_turnover,
        "n_dates": n_dates,
        "cost_bps": COST_BPS,
        "risk_free_annual": RISK_FREE_ANNUAL,
        "risk_free_per_period": float(rf_per_period),
        "cash_returns_earned": float(cash_returns_earned),
    }

def run_fold(
    h: int,
    cv_type: str,
    fold_id: int,
    config_name: str = "autogluon_quick",
    output_dir: Path | None = None,
    num_gpus: int | None = None,
    seed: int | None = None,
    momentum_filter: str = "dual",
    weight_method: str = "equal",
    smoothing_window: int = 1,
    regime_cooldown: int = REGIME_COOLDOWN_DAYS,
    risk_off_allocation: float = DEFAULT_RISK_OFF_ALLOCATION,
) -> dict:
    ws = _workspace_root()
    
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    panel = load_panel(h)
    folds = load_cv_folds(h, cv_type)
    config = load_model_config(config_name)
    
    if fold_id >= len(folds):
        raise ValueError(f"Fold {fold_id} not found (max: {len(folds) - 1})")
    
    fold = folds[fold_id]
    
    log.info("Running fold %d for h=%d, cv=%s", fold_id, h, cv_type)
    log.info("Train: %s to %s", fold["train_start"], fold["train_end"])
    log.info("Valid: %s to %s", fold.get("valid_start"), fold.get("valid_end"))
    log.info("Test: %s to %s", fold["test_start"], fold["test_end"])
    log.info("Cost: %d bps (%.4f)", COST_BPS, COST)
    
    train_start = pd.Timestamp(fold["train_start"])
    train_end = pd.Timestamp(fold["train_end"])
    test_start = pd.Timestamp(fold["test_start"])
    test_end = pd.Timestamp(fold["test_end"])
    
    train_mask = (panel["Date"] >= train_start) & (panel["Date"] <= train_end)
    test_mask = (panel["Date"] >= test_start) & (panel["Date"] <= test_end)
    
    train_data = panel.loc[train_mask].copy()
    test_data = panel.loc[test_mask].copy()
    
    tuning_data = None
    if fold.get("valid_start") and fold.get("valid_end"):
        valid_start = pd.Timestamp(fold["valid_start"])
        valid_end = pd.Timestamp(fold["valid_end"])
        valid_mask = (panel["Date"] >= valid_start) & (panel["Date"] <= valid_end)
        tuning_data = panel.loc[valid_mask].copy()

    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["Date"]).dt.normalize().unique()))
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    gap = int(fold.get("purge_gap", h + 5))
    
    def _idx(d: pd.Timestamp) -> int:
        return int(date_to_idx[pd.Timestamp(d).normalize()])

    if tuning_data is not None and len(tuning_data) > 0:
        te = pd.Timestamp(fold["train_end"]).normalize()
        vs = pd.Timestamp(fold["valid_start"]).normalize()
        ve = pd.Timestamp(fold["valid_end"]).normalize()
        ts = pd.Timestamp(fold["test_start"]).normalize()

        gap_train_valid = _idx(vs) - _idx(te) - 1
        gap_valid_test = _idx(ts) - _idx(ve) - 1
        if gap_train_valid < gap:
            raise AssertionError(f"CV gap train→valid too small: {gap_train_valid} < {gap}")
        if gap_valid_test < gap:
            raise AssertionError(f"CV gap valid→test too small: {gap_valid_test} < {gap}")
    
    feature_cols = get_feature_columns(panel, config)
    _validate_feature_columns(feature_cols, context="regressor")
    
    log.info("Features: %d columns", len(feature_cols))
    log.info("Momentum filter: %s", momentum_filter)
    log.info("Weight method: %s", weight_method)
    log.info("Smoothing window: %d", smoothing_window)
    
    if output_dir is None:
        output_dir = ws / "runs" / f"{cv_type}_h{h}_fold{fold_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )
    
    if momentum_filter == "pure_momentum":
        regressor = None
        log.info("Pure Momentum mode: SKIPPING ML model training (Traditional Dual Momentum)")
    else:
        regressor = train_regressor(
            train_data,
            tuning_data,
            feature_cols,
            config,
            output_dir,
            sample_weights=None,
            num_gpus=num_gpus,
            seed=seed,
        )
        
        if regressor is None:
            return {"error": "AutoGluon not available"}
    
    abs_momentum_window = ABSOLUTE_MOMENTUM_WINDOWS.get(h, DEFAULT_MOMENTUM_WINDOW)
    log.info("Using absolute momentum window: %d days (horizon=%d)", abs_momentum_window, h)
    
    rankings = generate_rankings(
        test_data, regressor, feature_cols,
        momentum_filter=momentum_filter,
        abs_momentum_window=abs_momentum_window,
        weight_method=weight_method,
        smoothing_window=smoothing_window,
        regime_cooldown=regime_cooldown,
        risk_off_allocation=risk_off_allocation,
        horizon=h,
    )
    
    rankings_path = output_dir / "rankings.csv"
    rankings.to_csv(rankings_path, index=False)
    log.info("Rankings saved to %s", rankings_path)
    
    metrics = compute_metrics(rankings, horizon=h)
    metrics.update({
        "horizon": int(h),
        "cv_type": str(cv_type),
        "fold_id": int(fold_id),
        "seed": int(seed) if seed is not None else None,
        "momentum_filter": str(momentum_filter),
        "weight_method": str(weight_method),
        "smoothing_window": int(smoothing_window),
        "regime_cooldown": int(regime_cooldown) if momentum_filter == "regime" else None,
    })
    
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Metrics saved to %s", metrics_path)
    log.info("Fold %d complete: Sharpe=%.2f, IC=%.4f, HitRate=%.1f%%, CashDays=%.0f%%",
             fold_id, metrics["sharpe_ratio"], metrics["ic_mean"], 
             metrics["hit_rate"] * 100, metrics["cash_pct"] * 100)
    
    return metrics

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AutoGluon regression for sector rotation")
    p.add_argument("horizon", type=int, default=21, help="Forecast horizon")
    p.add_argument("cv-type", default="rolling", 
                   choices=["rolling"])
    p.add_argument("fold", type=int, default=0, help="Fold ID to run")
    p.add_argument("config", default="autogluon_quick", help="Config name or path")
    p.add_argument("output-dir", default=None, help="Output directory")
    p.add_argument("num-gpus", default="auto", help="Number of GPUs (integer or 'auto')")
    p.add_argument("seed", type=int, default=0, help="Random seed")
    p.add_argument("momentum-filter", default="dual",
                   choices=["none", "dual", "relative", "absolute", "regime", "regime_dynamic", "regime_scalar", "regime_streak_decay", "layered_adaptive", "pure_momentum", "vix_adaptive", "trend_vix", "vix_zscore"],
                   help="Momentum filter: none, dual, relative, absolute, regime, regime_dynamic, regime_scalar, regime_streak_decay, layered_adaptive, pure_momentum, vix_adaptive, trend_vix, or vix_zscore")
    
    p.add_argument("weight-method", default="equal",
                   choices=["equal", "inverse_vol"],
                   help="Portfolio weighting: equal (1/N) or inverse_vol (Risk Parity Lite)")
    p.add_argument("smoothing-window", type=int, default=1,
                   help="EMA span for prediction smoothing (1 = no smoothing)")
    p.add_argument("regime-cooldown", type=int, default=REGIME_COOLDOWN_DAYS,
                   help="Days to block trading after Risk-OFF (only for regime filter)")
    p.add_argument("risk-off-allocation", type=float, default=DEFAULT_RISK_OFF_ALLOCATION,
                   help="Allocation during Risk-OFF: 0.0=binary/cash, 0.25=soft gating (regime filter only)")
    p.add_argument("log-level", default="INFO", help="Logging level")
    return p.parse_args()


def _resolve_num_gpus(x: str) -> int:
    s = str(x).strip().lower()
    if s == "auto":
        try:
            import torch
            return 1 if torch.cuda.is_available() else 0
        except Exception:
            pass
        try:
            proc = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
            )
            if proc.returncode == 0 and (proc.stdout or "").strip():
                return 1
        except Exception:
            pass
        return 0
    return int(s)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    
    out_dir = Path(args.output_dir) if args.output_dir else None
    num_gpus = _resolve_num_gpus(args.num_gpus)

    run_fold(
        h=args.horizon,
        cv_type=args.cv_type,
        fold_id=args.fold,
        config_name=args.config,
        output_dir=out_dir,
        num_gpus=num_gpus,
        seed=int(args.seed) if args.seed is not None else None,
        momentum_filter=args.momentum_filter,
        weight_method=args.weight_method,
        smoothing_window=args.smoothing_window,
        regime_cooldown=args.regime_cooldown,
        risk_off_allocation=args.risk_off_allocation,
    )