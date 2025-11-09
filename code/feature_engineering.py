import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SECTOR_DATA_PATH = DATA_DIR / "sector_data.csv"
FEATURES_DATA_PATH = DATA_DIR / "features_data.csv"
OUTPUT_PATH = DATA_DIR / "engineered_features.csv"

LAG_PERIODS = [1, 2, 3, 5, 10, 21]
# Updated windows: 1 week, 2 weeks, 1 month, 2 months, quarter, semester, year (trading days)
ROLLING_WINDOWS = [5, 10, 21, 42, 63, 126, 252]
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

def load_data():
    print("Loading data...")
    sectors = pd.read_csv(SECTOR_DATA_PATH, parse_dates=["Date"])
    features = pd.read_csv(FEATURES_DATA_PATH, parse_dates=["Date"])
    df = pd.merge(sectors, features, on="Date", how="inner")
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Loaded {len(df)} trading days")
    return df

def create_lag_features(df, columns, lag_periods=LAG_PERIODS):
    print(f"Creating lag features for {len(columns)} columns...")
    result = df.copy()
    for col in columns:
        for lag in lag_periods:
            result[f"{col}_lag{lag}"] = df[col].shift(lag)
    print(f"  Created {len(columns) * len(lag_periods)} lag features")
    return result

def create_rolling_statistics(df, columns, windows=ROLLING_WINDOWS):
    print(f"Creating rolling statistics for {len(columns)} columns...")
    result = df.copy()
    for col in columns:
        for window in windows:
            result[f"{col}_rollmean{window}"] = df[col].rolling(window=window, min_periods=1).mean()
            result[f"{col}_rollstd{window}"] = df[col].rolling(window=window, min_periods=1).std()
            result[f"{col}_rollmin{window}"] = df[col].rolling(window=window, min_periods=1).min()
            result[f"{col}_rollmax{window}"] = df[col].rolling(window=window, min_periods=1).max()
    print(f"  Created {len(columns) * len(windows) * 4} rolling features")
    return result

def create_temporal_features(df):
    print("Creating temporal features...")
    result = df.copy()
    result["month"] = df["Date"].dt.month
    result["day_of_week"] = df["Date"].dt.dayofweek
    result["quarter"] = df["Date"].dt.quarter
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    print("  Created 8 temporal features")
    return result

def create_volatility_features(df, columns, windows=ROLLING_WINDOWS):
    print(f"Creating volatility features for {len(columns)} columns...")
    result = df.copy()
    for col in columns:
        log_returns = np.log(df[col] / df[col].shift(1))
        for window in windows:
            result[f"{col}_realvol{window}"] = log_returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
    print(f"  Created {len(columns) * len(windows)} volatility features")
    return result

def create_momentum_features(df, columns, periods=[5, 10, 21, 63]):
    print(f"Creating momentum features for {len(columns)} columns...")
    result = df.copy()
    for col in columns:
        for period in periods:
            result[f"{col}_return{period}"] = (df[col] - df[col].shift(period)) / df[col].shift(period)
    print(f"  Created {len(columns) * len(periods)} momentum features")
    return result

def create_uncertainty_features(df):
    print("Creating uncertainty features...")
    result = df.copy()
    result["VIX_zscore"] = (df["VIX"] - df["VIX"].rolling(252, min_periods=1).mean()) / df["VIX"].rolling(252, min_periods=1).std()
    result["EPU_zscore"] = (df["EPU"] - df["EPU"].rolling(252, min_periods=1).mean()) / df["EPU"].rolling(252, min_periods=1).std()
    result["GPR_zscore"] = (df["GPR"] - df["GPR"].rolling(252, min_periods=1).mean()) / df["GPR"].rolling(252, min_periods=1).std()
    result["UNCERTAINTY_INDEX"] = (result["VIX_zscore"] + result["EPU_zscore"] + result["GPR_zscore"]) / 3
    result["MARKET_STRESS"] = (df["VIX"] > 30).astype(int)
    print("  Created 5 uncertainty features")
    return result

def create_market_features(df):
    print("Creating market features...")
    result = df.copy()
    result["SPREAD_change"] = df["SPREAD_10Y_13W"].diff()
    result["INVERTED_CURVE"] = (df["SPREAD_10Y_13W"] < 0).astype(int)
    print("  Created 2 market features")
    return result

def apply_final_lag(df, exclude_cols):
    print("Applying final 1-day lag to all features...")
    result = df.copy()
    cols_to_lag = [col for col in df.columns if col not in exclude_cols]
    for col in cols_to_lag:
        result[f"{col}_lag1"] = df[col].shift(1)
        result.drop(columns=[col], inplace=True)
    print(f"  Lagged {len(cols_to_lag)} features")
    return result

def engineer_features():
    print("="*80)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*80)
    df = load_data()
    base_features = ["OIL", "USD", "GOLD", "IRX", "TNX", "VIX", "EPU", "GPR", "SPREAD_10Y_13W"]
    df = create_lag_features(df, base_features + SECTORS, lag_periods=LAG_PERIODS)
    df = create_rolling_statistics(df, base_features + SECTORS, windows=ROLLING_WINDOWS)
    df = create_temporal_features(df)
    vol_features = SECTORS + ["VIX", "OIL", "GOLD"]
    df = create_volatility_features(df, vol_features, windows=ROLLING_WINDOWS)
    mom_features = SECTORS + ["OIL", "USD", "GOLD", "VIX"]
    df = create_momentum_features(df, mom_features, periods=[5, 10, 21, 63])
    df = create_uncertainty_features(df)
    df = create_market_features(df)
    df = apply_final_lag(df, exclude_cols=["Date"] + SECTORS)
    print(f"\nTotal engineered features: {len(df.columns) - 1 - len(SECTORS)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    df_clean = df.dropna()
    print(f"Rows after dropping NaN: {len(df_clean)}")
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")
    return df_clean

if __name__ == "__main__":
    df = engineer_features()
    print(f"\nFinal shape: {df.shape}")
