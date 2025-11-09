import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import technical indicators module
import sys
sys.path.append(str(Path(__file__).parent))
from technical_indicators import add_technical_indicators

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REGIME_DIR = BASE_DIR / "temporal_files" / "regime_detection"
SECTOR_DATA_PATH = DATA_DIR / "sector_data.csv"
FEATURES_DATA_PATH = DATA_DIR / "features_data.csv"
REGIME_PATH = REGIME_DIR / "regime_probabilities.csv"
OUTPUT_PATH = DATA_DIR / "engineered_features_v2.csv"

# Configuration
LAG_PERIODS = [1, 2, 3, 5, 10, 21]
ROLLING_WINDOWS = [5, 10, 21, 42, 63, 126, 252]
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
FEATURES = ["OIL", "USD", "GOLD", "IRX", "TNX", "VIX", "EPU", "GPR", "SPREAD_10Y_13W"]

def load_data():
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    print("\nLoading sector data...")
    sectors = pd.read_csv(SECTOR_DATA_PATH, parse_dates=["Date"])
    print(f"  {len(sectors)} days, {len(SECTORS)} sectors")
    
    print("Loading features data...")
    features = pd.read_csv(FEATURES_DATA_PATH, parse_dates=["Date"])
    print(f"  {len(features)} days, {len(FEATURES)} features")
    
    print("Loading regime probabilities...")
    regimes = pd.read_csv(REGIME_PATH, parse_dates=["Date"])
    regimes = regimes[[c for c in regimes.columns if c != 'VIX']]
    print(f"  {len(regimes)} days, 3 regimes")
    
    # Merge all
    df = pd.merge(sectors, features, on="Date", how="inner")
    df = pd.merge(df, regimes, on="Date", how="inner")
    df = df.sort_values("Date").reset_index(drop=True)
    
    print(f"\n Combined dataset: {len(df)} days")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Columns: {', '.join(df.columns[:20])}...")
    
    return df

def create_lag_features(df, columns, lag_periods=LAG_PERIODS):
    print(f"\n{'='*80}")
    print(f"CREATING LAG FEATURES")
    print(f"{'='*80}")
    print(f"  Columns: {len(columns)}")
    print(f"  Lag periods: {lag_periods}")
    
    result = df.copy()
    count = 0
    for col in columns:
        for lag in lag_periods:
            result[f"{col}_lag{lag}"] = df[col].shift(lag)
            count += 1
    
    print(f"Created {count} lag features")
    return result

def create_rolling_statistics(df, columns, windows=ROLLING_WINDOWS):
    print(f"\n{'='*80}")
    print(f"CREATING ROLLING STATISTICS")
    print(f"{'='*80}")
    print(f"  Columns: {len(columns)}")
    print(f"  Windows: {windows}")
    
    result = df.copy()
    count = 0
    
    for col in columns:
        for window in windows:
            result[f"{col}_rollmean{window}"] = df[col].rolling(window=window, min_periods=1).mean()
            result[f"{col}_rollstd{window}"] = df[col].rolling(window=window, min_periods=1).std()
            result[f"{col}_rollmin{window}"] = df[col].rolling(window=window, min_periods=1).min()
            result[f"{col}_rollmax{window}"] = df[col].rolling(window=window, min_periods=1).max()
            count += 4
    
    print(f"✓ Created {count} rolling features")
    return result

def create_temporal_features(df):
    print(f"\n{'='*80}")
    print(f"CREATING TEMPORAL FEATURES")
    print(f"{'='*80}")
    
    result = df.copy()
    result["month"] = df["Date"].dt.month
    result["day_of_week"] = df["Date"].dt.dayofweek
    result["quarter"] = df["Date"].dt.quarter
    result["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    result["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 5)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 5)
    
    count = 9
    print(f"Created {count} temporal features")
    return result

def create_technical_indicators(df):
    print(f"\n{'='*80}")
    print(f"CREATING TECHNICAL INDICATORS")
    print(f"{'='*80}")
    print(f"Assets: {len(SECTORS + FEATURES)} (9 sectors + 9 features)")
    print(f"Indicators per asset: 17 (RSI, MACD, Bollinger, Stochastic, ADX, etc.)")
    
    result = df.copy()
    count = 0
    
    # Add indicators for each sector
    for sector in SECTORS:
        result = add_technical_indicators(result, sector, vix_col='VIX')
        count += 17
    
    # Add indicators for each feature (except VIX)
    for feat in FEATURES:
        if feat != 'VIX':
            result = add_technical_indicators(result, feat, vix_col='VIX')
            count += 17
    
    print(f"✓ Created {count} technical indicator features")
    return result

def create_cross_asset_features(df):
    print(f"\n{'='*80}")
    print(f"CREATING CROSS-ASSET FEATURES")
    print(f"{'='*80}")
    
    result = df.copy()
    count = 0
    
    # Sector-sector correlations (rolling 21d)
    print("Computing sector-sector correlations...")
    for i, sector1 in enumerate(SECTORS):
        for sector2 in SECTORS[i+1:]:
            result[f'corr_{sector1}_{sector2}_21d'] = \
                df[sector1].rolling(21).corr(df[sector2])
            count += 1
    
    # Sector betas (vs MARKET)
    print("Computing sector betas...")
    result['MARKET'] = df[SECTORS].mean(axis=1)
    
    for sector in SECTORS:
        cov = df[sector].rolling(63).cov(result['MARKET'])
        var = result['MARKET'].rolling(63).var()
        result[f'{sector}_beta_63d'] = cov / (var + 1e-10)
        count += 1
    
    # Feature-sector correlations (selected pairs)
    print("Computing feature-sector correlations...")
    key_features = ['VIX', 'USD', 'OIL', 'GOLD']
    for feat in key_features:
        for sector in SECTORS:
            result[f'corr_{feat}_{sector}_21d'] = \
                df[feat].rolling(21).corr(df[sector])
            count += 1
    
    print(f"Created {count} cross-asset features")
    return result

def create_volatility_features(df):
    print(f"\n{'='*80}")
    print(f"CREATING VOLATILITY FEATURES")
    print(f"{'='*80}")
    
    result = df.copy()
    count = 0
    
    assets = SECTORS + ["OIL", "USD", "GOLD", "VIX"]
    
    for asset in assets:
        for window in [21, 63]:
            result[f"{asset}_realvol{window}"] = \
                df[asset].pct_change().rolling(window=window, min_periods=1).std()
            count += 1
    
    print(f"Created {count} volatility features")
    return result

def create_momentum_features(df):
    print(f"\n{'='*80}")
    print(f"CREATING MOMENTUM FEATURES")
    print(f"{'='*80}")
    
    result = df.copy()
    count = 0
    
    assets = SECTORS + ["OIL", "USD", "GOLD"]
    
    for asset in assets:
        for period in [5, 21, 63]:
            result[f"{asset}_return{period}"] = df[asset].pct_change(period)
            count += 1
    
    print(f"Created {count} momentum features")
    return result

def create_regime_interaction_features(df):
    print(f"\n{'='*80}")
    print(f"CREATING REGIME INTERACTION FEATURES")
    print(f"{'='*80}")
    
    result = df.copy()
    count = 0
    
    # Select top features to interact with regimes
    top_features = [
        'VIX',
        'XLK_rollstd63',
        'XLF_rollstd63',
        'USD_rollstd14',
        'OIL_return21'
    ]
    
    # Check which exist
    top_features = [f for f in top_features if f in df.columns]
    
    print(f"Interacting {len(top_features)} top features with 3 regimes...")
    
    for feat in top_features:
        for regime in [0, 1, 2]:
            result[f'{feat}_regime{regime}'] = \
                df[feat] * df[f'regime_{regime}_prob']
            count += 1
    
    print(f"✓ Created {count} regime interaction features")
    return result

def create_uncertainty_features(df):
    print(f"\n{'='*80}")
    print(f"CREATING UNCERTAINTY FEATURES")
    print(f"{'='*80}")
    
    result = df.copy()
    count = 0
    
    # Composite uncertainty
    result["uncertainty_composite"] = (
        df["EPU"] / df["EPU"].rolling(252).mean() +
        df["GPR"] / df["GPR"].rolling(252).mean() +
        df["VIX"] / df["VIX"].rolling(252).mean()
    ) / 3
    count += 1
    
    # Uncertainty change
    result["uncertainty_change"] = result["uncertainty_composite"].pct_change(21)
    count += 1
    
    # Policy vs geopolitical dominance
    result["policy_vs_geo"] = (
        (df["EPU"] / df["EPU"].rolling(252).mean()) -
        (df["GPR"] / df["GPR"].rolling(252).mean())
    )
    count += 1
    
    print(f"Created {count} uncertainty features")
    return result

def create_market_features(df):
    print(f"\n{'='*80}")
    print(f"CREATING MARKET FEATURES")
    print(f"{'='*80}")
    
    result = df.copy()
    count = 0
    
    # Market dispersion (std of sector returns)
    sector_returns = df[SECTORS].pct_change()
    result["market_dispersion"] = sector_returns.std(axis=1)
    count += 1
    
    # Market breadth (% of sectors with positive returns)
    result["market_breadth"] = (sector_returns > 0).mean(axis=1)
    count += 1
    
    print(f"Created {count} market features")
    return result

def apply_final_lag(df):
    print(f"\n{'='*80}")
    print(f"APPLYING FINAL 1-DAY LAG")
    print(f"{'='*80}")
    
    # Columns to keep without lagging
    keep_cols = ["Date"] + SECTORS + ["VIX"]
    
    # Regime columns - already lagged in their creation
    regime_cols = [c for c in df.columns if 'regime' in c]
    keep_cols.extend(regime_cols)
    
    # All other columns get lagged
    feature_cols = [c for c in df.columns if c not in keep_cols]
    
    result = df[keep_cols].copy()
    
    for col in feature_cols:
        result[f"{col}_lag1"] = df[col].shift(1)
    
    print(f"Lagged {len(feature_cols)} features")
    print(f"Kept {len(keep_cols)} columns without lag")
    
    return result

def main():
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING PIPELINE V2")
    print(f"{'='*80}\n")
    
    # Load data
    df = load_data()
    
    # Base features for engineering
    base_cols = SECTORS + FEATURES
    
    # Execute pipeline
    df = create_lag_features(df, base_cols)
    df = create_rolling_statistics(df, base_cols)
    df = create_temporal_features(df)
    df = create_technical_indicators(df)
    df = create_cross_asset_features(df)
    df = create_volatility_features(df)
    df = create_momentum_features(df)
    df = create_regime_interaction_features(df)
    df = create_uncertainty_features(df)
    df = create_market_features(df)
    df = apply_final_lag(df)
    
    # Count total features
    feature_cols = [c for c in df.columns if c not in ["Date"] + SECTORS]
    
    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Total engineered features: {len(feature_cols)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Rows before dropna: {len(df)}")
    
    # Drop NaN rows
    df_clean = df.dropna()
    print(f"Rows after dropna: {len(df_clean)}")
    print(f"Dropped: {len(df) - len(df_clean)} rows")
    
    # Save
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Final shape: {df_clean.shape}")
    
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING V2 COMPLETED")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
