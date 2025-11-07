import os
import numpy as np
import pandas as pd
try:
    from hurst import compute_Hc  # type: ignore
    HURST_AVAILABLE = True
except ImportError:
    HURST_AVAILABLE = False
    print("Warning: 'hurst' library not found. Skipping Hurst Exponent features.")

SECTOR_PATH = os.path.join("data", "sector_data.csv")
FEATURES_PATH = os.path.join("data", "features_data.csv")
GPR_PATH = os.path.join("data", "GPR.csv")
OUTPUT_PATH = os.path.join("data", "model_ready_data.csv")

# Config
ROLLING_WINDOWS = [5, 21, 63]   # ~1w, ~1m, ~1q
LAG_DAYS = [1, 2, 3, 5]

def load_inputs():
    # Load data
    sector_df = pd.read_csv(SECTOR_PATH, parse_dates=['Date'], index_col='Date')
    features_df = pd.read_csv(FEATURES_PATH, parse_dates=['Date'], index_col='Date')
    gpr_df = pd.read_csv(GPR_PATH, parse_dates=['Date'], index_col='Date')

    # Sort index
    sector_df = sector_df.sort_index()
    features_df = features_df.sort_index()
    gpr_df = gpr_df.sort_index()

    # Coerce to numeric
    sector_df = sector_df.apply(pd.to_numeric, errors='coerce')
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    gpr_df = gpr_df.apply(pd.to_numeric, errors='coerce')

    # Column name lists
    sector_columns = list(sector_df.columns)
    feature_columns = list(features_df.columns)

    # Inner join on dates
    df_master = sector_df.join(features_df, how='inner')

    return df_master, sector_columns, feature_columns, gpr_df


def add_targets(df, sector_columns):
    # Targets and returns
    sector_prices = df[sector_columns]
    sector_returns = sector_prices.pct_change()
    sector_returns.columns = [f"{c}_return" for c in sector_columns]

    # Next-day return and direction
    y_reg = sector_returns.shift(-1).copy()
    y_reg.columns = [f"y_reg_{c.replace('_return','')}" for c in sector_returns.columns]

    y_clf = (y_reg > 0).astype('Int64')
    y_clf.columns = [c.replace("y_reg_", "y_clf_") for c in y_reg.columns]

    df = pd.concat([df, sector_returns, y_reg, y_clf], axis=1)
    return df


def add_time_features(df):
    # Time features
    idx = df.index
    df['day_of_week'] = idx.dayofweek
    df['month_of_year'] = idx.month
    df['quarter'] = idx.quarter
    df['is_month_start'] = idx.is_month_start.astype(int)
    df['is_month_end'] = idx.is_month_end.astype(int)
    return df


def add_realized_vol_features(df, return_columns_list):
    # Realized volatility on shifted returns (no leakage)
    valid_cols = [c for c in return_columns_list if c in df.columns]
    if not valid_cols:
        return df

    shifted_returns = df[valid_cols].shift(1)
    frames = []
    for w in [5, 21]:
        vol = shifted_returns.rolling(window=w, min_periods=w).std()
        vol.columns = [f"{c}_realized_vol_{w}" for c in valid_cols]
        frames.append(vol)
    if frames:
        df = pd.concat([df] + frames, axis=1)
    return df


def add_feature_returns(df, feature_columns):
    # Feature returns
    valid_feats = [f for f in feature_columns if f in df.columns]
    if not valid_feats:
        return df
    feat_returns = df[valid_feats].pct_change()
    feat_returns.columns = [f"{c}_return" for c in valid_feats]
    df = pd.concat([df, feat_returns], axis=1)
    return df


def add_lags_and_rolling(df, feature_list):
    # Lags and rolling stats (use shift(1) to avoid leakage)
    valid_feats = [f for f in feature_list if f in df.columns]
    if not valid_feats:
        return df

    # Lags (vectorized)
    lag_frames = []
    for lag in LAG_DAYS:
        lag_df = df[valid_feats].shift(lag)
        lag_df.columns = [f"{c}_lag{lag}" for c in valid_feats]
        lag_frames.append(lag_df)

    # Rolling mean/std on shifted series
    shifted = df[valid_feats].shift(1)
    roll_frames = []
    for w in ROLLING_WINDOWS:
        mean_df = shifted.rolling(window=w, min_periods=w).mean()
        std_df = shifted.rolling(window=w, min_periods=w).std()
        mean_df.columns = [f"{c}_roll{w}_mean" for c in valid_feats]
        std_df.columns = [f"{c}_roll{w}_std" for c in valid_feats]
        roll_frames.extend([mean_df, std_df])

    df = pd.concat([df] + lag_frames + roll_frames, axis=1)
    return df


def handle_gpr(df_master, gpr_df):
    # Resample daily, ffill, join to master
    gpr_daily = gpr_df.resample('D').ffill()
    df_out = df_master.join(gpr_daily, how='left')

    gpr_col_name = None
    if 'GPR' in df_out.columns:
        gpr_col_name = 'GPR'
    elif gpr_daily.shape[1] == 1:
        gpr_col_name = gpr_daily.columns[0]
        df_out.rename(columns={gpr_col_name: 'GPR'}, inplace=True)

    if gpr_col_name:
        df_out['GPR'] = df_out['GPR'].ffill()
    else:
        raise ValueError("'GPR' column not found and no single-column fallback.")

    return df_out


def add_interactions(df):
    # Interactions on *_lag1 only
    try:
        # Products
        df['VIX_x_EPU_lag1'] = df['VIX_lag1'] * df['EPU_lag1']
        df['VIX_x_GPR_lag1'] = df['VIX_lag1'] * df['GPR_lag1']
        df['EPU_x_GPR_lag1'] = df['EPU_lag1'] * df['GPR_lag1']

        # 3-way
        df['VIX_x_EPU_x_GPR_lag1'] = df['VIX_lag1'] * df['EPU_lag1'] * df['GPR_lag1']

        # Ratios (safe denom)
        df['EPU_div_VIX_lag1'] = df['EPU_lag1'] / (df['VIX_lag1'] + 1e-6)
        df['GPR_div_VIX_lag1'] = df['GPR_lag1'] / (df['VIX_lag1'] + 1e-6)

        # Spread
        df['GOLD_minus_OIL_lag1'] = df['GOLD_lag1'] - df['OIL_lag1']

    except KeyError as e:
        print(f"[Warning] Missing column for interactions: {e}")

    return df


def add_hurst_features(df, feature_list):
    """Rolling Hurst exponent for selected columns over multiple windows.
    Outputs <col>_hurst_<W> for W in {21,63,126,252}. Lags are added later.
    """
    if not HURST_AVAILABLE:
        return df

    HURST_WINDOWS = [21, 63, 126, 252]

    def _safe_hurst(x: np.ndarray) -> float:
        try:
            h, _, _ = compute_Hc(x)
            return float(h) if np.isfinite(h) else np.nan
        except Exception:
            return np.nan

    print("Calculating rolling Hurst Exponent (multi-window; this will be slow)...")

    out_frames = []
    for col in feature_list:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors='coerce')
        for W in HURST_WINDOWS:
            h = series.rolling(window=W, min_periods=W).apply(_safe_hurst, raw=True)
            h.name = f"{col}_hurst_{W}"
            out_frames.append(h)

    if out_frames:
        df = pd.concat([df] + out_frames, axis=1)
        # Drop Hurst columns that are entirely NaN (prevent row wipeout at dropna)
        hurst_cols_added = [s.name for s in out_frames]
        all_nan_cols = [c for c in hurst_cols_added if df[c].isna().all()]
        if all_nan_cols:
            print(f"[HURST] Dropping {len(all_nan_cols)} all-NaN Hurst columns")
            df.drop(columns=all_nan_cols, inplace=True)
    return df


def final_cleanup_and_save(df):
    # Replace inf, drop NaNs, save
    df = df.replace([np.inf, -np.inf], np.nan)
    df_final = df.dropna()
    df_final.index.name = 'Date'
    df_final.to_csv(OUTPUT_PATH)

    return df_final


def main():
    print("--- Feature engineering started ---")
    # Load
    df_master, sector_columns, feature_columns, gpr_df = load_inputs()
    
    # Add Sector Ratio (Economic Feature)
    if 'XLY' in df_master.columns and 'XLP' in df_master.columns:
        print("Calculating XLY/XLP Sector Ratio feature...")
        df_master['XLY_div_XLP'] = df_master['XLY'] / (df_master['XLP'] + 1e-6)
    else:
        print("Warning: XLY or XLP not found. Skipping Sector Ratio feature.")
    
    # Targets
    df_master = add_targets(df_master, sector_columns)
    # Base features
    df_master = add_time_features(df_master)
    df_master = add_feature_returns(df_master, feature_columns)
    all_return_cols = [c for c in df_master.columns if c.endswith('_return')]
    df_master = add_realized_vol_features(df_master, all_return_cols)
    # GPR
    df_master = handle_gpr(df_master, gpr_df)
    # Step 4.5: Rolling Hurst (momentum/memory) features
    hurst_cols = feature_columns + sector_columns
    df_master = add_hurst_features(df_master, hurst_cols)
    
    # Register new features in feature_columns before building systematic list
    if 'XLY_div_XLP' in df_master.columns:
        feature_columns.append('XLY_div_XLP')
    
    # Build feature list
    systematic_feature_list = feature_columns.copy()
    if 'GPR' in df_master.columns:
        systematic_feature_list.append('GPR')
    systematic_feature_list.extend([col for col in df_master.columns if '_return' in col])
    systematic_feature_list.extend([col for col in df_master.columns if '_realized_vol_' in col])
    systematic_feature_list.extend([col for col in df_master.columns if '_hurst_' in col])

    print(f"Total features to process for lags/rolling: {len(systematic_feature_list)}")
    # Lags & rolling
    df_master = add_lags_and_rolling(df_master, systematic_feature_list)
    # Interactions (after _lag1 exists)
    df_master = add_interactions(df_master)
    # Clean & save
    df_final = final_cleanup_and_save(df_master)

    print(f"--- Feature engineering complete: {OUTPUT_PATH}")
    print(f"Rows: {len(df_final):,} | Columns: {len(df_final.columns):,}")


if __name__ == '__main__':
    main()