import os
import pandas as pd
import yfinance as yf

SECTOR_DATA_PATH = os.path.join("data", "sector_data.csv")
FEATURES_DATA_PATH = os.path.join("data", "features_data.csv")
EPU_DATA_PATH = os.path.join("data", "EPU.csv")
GPR_DATA_PATH = os.path.join("data", "GPR.csv")
SENTIMENT_DATA_PATH = os.path.join("data", "sentiment.csv")

# --- ETF Tickets---
MARKET_TICKERS = ['^VIX', 'GC=F', 'CL=F', 'DX=F', '^TNX', '^IRX']

# --- Column Names ---
COLUMN_NAMES = {
    '^VIX': 'VIX',
    'GC=F': 'GOLD',
    'CL=F': 'OIL',
    'DX=F': 'USD',
    '^TNX': 'TNX',
    '^IRX': 'IRX'
}

def load_all_daily_features(master_index, start_date, end_date):
    print("[JOB 1/2] Downloading market data...")
    try:
        data_yf = yf.download(
            tickers=MARKET_TICKERS,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        if data_yf.empty:
            print("[Error] No market data could be downloaded.")
            return False
            
        market_data = data_yf['Close'].rename(columns=COLUMN_NAMES)
        
        if market_data.index.tz is not None:
            market_data.index = market_data.index.tz_localize(None)

        print("[JOB 2/2] Loading EPU and GPR data...")
        if not os.path.exists(EPU_DATA_PATH):
            print(f"[Error] EPU file not found at: {EPU_DATA_PATH}")
            return False
        
        if not os.path.exists(GPR_DATA_PATH):
            print(f"[Error] GPR file not found at: {GPR_DATA_PATH}")
            return False
        
        if not os.path.exists(SENTIMENT_DATA_PATH):
            print(f"[Error] Sentiment file not found at: {SENTIMENT_DATA_PATH}")
            return False
            
        # EPU: Daily data
        epu_data = pd.read_csv(EPU_DATA_PATH, parse_dates=['Date'], index_col='Date')
        
        if epu_data.index.duplicated().any():
            epu_data = epu_data[~epu_data.index.duplicated(keep='first')]
        
        # GPR: Monthly data (1st of each month) - forward fill to daily
        gpr_data = pd.read_csv(GPR_DATA_PATH, parse_dates=['Date'], index_col='Date')
        
        if gpr_data.index.duplicated().any():
            gpr_data = gpr_data[~gpr_data.index.duplicated(keep='first')]
        
        # Sentiment: Weekly AAII data - parse and forward fill to daily
        sentiment_raw = pd.read_csv(SENTIMENT_DATA_PATH)
        
        # Parse date (format: '8-23-00' -> '2000-08-23')
        sentiment_raw['Date'] = pd.to_datetime(sentiment_raw['Reported Date'], format='%m-%d-%y')
        sentiment_raw = sentiment_raw.set_index('Date').sort_index()
        
        # Parse percentages (remove '%' and convert to float)
        sentiment_data = pd.DataFrame(index=sentiment_raw.index)
        sentiment_data['Bullish'] = sentiment_raw['Bullish'].str.rstrip('%').astype(float) / 100
        sentiment_data['Neutral'] = sentiment_raw['Neutral'].str.rstrip('%').astype(float) / 100
        sentiment_data['Bearish'] = sentiment_raw['Bearish'].str.rstrip('%').astype(float) / 100
        sentiment_data['Bullish_8wMA'] = sentiment_raw['Bullish 8-week Mov Avg'].str.rstrip('%').astype(float) / 100
        sentiment_data['Bull_Bear_Spread'] = sentiment_raw['Bull-Bear Spread'].str.rstrip('%').astype(float) / 100
        
        # Derived features
        sentiment_data['Bull_Bear_Ratio'] = sentiment_data['Bullish'] / (sentiment_data['Bearish'] + 1e-6)  # Avoid division by zero
        sentiment_data['Extreme_Bull'] = (sentiment_data['Bullish'] > 0.50).astype(int)  # Extreme bullishness (>50%)
        sentiment_data['Extreme_Bear'] = (sentiment_data['Bearish'] > 0.50).astype(int)  # Extreme bearishness (>50%)
        
        if sentiment_data.index.duplicated().any():
            sentiment_data = sentiment_data[~sentiment_data.index.duplicated(keep='first')]
        
        print(f"  Sentiment data: {len(sentiment_data)} weeks ({sentiment_data.index.min()} to {sentiment_data.index.max()})")

        # Reindex and align to master_index
        market_features = market_data.reindex(master_index).ffill()
        epu_features = epu_data.reindex(master_index)
        gpr_features = gpr_data.reindex(master_index).ffill().bfill()  # Forward fill GPR monthly to daily, backfill initial NaNs
        sentiment_features = sentiment_data.reindex(master_index).ffill().bfill()  # Forward fill weekly sentiment to daily
        
        final_features = pd.concat([market_features, epu_features, gpr_features, sentiment_features], axis=1)
        final_features['SPREAD_10Y_13W'] = final_features['TNX'] - final_features['IRX']
        
        final_features.to_csv(FEATURES_DATA_PATH)
        print(f"[JOB 2/2] Features saved to {FEATURES_DATA_PATH}")
        print(f"  Market features: {len(market_features.columns)}")
        print(f"  EPU features: {len(epu_features.columns)}")
        print(f"  GPR features: {len(gpr_features.columns)}")
        print(f"  Sentiment features: {len(sentiment_features.columns)}")
        print(f"  Total features: {len(final_features.columns)}")
        return True

    except Exception as e:
        print(f"[Error] Failed to process features: {e}")
        return False

# --- Main Program ---
if __name__ == '__main__':
    print("--- Features processing started ---")
    try:
        if not os.path.exists(SECTOR_DATA_PATH):
            print(f"[Fatal Error] File not found: {SECTOR_DATA_PATH}")
            print("Please run data_load.py first.")
        else:
            sector_data = pd.read_csv(SECTOR_DATA_PATH, parse_dates=['Date'], index_col='Date')
            master_index = sector_data.index
            start_date = master_index.min()
            end_date = master_index.max()
            
            print(f"Master index loaded: {len(master_index)} trading days.")
            print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            success = load_all_daily_features(master_index, start_date, end_date)
            
            if success:
                print("--- Features processed successfully ---")
            else:
                print("--- Pipeline finished with errors ---")
                
    except Exception as e:
        print(f"[Error] Main pipeline failed: {e}")
    
    print("--- Done ---")