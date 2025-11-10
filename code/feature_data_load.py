import os
import pandas as pd
import yfinance as yf

SECTOR_DATA_PATH = os.path.join("data", "sector_data.csv")
FEATURES_DATA_PATH = os.path.join("data", "features_data.csv")
EPU_DATA_PATH = os.path.join("data", "EPU.csv")
GPR_DATA_PATH = os.path.join("data", "GPR.csv")

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
            
        # EPU: Daily data
        epu_data = pd.read_csv(EPU_DATA_PATH, parse_dates=['Date'], index_col='Date')
        
        if epu_data.index.duplicated().any():
            epu_data = epu_data[~epu_data.index.duplicated(keep='first')]
        
        # GPR: Monthly data (1st of each month) - forward fill to daily
        gpr_data = pd.read_csv(GPR_DATA_PATH, parse_dates=['Date'], index_col='Date')
        
        if gpr_data.index.duplicated().any():
            gpr_data = gpr_data[~gpr_data.index.duplicated(keep='first')]

        # Reindex and align to master_index
        market_features = market_data.reindex(master_index).ffill()
        epu_features = epu_data.reindex(master_index)
        gpr_features = gpr_data.reindex(master_index).ffill().bfill()  # Forward fill GPR monthly to daily, backfill initial NaNs
        
        final_features = pd.concat([market_features, epu_features, gpr_features], axis=1)
        final_features['SPREAD_10Y_13W'] = final_features['TNX'] - final_features['IRX']
        
        final_features.to_csv(FEATURES_DATA_PATH)
        print(f"[JOB 2/2] Features saved to {FEATURES_DATA_PATH}")
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