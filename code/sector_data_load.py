import os
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime

# --- ETF Tickers---
TICKERS = ['SPY', 'XLK', 'XLE', 'XLF', 'XLV', 'XLU', 'XLI', 'XLB', 'XLY', 'XLP']
START_DATE = '2000-08-30'
END_DATE = '2025-09-30'
DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, "sector_data.csv")

# Get NYSE calendar
nyse = mcal.get_calendar('NYSE')

def get_trading_days():
    schedule = nyse.schedule(start_date=START_DATE, end_date=END_DATE)
    trading_days = pd.DatetimeIndex(schedule.index).normalize()
    return trading_days

def download_data():
    print("[JOB 1/2] Downloading data...")

    try:
        trading_days = get_trading_days()
        
        data_yf = yf.download(
            tickers=TICKERS,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=False
        )
        
        if data_yf.empty:
            print("[Error] No data found to download.")
            return None

        data_yf_close = data_yf['Adj Close']
        
        if data_yf_close.index.tz is not None:
            data_yf_close.index = data_yf_close.index.tz_localize(None)
        
        valid_days = data_yf_close.index.intersection(trading_days)
        data_yf_close = data_yf_close.loc[valid_days]
        
        if data_yf_close.empty:
            print("[Error] No valid trading days found in data.")
            return None
            
        print(f"[JOB 1/2] Download completed. Got {len(data_yf_close)} trading days.")
        return data_yf_close

    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return None

def save_data(df):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        
        df.index.name = 'Date'
        df.to_csv(FILE_PATH)
        print(f"[JOB 2/2] Data saved successfully at: {FILE_PATH}")
        return True
    except Exception as e:
        print(f"[Error] Could not save file: {e}")
        return False

# --- Main Program ---
if __name__ == '__main__':
    print("--- ETF download started ---")
    
    df_new = download_data()
    
    if df_new is not None:
        save_data(df_new)
    
    print("--- Done ---")