import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# RSI (Relative Strength Index)
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)

# MACD (Moving Average Convergence Divergence)
def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd.fillna(0), signal_line.fillna(0), histogram.fillna(0)

# Stochastic Oscillator
def calculate_stochastic(high, low, close, period=14, smooth_k=3):
    lowest_low = low.rolling(window=period, min_periods=1).min()
    highest_high = high.rolling(window=period, min_periods=1).max()
    
    k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k = k_raw.rolling(window=smooth_k, min_periods=1).mean()
    d = k.rolling(window=3, min_periods=1).mean()
    
    return k.fillna(50), d.fillna(50)

# Bollinger Bands
def calculate_bollinger(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period, min_periods=1).mean()
    std = prices.rolling(window=period, min_periods=1).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    bandwidth = (upper - lower) / (sma + 1e-10)
    
    position = (prices - lower) / (upper - lower + 1e-10)
    position = position.clip(0, 1)
    
    return (upper.fillna(prices), 
            lower.fillna(prices), 
            bandwidth.fillna(0), 
            position.fillna(0.5))

# ADX (Average Directional Index)
def calculate_adx(high, low, close, period=14):
    # Directional movement
    plus_dm = high.diff().where(lambda x: (x > 0) & (x > (-low.diff())), 0)
    minus_dm = (-low.diff()).where(lambda x: (x > 0) & (x > high.diff()), 0)
    
    # True range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
    
    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

# OBV Proxy (uses volatility as volume)
def calculate_obv_proxy(close, volatility):
    price_change = close.pct_change().fillna(0)
    obv = (np.sign(price_change) * volatility).cumsum()
    
    return obv.fillna(0)

# VWAP Proxy (uses volatility as volume weights)
def calculate_vwap_proxy(close, volatility, period=20):
    vwap = (close * volatility).rolling(window=period, min_periods=1).sum() / \
           (volatility.rolling(window=period, min_periods=1).sum() + 1e-10)
    
    return vwap.fillna(close)

# Bid-Ask Spread Proxy
def calculate_spread_proxy(high, low, close):
    spread = (high - low) / (close + 1e-10)
    return spread.fillna(0)

# Amihud Illiquidity Measure (Proxy)
def calculate_illiquidity(close, volatility):
    returns = close.pct_change().abs()
    illiquidity = returns / (volatility + 1e-10)
    
    return illiquidity.fillna(0)

# Price Impact Measure
def calculate_price_impact(close, volatility, period=5):
    returns = close.pct_change()
    cum_return = returns.rolling(window=period, min_periods=1).sum()
    cum_volume = volatility.rolling(window=period, min_periods=1).sum()
    
    impact = cum_return / (cum_volume + 1e-10)
    
    return impact.fillna(0)

# Create high/low proxy from close prices
def create_high_low_proxy(close, volatility_pct=0.01):
    high = close * (1 + volatility_pct)
    low = close * (1 - volatility_pct)
    
    return high, low

# Add all technical indicators for a single asset
def add_technical_indicators(df, asset_col, vix_col='VIX', 
                             rsi_periods=[14, 21], 
                             bb_period=20,
                             adx_period=14):
    result = df.copy()
    close = df[asset_col]
    volatility = df[vix_col] if vix_col in df.columns else pd.Series(1, index=df.index)
    
    high, low = create_high_low_proxy(close)
    
    # RSI
    for period in rsi_periods:
        result[f'{asset_col}_rsi{period}'] = calculate_rsi(close, period=period)
    
    # MACD
    macd, signal, histogram = calculate_macd(close)
    result[f'{asset_col}_macd'] = macd
    result[f'{asset_col}_macd_signal'] = signal
    result[f'{asset_col}_macd_hist'] = histogram
    
    # Bollinger Bands
    upper, lower, bandwidth, position = calculate_bollinger(close, period=bb_period)
    result[f'{asset_col}_bb_bandwidth'] = bandwidth
    result[f'{asset_col}_bb_position'] = position
    
    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    result[f'{asset_col}_stoch_k'] = stoch_k
    result[f'{asset_col}_stoch_d'] = stoch_d
    
    # ADX
    adx, plus_di, minus_di = calculate_adx(high, low, close)
    result[f'{asset_col}_adx'] = adx
    result[f'{asset_col}_plus_di'] = plus_di
    result[f'{asset_col}_minus_di'] = minus_di
    
    # Volume proxies
    result[f'{asset_col}_obv_proxy'] = calculate_obv_proxy(close, volatility)
    result[f'{asset_col}_vwap_proxy'] = calculate_vwap_proxy(close, volatility)
    
    # Microstructure
    result[f'{asset_col}_spread_proxy'] = calculate_spread_proxy(high, low, close)
    result[f'{asset_col}_illiquidity'] = calculate_illiquidity(close, volatility)
    result[f'{asset_col}_price_impact'] = calculate_price_impact(close, volatility)
    
    return result

def count_indicators_per_asset():
    return {
        'rsi': 2,
        'macd': 3,
        'bollinger': 2,
        'stochastic': 2,
        'adx': 3,
        'volume_proxy': 2,
        'microstructure': 3,
        'total': 17
    }

if __name__ == "__main__":
    print("Technical Indicators Module")
    print("=" * 50)
    print("\nIndicators per asset:")
    counts = count_indicators_per_asset()
    for name, count in counts.items():
        if name != 'total':
            print(f"  {name:20s}: {count:2d} indicators")
    print(f"  {'TOTAL':20s}: {counts['total']:2d} indicators per asset")
    print("\nFor 9 sectors + 10 features = 19 assets:")
    print(f"  Total new features: {19 * counts['total']} indicators")
