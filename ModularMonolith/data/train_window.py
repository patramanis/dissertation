import pandas as pd

TRAIN_DATE_START = pd.Timestamp("2000-09-29")
TRAIN_DATE_END = pd.Timestamp("2025-09-30")

def _to_naive_normalized_dates(x: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    if isinstance(x, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(x)
        if idx.tz is not None:
            idx = idx.tz_localize(None)   # no tz_convert for date-stamps
        return idx.normalize()

    dt = pd.to_datetime(x, errors="raise")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)      # no tz_convert
    return pd.DatetimeIndex(dt.dt.normalize())

def mask_train_window(dates: pd.Series | pd.DatetimeIndex):
    idx = _to_naive_normalized_dates(dates)
    return (idx >= TRAIN_DATE_START) & (idx <= TRAIN_DATE_END)

def filter_train_window_df(df: pd.DataFrame, *, date_col: str = "Date", name: str = "df") -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"[{name}] Missing required column: {date_col}")

    idx = _to_naive_normalized_dates(df[date_col])
    mask = (idx >= TRAIN_DATE_START) & (idx <= TRAIN_DATE_END)

    out = df.loc[mask].copy()
    out[date_col] = idx[mask].to_numpy()
    return out
