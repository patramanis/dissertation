from __future__ import annotations

import pandas as pd

TRAIN_DATE_START = pd.Timestamp("2005-09-08")
TRAIN_DATE_END = pd.Timestamp("2025-09-30")

def mask_train_window(dates: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.Index:
    if isinstance(dates, pd.DatetimeIndex):
        idx = dates
        return (idx >= TRAIN_DATE_START) & (idx <= TRAIN_DATE_END)

    dt = pd.to_datetime(dates, errors="raise")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return (dt >= TRAIN_DATE_START) & (dt <= TRAIN_DATE_END)


def filter_train_window_df(df: pd.DataFrame, *, date_col: str = "Date", name: str = "df") -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"[{name}] Missing required column: {date_col}")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise")
    if getattr(out[date_col].dt, "tz", None) is not None:
        out[date_col] = out[date_col].dt.tz_localize(None)
    m = (out[date_col] >= TRAIN_DATE_START) & (out[date_col] <= TRAIN_DATE_END)
    return out.loc[m].copy()
