from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd

MacroMonthlyName = Literal["CPIAUCSL", "INDPRO", "UNRATE"]

def _ensure_inputs(monthly_aligned: pd.DataFrame, trading_calendar: pd.DatetimeIndex) -> tuple[str, pd.Series]:
    if monthly_aligned.index.name != "Date" or not isinstance(monthly_aligned.index, pd.DatetimeIndex):
        raise ValueError("monthly_aligned must be indexed by Date (DatetimeIndex, name='Date')")
    if len(monthly_aligned.columns) != 1:
        raise ValueError("monthly_aligned must have exactly one column")
    if not isinstance(trading_calendar, pd.DatetimeIndex) or trading_calendar.name != "Date":
        trading_calendar = pd.DatetimeIndex(trading_calendar, name="Date")

    col = str(monthly_aligned.columns[0])
    level = monthly_aligned[col].astype("float64")
    return col, level

def _sparse_update_points(level: pd.Series) -> pd.Series:
    s = level
    updates = s.notna() & (s.ne(s.shift(1)) | s.shift(1).isna())
    return s.loc[updates]

def _logdiff(s: pd.Series, periods: int) -> pd.Series:
    values = s.to_numpy(dtype="float64", copy=False)
    logged = np.full(shape=values.shape, fill_value=np.nan, dtype="float64")
    mask = np.isfinite(values) & (values > 0)
    if mask.any():
        logged[mask] = np.log(values[mask])
    logged_s = pd.Series(logged, index=s.index, dtype="float64")
    return logged_s.diff(periods).astype("float64")

def build_monthly_growth(
    monthly_aligned: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    *,
    series_name: str,
    monthly_sparse_aligned: pd.DataFrame | None = None,
) -> pd.DataFrame:
    col, level = _ensure_inputs(monthly_aligned, trading_calendar)

    if monthly_sparse_aligned is not None:
        sparse_col, sparse_level = _ensure_inputs(monthly_sparse_aligned, trading_calendar)
        if sparse_col != col:
            raise ValueError(f"Sparse column mismatch: expected {col}, got {sparse_col}")
        base = sparse_level.loc[sparse_level.notna()]
    else:
        sparse = _sparse_update_points(level)
        base = sparse if sparse.dropna().shape[0] >= 3 else level

    if series_name in {"CPIAUCSL", "INDPRO"}:
        mom_sparse = _logdiff(base, 1)
        yoy_sparse = _logdiff(base, 12)
    elif series_name == "UNRATE":
        mom_sparse = base.diff(1).astype("float64")
        yoy_sparse = base.diff(12).astype("float64")
    else:
        raise ValueError(f"Unknown macro monthly series: {series_name}")

    mom_sparse.name = f"{col}_mom"
    yoy_sparse.name = f"{col}_yoy"

    mom = mom_sparse.reindex(trading_calendar).ffill(limit=30).astype("float64")
    yoy = yoy_sparse.reindex(trading_calendar).ffill(limit=30).astype("float64")

    out = pd.concat([mom, yoy], axis=1)
    out.index = pd.DatetimeIndex(out.index, name="Date")
    out = out[[f"{col}_mom", f"{col}_yoy"]]
    return out