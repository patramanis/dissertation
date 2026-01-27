from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np
import pandas as pd

Family = Literal[
    "always_positive_price_like",
    "rate_or_spread",
    "can_be_negative_level",
    "uncertainty_or_counts",
    "macro_monthly",
]

def _ensure_series_float64(series: pd.Series) -> pd.Series:
    if series.dtype != "float64":
        return series.astype("float64")
    return series

def safe_logret(series: pd.Series) -> pd.Series:
    s = _ensure_series_float64(series)
    values = s.to_numpy(dtype="float64", copy=False)
    logged = np.full(shape=values.shape, fill_value=np.nan, dtype="float64")
    mask = np.isfinite(values) & (values > 0)
    if mask.any():
        logged[mask] = np.log(values[mask])

    logged_s = pd.Series(logged, index=s.index, name=s.name, dtype="float64")
    return logged_s.diff(1).astype("float64")

def diff1(series: pd.Series) -> pd.Series:
    return _ensure_series_float64(series).diff(1).astype("float64")

def asinh_level(series: pd.Series) -> pd.Series:
    s = _ensure_series_float64(series)
    values = s.to_numpy(dtype="float64", copy=False)
    out = pd.Series(np.arcsinh(values), index=s.index, name=s.name, dtype="float64")
    return out.astype("float64")

def asinh_diff(series: pd.Series) -> pd.Series:
    return asinh_level(series).diff(1).astype("float64")

def log1p_level(series: pd.Series) -> pd.Series:
    s = _ensure_series_float64(series)
    values = s.to_numpy(dtype="float64", copy=False)
    out_arr = np.full(shape=values.shape, fill_value=np.nan, dtype="float64")
    mask = np.isfinite(values) & (values >= 0)
    if mask.any():
        out_arr[mask] = np.log1p(values[mask])
    return pd.Series(out_arr, index=s.index, name=s.name, dtype="float64")

def dlog1p(series: pd.Series) -> pd.Series:
    return log1p_level(series).diff(1).astype("float64")

def apply_family(
    df: pd.DataFrame,
    *,
    family: Family,
    assert_non_negative: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    if df.index.name != "Date" or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected df indexed by Date (DatetimeIndex, name='Date')")

    base_cols = list(df.columns)

    if family == "macro_monthly":
        raise ValueError("macro_monthly handled in monthly_growth module")

    out: dict[str, pd.Series] = {}

    if family == "always_positive_price_like":
        for col in base_cols:
            out[f"{col}_logret"] = safe_logret(df[col])

    elif family == "rate_or_spread":
        for col in base_cols:
            out[f"{col}_diff"] = diff1(df[col])

    elif family == "can_be_negative_level":
        for col in base_cols:
            out[f"{col}_asinh"] = asinh_level(df[col])
            out[f"{col}_asinh_diff"] = asinh_diff(df[col])

    elif family == "uncertainty_or_counts":
        if assert_non_negative:
            bad_cols = [c for c in base_cols if (df[c].dropna() < 0).any()]
            if bad_cols:
                raise AssertionError(f"Expected non-negative series for log1p family: {bad_cols}")
        for col in base_cols:
            out[f"{col}_log1p"] = log1p_level(df[col])
            out[f"{col}_dlog1p"] = dlog1p(df[col])

    else:
        raise ValueError(f"Unknown family: {family}")

    feature_cols = sorted(out.keys())
    features = pd.DataFrame({c: out[c].astype("float64") for c in feature_cols}, index=df.index)
    features.index = pd.DatetimeIndex(features.index, name="Date")
    return features, feature_cols

def recognize_macro_monthly(series_name: str) -> bool:
    return series_name in {"CPIAUCSL", "UNRATE", "INDPRO"}