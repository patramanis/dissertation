from __future__ import annotations
import numpy as np
import pandas as pd

def assert_prices_sane_after_first_valid(
    prices: pd.DataFrame | pd.Series,
    *,
    name: str,
) -> None:
    if isinstance(prices, pd.Series):
        s = prices.astype("float64")
        first = s.first_valid_index()
        if first is None:
            raise ValueError(f"{name}: all values are NaN")
        tail = s.loc[first:]
        vals = tail.to_numpy()
        finite = np.isfinite(vals)
        if not finite.all():
            raise ValueError(f"{name}: non-finite values after first_valid")
        if not (vals[finite] > 0).all():
            raise ValueError(f"{name}: non-positive values after first_valid")
        return

    df = prices.astype("float64")
    bad_all_nan = [c for c in df.columns if df[c].first_valid_index() is None]
    if bad_all_nan:
        raise ValueError(f"{name}: all-NaN columns: {bad_all_nan}")

    for c in df.columns:
        first = df[c].first_valid_index()
        if first is None:
            continue
        tail = df[c].loc[first:]
        vals = tail.to_numpy()
        finite = np.isfinite(vals)
        if not finite.all():
            raise ValueError(f"{name}: non-finite values after first_valid in {c}")
        if not (vals[finite] > 0).all():
            raise ValueError(f"{name}: non-positive values after first_valid in {c}")

def safe_log_returns(prices: pd.DataFrame | pd.Series[float], *, name: str = "prices") -> pd.DataFrame | pd.Series[float]:
    assert_prices_sane_after_first_valid(prices, name=name)
    if isinstance(prices, pd.DataFrame):
        log_prices = prices.apply(np.log)
    else:
        log_prices = pd.Series(np.log(prices.values), index=prices.index, name=prices.name)
    return log_prices.diff()