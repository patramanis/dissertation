from __future__ import annotations
import numpy as np
import pandas as pd

def _normalize_prices_close(prices_close: pd.DataFrame, *, sectors: list[str]) -> pd.DataFrame:
    if "Date" not in prices_close.columns:
        raise ValueError("prices_close missing Date column")

    required = ["Date", "SPY", *sectors]
    missing = [c for c in required if c not in prices_close.columns]
    if missing:
        raise ValueError(f"prices_close missing required columns: {missing}")

    df = prices_close.loc[:, required].copy()

    dt = pd.to_datetime(df["Date"], errors="raise")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    df["Date"] = dt.dt.normalize()

    df = df.sort_values("Date", kind="mergesort")
    if df["Date"].duplicated().any():
        dup = df.loc[df["Date"].duplicated(keep=False), "Date"].value_counts().head(5)
        examples = ", ".join([f"{pd.Timestamp(str(d)).date()}(x{c})" for d, c in dup.items()])
        raise ValueError(f"Duplicate dates in prices_close: {examples}")

    for c in ["SPY", *sectors]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")

    price_cols = ["SPY", *sectors]
    vals = df[price_cols]

    if vals.isna().any().any():
        bad_pos = int(np.flatnonzero(vals.isna().any(axis=1).to_numpy(dtype=bool))[0])
        bad_date = df["Date"].iloc[bad_pos]
        raise AssertionError(
            f"prices_close contains NaNs at row position {bad_pos}, date={pd.Timestamp(bad_date).date()}"
        )

    arr = vals.to_numpy(dtype="float64", copy=False)
    row_ok = np.isfinite(arr).all(axis=1)
    if not row_ok.all():
        bad_pos = int(np.flatnonzero(~row_ok)[0])
        bad_date = df["Date"].iloc[bad_pos]
        raise AssertionError(
            f"prices_close contains non-finite values at row position {bad_pos}, date={pd.Timestamp(bad_date).date()}"
        )

    row_ok = (arr > 0).all(axis=1)
    if not row_ok.all():
        bad_pos = int(np.flatnonzero(~row_ok)[0])
        bad_date = df["Date"].iloc[bad_pos]
        raise AssertionError(
            f"prices_close contains non-positive values at row position {bad_pos}, date={pd.Timestamp(bad_date).date()}"
        )

    return df

def build_excess_labels(
    prices_close: pd.DataFrame,
    horizons: list[int],
    sectors: list[str],
) -> dict[int, pd.DataFrame]:

    if not isinstance(horizons, list) or not horizons:
        raise ValueError("horizons must be a non-empty list")

    hs: list[int] = []
    for h in horizons:
        ih = int(h)
        if ih <= 0:
            raise ValueError(f"Invalid horizon: {h}")
        hs.append(ih)

    if not isinstance(sectors, list) or not sectors:
        raise ValueError("sectors must be a non-empty list")

    df = _normalize_prices_close(prices_close, sectors=sectors)

    px = df.set_index("Date")[["SPY", *sectors]].astype("float64")

    log_px: pd.DataFrame = pd.DataFrame(np.log(px), index=px.index, columns=px.columns)

    out_by_h: dict[int, pd.DataFrame] = {}
    n = len(px)

    for h in hs:
        log_ret = log_px.shift(-h) - log_px

        spy_log_ret = log_ret["SPY"].astype("float64")
        sec_log_ret = log_ret[sectors].astype("float64")

        label_excess = sec_log_ret.sub(spy_log_ret, axis=0)

        long = label_excess.reset_index().melt(
            id_vars=["Date"],
            var_name="Sector",
            value_name="label_excess"
        )
        long["Sector"] = pd.Categorical(
            long["Sector"].astype(str),
            categories=list(sectors),
            ordered=True
        )
        long = long.sort_values(["Date", "Sector"], kind="mergesort").reset_index(drop=True)
        long["Sector"] = long["Sector"].astype(str)

        long["horizon"] = int(h)

        long["label_excess"] = pd.to_numeric(long["label_excess"], errors="coerce").astype("float64")

        expected_rows = n * len(sectors)
        if len(long) != expected_rows:
            raise AssertionError(f"Label output rowcount mismatch: got {len(long)}, expected {expected_rows}")

        long = long[["Date", "Sector", "horizon", "label_excess"]]
        long["Sector"] = pd.Categorical(
            long["Sector"].astype(str),
            categories=list(sectors),
            ordered=True
        )
        long = long.sort_values(["Date", "Sector"], kind="mergesort").reset_index(drop=True)
        long["Sector"] = long["Sector"].astype(str)

        out_by_h[h] = long

    return out_by_h