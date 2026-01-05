from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW = 63
MIN_PERIODS = 40


def _find_mm_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        data_dir = candidate / "data"
        if (data_dir / "raw_data_2").exists() and (data_dir / "raw_data_3").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate ModularMonolith root containing data/raw_data_2 and data/raw_data_3. "
        f"Started search from: {start}"
    )


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def _require_columns(df: pd.DataFrame, path: Path, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")


def _load_spdr_prices(spdr_path: Path) -> pd.DataFrame:
    _require_file(spdr_path)

    df = pd.read_parquet(spdr_path)
    _require_columns(df, spdr_path, ["Date"])

    df["Date"] = pd.to_datetime(df["Date"], utc=False)
    df = df.sort_values("Date")

    sector_cols = [c for c in df.columns if c != "Date"]
    if not sector_cols:
        raise ValueError(f"No sector columns found in {spdr_path}")

    prices = df[["Date", *sector_cols]].copy()
    log_prices = np.log(prices[sector_cols])
    log_rets = log_prices.diff()

    out = pd.concat([prices[["Date"]], log_rets], axis=1)
    out = out.rename(columns={c: f"{c}_logret" for c in sector_cols})
    return out


def _load_macro(raw_data_3_dir: Path) -> pd.DataFrame:
    futures_path = raw_data_3_dir / "Futures.parquet"
    tlt_path = raw_data_3_dir / "TLT.parquet"

    _require_file(futures_path)
    _require_file(tlt_path)

    futures = pd.read_parquet(futures_path)
    _require_columns(futures, futures_path, ["Date", "^TNX_diff", "CL=F_diff", "DX-Y.NYB_logret"])
    futures = futures[["Date", "^TNX_diff", "CL=F_diff", "DX-Y.NYB_logret"]].copy()

    tlt = pd.read_parquet(tlt_path)
    _require_columns(tlt, tlt_path, ["Date", "TLT_logret"])
    tlt = tlt[["Date", "TLT_logret"]].copy()

    for df in (futures, tlt):
        df["Date"] = pd.to_datetime(df["Date"], utc=False)

    macro = futures.merge(tlt, on="Date", how="outer")
    macro = macro.sort_values("Date")
    return macro


def _rolling_corr_against_drivers(
    sector_logrets: pd.DataFrame,
    macro: pd.DataFrame,
    window: int = WINDOW,
    min_periods: int = MIN_PERIODS,
) -> list[pd.DataFrame]:
    sector_cols = [c for c in sector_logrets.columns if c != "Date"]
    sectors = [c.removesuffix("_logret") for c in sector_cols]

    df = sector_logrets.merge(macro, on="Date", how="left")
    df = df.sort_values("Date")
    sector_ret = df.set_index("Date")[sector_cols]

    drivers: dict[str, str] = {
        "rates": "^TNX_diff",
        "oil": "CL=F_diff",
        "usd": "DX-Y.NYB_logret",
        "bonds": "TLT_logret",
    }

    long_frames: list[pd.DataFrame] = []

    for driver_name, driver_col in drivers.items():
        driver = df.set_index("Date")[driver_col]

        corr_df = sector_ret.rolling(window=window, min_periods=min_periods).corr(driver)
        corr_df = corr_df.rename(columns={c: c.removesuffix("_logret") for c in corr_df.columns})

        feature = f"corr_{driver_name}_{window}"
        long_frames.append(_to_long_feature(corr_df, feature))

    roll_corr = sector_ret.rolling(window=window, min_periods=min_periods).corr()
    roll_corr = roll_corr.rename(columns={c: c.removesuffix("_logret") for c in roll_corr.columns})

    idx = pd.IndexSlice
    for s in sectors:
        if s in roll_corr.columns:
            roll_corr.loc[idx[:, f"{s}_logret"], s] = np.nan

    mean_corr_others = roll_corr.mean(axis=1, skipna=True)
    mean_corr_others.index.names = ["Date", "Sector_logret"]
    mean_corr_others = mean_corr_others.rename("value").reset_index()
    mean_corr_others["Sector"] = mean_corr_others["Sector_logret"].str.replace("_logret", "", regex=False)
    mean_corr_others = mean_corr_others.drop(columns=["Sector_logret"])
    mean_corr_others["Feature"] = f"mean_corr_others_{window}"
    mean_corr_others = mean_corr_others[["Date", "Sector", "Feature", "value"]]

    long_frames.append(mean_corr_others)

    return long_frames


def _to_long_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp = tmp.reset_index(names="Date")
    out = tmp.melt(id_vars=["Date"], var_name="Sector", value_name="value")
    out["Feature"] = feature
    return out[["Date", "Sector", "Feature", "value"]]


def main() -> None:
    mm_root = _find_mm_root(Path(__file__))
    data_dir = mm_root / "data"

    raw_data_2_dir = data_dir / "raw_data_2"
    raw_data_3_dir = data_dir / "raw_data_3"
    out_dir = data_dir / "processed_data_2"

    spdr_path = raw_data_2_dir / "SPDR.parquet"

    _require_file(spdr_path)
    if not raw_data_3_dir.exists():
        raise FileNotFoundError(f"Required input directory not found: {raw_data_3_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    sector_logrets = _load_spdr_prices(spdr_path)
    macro = _load_macro(raw_data_3_dir)

    long_frames = _rolling_corr_against_drivers(sector_logrets, macro, window=WINDOW, min_periods=MIN_PERIODS)
    features_long = pd.concat(long_frames, ignore_index=True)

    features_long = features_long.sort_values(["Date", "Sector", "Feature"], kind="mergesort")

    out_path = out_dir / "features_correlations.parquet"
    features_long.to_parquet(out_path, index=False)

    print(f"Wrote {len(features_long):,} rows to: {out_path}")

if __name__ == "__main__":
    main()