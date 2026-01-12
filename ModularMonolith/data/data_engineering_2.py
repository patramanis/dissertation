from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ModularMonolith.data.train_window import TRAIN_DATE_END, TRAIN_DATE_START


SECTOR_MACRO_WINDOWS: tuple[int, ...] = (21, 63, 126)
MIN_PERIODS_BY_WINDOW: dict[int, int] = {
    21: 15,
    63: 40,
    126: 80,
}

MACRO_MACRO_WINDOWS: tuple[int, ...] = (21, 63)


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


def _verify_raw2_shift_semantics(mm_root: Path) -> dict[str, object]:
    """CRITICAL: Verify raw2 shift semantics to prevent off-by-one in correlations."""
    raw1_spdr = mm_root / "data" / "raw_data_1" / "SPDR.csv"
    raw2_spdr = mm_root / "data" / "raw_data_2" / "SPDR.parquet"
    
    if not raw1_spdr.exists() or not raw2_spdr.exists():
        return {"status": "skip", "reason": "missing files"}
    
    # Load raw1 (unshifted)
    raw1 = pd.read_csv(raw1_spdr)
    raw1["Date"] = pd.to_datetime(raw1["Date"]).dt.normalize()
    raw1 = raw1.sort_values("Date").set_index("Date")
    
    # Load raw2
    raw2 = pd.read_parquet(raw2_spdr)
    raw2["Date"] = pd.to_datetime(raw2["Date"]).dt.normalize()
    raw2 = raw2.sort_values("Date").set_index("Date")
    
    # Find common dates and first common sector
    common = raw1.index.intersection(raw2.index)
    if len(common) < 10:
        return {"status": "insufficient_overlap", "n_common": len(common)}
    
    common_cols = [c for c in raw1.columns if c in raw2.columns]
    if not common_cols:
        return {"status": "no_common_columns"}
    
    test_col = common_cols[0]  # Test with first sector
    
    # Sample 20 dates
    sample_idx = np.linspace(10, len(common) - 2, min(20, len(common) - 11)).astype(int)
    sample_dates = common[sample_idx]
    
    shifted_matches = 0
    unshifted_matches = 0
    
    for date in sample_dates:
        raw2_val = raw2.loc[date, test_col]
        raw1_val_same = raw1.loc[date, test_col]
        
        prev_idx = common.get_loc(date) - 1
        if prev_idx >= 0:
            prev_date = common[prev_idx]
            raw1_val_prev = raw1.loc[prev_date, test_col]
            
            if abs(raw2_val - raw1_val_prev) < 1e-6:
                shifted_matches += 1
            if abs(raw2_val - raw1_val_same) < 1e-6:
                unshifted_matches += 1
    
    n_sample = len(sample_dates)
    result = {
        "n_tested": n_sample,
        "shifted_matches": shifted_matches,
        "unshifted_matches": unshifted_matches,
        "shifted_pct": shifted_matches / n_sample,
        "unshifted_pct": unshifted_matches / n_sample,
    }
    
    if result["shifted_pct"] > 0.9:
        result["semantics"] = "SHIFTED"
        result["interpretation"] = "raw2[t] = raw1[t-1] → correlations see PREVIOUS trading day"
    elif result["unshifted_pct"] > 0.9:
        result["semantics"] = "UNSHIFTED"
        result["interpretation"] = "raw2[t] = raw1[t] → correlations see SAME day"
    else:
        result["semantics"] = "INCONSISTENT"
        result["interpretation"] = "CRITICAL: inconsistent shift behavior"
    
    return result


def _get_trading_calendar(sector_logrets: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract and validate trading calendar from sector data."""
    dates = pd.to_datetime(sector_logrets["Date"], utc=False).sort_values()
    calendar = pd.DatetimeIndex(dates.unique())
    
    if not calendar.is_monotonic_increasing:
        raise AssertionError("Trading calendar is not monotonically increasing")
    
    if len(calendar) < len(dates):
        n_dup = len(dates) - len(calendar)
        raise AssertionError(f"Trading calendar has {n_dup} duplicate dates")
    
    return calendar


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


def _load_macro(raw_data_3_dir: Path, *, strict: bool = True) -> pd.DataFrame:
    futures_path = raw_data_3_dir / "Futures.parquet"
    tlt_path = raw_data_3_dir / "TLT.parquet"
    baml_path = raw_data_3_dir / "BAMLH0A0HYM2.parquet"

    _require_file(futures_path)
    _require_file(tlt_path)
    _require_file(baml_path)

    futures = pd.read_parquet(futures_path)
    if strict:
        _require_columns(
            futures,
            futures_path,
            ["Date", "^TNX_diff", "CL=F_asinh_diff", "DX-Y.NYB_logret", "GC=F_logret", "^VIX_dlog1p"],
        )
        futures = futures[["Date", "^TNX_diff", "CL=F_asinh_diff", "DX-Y.NYB_logret", "GC=F_logret", "^VIX_dlog1p"]].copy()
    else:
        _require_columns(futures, futures_path, ["Date"])
        wanted = ["^TNX_diff", "CL=F_asinh_diff", "DX-Y.NYB_logret", "GC=F_logret", "^VIX_dlog1p"]
        keep = [c for c in wanted if c in futures.columns]
        missing = [c for c in wanted if c not in futures.columns]
        if missing:
            print(f"WARNING: Futures.parquet missing driver columns: {missing}. Proceeding with available={keep}")
        futures = futures[["Date", *keep]].copy()

    tlt = pd.read_parquet(tlt_path)
    if strict:
        _require_columns(tlt, tlt_path, ["Date", "TLT_logret"])
        tlt = tlt[["Date", "TLT_logret"]].copy()
    else:
        _require_columns(tlt, tlt_path, ["Date"])
        keep = [c for c in ["TLT_logret"] if c in tlt.columns]
        if not keep:
            print("WARNING: TLT.parquet missing 'TLT_logret'. Proceeding without bonds driver.")
        tlt = tlt[["Date", *keep]].copy()

    for df in (futures, tlt):
        df["Date"] = pd.to_datetime(df["Date"], utc=False)

    baml = pd.read_parquet(baml_path)
    if strict:
        _require_columns(baml, baml_path, ["Date", "BAMLH0A0HYM2_diff"])
        baml = baml[["Date", "BAMLH0A0HYM2_diff"]].copy()
    else:
        _require_columns(baml, baml_path, ["Date"])
        keep = [c for c in ["BAMLH0A0HYM2_diff"] if c in baml.columns]
        if not keep:
            print("WARNING: BAMLH0A0HYM2.parquet missing 'BAMLH0A0HYM2_diff'. Proceeding without HY spread driver.")
        baml = baml[["Date", *keep]].copy()
    baml["Date"] = pd.to_datetime(baml["Date"], utc=False)

    macro = futures.merge(tlt, on="Date", how="outer")
    macro = macro.merge(baml, on="Date", how="outer")
    macro = macro.sort_values("Date")
    return macro


def _broadcast_series_to_sectors(s: pd.Series, *, sectors: list[str], feature: str, trading_calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    s = pd.to_numeric(s, errors="coerce")
    tmp = s.rename("value").to_frame()
    tmp.index = pd.to_datetime(tmp.index, utc=False)
    tmp.index.name = "Date"
    tmp = tmp.reset_index()
    if tmp.empty:
        return pd.DataFrame(columns=["Date", "Sector", "Feature", "value"])

    # FIX: Filter to trading calendar to avoid broadcasting on non-trading days
    if trading_calendar is not None:
        tmp = tmp[tmp["Date"].isin(trading_calendar)]
        if tmp.empty:
            return pd.DataFrame(columns=["Date", "Sector", "Feature", "value"])

    dates = tmp["Date"].to_numpy()
    values = tmp["value"].to_numpy()
    rep_dates = np.repeat(dates, len(sectors))
    rep_values = np.repeat(values, len(sectors))
    rep_sectors = np.tile(np.array(sectors, dtype=object), len(dates))

    out = pd.DataFrame({"Date": rep_dates, "Sector": rep_sectors, "Feature": feature, "value": rep_values})
    return out[["Date", "Sector", "Feature", "value"]]

def _macro_macro_corrs(macro: pd.DataFrame, *, windows: Iterable[int], sectors: list[str], trading_calendar: pd.DatetimeIndex) -> list[pd.DataFrame]:
    drivers: dict[str, str] = {
        "rates": "^TNX_diff",
        "oil": "CL=F_asinh_diff",
        "usd": "DX-Y.NYB_logret",
        "bonds": "TLT_logret",
        "vix": "^VIX_dlog1p",
        "hy": "BAMLH0A0HYM2_diff",
        "gold": "GC=F_logret",
    }

    df = macro.copy()
    df = df.sort_values("Date")
    df = df.set_index("Date")
    
    # CRITICAL FIX: Reindex to trading calendar so rolling window = N trading days, not calendar days
    df = df.reindex(trading_calendar)
    
    # Bounded ffill for macro-macro (prevent staleness)
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].ffill(limit=5)  # Conservative for daily drivers

    names = sorted(drivers)
    out: list[pd.DataFrame] = []

    for w in windows:
        minp = int(MIN_PERIODS_BY_WINDOW.get(int(w), max(2, int(w) // 2)))
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if drivers[a] not in df.columns or drivers[b] not in df.columns:
                    print(
                        f"WARNING: Skipping macro-macro corr {a}_{b}: "
                        f"missing {drivers[a]} or {drivers[b]}"
                    )
                    continue
                
                sa = pd.to_numeric(df[drivers[a]], errors="coerce")
                sb = pd.to_numeric(df[drivers[b]], errors="coerce")
                corr = sa.rolling(window=int(w), min_periods=minp).corr(sb)
                feat = f"corr_macro_{a}_{b}_{int(w)}"
                out.append(_broadcast_series_to_sectors(corr, sectors=sectors, feature=feat, trading_calendar=trading_calendar))

    return out


def _rolling_corr_against_drivers(
    sector_logrets: pd.DataFrame,
    macro: pd.DataFrame,
    window: int,
    min_periods: int,
    *,
    skip_missing_drivers: bool = False,
) -> list[pd.DataFrame]:
    sector_cols = [c for c in sector_logrets.columns if c != "Date"]
    bad_suffix = [c for c in sector_cols if not str(c).endswith("_logret")]
    if bad_suffix:
        raise ValueError(
            "Sector return columns must end with '_logret' to derive sector names safely. "
            f"Bad columns: {bad_suffix}"
        )
    sectors = [c.removesuffix("_logret") for c in sector_cols]

    df = sector_logrets.merge(macro, on="Date", how="left")
    df = df.sort_values("Date")
    
    # FIX: Bounded forward-fill macro (Last Known Value with staleness limit)
    # Conservative limits to prevent train/live mismatch
    macro_cols = [c for c in df.columns if c not in ["Date", *sector_cols]]
    ffill_limits = {
        "_logret": 5,      # Daily data
        "_diff": 5,        # Daily data  
        "_dlog1p": 5,      # Daily data
        "_asinh_diff": 5,  # Daily data
    }
    
    for col in macro_cols:
        if df[col].isna().any():
            # Determine limit based on column suffix
            limit = 5  # Conservative default for daily
            for suffix, lim in ffill_limits.items():
                if col.endswith(suffix):
                    limit = lim
                    break
            
            n_nan_before = df[col].isna().sum()
            df[col] = df[col].ffill(limit=limit)
            n_nan_after = df[col].isna().sum()
            
            if n_nan_after > 0 and n_nan_before > n_nan_after:
                first_valid = df[col].first_valid_index()
                if first_valid is not None:
                    interior_nans = df.loc[first_valid:, col].isna().sum()
                    if interior_nans > 0:
                        print(
                            f"WARNING: {col} has {interior_nans} interior NaNs after bounded ffill "
                            f"(limit={limit}). This will propagate to correlations."
                        )
    
    sector_ret = df.set_index("Date")[sector_cols]

    drivers: dict[str, str] = {
        "rates": "^TNX_diff",
        "oil": "CL=F_asinh_diff",
        "usd": "DX-Y.NYB_logret",
        "bonds": "TLT_logret",
        "vix": "^VIX_dlog1p",
        "hy": "BAMLH0A0HYM2_diff",
        "gold": "GC=F_logret",
    }

    long_frames: list[pd.DataFrame] = []

    for driver_name, driver_col in drivers.items():
        if driver_col not in df.columns:
            msg = (
                f"Missing macro driver column '{driver_col}' needed for '{driver_name}' correlations. "
                f"Available macro columns={sorted([c for c in df.columns if c != 'Date'])}"
            )
            if skip_missing_drivers:
                print(f"WARNING: {msg}. Skipping.")
                continue
            raise KeyError(msg)

        driver = df.set_index("Date")[driver_col]

        corr_df = sector_ret.rolling(window=window, min_periods=min_periods).corr(driver)
        corr_df = corr_df.rename(columns={c: c.removesuffix("_logret") for c in corr_df.columns})

        feature = f"corr_{driver_name}_{window}"
        long_frames.append(_to_long_feature(corr_df, feature))

    # IMPORTANT: This is correlation with equal-weighted market average, NOT mean pairwise correlation
    # Renamed to reflect actual computation (was misleadingly named mean_corr_others)
    market_avg = sector_ret.mean(axis=1)
    corr_vs_market = sector_ret.rolling(window=window, min_periods=min_periods).corr(market_avg)
    corr_vs_market = corr_vs_market.rename(columns={c: c.removesuffix("_logret") for c in corr_vs_market.columns})
    
    feature = f"corr_with_sector_mean_{window}"  # FIX: Accurate name
    long_frames.append(_to_long_feature(corr_vs_market, feature))

    return long_frames


def _to_long_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp.index.name = "Date"
    tmp = tmp.reset_index()
    if "Date" not in tmp.columns:
        tmp = tmp.rename(columns={tmp.columns[0]: "Date"})
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

    # CRITICAL: Verify shift semantics BEFORE processing correlations
    print("\n[CRITICAL] Verifying raw2 shift semantics...")
    shift_check = _verify_raw2_shift_semantics(mm_root)
    print(f"Shift verification: {shift_check}")
    
    if shift_check.get("semantics") == "SHIFTED":
        print(
            "\n⚠️  WARNING: raw2 uses SHIFTED semantics (raw2[t] = raw1[t-1])\n"
            "    → Correlations at date t use PREVIOUS trading day returns\n"
            "    → Ensure this aligns with feature timing from data_engineering_1\n"
            "    → If labels use unshifted prices, you have OFF-BY-ONE mismatch!\n"
        )
    elif shift_check.get("semantics") == "INCONSISTENT":
        raise AssertionError(
            f"CRITICAL: Inconsistent shift behavior!\n{shift_check}\n"
            "Check data_optimization_1.py output."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    sector_logrets = _load_spdr_prices(spdr_path)
    macro = _load_macro(raw_data_3_dir, strict=True)

    sector_cols = [c for c in sector_logrets.columns if c != "Date"]
    sectors = [c.removesuffix("_logret") for c in sector_cols]

    # Extract and validate trading calendar
    trading_calendar = _get_trading_calendar(sector_logrets)
    print(f"Trading calendar: {len(trading_calendar)} days from {trading_calendar.min().date()} to {trading_calendar.max().date()}")

    long_frames: list[pd.DataFrame] = []
    for w in SECTOR_MACRO_WINDOWS:
        minp = int(MIN_PERIODS_BY_WINDOW.get(int(w), max(2, int(w) // 2)))
        print(f"Computing sector-macro correlations for window={w}, min_periods={minp}...")
        long_frames.extend(
            _rolling_corr_against_drivers(
                sector_logrets,
                macro,
                window=int(w),
                min_periods=minp,
                skip_missing_drivers=False,
            )
        )

    print(f"Computing macro-macro correlations for windows={MACRO_MACRO_WINDOWS}...")
    long_frames.extend(_macro_macro_corrs(macro, windows=MACRO_MACRO_WINDOWS, sectors=sectors, trading_calendar=trading_calendar))
    
    features_long = pd.concat(long_frames, ignore_index=True)

    features_long = features_long.sort_values(["Date", "Sector", "Feature"], kind="mergesort")

    out_path = out_dir / "features_correlations.parquet"
    features_long["Date"] = pd.to_datetime(features_long["Date"], errors="raise").dt.tz_localize(None)
    features_long = features_long[
        (features_long["Date"] >= TRAIN_DATE_START) & (features_long["Date"] <= TRAIN_DATE_END)
    ].copy()
    features_long.to_parquet(out_path, index=False)

    print(f"\nWrote {len(features_long):,} rows to: {out_path}")
    
    # Summary statistics
    n_features = features_long["Feature"].nunique()
    n_dates = features_long["Date"].nunique()
    n_sectors = features_long["Sector"].nunique()
    print(f"Summary: {n_features} features × {n_sectors} sectors × {n_dates} dates")

if __name__ == "__main__":
    main()