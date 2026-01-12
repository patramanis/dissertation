from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

BUILD_ID = str(Path(__file__).resolve())

MM_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = MM_ROOT / "data" / "raw_data_1"
OUTPUT_DIR = MM_ROOT / "data" / "raw_data_2"

LABEL_INPUTS_DIR = MM_ROOT / "data" / "labels" / "unshifted_lagged_raw_data"

LABEL_DATE_START = pd.Timestamp("2000-09-29")
LABEL_DATE_END = pd.Timestamp("2025-09-30")

MANIFEST_PATH = OUTPUT_DIR / "raw_data_2_manifest.json"

SECTORS: tuple[str, ...] = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLY",
    "XLV",
    "XLU",
)


DAILY_FILES = {
    "SPDR",
    "SPY",
    "Futures",
    "TLT",
    "VUSTX",
    "EPU",
    "BAMLH0A0HYM2",
}

WEEKLY_FILES = {"ICSA", "NFCI"}

MONTHLY_FILES = {"CPIAUCSL", "UNRATE", "INDPRO"}


@dataclass(frozen=True)
class Policy:
    kind: Literal["daily", "weekly", "monthly"]
    lag_days: int
    lag_months: int = 0
    ref_floor: Callable[[pd.Series], pd.Series] | None = None
    ffill_limit: int | None = None
    trading_day_lag: int = 0  # For daily data: how many trading days to wait


def _normalize_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="raise")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt.dt.normalize()


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = _normalize_date_series(df["Date"])

    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("Date")
    if df["Date"].duplicated().any():
        dup_dates = df["Date"][df["Date"].duplicated(keep=False)].value_counts().head(5)
        examples = ", ".join([f"{d.date()}(x{c})" for d, c in dup_dates.items()])
        raise ValueError(f"Duplicate dates in {path.name}: {examples}")
    return df


def _load_trading_dates(spdr_path: Path) -> pd.DatetimeIndex:
    df = _read_csv(spdr_path)
    dates = pd.DatetimeIndex(df["Date"]).sort_values().unique()
    if len(dates) == 0:
        raise ValueError(f"Empty trading calendar from {spdr_path.name}")
    return dates


def _next_index_date(index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    """Find next available date in index."""
    idx = index.searchsorted(ts)
    return index[idx] if idx < len(index) else None





def _floor_to_weekday(dates: pd.Series, target_weekday: int) -> pd.Series:
    wd = dates.dt.weekday
    days_since = (wd - target_weekday) % 7
    return (dates - pd.to_timedelta(days_since, unit="D")).dt.normalize()


def _policy_for_basename(basename: str, df: pd.DataFrame) -> Policy:
    # Daily: trading_day_lag=1 so Date=T sees Close(T-1)
    # SOTA: ffill_limit=5 to prevent stale carries (De Prado)
    if basename in DAILY_FILES:
        return Policy(kind="daily", lag_days=0, ffill_limit=5, trading_day_lag=1)
    
    # Weekly: Use sets for consistency
    if basename in WEEKLY_FILES:
        if basename == "ICSA":
            return Policy(kind="weekly", lag_days=7, ref_floor=lambda s: _floor_to_weekday(s, 5), ffill_limit=10)
        if basename == "NFCI":
            return Policy(kind="weekly", lag_days=7, ref_floor=lambda s: _floor_to_weekday(s, 4), ffill_limit=10)
    
    # Monthly: Use set for consistency
    if basename in MONTHLY_FILES:
        if basename == "UNRATE":
            return Policy(kind="monthly", lag_months=1, lag_days=10, ffill_limit=30)
        if basename in {"CPIAUCSL", "INDPRO"}:
            return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=30)
    
    # Auto-detect GPR frequency
    if basename == "GPR":
        dates = pd.DatetimeIndex(df["Date"]).unique()
        if len(dates) < 3:
            return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=30)
        median_gap = float(pd.Series(dates[1:] - dates[:-1]).dt.days.median())
        if median_gap <= 3:
            return Policy(kind="daily", lag_days=0, ffill_limit=5, trading_day_lag=1)
        if median_gap <= 10:
            return Policy(kind="weekly", lag_days=7, ref_floor=lambda s: _floor_to_weekday(s, 4), ffill_limit=10)
        return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=30)
    
    raise ValueError(f"No policy for {basename}")


def _compute_available_from_calendar(df: pd.DataFrame, policy: Policy, trading_dates: pd.DatetimeIndex | None = None) -> pd.Series:
    dates = df["Date"]
    
    if policy.kind == "daily":
        if policy.trading_day_lag > 0 and trading_dates is not None:
            # Map to N trading days in the future
            result = []
            for d in dates:
                idx = trading_dates.searchsorted(pd.Timestamp(d), side="left")
                if idx + policy.trading_day_lag < len(trading_dates):
                    result.append(trading_dates[idx + policy.trading_day_lag])
                else:
                    result.append(pd.NaT)
            return pd.Series(result, index=dates.index)
        else:
            return dates + pd.Timedelta(days=policy.lag_days)
    
    if policy.kind == "weekly":
        if not policy.ref_floor:
            raise ValueError("Weekly policy requires ref_floor")
        return policy.ref_floor(dates) + pd.Timedelta(days=policy.lag_days)
    
    if policy.kind == "monthly":
        return dates + pd.DateOffset(months=policy.lag_months) + pd.Timedelta(days=policy.lag_days)
    
    raise ValueError(f"Unknown policy kind: {policy.kind}")


def _map_to_index(
    df: pd.DataFrame,
    index: pd.DatetimeIndex,
    available_from: pd.Series,
) -> pd.DataFrame:
    # Vectorized mapping: find first index date >= available_from for each row
    available_ts = pd.to_datetime(available_from).dropna()
    if len(available_ts) == 0:
        return pd.DataFrame(index=index)
    
    # searchsorted vectorized operation
    indices = index.searchsorted(available_ts.values, side="left")
    valid_mask = indices < len(index)
    
    if not valid_mask.any():
        return pd.DataFrame(index=index)
    
    # Filter to valid mappings
    valid_indices = indices[valid_mask]
    valid_rows = available_ts.index[valid_mask]
    mapped_dates = index[valid_indices]
    
    value_cols = [c for c in df.columns if c != "Date"]
    result = df.loc[valid_rows, value_cols].copy()
    result.index = pd.DatetimeIndex(mapped_dates, name="Date")
    return result.sort_index().groupby(level=0).last()


def _first_mapped_dates_by_col(mapped: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Track first non-null date per column for leakage detection."""
    return {col: mapped[col].dropna().index.min() for col in mapped.columns if mapped[col].notna().any()}


def _assert_no_values_before_first_actionable(
    basename: str,
    out: pd.DataFrame,
    first_mapped_by_col: dict[str, pd.Timestamp],
) -> None:
    """Verify no data leakage: values only appear after first actionable date."""
    # With trading_day_lag baked into available_from, first_mapped already IS the first actionable date
    for col, first_actionable in first_mapped_by_col.items():
        if col not in out.columns or pd.isna(first_actionable):
            continue
        
        pre = out.loc[out.index < first_actionable, col]
        if pre.notna().any():
            bad = pre[pre.notna()].index[0]
            raise AssertionError(
                f"{basename}.{col}: leakage at {bad.date()} before actionable {first_actionable.date()}"
            )


def _assert_min_coverage(out: pd.DataFrame, basename: str, min_rows: int = 252) -> None:
    """Verify sufficient non-null data after PIT mapping."""
    if out.empty:
        raise AssertionError(f"{basename}: empty output")
    
    for col in out.columns:
        n = out[col].notna().sum()
        if n == 0:
            raise AssertionError(f"{basename}.{col}: all NaN")
        if n < min_rows:
            raise AssertionError(f"{basename}.{col}: only {n} rows (need {min_rows})")
        
        # SOTA: Validate NaN % < 5% for data quality (SSRN/JPM standard)
        # Calculate only from first valid date to avoid false-fail for late-starting series
        first_valid_idx = out[col].first_valid_index()
        if first_valid_idx is None:
            continue
        
        valid_range = out.loc[first_valid_idx:, col]
        if len(valid_range) > 0:
            nan_pct = valid_range.isna().sum() / len(valid_range)
            if nan_pct > 0.05:
                raise AssertionError(
                    f"{basename}.{col}: {nan_pct:.1%} NaNs exceeds 5% threshold "
                    f"from {first_valid_idx.date()} onwards (data quality issue or insufficient ffill_limit)"
                )


def _mapped_ffilled_frame(
    csv_path: Path,
    trading_dates: pd.DatetimeIndex,
    value_cols: list[str],
    force_sparse: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp], Policy]:
    """Map data to trading_dates index with production lag, then ffill.
    
    trading_day_lag is now baked into available_from, no separate shift needed.
    force_sparse=True for label inputs (but now all data is on trading calendar).
    """
    df = _read_csv(csv_path)
    basename = csv_path.stem
    policy = _policy_for_basename(basename, df)
    available_from = _compute_available_from_calendar(df, policy, trading_dates)

    # All data directly mapped to trading_dates (no dense calendar needed)
    mapped = _map_to_index(df, trading_dates, available_from)
    first_mapped_by_col = _first_mapped_dates_by_col(mapped)

    out = mapped.reindex(trading_dates).ffill(limit=policy.ffill_limit)
    out = out.reindex(columns=value_cols)

    # Price series strictness
    if basename in {"SPDR", "SPY"}:
        if out.isna().any().any():
            raise AssertionError(f"{basename}: missing values after mapping")

    return out, first_mapped_by_col, policy


def _filter_label_date_range(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name != "Date":
        raise ValueError("Expected Date index for label input filter")
    idx = pd.DatetimeIndex(df.index)
    mask = (idx >= LABEL_DATE_START) & (idx <= LABEL_DATE_END)
    return df.loc[mask].copy()


def _export_unshifted_label_inputs(trading_dates: pd.DatetimeIndex) -> None:
    """Export unshifted price data for label computation (single file for all horizons)."""
    spdr, _, _ = _mapped_ffilled_frame(
        INPUT_DIR / "SPDR.csv", trading_dates, list(SECTORS), force_sparse=True
    )
    spy, _, _ = _mapped_ffilled_frame(
        INPUT_DIR / "SPY.csv", trading_dates, ["SPY"], force_sparse=True
    )

    prices = spdr.merge(spy, left_index=True, right_index=True, how="left")
    prices = prices[["SPY", *SECTORS]]
    prices.index.name = "Date"

    # Filter to label date range
    mask = (prices.index >= LABEL_DATE_START) & (prices.index <= LABEL_DATE_END)
    prices = prices[mask]

    # Verify calendar alignment
    expected = trading_dates[(trading_dates >= LABEL_DATE_START) & (trading_dates <= LABEL_DATE_END)]
    if not prices.index.equals(pd.DatetimeIndex(expected, name="Date")):
        raise AssertionError(
            f"Label calendar mismatch: expected {len(expected)} dates, got {len(prices)}"
        )

    LABEL_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Export once: horizons are computed by label builder, not here
    out = prices.reset_index()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out_path = LABEL_INPUTS_DIR / "prices_unshifted.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[labels_export] {out_path.name}: shape={out.shape}")


def _build_pit_dataset(csv_path: Path, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build PIT dataset: trading_day_lag baked into mapping, no separate shift needed."""
    df = _read_csv(csv_path)
    basename = csv_path.stem
    policy = _policy_for_basename(basename, df)

    # Determine value columns
    value_cols = [c for c in df.columns if c != "Date"]
    if basename == "SPDR":
        value_cols = list(SECTORS)
    elif basename == "SPY":
        value_cols = ["SPY"]

    # Get data directly on trading_dates (lag already in available_from)
    out, first_mapped, _ = _mapped_ffilled_frame(
        csv_path, trading_dates, value_cols
    )

    # Critical: verify no leakage
    _assert_no_values_before_first_actionable(basename, out, first_mapped)

    # Verify sufficient coverage
    _assert_min_coverage(out, basename)

    # Format output
    result = out.reset_index()
    result.columns = ["Date"] + list(out.columns)
    result["Date"] = _normalize_date_series(result["Date"])

    return result


def main() -> None:
    print(f"[data_optimization_1] BUILD_ID={BUILD_ID}")
    
    spdr_path = INPUT_DIR / "SPDR.csv"
    if not spdr_path.exists():
        raise FileNotFoundError(f"Missing {spdr_path}")

    trading_dates = _load_trading_dates(spdr_path)
    _export_unshifted_label_inputs(trading_dates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files in {INPUT_DIR}")

    manifest = {
        "artifact_dir": str(OUTPUT_DIR),
        "trading_calendar": str(spdr_path),
        "date_range": [str(trading_dates.min().date()), str(trading_dates.max().date())],
        "lag_policy": "daily=trading_day_lag(1), weekly/monthly=calendar_lag+days",
        "files": [],
    }

    for csv_path in csv_paths:
        df_out = _build_pit_dataset(csv_path, trading_dates)
        out_path = OUTPUT_DIR / f"{csv_path.stem}.parquet"
        df_out.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Wrote {out_path.name}: shape={df_out.shape}")

        policy = _policy_for_basename(csv_path.stem, _read_csv(csv_path))
        manifest["files"].append({
            "name": csv_path.stem,
            "policy": {k: v for k, v in policy.__dict__.items() if k != "ref_floor"},
            "columns": [c for c in df_out.columns if c != "Date"],
        })

    # Verify label calendar alignment
    spdr_df = pd.read_parquet(OUTPUT_DIR / "SPDR.parquet")
    lab_df = pd.read_parquet(LABEL_INPUTS_DIR / "prices_unshifted.parquet")
    
    spdr_dates = pd.to_datetime(spdr_df["Date"]).dt.normalize()
    lab_dates = pd.to_datetime(lab_df["Date"]).dt.normalize()
    
    mask = (spdr_dates >= LABEL_DATE_START) & (spdr_dates <= LABEL_DATE_END)
    if not spdr_dates[mask].reset_index(drop=True).equals(lab_dates.reset_index(drop=True)):
        raise AssertionError("Calendar mismatch: SPDR vs labels")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST_PATH.name}")


if __name__ == "__main__":
    main()