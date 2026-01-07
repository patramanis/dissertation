from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal

import pandas as pd


MM_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = MM_ROOT / "data" / "raw_data_1"
OUTPUT_DIR = MM_ROOT / "data" / "raw_data_2"

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


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    return df


def _load_trading_dates(spdr_path: Path) -> pd.DatetimeIndex:
    df = _read_csv(spdr_path)
    dates = pd.DatetimeIndex(df["Date"].dt.normalize()).sort_values().unique()
    if len(dates) == 0:
        raise ValueError("SPDR.csv produced empty TRADING_DATES")
    if not dates.is_monotonic_increasing:
        raise AssertionError("TRADING_DATES must be monotonic increasing")
    return dates


def _next_trading_date(trading_dates: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    i = int(trading_dates.searchsorted(ts, side="left"))
    if i >= len(trading_dates):
        return None
    return pd.Timestamp(trading_dates[i])


def _floor_to_weekday(dates: pd.Series, target_weekday: int) -> pd.Series:
    wd = dates.dt.weekday
    days_since = (wd - target_weekday) % 7
    return (dates - pd.to_timedelta(days_since, unit="D")).dt.normalize()


def _policy_for_basename(basename: str, df: pd.DataFrame) -> Policy:
    if basename in DAILY_FILES:
        return Policy(kind="daily", lag_days=0, ffill_limit=5)

    # Weekly releases
    if basename == "ICSA":
        return Policy(kind="weekly", lag_days=7, ref_floor=lambda s: _floor_to_weekday(s, 5), ffill_limit=None)
    if basename == "NFCI":
        return Policy(kind="weekly", lag_days=7, ref_floor=lambda s: _floor_to_weekday(s, 4), ffill_limit=None)

    # Monthly macro
    if basename == "UNRATE":
        return Policy(kind="monthly", lag_months=1, lag_days=10, ffill_limit=None)
    if basename in {"CPIAUCSL", "INDPRO"}:
        return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=None)

    if basename == "GPR":
        dates = pd.DatetimeIndex(df["Date"].dt.normalize()).sort_values().unique()
        if len(dates) < 3:
            return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=None)
        diffs = pd.Series(dates[1:] - dates[:-1]).dt.days
        median_diff = float(diffs.median())
        if median_diff <= 3.0:
            return Policy(kind="daily", lag_days=0, ffill_limit=5)
        if 4.0 <= median_diff <= 10.0:
            return Policy(kind="weekly", lag_days=7, ref_floor=lambda s: _floor_to_weekday(s, 4), ffill_limit=None)
        return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=None)

    raise ValueError(
        f"No policy configured for {basename}. "
    )


def _compute_available_from_calendar(df: pd.DataFrame, policy: Policy) -> pd.Series:
    src = df["Date"].dt.normalize()

    if policy.kind == "daily":
        return src + pd.to_timedelta(policy.lag_days, unit="D")

    if policy.kind == "weekly":
        if policy.ref_floor is None:
            raise ValueError("weekly policy requires ref_floor")
        ref = policy.ref_floor(src)
        return ref + pd.to_timedelta(policy.lag_days, unit="D")

    if policy.kind == "monthly":
        anchor = src.dt.to_period("M").dt.to_timestamp(how="start").dt.normalize()
        out = anchor + pd.DateOffset(months=int(policy.lag_months)) + pd.to_timedelta(policy.lag_days, unit="D")
        return pd.Series(out, index=df.index)

    raise ValueError(f"Unknown policy kind: {policy.kind}")


def _map_to_trading_dates(
    df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    available_from_calendar: pd.Series,
) -> pd.DataFrame:
    mapped_dates: list[pd.Timestamp] = []
    keep_rows: list[int] = []

    for i, ts in enumerate(available_from_calendar):
        mapped = _next_trading_date(trading_dates, pd.Timestamp(ts))
        if mapped is None:
            continue  
        keep_rows.append(i)
        mapped_dates.append(mapped)

    if not keep_rows:
        return pd.DataFrame(index=trading_dates)

    value_cols = [c for c in df.columns if c != "Date"]
    mapped = df.iloc[keep_rows][value_cols].copy()
    mapped.index = pd.DatetimeIndex(mapped_dates, name="Date")

    mapped = mapped.sort_index()
    mapped = mapped.groupby(level=0, sort=True).last()

    return mapped


def _build_pit_dataset(csv_path: Path, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    df_raw = _read_csv(csv_path)
    basename = csv_path.stem

    value_cols = [c for c in df_raw.columns if c != "Date"]

    if basename == "SPDR":
        if all(s in value_cols for s in SECTORS):
            value_cols = list(SECTORS)
        else:
            raise ValueError(
                "SPDR.csv does not contain the expected sector adjusted-close columns. "
                f"Expected={list(SECTORS)}; got={value_cols[:30]}"
            )

    if basename == "SPY":
        if "SPY" in value_cols:
            value_cols = ["SPY"]
        else:
            raise ValueError(f"SPY.csv missing expected column 'SPY'; got={value_cols[:30]}")

    policy = _policy_for_basename(basename, df_raw)
    available_from_calendar = _compute_available_from_calendar(df_raw, policy)

    mapped = _map_to_trading_dates(df_raw, trading_dates, available_from_calendar)

    if not mapped.empty:
        first_mapped = mapped.index.min()
        pre = mapped.reindex(trading_dates)
        if pre.loc[trading_dates < first_mapped].notna().any().any():
            raise AssertionError(f"Leakage: values present before first availability for {basename}")

    out = mapped.reindex(trading_dates)
    if policy.kind == "daily":
        out = out.ffill(limit=policy.ffill_limit)
    else:
        out = out.ffill()

    out = out.reindex(columns=value_cols)
    out = out.shift(1)

    if not out.index.equals(trading_dates):
        raise AssertionError(f"Output index is not exactly trading dates for {basename}")

    out_df = out.reset_index().rename(columns={"index": "Date"})
    if "Date" not in out_df.columns:
        out_df = out_df.rename(columns={out_df.columns[0]: "Date"})
    out_df["Date"] = pd.to_datetime(out_df["Date"]).astype("datetime64[ns]")

    expected_cols = ["Date", *value_cols]
    if list(out_df.columns) != expected_cols:
        raise AssertionError(
            f"Column mismatch for {basename}: expected={expected_cols}, got={list(out_df.columns)}"
        )

    return out_df


def main() -> None:
    spdr_path = INPUT_DIR / "SPDR.csv"
    if not spdr_path.exists():
        raise FileNotFoundError(f"Missing trading calendar source: {spdr_path}")

    trading_dates = _load_trading_dates(spdr_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    manifest: dict[str, object] = {
        "artifact_dir": str(OUTPUT_DIR.as_posix()),
        "trading_calendar_source": str(spdr_path.as_posix()),
        "trading_dates_start": str(pd.Timestamp(trading_dates.min()).date()),
        "trading_dates_end": str(pd.Timestamp(trading_dates.max()).date()),
        "shift_policy": "All series are shifted by +1 row so Date=T contains info available by Open(T), i.e., Close(T-1) for daily series.",
        "files": [],
    }

    for csv_path in csv_paths:
        out_path = OUTPUT_DIR / f"{csv_path.stem}.parquet"
        df_out = _build_pit_dataset(csv_path, trading_dates)
        df_out.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Wrote {out_path} shape={df_out.shape}")

        df_raw = _read_csv(csv_path)
        pol = _policy_for_basename(csv_path.stem, df_raw)
        file_meta = {
            "basename": csv_path.stem,
            "input": str(csv_path.as_posix()),
            "output": str(out_path.as_posix()),
            "policy": {
                "kind": pol.kind,
                "lag_days": int(pol.lag_days),
                "lag_months": int(pol.lag_months),
                "ffill_limit": None if pol.ffill_limit is None else int(pol.ffill_limit),
            },
            "columns": [c for c in df_out.columns if c != "Date"],
            "start": str(pd.to_datetime(df_out["Date"].min()).date()) if len(df_out) else None,
            "end": str(pd.to_datetime(df_out["Date"].max()).date()) if len(df_out) else None,
        }
        manifest["files"].append(file_meta)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
