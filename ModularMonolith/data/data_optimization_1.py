from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from ModularMonolith.data.train_window import TRAIN_DATE_END, TRAIN_DATE_START

try:
    from ModularMonolith import build_id

    BUILD_ID = build_id(__file__)
except Exception:
    BUILD_ID = str(Path(__file__).resolve())


MM_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = MM_ROOT / "data" / "raw_data_1"
OUTPUT_DIR = MM_ROOT / "data" / "raw_data_2"

LABEL_INPUTS_DIR = MM_ROOT / "data" / "labels" / "unshifted_lagged_raw_data"

LABEL_DATE_START = TRAIN_DATE_START
LABEL_DATE_END = TRAIN_DATE_END

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
    if basename == "ICSA":
        return Policy(kind="weekly", lag_days=7, ref_floor=lambda s: _floor_to_weekday(s, 5), ffill_limit=10)
    if basename == "NFCI":
        return Policy(kind="weekly", lag_days=7, ref_floor=lambda s: _floor_to_weekday(s, 4), ffill_limit=10)
    if basename == "UNRATE":
        # Monthly data: add 45-day ffill limit to prevent stale values propagating indefinitely
        # 45 trading days ≈ 2 months, so if a release is missing we still have a reasonable limit
        return Policy(kind="monthly", lag_months=1, lag_days=10, ffill_limit=45)
    if basename in {"CPIAUCSL", "INDPRO"}:
        # Monthly data: add 45-day ffill limit to prevent stale values propagating indefinitely
        return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=45)

    if basename == "GPR":
        dates = pd.DatetimeIndex(df["Date"].dt.normalize()).sort_values().unique()
        if len(dates) < 3:
            return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=45)
        diffs = pd.Series(dates[1:] - dates[:-1]).dt.days
        median_diff = float(diffs.median())
        if median_diff <= 3.0:
            return Policy(kind="daily", lag_days=0, ffill_limit=5)
        if 4.0 <= median_diff <= 10.0:
            return Policy(kind="weekly", lag_days=7, ref_floor=lambda s: _floor_to_weekday(s, 4), ffill_limit=10)
        return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=45)

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


def _mapped_ffilled_frame(csv_path: Path, trading_dates: pd.DatetimeIndex, *, value_cols: list[str]) -> pd.DataFrame:
    df_raw = _read_csv(csv_path)
    basename = csv_path.stem
    policy = _policy_for_basename(basename, df_raw)
    available_from_calendar = _compute_available_from_calendar(df_raw, policy)

    mapped = _map_to_trading_dates(df_raw, trading_dates, available_from_calendar)

    if not mapped.empty:
        first_mapped = mapped.index.min()
        pre = mapped.reindex(trading_dates)
        if pre.loc[trading_dates < first_mapped].notna().any().any():
            raise AssertionError(f"Leakage: values present before first availability for {basename}")

    out = mapped.reindex(trading_dates)
    # Apply ffill with limit for all data types to prevent stale values
    # For daily: limit=5 trading days
    # For weekly: limit=10 trading days (~2 weeks)  
    # For monthly: limit=45 trading days (~2 months)
    if policy.ffill_limit is not None:
        out = out.ffill(limit=policy.ffill_limit)
    else:
        out = out.ffill()

    out = out.reindex(columns=value_cols)

    if not out.index.equals(trading_dates):
        raise AssertionError(f"Output index is not exactly trading dates for {basename}")

    return out


def _filter_label_date_range(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name != "Date":
        raise ValueError("Expected Date index for label input filter")
    idx = pd.DatetimeIndex(df.index)
    mask = (idx >= LABEL_DATE_START) & (idx <= LABEL_DATE_END)
    return df.loc[mask].copy()


def _export_unshifted_label_inputs(trading_dates: pd.DatetimeIndex) -> None:
    spdr_path = INPUT_DIR / "SPDR.csv"
    spy_path = INPUT_DIR / "SPY.csv"
    for p in (spdr_path, spy_path):
        if not p.exists():
            raise FileNotFoundError(p)

    spdr_pre = _mapped_ffilled_frame(spdr_path, trading_dates, value_cols=list(SECTORS))
    spy_pre = _mapped_ffilled_frame(spy_path, trading_dates, value_cols=["SPY"])

    prices = spdr_pre.merge(spy_pre, left_index=True, right_index=True, how="left")
    prices = prices[["SPY", *SECTORS]].copy()

    if prices.index.has_duplicates:
        raise AssertionError("Unshifted label input prices has duplicate Date index")

    prices = _filter_label_date_range(prices)

    expected_dates = trading_dates[(trading_dates >= LABEL_DATE_START) & (trading_dates <= LABEL_DATE_END)]
    if not prices.index.equals(pd.DatetimeIndex(expected_dates, name="Date")):
        got = pd.DatetimeIndex(prices.index)
        raise AssertionError(
            "Unshifted label inputs calendar mismatch vs trading_dates (after canonical cut). "
            f"expected_len={len(expected_dates)} got_len={len(got)} "
            f"expected_range=[{pd.Timestamp(expected_dates.min()).date()},{pd.Timestamp(expected_dates.max()).date()}] "
            f"got_range=[{pd.Timestamp(got.min()).date()},{pd.Timestamp(got.max()).date()}]"
        )

    LABEL_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for h in (5, 21, 63):
        out_path = LABEL_INPUTS_DIR / f"prices_h{int(h)}.parquet"
        out_df = prices.reset_index().rename(columns={"index": "Date"})
        if "Date" not in out_df.columns:
            out_df = out_df.rename(columns={out_df.columns[0]: "Date"})
        out_df["Date"] = pd.to_datetime(out_df["Date"], errors="raise").dt.normalize().dt.tz_localize(None)

        out_df = out_df[["Date", "SPY", *SECTORS]].copy()
        out_df.to_parquet(out_path, index=False, engine="pyarrow")

        start = pd.to_datetime(out_df["Date"].min()).date() if len(out_df) else None
        end = pd.to_datetime(out_df["Date"].max()).date() if len(out_df) else None
        print(
            f"[labels_export] Wrote {out_path} shape={out_df.shape} date_range=[{start},{end}] "
            f"cols={len(out_df.columns)}"
        )


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

    out = _mapped_ffilled_frame(csv_path, trading_dates, value_cols=value_cols)
    out = out.shift(1)

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
    print(f"[data_optimization_1] BUILD_ID={BUILD_ID}")
    spdr_path = INPUT_DIR / "SPDR.csv"
    if not spdr_path.exists():
        raise FileNotFoundError(f"Missing trading calendar source: {spdr_path}")

    trading_dates = _load_trading_dates(spdr_path)

    _export_unshifted_label_inputs(trading_dates)

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

    spdr_out = OUTPUT_DIR / "SPDR.parquet"
    label_prices = LABEL_INPUTS_DIR / "prices_h5.parquet"
    if spdr_out.exists() and label_prices.exists():
        spdr_df = pd.read_parquet(spdr_out, engine="pyarrow")
        lab_df = pd.read_parquet(label_prices, engine="pyarrow")
        spdr_dates = pd.to_datetime(spdr_df["Date"], errors="raise").dt.normalize().dt.tz_localize(None)
        lab_dates = pd.to_datetime(lab_df["Date"], errors="raise").dt.normalize().dt.tz_localize(None)
        spdr_dates = spdr_dates[(spdr_dates >= LABEL_DATE_START) & (spdr_dates <= LABEL_DATE_END)].reset_index(drop=True)
        lab_dates = lab_dates.reset_index(drop=True)
        if len(spdr_dates) != len(lab_dates) or not spdr_dates.equals(lab_dates):
            raise AssertionError(
                "Calendar mismatch: raw_data_2/SPDR.parquet Date column != labels/unshifted_lagged_raw_data/prices_h5.parquet Date column. "
                f"spdr_len={len(spdr_dates)} lab_len={len(lab_dates)}"
            )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
