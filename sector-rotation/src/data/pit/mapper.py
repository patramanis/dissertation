from __future__ import annotations
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
from .policies import Policy, floor_to_weekday

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class ReadCsvReport:
    path: Path
    coerced_to_nan: dict[str, int]

def normalize_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="raise")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt.dt.normalize()

def read_raw_csv(path: Path) -> tuple[pd.DataFrame, ReadCsvReport]:
    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")

    df["Date"] = normalize_date_series(df["Date"])

    coerced: dict[str, int] = {}
    for col in df.columns:
        if col == "Date":
            continue
        original = df[col]
        numeric = pd.to_numeric(original, errors="coerce")
        bad = int(original.notna().sum() - numeric.notna().sum())
        if bad:
            coerced[col] = bad
            log.warning("%s: coerced %d values to NaN in column %s", path.name, bad, col)
        df[col] = numeric.astype("float64")

    df = df.sort_values("Date", kind="mergesort")

    if df["Date"].duplicated().any():
        dup = df.loc[df["Date"].duplicated(keep=False), "Date"]
        vc = dup.value_counts().head(5)
        examples = ", ".join([f"{pd.Timestamp(str(d)).date()}(x{c})" for d, c in vc.items()])
        raise ValueError(f"Duplicate dates in {path.name}: {examples}")

    df = df.set_index("Date")
    df.index = pd.DatetimeIndex(df.index, name="Date")

    return df, ReadCsvReport(path=path, coerced_to_nan=coerced)

def load_trading_calendar(spy_csv_path: Path) -> pd.DatetimeIndex:
    df, _ = read_raw_csv(spy_csv_path)
    idx = pd.DatetimeIndex(df.index).sort_values().unique()
    if len(idx) == 0:
        raise ValueError(f"Empty trading calendar from {spy_csv_path}")
    return idx

def compute_available_from(
    observed_dates: pd.DatetimeIndex,
    policy: Policy,
    trading_calendar: pd.DatetimeIndex,
) -> pd.Series:
    dates = pd.Series(pd.DatetimeIndex(observed_dates), index=range(len(observed_dates)))

    if policy.kind == "daily":
        if policy.trading_day_lag > 0:
            base_pos = trading_calendar.searchsorted(dates.values, side="left")
            pos = base_pos + policy.trading_day_lag
            ok = pos < len(trading_calendar)
            out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
            if ok.any():
                out.loc[ok] = trading_calendar.take(pos[ok])
            return out

        return dates + pd.Timedelta(days=policy.lag_days)

    if policy.kind == "weekly":
        if policy.weekday_floor is None:
            raise ValueError("Weekly policy requires weekday_floor")
        floored = floor_to_weekday(dates.astype("datetime64[ns]"), policy.weekday_floor)
        return floored + pd.Timedelta(days=policy.lag_days)

    if policy.kind == "monthly":
        return dates + pd.DateOffset(months=policy.lag_months) + pd.Timedelta(days=policy.lag_days)

    raise ValueError(f"Unknown policy kind: {policy.kind}")

def map_to_calendar(
    raw: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    available_from: pd.Series,
) -> pd.DataFrame:
    available_ts = pd.to_datetime(available_from, errors="coerce")

    valid_mask = available_ts.notna().to_numpy(dtype=bool)
    if not valid_mask.any():
        return pd.DataFrame(index=trading_calendar)

    valid_ts = available_ts.to_numpy(dtype="datetime64[ns]")[valid_mask]
    valid_rows = raw.index.to_numpy()[valid_mask]

    positions = trading_calendar.searchsorted(valid_ts, side="left")
    ok = positions < len(trading_calendar)
    if not ok.any():
        return pd.DataFrame(index=trading_calendar)

    positions = positions[ok]
    valid_rows = valid_rows[ok]

    mapped_dates = trading_calendar.take(positions)
    mapped = raw.loc[pd.DatetimeIndex(valid_rows), :].copy()
    mapped.index = pd.DatetimeIndex(mapped_dates, name="Date")
    mapped = mapped.sort_index(kind="mergesort")
    mapped = mapped.groupby(level=0, sort=True).last()

    return mapped

def apply_ffill_limit(
    mapped: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    ffill_limit: int | None,
) -> pd.DataFrame:
    out = mapped.reindex(trading_calendar)
    out = out.ffill(limit=ffill_limit)

    for col in out.columns:
        out[col] = out[col].astype("float64")

    out.index = pd.DatetimeIndex(out.index, name="Date")
    return out

def first_actionable_dates(mapped: pd.DataFrame) -> dict[str, pd.Timestamp]:
    return {
        col: mapped[col].dropna().index.min()
        for col in mapped.columns
        if mapped[col].notna().any()
    }

def assert_no_values_before_first_actionable(
    series_name: str,
    ffilled: pd.DataFrame,
    first_actionable_by_col: dict[str, pd.Timestamp],
) -> None:
    for col, first_actionable in first_actionable_by_col.items():
        pre = ffilled.loc[ffilled.index < first_actionable, col]
        if pre.notna().any():
            bad = pre[pre.notna()].index[0]
            raise AssertionError(
                f"{series_name}.{col}: leakage at {bad.date()} before actionable {first_actionable.date()}"
            )

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stable_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        df.to_csv(
            f,
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.10g",
            lineterminator="\n",
        )

def nan_rates(df: pd.DataFrame) -> dict[str, float]:
    rates: dict[str, float] = {}
    for col in df.columns:
        if col == "Date":
            continue
        denom = len(df)
        rates[col] = float(df[col].isna().sum() / denom) if denom else 0.0
    return rates

def schema_snapshot(df: pd.DataFrame) -> dict[str, str]:
    return {c: str(df[c].dtype) for c in df.columns}

def ensure_columns_present(df: pd.DataFrame, required: Iterable[str], *, series_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{series_name}: missing required columns: {missing}")