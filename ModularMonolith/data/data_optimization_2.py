from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd


MM_ROOT = Path(__file__).resolve().parents[1]
RAW1_DIR = MM_ROOT / "data" / "raw_data_1"
RAW2_DIR = MM_ROOT / "data" / "raw_data_2"
RAW3_DIR = MM_ROOT / "data" / "raw_data_3"


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


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    return df


def _load_trading_dates_from_raw2() -> pd.DatetimeIndex:
    spdr_path = RAW2_DIR / "SPDR.parquet"
    if not spdr_path.exists():
        raise FileNotFoundError(f"Missing {spdr_path}; build raw_data_2 first")
    df = _read_parquet(spdr_path)
    dates = pd.DatetimeIndex(df["Date"].dt.normalize()).sort_values().unique()
    if len(dates) == 0:
        raise ValueError("SPDR.parquet produced empty trading dates")
    if not dates.is_monotonic_increasing:
        raise AssertionError("Trading dates must be monotonic increasing")
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


def _policy_for_monthly_basename(basename: str) -> Policy:
    if basename == "UNRATE":
        return Policy(kind="monthly", lag_months=1, lag_days=10)
    if basename in {"CPIAUCSL", "INDPRO"}:
        return Policy(kind="monthly", lag_months=1, lag_days=20)
    raise ValueError(f"No monthly policy configured for {basename}")


def _compute_available_from_calendar_monthly(df: pd.DataFrame, policy: Policy) -> pd.Series:
    src = df["Date"].dt.normalize()
    anchor = src.dt.to_period("M").dt.to_timestamp(how="start").dt.normalize()
    out = anchor + pd.DateOffset(months=int(policy.lag_months)) + pd.to_timedelta(policy.lag_days, unit="D")
    return pd.Series(out, index=df.index)


def _map_observations_to_trading_dates(
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


def _safe_log_returns(level: pd.Series) -> pd.Series:
    log_level = pd.Series(np.nan, index=level.index, dtype="float64")
    valid = level > 0
    log_level.loc[valid] = np.log(level.loc[valid].astype("float64"))
    return log_level.diff(1)


def _asinh(level: pd.Series) -> pd.Series:
    return pd.Series(np.arcsinh(level.astype("float64")), index=level.index)


def _log1p(level: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=level.index, dtype="float64")
    valid = level >= 0
    out.loc[valid] = np.log1p(level.loc[valid].astype("float64"))
    return out


def _build_monthly_growth_features(
    basename: str,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    csv_path = RAW1_DIR / f"{basename}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df_raw = _read_csv(csv_path)
    value_cols = [c for c in df_raw.columns if c != "Date"]
    if len(value_cols) != 1:
        raise ValueError(f"Expected single value column in {csv_path}, got {value_cols}")

    col = value_cols[0]
    df_native = df_raw[["Date", col]].copy()
    df_native[col] = pd.to_numeric(df_native[col], errors="coerce")

    df_native = df_native.dropna(subset=[col])

    df_native[f"{col}_mom"] = df_native[col].pct_change(1)
    df_native[f"{col}_yoy"] = df_native[col].pct_change(12)

    policy = _policy_for_monthly_basename(basename)
    available_from_calendar = _compute_available_from_calendar_monthly(df_native, policy)

    mapped = _map_observations_to_trading_dates(
        df_native[["Date", f"{col}_mom", f"{col}_yoy"]],
        trading_dates,
        available_from_calendar,
    )

    out = mapped.reindex(trading_dates).ffill().shift(1)
    out_df = out.reset_index().rename(columns={"index": "Date"})
    if "Date" not in out_df.columns:
        out_df = out_df.rename(columns={out_df.columns[0]: "Date"})
    out_df["Date"] = pd.to_datetime(out_df["Date"]).astype("datetime64[ns]")
    return out_df


def _families_for_dataset(basename: str, columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}

    if basename == "SPDR":
        for c in columns:
            out[c] = "always_positive_price_like"
        return out

    if basename == "SPY":
        for c in columns:
            out[c] = "always_positive_price_like"
        return out

    if basename in {"TLT", "VUSTX"}:
        for c in columns:
            out[c] = "always_positive_price_like"
        return out

    if basename == "Futures":
        for c in columns:
            if c == "CL=F":
                out[c] = "can_be_nonpositive"
            elif c in {"^IRX", "^TNX"}:
                out[c] = "rate_or_spread"
            elif c in {"^VIX"}:
                out[c] = "uncertainty_or_counts"
            else:
                out[c] = "always_positive_price_like"
        return out

    if basename == "BAMLH0A0HYM2":
        for c in columns:
            out[c] = "rate_or_spread"
        return out

    if basename in {"ICSA", "EPU", "GPR"}:
        for c in columns:
            out[c] = "uncertainty_or_counts"
        return out

    if basename == "NFCI":
        for c in columns:
            out[c] = "can_be_negative_level"
        return out

    if basename in MONTHLY_FILES:
        for c in columns:
            out[c] = "macro_monthly"
        return out

    for c in columns:
        out[c] = "unknown"
    return out


def _apply_transforms(
    basename: str,
    df_raw2: pd.DataFrame,
    manifest_rows: list[dict],
) -> pd.DataFrame:
    df = df_raw2.copy()
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date in raw2 {basename}")

    value_cols = [c for c in df.columns if c != "Date"]
    family_map = _families_for_dataset(basename, value_cols)

    for col in value_cols:
        family = family_map.get(col, "unknown")
        transforms: list[str] = []

        s = pd.to_numeric(df[col], errors="coerce")

        if family == "always_positive_price_like":
            new_name = f"{col}_logret"
            df[new_name] = _safe_log_returns(s)
            transforms.append("logret")

        elif family == "can_be_nonpositive":
            df[f"{col}_asinh"] = _asinh(s)
            df[f"{col}_diff"] = s.diff(1)
            transforms.extend(["asinh", "diff"])

        elif family == "rate_or_spread":
            df[f"{col}_diff"] = s.diff(1)
            transforms.append("diff")

        elif family == "can_be_negative_level":
            df[f"{col}_asinh"] = _asinh(s)
            df[f"{col}_diff"] = s.diff(1)
            transforms.extend(["asinh", "diff"])

        elif family == "uncertainty_or_counts":
            df[f"{col}_log1p"] = _log1p(s)
            df[f"{col}_dlog1p"] = df[f"{col}_log1p"].diff(1)
            transforms.extend(["log1p", "dlog1p"])

        elif family == "macro_monthly":
            transforms = ["mom", "yoy"]

        manifest_rows.append(
            {
                "dataset": basename,
                "column": col,
                "family": family,
                "derived": [f"{col}_{t}" for t in transforms],
            }
        )

    return df


def main() -> None:
    if not RAW2_DIR.exists():
        raise FileNotFoundError(f"Missing {RAW2_DIR}")

    trading_dates = _load_trading_dates_from_raw2()

    RAW3_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    raw2_paths = sorted(RAW2_DIR.glob("*.parquet"))
    if not raw2_paths:
        raise FileNotFoundError(f"No parquet files found in {RAW2_DIR}")

    for p in raw2_paths:
        basename = p.stem
        df_raw2 = _read_parquet(p)

        df_out = _apply_transforms(basename, df_raw2, manifest_rows)

        if basename in MONTHLY_FILES:
            growth = _build_monthly_growth_features(basename, trading_dates)
            df_out = df_out.merge(growth, on="Date", how="left")

        out_path = RAW3_DIR / f"{basename}.parquet"
        df_out.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Wrote {out_path} shape={df_out.shape}")

    manifest = {
        "raw_data_3": {
            "inherits_timing_from": "raw_data_2 (publication-lag safe + shift(1))",
            "no_additional_shift_applied": True,
            "suffix_semantics": {
                "_logret": "log return (diff of log level)",
                "_asinh": "arcsinh(level) for nonpositive-safe compression",
                "_diff": "first difference (t - t-1)",
                "_log1p": "log1p(level) for nonnegative levels",
                "_dlog1p": "diff(log1p(level))",
                "_mom": "monthly pct_change(1) computed on native monthly series, then PIT-projected",
                "_yoy": "monthly pct_change(12) computed on native monthly series, then PIT-projected",
            },
        },
        "columns": sorted(manifest_rows, key=lambda r: (r["dataset"], r["column"])),
    }

    sanity_dir = RAW3_DIR / "Sanity_check"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sanity_dir / "raw_data_3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
