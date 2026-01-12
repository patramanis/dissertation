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

RAW2_MANIFEST_PATH = RAW2_DIR / "raw_data_2_manifest.json"


MONTHLY_FILES = {"CPIAUCSL", "UNRATE", "INDPRO"}


# Regime indicators: Keep levels (absolute values have regime signal)
# SOTA: VIX-like indicators - the LEVEL matters (VIX=75 vs VIX=25)
REGIME_INDICATOR_DATASETS = {
    "NFCI",          # Financial Stress: negative = easy conditions, positive = stress
    "BAMLH0A0HYM2",  # Credit Spread: >8% = distress, <4% = calm
    "UNRATE",        # Unemployment: 3-15% bounded, levels matter for regime
    "EPU",           # Economic Policy Uncertainty: bounded oscillator
    "GPR",           # Geopolitical Risk: bounded oscillator
    "ICSA",          # Initial Claims: bounded counts, spikes = recession signal
}


# Feature hygiene toggle: keep original level columns alongside transforms.
# SOTA: Set False for stationarity (De Prado/Quantstart: always diff levels)
KEEP_LEVELS = False


@dataclass(frozen=True)
class Policy:
    kind: Literal["daily", "weekly", "monthly"]
    lag_days: int
    lag_months: int = 0
    ref_floor: Callable[[pd.Series], pd.Series] | None = None
    ffill_limit: int | None = None


def _read_data(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet with date normalization and duplicate detection."""
    df = pd.read_parquet(path, engine="pyarrow") if path.suffix == ".parquet" else pd.read_csv(path)
    
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date in {path}")
    
    # Normalize dates and remove timezone (robust handling)
    dt = pd.to_datetime(df["Date"], errors="raise")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    df["Date"] = dt.dt.normalize()
    
    # Coerce non-Date columns to numeric
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Check for duplicates
    df = df.sort_values("Date")
    dup_mask = df["Date"].duplicated(keep=False)
    if dup_mask.any():
        dup_dates = df.loc[dup_mask, "Date"].value_counts().head(3)
        examples = ", ".join([f"{d.date()}(x{c})" for d, c in dup_dates.items()])
        raise ValueError(f"Duplicate dates in {path.name}: {examples}")
    
    return df


def _load_raw2_shift_rows() -> int:
    """Load shift_rows from raw_data_2 manifest."""
    if not RAW2_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Run data_optimization_1.py first to build {RAW2_MANIFEST_PATH}")
    
    manifest = json.loads(RAW2_MANIFEST_PATH.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest format: {RAW2_MANIFEST_PATH}")
    
    # Try direct field first
    if "shift_rows" in manifest:
        return int(manifest["shift_rows"])
    
    # Fallback: parse from shift_policy string
    policy = str(manifest.get("shift_policy", ""))
    if any(k in policy for k in ["+1 row", "shift(1)", "shifted by +1"]):
        return 1
    
    raise ValueError(f"Cannot determine shift_rows from {RAW2_MANIFEST_PATH}")


def _load_raw2_shift_policy_summary() -> dict[str, object]:
    """Load per-file shift policy from raw_data_2 manifest."""
    if not RAW2_MANIFEST_PATH.exists():
        return {"missing": True}
    
    try:
        manifest = json.loads(RAW2_MANIFEST_PATH.read_text())
        files = manifest.get("files", []) if isinstance(manifest, dict) else []
        
        per_file = {}
        for f in files:
            try:
                basename = f.get("name", "?")  # FIX: was 'basename'
                shift_rows = int(f.get("policy", {}).get("shift_rows", 0))  # Safe default
                per_file[basename] = shift_rows
            except Exception:
                continue
        
        # Count files by shift_rows value
        counts = {}
        for sr in per_file.values():
            counts[str(sr)] = counts.get(str(sr), 0) + 1
        
        return {"per_file_shift_rows": per_file, "counts": counts}
    except Exception as e:
        return {"error": str(e)}


def _load_trading_dates_from_raw2() -> pd.DatetimeIndex:
    """Extract trading calendar from SPDR.parquet."""
    spdr_path = RAW2_DIR / "SPDR.parquet"
    if not spdr_path.exists():
        raise FileNotFoundError(f"Build raw_data_2 first: {spdr_path} missing")
    
    df = _read_data(spdr_path)
    dates = pd.DatetimeIndex(df["Date"]).sort_values().unique()
    
    if len(dates) == 0:
        raise ValueError("Empty trading calendar from SPDR.parquet")
    if not dates.is_monotonic_increasing:
        raise AssertionError("Trading dates not monotonic")
    
    return dates


def _assert_calendar(df: pd.DataFrame, trading_dates: pd.DatetimeIndex, *, name: str) -> None:
    """Verify DataFrame has exact trading calendar (prevents gap-induced errors)."""
    got = pd.DatetimeIndex(df["Date"]).normalize()
    exp = pd.DatetimeIndex(trading_dates).normalize()
    
    if not got.equals(exp):
        raise AssertionError(
            f"{name} calendar mismatch: expected {len(exp)} dates [{exp.min().date()}, {exp.max().date()}], "
            f"got {len(got)} dates [{got.min().date()}, {got.max().date()}]"
        )


def _next_trading_date(trading_dates: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    """Find first trading date >= ts."""
    idx = trading_dates.searchsorted(ts, side="left")
    return pd.Timestamp(trading_dates[idx]) if idx < len(trading_dates) else None


def _policy_for_monthly_basename(basename: str) -> Policy:
    """Publication lag policy for monthly macro data."""
    if basename == "UNRATE":
        return Policy(kind="monthly", lag_months=1, lag_days=10, ffill_limit=30)
    if basename in {"CPIAUCSL", "INDPRO"}:
        return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=30)
    raise ValueError(f"No policy for {basename}")


def _map_observations_to_trading_dates(
    df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    available_from: pd.Series,
) -> pd.DataFrame:
    """Map observations to first trading date >= available_from (vectorized)."""
    available_ts = pd.to_datetime(available_from).dropna()
    if len(available_ts) == 0:
        return pd.DataFrame(index=trading_dates)
    
    # Vectorized: searchsorted finds first trading_date >= available_from
    indices = trading_dates.searchsorted(available_ts.values, side="left")
    valid_mask = indices < len(trading_dates)
    
    if not valid_mask.any():
        return pd.DataFrame(index=trading_dates)
    
    # Filter to valid mappings
    valid_indices = indices[valid_mask]
    valid_rows = available_ts.index[valid_mask]
    mapped_dates = trading_dates[valid_indices]
    
    value_cols = [c for c in df.columns if c != "Date"]
    result = df.loc[valid_rows, value_cols].copy()
    result.index = pd.DatetimeIndex(mapped_dates, name="Date")
    return result.sort_index().groupby(level=0).last()


def _safe_log_returns(level: pd.Series) -> pd.Series:
    """Log returns with NaN for non-positive values."""
    log_level = pd.Series(np.nan, index=level.index, dtype="float64")
    valid = level > 0
    log_level[valid] = np.log(level[valid])
    return log_level.diff(1)

def _asinh(level: pd.Series) -> pd.Series:
    """Inverse hyperbolic sine (negative-safe log-like transform)."""
    return pd.Series(np.arcsinh(level.astype("float64")), index=level.index)

def _log1p(level: pd.Series) -> pd.Series:
    """Log(1+x) for non-negative values."""
    out = pd.Series(np.nan, index=level.index, dtype="float64")
    valid = level >= 0
    out[valid] = np.log1p(level[valid])
    return out


def _build_monthly_growth_features(
    basename: str,
    df_raw2: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build MoM/YoY growth features from already PIT-aligned raw2 data.
    
    CRITICAL: Uses raw2 (already PIT-mapped) to avoid timing split-brain.
    Resamples to month-end to ensure consecutive months for pct_change.
    """
    value_cols = [c for c in df_raw2.columns if c != "Date"]
    if len(value_cols) != 1:
        raise ValueError(f"Expected 1 value column in {basename}, got {value_cols}")
    
    col = value_cols[0]
    df = df_raw2[["Date", col]].copy()
    df = df.set_index("Date").sort_index()
    
    # Resample to month-end (ME) to ensure consecutive months
    # Use 'ME' for month-end (or 'M' in older pandas)
    try:
        df_monthly = df.resample("ME").last()
    except ValueError:
        # Fallback for older pandas versions
        df_monthly = df.resample("M").last()
    
    df_monthly[f"{col}_mom"] = df_monthly[col].pct_change(1)
    df_monthly[f"{col}_yoy"] = df_monthly[col].pct_change(12)
    df_monthly = df_monthly[[f"{col}_mom", f"{col}_yoy"]].dropna(how="all")
    
    # Map back to trading dates (already PIT-aligned, no additional lag needed)
    out = df_monthly.reindex(trading_dates, method="ffill", limit=30)
    out.index.name = "Date"
    
    return out.reset_index()


def _families_for_dataset(basename: str, columns: list[str]) -> dict[str, str]:
    """Assign transform family per column. Raises ValueError for unknown datasets."""
    # Price-like datasets (always positive)
    if basename in {"SPDR", "SPY", "TLT", "VUSTX"}:
        return {c: "always_positive_price_like" for c in columns}
    
    # Futures (mixed)
    if basename == "Futures":
        families = {
            "CL=F": "can_be_nonpositive",
            "^IRX": "rate_or_spread",
            "^TNX": "rate_or_spread",
            "^VIX": "uncertainty_or_counts",
        }
        return {c: families.get(c, "always_positive_price_like") for c in columns}
    
    # Rates and spreads
    if basename == "BAMLH0A0HYM2":
        return {c: "rate_or_spread" for c in columns}
    
    # Uncertainty/counts
    if basename in {"ICSA", "EPU", "GPR"}:
        return {c: "uncertainty_or_counts" for c in columns}
    
    # Can be negative
    if basename == "NFCI":
        return {c: "can_be_negative_level" for c in columns}
    
    # Monthly macro
    if basename in MONTHLY_FILES:
        return {c: "macro_monthly" for c in columns}
    
    raise ValueError(f"Unknown dataset: {basename}. Add to _families_for_dataset()")


def _apply_transforms(
    basename: str,
    df_raw2: pd.DataFrame,
    manifest_rows: list[dict],
) -> pd.DataFrame:
    """Apply stationarity transforms. Level transforms (asinh, log1p) are non-stationary."""
    df = df_raw2.copy()
    value_cols = [c for c in df.columns if c != "Date"]
    family_map = _families_for_dataset(basename, value_cols)
    
    for col in value_cols:
        family = family_map[col]
        s = pd.to_numeric(df[col], errors="coerce")
        transforms = []
        
        if family == "always_positive_price_like":
            df[f"{col}_logret"] = _safe_log_returns(s)
            transforms = ["logret"]
        
        elif family in {"can_be_nonpositive", "can_be_negative_level"}:
            df[f"{col}_asinh"] = _asinh(s)
            df[f"{col}_asinh_diff"] = df[f"{col}_asinh"].diff(1)
            transforms = ["asinh", "asinh_diff"]
        
        elif family == "rate_or_spread":
            df[f"{col}_diff"] = s.diff(1)
            transforms = ["diff"]
        
        elif family == "uncertainty_or_counts":
            # Validate non-negativity for log1p
            if basename in {"ICSA", "EPU", "GPR"} and (s < 0).any():
                raise AssertionError(f"{basename}:{col} has negative values (invalid for log1p)")
            df[f"{col}_log1p"] = _log1p(s)
            df[f"{col}_dlog1p"] = df[f"{col}_log1p"].diff(1)
            transforms = ["log1p", "dlog1p"]
        
        elif family == "macro_monthly":
            transforms = ["mom", "yoy"]
        
        manifest_rows.append({
            "dataset": basename,
            "column": col,
            "family": family,
            "derived": [f"{col}_{t}" for t in transforms],
        })
    
    # Selective level retention: Keep levels ONLY for regime indicators
    # Price-like data: ALWAYS drop levels (non-stationary)
    # Regime indicators: Keep levels (absolute value = regime signal)
    if not KEEP_LEVELS:
        if basename in REGIME_INDICATOR_DATASETS:
            # Keep levels for regime indicators (VIX-like: level matters)
            print(f"INFO: {basename} keeping levels (regime indicator)")
        else:
            # Drop levels for price-like data (stationarity requirement)
            drop_families = {"always_positive_price_like", "can_be_nonpositive", 
                            "rate_or_spread", "can_be_negative_level", "uncertainty_or_counts"}
            drop_cols = [c for c in value_cols if family_map[c] in drop_families]
            if drop_cols:
                df = df.drop(columns=drop_cols, errors="ignore")
                print(f"INFO: {basename} dropped levels {drop_cols} (stationarity)")
    
    return df


def main() -> None:
    if not RAW2_DIR.exists():
        raise FileNotFoundError(f"Build raw_data_2 first: {RAW2_DIR}")
    
    # Load shift policy (per-file shift_rows supported)
    shift_summary = _load_raw2_shift_policy_summary()
    
    # Load ffill_limits from raw2 manifest
    ffill_limits = _load_ffill_limits_from_raw2()
    
    # Verify monthly files have shift_rows=0 (no double-shift)
    per_file = shift_summary.get("per_file_shift_rows", {})
    for basename in MONTHLY_FILES:
        if per_file.get(basename, 0) != 0:
            raise AssertionError(f"{basename} has non-zero shift in raw2 (monthly logic assumes 0)")
    
    trading_dates = _load_trading_dates_from_raw2()
    RAW3_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = []
    raw2_paths = sorted(RAW2_DIR.glob("*.parquet"))
    
    if not raw2_paths:
        raise FileNotFoundError(f"No parquet files in {RAW2_DIR}")
    
    for path in raw2_paths:
        basename = path.stem
        df_raw2 = _read_data(path)
        
        # Enforce calendar alignment (prevents gap-induced errors)
        _assert_calendar(df_raw2, trading_dates, name=basename)
        
        # REMOVED: pre-transform unbounded ffill (was causing staleness)
        # Transforms must handle NaN propagation gracefully
        
        df_out = _apply_transforms(basename, df_raw2, manifest_rows)
        
        # Add monthly growth features if applicable
        if basename in MONTHLY_FILES:
            # FIXED: Compute from raw2 (already PIT-aligned) not raw1
            growth = _build_monthly_growth_features(basename, df_raw2, trading_dates)
            df_out = df_out.merge(growth, on="Date", how="left")
            
            # Validate columns exist
            col = [c for c in df_raw2.columns if c != "Date"][0]
            for suffix in ["_mom", "_yoy"]:
                if f"{col}{suffix}" not in df_out.columns:
                    raise AssertionError(f"{basename}: missing {col}{suffix}")
        
        # FIXED: Use bounded ffill from raw2 manifest, not unbounded
        ffill_limit = ffill_limits.get(basename, 5)  # Conservative default
        if ffill_limit is None:
            print(f"WARNING: {basename} has unbounded ffill_limit from raw2")
            ffill_limit = 30  # Cap at 30 to prevent staleness
        
        # Handle NaNs with bounded forward-fill (no backfill - would leak future)
        nan_cols = df_out.columns[df_out.isna().any()].tolist()
        if nan_cols:
            nan_summary = ", ".join([f"{c}({df_out[c].isna().sum()})" for c in nan_cols])
            print(f"INFO: {basename} NaNs: {nan_summary} (ffill limit={ffill_limit})")
            
            df_out = df_out.ffill(limit=ffill_limit)
            # NO bfill - would leak future data
            
            # Check for interior NaNs after bounded ffill
            for col in nan_cols:
                if col not in df_out.columns:
                    continue
                first_valid = df_out[col].first_valid_index()
                if first_valid is None:
                    raise AssertionError(f"{basename}:{col} no valid data")
                interior_nans = df_out.loc[first_valid:, col].isna().sum()
                if interior_nans > 0:
                    # Allow some interior NaNs (ffill_limit exceeded) but warn
                    interior_pct = interior_nans / len(df_out.loc[first_valid:])
                    if interior_pct > 0.05:
                        raise AssertionError(
                            f"{basename}:{col} has {interior_pct:.1%} interior NaNs after bounded ffill "
                            f"(ffill_limit={ffill_limit} exceeded or data quality issue)"
                        )
                    else:
                        print(f"WARNING: {basename}:{col} has {interior_nans} interior NaNs ({interior_pct:.2%})")
        
        out_path = RAW3_DIR / f"{basename}.parquet"
        df_out.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Wrote {out_path} shape={df_out.shape}")
    
    # Write manifest
    manifest = {
        "raw_data_3": {
            "inherits_timing_from": "raw_data_2 (PIT-aligned with per-series shift)",
            "shift_summary": shift_summary,
            "stationarity": {
                "stationary": ["_logret", "_asinh_diff", "_diff", "_dlog1p"],
                "quasi_stationary": ["_mom", "_yoy"],  # Can drift in regime changes
                "level": ["_asinh", "_log1p"],
            },
            "transforms": {
                "_logret": "log return = diff(log(level))",
                "_asinh": "arcsinh(level) - level, non-stationary",
                "_asinh_diff": "diff(arcsinh(level)) - stationary, variance-stabilized",
                "_diff": "first difference",
                "_log1p": "log(1+x) - level, non-stationary",
                "_dlog1p": "diff(log1p(level)) - stationary",
                "_mom": "monthly pct_change(1) from PIT-aligned raw2",
                "_yoy": "monthly pct_change(12) from PIT-aligned raw2",
            },
            "nan_policy": {
                "ffill": "Bounded by raw2 manifest limits (prevents staleness)",
                "bfill": "NEVER (would leak future)",
                "leading_nans": "Allowed before first observation",
                "interior_nans": "Allowed <5% when ffill_limit exceeded",
            },
            "feature_hygiene": {
                "KEEP_LEVELS": KEEP_LEVELS,
                "regime_indicators": list(REGIME_INDICATOR_DATASETS),
                "note": "Levels kept ONLY for regime indicators (VIX-like). Prices always diff (SOTA: De Prado)",
            },
        },
        "columns": sorted(manifest_rows, key=lambda r: (r["dataset"], r["column"])),
    }
    
    (RAW3_DIR / "Sanity_check").mkdir(exist_ok=True)
    manifest_path = RAW3_DIR / "Sanity_check" / "raw_data_3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
