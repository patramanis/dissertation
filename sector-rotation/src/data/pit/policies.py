from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import pandas as pd

PolicyKind = Literal["daily", "weekly", "monthly"]

@dataclass(frozen=True)
class Policy:
    kind: PolicyKind
    lag_days: int
    lag_months: int = 0
    weekday_floor: int | None = None
    ffill_limit: int | None = None
    trading_day_lag: int = 0

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

DAILY_FILES: set[str] = {
    "SPDR",
    "SPY",
    "Futures",
    "TLT",
    "VUSTX",
    "EPU",
    "BAMLH0A0HYM2",
}

WEEKLY_FILES: set[str] = {"ICSA", "NFCI"}

MONTHLY_FILES: set[str] = {"CPIAUCSL", "UNRATE", "INDPRO"}

def floor_to_weekday(dates: pd.Series, target_weekday: int) -> pd.Series:
    idx = pd.DatetimeIndex(dates)
    days_since = (idx.dayofweek - target_weekday) % 7
    floored = (idx - pd.to_timedelta(days_since, unit="D")).normalize()
    return pd.Series(floored, index=dates.index)

def _parse_policy_dict(d: dict[str, Any]) -> Policy:
    kind = d.get("kind")
    if kind not in {"daily", "weekly", "monthly"}:
        raise ValueError(f"Invalid policy kind: {kind!r}")

    lag_days = int(d.get("lag_days", 0))
    lag_months = int(d.get("lag_months", 0))

    weekday_floor = d.get("weekday_floor", None)
    if weekday_floor is not None:
        weekday_floor = int(weekday_floor)

    ffill_limit = d.get("ffill_limit", None)
    if ffill_limit is not None:
        ffill_limit = int(ffill_limit)

    trading_day_lag = int(d.get("trading_day_lag", 0))

    return Policy(
        kind=kind,
        lag_days=lag_days,
        lag_months=lag_months,
        weekday_floor=weekday_floor,
        ffill_limit=ffill_limit,
        trading_day_lag=trading_day_lag,
    )

def policies_from_config(config: dict[str, Any] | None) -> dict[str, Policy]:
    if not config:
        return {}

    mapping: Any = None
    if isinstance(config.get("pit"), dict) and isinstance(config["pit"].get("policies"), dict):
        mapping = config["pit"]["policies"]
    elif isinstance(config.get("pit_policies"), dict):
        mapping = config["pit_policies"]

    if not isinstance(mapping, dict):
        return {}

    out: dict[str, Policy] = {}
    for name, raw in mapping.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ValueError("Policies config must be a dict[str, dict]")
        out[name] = _parse_policy_dict(raw)
    return out

def default_policy_for_series(series_name: str, dates: pd.DatetimeIndex | None = None) -> Policy:
    if series_name in DAILY_FILES:
        return Policy(kind="daily", lag_days=0, ffill_limit=5, trading_day_lag=1)

    if series_name in WEEKLY_FILES:
        if series_name == "ICSA":
            return Policy(kind="weekly", lag_days=7, weekday_floor=5, ffill_limit=10)
        if series_name == "NFCI":
            return Policy(kind="weekly", lag_days=7, weekday_floor=4, ffill_limit=10)

    if series_name in MONTHLY_FILES:
        if series_name == "UNRATE":
            return Policy(kind="monthly", lag_months=1, lag_days=10, ffill_limit=30)
        if series_name in {"CPIAUCSL", "INDPRO"}:
            return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=30)

    if series_name == "GPR":
        if dates is None or len(dates) < 3:
            return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=30)

        uniq = pd.DatetimeIndex(dates).sort_values().unique()
        if len(uniq) < 3:
            return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=30)

        gaps_days = (uniq[1:] - uniq[:-1]) / pd.Timedelta(days=1)
        median_gap = float(pd.Series(gaps_days).median())

        if median_gap <= 3:
            return Policy(kind="daily", lag_days=0, ffill_limit=5, trading_day_lag=1)
        if median_gap <= 10:
            return Policy(kind="weekly", lag_days=7, weekday_floor=4, ffill_limit=10)
        return Policy(kind="monthly", lag_months=1, lag_days=20, ffill_limit=30)

    raise ValueError(f"No policy for series {series_name!r} (missing config?)")

def policy_for_series(series_name: str, *, config: dict[str, Any] | None = None, dates: pd.DatetimeIndex | None = None) -> Policy:
    cfg = policies_from_config(config)
    if series_name in cfg:
        return cfg[series_name]
    return default_policy_for_series(series_name, dates=dates)