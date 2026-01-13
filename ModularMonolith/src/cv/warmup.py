from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WarmupReport:
    warmup_start_date: pd.Timestamp
    nan_ratio_at_start: float
    max_lookback_days_used: int
    max_lookback_days_inferred: int
    dates_cut_before_warmup: int


_INT_RE = re.compile(r"(\d+)")


def infer_max_lookback_days(feature_cols: list[str], *, conservative_default: int = 252) -> int:
    """Infer a conservative maximum lookback from feature names."""
    candidates: list[int] = []
    for c in feature_cols:
        for m in _INT_RE.findall(str(c)):
            try:
                v = int(m)
            except ValueError:
                continue
            if 2 <= v <= 2000:
                candidates.append(v)
    return int(max(candidates)) if candidates else int(conservative_default)


def compute_warmup_start_date(
    *,
    X: pd.DataFrame,
    keys: pd.DataFrame,
    feature_cols: list[str],
    nan_ratio_threshold: float = 0.01,
    stability_days: int = 60,
    purge_gap: int = 21,
    max_lookback_days: int | None = None,
    conservative_default_lookback: int = 252,
) -> WarmupReport:
    """Find first trading date where NaNs are stable and enough history exists.

    keys must include: Date, Sector (Sector used only for sanity; not required).

    nan_ratio(date) = (#NaNs in X[feature_cols] rows on that date) / (n_rows_date * n_features)

    warmup_start_date: first date where:
      - nan_ratio <= threshold
      - holds for `stability_days` consecutive trading dates
      - has at least `max_lookback_days_used` trading dates of history before it
      - has at least `purge_gap` trading dates of history before it
    """
    if "Date" not in keys.columns:
        raise ValueError("keys must include Date")

    d = pd.to_datetime(keys["Date"], errors="raise").dt.normalize()

    feat_cols = list(feature_cols)
    if not feat_cols:
        raise ValueError("feature_cols is empty")

    inferred = infer_max_lookback_days(feat_cols, conservative_default=conservative_default_lookback)
    used = int(max_lookback_days) if max_lookback_days is not None else int(max(inferred, conservative_default_lookback))

    X_num = X[feat_cols].apply(pd.to_numeric, errors="coerce")

    uniq_dates = pd.Index(d.unique()).sort_values()
    if uniq_dates.empty:
        raise ValueError("No dates in keys")

    # Map each row to date code
    codes = uniq_dates.get_indexer(d)
    if (codes < 0).any():
        raise ValueError("Date mapping failed")

    n_dates = int(len(uniq_dates))
    n_features = int(X_num.shape[1])

    nan_ratio_by_date = np.full(n_dates, np.nan, dtype=np.float64)
    X_arr = X_num.to_numpy(dtype=np.float32, copy=False)

    for i in range(n_dates):
        mask = codes == i
        n_rows = int(mask.sum())
        if n_rows <= 0:
            continue
        block = X_arr[mask]
        nan_ratio_by_date[i] = float(np.isnan(block).sum()) / float(n_rows * n_features)

    ok = np.isfinite(nan_ratio_by_date) & (nan_ratio_by_date <= float(nan_ratio_threshold))

    run = 0
    stab = int(stability_days)
    for i in range(n_dates):
        run = (run + 1) if ok[i] else 0
        if run < stab:
            continue
        start_idx = i - stab + 1

        if start_idx < int(used):
            continue
        if start_idx < int(purge_gap):
            continue

        warmup_start = pd.Timestamp(uniq_dates[start_idx]).normalize()
        return WarmupReport(
            warmup_start_date=warmup_start,
            nan_ratio_at_start=float(nan_ratio_by_date[start_idx]),
            max_lookback_days_used=int(used),
            max_lookback_days_inferred=int(inferred),
            dates_cut_before_warmup=int(start_idx),
        )

    raise RuntimeError(
        "Unable to find warmup_start_date: no date satisfies nan_ratio threshold with required stability + history"
    )
