from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd

def compute_linear_decay_weights(
    panel: pd.DataFrame,
    train_start: str,
    train_end: str,
    decay_len: int | None = None,
    min_weight: float = 0.05,
) -> pd.Series:

    dates = pd.to_datetime(panel["Date"]).dt.normalize()

    train_start_dt = pd.Timestamp(train_start).normalize()
    train_end_dt = pd.Timestamp(train_end).normalize()

    in_train = (dates >= train_start_dt) & (dates <= train_end_dt)

    weights = pd.Series(np.nan, index=panel.index)

    if not in_train.any():
        return weights

    train_dates = dates[in_train].unique()
    train_dates = pd.DatetimeIndex(sorted(train_dates))
    n_dates = len(train_dates)

    if n_dates <= 1:
        weights.loc[in_train] = 1.0
        return weights

    date_to_age = {d: n_dates - 1 - i for i, d in enumerate(train_dates)}
    ages = np.asarray(dates[in_train].map(date_to_age).to_numpy(), dtype=np.float64)

    if decay_len is None or decay_len >= n_dates:
        effective_decay_len = float(n_dates - 1)
    else:
        effective_decay_len = float(decay_len)

    if effective_decay_len > 0:
        w = np.maximum(min_weight, 1.0 - ages / effective_decay_len)
    else:
        w = np.ones_like(ages)

    weights.loc[in_train] = w

    return weights

def validate_weights(
    weights: pd.Series,
    panel: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> dict:

    dates = pd.to_datetime(panel["Date"]).dt.normalize()
    train_start_dt = pd.Timestamp(train_start).normalize()
    train_end_dt = pd.Timestamp(train_end).normalize()
    in_train = (dates >= train_start_dt) & (dates <= train_end_dt)

    train_weights = weights.loc[in_train]

    results = {
        "has_nan": bool(train_weights.isna().any()),
        "has_negative": bool((train_weights < 0).any()),
        "has_zero": bool((train_weights == 0).any()),
        "min_weight": float(train_weights.min()) if len(train_weights) else None,
        "max_weight": float(train_weights.max()) if len(train_weights) else None,
        "mean_weight": float(train_weights.mean()) if len(train_weights) else None,
    }

    if len(train_weights) > 0:
        date_weights = train_weights.groupby(dates[in_train]).mean()
        date_weights = date_weights.sort_index()
        diffs = date_weights.diff().dropna()
        results["monotonic_increasing"] = bool((diffs >= -1e-10).all())
    else:
        results["monotonic_increasing"] = True

    results["valid"] = (
        not results["has_nan"]
        and not results["has_negative"]
        and results["monotonic_increasing"]
    )

    return results