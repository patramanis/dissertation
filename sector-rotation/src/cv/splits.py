from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Literal
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class CVFold:

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    purge_gap: int

    valid_start: str | None = None
    valid_end: str | None = None

    train_n_dates: int | None = None
    test_n_dates: int | None = None
    valid_n_dates: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

def _trading_days_index(panel: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.to_datetime(panel["Date"]).dt.normalize()
    return pd.DatetimeIndex(sorted(dates.unique()))

def _date_to_str(d: pd.Timestamp) -> str:
    return pd.Timestamp(d).date().isoformat()

class UnifiedCVSplit:

    def __init__(
        self,
        horizon: int = 21,
        purge_buffer: int = 5,
        min_train_years: int = 5,
        test_months: int = 6,
        step_months: int = 6,
        valid_fraction: float = 0.15,
    ):
        self.horizon = horizon
        self.purge_buffer = purge_buffer
        self.gap = horizon + purge_buffer
        self.min_train_years = min_train_years
        self.test_months = test_months
        self.step_months = step_months
        self.valid_fraction = valid_fraction

    def generate_test_windows(self, panel: pd.DataFrame) -> list[tuple[int, int]]:
        dates = _trading_days_index(panel)
        n_dates = len(dates)

        trading_days_per_month = 21
        trading_days_per_year = 252

        min_train_days = self.min_train_years * trading_days_per_year
        test_days = self.test_months * trading_days_per_month
        step_days = self.step_months * trading_days_per_month

        first_test_start = min_train_days + self.gap

        windows = []
        test_start_idx = first_test_start

        while test_start_idx < n_dates:
            test_end_idx = min(test_start_idx + test_days - 1, n_dates - 1)

            if test_end_idx - test_start_idx + 1 >= test_days // 2:
                windows.append((test_start_idx, test_end_idx))

            test_start_idx += step_days

        return windows

    def split(
        self,
        panel: pd.DataFrame,
        mode: Literal["rolling"] = "rolling",
        rolling_years: int = 5,
    ) -> Iterator[CVFold]:
        dates = _trading_days_index(panel)
        n_dates = len(dates)

        trading_days_per_year = 252
        rolling_days = rolling_years * trading_days_per_year

        test_windows = self.generate_test_windows(panel)

        for fold_id, (test_start_idx, test_end_idx) in enumerate(test_windows):
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]

            train_cutoff_idx = test_start_idx - self.gap - 1
            if train_cutoff_idx < 0:
                continue

            train_start_idx = max(0, train_cutoff_idx - rolling_days + 1)

            train_end_idx = train_cutoff_idx

            if train_end_idx <= train_start_idx:
                continue

            train_size = train_end_idx - train_start_idx + 1
            valid_days = int(train_size * self.valid_fraction)
            valid_days = max(1, valid_days)

            valid_end_idx = train_end_idx
            valid_start_idx = valid_end_idx - valid_days + 1

            train_main_end_idx = valid_start_idx - self.gap - 1

            if train_main_end_idx <= train_start_idx:
                train_main_end_idx = train_end_idx
                valid_start_idx = None
                valid_end_idx = None
                valid_days = 0

            train_start = dates[train_start_idx]
            train_end = dates[train_main_end_idx]

            valid_start = dates[valid_start_idx] if valid_start_idx is not None else None
            valid_end = dates[valid_end_idx] if valid_end_idx is not None else None

            fold = CVFold(
                fold_id=fold_id,
                train_start=_date_to_str(train_start),
                train_end=_date_to_str(train_end),
                test_start=_date_to_str(test_start),
                test_end=_date_to_str(test_end),
                purge_gap=self.gap,
                valid_start=_date_to_str(valid_start) if valid_start is not None else None,
                valid_end=_date_to_str(valid_end) if valid_end is not None else None,
                train_n_dates=int(train_main_end_idx - train_start_idx + 1),
                test_n_dates=int(test_end_idx - test_start_idx + 1),
                valid_n_dates=int(valid_days) if valid_days > 0 else None,
            )

            yield fold

    def get_n_splits(self, panel: pd.DataFrame) -> int:
        return len(self.generate_test_windows(panel))

def generate_cv_folds(
    panel: pd.DataFrame,
    cv_type: Literal["rolling"] = "rolling",
    horizon: int = 21,
    purge_buffer: int = 5,
    min_train_years: int = 5,
    test_months: int = 6,
    step_months: int = 6,
    valid_fraction: float = 0.15,
    rolling_years: int = 5,
    n_folds: int | None = None,
) -> list[CVFold]:

    splitter = UnifiedCVSplit(
        horizon=horizon,
        purge_buffer=purge_buffer,
        min_train_years=min_train_years,
        test_months=test_months,
        step_months=step_months,
        valid_fraction=valid_fraction,
    )

    folds = list(splitter.split(panel, mode=cv_type, rolling_years=rolling_years))

    if n_folds is not None and len(folds) > n_folds:
        folds = folds[:n_folds]

    return folds

def save_cv_folds(
    folds: list[CVFold],
    output_path: Path,
    cv_type: str,
    horizon: int,
    metadata: dict | None = None,
) -> None:

    output = {
        "cv_type": cv_type,
        "horizon": horizon,
        "n_folds": len(folds),
        "folds": [f.to_dict() for f in folds],
    }

    if metadata:
        output["metadata"] = metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

def load_cv_folds(input_path: Path) -> list[CVFold]:

    data = json.loads(input_path.read_text(encoding="utf-8"))

    folds = []
    for f in data["folds"]:
        if "embargo" in f and "purge_gap" not in f:
            f["purge_gap"] = f.pop("embargo") + f.get("purge_gap", 0)
        folds.append(CVFold(**{k: v for k, v in f.items() if k in CVFold.__dataclass_fields__}))

    return folds