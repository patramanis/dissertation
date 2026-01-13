from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise ImportError("pandas is required") from e

try:
    from sklearn.model_selection import BaseCrossValidator
except Exception as e:  # pragma: no cover
    raise ImportError("scikit-learn error.") from e


@dataclass(frozen=True)
class RollingFoldInfo:
    split_index: int
    train_start: object
    train_end: object
    test_start: object
    test_end: object
    purge_gap: int


class RollingWindowCV(BaseCrossValidator):
    """Rolling window walk-forward CV with purge gap.

    This splitter operates on *group dates* (unique trading dates) derived from `groups`.

    Windows are calendar-year based but snapped to available trading dates:
    - Train window: [test_start - train_years, test_start)
    - Purge: drop the last `purge_gap` trading dates before test_start
      (equivalently, train_end = test_start_idx - purge_gap)
    - Test window: [test_start, test_start + test_years)
    - Step: test_start += step_years

    Warmup:
    - The first test_start is the first trading date >= warmup_start_date.
    - Folds are skipped until there is a full `train_years` of history available
      before test_start (and after purge).

    Notes:
    - `purge_gap` is in TRADING DATES (index-based), not calendar days.
    - Embargo is intentionally not supported here.
    """

    def __init__(
        self,
        *,
        train_years: int = 5,
        test_years: int = 1,
        step_years: int = 1,
        purge_gap: int = 21,
        warmup_start_date: object | None = None,
        min_train_years: int = 3,
    ) -> None:
        if train_years <= 0:
            raise ValueError("train_years must be positive")
        if test_years <= 0:
            raise ValueError("test_years must be positive")
        if step_years <= 0:
            raise ValueError("step_years must be positive")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")
        if min_train_years <= 0:
            raise ValueError("min_train_years must be positive")

        self.train_years = int(train_years)
        self.test_years = int(test_years)
        self.step_years = int(step_years)
        self.purge_gap = int(purge_gap)
        self.warmup_start_date = warmup_start_date
        self.min_train_years = int(min_train_years)

        # cached after first split
        self._n_splits_cached: int | None = None

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if self._n_splits_cached is not None:
            return int(self._n_splits_cached)
        if groups is None:
            raise ValueError("groups is required")
        # Compute once by iterating splits
        g = self._as_1d_datetime(groups)
        uniq = self._unique_sorted(g)
        n = 0
        for _ in self._iter_fold_boundaries(uniq):
            n += 1
        self._n_splits_cached = int(n)
        return int(n)

    @staticmethod
    def _as_1d_datetime(groups) -> np.ndarray:
        g = np.asarray(groups)
        if g.ndim != 1:
            g = np.ravel(g)
        try:
            g = pd.to_datetime(g, errors="raise").to_numpy(dtype="datetime64[ns]")
        except Exception as e:
            raise ValueError(f"groups must be datetime-like; got dtype={g.dtype}") from e
        return g

    @staticmethod
    def _unique_sorted(groups_1d: np.ndarray) -> np.ndarray:
        uniq = np.unique(groups_1d)
        return np.sort(uniq)

    def _first_test_start_date(self, unique_groups: np.ndarray) -> np.datetime64:
        if unique_groups.size == 0:
            raise ValueError("Empty groups")
        if self.warmup_start_date is None:
            return unique_groups[0]
        try:
            ts = np.datetime64(pd.Timestamp(self.warmup_start_date).to_datetime64())
        except Exception as e:
            raise ValueError(f"Invalid warmup_start_date={self.warmup_start_date!r}") from e
        i = int(np.searchsorted(unique_groups, ts, side="left"))
        if i < 0:
            i = 0
        if i >= int(unique_groups.size):
            return unique_groups[-1]
        return unique_groups[i]

    def _iter_fold_boundaries(self, unique_groups: np.ndarray) -> Iterator[tuple[int, int, int, int, RollingFoldInfo]]:
        # unique_groups is datetime64[ns] sorted.
        dates = pd.to_datetime(unique_groups)
        first_test_start = pd.Timestamp(self._first_test_start_date(unique_groups)).normalize()

        # Ensure monotone stepping over available trading dates.
        idx = pd.Index(dates)
        test_start = first_test_start
        split_index = 0

        earliest = pd.Timestamp(dates.min()).normalize()

        while True:
            test_end_cal = (test_start + pd.DateOffset(years=self.test_years)).normalize()
            next_test_start_cal = (test_start + pd.DateOffset(years=self.step_years)).normalize()

            # Snap test_start to trading date index (it should already be).
            test_start_idx = int(idx.searchsorted(test_start, side="left"))
            if test_start_idx >= len(idx):
                break
            test_start = pd.Timestamp(idx[test_start_idx]).normalize()

            # Test end idx
            test_end_idx = int(idx.searchsorted(test_end_cal, side="left"))
            test_end_idx = min(test_end_idx, len(idx))
            if test_end_idx <= test_start_idx:
                break

            # Require full calendar train_years history available
            train_start_cal = (test_start - pd.DateOffset(years=self.train_years)).normalize()
            if earliest > train_start_cal:
                # Not enough history yet; step forward.
                ns = int(idx.searchsorted(next_test_start_cal, side="left"))
                if ns >= len(idx):
                    break
                test_start = pd.Timestamp(idx[ns]).normalize()
                continue

            # Purge: drop last purge_gap trading dates before test_start
            train_end_idx = max(0, test_start_idx - int(self.purge_gap))
            if train_end_idx <= 0:
                ns = int(idx.searchsorted(next_test_start_cal, side="left"))
                if ns >= len(idx):
                    break
                test_start = pd.Timestamp(idx[ns]).normalize()
                continue

            # Train start index based on calendar boundary but clipped to train_end
            train_start_idx = int(idx.searchsorted(train_start_cal, side="left"))
            train_start_idx = min(train_start_idx, train_end_idx)

            # Min train size sanity
            min_train_days = int(self.min_train_years) * 252
            if (train_end_idx - train_start_idx) < min_train_days:
                ns = int(idx.searchsorted(next_test_start_cal, side="left"))
                if ns >= len(idx):
                    break
                test_start = pd.Timestamp(idx[ns]).normalize()
                continue

            info = RollingFoldInfo(
                split_index=int(split_index),
                train_start=idx[train_start_idx],
                train_end=idx[train_end_idx - 1] if train_end_idx > train_start_idx else None,
                test_start=idx[test_start_idx],
                test_end=idx[test_end_idx - 1],
                purge_gap=int(self.purge_gap),
            )

            yield train_start_idx, train_end_idx, test_start_idx, test_end_idx, info

            split_index += 1

            # Step to next test_start
            ns = int(idx.searchsorted(next_test_start_cal, side="left"))
            if ns >= len(idx):
                break
            next_ts = pd.Timestamp(idx[ns]).normalize()
            if next_ts <= test_start:
                raise RuntimeError("Non-monotone rolling CV stepping detected")
            test_start = next_ts

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups is required")
        g = self._as_1d_datetime(groups)
        uniq = self._unique_sorted(g)
        n_groups = int(uniq.shape[0])
        if n_groups == 0:
            raise ValueError("Empty dataset")

        group_codes = np.searchsorted(uniq, g)
        if not np.all((group_codes >= 0) & (group_codes < n_groups)):
            raise ValueError("groups contain values not in unique_groups")

        # Iterate boundaries, convert to sample indices
        count = 0
        for train_start_idx, train_end_idx, test_start_idx, test_end_idx, info in self._iter_fold_boundaries(uniq):

            train_mask = (group_codes >= int(train_start_idx)) & (group_codes < int(train_end_idx))
            test_mask = (group_codes >= int(test_start_idx)) & (group_codes < int(test_end_idx))

            tr_idx = np.flatnonzero(train_mask)
            te_idx = np.flatnonzero(test_mask)

            if tr_idx.size == 0 or te_idx.size == 0:
                continue

            # Safety: no overlap
            if np.intersect1d(tr_idx, te_idx).size > 0:
                raise RuntimeError("Train/Test overlap detected in rolling CV")

            count += 1
            yield tr_idx, te_idx

        self._n_splits_cached = int(count)
