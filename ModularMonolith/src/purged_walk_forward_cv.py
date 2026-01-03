from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


try:
    from sklearn.model_selection import BaseCrossValidator
except Exception as e:
    raise ImportError("scikit-learn error.") from e


@dataclass(frozen=True)
class FoldInfo:
    split_index: int
    train_start: object
    train_end: object
    test_start: object
    test_end: object
    purge_gap: int
    embargo: int


class PurgedWalkForwardCV(BaseCrossValidator):

    def __init__(
        self,
        *,
        n_splits: int = 5,
        test_size: int = 63,
        purge_gap: int = 21,
        embargo: int = 0,
    ) -> None:
        if n_splits <= 0:
            raise ValueError("n_splits must be positive")
        if test_size <= 0:
            raise ValueError("test_size must be positive")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")

        self.n_splits = int(n_splits)
        self.test_size = int(test_size)
        self.purge_gap = int(purge_gap)
        self.embargo = int(embargo)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    @staticmethod
    def _as_1d_array(groups) -> np.ndarray:
        if groups is None:
            raise ValueError("groups is required and must contain the Date for each sample")
        g = np.asarray(groups)
        if g.ndim != 1:
            g = np.ravel(g)
        return g

    @staticmethod
    def _unique_sorted(groups_1d: np.ndarray) -> np.ndarray:
        uniq = np.unique(groups_1d)
        return np.sort(uniq)

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:  # noqa: N803
        groups_1d = self._as_1d_array(groups)
        n_samples = groups_1d.shape[0]

        unique_groups = self._unique_sorted(groups_1d)
        n_groups = unique_groups.shape[0]

        required = self.n_splits * self.test_size
        if n_groups < required:
            raise ValueError(
                f"Not enough unique groups for n_splits*test_size: have {n_groups}, need {required}"
            )

        first_test_start = n_groups - required
        embargoed_groups: set[object] = set()

        for i in range(self.n_splits):
            test_start_idx = first_test_start + i * self.test_size
            test_end_idx = test_start_idx + self.test_size

            test_groups = unique_groups[test_start_idx:test_end_idx]
            if test_groups.size == 0:
                raise RuntimeError("Empty test window; check n_splits/test_size")

            train_end_idx = max(0, test_start_idx - self.purge_gap)
            train_groups = unique_groups[:train_end_idx]

            if embargoed_groups:
                mask = np.array([g not in embargoed_groups for g in train_groups], dtype=bool)
                train_groups = train_groups[mask]

            train_mask = np.isin(groups_1d, train_groups)
            test_mask = np.isin(groups_1d, test_groups)

            train_index = np.flatnonzero(train_mask)
            test_index = np.flatnonzero(test_mask)

            if self.embargo > 0:
                emb_start = test_end_idx
                emb_end = min(n_groups, test_end_idx + self.embargo)
                embargoed = unique_groups[emb_start:emb_end]
                embargoed_groups.update(embargoed.tolist())

            yield train_index, test_index

    def split_with_info(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray, FoldInfo]]:  # noqa: N803
        groups_1d = self._as_1d_array(groups)
        unique_groups = self._unique_sorted(groups_1d)

        required = self.n_splits * self.test_size
        if unique_groups.shape[0] < required:
            raise ValueError(
                f"Not enough unique groups for n_splits*test_size: have {unique_groups.shape[0]}, need {required}"
            )

        first_test_start = unique_groups.shape[0] - required
        embargoed_groups: set[object] = set()

        for i in range(self.n_splits):
            test_start_idx = first_test_start + i * self.test_size
            test_end_idx = test_start_idx + self.test_size

            test_groups = unique_groups[test_start_idx:test_end_idx]
            train_end_idx = max(0, test_start_idx - self.purge_gap)
            train_groups = unique_groups[:train_end_idx]

            if embargoed_groups:
                mask = np.array([g not in embargoed_groups for g in train_groups], dtype=bool)
                train_groups = train_groups[mask]

            train_index = np.flatnonzero(np.isin(groups_1d, train_groups))
            test_index = np.flatnonzero(np.isin(groups_1d, test_groups))

            info = FoldInfo(
                split_index=i,
                train_start=train_groups[0] if train_groups.size else None,
                train_end=train_groups[-1] if train_groups.size else None,
                test_start=test_groups[0],
                test_end=test_groups[-1],
                purge_gap=self.purge_gap,
                embargo=self.embargo,
            )

            if self.embargo > 0:
                emb_start = test_end_idx
                emb_end = min(unique_groups.shape[0], test_end_idx + self.embargo)
                embargoed = unique_groups[emb_start:emb_end]
                embargoed_groups.update(embargoed.tolist())

            yield train_index, test_index, info


def plot_cv_indices(
    cv: BaseCrossValidator,
    X,
    y=None,
    groups=None,
    *,
    ax=None,
    title: str | None = None,
):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 1 + 0.6 * cv.get_n_splits(X, y, groups)))

    n_samples = len(X)

    cmap = mcolors.ListedColormap(["#ffffff", "#1f77b4", "#d62728"])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    for ii, (train, test) in enumerate(cv.split(X, y, groups)):
        arr = np.zeros(n_samples, dtype=int)
        arr[train] = 1
        arr[test] = 2

        ax.scatter(
            np.arange(n_samples),
            np.full(n_samples, ii),
            c=arr,
            marker="|",
            s=120,
            cmap=cmap,
            norm=norm,
            linewidths=3,
        )

    ax.set_yticks(np.arange(cv.get_n_splits(X, y, groups)))
    ax.set_yticklabels([f"split {i}" for i in range(cv.get_n_splits(X, y, groups))])
    ax.set_xlabel("Sample index")
    ax.set_ylabel("CV split")
    if title is not None:
        ax.set_title(title)
    ax.set_xlim(-1, n_samples)
    ax.grid(False)

    return ax
