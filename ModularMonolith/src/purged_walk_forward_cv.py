from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    x_axis: str = "date",
):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 1 + 0.6 * cv.get_n_splits(X, y, groups)))

    n_samples = len(X)

    if x_axis not in {"index", "date"}:
        raise ValueError("x_axis must be 'index' or 'date'")

    if x_axis == "date":
        if groups is None:
            raise ValueError("groups is required when x_axis='date'")
        x = np.asarray(groups)
        x_label = "Date"
    else:
        x = np.arange(n_samples)
        x_label = "Sample index"

    cmap = mcolors.ListedColormap(["#ffffff", "#1f77b4", "#d62728"])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    for ii, (train, test) in enumerate(cv.split(X, y, groups)):
        arr = np.zeros(n_samples, dtype=int)
        arr[train] = 1
        arr[test] = 2

        ax.scatter(
            x,
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
    ax.set_xlabel(x_label)
    ax.set_ylabel("CV split")
    if title is not None:
        ax.set_title(title)

    if x_axis == "date":
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        try:
            ax.set_xlim(x.min(), x.max())
        except Exception:
            pass
    else:
        ax.set_xlim(-1, n_samples)
    ax.grid(False)

    return ax


def _repo_root() -> Path:
    # .../dissertation/ModularMonolith/src/purged_walk_forward_cv.py -> .../dissertation
    return Path(__file__).resolve().parents[2]


def _dataset_dir() -> Path:
    return _repo_root() / "ModularMonolith" / "data" / "dataset"


def _out_dir() -> Path:
    return _repo_root() / "ModularMonolith" / "data" / "traintestperiods"


def _load_groups_from_keys(keys_path: Path) -> np.ndarray:
    import pandas as pd

    keys = pd.read_parquet(keys_path)
    if "Date" not in keys.columns:
        raise ValueError(f"Missing Date column in keys file: {keys_path}")
    dt = pd.to_datetime(keys["Date"], errors="raise")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt.to_numpy()


def _simulate_synthetic_price(dates: np.ndarray, *, seed: int = 123, start: float = 100.0) -> np.ndarray:
    """Synthetic price series to visualize CV windows.

    Uses a simple geometric random walk so the plotted line looks like a market price.
    """

    rng = np.random.default_rng(seed)
    n = int(len(dates))
    if n <= 0:
        raise ValueError("dates must be non-empty")

    # Daily log-return random walk (roughly 1% daily vol).
    lr = rng.normal(loc=0.0, scale=0.01, size=n)
    price = float(start) * np.exp(np.cumsum(lr))
    return price


def plot_cv_price_panels(
    *,
    cv: PurgedWalkForwardCV,
    unique_dates: np.ndarray,
    price: np.ndarray,
    title: str,
) -> "plt.Figure":
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    if len(unique_dates) != len(price):
        raise ValueError("unique_dates and price must have same length")

    n_splits = cv.get_n_splits()
    fig_h = max(6.0, 0.55 * n_splits)
    fig, axes = plt.subplots(n_splits, 1, figsize=(14, fig_h), sharex=True, constrained_layout=True)
    if n_splits == 1:
        axes = [axes]

    groups = unique_dates
    X_dummy = np.zeros((len(groups), 1), dtype=float)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)

    for ax, (_, _, info) in zip(axes, cv.split_with_info(X_dummy, groups=groups), strict=False):
        ax.plot(unique_dates, price, color="black", linewidth=1.0)

        # Shade Train (blue) and Test (red). Purge gap remains unshaded by design.
        if info.train_start is not None and info.train_end is not None:
            ax.axvspan(
                pd.Timestamp(info.train_start),
                pd.Timestamp(info.train_end) + pd.Timedelta(days=1),
                color="#1f77b4",
                alpha=0.15,
                lw=0,
            )

        ax.axvspan(
            pd.Timestamp(info.test_start),
            pd.Timestamp(info.test_end) + pd.Timedelta(days=1),
            color="#d62728",
            alpha=0.15,
            lw=0,
        )

        ax.set_ylabel(f"fold {info.split_index}")
        ax.grid(True, alpha=0.15)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        t0 = str(info.train_start)[:10] if info.train_start is not None else "None"
        t1 = str(info.train_end)[:10] if info.train_end is not None else "None"
        s0 = str(info.test_start)[:10]
        s1 = str(info.test_end)[:10]
        ax.set_title(
            f"Fold {info.split_index}: Train {t0} → {t1} | Purge gap={info.purge_gap} | Test {s0} → {s1}",
            fontsize=9,
        )

    fig.suptitle(title, fontsize=12)
    axes[-1].set_xlabel("Date")
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    horizons = (5, 21, 63)
    dataset_dir = _dataset_dir()
    out_dir = _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        keys_path = dataset_dir / f"h{h}" / "keys.parquet"
        if not keys_path.exists():
            raise FileNotFoundError(keys_path)

        groups = _load_groups_from_keys(keys_path)
        unique_dates = np.unique(groups)
        unique_dates = np.sort(unique_dates)
        price = _simulate_synthetic_price(unique_dates, seed=123 + int(h), start=100.0)

        purge_gap = int(h)
        test_size = 126 if h == 63 else 63
        n_splits = 10 if h == 63 else 20

        cv = PurgedWalkForwardCV(
            n_splits=n_splits,
            test_size=test_size,
            purge_gap=purge_gap,
            embargo=0,
        )

        out_path = out_dir / f"cv_h{h}.png"

        title = (
            f"Purged Walk-Forward CV on synthetic price (h={h}) | "
            f"folds={n_splits} | test_size={test_size} | purge_gap={purge_gap}"
        )
        fig = plot_cv_price_panels(cv=cv, unique_dates=unique_dates, price=price, title=title)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved CV plot: {out_path}")
