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
    """
    EXPANDING-ONLY Walk-Forward CV with Date-based Purging.
    
    Design:
    - Train window GROWS monotonically (never shrinks).
    - Train is ALWAYS chronologically before Test (no overlap, no Swiss cheese).
    - Embargo is NOT supported (enforce_expanding_only=True raises if embargo>0).
    - For Rolling Window or K-Fold with embargo, use a different CV class.
    
    Purging:
    - Index-based on unique trading dates (assumes daily calendar with no gaps).
    - purge_gap should match label horizon h for contract Return(T-1 → T-1+h).
    
    Performance:
    - O(N) indexing per fold via factorized group codes (not O(N×G) isin).
    - Suitable for large panel datasets (Date×Sector with N~50k rows).
    """

    def __init__(
        self,
        *,
        n_splits: int | None = 5,
        test_size: int = 63,
        purge_gap: int = 21,
        min_train_size: int = 252,
        embargo: int = 0,
        test_start: object | None = None,
        enforce_expanding_only: bool = True,
    ) -> None:
        """
        CRITICAL ASSUMPTIONS:
        1. EXPANDING WINDOW ONLY: embargo must be 0 (Train grows monotonically and is always before Test).
           For K-Fold or Rolling Window with embargo, use a different CV class.
        2. Purge gap is INDEX-based on unique trading dates (not timedelta-based).
           Safe for Daily trading calendar. RISKY for intraday or sparse data.
        3. Groups must be Date (one per sample) with repeats for panel data (e.g., Date×Sector).
        """
        if n_splits is not None and n_splits <= 0:
            raise ValueError("n_splits must be positive (or None for dynamic)")
        if test_size <= 0:
            raise ValueError("test_size must be positive")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")
        if min_train_size <= 0:
            raise ValueError("min_train_size must be positive")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        
        # FIX (1): Enforce embargo=0 for expanding-only design (no "fake safety")
        if enforce_expanding_only and embargo != 0:
            raise ValueError(
                f"This CV is EXPANDING-ONLY: embargo must be 0 (got embargo={embargo}). "
                "Expanding Window means Train is ALWAYS chronologically before Test, "
                "so embargo (which excludes dates immediately after Test from Train) has no effect. "
                "If you need embargo>0, use a Rolling Window or K-Fold CV class instead."
            )

        self.n_splits = int(n_splits) if n_splits is not None else None
        self.test_size = int(test_size)
        self.purge_gap = int(purge_gap)
        self.min_train_size = int(min_train_size)
        self.embargo = int(embargo)
        self.test_start = test_start

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if self.n_splits is not None:
            return self.n_splits
        if groups is None:
            raise ValueError("groups is required to compute dynamic n_splits")
        groups_1d = self._as_1d_array(groups)
        unique_groups = self._unique_sorted(groups_1d)
        n_groups = int(unique_groups.shape[0])
        first_test_start = self._resolve_first_test_start(unique_groups)
        remaining = max(0, n_groups - first_test_start)
        return int((remaining + self.test_size - 1) // self.test_size)

    def _resolve_first_test_start(self, unique_groups: np.ndarray) -> int:
        n_groups = int(unique_groups.shape[0])
        if self.test_start is None:
            if self.n_splits is None:
                raise ValueError("n_splits cannot be None when test_start is None")
            return n_groups - (int(self.n_splits) * self.test_size)

        ts = self.test_start
        if np.issubdtype(unique_groups.dtype, np.datetime64):
            try:
                import pandas as pd

                ts = np.datetime64(pd.Timestamp(ts).to_datetime64())
            except Exception:
                ts = np.asarray(ts, dtype=unique_groups.dtype).item()
        else:
            try:
                ts = np.asarray(ts, dtype=unique_groups.dtype).item()
            except Exception:
                ts = ts

        i = int(np.searchsorted(unique_groups, ts, side="left"))
        if i < 0:
            i = 0
        if i > n_groups:
            i = n_groups
        return i

    @staticmethod
    def _as_1d_array(groups) -> np.ndarray:
        if groups is None:
            raise ValueError("groups is required and must contain the Date for each sample")
        g = np.asarray(groups)
        if g.ndim != 1:
            g = np.ravel(g)
        
        # FIX (3): Normalize object dtype (e.g., pd.Timestamp) to datetime64[ns]
        # This prevents sort/comparison failures with mixed types
        if g.dtype == object or str(g.dtype).startswith("datetime"):
            try:
                import pandas as pd
                g = pd.to_datetime(g, errors="raise").to_numpy(dtype="datetime64[ns]")
            except Exception as e:
                raise ValueError(
                    f"groups contains non-datetime or incompatible types. "
                    f"Expected datetime-like array, got dtype={g.dtype}"
                ) from e
        
        return g

    @staticmethod
    def _unique_sorted(groups_1d: np.ndarray) -> np.ndarray:
        uniq = np.unique(groups_1d)
        return np.sort(uniq)

    def _iter_splits(
        self,
        groups_1d: np.ndarray,
        *,
        want_info: bool,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, FoldInfo | None]]:
        n_samples = groups_1d.shape[0]
        if n_samples <= 0:
            raise ValueError("Empty dataset")

        unique_groups = self._unique_sorted(groups_1d)
        n_groups = unique_groups.shape[0]

        first_test_start = self._resolve_first_test_start(unique_groups)
        if int(first_test_start) >= int(n_groups):
            raise ValueError(
                "test_start is after the last available group/date. "
                f"test_start={self.test_start} last_group={unique_groups[-1] if int(n_groups) else None}"
            )

        max_by_data = int((int(n_groups) - int(first_test_start) + self.test_size - 1) // self.test_size)
        effective_splits = max_by_data if self.n_splits is None else min(int(self.n_splits), max_by_data)
        if effective_splits <= 0:
            raise ValueError("No feasible splits: not enough groups after test_start")

        # FIX (6): Remove embargo from required check (it's not applied in expanding window)
        required = effective_splits * self.test_size + self.purge_gap + self.min_train_size
        if n_groups < required and self.test_start is None:
            raise ValueError(
                f"Not enough unique groups for n_splits*test_size+purge_gap+min_train_size: "
                f"have {n_groups}, need {required}"
            )
        
        # FIX (4): Factorize groups_1d to integer codes for O(N) performance (not O(N×G))
        # Map each sample → group_code (0..n_groups-1)
        group_codes = np.searchsorted(unique_groups, groups_1d)
        if not np.all((group_codes >= 0) & (group_codes < n_groups)):
            raise ValueError("groups_1d contains values not in unique_groups (corrupted data)")

        for i in range(effective_splits):
            test_start_idx = first_test_start + i * self.test_size
            test_end_idx = test_start_idx + self.test_size

            if int(test_start_idx) >= int(n_groups):
                break
            test_end_idx = min(int(test_end_idx), int(n_groups))

            test_groups = unique_groups[test_start_idx:test_end_idx]
            if test_groups.size == 0:
                raise RuntimeError("Empty test window; check n_splits/test_size")

            # Index-based purging assumes uniform date spacing (trading days).
            # For label contract Return(T-1 → T-1+h), purge_gap should be == h (not h+1).
            train_end_idx = max(0, test_start_idx - self.purge_gap)
            train_groups = unique_groups[:train_end_idx]

            if train_groups.size < self.min_train_size:
                raise RuntimeError(
                    f"Split {i}: train_groups.size={int(train_groups.size)} < min_train_size={self.min_train_size}. "
                    f"n_groups={int(n_groups)}, purge_gap={self.purge_gap}"
                )

            # FIX (4): Use factorized group_codes for O(N) indexing (not O(N×G) isin)
            train_mask = group_codes < train_end_idx
            test_mask = (group_codes >= test_start_idx) & (group_codes < test_end_idx)

            train_index = np.flatnonzero(train_mask)
            test_index = np.flatnonzero(test_mask)

            if train_index.size == 0:
                raise RuntimeError(
                    f"Split {i}: Training set is empty. "
                    f"Too many splits ({self.n_splits}) or purge_gap ({self.purge_gap}) is too large for the dataset size."
                )
            
            # FIX (7): Correctness asserts to catch data corruption early
            overlap = np.intersect1d(train_index, test_index)
            if overlap.size > 0:
                raise RuntimeError(
                    f"Split {i}: Train/Test overlap detected ({overlap.size} samples). "
                    "This indicates a bug in the CV logic or corrupted groups data."
                )
            
            # Expanding window: Train must be chronologically before Test
            if train_groups.size > 0 and test_groups.size > 0:
                if train_groups[-1] >= test_groups[0]:
                    raise RuntimeError(
                        f"Split {i}: Train end ({train_groups[-1]}) >= Test start ({test_groups[0]}). "
                        f"Expanding window violated. purge_gap={self.purge_gap} may be too small."
                    )

            info: FoldInfo | None = None
            if want_info:
                info = FoldInfo(
                    split_index=i,
                    train_start=train_groups[0] if train_groups.size else None,
                    train_end=train_groups[-1] if train_groups.size else None,
                    test_start=test_groups[0],
                    test_end=test_groups[-1],
                    purge_gap=self.purge_gap,
                    embargo=self.embargo,
                )

            # REMOVED: Cumulative embargo persistence (Swiss Cheese bug)
            # Old code:
            # if self.embargo > 0:
            #     emb_start = test_end_idx
            #     emb_end = min(n_groups, test_end_idx + self.embargo)
            #     embargoed = unique_groups[emb_start:emb_end]
            #     embargoed_groups.update(embargoed.tolist())
            # 
            # This created permanent "holes" in the training set for all future splits.
            # In Expanding Window, Train[t] ⊂ Train[t+1], so past embargos should NOT persist.

            yield train_index, test_index, info

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups_1d = self._as_1d_array(groups)
        for train_index, test_index, _ in self._iter_splits(groups_1d, want_info=False):
            yield train_index, test_index

    def split_with_info(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray, FoldInfo]]:
        groups_1d = self._as_1d_array(groups)
        for train_index, test_index, info in self._iter_splits(groups_1d, want_info=True):
            if info is None:
                raise RuntimeError("Internal error: want_info=True but info is None")
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
        # FIX (5): Pass groups to get_n_splits (required if n_splits=None)
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

    # FIX (5): Pass groups consistently
    n_splits_val = cv.get_n_splits(X, y, groups)
    ax.set_yticks(np.arange(n_splits_val))
    ax.set_yticklabels([f"split {i}" for i in range(n_splits_val)])
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
    return Path(__file__).resolve().parents[2]


def _mm_root() -> Path:
    return _repo_root()


def _dataset_dir() -> Path:
    return _mm_root() / "data" / "dataset"


def _out_dir() -> Path:
    return _mm_root() / "data" / "traintestperiods"


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
    rng = np.random.default_rng(seed)
    n = int(len(dates))
    if n <= 0:
        raise ValueError("dates must be non-empty")

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

    # FIX (5): Pass groups to get_n_splits
    groups = unique_dates
    X_dummy = np.zeros((len(groups), 1), dtype=float)
    n_splits = cv.get_n_splits(X_dummy, None, groups)
    fig_h = max(6.0, 0.55 * n_splits)
    fig, axes = plt.subplots(n_splits, 1, figsize=(14, fig_h), sharex=True, constrained_layout=True)
    if n_splits == 1:
        axes = [axes]

    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)

    for ax, (_, _, info) in zip(axes, cv.split_with_info(X_dummy, groups=groups), strict=False):
        ax.plot(unique_dates, price, color="black", linewidth=1.0)

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

        # FIX (8): Load full panel groups (Date with repeats for each Sector)
        panel_groups = _load_groups_from_keys(keys_path)
        unique_dates = np.unique(panel_groups)
        unique_dates = np.sort(unique_dates)
        price = _simulate_synthetic_price(unique_dates, seed=123 + int(h), start=100.0)

        # FIX (2): purge_gap should match label contract (h for Return(T-1→T-1+h))
        purge_gap = int(h)
        test_size = 126 if h == 63 else 63
        n_splits = 10 if h == 63 else 20

        cv = PurgedWalkForwardCV(
            n_splits=n_splits,
            test_size=test_size,
            purge_gap=purge_gap,
            embargo=0,
            enforce_expanding_only=True,
        )

        # Validate CV correctness on panel data
        print(f"\n[h={h}] Validating CV on panel data (n_samples={len(panel_groups)})...")
        X_panel = np.zeros((len(panel_groups), 1), dtype=float)
        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_panel, groups=panel_groups)):
            train_dates = panel_groups[train_idx]
            test_dates = panel_groups[test_idx]
            if fold_i == 0:
                print(f"  Fold {fold_i}: train_size={len(train_idx)} test_size={len(test_idx)}")
                print(f"    train_dates: {np.min(train_dates)} to {np.max(train_dates)}")
                print(f"    test_dates: {np.min(test_dates)} to {np.max(test_dates)}")

        out_path = out_dir / f"cv_h{h}.png"

        title = (
            f"Purged Walk-Forward CV (h={h}) | "
            f"folds={n_splits} | test_size={test_size} | purge_gap={purge_gap} (expanding-only)"
        )
        fig = plot_cv_price_panels(cv=cv, unique_dates=unique_dates, price=price, title=title)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved CV plot: {out_path}")
