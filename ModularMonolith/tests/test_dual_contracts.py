from __future__ import annotations

import numpy as np
import pandas as pd

from ModularMonolith.data.dataset_shaping import EXPECTED_SECTORS


def _make_synthetic_dual_panel(*, n_dates: int = 50, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Synthetic dual-stage dataset that matches ranker contracts.

    - Panel: Date×Sector with EXPECTED_SECTORS
    - label_excess: per-date strict ordering with exactly 3 winners (positive) and 6 losers (negative)
    - y_rank: within-date rank target (best gets n-1)
    - groups: Date per row
    """
    rng = np.random.default_rng(int(seed))

    dates = pd.date_range("2000-01-03", periods=int(n_dates), freq="B")
    sectors = list(EXPECTED_SECTORS)

    idx = pd.MultiIndex.from_product([dates, sectors], names=["Date", "Sector"])
    base = idx.to_frame(index=False)

    # Raw signal that models can learn (sector effect + slow drift + tiny noise).
    sec_code = base["Sector"].map({s: i for i, s in enumerate(sectors)}).to_numpy(dtype=float)
    t = (pd.to_datetime(base["Date"]).astype("int64") // 86_400_000_000_000).to_numpy(dtype=float)

    f_sector = (sec_code - sec_code.mean()) / (sec_code.std() + 1e-12)
    f_regime = np.sin(t / 17.0)
    f_noise = rng.standard_normal(len(base)) * 0.05

    raw = 0.70 * f_sector + 0.25 * f_regime + 0.05 * f_noise

    # Build strict per-date ranking, deterministic tie-break by EXPECTED_SECTORS order.
    df = base.copy()
    df["raw"] = raw
    df["_sector_order"] = df["Sector"].map({s: i for i, s in enumerate(sectors)}).astype(int)

    label_vals = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=float)
    if len(label_vals) != len(sectors):
        raise RuntimeError("EXPECTED_SECTORS length changed; update label_vals")

    out_label = np.zeros(len(df), dtype=float)
    out_rank = np.zeros(len(df), dtype=np.int16)

    for dt, sub in df.groupby("Date", sort=True):
        sub_sorted = sub.sort_values(["raw", "_sector_order"], ascending=[False, True], kind="mergesort")
        idx_sorted = sub_sorted.index.to_numpy()
        out_label[idx_sorted] = label_vals

        # Rank target: best gets n-1, worst gets 0.
        n = len(sub_sorted)
        out_rank[idx_sorted] = (n - 1) - np.arange(n, dtype=np.int16)

    X = pd.DataFrame(
        {
            "f_sector": f_sector,
            "f_regime": f_regime,
            "f_noise": f_noise,
        }
    )

    y = pd.DataFrame(
        {
            "Date": df["Date"].to_numpy(dtype="datetime64[ns]"),
            "Sector": df["Sector"].astype("string"),
            "label_excess": out_label,
            "y_gate": (out_label > 0.0).astype(np.int8),
            "y_rank": out_rank.astype(np.float32),
        }
    )

    groups = y["Date"].to_numpy(dtype="datetime64[ns]")
    return X, y, groups


def test_rank_target_semantics_argmax_matches_label_excess_per_date() -> None:
    X, y, groups = _make_synthetic_dual_panel(n_dates=40, seed=1)
    assert len(X) == len(y) == len(groups)

    for dt, sub in y.groupby("Date", sort=True):
        i_best_excess = int(sub["label_excess"].to_numpy().argmax())
        i_best_rank = int(sub["y_rank"].to_numpy().argmax())
        assert i_best_rank == i_best_excess


def test_ranker_groups_equal_y_date() -> None:
    _X, y, groups = _make_synthetic_dual_panel(n_dates=20, seed=2)
    np.testing.assert_array_equal(groups, y["Date"].to_numpy(dtype="datetime64[ns]"))


def test_group_size_equals_expected_sectors() -> None:
    _X, y, _groups = _make_synthetic_dual_panel(n_dates=15, seed=3)
    sizes = y.groupby("Date", sort=True).size().to_numpy()
    assert int(sizes.min()) == int(sizes.max()) == len(EXPECTED_SECTORS)
