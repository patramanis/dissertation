from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd


def get_cv_config(horizon: int) -> dict[str, Any]:
    h = int(horizon)
    embargo_map = {5: 2, 21: 5, 63: 10}
    if h not in embargo_map:
        raise ValueError(f"Unsupported horizon={h}. Expected one of {sorted(embargo_map)}")

    if h == 63:
        n_splits = 10
        test_size = 126
    else:
        n_splits = 20
        test_size = 63

    return {
        "n_splits": n_splits,
        "test_size": test_size,
        "purge_gap": h,
        "embargo": embargo_map[h],
    }


def get_xgboost_groups(df: pd.DataFrame, group_col: str = "Date") -> np.ndarray:
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col={group_col!r} in df")

    df_sorted = df.sort_values(group_col, kind="mergesort")
    group_sizes = df_sorted.groupby(group_col, sort=True).size().to_numpy(dtype=np.int32)

    if int(group_sizes.sum()) != len(df_sorted):
        raise RuntimeError("Group sizes do not sum to number of rows")

    return group_sizes