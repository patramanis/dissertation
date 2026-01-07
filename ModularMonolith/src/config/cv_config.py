from __future__ import annotations

from typing import Any


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
