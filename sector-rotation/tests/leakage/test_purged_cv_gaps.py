from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
WS = THIS_DIR.parents[1]

@pytest.mark.parametrize("cv_type", ["rolling"])
def test_purged_cv_gaps_h21(cv_type: str) -> None:
    h = 21
    purge_buffer = 5
    expected_gap = h + purge_buffer

    panel_path = WS / "data" / "processed" / "panel" / f"panel_h{h}.csv"
    cv_path = WS / "configs" / "cv" / f"cv_{cv_type}_h{h}.json"

    if not panel_path.exists():
        pytest.skip(f"Panel not found: {panel_path}")
    if not cv_path.exists():
        pytest.skip(f"CV config not found: {cv_path}")

    panel = pd.read_csv(panel_path, usecols=["Date"], parse_dates=["Date"])
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["Date"]).dt.normalize().unique()))
    date_to_idx = {d: i for i, d in enumerate(dates)}

    folds = json.loads(cv_path.read_text(encoding="utf-8")).get("folds", [])
    assert folds, f"No folds found in {cv_path}"

    def idx(d: str) -> int:
        return int(date_to_idx[pd.Timestamp(d).normalize()])

    for f in folds:
        gap = int(f.get("purge_gap", expected_gap))

        te = idx(f["train_end"])
        ts = idx(f["test_start"])

        gap_train_test = ts - te - 1
        assert gap_train_test >= gap, (
            f"Fold {f.get('fold_id')} {cv_type}: train→test gap {gap_train_test} < {gap}"
        )

        if f.get("valid_start") and f.get("valid_end"):
            ve = idx(f["valid_end"])
            gap_valid_test = ts - ve - 1
            assert gap_valid_test >= gap, (
                f"Fold {f.get('fold_id')} {cv_type}: valid→test gap {gap_valid_test} < {gap}"
            )


def test_rolling_reaches_panel_end_h21() -> None:
    h = 21

    panel_path = WS / "data" / "processed" / "panel" / f"panel_h{h}.csv"
    cv_path = WS / "configs" / "cv" / f"cv_rolling_h{h}.json"

    if not panel_path.exists() or not cv_path.exists():
        pytest.skip("Missing panel or CV config")

    panel = pd.read_csv(panel_path, usecols=["Date"], parse_dates=["Date"])
    panel_end = pd.to_datetime(panel["Date"]).dt.normalize().max()

    folds = json.loads(cv_path.read_text(encoding="utf-8")).get("folds", [])
    assert folds

    last_test_end = pd.Timestamp(folds[-1]["test_end"]).normalize()
    assert last_test_end == panel_end, f"Last test_end {last_test_end} != panel_end {panel_end}"