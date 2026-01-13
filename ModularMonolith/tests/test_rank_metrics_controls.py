from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ModularMonolith.src.reporting.metrics_diagnostics import oracle_ranker_control, random_ranker_control
from ModularMonolith.src.reporting.metrics_rank import compute_rank_metrics, lift_at_k, precision_at_k, uplift_at_k


def test_k_eff_contract_no_skip_precision_lift_uplift() -> None:
    # Contract: for each group use k_eff=min(k, group_size) and do NOT skip.
    # Single group with size=2, ask k=3.
    y = np.array([1.0, 0.0])
    pred = np.array([0.0, 1.0])  # lower score better -> selects index 0 first
    g = np.array(["2000-01-03", "2000-01-03"], dtype="datetime64[ns]")

    p = precision_at_k(y, pred, g, k=3, threshold=0.5)
    assert p == pytest.approx(0.5, abs=1e-12)  # 1 winner out of k_eff=2

    l = lift_at_k(y, pred, g, k=3, cost_threshold=0.5)
    assert l == pytest.approx(0.0, abs=1e-12)  # precision==base_rate==0.5

    u = uplift_at_k(y, pred, g, k=3)
    assert u == pytest.approx(0.0, abs=1e-12)  # selects all -> no uplift


def _make_strict_rank_panel(*, n_dates: int = 250, n_items: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Strictly ordered within each group with both winners/losers (no ties).
    # We want the default winner definition (y > 0) to yield base_rate=3/9.
    # This makes Spearman IC exactly -1 for oracle score = -y.
    dates = pd.date_range("2000-01-03", periods=int(n_dates), freq="B").to_numpy(dtype="datetime64[ns]")
    if int(n_items) != 9:
        raise ValueError("This helper assumes n_items=9")

    # 3 winners (positive) + 6 losers (negative), strictly decreasing.
    y_one = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=float)

    y = np.tile(y_one, int(n_dates))
    g = np.repeat(dates, int(n_items))
    # Oracle score under LOWER-is-better convention
    pred_oracle = -y
    return y, pred_oracle, g


def test_oracle_metrics_consistent_with_lower_is_better_convention() -> None:
    y, pred_oracle, g = _make_strict_rank_panel(n_dates=200, n_items=9)

    # Default winner definition for controls is y > 0.
    m = compute_rank_metrics(y, pred_oracle, g, cost_threshold=0.0)
    assert m.rank_ic == pytest.approx(-1.0, abs=1e-12)
    assert m.precision_at_3 == pytest.approx(1.0, abs=1e-12)
    assert m.lift_at_3 == pytest.approx(2.0, abs=1e-12)  # base_rate=3/9 => lift=(1/(1/3))-1=2

    diag = oracle_ranker_control(y, g)
    assert diag["oracle_rank_ic"] == pytest.approx(1.0, abs=1e-12)
    assert diag["oracle_rank_ic_signed"] == pytest.approx(-1.0, abs=1e-12)


def test_random_ranker_control_expected_behavior_in_expectation() -> None:
    # With many dates, random should concentrate near IC≈0, lift≈0, precision≈base_rate.
    y, _pred_oracle, g = _make_strict_rank_panel(n_dates=300, n_items=9)
    base_rate = 3.0 / 9.0

    rc = random_ranker_control(y, g, seed=123)

    assert abs(float(rc["random_rank_ic"])) < 0.15

    # precision@3 should be close to base rate in expectation.
    assert float(rc["random_precision@3"]) == pytest.approx(base_rate, abs=0.06)

    # lift should be close to 0 in expectation.
    assert float(rc["random_lift@3"]) == pytest.approx(0.0, abs=0.20)

    # Sanity: oracle with the same y/gs still works and stays finite.
    oc = oracle_ranker_control(y, g)
    for k, v in oc.items():
        assert np.isfinite(float(v)), f"oracle control produced non-finite {k}={v}"
