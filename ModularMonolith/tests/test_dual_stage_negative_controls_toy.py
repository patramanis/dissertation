from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from ModularMonolith.data.dataset_shaping import EXPECTED_SECTORS
from ModularMonolith.src.cv.purged_walk_forward_cv import PurgedWalkForwardCV
from ModularMonolith.src.reporting.metrics_rank import compute_rank_metrics


@dataclass(frozen=True)
class ToyDualOOF:
    df: pd.DataFrame
    gate_auc: float
    rank_metrics: object
    rank_metrics_gated: object


def _make_synthetic_dual_panel(*, n_dates: int = 260, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(int(seed))

    dates = pd.date_range("2000-01-03", periods=int(n_dates), freq="B")
    sectors = list(EXPECTED_SECTORS)

    idx = pd.MultiIndex.from_product([dates, sectors], names=["Date", "Sector"])
    base = idx.to_frame(index=False)

    sec_code = base["Sector"].map({s: i for i, s in enumerate(sectors)}).to_numpy(dtype=float)
    t = (pd.to_datetime(base["Date"]).astype("int64") // 86_400_000_000_000).to_numpy(dtype=float)

    f_sector = (sec_code - sec_code.mean()) / (sec_code.std() + 1e-12)
    f_regime = np.sin(t / 17.0)
    f_noise = rng.standard_normal(len(base)) * 0.05

    # Learnable raw signal.
    raw = 0.70 * f_sector + 0.25 * f_regime + 0.05 * f_noise

    df = base.copy()
    df["raw"] = raw
    df["_sector_order"] = df["Sector"].map({s: i for i, s in enumerate(sectors)}).astype(int)

    # Deterministic winners/losers per date: 3 positives + 6 negatives.
    label_vals = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=float)
    if len(label_vals) != len(sectors):
        raise RuntimeError("EXPECTED_SECTORS length changed; update label_vals")

    out_label = np.zeros(len(df), dtype=float)
    out_rank = np.zeros(len(df), dtype=np.int16)

    for _dt, sub in df.groupby("Date", sort=True):
        sub_sorted = sub.sort_values(["raw", "_sector_order"], ascending=[False, True], kind="mergesort")
        idx_sorted = sub_sorted.index.to_numpy()
        out_label[idx_sorted] = label_vals

        n = len(sub_sorted)
        out_rank[idx_sorted] = (n - 1) - np.arange(n, dtype=np.int16)

    X = pd.DataFrame({"f_sector": f_sector, "f_regime": f_regime, "f_noise": f_noise})

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


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel().astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _permute_within_group(values: np.ndarray, groups: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute values within each group with a per-group deterministic RNG.

    IMPORTANT: Using a single RNG across groups can accidentally introduce
    repeated/structured permutations that create a spurious sector bias.
    Seeding per-group (per Date) keeps it deterministic while breaking any
    cross-date mapping.
    """
    out = np.asarray(values).copy()
    df = pd.DataFrame({"g": np.asarray(groups).ravel()})
    for gval, idx in df.groupby("g", sort=False).indices.items():
        ii = np.asarray(idx)
        # Stable per-date seed (datetime64 -> int64 nanoseconds)
        g_int = int(np.datetime64(gval).astype("datetime64[ns]").astype("int64"))
        group_seed = (int(seed) ^ (g_int & 0xFFFFFFFF)) & 0xFFFFFFFF
        rng_g = np.random.default_rng(group_seed)
        out[ii] = rng_g.permutation(out[ii])
    return out


def _randomize_within_group(values: np.ndarray, groups: np.ndarray, *, seed: int) -> np.ndarray:
    """Replace values within each group by deterministic random noise.

    This is a stronger negative control than permutation: it guarantees there is
    no stable mapping between X and the training target, while preserving group
    sizes/order and keeping the pipeline contracts intact.
    """
    out = np.asarray(values).copy().astype(float)
    df = pd.DataFrame({"g": np.asarray(groups).ravel()})
    for gval, idx in df.groupby("g", sort=False).indices.items():
        ii = np.asarray(idx)
        g_int = int(np.datetime64(gval).astype("datetime64[ns]").astype("int64"))
        group_seed = (int(seed) ^ (g_int & 0xFFFFFFFF) ^ 0x9E3779B9) & 0xFFFFFFFF
        rng_g = np.random.default_rng(group_seed)
        out[ii] = rng_g.standard_normal(size=len(ii))
    return out


def run_toy_dual_cv(
    *,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
    outer_cv: PurgedWalkForwardCV,
    k: int = 3,
    gate_threshold: float = 0.5,
    permute_gate_train: bool = False,
    permute_ranker_train: bool = False,
    seed: int = 0,
) -> ToyDualOOF:
    from sklearn.linear_model import LogisticRegression, Ridge

    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X.to_numpy(), groups=groups), start=1):
        X_tr = X.iloc[train_idx].to_numpy()
        X_te = X.iloc[test_idx].to_numpy()

        y_gate_tr = y.iloc[train_idx]["y_gate"].to_numpy(dtype=int)
        y_gate_te = y.iloc[test_idx]["y_gate"].to_numpy(dtype=int)

        y_rank_tr = y.iloc[train_idx]["y_rank"].to_numpy(dtype=float)

        g_tr = groups[train_idx]

        if permute_gate_train:
            rng = np.random.default_rng(int(seed))
            y_gate_tr = rng.permutation(y_gate_tr)

        if permute_ranker_train:
            # Contract: preserve group structure; destroy signal by permuting relevance within Date.
            y_rank_tr = _randomize_within_group(y_rank_tr, g_tr, seed=seed)

        gate = LogisticRegression(solver="liblinear", random_state=int(seed), max_iter=200)
        gate.fit(X_tr, y_gate_tr)
        p_gate = gate.predict_proba(X_te)[:, 1]

        ranker = Ridge(alpha=1.0, random_state=int(seed))
        ranker.fit(X_tr, y_rank_tr)
        # Higher predicted relevance = better; convert to LOWER-is-better score for evaluator.
        rank_score = -ranker.predict(X_te).astype(float)

        part = y.iloc[test_idx][["Date", "Sector", "label_excess", "y_gate"]].copy()
        part["fold_id"] = int(fold_id)
        part["p_gate"] = p_gate
        part["rank_score"] = rank_score
        rows.append(part)

    oof = pd.concat(rows, ignore_index=True)

    gate_auc = _auc(oof["y_gate"].to_numpy(), oof["p_gate"].to_numpy())

    # Ungated rank metrics: evaluate ranking scores on the whole universe.
    rank_metrics = compute_rank_metrics(
        oof["label_excess"].to_numpy(),
        oof["rank_score"].to_numpy(),
        oof["Date"].to_numpy(dtype="datetime64[ns]"),
        cost_threshold=0.0,
    )

    # Gated: evaluate only among sectors where gate passes.
    gated = oof.loc[oof["p_gate"].to_numpy(dtype=float) >= float(gate_threshold)].copy()
    if gated.empty:
        rank_metrics_gated = compute_rank_metrics(
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype="datetime64[ns]"),
            cost_threshold=0.0,
        )
    else:
        rank_metrics_gated = compute_rank_metrics(
            gated["label_excess"].to_numpy(),
            gated["rank_score"].to_numpy(),
            gated["Date"].to_numpy(dtype="datetime64[ns]"),
            cost_threshold=0.0,
        )

    return ToyDualOOF(df=oof, gate_auc=gate_auc, rank_metrics=rank_metrics, rank_metrics_gated=rank_metrics_gated)


def _selected_topk(oof: pd.DataFrame, *, k: int, gate_threshold: float) -> pd.DataFrame:
    # Deterministic selection set per date (used for invariance tests).
    df = oof.loc[oof["p_gate"].to_numpy(dtype=float) >= float(gate_threshold)].copy()
    if df.empty:
        return df[["Date", "Sector"]].copy()

    picked = []
    for dt, sub in df.groupby("Date", sort=True):
        k_eff = min(int(k), len(sub))
        sub_sorted = sub.sort_values(["rank_score", "Sector"], ascending=[True, True], kind="mergesort")
        picked.append(sub_sorted.head(k_eff)[["Date", "Sector"]])
    return pd.concat(picked, ignore_index=True) if picked else df[["Date", "Sector"]].head(0)


def test_toy_dual_permute_gate_train_collapses_combined_metrics() -> None:
    X, y, groups = _make_synthetic_dual_panel(n_dates=260, seed=1)
    uniq_dates = np.sort(np.unique(groups))

    cv = PurgedWalkForwardCV(
        n_splits=3,
        test_size=20,
        purge_gap=10,
        min_train_size=80,
        embargo=0,
        test_start=uniq_dates[180],
    )

    good = run_toy_dual_cv(X=X, y=y, groups=groups, outer_cv=cv, permute_gate_train=False, permute_ranker_train=False, seed=0)
    assert good.gate_auc > 0.60
    assert abs(float(good.rank_metrics.rank_ic)) > 0.80

    bad_gate = run_toy_dual_cv(X=X, y=y, groups=groups, outer_cv=cv, permute_gate_train=True, permute_ranker_train=False, seed=0)
    assert bad_gate.gate_auc < 0.58

    # Combined (gated) ranking should collapse when the gate is permuted.
    assert abs(float(bad_gate.rank_metrics_gated.rank_ic)) < 0.40


def test_toy_dual_permute_ranker_train_collapses_rank_metrics_only() -> None:
    # Use more dates so the random-label negative control concentrates near IC≈0.
    X, y, groups = _make_synthetic_dual_panel(n_dates=360, seed=2)
    uniq_dates = np.sort(np.unique(groups))

    cv = PurgedWalkForwardCV(
        n_splits=4,
        test_size=25,
        purge_gap=10,
        min_train_size=100,
        embargo=0,
        test_start=uniq_dates[220],
    )

    good = run_toy_dual_cv(X=X, y=y, groups=groups, outer_cv=cv, permute_gate_train=False, permute_ranker_train=False, seed=0)
    assert good.gate_auc > 0.60
    assert abs(float(good.rank_metrics.rank_ic)) > 0.80

    bad_rank = run_toy_dual_cv(X=X, y=y, groups=groups, outer_cv=cv, permute_gate_train=False, permute_ranker_train=True, seed=0)
    # Gate stays fine.
    assert bad_rank.gate_auc > 0.60
    # But ranking collapses.
    assert abs(float(bad_rank.rank_metrics.rank_ic)) < 0.25
    assert abs(float(bad_rank.rank_metrics_gated.rank_ic)) < 0.25


def test_toy_dual_future_perturbation_invariance_before_cutoff() -> None:
    X, y, groups = _make_synthetic_dual_panel(n_dates=260, seed=3)
    uniq_dates = np.sort(np.unique(groups))

    cutoff = uniq_dates[210]

    cv = PurgedWalkForwardCV(
        n_splits=2,
        test_size=20,
        purge_gap=10,
        min_train_size=80,
        embargo=0,
        test_start=uniq_dates[180],
    )

    base = run_toy_dual_cv(X=X, y=y, groups=groups, outer_cv=cv, seed=0)

    X2 = X.copy()
    future_mask = pd.to_datetime(y["Date"]) > pd.Timestamp(cutoff)
    X2.loc[future_mask.to_numpy(), "f_sector"] = X2.loc[future_mask.to_numpy(), "f_sector"] + 1000.0

    pert = run_toy_dual_cv(X=X2, y=y, groups=groups, outer_cv=cv, seed=0)

    a = base.df.loc[pd.to_datetime(base.df["Date"]) <= pd.Timestamp(cutoff)].sort_values(["Date", "Sector"]).reset_index(drop=True)
    b = pert.df.loc[pd.to_datetime(pert.df["Date"]) <= pd.Timestamp(cutoff)].sort_values(["Date", "Sector"]).reset_index(drop=True)

    np.testing.assert_allclose(a["p_gate"].to_numpy(), b["p_gate"].to_numpy(), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(a["rank_score"].to_numpy(), b["rank_score"].to_numpy(), rtol=0.0, atol=1e-12)

    sel_a = _selected_topk(base.df, k=3, gate_threshold=0.5)
    sel_b = _selected_topk(pert.df, k=3, gate_threshold=0.5)

    sa = sel_a.loc[pd.to_datetime(sel_a["Date"]) <= pd.Timestamp(cutoff)].sort_values(["Date", "Sector"]).reset_index(drop=True)
    sb = sel_b.loc[pd.to_datetime(sel_b["Date"]) <= pd.Timestamp(cutoff)].sort_values(["Date", "Sector"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(sa, sb, check_dtype=False)
