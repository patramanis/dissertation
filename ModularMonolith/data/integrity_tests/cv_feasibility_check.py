from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ModularMonolith.src.config.cv_config import get_cv_config
from ModularMonolith.src.cv.purged_walk_forward_cv import PurgedWalkForwardCV


def _load_dataset_dir(h: int) -> Path:
    here = Path(__file__).resolve().parents[1]
    ds = here / "dataset" / f"h{int(h)}"
    if not ds.exists():
        raise FileNotFoundError(f"Missing dataset dir: {ds}")
    return ds


def _load_meta(ds: Path) -> dict:
    p = ds / "meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _groups_from_keys(keys: pd.DataFrame) -> np.ndarray:
    if "Date" not in keys.columns:
        raise ValueError("keys.parquet must contain Date")
    dt = pd.to_datetime(keys["Date"], errors="raise").dt.tz_localize(None)
    return dt.to_numpy()


def _ranker_fold_gating_summary(
    *,
    y: pd.DataFrame,
    keys: pd.DataFrame,
    cost: float,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    min_rank_train_groups: int,
    min_rank_train_winners: int,
    min_rank_train_winner_dates: int,
) -> tuple[bool, str]:
    y_tr = y.iloc[outer_train_idx].copy()
    y_te = y.iloc[outer_test_idx].copy()
    k_tr = keys.iloc[outer_train_idx].copy()
    k_te = keys.iloc[outer_test_idx].copy()

    if "label_excess" not in y_tr.columns:
        return False, "missing_label_excess"

    tr_mask = pd.to_numeric(y_tr["label_excess"], errors="coerce").notna().to_numpy()
    te_mask = pd.to_numeric(y_te["label_excess"], errors="coerce").notna().to_numpy()

    y_tr = y_tr.iloc[tr_mask]
    k_tr = k_tr.iloc[tr_mask]
    y_te = y_te.iloc[te_mask]
    k_te = k_te.iloc[te_mask]

    if y_tr.empty or y_te.empty:
        return False, "empty_after_valid_mask"

    g_tr = pd.to_datetime(k_tr["Date"], errors="raise").dt.tz_localize(None)
    n_train_groups = int(g_tr.nunique())
    if n_train_groups < int(min_rank_train_groups):
        return False, f"train_groups<{min_rank_train_groups}"

    winners = pd.to_numeric(y_tr["label_excess"], errors="coerce") > float(cost)
    n_winners = int(winners.sum())
    if n_winners < int(min_rank_train_winners):
        return False, f"winners<{min_rank_train_winners}"

    win_dates = int(g_tr.loc[winners].nunique())
    if win_dates < int(min_rank_train_winner_dates):
        return False, f"winner_dates<{min_rank_train_winner_dates}"

    return True, "ok"


def main() -> None:
    horizons = (5, 21, 63)

    for h in horizons:
        ds = _load_dataset_dir(h)
        meta = _load_meta(ds)

        keys = pd.read_parquet(ds / "keys.parquet", engine="pyarrow")
        y = pd.read_parquet(ds / "y.parquet", engine="pyarrow")
        groups = _groups_from_keys(keys)
        n_groups = int(pd.Series(groups).nunique())

        cfg = get_cv_config(h)
        cv = PurgedWalkForwardCV(**cfg)

        dummy_X = np.zeros((len(groups), 1), dtype=float)

        print(f"\n[h{h}] samples={len(groups)} unique_dates={n_groups} cv={cfg}")

        fold_infos = list(cv.split_with_info(dummy_X, None, groups=groups))
        print(f"[h{h}] folds={len(fold_infos)}")

        cost_threshold = float(meta.get("cost_threshold", 0.0))
        cost_bps = int(round(cost_threshold * 10_000))
        cost = float(cost_bps) / 10_000.0

        skip_reasons: dict[str, int] = {}
        ok_folds = 0
        for fold_id, (tr, te, info) in enumerate(fold_infos):
            ok, reason = _ranker_fold_gating_summary(
                y=y,
                keys=keys,
                cost=cost,
                outer_train_idx=np.asarray(tr, dtype=int),
                outer_test_idx=np.asarray(te, dtype=int),
                min_rank_train_groups=10,
                min_rank_train_winners=25,
                min_rank_train_winner_dates=5,
            )
            if ok:
                ok_folds += 1
            else:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

            if fold_id < 3:
                print(
                    f"[h{h}] fold={fold_id} train=[{pd.Timestamp(info.train_start).date()},{pd.Timestamp(info.train_end).date()}] "
                    f"test=[{pd.Timestamp(info.test_start).date()},{pd.Timestamp(info.test_end).date()}] gate={reason}"
                )

        print(f"[h{h}] nested_ranker_gate_ok_folds={ok_folds}/{len(fold_infos)} cost_bps={cost_bps}")
        if skip_reasons:
            print(f"[h{h}] skip_reasons={skip_reasons}")


if __name__ == "__main__":
    main()
