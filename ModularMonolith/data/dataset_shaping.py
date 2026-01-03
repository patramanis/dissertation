from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_SECTORS: tuple[str, ...] = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLY",
    "XLV",
    "XLU",
)

HORIZONS: tuple[int, ...] = (5, 21, 63)


DATASET_OUT_DIR: Path = Path(__file__).resolve().parent / "dataset"


@dataclass(frozen=True)
class ShapingResult:
    horizon: int
    target_col: str
    features_path: Path
    labels_path: Path
    full_df: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    group_sizes: np.ndarray
    feature_cols: list[str]
    dropped_rows_nan_target: int
    dropped_dates_nan_target: int
    dropped_dates_universe_policy: int


def _normalize_date_column(df: pd.DataFrame, *, col: str = "Date", name: str = "df") -> None:
    if col not in df.columns:
        raise ValueError(f"[{name}] Missing required column: {col}")

    s = df[col]

    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, errors="raise")
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_localize(None)
        df[col] = dt
        return

    if pd.api.types.is_numeric_dtype(s):
        df[col] = pd.to_datetime(s, unit="ms", errors="raise")
        return

    df[col] = pd.to_datetime(s, errors="raise")
    dt = df[col]
    if getattr(dt.dt, "tz", None) is not None:
        df[col] = dt.dt.tz_localize(None)


def _normalize_sector_column(df: pd.DataFrame, *, col: str = "Sector", name: str = "df") -> None:
    if col not in df.columns:
        raise ValueError(f"[{name}] Missing required column: {col}")

    df[col] = df[col].astype("string")
    df[col] = pd.Categorical(df[col], categories=list(EXPECTED_SECTORS), ordered=True)


def _assert_required_columns(df: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")


def _report_duplicates(df: pd.DataFrame, key_cols: list[str], *, name: str, max_rows: int = 40) -> None:
    dups = df[df.duplicated(subset=key_cols, keep=False)].copy()
    if dups.empty:
        return

    dups = dups.sort_values(key_cols)
    counts = dups.groupby(key_cols, dropna=False).size().reset_index(name="n")

    msg_lines = [
        f"[{name}] Duplicate keys detected for {key_cols}: {len(counts)} unique duplicated keys, {len(dups)} rows total.",
        "Showing first occurrences:",
        counts.head(max_rows).to_string(index=False),
    ]
    raise ValueError("\n".join(msg_lines))


def _ensure_unique_key(df: pd.DataFrame, key_cols: list[str], *, name: str) -> None:
    _report_duplicates(df, key_cols, name=name)


def _default_paths(horizon: int) -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = here
    features_path = data_dir / "processed_data_1" / f"features_h{horizon}.parquet"
    labels_path = data_dir / "labels" / f"h{horizon}.parquet"
    return features_path, labels_path


def _build_rank_target(
    df: pd.DataFrame,
    *,
    target_col: str,
    n_expected: int,
) -> pd.Series:
    tie = pd.util.hash_pandas_object(df[["Date", "Sector"]], index=False)

    tmp = df[["Date", "Sector", target_col]].copy()
    tmp["_tie"] = tie.to_numpy()

    tmp = tmp.sort_values(["Date", target_col, "_tie"], ascending=[True, False, True], kind="mergesort")
    tmp["_pos"] = tmp.groupby("Date", sort=False).cumcount()

    tmp["rank_target"] = (n_expected - 1) - tmp["_pos"]
    out = tmp.set_index(["Date", "Sector"])["rank_target"]

    if out.isna().any():
        raise RuntimeError("rank_target construction produced NaNs")

    if (out < 0).any() or (out > (n_expected - 1)).any():
        raise RuntimeError("rank_target outside expected range")

    return out.astype(np.int16)


def _select_feature_columns(df: pd.DataFrame, *, target_col: str) -> list[str]:
    non_feature = {"Date", "Sector", "rank_target"}

    for c in df.columns:
        if c.startswith("label_"):
            non_feature.add(c)

    non_feature.add(target_col)

    forbidden = (
        "label",
        "target",
        "forward",
        "fwd",
        "excess_5d",
        "excess_21d",
        "excess_63d",
    )

    feature_cols: list[str] = []
    bad: list[str] = []
    for c in df.columns:
        if c in non_feature:
            continue
        lc = c.lower()
        if any(tok in lc for tok in forbidden):
            bad.append(c)
            continue
        feature_cols.append(c)

    if bad:
        raise ValueError(f"Leakage-like columns would enter X: {sorted(set(bad))}")

    if not feature_cols:
        raise ValueError("No feature columns selected")

    return feature_cols


def load_ranker_dataset(
    horizon: int,
    *,
    enforce_full_universe: bool = True,
    verbose: bool = True,
    features_path: Path | None = None,
    labels_path: Path | None = None,
) -> ShapingResult:
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon}; expected one of {HORIZONS}")

    target_col = f"label_excess_{horizon}d"

    if features_path is None or labels_path is None:
        d_features, d_labels = _default_paths(horizon)
        features_path = features_path or d_features
        labels_path = labels_path or d_labels

    if verbose:
        print(f"\n[load] horizon={horizon} target_col={target_col}")
        print(f"[load] features_path={features_path}")
        print(f"[load] labels_path={labels_path}")

    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    drop_from_features = [c for c in features.columns if isinstance(c, str) and c.startswith("label_")]
    if drop_from_features:
        features = features.drop(columns=drop_from_features)

    _assert_required_columns(features, ["Date", "Sector"], name="features")
    _assert_required_columns(labels, ["Date", "Sector", target_col], name="labels")

    _normalize_date_column(features, name="features")
    _normalize_date_column(labels, name="labels")

    _normalize_sector_column(features, name="features")
    _normalize_sector_column(labels, name="labels")

    _ensure_unique_key(features, ["Date", "Sector"], name="features")
    _ensure_unique_key(labels, ["Date", "Sector"], name="labels")

    merged = features.merge(labels, on=["Date", "Sector"], how="inner")

    _ensure_unique_key(merged, ["Date", "Sector"], name="merged")

    if target_col not in merged.columns:
        raise RuntimeError(f"target_col missing after merge: {target_col}")
    if merged[target_col].isna().all():
        raise RuntimeError(f"target_col is all-NaN after merge: {target_col}")

    n0 = len(merged)
    dates0 = merged["Date"].nunique()

    merged = merged.dropna(subset=[target_col])

    n1 = len(merged)
    dates1 = merged["Date"].nunique()
    dropped_rows_nan_target = n0 - n1
    dropped_dates_nan_target = dates0 - dates1

    if verbose:
        print(f"[drop] NaN target rows dropped: {dropped_rows_nan_target}")
        print(f"[drop] Dates lost fully due to NaN target drop: {dropped_dates_nan_target}")

    group_sizes_pre = merged.groupby("Date", sort=False).size()
    if verbose:
        dist = group_sizes_pre.value_counts().sort_index()
        print("[groups] Group size distribution (pre-universe-policy):")
        print(dist.to_string())

    dropped_dates_universe_policy = 0

    if enforce_full_universe:
        keep_dates = group_sizes_pre[group_sizes_pre == len(EXPECTED_SECTORS)].index
        merged2 = merged[merged["Date"].isin(keep_dates)].copy()

        bad_dates: list[pd.Timestamp] = []
        for dt, g in merged2.groupby("Date", sort=False):
            got = set(g["Sector"].astype("string").tolist())
            exp = set(EXPECTED_SECTORS)
            if got != exp:
                bad_dates.append(dt)
        if bad_dates:
            sample = bad_dates[:10]
            raise ValueError(
                "Universe policy violated: Dates with 9 rows but wrong sector set. "
                f"Count={len(bad_dates)} sample={sample}"
            )

        dropped_dates_universe_policy = merged["Date"].nunique() - merged2["Date"].nunique()
        merged = merged2

        if verbose:
            print(f"[groups] Dates dropped by 9-sector policy: {dropped_dates_universe_policy}")

    merged = merged.sort_values(["Date", "Sector"], kind="mergesort").reset_index(drop=True)

    rt = _build_rank_target(merged, target_col=target_col, n_expected=len(EXPECTED_SECTORS))
    merged = merged.merge(rt.rename("rank_target"), on=["Date", "Sector"], how="left", validate="one_to_one")

    if merged["rank_target"].isna().any():
        raise RuntimeError("rank_target contains NaNs after merge")

    feature_cols = _select_feature_columns(merged, target_col=target_col)
    X = merged[feature_cols].copy()
    y = merged["rank_target"].to_numpy(dtype=np.int16)

    group_sizes = merged.groupby("Date", sort=False).size().to_numpy(dtype=np.int32)

    if group_sizes.sum() != len(y) or len(y) != len(X):
        raise RuntimeError("Alignment assertion failed: sum(group_sizes) != len(y) != len(X)")

    if enforce_full_universe and not np.all(group_sizes == len(EXPECTED_SECTORS)):
        raise RuntimeError("Universe policy expected all group_sizes == 9")

    return ShapingResult(
        horizon=horizon,
        target_col=target_col,
        features_path=features_path,
        labels_path=labels_path,
        full_df=merged,
        X=X,
        y=y,
        group_sizes=group_sizes,
        feature_cols=feature_cols,
        dropped_rows_nan_target=int(dropped_rows_nan_target),
        dropped_dates_nan_target=int(dropped_dates_nan_target),
        dropped_dates_universe_policy=int(dropped_dates_universe_policy),
    )


def _sanity_report(res: ShapingResult, *, show_first_dates: int = 3) -> None:
    df = res.full_df

    print("\nFINAL SANITY CHECK !")
    print(f"horizon={res.horizon} target_col={res.target_col}")

    is_unique = not df.duplicated(["Date", "Sector"]).any()
    print(f"Key integrity: unique(Date,Sector)={is_unique}")

    print(f"X shape={res.X.shape} y len={len(res.y)}")

    n_dates = df["Date"].nunique()
    print(f"group_sizes len={len(res.group_sizes)} unique_dates={n_dates}")
    print(f"sum(group_sizes)={int(res.group_sizes.sum())}")
    print(f"min(gs)={int(res.group_sizes.min())} max(gs)={int(res.group_sizes.max())}")

    dates = df["Date"].drop_duplicates().iloc[:show_first_dates].tolist()
    for dt in dates:
        g = df[df["Date"] == dt][["Sector", res.target_col, "rank_target"]].copy()
        g = g.sort_values("rank_target", ascending=False, kind="mergesort")
        best = g.iloc[0]
        worst = g.iloc[-1]
        print(f"\nDate={dt.date()} (sorted by rank_target desc)")
        print(g.to_string(index=False))
        print(
            f"Check: best target={best[res.target_col]:.6g} rank={int(best['rank_target'])}; "
            f"worst target={worst[res.target_col]:.6g} rank={int(worst['rank_target'])}"
        )

    forbidden_substrings = ("label", "target", "forward", "fwd", "excess_5d", "excess_21d", "excess_63d")
    bad = [c for c in res.feature_cols if any(tok in c.lower() for tok in forbidden_substrings)]
    print(f"No-leakage columns: bad_in_X={len(bad)}")
    if bad:
        print("Bad columns:", bad)
        raise RuntimeError("Leakage-like columns found in X")


def _save_shaping_result(res: ShapingResult, *, out_root: Path = DATASET_OUT_DIR, verify_reload: bool = True) -> None:
    out_dir = out_root / f"h{res.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist keys separately to make post-load alignment checks trivial.
    keys = res.full_df[["Date", "Sector"]].copy()
    keys["Sector"] = keys["Sector"].astype("string")

    X_path = out_dir / "X.parquet"
    y_path = out_dir / "y.npy"
    group_path = out_dir / "group_sizes.npy"
    keys_path = out_dir / "keys.parquet"
    meta_path = out_dir / "meta.json"

    res.X.to_parquet(X_path, index=False, engine="pyarrow")
    np.save(y_path, res.y)
    np.save(group_path, res.group_sizes)
    keys.to_parquet(keys_path, index=False, engine="pyarrow")

    meta = {
        "horizon": int(res.horizon),
        "target_col": res.target_col,
        "n_rows": int(len(res.y)),
        "n_dates": int(res.full_df["Date"].nunique()),
        "n_features": int(res.X.shape[1]),
        "expected_sectors": list(EXPECTED_SECTORS),
        "dropped_rows_nan_target": int(res.dropped_rows_nan_target),
        "dropped_dates_nan_target": int(res.dropped_dates_nan_target),
        "dropped_dates_universe_policy": int(res.dropped_dates_universe_policy),
        "feature_cols": list(res.feature_cols),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if verify_reload:
        X2 = pd.read_parquet(X_path)
        y2 = np.load(y_path)
        gs2 = np.load(group_path)
        keys2 = pd.read_parquet(keys_path)

        if len(X2) != len(y2) or len(y2) != len(keys2):
            raise RuntimeError(
                f"Reloaded lengths mismatch for h={res.horizon}: len(X)={len(X2)} len(y)={len(y2)} len(keys)={len(keys2)}"
            )
        if gs2.sum() != len(y2):
            raise RuntimeError(
                f"Reloaded group_sizes mismatch for h={res.horizon}: sum(gs)={int(gs2.sum())} len(y)={int(len(y2))}"
            )


def save_all_results(results: dict[int, ShapingResult], *, out_root: Path = DATASET_OUT_DIR) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for h, res in results.items():
        _save_shaping_result(res, out_root=out_root, verify_reload=True)
    print(f"\n[saved] Wrote datasets to: {out_root}")


def run_all_horizons(*, enforce_full_universe: bool = True, save_outputs: bool = True) -> dict[int, ShapingResult]:
    out: dict[int, ShapingResult] = {}

    for h in HORIZONS:
        res = load_ranker_dataset(h, enforce_full_universe=enforce_full_universe, verbose=True)
        _sanity_report(res, show_first_dates=3)
        out[h] = res

    print("\n--- HORIZON CONSISTENCY ---")
    for h, res in out.items():
        print(
            f"h={h}: target_col={res.target_col} "
            f"dropped_rows_nan_target={res.dropped_rows_nan_target} "
            f"dropped_dates_nan_target={res.dropped_dates_nan_target} "
            f"dropped_dates_universe_policy={res.dropped_dates_universe_policy}"
        )

    if save_outputs:
        save_all_results(out, out_root=DATASET_OUT_DIR)

    return out


if __name__ == "__main__":
    run_all_horizons(enforce_full_universe=True, save_outputs=True)
