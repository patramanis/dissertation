from __future__ import annotations

import json
import re
from pathlib import Path

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


def _normalize_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="raise")
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
        return out
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ms", errors="raise")
    out = pd.to_datetime(s, errors="raise")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def _print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _pick_heavy_feature(feature_cols: list[str]) -> str | None:
    preferred_patterns = [
        r"(?i)beta.*(252|126)",
        r"(?i)idio.*(252|126)",
        r"(?i)vol.*(252|126|63)",
        r"(?i)mdd.*(252|126)",
        r"(?i)trend.*(252|126|63)",
        r"(?i)ratio_z.*(252|126|63)",
    ]
    for pat in preferred_patterns:
        for c in feature_cols:
            if re.search(pat, c):
                return c
    for c in feature_cols:
        if re.search(r"(252|126)", c):
            return c
    return feature_cols[0] if feature_cols else None


def _integer_like(a: np.ndarray) -> bool:
    if a.size == 0:
        return True
    if np.issubdtype(a.dtype, np.integer):
        return True
    return np.nanmax(np.abs(a - np.round(a))) < 1e-9


def _is_multiple_of_step(a: np.ndarray, step: float, tol: float = 1e-9) -> bool:
    if a.size == 0:
        return True
    q = a / step
    return np.nanmax(np.abs(q - np.round(q))) < tol


def check_horizon(h: int) -> None:
    data_dir = Path(__file__).resolve().parents[1]
    dataset_dir = data_dir / "dataset" / f"h{h}"

    X_path = dataset_dir / "X.parquet"
    keys_path = dataset_dir / "keys.parquet"
    y_gate_path = dataset_dir / "y_gate.npy"
    y_rank_path = dataset_dir / "y_rank.npy"
    group_path = dataset_dir / "group_sizes.npy"
    meta_path = dataset_dir / "meta.json"

    labels_path = data_dir / "labels" / f"h{h}.parquet"

    _print_header(f"Dataset Integrity Check — h={h}")

    X = pd.read_parquet(X_path)
    keys = pd.read_parquet(keys_path)
    y_gate = np.load(y_gate_path)
    y_rank = np.load(y_rank_path)
    group_sizes = np.load(group_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    keys = keys.copy()
    keys["Date"] = _normalize_date(keys["Date"]).dt.normalize()
    keys["Sector"] = keys["Sector"].astype("string")

    assert len(X) == len(keys) == len(y_gate) == len(
        y_rank
    ), f"len mismatch: X={len(X)} keys={len(keys)} y_gate={len(y_gate)} y_rank={len(y_rank)}"
    assert int(group_sizes.sum()) == len(
        y_rank
    ), f"sum(group_sizes)={group_sizes.sum()} len(y_rank)={len(y_rank)}"

    print(f"Paths: {dataset_dir}")
    print(
        f"X shape: {X.shape} | y_gate len: {len(y_gate)} | y_rank len: {len(y_rank)} | #dates: {keys['Date'].nunique()}"
    )
    print(
        f"Meta target_col: {meta.get('target_col')} | cost_threshold: {meta.get('cost_threshold')} | horizon: {meta.get('horizon')}"
    )

    dup_mask = keys.duplicated(["Date", "Sector"], keep=False)
    dup_count = int(dup_mask.sum())
    print("\n[Step 1] Primary key uniqueness (Date,Sector)")
    print(f"Duplicate rows: {dup_count}")
    if dup_count:
        print(keys.loc[dup_mask].sort_values(["Date", "Sector"]).head(30).to_string(index=False))

    print("\n[Step 2] Query group consistency (9 sectors per Date)")
    counts = keys.groupby("Date", sort=True).size()
    bad = counts[counts != len(EXPECTED_SECTORS)]
    print(f"Unique dates: {len(counts)}")
    print(f"Bad group dates (count != 9): {len(bad)}")
    if len(bad):
        print(bad.head(25).to_string())

    if len(group_sizes) != len(counts):
        print(f"WARNING: stored group_sizes len={len(group_sizes)} != computed dates={len(counts)}")
    else:
        if not np.array_equal(group_sizes.astype(np.int64), counts.to_numpy(dtype=np.int64)):
            print("WARNING: stored group_sizes do not match recomputed counts")

    print("\n[Step 3] Target horizon check (end-of-sample NaNs)")
    target_col = f"label_excess_{h}d"
    labels = pd.read_parquet(labels_path)
    if "Date" not in labels.columns or "Sector" not in labels.columns or target_col not in labels.columns:
        print(f"ERROR: labels parquet missing required columns: Date/Sector/{target_col}")
    else:
        labels = labels.copy()
        labels["Date"] = _normalize_date(labels["Date"]).dt.normalize()
        labels["Sector"] = labels["Sector"].astype("string")
        labels = labels.sort_values(["Date", "Sector"]).reset_index(drop=True)

        last_dates = labels["Date"].drop_duplicates().sort_values().tail(h).tolist()
        tail = labels[labels["Date"].isin(last_dates)][["Date", "Sector", target_col]]
        non_nan = tail[tail[target_col].notna()]
        print(f"Last {h} trading dates in labels: {len(last_dates)}")
        print(f"Rows in tail block: {len(tail)} (expected {len(last_dates) * len(EXPECTED_SECTORS)})")
        print(f"Non-NaN target rows in last {h} dates: {len(non_nan)}")
        if len(non_nan):
            print("WARNING: found non-NaN targets in the last h trading days (unexpected)")
            print(non_nan.head(25).to_string(index=False))

        non_nan_labels = labels[labels[target_col].notna()]
        if non_nan_labels.empty:
            print("ERROR: labels contain no non-NaN targets")
        else:
            last_label_date = non_nan_labels["Date"].max()
            last_key_date = keys["Date"].max()
            print(f"Last non-NaN label date: {last_label_date}")
            print(f"Last dataset key date: {last_key_date}")
            if last_key_date > last_label_date:
                print("ERROR: dataset contains dates beyond last non-NaN label date (leakage risk)")

    print("\n[Step 4] Warm-up period check (heavy rolling feature NaNs at start)")
    feature_cols = list(X.columns)
    heavy = _pick_heavy_feature(feature_cols)
    print(f"Chosen heavy feature: {heavy}")
    if heavy is None or heavy not in X.columns:
        print("ERROR: could not select heavy feature")
    else:
        tmp = pd.concat([keys[["Date", "Sector"]].reset_index(drop=True), X[[heavy]].reset_index(drop=True)], axis=1)
        tmp = tmp.sort_values(["Date", "Sector"], kind="mergesort")
        first_dates = tmp["Date"].drop_duplicates().sort_values().head(5).tolist()
        head_block = tmp[tmp["Date"].isin(first_dates)]
        nan_rate = float(head_block[heavy].isna().mean())
        first_non_na_date = tmp.loc[tmp[heavy].notna(), "Date"].min()
        print(f"NaN rate of heavy feature in first 5 dates: {nan_rate:.3f}")
        print(f"First non-NaN date for heavy feature: {first_non_na_date}")
        if nan_rate == 0.0:
            print("WARNING: No NaNs found at the very start for the heavy feature (possible aggressive backfill).")

    print("\n[Step 5] Range & Infinity check")
    num = X.select_dtypes(include=["number"])
    arr = num.to_numpy(dtype=float)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    print(f"NaN values in numeric features: {nan_count}")
    print(f"Infinite values in numeric features: {inf_count}")
    if inf_count:
        inf_mask = np.isinf(arr)
        bad_cols = num.columns[np.any(inf_mask, axis=0)].tolist()
        print(f"Columns containing inf: {bad_cols[:30]}")

    rank_cols = [c for c in X.columns if c.startswith("rank_") or c.lower().startswith("rank_")]
    print(f"Rank-like feature columns: {len(rank_cols)}")
    if rank_cols:
        bad_rank = []
        for c in rank_cols:
            v = pd.to_numeric(X[c], errors="coerce").to_numpy()
            v2 = v[~np.isnan(v)]
            if v2.size == 0:
                continue
            mn = float(np.min(v2))
            mx = float(np.max(v2))
            if mn < (1.0 / 9.0 - 1e-12) or mx > (1.0 + 1e-12):
                bad_rank.append((c, f"out-of-range [{mn:.6g},{mx:.6g}] expected in (0,1]"))
                continue
            if not _is_multiple_of_step(v2, step=1.0 / 18.0, tol=1e-8):
                bad_rank.append((c, "unexpected step (expected ~multiples of 1/18 for pct ranks with ties)"))
        print(f"Bad rank columns: {len(bad_rank)}")
        if bad_rank:
            print("Examples:", bad_rank[:20])

    print("\n[Step 6] Date continuity / gap detection")
    dates = keys["Date"].drop_duplicates().sort_values().to_numpy()
    if dates.size < 2:
        print("Not enough dates to check gaps")
    else:
        diffs = np.diff(dates).astype("timedelta64[D]").astype(int)
        max_gap = int(diffs.max())
        print(f"Max calendar-day gap between consecutive trading dates: {max_gap} days")
        big = np.where(diffs >= 10)[0]
        print(f"#gaps >= 10 days: {int(big.size)}")
        if big.size:
            idx = np.argsort(diffs)[-10:][::-1]
            rows = []
            for i in idx:
                rows.append((pd.Timestamp(dates[i]).date(), pd.Timestamp(dates[i + 1]).date(), int(diffs[i])))
            print("Top gaps (date_from -> date_to, days):")
            for a, b, g in rows:
                print(f"  {a} -> {b}: {g}")


def main() -> None:
    for h in HORIZONS:
        check_horizon(h)


if __name__ == "__main__":
    main()
