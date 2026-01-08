from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ModularMonolith.data.train_window import TRAIN_DATE_END, TRAIN_DATE_START

try:
    from ModularMonolith import build_id

    BUILD_ID = build_id(__file__)
except Exception:
    BUILD_ID = str(Path(__file__).resolve())


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


FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "label",
    "target",
    "forward",
    "fwd",
    "excess_5d",
    "excess_21d",
    "excess_63d",
)


COST_THRESHOLD: dict[int, float] = {
    5: 0.001,
    21: 0.001,
    63: 0.001,
}


def _resolve_label_excess_column(labels: pd.DataFrame, *, horizon: int) -> str:
    legacy = f"label_excess_{int(horizon)}d"
    if "label_excess" in labels.columns:
        return "label_excess"
    if legacy in labels.columns:
        labels["label_excess"] = labels[legacy]
        return "label_excess"
    raise ValueError(
        "Labels parquet is missing the required canonical target column 'label_excess'. "
        f"Columns={sorted(map(str, labels.columns))}"
    )


def _maybe_validate_labels_horizon(labels: pd.DataFrame, *, horizon: int, labels_path: Path) -> None:
    for col in ("horizon", "horizon_days"):
        if col not in labels.columns:
            continue
        h_unique = pd.to_numeric(labels[col], errors="coerce").dropna().unique()
        if h_unique.size == 0:
            continue
        if h_unique.size != 1 or int(h_unique[0]) != int(horizon):
            raise ValueError(
                f"Labels horizon mismatch for {labels_path}: expected horizon={int(horizon)}; got {col} unique={h_unique.tolist()}"
            )


def _rel_rank_per_date_from_excess(excess: pd.Series, *, cost: float) -> pd.Series:

    mask = excess.notna()
    if int(mask.sum()) < 2:
        return pd.Series(pd.NA, index=excess.index, dtype="Int8")

    valid = pd.to_numeric(excess[mask], errors="coerce").dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=excess.index, dtype="Int8")

    q60 = float(valid.quantile(0.60))
    q80 = float(valid.quantile(0.80))

    rel = pd.Series(pd.NA, index=excess.index, dtype="Int8")
    rel.loc[mask & (excess <= cost)] = 0
    rel.loc[mask & (excess > cost) & (excess <= q60)] = 1
    rel.loc[mask & (excess > q60) & (excess <= q80)] = 2
    rel.loc[mask & (excess > q80)] = 3
    return rel


DATASET_OUT_DIR: Path = Path(__file__).resolve().parent / "dataset"


RAW3_SKIP_DATASETS: tuple[str, ...] = (
    "SPDR",
    "SPY",
)


RAW3_ALLOWED_SUFFIXES: tuple[str, ...] = (
    "_diff",
    "_logret",
    "_dlog1p",
    "_log1p",
    "_asinh",
    "_mom",
    "_yoy",
)


@dataclass(frozen=True)
class ShapingResult:
    horizon: int
    target_col: str
    features_path: Path
    labels_path: Path
    correlations_path: Path
    full_df: pd.DataFrame
    X: pd.DataFrame
    y_gate: np.ndarray
    y_rank: np.ndarray
    group_sizes: np.ndarray
    feature_cols: list[str]
    dropped_rows_nan_target: int
    dropped_dates_nan_target: int
    dropped_dates_universe_policy: int
    cost_threshold: float


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


def _paths_for_fusion(horizon: int) -> tuple[Path, Path, Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = here
    proc1_dir = data_dir / "processed_data_1"
    proc2_dir = data_dir / "processed_data_2"
    raw3_dir = data_dir / "raw_data_3"

    features_path = proc1_dir / f"features_h{horizon}.parquet"
    labels_path = data_dir / "labels" / f"h{horizon}.parquet"
    correlations_path = proc2_dir / "features_correlations.parquet"
    return features_path, labels_path, correlations_path, raw3_dir


def _load_correlations_wide(correlations_path: Path) -> pd.DataFrame:
    if not correlations_path.exists():
        raise FileNotFoundError(f"Missing correlations parquet: {correlations_path}")

    corr_long = pd.read_parquet(correlations_path)
    _assert_required_columns(corr_long, ["Date", "Sector", "Feature", "value"], name="correlations_long")
    _normalize_date_column(corr_long, name="correlations_long")
    _normalize_sector_column(corr_long, name="correlations_long")

    _ensure_unique_key(corr_long, ["Date", "Sector", "Feature"], name="correlations_long")

    wide = corr_long.pivot(index=["Date", "Sector"], columns="Feature", values="value")
    wide = wide.reset_index()
    wide.columns = [str(c) for c in wide.columns]
    _ensure_unique_key(wide, ["Date", "Sector"], name="correlations_wide")
    return wide


def _load_raw3_macro_broadcast(raw3_dir: Path) -> pd.DataFrame:
    if not raw3_dir.exists():
        raise FileNotFoundError(f"Missing raw_data_3 directory: {raw3_dir}")

    files = sorted(raw3_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in raw_data_3: {raw3_dir}")

    macro: pd.DataFrame | None = None
    seen_cols: set[str] = set()

    def is_allowed_transformed_column(col: str) -> bool:
        c = str(col)
        return any(c.endswith(suf) for suf in RAW3_ALLOWED_SUFFIXES)

    for p in files:
        if p.stem in set(RAW3_SKIP_DATASETS):
            continue
        df = pd.read_parquet(p)
        _assert_required_columns(df, ["Date"], name=f"raw3:{p.name}")
        _normalize_date_column(df, name=f"raw3:{p.name}")

        df = df.sort_values("Date").drop_duplicates("Date", keep="last")
        cols_all = [c for c in df.columns if c != "Date"]
        cols = [c for c in cols_all if is_allowed_transformed_column(str(c))]
        if not cols:
            continue

        suspicious = [c for c in cols if any(tok in str(c).lower() for tok in FORBIDDEN_SUBSTRINGS)]
        if suspicious:
            raise ValueError(
                "Forbidden/leakage-like column names detected in raw_data_3 input. "
                f"File={p.name} columns={sorted(set(suspicious))}"
            )

        overlap = [c for c in cols if c in seen_cols]
        if overlap:
            raise ValueError(
                "Duplicate macro column names across raw_data_3 files. "
                f"File={p.name} duplicates={overlap}. "
                "Rename upstream or implement a prefixing scheme."
            )
        seen_cols.update(cols)

        df = df[["Date", *cols]].copy()
        macro = df if macro is None else macro.merge(df, on="Date", how="outer")

    if macro is None:
        raise RuntimeError(f"raw_data_3 had no usable columns (only Date): {raw3_dir}")

    macro = macro.sort_values("Date")
    _ensure_unique_key(macro, ["Date"], name="raw3_macro")
    return macro


def _load_regimes_features(data_dir: Path) -> pd.DataFrame | None:
    regimes_path = data_dir / "regimes" / "spy_regimes.parquet"
    if not regimes_path.exists():
        return None

    rg = pd.read_parquet(regimes_path)
    if "Date" in rg.columns:
        rg = rg[["Date", *[c for c in ("hmm_raw", "hmm_low_vol", "hmm_trend", "hmm_delta") if c in rg.columns]]].copy()
        _normalize_date_column(rg, name="regimes")
        rg = rg.sort_values("Date").drop_duplicates("Date", keep="last")
    else:
        idx = pd.to_datetime(rg.index, errors="raise")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        keep = [c for c in ("hmm_raw", "hmm_low_vol", "hmm_trend", "hmm_delta") if c in rg.columns]
        if not keep:
            return None
        rg = rg[keep].copy()
        rg.index = pd.DatetimeIndex(idx, name="Date")
        rg = rg.sort_index().reset_index()

    keep_cols = [c for c in ("Date", "hmm_raw", "hmm_low_vol", "hmm_trend", "hmm_delta") if c in rg.columns]
    if keep_cols == ["Date"]:
        return None
    out = rg[keep_cols].copy()
    _ensure_unique_key(out, ["Date"], name="regimes")
    return out


def _assert_no_merge_suffix_columns(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if isinstance(c, str) and (c.endswith("_x") or c.endswith("_y"))]
    if bad:
        raise ValueError(
            f"[{name}] Found merge-suffix duplicate columns (likely double-merged features): {sorted(set(bad))}"
        )


def _filter_train_window(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError(f"[{name}] Missing Date column for train-window filter")
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="raise")
    if getattr(out["Date"].dt, "tz", None) is not None:
        out["Date"] = out["Date"].dt.tz_localize(None)
    m = (out["Date"] >= TRAIN_DATE_START) & (out["Date"] <= TRAIN_DATE_END)
    return out.loc[m].copy()


def _assert_same_date_set(a: pd.DataFrame, b: pd.DataFrame, *, name_a: str, name_b: str) -> None:
    da = pd.DatetimeIndex(pd.to_datetime(a["Date"], errors="raise").dt.tz_localize(None).unique()).sort_values()
    db = pd.DatetimeIndex(pd.to_datetime(b["Date"], errors="raise").dt.tz_localize(None).unique()).sort_values()
    if da.equals(db):
        return

    sa = set(da.tolist())
    sb = set(db.tolist())
    only_a = sorted(sa - sb)
    only_b = sorted(sb - sa)
    raise AssertionError(
        "Calendar mismatch between inputs after canonical 2005–2025 cut. "
        f"{name_a}_dates={len(da)} {name_b}_dates={len(db)} "
        f"{name_a}_only={len(only_a)} {name_b}_only={len(only_b)} "
        f"{name_a}_range=[{(da.min().date() if len(da) else None)},{(da.max().date() if len(da) else None)}] "
        f"{name_b}_range=[{(db.min().date() if len(db) else None)},{(db.max().date() if len(db) else None)}] "
        f"sample_{name_a}_only={[d.date() for d in only_a[:5]]} sample_{name_b}_only={[d.date() for d in only_b[:5]]}"
    )


def _assert_same_panel_keys(a: pd.DataFrame, b: pd.DataFrame, *, name_a: str, name_b: str) -> None:
    mi_a = pd.MultiIndex.from_frame(a[["Date", "Sector"]])
    mi_b = pd.MultiIndex.from_frame(b[["Date", "Sector"]])
    if mi_a.equals(mi_b):
        return

    sa = set(mi_a.tolist())
    sb = set(mi_b.tolist())
    only_a = list(sa - sb)
    only_b = list(sb - sa)
    only_a.sort()
    only_b.sort()
    fmt = lambda pairs: [(pd.Timestamp(d).date(), str(s)) for d, s in pairs]
    raise AssertionError(
        "Panel key mismatch between inputs (Date,Sector) after canonical cut. "
        f"{name_a}_keys={len(mi_a)} {name_b}_keys={len(mi_b)} "
        f"{name_a}_only={len(only_a)} {name_b}_only={len(only_b)} "
        f"sample_{name_a}_only={fmt(only_a[:8])} sample_{name_b}_only={fmt(only_b[:8])}"
    )


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
    non_feature = {"Date", "Sector", "rank_target", "y_gate", "rel_rank", "cost_bps", "horizon"}

    for c in df.columns:
        if c.startswith("label_"):
            non_feature.add(c)

    non_feature.add(target_col)

    forbidden = FORBIDDEN_SUBSTRINGS

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


def load_dual_model_dataset(
    horizon: int,
    *,
    enforce_full_universe: bool = True,
    require_regimes: bool = True,
    verbose: bool = True,
    features_path: Path | None = None,
    labels_path: Path | None = None,
    correlations_path: Path | None = None,
    raw3_dir: Path | None = None,
) -> ShapingResult:
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon}; expected one of {HORIZONS}")

    target_col = "label_excess"

    if horizon not in COST_THRESHOLD:
        raise ValueError(f"Missing COST_THRESHOLD for horizon={horizon}; have keys={sorted(COST_THRESHOLD)}")
    default_cost_threshold = float(COST_THRESHOLD[horizon])
    cost_threshold = float(default_cost_threshold)

    if features_path is None or labels_path is None or correlations_path is None or raw3_dir is None:
        d_features, d_labels, d_corr, d_raw3 = _paths_for_fusion(horizon)
        features_path = features_path or d_features
        labels_path = labels_path or d_labels
        correlations_path = correlations_path or d_corr
        raw3_dir = raw3_dir or d_raw3

    if verbose:
        print(f"\n[load] horizon={horizon} target_col={target_col}")
        print(f"[load] features_path={features_path}")
        print(f"[load] labels_path={labels_path}")
        print(f"[load] correlations_path={correlations_path}")
        print(f"[load] raw3_dir={raw3_dir}")
        print(f"[gate] cost_threshold={cost_threshold}")

    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    labels = pd.read_parquet(labels_path)
    features_p1 = pd.read_parquet(features_path)

    _maybe_validate_labels_horizon(labels, horizon=horizon, labels_path=labels_path)
    label_excess_col = _resolve_label_excess_column(labels, horizon=horizon)
    target_col = label_excess_col

    if "cost_bps" in labels.columns:
        cb = pd.to_numeric(labels["cost_bps"], errors="coerce").dropna().unique()
        if cb.size == 1:
            cost_bps = float(cb[0])
            derived = float(cost_bps) / 10_000.0
            if not np.isfinite(derived) or derived < 0:
                raise ValueError(f"Invalid cost_bps in labels: {cost_bps}")
            if not np.isclose(derived, default_cost_threshold, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "Cost inconsistency: labels cost_bps disagrees with COST_THRESHOLD. "
                    f"h={horizon} labels_cost_bps={cost_bps} -> {derived} vs COST_THRESHOLD={default_cost_threshold}"
                )
            cost_threshold = float(derived)
        elif cb.size > 1:
            raise ValueError(
                f"Labels contain multiple cost_bps values for h={horizon}: {cb.tolist()}. Expected a single constant."
            )

    keep_label_cols = ["Date", "Sector", label_excess_col]
    for extra in ("y_gate", "rel_rank", "cost_bps", "horizon"):
        if extra in labels.columns:
            keep_label_cols.append(extra)

    labels = labels[keep_label_cols].copy()
    if label_excess_col != "label_excess":
        labels = labels.rename(columns={label_excess_col: "label_excess"})
        target_col = "label_excess"

    correlations_wide = _load_correlations_wide(correlations_path)
    macro = _load_raw3_macro_broadcast(raw3_dir)

    regimes_dir = Path(__file__).resolve().parent
    regimes_path = regimes_dir / "regimes" / "spy_regimes.parquet"
    regimes = _load_regimes_features(regimes_dir)

    if require_regimes and (regimes is None or regimes.empty):
        raise FileNotFoundError(
            "Required HMM regimes parquet is missing or empty. "
            f"Expected: {regimes_path}"
        )

    if regimes is not None and not regimes.empty:
        req_cols = ["hmm_raw", "hmm_trend", "hmm_delta"]
        missing = [c for c in req_cols if c not in regimes.columns]
        if require_regimes and missing:
            raise ValueError(
                "HMM regimes parquet is missing required columns. "
                f"Missing={missing} columns={sorted(map(str, regimes.columns))} path={regimes_path}"
            )
        macro = macro.merge(regimes, on="Date", how="outer")

    drop_from_p1 = [c for c in features_p1.columns if isinstance(c, str) and c.startswith("label_")]
    if drop_from_p1:
        features_p1 = features_p1.drop(columns=drop_from_p1)

    _assert_required_columns(features_p1, ["Date", "Sector"], name="features_p1")
    _assert_required_columns(labels, ["Date", "Sector", target_col], name="labels")

    _normalize_date_column(features_p1, name="features_p1")
    _normalize_date_column(labels, name="labels")

    _normalize_sector_column(features_p1, name="features_p1")
    _normalize_sector_column(labels, name="labels")

    _ensure_unique_key(features_p1, ["Date", "Sector"], name="features_p1")
    _ensure_unique_key(labels, ["Date", "Sector"], name="labels")

    labels = _filter_train_window(labels, name="labels")
    features_p1 = _filter_train_window(features_p1, name="features_p1")
    correlations_wide = _filter_train_window(correlations_wide, name="correlations_wide")
    macro = _filter_train_window(macro, name="raw3_macro")

    _assert_same_date_set(labels, features_p1, name_a="labels", name_b="features_p1")
    _assert_same_panel_keys(labels, features_p1, name_a="labels", name_b="features_p1")

    merged = labels.copy()

    merged = merged.merge(features_p1, on=["Date", "Sector"], how="inner")

    _normalize_date_column(correlations_wide, name="correlations_wide")
    _normalize_sector_column(correlations_wide, name="correlations_wide")
    _ensure_unique_key(correlations_wide, ["Date", "Sector"], name="correlations_wide")
    merged = merged.merge(correlations_wide, on=["Date", "Sector"], how="left")

    _normalize_date_column(macro, name="raw3_macro")
    _ensure_unique_key(macro, ["Date"], name="raw3_macro")
    merged = merged.merge(macro, on=["Date"], how="left")

    if require_regimes:
        req_cols = ["hmm_raw", "hmm_trend", "hmm_delta"]
        missing = [c for c in req_cols if c not in merged.columns]
        if missing:
            raise RuntimeError(
                "HMM regimes columns are missing after merge; expected them to be present in merged dataset. "
                f"Missing={missing}"
            )
        nan_any = merged[req_cols].isna().any(axis=1)
        if bool(nan_any.any()):
            bad_dates = (
                pd.to_datetime(merged.loc[nan_any, "Date"], errors="raise")
                .dt.tz_localize(None)
                .drop_duplicates()
                .sort_values()
            )
            sample = [d.date() for d in bad_dates.iloc[:10].tolist()]
            raise ValueError(
                "HMM regimes contain NaNs after merge into dataset. "
                "Regimes must be fully populated over the training window to avoid silent feature holes. "
                f"bad_dates={len(bad_dates)} sample={sample}"
            )

    _assert_no_merge_suffix_columns(merged, name="merged")

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

    if "y_gate" not in merged.columns:
        merged["y_gate"] = (merged[target_col] > cost_threshold).astype(np.int8)
    else:
        merged["y_gate"] = pd.to_numeric(merged["y_gate"], errors="coerce").astype("Int64").astype(np.int8)

    if "rel_rank" not in merged.columns:
        merged["rel_rank"] = (
            merged.groupby("Date", sort=False)[target_col]
            .apply(lambda s: _rel_rank_per_date_from_excess(s, cost=cost_threshold))
            .reset_index(level=0, drop=True)
            .astype("Int8")
        )

    rt = _build_rank_target(merged, target_col=target_col, n_expected=len(EXPECTED_SECTORS))
    merged = merged.merge(rt.rename("rank_target"), on=["Date", "Sector"], how="left", validate="one_to_one")

    if merged["rank_target"].isna().any():
        raise RuntimeError("rank_target contains NaNs after merge")

    feature_cols = _select_feature_columns(merged, target_col=target_col)
    X = merged[feature_cols].copy()
    y_gate = merged["y_gate"].to_numpy(dtype=np.int8)
    y_rank = merged["rank_target"].to_numpy(dtype=np.int16)

    group_sizes = merged.groupby("Date", sort=False).size().to_numpy(dtype=np.int32)

    if group_sizes.sum() != len(y_rank) or len(y_rank) != len(X) or len(y_gate) != len(X):
        raise RuntimeError("Alignment assertion failed: sum(group_sizes) != len(y_rank)/len(y_gate)/len(X)")

    if enforce_full_universe and not np.all(group_sizes == len(EXPECTED_SECTORS)):
        raise RuntimeError("Universe policy expected all group_sizes == 9")

    return ShapingResult(
        horizon=horizon,
        target_col=target_col,
        features_path=features_path,
        labels_path=labels_path,
        correlations_path=correlations_path,
        full_df=merged,
        X=X,
        y_gate=y_gate,
        y_rank=y_rank,
        group_sizes=group_sizes,
        feature_cols=feature_cols,
        dropped_rows_nan_target=int(dropped_rows_nan_target),
        dropped_dates_nan_target=int(dropped_dates_nan_target),
        dropped_dates_universe_policy=int(dropped_dates_universe_policy),
        cost_threshold=cost_threshold,
    )


def _sanity_report(res: ShapingResult, *, show_first_dates: int = 3) -> None:
    df = res.full_df

    print("\nFINAL SANITY CHECK !")
    print(f"horizon={res.horizon} target_col={res.target_col}")

    is_unique = not df.duplicated(["Date", "Sector"]).any()
    print(f"Key integrity: unique(Date,Sector)={is_unique}")

    print(f"X shape={res.X.shape} y_gate len={len(res.y_gate)} y_rank len={len(res.y_rank)}")

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

    keys = res.full_df[["Date", "Sector"]].copy()
    keys["Sector"] = keys["Sector"].astype("string")

    X_path = out_dir / "X.parquet"
    y_gate_path = out_dir / "y_gate.npy"
    y_rank_path = out_dir / "y_rank.npy"
    y_path = out_dir / "y.parquet"
    group_path = out_dir / "group_sizes.npy"
    keys_path = out_dir / "keys.parquet"
    meta_path = out_dir / "meta.json"

    res.X.to_parquet(X_path, index=False, engine="pyarrow")
    np.save(y_gate_path, res.y_gate)
    np.save(y_rank_path, res.y_rank)
    np.save(group_path, res.group_sizes)
    keys.to_parquet(keys_path, index=False, engine="pyarrow")

    y_df = res.full_df[["Date", "Sector"]].copy()
    for c in ("label_excess", "y_gate", "rel_rank", "rank_target", "cost_bps", "horizon"):
        if c in res.full_df.columns:
            y_df[c] = res.full_df[c].to_numpy()
    y_df.to_parquet(y_path, index=False, engine="pyarrow")

    meta = {
        "horizon": int(res.horizon),
        "target_col": res.target_col,
        "cost_threshold": float(res.cost_threshold),
        "n_rows": int(len(res.y_rank)),
        "n_dates": int(res.full_df["Date"].nunique()),
        "n_features": int(res.X.shape[1]),
        "y_artifacts": {
            "y_gate": "y_gate.npy",
            "y_rank": "y_rank.npy",
            "y": "y.parquet",
        },
        "expected_sectors": list(EXPECTED_SECTORS),
        "dropped_rows_nan_target": int(res.dropped_rows_nan_target),
        "dropped_dates_nan_target": int(res.dropped_dates_nan_target),
        "dropped_dates_universe_policy": int(res.dropped_dates_universe_policy),
        "feature_cols": list(res.feature_cols),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if verify_reload:
        X2 = pd.read_parquet(X_path)
        y_gate2 = np.load(y_gate_path)
        y_rank2 = np.load(y_rank_path)
        gs2 = np.load(group_path)
        keys2 = pd.read_parquet(keys_path)
        _ = pd.read_parquet(y_path)

        if len(X2) != len(y_gate2) or len(X2) != len(y_rank2) or len(X2) != len(keys2):
            raise RuntimeError(
                f"Reloaded lengths mismatch for h={res.horizon}: len(X)={len(X2)} len(y_gate)={len(y_gate2)} len(y_rank)={len(y_rank2)} len(keys)={len(keys2)}"
            )
        if gs2.sum() != len(y_rank2):
            raise RuntimeError(
                f"Reloaded group_sizes mismatch for h={res.horizon}: sum(gs)={int(gs2.sum())} len(y_rank)={int(len(y_rank2))}"
            )


def save_all_results(results: dict[int, ShapingResult], *, out_root: Path = DATASET_OUT_DIR) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for h, res in results.items():
        _save_shaping_result(res, out_root=out_root, verify_reload=True)
    print(f"\n[saved] Wrote datasets to: {out_root}")


def run_all_horizons(*, enforce_full_universe: bool = True, save_outputs: bool = True) -> dict[int, ShapingResult]:
    out: dict[int, ShapingResult] = {}

    for h in HORIZONS:
        res = load_dual_model_dataset(h, enforce_full_universe=enforce_full_universe, verbose=True)
        _sanity_report(res, show_first_dates=3)
        out[h] = res

    print("\nHORIZON CONSISTENCY")
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
    print(f"[dataset_shaping] BUILD_ID={BUILD_ID}")
    run_all_horizons(enforce_full_universe=True, save_outputs=True)
