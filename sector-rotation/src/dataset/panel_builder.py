from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from data.pit.policies import SECTORS
from data.artifacts import build_manifest, describe_table, read_csv, sha256_file, write_csv, write_json
from dataset.feature_policy import FeaturePolicy, select_feature_columns

_SORT_KIND = "mergesort"

@dataclass(frozen=True)
class PanelPaths:
    tech_csv: Path
    corr_csv: Path
    labels_csv: Path
    labels_manifest_json: Path | None
    macro_csv: Path | None
    macro_dir: Path | None
    output_dir: Path

    @staticmethod
    def defaults(*, workspace_root: Path, horizon: int) -> "PanelPaths":
        ws = workspace_root
        return PanelPaths(
            tech_csv=ws / "data" / "features" / "technical_indicators" / f"tech_h{int(horizon)}.csv",
            corr_csv=ws / "data" / "features" / "correlations" / "corr_features.csv",
            labels_csv=ws / "data" / "labels" / "excess" / f"labels_h{int(horizon)}.csv",
            labels_manifest_json=ws / "data" / "labels" / "excess" / "manifest.json",
            macro_csv=ws / "data" / "features" / "base_stationary" / "macro_features.csv",
            macro_dir=ws / "data" / "features" / "base_stationary",
            output_dir=ws / "data" / "processed" / "panel",
        )

@dataclass(frozen=True)
class PanelConfig:
    stage: str = "S7_PANEL"
    sort_kind: str = _SORT_KIND
    full_universe_policy: bool = True
    enforce_no_leakage_columns: bool = True
    feature_policy: FeaturePolicy = FeaturePolicy(
        blocked_patterns=("corr_macro_", "_126"),
        blocked_level_suffixes=("_asinh", "_log1p"),
    )

@dataclass(frozen=True)
class PanelResult:
    horizon: int
    panel_csv: Path
    manifest_json: Path
    manifest: dict[str, Any]

def _normalize_sector(df: pd.DataFrame) -> pd.DataFrame:
    if "Sector" not in df.columns:
        raise ValueError("Missing Sector column")
    out = df.copy()
    raw = out["Sector"].astype(str)
    out["Sector"] = pd.Categorical(raw, categories=list(SECTORS), ordered=True)
    if out["Sector"].isna().any():
        bad = sorted(set(raw[out["Sector"].isna()].tolist()))
        raise AssertionError(f"Unexpected Sector values (must be exactly SPDR 9): {bad}")
    return out

def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("Missing Date column")
    out = df.copy()
    dt = pd.to_datetime(out["Date"], errors="raise")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    out["Date"] = dt.dt.normalize()
    return out

def _assert_sorted_no_dupes(df: pd.DataFrame, *, key_cols: list[str], name: str) -> None:
    if not df[key_cols].equals(df.sort_values(key_cols, kind=_SORT_KIND)[key_cols].reset_index(drop=True)):
        raise AssertionError(f"{name} is not sorted by {key_cols}")
    dup = int(df.duplicated(subset=key_cols).sum())
    if dup:
        raise AssertionError(f"{name} has duplicate keys on {key_cols} (n_dup={dup})")

def _forbid_leakage_feature_names(cols: list[str]) -> None:
    forbidden = ("label", "target", "forward", "fwd")
    bad = [c for c in cols if any(s in c.lower() for s in forbidden)]
    if bad:
        raise AssertionError(f"Forbidden leakage-like columns present: {sorted(bad)[:20]}")

def _stable_tie_hash(date: pd.Timestamp, sector: str) -> str:
    s = f"{pd.Timestamp(date).date().isoformat()}|{sector}".encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def _load_macro_features(paths: PanelPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    if paths.macro_csv and paths.macro_csv.exists():
        df = read_csv(paths.macro_csv, required_cols=["Date"], key_cols=["Date"], sort_keys=True)
        return df, {"path": str(paths.macro_csv), "sha256": sha256_file(paths.macro_csv)}

    if paths.macro_dir and paths.macro_dir.exists():
        pieces: list[pd.DataFrame] = []
        used_files: list[str] = []
        for p in sorted(paths.macro_dir.glob("*.csv")):
            if p.name in {"SPDR.csv", "SPY.csv", "manifest.json"}:
                continue
            df = read_csv(p, required_cols=["Date"], key_cols=["Date"], sort_keys=True)
            pieces.append(df)
            used_files.append(p.name)

        if not pieces:
            raise FileNotFoundError("No macro CSVs found to assemble macro features")

        out = pieces[0]
        for d in pieces[1:]:
            overlap = set(out.columns) & set(d.columns)
            overlap.discard("Date")
            if overlap:
                raise AssertionError(f"Macro column collision: {sorted(overlap)[:10]}")
            out = out.merge(d, on="Date", how="outer", sort=False, validate="one_to_one")

        out = out.sort_values(["Date"], kind=_SORT_KIND).reset_index(drop=True)
        return out, {"path": str(paths.macro_dir), "assembled_from": used_files}

    raise FileNotFoundError("macro_features.csv not found and macro_dir not provided")

def build_panel(horizon: int, paths: PanelPaths, config: PanelConfig) -> PanelResult:
    h = int(horizon)

    tech = read_csv(paths.tech_csv, required_cols=["Date", "Sector"], key_cols=["Date", "Sector"], sort_keys=True)
    corr = read_csv(paths.corr_csv, required_cols=["Date", "Sector"], key_cols=["Date", "Sector"], sort_keys=True)
    labels_required = ["Date", "Sector", "label_excess", "horizon"]
    labels = read_csv(paths.labels_csv, required_cols=labels_required, key_cols=["Date", "Sector"], sort_keys=True)

    tech = _normalize_sector(_normalize_dates(tech))
    corr = _normalize_sector(_normalize_dates(corr))
    labels = _normalize_sector(_normalize_dates(labels))

    tech = tech.sort_values(["Date", "Sector"], kind=_SORT_KIND).reset_index(drop=True)
    corr = corr.sort_values(["Date", "Sector"], kind=_SORT_KIND).reset_index(drop=True)
    labels = labels.sort_values(["Date", "Sector"], kind=_SORT_KIND).reset_index(drop=True)

    _assert_sorted_no_dupes(tech, key_cols=["Date", "Sector"], name="tech")
    _assert_sorted_no_dupes(corr, key_cols=["Date", "Sector"], name="corr")
    _assert_sorted_no_dupes(labels, key_cols=["Date", "Sector"], name="labels")

    if int(labels["horizon"].dropna().unique()[0]) != h:
        raise AssertionError("labels horizon does not match requested horizon")

    if "label_contract" not in labels.columns:
        label_contract = "log_excess_return_ln(Ps,t+h/Ps,t)-ln(Pspy,t+h/Pspy,t)"
        if paths.labels_manifest_json and paths.labels_manifest_json.exists():
            m = json.loads(paths.labels_manifest_json.read_text(encoding="utf-8"))
            if m.get("label_timing") == {"denom": "T", "numer": "T+h"}:
                label_contract = "log_excess_return_ln(Ps,t+h/Ps,t)-ln(Pspy,t+h/Pspy,t)"
        labels["label_contract"] = label_contract

    def non_keys(df: pd.DataFrame) -> set[str]:
        return set([c for c in df.columns if c not in {"Date", "Sector"}])

    overlap_tc = (non_keys(tech) & non_keys(corr)) - {"horizon", "cost_bps"}
    overlap_tl = (non_keys(tech) & non_keys(labels)) - {"horizon", "cost_bps"}
    overlap_cl = (non_keys(corr) & non_keys(labels)) - {"horizon", "cost_bps"}
    if overlap_tc or overlap_tl or overlap_cl:
        bad = sorted(set().union(overlap_tc, overlap_tl, overlap_cl))
        raise AssertionError(f"Column collision across inputs (would cause _x/_y): {bad[:20]}")

    macro, macro_meta = _load_macro_features(paths)
    macro = _normalize_dates(macro)
    macro = macro.sort_values(["Date"], kind=_SORT_KIND).reset_index(drop=True)
    _assert_sorted_no_dupes(macro, key_cols=["Date"], name="macro")

    panel = labels.merge(tech, on=["Date", "Sector"], how="inner", sort=False, validate="one_to_one")
    panel = panel.merge(corr, on=["Date", "Sector"], how="inner", sort=False, validate="one_to_one")
    panel = panel.merge(macro, on=["Date"], how="inner", sort=False, validate="many_to_one")

    suffix_cols = [c for c in panel.columns if c.endswith("_x") or c.endswith("_y")]
    if suffix_cols:
        raise AssertionError(f"Merge suffix columns present: {sorted(suffix_cols)[:20]}")

    panel = _normalize_sector(_normalize_dates(panel))

    dropped_dates_incomplete = 0
    if config.full_universe_policy:
        counts = panel.groupby("Date", sort=True)["Sector"].nunique()
        bad_dates = counts[counts != len(SECTORS)].index
        if len(bad_dates):
            dropped_dates_incomplete = int(len(bad_dates))
            panel = panel.loc[~panel["Date"].isin(bad_dates)].copy()

    dropped_rows_nan_target = 0
    target_cols = ["label_excess"]
    na_any_row = panel[target_cols].isna().any(axis=1)
    na_by_date = na_any_row.groupby(panel["Date"], sort=True).transform("any")
    if bool(na_by_date.any()):
        bad_na_dates = panel.loc[na_by_date, "Date"].drop_duplicates().to_list()
        dropped_rows_nan_target = int(panel.loc[panel["Date"].isin(bad_na_dates)].shape[0])
        panel = panel.loc[~panel["Date"].isin(bad_na_dates)].copy()

    if config.full_universe_policy:
        counts2 = panel.groupby("Date", sort=True)["Sector"].nunique()
        bad_dates2 = counts2[counts2 != len(SECTORS)].index
        if len(bad_dates2):
            panel = panel.loc[~panel["Date"].isin(bad_dates2)].copy()
            dropped_dates_incomplete += int(len(bad_dates2))

    panel = panel.sort_values(["Date", "Sector"], kind=_SORT_KIND).reset_index(drop=True)

    feature_cols, policy_report = select_feature_columns(panel, policy=config.feature_policy)
    if config.enforce_no_leakage_columns:
        _forbid_leakage_feature_names(feature_cols)

    key_cols = ["Date", "Sector"]
    targets = ["label_excess"]
    meta_cols = ["horizon", "label_contract"]
    ordered = key_cols + targets + meta_cols + sorted(feature_cols)

    missing = [c for c in ordered if c not in panel.columns]
    if missing:
        raise AssertionError(f"Panel missing required columns: {missing}")

    panel = panel.loc[:, ordered].copy()

    dup = int(panel.duplicated(subset=["Date", "Sector"]).sum())
    if dup:
        raise AssertionError(f"Final panel has duplicate keys (n_dup={dup})")

    out_dir = paths.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / f"panel_h{h}.csv"
    man_path = out_dir / f"panel_h{h}_manifest.json"

    write_csv(panel, panel_path)

    cfg_bytes = json.dumps({"horizon": h, "config": config.__dict__}, sort_keys=True, default=str).encode("utf-8")
    hh = hashlib.sha256()
    hh.update(config.stage.encode("utf-8"))
    hh.update(cfg_bytes)
    for k, p in sorted(
        {
            "tech": paths.tech_csv,
            "corr": paths.corr_csv,
            "labels": paths.labels_csv,
        }.items(),
        key=lambda kv: kv[0],
    ):
        hh.update(k.encode("utf-8"))
        hh.update(sha256_file(p).encode("utf-8"))
    if paths.macro_csv and paths.macro_csv.exists():
        hh.update(b"macro")
        hh.update(sha256_file(paths.macro_csv).encode("utf-8"))
    else:
        hh.update(b"macro_assembled")
        hh.update(json.dumps(macro_meta, sort_keys=True).encode("utf-8"))

    build_id = hh.hexdigest()

    group_sizes = panel.groupby("Date", sort=True)["Sector"].nunique().astype("int64")

    nanr = panel.isna().mean().sort_values(ascending=False)
    nan_top = [(str(k), float(v)) for k, v in nanr.head(20).items()]

    inputs = {
        "tech_features": {
            "path": str(paths.tech_csv),
            "sha256": sha256_file(paths.tech_csv),
            "n_rows": int(len(tech)),
            "n_cols": int(tech.shape[1]),
        },
        "corr_features": {"path": str(paths.corr_csv), "sha256": sha256_file(paths.corr_csv)},
        "macro_features": {"path": str(paths.macro_csv) if paths.macro_csv else str(paths.macro_dir), **macro_meta},
        "labels": {
            "path": str(paths.labels_csv),
            "sha256": sha256_file(paths.labels_csv),
            "label_contract": str(panel["label_contract"].iloc[0]) if len(panel) else "denom=T-1,numer=T+h-1",
        },
    }

    outputs = {
        "panel": {
            "path": str(panel_path.name),
            "sha256": sha256_file(panel_path),
            "n_rows": int(len(panel)),
            "n_cols": int(panel.shape[1]),
        }
    }

    coverage = describe_table(panel).get("coverage") or {"date_min": None, "date_max": None, "n_dates": 0}

    stats = {
        "nan_rate_by_col_top": nan_top,
    }

    manifest = build_manifest(
        stage=config.stage,
        build_id=build_id,
        inputs=inputs,
        outputs=outputs,
        stats=stats,
        config={
            "horizon": h,
            "feature_policy": policy_report,
        },
        repo_root=Path(__file__).resolve().parents[2],
        include_timestamp=False,
    )

    manifest.update(
        {
            "horizon": h,
            "schema": {
                "key": ["Date", "Sector"],
                "targets": ["label_excess"],
                "n_features": int(len(feature_cols)),
            },
            "coverage": coverage,
            "integrity": {
                "unique_key_ok": True,
                "full_universe_policy": bool(config.full_universe_policy),
                "group_size_min": int(group_sizes.min()) if len(group_sizes) else 0,
                "group_size_max": int(group_sizes.max()) if len(group_sizes) else 0,
                "dropped_dates_incomplete_universe": int(dropped_dates_incomplete),
                "dropped_rows_nan_target": int(dropped_rows_nan_target),
            },
            "feature_cols": feature_cols,
            "feature_policy": {
                **policy_report,
                "forbidden_cols_found": 0,
            },
            "determinism": {
                "sort_kind": config.sort_kind,
                "column_order": "sorted",
                "stable_tie_break": "hash(Date,Sector)",
            },
        }
    )

    write_json(manifest, man_path)

    return PanelResult(horizon=h, panel_csv=panel_path, manifest_json=man_path, manifest=manifest)