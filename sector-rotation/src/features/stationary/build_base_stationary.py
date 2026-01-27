from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

from data.pit.mapper import nan_rates, schema_snapshot, sha256_file, stable_write_csv
from data.pit.policies import SECTORS
from utils.paths import find_workspace_root
from .monthly_growth import build_monthly_growth
from .transforms import Family, apply_family, recognize_macro_monthly

log = logging.getLogger(__name__)

def _load_config(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML (not installed)") from e
        return yaml.safe_load(text)

    raise ValueError(f"Unsupported config extension: {config_path.suffix}")

def _workspace_root() -> Path:
    return find_workspace_root(Path(__file__))

def _resolve_config_path(p: Path, base_dir: Path) -> Path:
    return p if p.is_absolute() else (base_dir / p).resolve()

def _select_input_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit

    cfg = _base_stationary_cfg(config)
    v = cfg.get("input_root")
    if isinstance(v, str) and v:
        return _resolve_config_path(Path(v), base_dir)

    ws = _workspace_root()
    return ws / "data" / "interim" / "aligned_pit"

def _select_output_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit

    cfg = _base_stationary_cfg(config)
    v = cfg.get("output_root")
    if isinstance(v, str) and v:
        return _resolve_config_path(Path(v), base_dir)

    ws = _workspace_root()
    return ws / "data" / "features" / "base_stationary"

def _build_id(config_bytes: bytes, input_hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    h.update(b"S3")
    h.update(config_bytes)
    for k in sorted(input_hashes):
        h.update(k.encode("utf-8"))
        h.update(input_hashes[k].encode("utf-8"))
    return h.hexdigest()

def _read_aligned_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date in {path.name}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    value_cols = [c for c in df.columns if c != "Date"]

    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    df = df.sort_values("Date", kind="mergesort")
    df = df.set_index("Date")
    df.index = pd.DatetimeIndex(df.index, name="Date")
    return df

def _canonical_calendar_from_spy(input_root: Path) -> pd.DatetimeIndex:
    spy_path = input_root / "SPY.csv"
    if not spy_path.exists():
        raise FileNotFoundError(f"Missing SPY.csv under {input_root}")

    spy = _read_aligned_csv(spy_path)
    idx = pd.DatetimeIndex(spy.index, name="Date")
    return idx

def _enforce_calendar_equality(df: pd.DataFrame, calendar: pd.DatetimeIndex, *, name: str) -> None:
    if not df.index.equals(calendar):
        raise AssertionError(f"Calendar mismatch for {name}: does not match SPY")

def _base_stationary_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("base_stationary")
    return cfg if isinstance(cfg, dict) else {}

def _family_map(config: dict[str, Any]) -> dict[str, Family]:
    cfg = _base_stationary_cfg(config)
    m = cfg.get("family_map")
    if not isinstance(m, dict):
        return {}
    out: dict[str, Family] = {}
    valid_families = {"always_positive_price_like", "rate_or_spread", "can_be_negative_level", "uncertainty_or_counts", "macro_monthly"}
    for k, v in m.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if v in valid_families:
            out[k] = v  # type: ignore[assignment]
    return out

def _default_family_for_series(series_name: str) -> Family:
    if recognize_macro_monthly(series_name):
        return "macro_monthly"

    if series_name in {"SPY", "SPDR", "TLT", "VUSTX", "Futures"}:
        return "always_positive_price_like"

    if series_name in {"BAMLH0A0HYM2", "NFCI"}:
        return "rate_or_spread"

    if series_name in {"ICSA"}:
        return "uncertainty_or_counts"

    if series_name in {"EPU", "GPR"}:
        return "uncertainty_or_counts"

    raise ValueError(
        f"Unknown family for series {series_name!r}. Provide base_stationary.family_map in config."
    )

def _deterministic_value_cols(series_name: str, cols: list[str]) -> list[str]:
    if series_name == "SPDR":
        missing = [c for c in SECTORS if c not in cols]
        if missing:
            raise ValueError(f"SPDR missing sectors: {missing}")
        return list(SECTORS)
    return sorted(cols)

def build_stage(
    *,
    config: dict[str, Any],
    config_path: Path,
    input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    upstream_manifest = input_root / "manifest.json"
    if not upstream_manifest.exists():
        raise FileNotFoundError(f"Missing upstream manifest.json under {input_root}")

    upstream_manifest_sha256 = sha256_file(upstream_manifest)
    upstream = json.loads(upstream_manifest.read_text(encoding="utf-8"))

    calendar = _canonical_calendar_from_spy(input_root)

    csv_paths_all = sorted([p for p in input_root.glob("*.csv") if p.name != "manifest.json"], key=lambda p: p.name)
    csv_paths = [p for p in csv_paths_all if not p.stem.endswith("_sparse")]
    if not csv_paths:
        raise FileNotFoundError(f"No aligned PIT CSVs found in {input_root}")

    input_hashes: dict[str, str] = {}
    for p in csv_paths_all:
        rel = p.relative_to(input_root).as_posix()
        input_hashes[rel] = sha256_file(p)

    config_bytes = config_path.read_bytes()
    build_id = _build_id(config_bytes, input_hashes)

    config_sha256 = sha256_file(config_path)

    created_at_utc = datetime.now(timezone.utc).isoformat()

    cfg = _base_stationary_cfg(config)
    keep_levels = cfg.get("keep_levels", False)
    assert_non_negative = bool(cfg.get("assert_non_negative", False))

    family_map = _family_map(config)

    series: dict[str, Any] = {}
    output_hashes: dict[str, str] = {}

    output_root.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        name = csv_path.stem
        aligned = _read_aligned_csv(csv_path)
        aligned = aligned.reindex(columns=_deterministic_value_cols(name, list(aligned.columns)))

        _enforce_calendar_equality(aligned, calendar, name=name)

        policy = None
        try:
            policy = upstream.get("series", {}).get(name, {}).get("policy")
        except Exception:
            policy = None

        family = family_map.get(name)
        if family is None:
            family = _default_family_for_series(name)
        if family is None:
            raise ValueError(
                f"Unknown family for series '{name}'. Add base_stationary.family_map['{name}'] in config."
            )

        families_used: list[str] = [str(family)]

        derived_cols: list[str] = []
        features_df: pd.DataFrame

        if family == "macro_monthly":
            if len(aligned.columns) != 1:
                raise ValueError(f"{name}: macro_monthly expects exactly one column")
            sparse_path = input_root / f"{name}_sparse.csv"
            if not sparse_path.exists():
                raise FileNotFoundError(
                    f"Missing sparse mapped monthly series for {name}: {sparse_path.name}. "
                    "Re-run S1 with updated pipeline that exports *_sparse.csv for monthly series."
                )
            sparse_aligned = _read_aligned_csv(sparse_path)
            _enforce_calendar_equality(sparse_aligned, calendar, name=f"{name}_sparse")
            features_df = build_monthly_growth(
                aligned,
                calendar,
                series_name=name,
                monthly_sparse_aligned=sparse_aligned,
            )
            derived_cols = list(features_df.columns)
        else:
            features_df, derived_cols = apply_family(aligned, family=family, assert_non_negative=assert_non_negative)

        if keep_levels:
            for col in aligned.columns:
                features_df[f"{col}_level"] = aligned[col].astype("float64")
                derived_cols.append(f"{col}_level")

        features_df = features_df.reindex(columns=sorted(features_df.columns))

        export_df = features_df.reset_index()
        out_path = output_root / f"{name}.csv"
        stable_write_csv(export_df, out_path)
        out_rel = out_path.relative_to(output_root).as_posix()
        output_hashes[out_rel] = sha256_file(out_path)

        series[name] = {
            "policy": policy,
            "families": families_used,
            "derived_columns": list(features_df.columns),
            "schema": schema_snapshot(export_df),
            "nan_rates": nan_rates(export_df),
            "date_coverage": {
                "start": pd.to_datetime(export_df["Date"]).min().date().isoformat(),
                "end": pd.to_datetime(export_df["Date"]).max().date().isoformat(),
                "n_dates": int(len(export_df)),
            },
            "output_file": out_rel,
        }

    canonical_calendar = {
        "name": "SPY",
        "start": calendar.min().date().isoformat(),
        "end": calendar.max().date().isoformat(),
        "n_dates": int(len(calendar)),
    }

    manifest = {
        "build_id": build_id,
        "created_at_utc": created_at_utc,
        "stage": "S3",
        "config": {"path": str(config_path.resolve()), "sha256": config_sha256},
        "upstream": {
            "s1_manifest_path": str(upstream_manifest.resolve()),
            "s1_manifest_sha256": upstream_manifest_sha256,
        },
        "input_root": str(input_root),
        "output_root": str(output_root),
        "canonical_calendar": canonical_calendar,
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "series": series,
    }

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log.info("S3 complete: wrote %d series to %s", len(series), output_root)
    return manifest

def main(
    config_path: str | Path,
    *,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    cfg_path = Path(config_path)
    config = _load_config(cfg_path)
    base_dir = cfg_path.resolve().parent

    in_root = _select_input_root(Path(input_root) if input_root else None, config, base_dir=base_dir)
    out_root = _select_output_root(Path(output_root) if output_root else None, config, base_dir=base_dir)

    return build_stage(config=config, config_path=cfg_path, input_root=in_root, output_root=out_root)