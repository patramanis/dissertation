from __future__ import annotations
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

from .mapper import (
    apply_ffill_limit,
    assert_no_values_before_first_actionable,
    compute_available_from,
    first_actionable_dates,
    load_trading_calendar,
    map_to_calendar,
    nan_rates,
    read_raw_csv,
    schema_snapshot,
    sha256_file,
    stable_write_csv,
)
from .policies import SECTORS, default_policy_for_series, policies_from_config

from utils.paths import find_workspace_root

log = logging.getLogger(__name__)

def _iso_date(x: object) -> str:
    return pd.Timestamp(str(x)).date().isoformat()
def _workspace_root() -> Path:
    return find_workspace_root(Path(__file__))
def _load_config(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".json"}:
        return json.loads(text)
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML (not installed)") from e
        return yaml.safe_load(text)

    raise ValueError(f"Unsupported config extension: {config_path.suffix}")

def _resolve_config_path(p: Path, base_dir: Path) -> Path:
    return p if p.is_absolute() else (base_dir / p).resolve()

def _select_input_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit

    cfg = config.get("input_root")
    if isinstance(cfg, str) and cfg:
        return _resolve_config_path(Path(cfg), base_dir)

    ws = _workspace_root()
    candidates = [
        ws / "data" / "raw",
    ]
    for p in candidates:
        if p.exists():
            return p

    return candidates[0]

def _select_output_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit

    cfg = config.get("output_root")
    if isinstance(cfg, str) and cfg:
        return _resolve_config_path(Path(cfg), base_dir)

    ws = _workspace_root()
    return ws / "data" / "interim" / "aligned_pit"

def _build_id(config_bytes: bytes, input_hashes: dict[str, str]) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(b"S1")
    h.update(config_bytes)
    for k in sorted(input_hashes):
        h.update(k.encode("utf-8"))
        h.update(input_hashes[k].encode("utf-8"))
    return h.hexdigest()

def _deterministic_value_cols(series_name: str, cols: list[str]) -> list[str]:
    if series_name == "SPDR":
        present = [c for c in SECTORS if c in cols]
        missing = [c for c in SECTORS if c not in cols]
        if missing:
            raise ValueError(f"SPDR missing sectors: {missing}")
        return present

    return sorted(cols)

def _pit_config(config: dict[str, Any]) -> dict[str, Any]:
    pit = config.get("pit")
    return pit if isinstance(pit, dict) else {}

def _series_pit_config(pit: dict[str, Any], series_name: str) -> dict[str, Any]:
    out = dict(pit)
    overrides = pit.get("series")
    if isinstance(overrides, dict):
        ov = overrides.get(series_name)
        if isinstance(ov, dict):
            out.update(ov)
    return out

def _require_complete_series(pit: dict[str, Any]) -> set[str]:
    raw = pit.get("require_complete_series", None)
    if raw is None:
        return {"SPY", "SPDR"}
    if isinstance(raw, list):
        return {str(x) for x in raw}
    raise ValueError("pit.require_complete_series must be a list[str] or null")

def _assert_coverage(
    series_name: str,
    aligned: pd.DataFrame,
    *,
    min_rows: int | None,
    max_nan_pct: float | None,
) -> None:
    if min_rows is None and max_nan_pct is None:
        return

    for col in aligned.columns:
        s = aligned[col]

        if min_rows is not None:
            n = int(s.notna().sum())
            if n < int(min_rows):
                raise AssertionError(f"{series_name}.{col}: only {n} non-null rows (min_rows={min_rows})")

        if max_nan_pct is not None:
            first_valid = s.first_valid_index()
            if first_valid is None:
                raise AssertionError(f"{series_name}.{col}: all NaN (max_nan_pct={max_nan_pct})")
            tail = s.loc[first_valid:]
            if len(tail) == 0:
                raise AssertionError(f"{series_name}.{col}: empty tail after first valid")
            nan_pct = float(tail.isna().sum() / len(tail))
            if nan_pct > float(max_nan_pct):
                raise AssertionError(
                    f"{series_name}.{col}: {nan_pct:.1%} NaNs exceeds max_nan_pct={max_nan_pct} from {_iso_date(first_valid)}"
                )

def build_stage(
    *,
    config: dict[str, Any],
    config_path: Path,
    input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    csv_paths = sorted(input_root.glob("*.csv"), key=lambda p: p.name)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_root}")

    spy_path = input_root / "SPY.csv"
    if not spy_path.exists():
        raise FileNotFoundError(f"Missing SPY.csv in {input_root}")

    trading_calendar = load_trading_calendar(spy_path)

    cfg_policies = policies_from_config(config)
    pit_cfg = _pit_config(config)
    require_complete = _require_complete_series(pit_cfg)

    input_hashes: dict[str, str] = {}
    for p in csv_paths:
        input_hashes[str(p.relative_to(input_root)).replace("\\", "/")] = sha256_file(p)

    config_bytes = config_path.read_bytes()
    build_id = _build_id(config_bytes, input_hashes)

    created_at_utc = datetime.now(timezone.utc).isoformat()

    per_series: dict[str, Any] = {}
    output_hashes: dict[str, str] = {}

    out_dir = output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        name = csv_path.stem
        raw_df, read_report = read_raw_csv(csv_path)

        value_cols = _deterministic_value_cols(name, list(raw_df.columns))
        raw_df = raw_df.reindex(columns=value_cols)

        if name in cfg_policies:
            policy = cfg_policies[name]
        else:
            policy = default_policy_for_series(name, dates=pd.DatetimeIndex(raw_df.index))

        available_from = compute_available_from(
            pd.DatetimeIndex(raw_df.index),
            policy,
            trading_calendar,
        )

        mapped = map_to_calendar(raw_df, trading_calendar, available_from)
        first_actionable = first_actionable_dates(mapped)

        if policy.kind == "monthly":
            sparse_aligned = mapped.reindex(trading_calendar)
            for c in sparse_aligned.columns:
                sparse_aligned[c] = pd.to_numeric(sparse_aligned[c], errors="coerce").astype("float64")
            sparse_aligned.index = pd.DatetimeIndex(sparse_aligned.index, name="Date")
            mapped_export = sparse_aligned.reindex(columns=value_cols).reset_index()
            sparse_path = out_dir / f"{name}_sparse.csv"
            stable_write_csv(mapped_export, sparse_path)
            output_hashes[str(sparse_path.relative_to(out_dir)).replace("\\", "/")] = sha256_file(sparse_path)

        aligned = apply_ffill_limit(mapped, trading_calendar, policy.ffill_limit)
        assert_no_values_before_first_actionable(name, aligned, first_actionable)

        if name in require_complete:
            for col in aligned.columns:
                first_valid = aligned[col].first_valid_index()
                if first_valid is None:
                    raise AssertionError(f"{name}.{col}: all NaN but is marked require_complete_series")
                tail = aligned.loc[first_valid:, col]
                if tail.isna().any():
                    raise AssertionError(f"{name}.{col}: has missing values after {_iso_date(first_valid)} (require_complete_series)")

        series_cfg = _series_pit_config(pit_cfg, name)
        min_rows = series_cfg.get("min_rows", None)
        if min_rows is not None:
            min_rows = int(min_rows)
        max_nan_pct = series_cfg.get("max_nan_pct", None)
        if max_nan_pct is not None:
            max_nan_pct = float(max_nan_pct)
        _assert_coverage(name, aligned, min_rows=min_rows, max_nan_pct=max_nan_pct)

        aligned = aligned.reindex(columns=value_cols)
        export_df = aligned.reset_index()

        out_path = out_dir / f"{name}.csv"
        stable_write_csv(export_df, out_path)
        output_hashes[str(out_path.relative_to(out_dir)).replace("\\", "/")] = sha256_file(out_path)

        per_series[name] = {
            "policy": asdict(policy),
            "input": {
                "file": str(csv_path.relative_to(input_root)).replace("\\", "/"),
                "sha256": input_hashes[str(csv_path.relative_to(input_root)).replace("\\", "/")],
                "coerced_to_nan": read_report.coerced_to_nan,
            },
            "output": {
                "file": str(out_path.relative_to(out_dir)).replace("\\", "/"),
                "sha256": output_hashes[str(out_path.relative_to(out_dir)).replace("\\", "/")],
                "schema": schema_snapshot(export_df),
                "nan_rates": nan_rates(export_df),
                "date_coverage": {
                    "start": pd.to_datetime(export_df["Date"]).min().date().isoformat(),
                    "end": pd.to_datetime(export_df["Date"]).max().date().isoformat(),
                    "n_dates": int(len(export_df)),
                },
                "first_actionable": {k: v.date().isoformat() for k, v in first_actionable.items()},
            },
        }

    manifest = {
        "build_id": build_id,
        "created_at_utc": created_at_utc,
        "stage": "S1",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "input_root": str(input_root),
        "output_root": str(output_root),
        "trading_calendar": {
            "canonical": "SPY",
            "n_dates": int(len(trading_calendar)),
            "start": trading_calendar.min().date().isoformat(),
            "end": trading_calendar.max().date().isoformat(),
        },
        "pit_checks": {
            "require_complete_series": sorted(require_complete),
            "min_rows_default": pit_cfg.get("min_rows", None),
            "max_nan_pct_default": pit_cfg.get("max_nan_pct", None),
        },
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "series": per_series,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log.info("S1 complete: wrote %d series to %s", len(per_series), out_dir)
    return manifest

def main(config_path: str | Path, *, input_root: str | Path | None = None, output_root: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(config_path)
    config = _load_config(cfg_path)

    base_dir = cfg_path.resolve().parent

    in_root = _select_input_root(Path(input_root) if input_root else None, config, base_dir=base_dir)
    out_root = _select_output_root(Path(output_root) if output_root else None, config, base_dir=base_dir)

    return build_stage(config=config, config_path=cfg_path, input_root=in_root, output_root=out_root)