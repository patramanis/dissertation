from __future__ import annotations
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from data.pit.mapper import nan_rates, read_raw_csv, schema_snapshot, sha256_file, stable_write_csv
from data.pit.policies import SECTORS
from utils.paths import find_workspace_root

log = logging.getLogger(__name__)

def _workspace_root() -> Path:
    return find_workspace_root(Path(__file__))

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

def _resolve_config_path(p: Path, base_dir: Path) -> Path:
    return p if p.is_absolute() else (base_dir / p).resolve()

def _get_label_date_range(config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    labels_raw = config.get("labels")
    labels: dict[str, Any] = labels_raw if isinstance(labels_raw, dict) else {}

    start_s = labels.get("date_start") or config.get("label_date_start")
    end_s = labels.get("date_end") or config.get("label_date_end")

    if not start_s or not end_s:
        raise ValueError("Missing label date range in config (labels.date_start/date_end)")

    start = pd.Timestamp(start_s)
    end = pd.Timestamp(end_s)
    return start.normalize(), end.normalize()

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

    cfg = config.get("label_output_root")
    if isinstance(cfg, str) and cfg:
        return _resolve_config_path(Path(cfg), base_dir)

    ws = _workspace_root()
    return ws / "data" / "interim" / "label_inputs"

def build_prices_close(
    *,
    config_path: Path,
    input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)

    label_start, label_end = _get_label_date_range(config)

    spy_path = input_root / "SPY.csv"
    spdr_path = input_root / "SPDR.csv"

    if not spy_path.exists():
        raise FileNotFoundError(f"Missing SPY.csv in {input_root}")
    if not spdr_path.exists():
        raise FileNotFoundError(f"Missing SPDR.csv in {input_root}")

    spy_raw, spy_report = read_raw_csv(spy_path)
    spdr_raw, spdr_report = read_raw_csv(spdr_path)

    if not isinstance(spy_raw.index, pd.DatetimeIndex) or not isinstance(spdr_raw.index, pd.DatetimeIndex):
        raise AssertionError("read_raw_csv must return Date-indexed DataFrames")

    if spy_raw.index.tz is not None or spdr_raw.index.tz is not None:
        raise AssertionError("Raw indices must be tz-naive")

    spy_raw = spy_raw.copy()
    spdr_raw = spdr_raw.copy()
    spy_raw.index = pd.DatetimeIndex(spy_raw.index).normalize()
    spdr_raw.index = pd.DatetimeIndex(spdr_raw.index).normalize()

    if "SPY" not in spy_raw.columns:
        raise ValueError("SPY.csv must contain a 'SPY' column")

    missing_sectors = [s for s in SECTORS if s not in spdr_raw.columns]
    if missing_sectors:
        raise ValueError(f"SPDR.csv missing sectors: {missing_sectors}")

    calendar = pd.DatetimeIndex(spy_raw.index).sort_values().unique()
    mask = (calendar >= label_start) & (calendar <= label_end)
    expected = pd.DatetimeIndex(calendar[mask], name="Date")
    if len(expected) == 0:
        raise ValueError("Label date range produced empty calendar")

    spy = spy_raw[["SPY"]].reindex(expected)
    spdr = spdr_raw[list(SECTORS)].reindex(expected)

    if not spy.index.equals(expected) or not spdr.index.equals(expected):
        raise AssertionError("Calendar reindex mismatch")

    if spy.isna().any().any():
        raise AssertionError("SPY has missing values in label range")
    if spdr.isna().any().any():
        raise AssertionError("SPDR sectors have missing values in label range")

    out = pd.concat([spy, spdr], axis=1)
    out.index = pd.DatetimeIndex(out.index, name="Date")

    export_df = out.reset_index()
    export_df = export_df[["Date", "SPY", *SECTORS]]

    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "prices_close.csv"
    stable_write_csv(export_df, out_path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "S2",
        "label_date_range": {
            "start": label_start.date().isoformat(),
            "end": label_end.date().isoformat(),
        },
        "input_root": str(input_root),
        "output_root": str(output_root),
        "inputs": {
            "SPY.csv": {"sha256": sha256_file(spy_path), "coerced_to_nan": spy_report.coerced_to_nan},
            "SPDR.csv": {"sha256": sha256_file(spdr_path), "coerced_to_nan": spdr_report.coerced_to_nan},
        },
        "outputs": {
            "prices_close.csv": {
                "sha256": sha256_file(out_path),
                "schema": schema_snapshot(export_df),
                "nan_rates": nan_rates(export_df),
                "date_coverage": {
                    "start": pd.to_datetime(export_df["Date"]).min().date().isoformat(),
                    "end": pd.to_datetime(export_df["Date"]).max().date().isoformat(),
                    "n_dates": int(len(export_df)),
                },
            }
        },
    }

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log.info("S2 complete: wrote %s", out_path)
    return manifest

def main(config_path: str | Path, *, input_root: str | Path | None = None, output_root: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(config_path)
    config = _load_config(cfg_path)

    base_dir = cfg_path.resolve().parent

    in_root = _select_input_root(Path(input_root) if input_root else None, config, base_dir=base_dir)
    out_root = _select_output_root(Path(output_root) if output_root else None, config, base_dir=base_dir)

    return build_prices_close(config_path=cfg_path, input_root=in_root, output_root=out_root)