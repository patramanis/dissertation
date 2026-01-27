from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from data.pit.mapper import nan_rates, normalize_date_series, schema_snapshot, sha256_file, stable_write_csv
from data.pit.policies import SECTORS
from utils.paths import find_workspace_root
from .compute import build_excess_labels

log = logging.getLogger(__name__)

def _workspace_root() -> Path:
    return find_workspace_root(Path(__file__))

def _load_config(config_path: Path) -> tuple[dict[str, Any], bytes]:
    config_bytes = config_path.read_bytes()
    text = config_bytes.decode("utf-8")

    if config_path.suffix.lower() == ".json":
        return json.loads(text), config_bytes

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML (not installed)") from e
        return yaml.safe_load(text), config_bytes

    raise ValueError(f"Unsupported config extension: {config_path.suffix}")

def _resolve_config_path(p: Path, base_dir: Path) -> Path:
    return p if p.is_absolute() else (base_dir / p).resolve()

def _labels_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("labels")
    return raw if isinstance(raw, dict) else {}

def _excess_cfg(config: dict[str, Any]) -> dict[str, Any]:
    for k in ("labels_excess", "excess_labels", "excess"):
        raw = config.get(k)
        if isinstance(raw, dict):
            return raw
    return {}

def _get_label_date_range(config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    labels = _labels_cfg(config)

    start_s = labels.get("date_start") or config.get("label_date_start")
    end_s = labels.get("date_end") or config.get("label_date_end")
    if not start_s or not end_s:
        raise ValueError("Missing label date range in config (labels.date_start/date_end)")

    start = pd.Timestamp(start_s).normalize()
    end = pd.Timestamp(end_s).normalize()
    return start, end

def _select_input_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit

    c = _excess_cfg(config)
    v = c.get("input_root")
    if isinstance(v, str) and v:
        return _resolve_config_path(Path(v), base_dir)

    ws = _workspace_root()
    return ws / "data" / "interim" / "label_inputs"

def _select_output_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit

    c = _excess_cfg(config)
    v = c.get("output_root")
    if isinstance(v, str) and v:
        return _resolve_config_path(Path(v), base_dir)

    ws = _workspace_root()
    return ws / "artifacts" / "labels" / "excess"

def _parse_horizons(config: dict[str, Any]) -> list[int]:
    c = _excess_cfg(config)
    v = c.get("horizons")
    if isinstance(v, list) and v:
        return [int(x) for x in v]
    return [5, 21, 63]

def _parse_cost_bps(config: dict[str, Any]) -> float:
    c = _excess_cfg(config)
    v = c.get("cost_bps")
    return float(v) if v is not None else 10.0

def _build_id(*, config_bytes: bytes, input_hashes: dict[str, str], upstream_manifest_sha: str | None) -> str:
    h = hashlib.sha256()
    h.update(b"S4")
    h.update(config_bytes)

    for k in sorted(input_hashes.keys()):
        h.update(k.encode("utf-8"))
        h.update(str(input_hashes[k]).encode("utf-8"))

    if upstream_manifest_sha:
        h.update(str(upstream_manifest_sha).encode("utf-8"))

    return h.hexdigest()

def _date_coverage(df: pd.DataFrame) -> dict[str, Any]:
    d = normalize_date_series(df["Date"]) if "Date" in df.columns else pd.Series([], dtype="datetime64[ns]")
    if len(d) == 0:
        return {"start": None, "end": None, "n_dates": 0}
    return {
        "start": pd.Timestamp(d.min()).date().isoformat(),
        "end": pd.Timestamp(d.max()).date().isoformat(),
        "n_dates": int(d.nunique()),
    }

def _stable_csv_sha256(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.10g", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def build_stage(
    *,
    config_path: Path,
    input_root: Path,
    output_root: Path,
    diagnostic: bool = False,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config, config_bytes = _load_config(config_path)

    label_start, label_end = _get_label_date_range(config)
    horizons = _parse_horizons(config)
    cost_bps = _parse_cost_bps(config)

    prices_path = input_root / "prices_close.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing prices_close.csv in {input_root}")

    prices = pd.read_csv(prices_path)

    prices["Date"] = normalize_date_series(prices["Date"])
    prices = prices.sort_values("Date", kind="mergesort")
    prices = prices.loc[(prices["Date"] >= label_start) & (prices["Date"] <= label_end)].copy()
    if len(prices) == 0:
        raise ValueError("Label date range produced empty prices_close")

    labels_by_h = build_excess_labels(prices, horizons=horizons, sectors=list(SECTORS))

    outputs: dict[str, Any] = {}
    for h, df in sorted(labels_by_h.items(), key=lambda kv: int(kv[0])):
        out_path = output_root / f"labels_h{int(h)}.csv"
        stable_write_csv(df, out_path)

        le = df["label_excess"]
        le_valid = le.dropna()
        label_stats = {
            "mean": float(le_valid.mean()) if len(le_valid) > 0 else None,
            "std": float(le_valid.std()) if len(le_valid) > 0 else None,
            "min": float(le_valid.min()) if len(le_valid) > 0 else None,
            "max": float(le_valid.max()) if len(le_valid) > 0 else None,
            "n_valid": int(len(le_valid)),
            "n_nan": int(le.isna().sum()),
        }

        outputs[out_path.name] = {
            "file": out_path.name,
            "sha256": sha256_file(out_path),
            "schema": schema_snapshot(df),
            "nan_rates": nan_rates(df),
            "date_coverage": _date_coverage(df),
            "label_stats": label_stats,
        }

    sha_full = sha256_file(prices_path)
    prices_for_hash = prices[["Date", "SPY", *SECTORS]].copy()
    sha_filtered = _stable_csv_sha256(prices_for_hash)
    input_hashes = {
        "prices_close.csv_full": sha_full,
        "prices_close.csv_filtered": sha_filtered,
    }

    upstream_manifest_path = input_root / "manifest.json"
    upstream_manifest_sha = sha256_file(upstream_manifest_path) if upstream_manifest_path.exists() else None

    build_id = _build_id(config_bytes=config_bytes, input_hashes=input_hashes, upstream_manifest_sha=upstream_manifest_sha)

    manifest: dict[str, Any] = {
        "build_id": build_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "S4",
        "config": {"path": str(config_path.resolve()), "sha256": hashlib.sha256(config_bytes).hexdigest()},
        "input_root": str(input_root),
        "output_root": str(output_root),
        "label_timing": {
            "formula": "ln(P_t+h / P_t) - ln(P_SPY_t+h / P_SPY_t)",
            "description": "Forward log-excess-return from decision date t over h trading days",
        },
        "cost_note": "Cost NOT embedded in labels. Apply 10bps in trading rule.",
        "inputs": {
            "prices_close.csv": {
                "sha256": sha_full,
                "sha256_full_file": sha_full,
                "sha256_filtered": sha_filtered,
                "schema": schema_snapshot(prices),
                "nan_rates": nan_rates(prices),
                "date_coverage": _date_coverage(prices),
            }
        },
        "label_date_range": {"start": label_start.date().isoformat(), "end": label_end.date().isoformat()},
        "horizons": [int(h) for h in sorted(horizons)],
        "cost_bps_for_trading": float(cost_bps),
        "outputs": outputs,
    }

    if upstream_manifest_sha:
        manifest["upstream_manifest"] = {
            "path": str(upstream_manifest_path.resolve()),
            "sha256": upstream_manifest_sha,
        }

    man_path = output_root / "manifest.json"
    man_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    if diagnostic:
        log.info("S4 labels built: %s", man_path)

    return manifest

def main(
    config_path: Path,
    *,
    input_root: Path | None = None,
    output_root: Path | None = None,
    diagnostic: bool = False,
) -> dict[str, Any]:
    base_dir = config_path.resolve().parent
    config, _ = _load_config(config_path)

    in_root = _select_input_root(input_root, config, base_dir=base_dir)
    out_root = _select_output_root(output_root, config, base_dir=base_dir)

    return build_stage(
        config_path=config_path,
        input_root=in_root,
        output_root=out_root,
        diagnostic=diagnostic,
    )