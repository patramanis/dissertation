from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from data.pit.mapper import nan_rates, schema_snapshot, sha256_file, stable_write_csv
from data.pit.policies import SECTORS
from utils.paths import find_workspace_root
from .compute import DRIVERS_REQUIRED, compute_corr_features

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

def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    c = config.get("correlations")
    return c if isinstance(c, dict) else {}

def _resolve_config_path(p: Path, base_dir: Path) -> Path:
    return p if p.is_absolute() else (base_dir / p).resolve()

def _select_input_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit
    c = _cfg(config)
    v = c.get("input_root")
    if isinstance(v, str) and v:
        return _resolve_config_path(Path(v), base_dir)
    ws = _workspace_root()
    return ws / "data" / "interim" / "aligned_pit"

def _select_stationary_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit
    c = _cfg(config)
    v = c.get("stationary_root")
    if isinstance(v, str) and v:
        return _resolve_config_path(Path(v), base_dir)
    ws = _workspace_root()
    return ws / "data" / "features" / "base_stationary"

def _select_output_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit
    c = _cfg(config)
    v = c.get("output_root")
    if isinstance(v, str) and v:
        return _resolve_config_path(Path(v), base_dir)
    ws = _workspace_root()
    return ws / "data" / "features" / "correlations"

def _read_aligned_prices(input_root: Path) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, dict[str, dict[str, Any]]]:
    spdr_path = input_root / "SPDR.csv"
    spy_path = input_root / "SPY.csv"
    s1_manifest_path = input_root / "manifest.json"

    if not spdr_path.exists() or not spy_path.exists():
        raise FileNotFoundError("S5 requires SPDR.csv and SPY.csv under input_root")

    spdr = pd.read_csv(spdr_path)
    spy = pd.read_csv(spy_path)
    if "Date" not in spdr.columns or "Date" not in spy.columns:
        raise ValueError("Missing Date column in SPDR/SPY")

    spdr["Date"] = pd.to_datetime(spdr["Date"]).dt.normalize()
    spy["Date"] = pd.to_datetime(spy["Date"]).dt.normalize()

    missing = [c for c in SECTORS if c not in spdr.columns]
    if missing:
        raise ValueError(f"SPDR missing sectors: {missing}")
    if "SPY" not in spy.columns:
        raise ValueError("SPY.csv missing SPY column")

    spdr = spdr[["Date", *SECTORS]].copy()
    spy = spy[["Date", "SPY"]].copy()
    for c in SECTORS:
        spdr[c] = pd.to_numeric(spdr[c], errors="raise").astype("float64")
    spy["SPY"] = pd.to_numeric(spy["SPY"], errors="raise").astype("float64")

    spdr = spdr.sort_values("Date", kind="mergesort").set_index("Date")
    spy = spy.sort_values("Date", kind="mergesort").set_index("Date")

    cal = pd.DatetimeIndex(spy.index, name="Date")
    if not pd.DatetimeIndex(spdr.index, name="Date").equals(cal):
        raise AssertionError("Calendar mismatch: SPDR vs SPY")

    inputs: dict[str, dict[str, Any]] = {
        "SPDR.csv": {"path": str(spdr_path.resolve()), "sha256": sha256_file(spdr_path)},
        "SPY.csv": {"path": str(spy_path.resolve()), "sha256": sha256_file(spy_path)},
    }
    if s1_manifest_path.exists():
        inputs["manifest.json"] = {"path": str(s1_manifest_path.resolve()), "sha256": sha256_file(s1_manifest_path)}

    return spdr, spy["SPY"].rename("SPY"), cal, inputs

def _load_stationary_macro(stationary_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if not stationary_root.exists():
        raise FileNotFoundError(f"stationary_root does not exist: {stationary_root}")

    manifest_path = stationary_root / "manifest.json"
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid JSON in {manifest_path}") from e
        if isinstance(m, dict) and m.get("stage") not in {"S3", None}:
            raise AssertionError(f"stationary_root manifest stage must be 'S3' (got {m.get('stage')!r})")

    csvs = sorted([p for p in stationary_root.glob("*.csv") if p.name.lower() != "manifest.json"])
    if not csvs:
        raise FileNotFoundError(f"No stationary macro CSVs found under: {stationary_root}")

    frames: list[pd.DataFrame] = []
    inputs: dict[str, dict[str, Any]] = {}
    for p in csvs:
        df = pd.read_csv(p)
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        df = df.sort_values("Date", kind="mergesort").set_index("Date")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
        frames.append(df)
        inputs[p.name] = {"path": str(p.resolve()), "sha256": sha256_file(p)}

    if not frames:
        raise ValueError("No usable stationary macro CSVs with a Date column")

    macro = pd.concat(frames, axis=1)
    dupes = macro.columns[macro.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate macro columns across stationary CSVs: {sorted(set(map(str, dupes)))}")
    macro = macro.astype("float64")

    c = _cfg(config)
    driver_map = c.get("driver_map")
    if isinstance(driver_map, dict) and driver_map:
        renamed: dict[str, str] = {}
        for target, src in driver_map.items():
            if isinstance(target, str) and isinstance(src, str) and src in macro.columns:
                renamed[src] = target
        if renamed:
            macro = macro.rename(columns=renamed)

    missing = [d for d in DRIVERS_REQUIRED if d not in macro.columns]
    if missing:
        raise ValueError(
            f"Missing required stationary macro drivers: {missing}. "
            f"Available columns: {sorted(map(str, macro.columns))}. "
            "Provide correlations.driver_map in config if needed."
        )

    macro = macro[DRIVERS_REQUIRED].copy()
    return macro, inputs

def _build_id(config_bytes: bytes, inputs: dict[str, dict[str, Any]], upstream_sha: str) -> str:
    h = hashlib.sha256()
    h.update(b"S5")
    h.update(config_bytes)
    h.update(upstream_sha.encode("utf-8"))
    for k in sorted(inputs.keys()):
        h.update(k.encode("utf-8"))
        h.update(str(inputs[k]["sha256"]).encode("utf-8"))
    return h.hexdigest()

def _parse_export_range(cfg: dict[str, Any]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_raw = cfg.get("export_date_start")
    end_raw = cfg.get("export_date_end")
    start = pd.Timestamp(start_raw).normalize() if isinstance(start_raw, str) and start_raw else None
    end = pd.Timestamp(end_raw).normalize() if isinstance(end_raw, str) and end_raw else None
    return start, end

def _filter_calendar(cal: pd.DatetimeIndex, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DatetimeIndex:
    out = cal
    if start is not None:
        out = out[out >= start]
    if end is not None:
        out = out[out <= end]
    return pd.DatetimeIndex(out, name="Date")

def build_stage(
    *,
    config: dict[str, Any],
    config_path: Path,
    input_root: Path,
    stationary_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    stationary_root = stationary_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = _cfg(config)
    spdr, spy, cal_full, pit_inputs = _read_aligned_prices(input_root)
    macro, macro_inputs = _load_stationary_macro(stationary_root, config)

    export_start, export_end = _parse_export_range(cfg)
    cal = _filter_calendar(cal_full, export_start, export_end)
    if len(cal) == 0:
        raise ValueError("Empty export calendar")

    spdr = spdr.loc[cal]
    spy = spy.loc[cal]
    macro = macro.reindex(cal)

    windows_raw = cfg.get("windows")
    if isinstance(windows_raw, list) and windows_raw:
        windows = [int(x) for x in windows_raw]
    else:
        windows = [5, 10, 21, 63, 126, 252]

    mp_raw = cfg.get("min_periods_by_window")
    min_periods_by_window: dict[int, int] = {}
    if isinstance(mp_raw, dict):
        for k, v in mp_raw.items():
            try:
                min_periods_by_window[int(k)] = int(v)
            except Exception:
                continue

    ffill_limits_raw = cfg.get("ffill_limits")
    ffill_limits: dict[str, int] | None = None
    if isinstance(ffill_limits_raw, dict):
        ffill_limits = {}
        for k, v in ffill_limits_raw.items():
            if isinstance(k, str):
                try:
                    ffill_limits[k] = int(v)
                except Exception:
                    continue

    default_ffill = int(cfg.get("default_macro_ffill_limit", 5))
    shift_by_1 = bool(cfg.get("shift_by_1", False))

    created_at_utc = datetime.now(timezone.utc).isoformat()

    config_bytes = config_path.read_bytes()
    upstream_sha = pit_inputs.get("manifest.json", {}).get("sha256", "")

    s3_manifest_path = stationary_root / "manifest.json"
    s3_sha = sha256_file(s3_manifest_path) if s3_manifest_path.exists() else None
    upstream_sha_combined = upstream_sha + ("|" + s3_sha if s3_sha else "")

    inputs = {f"pit/{k}": v for k, v in pit_inputs.items()} | {f"macro/{k}": v for k, v in macro_inputs.items()}
    build_id = _build_id(config_bytes, inputs, upstream_sha_combined)

    ffill_limits_raw = cfg.get("ffill_limits")
    allow_defaulted_ffill = bool(cfg.get("allow_default_ffill_limits", False))
    if isinstance(ffill_limits_raw, dict):
        ffill_limits = {str(k): int(v) for k, v in ffill_limits_raw.items() if isinstance(k, str)}
    else:
        if not allow_defaulted_ffill:
            raise ValueError(
                "Missing correlations.ffill_limits. Provide per-driver limits "
                "(e.g. {'rates':5,'oil':5,'usd':5,'bonds':5,'vix':5,'hy':5,'gold':5})."
            )
        ffill_limits = {d: default_ffill for d in DRIVERS_REQUIRED}

    panel, ffill_used = compute_corr_features(
        spdr_prices=spdr,
        macro_drivers=macro,
        trading_calendar=cal,
        windows=windows,
        min_periods_by_window=min_periods_by_window,
        ffill_limits=ffill_limits,
        default_macro_ffill_limit=default_ffill,
        shift_by_1=shift_by_1,
    )

    out_path = output_root / "corr_features.csv"
    stable_write_csv(panel, out_path)

    outputs = {
        "corr_features.csv": {
            "sha256": sha256_file(out_path),
            "schema": schema_snapshot(panel),
            "nan_rates": nan_rates(panel),
            "date_coverage": {
                "start": pd.Timestamp(cal.min()).date().isoformat(),
                "end": pd.Timestamp(cal.max()).date().isoformat(),
                "n_dates": int(len(cal)),
            },
        }
    }

    features = [c for c in panel.columns if c not in {"Date", "Sector"}]
    manifest: dict[str, Any] = {
        "build_id": build_id,
        "created_at_utc": created_at_utc,
        "stage": "S5",
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "output_root": str(output_root),
        "input_roots": {"aligned_pit": str(input_root), "stationary": str(stationary_root)},
        "upstream": {
            "s1_manifest_sha256": upstream_sha or None,
            "s3_manifest_sha256": s3_sha,
        },
        "canonical_calendar": {
            "name": "SPY",
            "start": pd.Timestamp(cal.min()).date().isoformat(),
            "end": pd.Timestamp(cal.max()).date().isoformat(),
            "n_dates": int(len(cal)),
        },
        "inputs": inputs,
        "ffill_policy_used": dict(ffill_used),
        "shift_by_1": shift_by_1,
        "windows": [int(x) for x in windows],
        "min_periods_by_window": {str(k): int(v) for k, v in sorted(min_periods_by_window.items())},
        "features": sorted(features),
        "outputs": outputs,
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    log.info("S5 complete: wrote %s", out_path)
    return manifest

def main(
    config_path: str | Path,
    *,
    input_root: str | Path | None = None,
    stationary_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    cfg_path = Path(config_path)
    config = _load_config(cfg_path)
    base_dir = cfg_path.resolve().parent
    in_root = _select_input_root(Path(input_root) if input_root else None, config, base_dir=base_dir)
    st_root = _select_stationary_root(Path(stationary_root) if stationary_root else None, config, base_dir=base_dir)
    out_root = _select_output_root(Path(output_root) if output_root else None, config, base_dir=base_dir)
    return build_stage(
        config=config,
        config_path=cfg_path,
        input_root=in_root,
        stationary_root=st_root,
        output_root=out_root,
    )