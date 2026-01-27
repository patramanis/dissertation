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
from .indicators import TechHorizonSpec, build_wide_features_for_horizon
from .ranks import RankConfig, add_rank_columns, assert_group_size

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

def _tech_cfg(config: dict[str, Any]) -> dict[str, Any]:
    c = config.get("tech")
    return c if isinstance(c, dict) else {}

def _select_input_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit
    cfg = _tech_cfg(config)
    v = cfg.get("input_root")
    if isinstance(v, str) and v:
        return _resolve_config_path(Path(v), base_dir)
    ws = _workspace_root()
    return ws / "data" / "interim" / "aligned_pit"

def _select_output_root(explicit: Path | None, config: dict[str, Any], *, base_dir: Path) -> Path:
    if explicit is not None:
        return explicit
    cfg = _tech_cfg(config)
    v = cfg.get("output_root")
    if isinstance(v, str) and v:
        return _resolve_config_path(Path(v), base_dir)
    ws = _workspace_root()
    return ws / "data" / "features" / "technical_indicators"

def _parse_horizons(tech_cfg: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    default_horizons = [5, 10, 21, 63, 126, 252]
    hcfg = tech_cfg.get("horizons")

    if isinstance(hcfg, list):
        hs: list[int] = []
        for x in hcfg:
            try:
                hs.append(int(x))
            except Exception:
                continue
        hs = sorted(set(hs))
        return (hs if hs else default_horizons), {}

    if isinstance(hcfg, dict) and hcfg:
        parsed: list[int] = []
        for k in hcfg.keys():
            try:
                parsed.append(int(k))
            except Exception:
                continue
        hs = sorted(set(parsed))
        return (hs if hs else default_horizons), hcfg

    return default_horizons, {}

def _read_aligned_prices(input_root: Path) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, dict[str, str], Path]:
    spdr_path = input_root / "SPDR.csv"
    spy_path = input_root / "SPY.csv"
    s1_manifest_path = input_root / "manifest.json"

    if not spdr_path.exists() or not spy_path.exists() or not s1_manifest_path.exists():
        raise FileNotFoundError("S4 requires SPDR.csv, SPY.csv, and manifest.json under input_root")

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

    input_hashes = {
        "SPDR.csv": sha256_file(spdr_path),
        "SPY.csv": sha256_file(spy_path),
        "manifest.json": sha256_file(s1_manifest_path),
    }

    return spdr, spy["SPY"].rename("SPY"), cal, input_hashes, s1_manifest_path

def _build_id(config_bytes: bytes, input_hashes: dict[str, str], upstream_manifest_sha256: str) -> str:
    h = hashlib.sha256()
    h.update(b"S4")
    h.update(config_bytes)
    h.update(upstream_manifest_sha256.encode("utf-8"))
    for k in sorted(input_hashes):
        h.update(k.encode("utf-8"))
        h.update(input_hashes[k].encode("utf-8"))
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

def _panel_base(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([calendar, list(SECTORS)], names=["Date", "Sector"])
    base = pd.DataFrame(index=idx)
    return base

def _assert_min_non_nan_sectors(
    panel: pd.DataFrame,
    *,
    feature: str,
    min_sectors: int,
) -> None:
    if feature not in panel.columns:
        raise ValueError(f"Missing core feature for coverage check: {feature}")

    df = panel.reset_index()
    counts = df.groupby("Date", sort=False)[feature].apply(lambda s: int(s.notna().sum()))
    first_idx = next((i for i, v in enumerate(counts.to_list()) if v > 0), None)
    if first_idx is None:
        raise AssertionError(f"Core feature {feature} is all-NaN")
    tail = counts.iloc[first_idx:]
    bad = tail[tail < min_sectors]
    if len(bad) > 0:
        d0 = bad.index[0]
        raise AssertionError(
            f"Dead-sector check failed for {feature}: {int(bad.iloc[0])} non-NaN sectors at {pd.Timestamp(d0).date()} (min {min_sectors})"
        )

def _wide_to_panel_series(wide: pd.DataFrame, *, name: str, base_index: pd.MultiIndex) -> pd.Series[Any]:
    wide2 = wide.reindex(columns=list(SECTORS))
    try:
        s = wide2.stack(future_stack=True)
    except TypeError:
        s = wide2.stack(dropna=False)
    s.index = pd.MultiIndex.from_arrays([s.index.get_level_values(0), s.index.get_level_values(1)], names=["Date", "Sector"])
    s = s.reindex(base_index)
    s.name = name
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s

def build_stage(
    *,
    config: dict[str, Any],
    config_path: Path,
    input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tech_cfg = _tech_cfg(config)
    spdr, spy, cal_full, input_hashes, s1_manifest_path = _read_aligned_prices(input_root)

    export_start, export_end = _parse_export_range(tech_cfg)
    cal = _filter_calendar(cal_full, export_start, export_end)

    if len(cal) == 0:
        raise ValueError("Empty export calendar")

    spdr = spdr.loc[cal]
    spy = spy.loc[cal]

    config_bytes = config_path.read_bytes()
    upstream_sha = input_hashes["manifest.json"]
    build_id = _build_id(config_bytes, input_hashes, upstream_sha)
    created_at_utc = datetime.now(timezone.utc).isoformat()

    horizons, horizons_cfg = _parse_horizons(tech_cfg)

    rank_method_str = str(tech_cfg.get("rank_method", "first")).lower()
    if rank_method_str not in ("average", "first"):
        rank_method_str = "first"
    rank_cfg = RankConfig(method=rank_method_str)  # type: ignore[arg-type]

    horizon_meta: dict[str, Any] = {}
    outputs_meta: dict[str, Any] = {}
    output_hashes: dict[str, str] = {}

    for h in horizons:
        h_str = str(int(h))
        hc_raw = horizons_cfg.get(h_str) if isinstance(horizons_cfg, dict) else None
        hc = hc_raw if isinstance(hc_raw, dict) else {}

        window = int(hc.get("window", h))
        eps = float(hc.get("eps", 1e-12))
        annualize_vol = bool(hc.get("annualize_vol", False))
        feature_set = str(hc.get("feature_set", "default"))

        rank_features = hc.get("rank_features")
        if not isinstance(rank_features, list) or not rank_features:
            rank_features = ["relmom", "ratio_z", "beta", "corr", "vol", "idio_vol", "mdd", "semi"]
        rank_features = [str(x) for x in rank_features]

        core_feature = str(hc.get("core_feature", "relmom"))
        min_non_nan_sectors = int(hc.get("min_non_nan_sectors", 9))

        spec = TechHorizonSpec(horizon=int(h), window=window, eps=eps, annualize_vol=annualize_vol)
        wide = build_wide_features_for_horizon(spec, spdr, spy, feature_set=feature_set)

        base = _panel_base(cal)
        panel = base
        feature_names = sorted(wide.keys())
        panel_index = panel.index
        assert isinstance(panel_index, pd.MultiIndex), "panel.index must be MultiIndex"
        for name in feature_names:
            panel[name] = _wide_to_panel_series(wide[name], name=name, base_index=panel_index)

        panel = add_rank_columns(panel, feature_cols=rank_features, cfg=rank_cfg)
        assert_group_size(panel, expected=9)

        _assert_min_non_nan_sectors(panel, feature=core_feature, min_sectors=min_non_nan_sectors)

        out_df = panel.reset_index()
        out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.normalize()
        out_df["Sector"] = pd.Categorical(out_df["Sector"].astype(str), categories=list(SECTORS), ordered=True)
        out_df = out_df.sort_values(["Date", "Sector"], kind="mergesort")
        out_df["Sector"] = out_df["Sector"].astype(str)

        value_cols = [c for c in out_df.columns if c not in {"Date", "Sector"}]
        out_df = out_df[["Date", "Sector", *sorted(value_cols)]]

        out_path = output_root / f"tech_h{int(h)}.csv"
        stable_write_csv(out_df, out_path)
        rel = out_path.relative_to(output_root).as_posix()
        output_hashes[rel] = sha256_file(out_path)

        horizon_meta[h_str] = {
            "spec": {
                "horizon": int(h),
                "window": window,
                "eps": eps,
                "annualize_vol": annualize_vol,
                "feature_set": feature_set,
            },
            "features": feature_names,
            "rank_features": rank_features,
        }

        outputs_meta[rel] = {
            "schema": schema_snapshot(out_df),
            "nan_rates": nan_rates(out_df),
            "date_coverage": {
                "start": pd.Timestamp(cal.min()).date().isoformat(),
                "end": pd.Timestamp(cal.max()).date().isoformat(),
                "n_dates": int(len(cal)),
            },
        }

    canonical_calendar = {
        "name": "SPY",
        "start": pd.Timestamp(cal.min()).date().isoformat(),
        "end": pd.Timestamp(cal.max()).date().isoformat(),
        "n_dates": int(len(cal)),
    }

    manifest_core: dict[str, Any] = {
        "build_id": build_id,
        "created_at_utc": created_at_utc,
        "stage": "S4",
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "upstream": {
            "s1_manifest_path": str(s1_manifest_path.resolve()),
            "s1_manifest_sha256": upstream_sha,
        },
        "input_root": str(input_root),
        "output_root": str(output_root),
        "canonical_calendar": canonical_calendar,
        "input_hashes": dict(input_hashes),
        "output_hashes": dict(output_hashes),
        "horizons": horizon_meta,
        "outputs": outputs_meta,
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_core, indent=2, sort_keys=True), encoding="utf-8")

    manifest_core["manifest_sha256"] = sha256_file(manifest_path)
    manifest_core["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(manifest_core, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest_core, indent=2, sort_keys=True), encoding="utf-8")

    log.info("S4 complete: wrote %d horizons to %s", len(horizon_meta), output_root)
    return manifest_core

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