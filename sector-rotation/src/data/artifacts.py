from __future__ import annotations
import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
from data.pit.mapper import nan_rates, schema_snapshot

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _normalize_date_col(df: pd.DataFrame, *, date_col: str = "Date") -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Missing {date_col} column")
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="raise")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    out[date_col] = dt.dt.normalize()
    return out

def read_csv(
    path: Path,
    *,
    required_cols: Iterable[str] | None = None,
    key_cols: list[str] | None = None,
    sort_keys: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    if required_cols is not None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {missing}")

    if "Date" in df.columns:
        df = _normalize_date_col(df, date_col="Date")

    if key_cols:
        if sort_keys:
            df = df.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
        dup = int(df.duplicated(subset=key_cols).sum())
        if dup:
            raise ValueError(f"{path.name} has duplicate keys on {key_cols} (n_dup={dup})")

    return df

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if "Date" in out.columns:
        out = _normalize_date_col(out, date_col="Date")

    out.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10g",
        lineterminator="\n",
    )

def _jsonable(x: Any) -> Any:
    if is_dataclass(x) and not isinstance(x, type):
        return asdict(x)
    if isinstance(x, Path):
        return str(x)
    return x

def git_hash(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        s = (r.stdout or "").strip()
        return s or None
    except Exception:
        return None

def build_manifest(
    *,
    stage: str,
    build_id: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    stats: dict[str, Any],
    config: Any,
    repo_root: Path | None = None,
    include_timestamp: bool = True,
) -> dict[str, Any]:
    man: dict[str, Any] = {
        "stage": str(stage),
        "build_id": str(build_id),
        "git_hash": git_hash(repo_root) if repo_root else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat() if include_timestamp else None,
        "inputs": inputs,
        "outputs": outputs,
        "stats": stats,
        "config": _jsonable(config),
    }
    return man

def describe_table(df: pd.DataFrame) -> dict[str, Any]:
    d: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "schema": schema_snapshot(df),
        "nan_rates": nan_rates(df),
    }
    if "Date" in df.columns and len(df):
        dt = pd.to_datetime(df["Date"]).dt.normalize()
        d["coverage"] = {
            "date_min": pd.Timestamp(dt.min()).date().isoformat(),
            "date_max": pd.Timestamp(dt.max()).date().isoformat(),
            "n_dates": int(dt.nunique()),
        }
    return d

def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2), encoding="utf-8")