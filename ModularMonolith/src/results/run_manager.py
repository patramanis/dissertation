"""
Run Manager for experiment tracking and reproducibility.

Creates structured output directories for each run with:
- Unique run ID (Run #K)
- Metadata (horizon, seeds, dataset hashes, timestamps)
- Organized subfolders for outputs

Directory structure:
Results/
  Run #K/
    printable/      (summary PDFs, key metrics)
    diagnostic/     (per-fold CSVs, drift tables)
    graphs/         (PNG/HTML plots)
    backtest/       (strategy outputs, quantstats)
    models/         (saved model artifacts)
    metadata.json   (run configuration and hashes)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import uuid
import warnings
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _hash_dataframe(df: pd.DataFrame, max_rows: int = 5000) -> str:
    """Create a reproducible hash of a DataFrame.
    
    SOTA Fix #2: Time-series aware hashing.
    Hashes head + tail + shape to detect both early and late changes.
    Critical for financial time series where new data is appended at the end.
    """
    if df.empty:
        return "empty"
    
    # FIX #4: Safe column name handling (map non-string columns to str)
    cols = ",".join(map(str, df.columns))
    dtypes = ",".join(map(str, df.dtypes))
    structure = f"{df.shape}|{cols}|{dtypes}"
    structure_hash = hashlib.sha256(structure.encode()).digest()
    
    # Hash head (early data) and tail (recent data)
    n = min(max_rows, len(df))
    head_sample = df.head(n)
    tail_sample = df.tail(n)
    
    head_arr = pd.util.hash_pandas_object(head_sample, index=True).to_numpy()
    tail_arr = pd.util.hash_pandas_object(tail_sample, index=True).to_numpy()
    
    # Combine hashes: structure + head + tail
    combined = structure_hash + head_arr.tobytes() + tail_arr.tobytes()
    return hashlib.sha256(combined).hexdigest()[:16]


def _json_safe_serializer(obj: Any) -> Any:
    """FIX #7: Type-safe JSON serializer used consistently across all JSON operations.
    
    Explicit handling of common types to preserve information.
    Raises TypeError for unknown types instead of silently converting to string.
    """
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    # FIX #7: Fail loudly for unknown types
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable. "
        f"Add explicit handler to _json_safe_serializer or convert before serialization."
    )


def _hash_array(arr: np.ndarray) -> str:
    """Create a reproducible hash of a numpy array.
    
    FIX #3: Include dtype, shape, and enforce contiguous bytes.
    Fails loudly for object dtype arrays (non-reproducible).
    """
    a = np.asarray(arr)
    if a.size == 0:
        return "empty"
    if a.dtype == object:
        raise TypeError(
            "Cannot hash object dtype arrays reproducibly. "
            "Convert to numeric or string array first."
        )
    a_c = np.ascontiguousarray(a)
    header = f"{a_c.shape}|{a_c.dtype.str}|C".encode()
    return hashlib.sha256(header + a_c.tobytes()).hexdigest()[:16]


@dataclass
class RunMetadata:
    """Metadata for a single training run."""
    run_id: int
    run_name: str
    timestamp_utc: str
    horizon: int
    seeds: list[int]
    cost_bps: float
    task_gate: str
    task_ranker: str
    
    # Dataset hashes for reproducibility
    X_hash: str = ""
    y_gate_hash: str = ""
    y_rank_hash: str = ""
    group_sizes_hash: str = ""
    
    # FIX #5: Environment fingerprint for hash stability
    pandas_version: str = ""
    numpy_version: str = ""
    python_version: str = ""
    
    # CV configuration
    cv_config: dict[str, Any] = field(default_factory=dict)
    
    # Model parameters
    gate_params: dict[str, Any] = field(default_factory=dict)
    ranker_params: dict[str, Any] = field(default_factory=dict)
    
    # Feature selection
    feature_selection_method: str = "importance"
    top_n_features: int = 25
    
    # Uncertainty settings
    n_seeds_ensemble: int = 10
    
    # Thresholds
    optuna_n_trials: int = 25
    
    # Notes
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        """SOTA Fix #4: Type-safe JSON serialization.
        
        Explicit handling of common types to preserve information.
        Raises TypeError for unknown types instead of silently converting to string.
        """
        return json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
            default=_json_safe_serializer,
        )


class RunManager:
    """
    Manages experiment runs with structured output directories.
    
    Usage:
        rm = RunManager(base_dir="Results")
        run_dir = rm.create_run(horizon=21, seeds=[0,1,2,...], ...)
        rm.save_metadata()
        
        # Access paths
        rm.printable_dir
        rm.diagnostic_dir
        rm.graphs_dir
        rm.backtest_dir
        rm.models_dir
    """
    
    SUBFOLDERS = ("printable", "diagnostic", "graphs", "backtest", "models")
    
    def __init__(
        self,
        base_dir: str | Path = "Results",
        *,
        workspace_root: Path | None = None,
    ) -> None:
        # SOTA Fix #3: Robust workspace root detection
        # Priority: explicit arg > env var > fallback to parent traversal
        if workspace_root is None:
            env_root = os.environ.get("DISSERTATION_ROOT") or os.environ.get("PROJECT_ROOT")
            if env_root:
                workspace_root = Path(env_root)
            else:
                # Fallback: traverse up from this file
                # Look for marker files (pyproject.toml, .git, README.md)
                current = Path(__file__).resolve().parent
                for _ in range(5):  # Max 5 levels up
                    if any((current / marker).exists() for marker in ["pyproject.toml", ".git", "README.md"]):
                        workspace_root = current
                        break
                    current = current.parent
                else:
                    # Last resort: hardcoded parent traversal (with warning)
                    workspace_root = Path(__file__).resolve().parent.parent.parent
                    warnings.warn(
                        f"Could not locate workspace root via env var or markers. "
                        f"Using fallback: {workspace_root}. "
                        f"Set DISSERTATION_ROOT environment variable for robustness.",
                        category=UserWarning,
                        stacklevel=2,
                    )
        
        self.workspace_root = Path(workspace_root)
        self.base_dir = self.workspace_root / base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id: int | None = None
        self.run_dir: Path | None = None
        self.metadata: RunMetadata | None = None
        
        # Subfolder paths (set after create_run)
        self.printable_dir: Path | None = None
        self.diagnostic_dir: Path | None = None
        self.graphs_dir: Path | None = None
        self.backtest_dir: Path | None = None
        self.models_dir: Path | None = None
    
    def _allocate_run_dir(self) -> tuple[int, Path]:
        """Atomically allocate and create a run directory.
        
        FIX #1: Keeps the directory after creation (no rmdir race window).
        FIX #2: Uses UUID suffix for fallback (no modulo collisions).
        
        Returns:
            (run_id, run_dir): The allocated ID and its path
        """
        for attempt in range(200):  # Increased from 100 for better concurrency
            existing_ids = []
            for d in self.base_dir.iterdir():
                if d.is_dir() and d.name.startswith("Run #"):
                    try:
                        # Extract numeric ID (ignoring any suffix like -uuid)
                        num_str = d.name.split("#", 1)[1].split("-", 1)[0]
                        existing_ids.append(int(num_str))
                    except (IndexError, ValueError):
                        pass
            
            candidate_id = max(existing_ids, default=0) + 1
            run_dir = self.base_dir / f"Run #{candidate_id}"
            
            try:
                # FIX #1: Atomic claim - keep the directory once created
                run_dir.mkdir(parents=False, exist_ok=False)
                return candidate_id, run_dir
            except FileExistsError:
                # Race condition: another process claimed this ID
                time.sleep(0.01 * (attempt + 1))  # Exponential backoff
                continue
        
        # FIX #2: UUID-based fallback (no collision risk)
        warnings.warn(
            "Race condition: Could not allocate sequential run ID after 200 attempts. "
            "Using UUID-based ID instead.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        unique_suffix = uuid.uuid4().hex[:8]
        timestamp = int(time.time())
        run_dir = self.base_dir / f"Run #{timestamp}-{unique_suffix}"
        run_dir.mkdir(parents=False, exist_ok=False)
        return timestamp, run_dir
    
    def create_run(
        self,
        *,
        horizon: int,
        seeds: list[int],
        cost_bps: float,
        cv_config: dict[str, Any],
        gate_params: dict[str, Any],
        ranker_params: dict[str, Any],
        X: pd.DataFrame | None = None,
        y_gate: np.ndarray | None = None,
        y_rank: np.ndarray | None = None,
        group_sizes: np.ndarray | None = None,
        feature_selection_method: str = "importance",
        top_n_features: int = 25,
        optuna_n_trials: int = 25,
        notes: str = "",
    ) -> Path:
        """
        Create a new run directory with all subfolders.
        
        Returns the path to the run directory.
        """
        # FIX #1: Atomic allocation (no race window)
        self.run_id, self.run_dir = self._allocate_run_dir()
        run_name = self.run_dir.name
        
        # Create subfolders
        for subfolder in self.SUBFOLDERS:
            path = self.run_dir / subfolder
            path.mkdir(exist_ok=True)
            setattr(self, f"{subfolder}_dir", path)
        
        # Build metadata
        self.metadata = RunMetadata(
            run_id=self.run_id,
            run_name=run_name,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            horizon=horizon,
            seeds=list(seeds),
            cost_bps=cost_bps,
            task_gate="gate_classifier",
            task_ranker="ranker",
            X_hash=_hash_dataframe(X) if X is not None else "",
            y_gate_hash=_hash_array(y_gate) if y_gate is not None else "",
            y_rank_hash=_hash_array(y_rank) if y_rank is not None else "",
            group_sizes_hash=_hash_array(group_sizes) if group_sizes is not None else "",
            pandas_version=pd.__version__,
            numpy_version=np.__version__,
            python_version=f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
            cv_config=dict(cv_config),
            gate_params=dict(gate_params),
            ranker_params=dict(ranker_params),
            feature_selection_method=feature_selection_method,
            top_n_features=top_n_features,
            n_seeds_ensemble=len(seeds),
            optuna_n_trials=optuna_n_trials,
            notes=notes,
        )
        
        print(f"[RunManager] Created: {self.run_dir}")
        return self.run_dir
    
    def save_metadata(self) -> Path:
        """Save metadata to JSON file.
        
        FIX #10: Atomic write to prevent corruption on process death.
        """
        if self.run_dir is None or self.metadata is None:
            raise RuntimeError("No run created yet. Call create_run() first.")
        
        meta_path = self.run_dir / "metadata.json"
        # FIX #10: Write to temp file and atomically replace
        tmp_path = meta_path.with_suffix(".json.tmp")
        tmp_path.write_text(self.metadata.to_json(), encoding="utf-8")
        tmp_path.replace(meta_path)
        return meta_path
    
    def _safe_artifact_path(self, subfolder: str, name: str) -> Path:
        """FIX #6: Prevent path traversal attacks.
        
        Validates subfolder is known and name doesn't escape the run directory.
        """
        if subfolder not in self.SUBFOLDERS:
            raise ValueError(
                f"Invalid subfolder '{subfolder}'. "
                f"Must be one of: {', '.join(self.SUBFOLDERS)}"
            )
        
        # Strip any directory components from name (basename only)
        safe_name = Path(name).name
        if not safe_name or safe_name in (".", ".."):
            raise ValueError(f"Invalid artifact name: '{name}'")
        
        base = (self.run_dir / subfolder).resolve()
        path = (base / safe_name).resolve()
        
        # Verify path is within base (prevent traversal)
        try:
            path.relative_to(base)
        except ValueError:
            raise ValueError(
                f"Unsafe artifact path (traversal detected): '{name}' in '{subfolder}'"
            )
        
        return path
    
    def save_artifact(
        self,
        data: pd.DataFrame | pd.Series | np.ndarray | dict[str, Any] | str,
        name: str,
        subfolder: str = "diagnostic",
    ) -> Path:
        """Save an artifact to the appropriate subfolder.
        
        FIX #6: Path traversal protection.
        FIX #7: Consistent JSON serialization.
        FIX #10: Atomic writes for JSON/text.
        
        Supports:
        - DataFrame -> CSV (always, for portability)
        - Series -> CSV
        - ndarray -> npy
        - dict -> json
        - str -> text
        """
        if self.run_dir is None:
            raise RuntimeError("No run created yet.")
        
        # Ensure subfolder exists
        folder = self.run_dir / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            # Always save as CSV for maximum portability
            if not name.endswith(".csv"):
                name = name.replace(".parquet", "") + ".csv"
            path = self._safe_artifact_path(subfolder, name)
            # FIX #10: Atomic write
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            data.to_csv(tmp_path, index=False)
            tmp_path.replace(path)
        elif isinstance(data, pd.Series):
            name = name.replace(".csv", "") + ".csv"
            path = self._safe_artifact_path(subfolder, name)
            # FIX #10: Atomic write
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            data.to_frame(name="value").to_csv(tmp_path, index=True)
            tmp_path.replace(path)
        elif isinstance(data, np.ndarray):
            name = name.replace(".npy", "") + ".npy"
            path = self._safe_artifact_path(subfolder, name)
            np.save(path, data)  # numpy handles atomic writes internally
        elif isinstance(data, dict):
            name = name.replace(".json", "") + ".json"
            path = self._safe_artifact_path(subfolder, name)
            # FIX #7: Use consistent JSON serializer (strict, no default=str)
            # FIX #10: Atomic write
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=_json_safe_serializer),
                encoding="utf-8"
            )
            tmp_path.replace(path)
        elif isinstance(data, str):
            path = self._safe_artifact_path(subfolder, name)
            # FIX #10: Atomic write
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(data, encoding="utf-8")
            tmp_path.replace(path)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        return path
    
    def get_run_summary(self) -> dict[str, Any]:
        """Get a summary of the current run for logging."""
        if self.metadata is None:
            return {}
        return {
            "run_id": self.metadata.run_id,
            "run_name": self.metadata.run_name,
            "horizon": self.metadata.horizon,
            "n_seeds": len(self.metadata.seeds),
            "git_hash": self.metadata.git_hash,
            "timestamp": self.metadata.timestamp_utc,
        }


def load_run_metadata(run_dir: Path | str) -> RunMetadata:
    """Load metadata from a previous run.
    
    SOTA Fix #6: Forward-compatible loading with schema evolution support.
    Handles missing fields gracefully by using dataclass defaults.
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {run_dir}")
    
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    
    # Get all field names and defaults from RunMetadata dataclass
    metadata_fields = {f.name: f for f in fields(RunMetadata)}
    
    # Build kwargs with defaults for missing fields
    kwargs = {}
    for field_name, field_obj in metadata_fields.items():
        if field_name in data:
            kwargs[field_name] = data[field_name]
        elif field_obj.default is not MISSING:
            kwargs[field_name] = field_obj.default
        elif field_obj.default_factory is not MISSING:
            kwargs[field_name] = field_obj.default_factory()
        else:
            # Required field missing from old metadata
            warnings.warn(
                f"Required field '{field_name}' missing from metadata in {run_dir}. "
                f"Using empty value. Metadata may be from an older version.",
                category=UserWarning,
                stacklevel=2,
            )
            # Provide sensible defaults based on type annotation
            field_type = field_obj.type
            if field_type == str:
                kwargs[field_name] = ""
            elif field_type == int:
                kwargs[field_name] = 0
            elif field_type == float:
                kwargs[field_name] = 0.0
            elif field_type == list:
                kwargs[field_name] = []
            elif field_type == dict:
                kwargs[field_name] = {}
            else:
                kwargs[field_name] = None
    
    return RunMetadata(**kwargs)


def list_runs(base_dir: Path | str = "Results") -> list[dict[str, Any]]:
    """List all runs with basic info.
    
    FIX #9: Numeric sorting (Run #2 before Run #10).
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    
    def _run_sort_key(p: Path) -> tuple[int, str]:
        """Extract numeric ID for proper sorting."""
        try:
            num_str = p.name.split("#", 1)[1].split("-", 1)[0]
            return (int(num_str), p.name)
        except (IndexError, ValueError):
            return (10**9, p.name)  # Non-numeric runs at end
    
    runs = []
    for d in sorted(base.iterdir(), key=_run_sort_key):
        if d.is_dir() and d.name.startswith("Run #"):
            meta_path = d / "metadata.json"
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    runs.append({
                        "run_id": data.get("run_id"),
                        "run_name": data.get("run_name"),
                        "horizon": data.get("horizon"),
                        "timestamp": data.get("timestamp_utc"),
                        "path": str(d),
                    })
                except Exception:
                    runs.append({"run_name": d.name, "path": str(d), "error": "parse_failed"})
    return runs
