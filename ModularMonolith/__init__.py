from __future__ import annotations

from datetime import datetime
from pathlib import Path

__version__ = "0.1.0"


def build_id(module_file: str) -> str:
	p = Path(module_file).resolve()
	try:
		mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
	except Exception:
		mtime = "unknown"
	return f"{p} (mtime={mtime})"
