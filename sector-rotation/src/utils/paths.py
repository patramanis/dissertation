from __future__ import annotations
from pathlib import Path

def find_workspace_root(start: Path) -> Path:
    start = start.resolve()

    markers = (
        ".git",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
    )

    for parent in (start, *start.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent

    return start.parent