from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = (THIS_DIR.parent / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from labels.excess.build_labels import main

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage S4: Build excess-return labels")
    p.add_argument("config", required=True, help="Path to JSON/YAML config")
    p.add_argument("input-root", default=None, help="Override input root (expects prices_close.csv)")
    p.add_argument("output-root", default=None, help="Override output root (default: artifacts/labels/excess)")
    p.add_argument("log-level", default="INFO", help="Logging level")
    p.add_argument("diagnostic", action="store_true", help="Enable extra diagnostic logging")
    return p.parse_args()

def cli() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    main(
        Path(args.config),
        input_root=Path(args.input_root) if args.input_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
        diagnostic=bool(args.diagnostic),
    )

if __name__ == "__main__":
    cli()