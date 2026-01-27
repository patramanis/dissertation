from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = (THIS_DIR.parent / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features.stationary.build_base_stationary import main

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage S3: Build base stationary features from PIT-aligned CSVs")
    p.add_argument("config", required=True, help="Path to JSON/YAML config")
    p.add_argument("input-root", default=None, help="Override S1 aligned PIT root (data/interim/aligned_pit)")
    p.add_argument("output-root", default=None, help="Override output root (data/features/base_stationary)")
    p.add_argument("log-level", default="INFO", help="Logging level")
    return p.parse_args()

def cli() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    main(
        Path(args.config),
        input_root=Path(args.input_root) if args.input_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
    )

if __name__ == "__main__":
    cli()