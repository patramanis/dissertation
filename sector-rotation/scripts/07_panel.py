from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = (THIS_DIR.parent / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset.panel_builder import PanelConfig, PanelPaths, build_panel

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage S7: Build model panels (panel_h{h}.csv)")
    p.add_argument("config", required=True, help="Path to JSON/YAML config (currently unused; kept for parity)")
    p.add_argument(
        "horizons",
        default="5,21,63",
        help="Comma-separated horizons to build (default: 5,21,63)",
    )
    p.add_argument("output-root", default=None, help="Override output root (default: data/processed/panel)")
    p.add_argument("log-level", default="INFO", help="Logging level")
    return p.parse_args()

def cli() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    ws = THIS_DIR.parent
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]

    for h in horizons:
        paths = PanelPaths.defaults(workspace_root=ws, horizon=int(h))
        out_dir = Path(args.output_root).resolve() if args.output_root else paths.output_dir
        build_panel(int(h), PanelPaths(**{**paths.__dict__, "output_dir": out_dir}), PanelConfig(stage="S7_PANEL"))

if __name__ == "__main__":
    cli()