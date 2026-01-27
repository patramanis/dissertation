from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = (THIS_DIR.parent / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from cv.splits import UnifiedCVSplit, CVFold, save_cv_folds

log = logging.getLogger(__name__)

CVMode = Literal["rolling"]

def _workspace_root() -> Path:
    return THIS_DIR.parent


def generate_folds_for_cv_type(
    panel: pd.DataFrame,
    horizon: int,
    cv_type: CVMode = "rolling",
    min_train_years: int = 5,
    test_months: int = 12,
    step_months: int = 12,
    purge_buffer: int = 5,
    valid_fraction: float = 0.15,
) -> list[CVFold]:
    splitter = UnifiedCVSplit(
        horizon=horizon,
        purge_buffer=purge_buffer,
        min_train_years=min_train_years,
        test_months=test_months,
        step_months=step_months,
        valid_fraction=valid_fraction,
    )

    return list(splitter.split(panel, mode=cv_type))

def main(
    output_dir: Path | None = None,
    horizons: list[int] | None = None,
) -> dict:

    ws = _workspace_root()
    out = output_dir or (ws / "configs" / "cv")
    out.mkdir(parents=True, exist_ok=True)
    hs = horizons or [5, 21, 63]
    results = {}

    for h in hs:
        panel_path = ws / "data" / "processed" / "panel" / f"panel_h{h}.csv"

        if not panel_path.exists():
            log.warning("Panel not found: %s", panel_path)
            continue

        log.info("Loading panel for h=%d from %s", h, panel_path)
        panel = pd.read_csv(panel_path, usecols=["Date", "Sector"])
        panel["Date"] = pd.to_datetime(panel["Date"]).dt.normalize()

        n_unique_dates = panel["Date"].nunique()
        date_range = (panel["Date"].min(), panel["Date"].max())
        log.info("  Dates: %d unique, from %s to %s",
                 n_unique_dates, date_range[0].date(), date_range[1].date())

        purge_buffer = 5
        test_months = 12
        step_months = 6
        valid_fraction = 0.15

        log.info("Generating rolling folds (5yr train)...")
        rolling_folds = generate_folds_for_cv_type(
            panel, horizon=h, cv_type="rolling",
            min_train_years=5, test_months=test_months, step_months=step_months,
            purge_buffer=purge_buffer, valid_fraction=valid_fraction,
        )
        log.info("Generated %d rolling folds", len(rolling_folds))

        rolling_path = out / f"cv_rolling_h{h}.json"
        save_cv_folds(
            rolling_folds, rolling_path,
            cv_type="rolling", horizon=h,
            metadata={
                "min_train_years": 5,
                "test_months": test_months,
                "step_months": step_months,
                "valid_fraction": valid_fraction,
                "purge_buffer": purge_buffer,
                "gap_formula": f"gap = horizon({h}) + purge_buffer({purge_buffer}) = {h + purge_buffer}",
                "description": "Rolling window: fixed 5yr train window",
            }
        )

        results[h] = {
            "rolling": {
                "path": str(rolling_path),
                "n_folds": len(rolling_folds),
            },
        }

    summary_path = out / "cv_summary.json"
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cv_type": "rolling",
        "horizons": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log.info("CV summary written to %s", summary_path)

    return results

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CV fold configurations")
    p.add_argument("--output-dir", default=None, help="Output directory for CV configs")
    p.add_argument("--horizons", default="5,21,63", help="Comma-separated horizons")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    out_dir = Path(args.output_dir) if args.output_dir else None

    main(output_dir=out_dir, horizons=horizons)