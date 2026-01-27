from __future__ import annotations
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_common import run_all_horizons

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    run_all_horizons(
        filter_type="regime_streak_decay",
        category="normal",
        num_folds=39,
        config="autogluon_medium",
    )