from __future__ import annotations
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_common import run_all_horizons

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

if __name__ == "__main__":
    run_all_horizons(
        category="maximum",
        filter_type="regime_scalar",
        num_folds=39,
        config="autogluon_extreme",
        log_level="INFO",
    )