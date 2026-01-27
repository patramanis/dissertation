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
        category="test",
        filter_type="dual",
        num_folds=5,
        config="autogluon_medium",
        log_level="INFO",
    )