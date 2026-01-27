from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_common import run_all_horizons

def main():
    results = run_all_horizons(
        category="test",
        filter_type="trend_vix",
        num_folds=5,
        horizons=[5, 21, 63],
        config="autogluon_medium",
    )
    
    print("TREND + VIX TEST COMPLETE")
    for h, metrics in results.items():
        if "error" in metrics:
            print(f"h={h}: ERROR - {metrics['error']}")
        else:
            ext = metrics.get("external_validation", {})
            sharpe = ext.get("vectorbt_sharpe", metrics.get("sharpe_ratio_mean", 0))
            print(f"h={h}: Sharpe={sharpe:.2f}")
    
    return results


if __name__ == "__main__":
    main()