#!/usr/bin/env python
"""
Training script for Dual-Stage Model (Gate Classifier + Ranker)
- Horizon: 21 trading days
- Optuna: 25 trials for hyperparameter tuning
- Seeds: 3 independent runs for ensemble
- Run allocation: Auto-continues from last Run # in Results/
"""
import sys
from pathlib import Path

# Add repo root to path so `import ModularMonolith...` works when executing
# this file directly (e.g., `python ModularMonolith/run_training_h21.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ModularMonolith.src.models.train_dual_system import train_dual_model

if __name__ == "__main__":
    print("=" * 80)
    print("TRAINING DUAL-STAGE MODEL")
    print("=" * 80)
    print("Configuration:")
    print("  Horizon: 21 trading days")
    print("  Optuna trials: 25 (TPE sampler)")
    print("  Seeds: [42, 43, 44]")
    print("  Run allocation: Auto-continue from last Run #")
    print("  Time-decay: λ=1260 days (5 years half-life)")
    print("  CV: Rolling window (train=5y, test=1y, step=1y) with purge_gap=h")
    print("=" * 80)
    print()
    
    result = train_dual_model(
        horizon=21,
        seeds=[42, 43, 44],
        optuna_n_trials=25,
        use_gpu=True,
        outer_cv_mode="rolling_5y1y",
    )
    
    print()
    print("=" * 80)
    print(f"✓ Training complete")
    print(f"  Results: {result.run_dir}")
    print(f"  Metrics: {result.run_dir / 'metrics.json'}")
    print(f"  Models: {result.run_dir / 'models' / 'final_models.pkl'}")
    print("=" * 80)
