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

# Add ModularMonolith to path
sys.path.insert(0, str(Path(__file__).parent))

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
    print("  CV: Expanding window, annual training periods")
    print("=" * 80)
    print()
    
    result = train_dual_model(
        horizon=21,
        seeds=[42, 43, 44],
        optuna_n_trials=25,
        use_gpu=True,
    )
    
    print()
    print("=" * 80)
    print(f"✓ Training complete")
    print(f"  Results: {result.run_dir}")
    print(f"  Metrics: {result.run_dir / 'metrics.json'}")
    print(f"  Models: {result.run_dir / 'models' / 'final_models.pkl'}")
    print("=" * 80)
