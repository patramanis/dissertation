import os
import argparse
import json
import pickle
from dataclasses import dataclass
from typing import Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor, XGBClassifier

from feature_engineering import FeatureEngineer

RESULTS_DIR = os.path.join("results")
DATA_DIR = os.path.join("data")
SECTOR_DATA_PATH = os.path.join(DATA_DIR, "sector_data.csv")
FEATURES_DATA_PATH = os.path.join(DATA_DIR, "features_data.csv")

# ---------------------------- Utils ---------------------------- #

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    sector = pd.read_csv(SECTOR_DATA_PATH, parse_dates=["Date"])  # prices
    features = pd.read_csv(FEATURES_DATA_PATH, parse_dates=["Date"])  # market indicators
    return sector, features


def build_dataset(horizon: int, target: str = "SPY", task: str = "regression", 
                  use_engineered_features: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build X (features) and y (target) with temporal integrity.
    - X: uses FEATURES_DATA only (no sector forward-returns leakage)
    - y: future return of target over horizon days (log-return)
    
    Args:
        horizon: Prediction horizon (days)
        target: Target sector to predict
        task: "regression" or "classification"
        use_engineered_features: Whether to include engineered features (default: True)
    """
    sector, features = load_datasets()

    # Align by Date
    df = pd.merge(features, sector[["Date", target]], on="Date", how="inner")

    # Engineer new features if requested
    if use_engineered_features:
        # Remove Date for feature engineering
        features_only = df[[c for c in features.columns if c != "Date"]]
        sectors_only = sector[[c for c in sector.columns if c != "Date"]]
        
        # Align sectors to match dates in df
        sector_aligned = sectors_only.loc[df.index]
        
        engineer = FeatureEngineer(features_only, sector_aligned)
        df_engineered = engineer.engineer_all_features()
        
        # Replace features in df with engineered ones
        df = df[["Date", target]].copy()
        for col in df_engineered.columns:
            df[col] = df_engineered[col].values
    
    # Create returns for target (log-returns safer for regression)
    df = df.sort_values("Date").reset_index(drop=True)
    df["price"] = df[target]
    df["ret_h"] = np.log(df["price"].shift(-horizon) / df["price"])  # future log return

    # Drop rows with NaNs created by shift or fills
    feature_cols = [c for c in df.columns if c not in ["Date", target, "price", "ret_h"]]
    X = df[feature_cols].copy()
    y = df["ret_h"].copy()

    valid_idx = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    return X.reset_index(drop=True), y.reset_index(drop=True), feature_cols


@dataclass
class XGBConfig:
    task: str = "regression"  # or "classification"
    horizon: int = 10
    n_splits: int = 5
    early_stopping_rounds: int = 50
    random_state: int = 42
    # Core params (conservative to reduce overfit)
    n_estimators: int = 1000
    learning_rate: float = 0.03
    max_depth: int = 3
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_lambda: float = 3.0
    reg_alpha: float = 0.5


def get_model(cfg: XGBConfig):
    common = dict(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        random_state=cfg.random_state,
        tree_method="hist",
        n_jobs=0,
    )
    if cfg.task == "regression":
        return XGBRegressor(**common)
    else:
        # Binary classification: positive if future return > 0
        return XGBClassifier(**common, eval_metric="logloss")


def time_series_cv_train(X: pd.DataFrame, y: pd.Series, cfg: XGBConfig):
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)

    oof_preds = np.zeros_like(y, dtype=float)
    fold_metrics = []
    models = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner validation split (last 20% of train) to avoid test leakage
        val_size = max(1, int(0.2 * len(X_train)))
        X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
        y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

        model = get_model(cfg)
        if cfg.task == "regression":
            model.fit(
                X_tr,
                y_tr,
                verbose=False,
            )
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            metrics = {"fold": fold, "mae": float(mae), "r2": float(r2)}
        else:
            y_tr_bin = (y_tr > 0).astype(int)
            y_val_bin = (y_val > 0).astype(int)
            y_test_bin = (y_test > 0).astype(int)
            model.fit(
                X_tr,
                y_tr_bin,
                verbose=False,
            )
            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)
            # Basic metrics inline to avoid extra deps
            acc = (preds == y_test_bin).mean()
            metrics = {"fold": fold, "accuracy": float(acc)}

        fold_metrics.append(metrics)
        models.append(model)

        # store oof preds (regression only)
        if cfg.task == "regression":
            oof_preds[test_idx] = preds

    return models, fold_metrics, oof_preds if cfg.task == "regression" else None


def save_results(cfg: XGBConfig, feature_cols: List[str], fold_metrics, oof_preds, models, 
                 use_engineered: bool = True):
    """Save results with timestamped directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if cfg.task == "regression" and use_engineered:
        base_dir = os.path.join(RESULTS_DIR, "regression-baseline", "runs", timestamp)
    else:
        base_dir = os.path.join(RESULTS_DIR, "baseline", "runs", timestamp)
    
    summary_dir = os.path.join(base_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save configuration
    cfg_path = os.path.join(summary_dir, "xgb_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)

    # Save metrics
    metrics_path = os.path.join(summary_dir, "xgb_cv_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(fold_metrics, f, indent=2)

    # Save feature list
    features_path = os.path.join(summary_dir, "xgb_feature_list.json")
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save OOF predictions if regression
    if oof_preds is not None:
        oof_path = os.path.join(summary_dir, "xgb_oof_preds.csv")
        pd.DataFrame({"oof_pred": oof_preds}).to_csv(oof_path, index=False)

    # Save best model
    model_path = os.path.join(summary_dir, "xgb_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(models[-1], f)
    
    # Print summary statistics
    print("\n" + "="*70)
    print(f"Results saved to: {base_dir}")
    print("="*70)
    print(f"Configuration: {cfg.task} (horizon={cfg.horizon}d, splits={cfg.n_splits})")
    print(f"Features: {len(feature_cols)} total")
    print(f"Samples: {oof_preds.shape[0] if oof_preds is not None else 'N/A'}")
    
    if fold_metrics:
        print("\nCross-validation Results:")
        for metric in fold_metrics:
            if cfg.task == "regression":
                print(f"  Fold {metric['fold']}: MAE={metric['mae']:.6f}, R²={metric['r2']:.4f}")
            else:
                print(f"  Fold {metric['fold']}: Accuracy={metric['accuracy']:.4f}")
    
    print("\nFiles saved:")
    print(f"  ✓ {os.path.relpath(cfg_path)}")
    print(f"  ✓ {os.path.relpath(metrics_path)}")
    print(f"  ✓ {os.path.relpath(features_path)}")
    if oof_preds is not None:
        print(f"  ✓ {os.path.relpath(oof_path)}")
    print(f"  ✓ {os.path.relpath(model_path)}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost with TSCV and save metrics")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--engineered", action="store_true", default=True,
                        help="Use engineered features (default: True)")
    parser.add_argument("--no-engineered", dest="engineered", action="store_false",
                        help="Use only base features")
    args = parser.parse_args()

    cfg = XGBConfig(task=args.task, horizon=args.horizon, n_splits=args.splits)

    X, y, feature_cols = build_dataset(horizon=cfg.horizon, task=cfg.task, 
                                        use_engineered_features=args.engineered)
    models, fold_metrics, oof_preds = time_series_cv_train(X, y, cfg)
    save_results(cfg, feature_cols, fold_metrics, oof_preds, models, args.engineered)


if __name__ == "__main__":
    main()
