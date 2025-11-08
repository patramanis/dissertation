"""
Feature backtest harness: validate each feature independently.
Train tiny models per-feature, report OOS metrics, rank features.
Detects if features have genuine signal vs. random noise.
"""
import os
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from feature_engineering import FeatureEngineer

RESULTS_DIR = os.path.join("results")
DATA_DIR = os.path.join("data")
SECTOR_DATA_PATH = os.path.join(DATA_DIR, "sector_data.csv")
FEATURES_DATA_PATH = os.path.join(DATA_DIR, "features_data.csv")


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    sector = pd.read_csv(SECTOR_DATA_PATH, parse_dates=["Date"])
    features = pd.read_csv(FEATURES_DATA_PATH, parse_dates=["Date"])
    return sector, features


def build_dataset(horizon: int, use_engineered_features: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    sector = pd.read_csv(SECTOR_DATA_PATH, parse_dates=["Date"])
    features = pd.read_csv(FEATURES_DATA_PATH, parse_dates=["Date"])
    
    df = pd.merge(features, sector[["Date", "SPY"]], on="Date", how="inner")
    
    # Engineer features if requested
    if use_engineered_features:
        features_only = df[[c for c in features.columns if c != "Date"]]
        sectors_only = sector[[c for c in sector.columns if c != "Date"]]
        sector_aligned = sectors_only.loc[df.index]
        
        engineer = FeatureEngineer(features_only, sector_aligned)
        df_engineered = engineer.engineer_all_features()
        
        df = df[["Date", "SPY"]].copy()
        for col in df_engineered.columns:
            df[col] = df_engineered[col].values
    
    df = df.sort_values("Date").reset_index(drop=True)
    df["price"] = df["SPY"]
    df["ret_h"] = np.log(df["price"].shift(-horizon) / df["price"])

    feature_cols = [c for c in df.columns if c not in ["Date", "SPY", "price", "ret_h"]]
    X = df[feature_cols].copy()
    y = df["ret_h"].copy()

    valid_idx = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    return X.reset_index(drop=True), y.reset_index(drop=True), feature_cols


def backtest_single_feature(feature_name: str, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
    """
    Train XGBoost using only one feature.
    Returns OOS metrics (MAE, R2) averaged across CV folds.
    """
    X_feat = X[[feature_name]].copy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for train_idx, test_idx in tscv.split(X_feat):
        X_train, X_test = X_feat.iloc[train_idx], X_feat.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner validation
        val_size = max(1, int(0.2 * len(X_train)))
        X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
        y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            n_jobs=0,
        )
        model.fit(X_tr, y_tr, verbose=False)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        fold_metrics.append({"mae": mae, "r2": r2})

    # Aggregate
    mean_mae = np.mean([m["mae"] for m in fold_metrics])
    mean_r2 = np.mean([m["r2"] for m in fold_metrics])

    return {
        "feature": feature_name,
        "mean_mae": mean_mae,
        "mean_r2": mean_r2,
        "fold_count": n_splits,
    }


def backtest_all_features(horizon: int = 10, n_splits: int = 5, use_engineered: bool = True):
    """Backtest all features independently."""
    X, y, feature_cols = build_dataset(horizon=horizon, use_engineered_features=use_engineered)
    print(f"\nBacktesting {len(feature_cols)} features ({len(X)} samples) with {n_splits}-fold TSCV...")

    results = []
    for i, feat in enumerate(feature_cols, 1):
        print(f"  [{i:3d}/{len(feature_cols)}] {feat:40s}", end="", flush=True)
        result = backtest_single_feature(feat, X, y, n_splits=n_splits)
        results.append(result)
        print(f" → R²={result['mean_r2']:7.4f}, MAE={result['mean_mae']:8.6f}")

    # Sort by R2
    results_df = pd.DataFrame(results).sort_values("mean_r2", ascending=False)

    # Save with timestamped directory
    if use_engineered:
        base_dir = os.path.join(RESULTS_DIR, "regression-baseline", "runs", 
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        base_dir = os.path.join(RESULTS_DIR, "baseline", "runs",
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    summary_dir = os.path.join(base_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    csv_path = os.path.join(summary_dir, "feature_backtest_results.csv")
    results_df.to_csv(csv_path, index=False)

    # Print results
    print(f"\n{'='*80}")
    print("FEATURE BACKTEST RANKING (by R²)")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total features tested:     {len(results_df)}")
    print(f"Top feature:               {results_df.iloc[0]['feature']} (R²={results_df.iloc[0]['mean_r2']:.4f})")
    print(f"Bottom feature:            {results_df.iloc[-1]['feature']} (R²={results_df.iloc[-1]['mean_r2']:.4f})")
    print(f"Mean R²:                   {results_df['mean_r2'].mean():.4f}")
    print(f"Median R²:                 {results_df['mean_r2'].median():.4f}")
    print(f"Std R²:                    {results_df['mean_r2'].std():.4f}")
    print(f"Features with R² > 0.01:   {(results_df['mean_r2'] > 0.01).sum()}")
    print(f"Features with R² > 0.005:  {(results_df['mean_r2'] > 0.005).sum()}")
    print(f"\nSaved: {csv_path}")
    print(f"{'='*80}\n")

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineered", action="store_true", default=True,
                        help="Use engineered features (default: True)")
    parser.add_argument("--no-engineered", dest="engineered", action="store_false",
                        help="Use only base features")
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    
    backtest_all_features(horizon=args.horizon, use_engineered=args.engineered)
