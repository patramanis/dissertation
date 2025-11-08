"""
SHAP-based feature interpretability analysis.
Loads trained XGBoost model and explains predictions using SHAP.
Saves: summary_plot.png, bar_plot.png, feature_importance.csv
"""
import os
import pickle
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor, XGBClassifier

RESULTS_DIR = os.path.join("results")
DATA_DIR = os.path.join("data")
SECTOR_DATA_PATH = os.path.join(DATA_DIR, "sector_data.csv")
FEATURES_DATA_PATH = os.path.join(DATA_DIR, "features_data.csv")


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    sector = pd.read_csv(SECTOR_DATA_PATH, parse_dates=["Date"])
    features = pd.read_csv(FEATURES_DATA_PATH, parse_dates=["Date"])
    return sector, features


def build_dataset(horizon: int, task: str = "regression"):
    sector, features = load_datasets()
    df = pd.merge(features, sector[["Date", "SPY"]], on="Date", how="inner")
    df = df.sort_values("Date").reset_index(drop=True)
    df["price"] = df["SPY"]
    df["ret_h"] = np.log(df["price"].shift(-horizon) / df["price"])

    feature_cols = [c for c in features.columns if c != "Date"]
    X = df[feature_cols].copy()
    y = df["ret_h"].copy()

    valid_idx = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    return X.reset_index(drop=True), y.reset_index(drop=True), feature_cols


def load_model_and_config():
    """Load last trained model and config from results/"""
    model_path = os.path.join(RESULTS_DIR, "xgb_model.pkl")
    config_path = os.path.join(RESULTS_DIR, "xgb_config.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train model first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(config_path, "r") as f:
        config = json.load(f)

    return model, config


def compute_shap_values(model, X: pd.DataFrame):
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def save_shap_plots(explainer, shap_values, X: pd.DataFrame, feature_cols: List[str]):
    """Save SHAP summary and bar plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Summary plot (SHAP values vs feature values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    summary_path = os.path.join(RESULTS_DIR, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=100, bbox_inches="tight")
    plt.close()

    # Bar plot (mean absolute SHAP value per feature)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, plot_type="bar", show=False)
    bar_path = os.path.join(RESULTS_DIR, "shap_bar_plot.png")
    plt.savefig(bar_path, dpi=100, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP plots:")
    print(f"  - {summary_path}")
    print(f"  - {bar_path}")


def save_feature_importance_csv(shap_values, feature_cols: List[str]):
    """Save mean absolute SHAP per feature to CSV."""
    # Handle both regression (1D array) and classification (list of arrays)
    if isinstance(shap_values, list):
        # Multi-class or binary classification
        shap_values = shap_values[0] if len(shap_values) == 1 else np.mean(np.abs(shap_values), axis=0)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    csv_path = os.path.join(RESULTS_DIR, "shap_feature_importance.csv")
    shap_importance.to_csv(csv_path, index=False)

    print(f"\nFeature Importance (SHAP):")
    print(shap_importance.to_string(index=False))
    print(f"\nSaved: {csv_path}")


def main():
    model, config = load_model_and_config()
    print(f"Loaded model with config: {config}")

    X, y, feature_cols = build_dataset(horizon=config["horizon"], task=config.get("task", "regression"))
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    explainer, shap_values = compute_shap_values(model, X)
    save_shap_plots(explainer, shap_values, X, feature_cols)
    save_feature_importance_csv(shap_values, feature_cols)


if __name__ == "__main__":
    main()
