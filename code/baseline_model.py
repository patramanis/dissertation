import os
import json
import numpy as np
import pandas as pd

# Non-interactive backend for saving plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Paths
DATA_PATH = os.path.join("data", "model_ready_data.csv")
# Baseline output layout
BASELINE_ROOT = os.path.join("results", "baseline")
PER_SECTOR_ROOT = os.path.join(BASELINE_ROOT, "per_sector")
SUMMARY_ROOT = os.path.join(BASELINE_ROOT, "summary")

# CV and model config (env override via BASELINE_N_SPLITS)
N_SPLITS = int(os.getenv("BASELINE_N_SPLITS", 5))
EARLY_STOPPING_ROUNDS = 50

FIXED_XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
}


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
    return df


def build_feature_matrix(df):
    all_columns = df.columns.tolist()

    # Targets
    target_cols = [c for c in all_columns if c.startswith("y_reg_") or c.startswith("y_clf_")]

    # Leakage columns (current returns and realized vols)
    return_cols = [c for c in all_columns if c.endswith("_return")]
    vol_cols = [c for c in all_columns if "_realized_vol_" in c]

    # Base price-level columns (raw inputs before FE)
    base_sectors = [c for c in all_columns if c.startswith("XL")]
    base_features = ["VIX", "EPU", "GOLD", "OIL", "USD", "IRX", "TNX", "SPREAD_10Y_13W", "GPR"]
    known_price_cols = [c for c in base_sectors + base_features if c in all_columns]

    # Final drop list
    features_to_drop = list(dict.fromkeys(target_cols + return_cols + vol_cols + known_price_cols))

    X = df.drop(columns=features_to_drop)
    return X, target_cols


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_sector_baseline(sector_name, X, y, base_results_dir):
    # Output dir
    sector_dir = os.path.join(base_results_dir, sector_name)
    ensure_dir(sector_dir)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics = []
    trained_models = []
    all_test_indices = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = XGBClassifier(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **FIXED_XGB_PARAMS)
        eval_set = [(X_test, y_test)]

        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        preds = model.predict(X_test)
        # Probabilities if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = preds.astype(float)

        metrics = {
            "fold": fold,
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan,
        }
        fold_metrics.append(metrics)
        trained_models.append(model)
        all_test_indices.append(test_index)

    # Save per-fold metrics
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(os.path.join(sector_dir, "fold_metrics.csv"), index=False)

    # Aggregate summary for sector
    summary = {
        "sector": sector_name,
        "accuracy_mean": float(metrics_df["accuracy"].mean()),
        "accuracy_std": float(metrics_df["accuracy"].std()),
        "f1_mean": float(metrics_df["f1"].mean()),
        "f1_std": float(metrics_df["f1"].std()),
        "precision_mean": float(metrics_df["precision"].mean()),
        "precision_std": float(metrics_df["precision"].std()),
        "recall_mean": float(metrics_df["recall"].mean()),
        "recall_std": float(metrics_df["recall"].std()),
        "roc_auc_mean": float(metrics_df["roc_auc"].mean(skipna=True)),
        "roc_auc_std": float(metrics_df["roc_auc"].std(skipna=True)),
    }

    # Native importance plot
    try:
        last_model = trained_models[-1]
        from xgboost import plot_importance

        ax = plot_importance(last_model, max_num_features=30)
        plt.tight_layout()
        plt.savefig(os.path.join(sector_dir, "native_importance_plot.png"))
        plt.close()
    except Exception as e:
        with open(os.path.join(sector_dir, "warnings.txt"), "a", encoding="utf-8") as f:
            f.write(f"Native importance plot failed: {e}\n")

    # SHAP summary (optional)
    try:
        last_idx = all_test_indices[-1]
        X_test_to_explain = X.iloc[last_idx]
        model_to_explain = trained_models[-1]

        # Lazy import
        try:
            import shap as _shap
        except Exception as ie:
            raise RuntimeError(f"SHAP not available: {ie}")

        explainer = _shap.TreeExplainer(model_to_explain)
        shap_values = explainer(X_test_to_explain)

        _shap.summary_plot(shap_values, X_test_to_explain, max_display=30, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(sector_dir, "shap_summary_plot.png"), dpi=150)
        plt.close()
    except Exception as e:
        with open(os.path.join(sector_dir, "warnings.txt"), "a", encoding="utf-8") as f:
            f.write(f"SHAP summary failed: {e}\n")

    # SHAP dependence plots for top features (optional)
    try:
        if 'shap_values' in locals() and hasattr(shap_values, "values"):
            shap_arr = np.abs(shap_values.values)
            mean_abs = shap_arr.mean(axis=0)
            top_idx = np.argsort(-mean_abs)[:3]
            top_features = X_test_to_explain.columns[top_idx]
        else:
            top_features = X_test_to_explain.columns[:3]

        for feat in top_features:
            try:
                import shap as _shap
                _shap.dependence_plot(
                    feat,
                    shap_values.values if ('shap_values' in locals() and hasattr(shap_values, "values")) else shap_values,
                    X_test_to_explain,
                    show=False,
                )
                plt.tight_layout()
                plt.savefig(os.path.join(sector_dir, f"shap_dependence_{feat}.png"), dpi=150)
                plt.close()
            except Exception as ie:
                with open(os.path.join(sector_dir, "warnings.txt"), "a", encoding="utf-8") as f:
                    f.write(f"SHAP dependence for {feat} failed: {ie}\n")
    except Exception:
        pass

    # Save sector summary JSON
    with open(os.path.join(sector_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    print("--- Baseline model run started ---")
    # Ensure output folders
    ensure_dir(BASELINE_ROOT)
    ensure_dir(PER_SECTOR_ROOT)
    ensure_dir(SUMMARY_ROOT)

    df = load_data()
    X, target_cols = build_feature_matrix(df)

    classification_targets = [c for c in target_cols if c.startswith("y_clf_")]
    all_sectors_summary = []

    for target_name in classification_targets:
        sector_name = target_name.replace("y_clf_", "")
        print(f"--- Processing Baseline Model for: {sector_name} ---")
        y = df[target_name]
        summary = run_sector_baseline(sector_name, X, y, PER_SECTOR_ROOT)
        print(
            f"Robust F1 for {sector_name}: {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}"
        )
        all_sectors_summary.append(summary)

    df_summary = pd.DataFrame(all_sectors_summary)
    df_summary.sort_values(by="f1_mean", ascending=False, inplace=True)
    # Save summary
    df_summary.to_csv(os.path.join(SUMMARY_ROOT, "Sectors_Symmary.csv"), index=False)

    print("Baseline analysis complete. Summary saved to results.")

if __name__ == "__main__":
    main()
