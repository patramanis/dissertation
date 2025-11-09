import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                             precision_score, recall_score, log_loss,
                             confusion_matrix, classification_report)
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "classification-model" / "runs"

# Configuration
SECTOR = "XLK"
HORIZON = 21
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
N_SPLITS = 5

# XGBoost parameters (optimized from diagnostics)
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_lambda': 10,
    'reg_alpha': 1,
    'min_child_weight': 5,
    'random_state': 42,
    'n_jobs': -1
}

def load_data(use_selected=True):
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    if use_selected:
        data_path = DATA_DIR / "engineered_features_selected_v2.csv"
        print(f"\nUsing selected features: {data_path}")
    else:
        data_path = DATA_DIR / "engineered_features_v2.csv"
        print(f"\nUsing all features: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape}")
    
    return df

def prepare_target(df):
    print(f"\n{'='*80}")
    print("PREPARING TARGET")
    print(f"{'='*80}")
    
    # Calculate returns
    sector_return = df[SECTOR].pct_change(HORIZON).shift(-HORIZON)
    df['SPY_proxy'] = df[SECTORS].mean(axis=1)
    spy_return = df['SPY_proxy'].pct_change(HORIZON).shift(-HORIZON)
    
    # Target: 1 if sector beats market, 0 otherwise
    df['target'] = (sector_return > spy_return).astype(int)
    
    print(f"Target: {SECTOR} beats SPY over {HORIZON} days")
    print(f"Distribution: {df['target'].value_counts().to_dict()}")
    
    return df

def prepare_features(df):
    print(f"\n{'='*80}")
    print("PREPARING FEATURES")
    print(f"{'='*80}")
    
    # Feature columns: exclude Date, sectors, target, SPY_proxy
    feature_cols = [c for c in df.columns 
                    if c not in ['Date'] + SECTORS + ['target', 'SPY_proxy']]
    
    # Clean data
    df_clean = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    dates = df_clean['Date']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(y)}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    
    return X, y, dates, feature_cols

def train_and_evaluate(X, y):
    print(f"\n{'='*80}")
    print("TRAINING MODEL WITH TIME SERIES CV")
    print(f"{'='*80}")
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{N_SPLITS}")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Test:  {len(test_idx)} samples")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Calculate class balance for this fold
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        print(f"  Class balance: {n_neg} neg, {n_pos} pos")
        print(f"  scale_pos_weight: {scale_pos_weight:.3f}")
        
        # Train model with class balancing
        xgb_params = XGB_PARAMS.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight
        
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'scale_pos_weight': scale_pos_weight,
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'log_loss': log_loss(y_test, y_proba)
        }
        
        fold_metrics.append(metrics)
        
        # Store predictions
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC:      {metrics['auc']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
    
    return fold_metrics, all_y_true, all_y_pred, all_y_proba, model

def calculate_summary_metrics(fold_metrics, all_y_true, all_y_pred, all_y_proba):
    print(f"\n{'='*80}")
    print("SUMMARY METRICS")
    print(f"{'='*80}")
    
    # Aggregate metrics
    summary = {
        'accuracy_mean': np.mean([m['accuracy'] for m in fold_metrics]),
        'accuracy_std': np.std([m['accuracy'] for m in fold_metrics]),
        'auc_mean': np.mean([m['auc'] for m in fold_metrics]),
        'auc_std': np.std([m['auc'] for m in fold_metrics]),
        'f1_mean': np.mean([m['f1'] for m in fold_metrics]),
        'f1_std': np.std([m['f1'] for m in fold_metrics]),
        'precision_mean': np.mean([m['precision'] for m in fold_metrics]),
        'precision_std': np.std([m['precision'] for m in fold_metrics]),
        'recall_mean': np.mean([m['recall'] for m in fold_metrics]),
        'recall_std': np.std([m['recall'] for m in fold_metrics]),
        'log_loss_mean': np.mean([m['log_loss'] for m in fold_metrics]),
        'log_loss_std': np.std([m['log_loss'] for m in fold_metrics]),
    }
    
    # Overall metrics on all predictions
    summary['overall_accuracy'] = accuracy_score(all_y_true, all_y_pred)
    summary['overall_auc'] = roc_auc_score(all_y_true, all_y_proba)
    summary['overall_f1'] = f1_score(all_y_true, all_y_pred)
    summary['overall_precision'] = precision_score(all_y_true, all_y_pred, zero_division=0)
    summary['overall_recall'] = recall_score(all_y_true, all_y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    summary['confusion_matrix'] = cm.tolist()
    
    print(f"\nCV Metrics (mean ± std):")
    print(f"  Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  AUC:       {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  F1:        {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    print(f"  Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {summary['overall_accuracy']:.4f}")
    print(f"  AUC:       {summary['overall_auc']:.4f}")
    print(f"  F1:        {summary['overall_f1']:.4f}")
    print(f"  Precision: {summary['overall_precision']:.4f}")
    print(f"  Recall:    {summary['overall_recall']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return summary

def compute_shap_importance(model, X, feature_cols, output_dir):
    print(f"\n{'='*80}")
    print("COMPUTING SHAP IMPORTANCE")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a sample for SHAP (faster)
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    
    print(f"Computing SHAP values for {sample_size} samples...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"\nTop 10 features by SHAP importance:")
    print(shap_df.head(10).to_string(index=False))
    
    # Save
    shap_df.to_csv(output_dir / 'shap_importance.csv', index=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_n = 20
    top_features = shap_df.head(top_n)
    plt.barh(range(top_n), top_features['mean_abs_shap'])
    plt.yticks(range(top_n), top_features['feature'])
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Top {top_n} Features by SHAP Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_importance_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SHAP importance to {output_dir}")
    
    return shap_df

def save_results(output_dir, fold_metrics, summary, feature_cols, model):
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fold metrics
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(output_dir / 'fold_metrics.csv', index=False)
    print(f"Saved fold_metrics.csv")
    
    # 2. Summary JSON
    summary_json = summary.copy()
    summary_json['timestamp'] = datetime.now().isoformat()
    summary_json['sector'] = SECTOR
    summary_json['horizon'] = HORIZON
    summary_json['n_features'] = len(feature_cols)
    summary_json['xgb_params'] = XGB_PARAMS
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved summary.json")
    
    # 3. Feature importance (native XGBoost)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    print(f"Saved feature_importance.csv")
    
    # 4. Native importance plot
    plt.figure(figsize=(10, 8))
    top_n = 20
    top_features = importance_df.head(top_n)
    plt.barh(range(top_n), top_features['importance'])
    plt.yticks(range(top_n), top_features['feature'])
    plt.xlabel('XGBoost Gain Importance')
    plt.title(f'Top {top_n} Features by XGBoost Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'native_importance_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved native_importance_plot.png")
    
    print(f"\nAll results saved to: {output_dir}")

def main():
    print(f"\n{'='*80}")
    print("XGBOOST CLASSIFICATION MODEL - WITH CLASS BALANCING")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sector: {SECTOR}")
    print(f"Horizon: {HORIZON} days")
    print(f"CV Splits: {N_SPLITS}")
    print(f"Improvement: Auto class balancing (scale_pos_weight per fold)")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_DIR / timestamp / SECTOR
    
    # Load data
    df = load_data(use_selected=True)
    
    # Prepare target
    df = prepare_target(df)
    
    # Prepare features
    X, y, dates, feature_cols = prepare_features(df)
    
    # Train and evaluate
    fold_metrics, all_y_true, all_y_pred, all_y_proba, model = train_and_evaluate(X, y)
    
    # Calculate summary metrics
    summary = calculate_summary_metrics(fold_metrics, all_y_true, all_y_pred, all_y_proba)
    
    # Print comparison to baseline
    print(f"\n{'='*80}")
    print("COMPARISON TO BASELINE")
    print(f"{'='*80}")
    baseline_auc = 0.5718
    baseline_recall_std = 0.2898
    improvement_auc = summary['overall_auc'] - baseline_auc
    improvement_recall_std = baseline_recall_std - summary['recall_std']
    
    print(f"Baseline AUC:           {baseline_auc:.4f}")
    print(f"Class Balanced AUC:     {summary['overall_auc']:.4f}")
    print(f"Improvement:            {improvement_auc:+.4f} ({100*improvement_auc/baseline_auc:+.1f}%)")
    print(f"\nBaseline Recall Std:    {baseline_recall_std:.4f}")
    print(f"Class Balanced Std:     {summary['recall_std']:.4f}")
    print(f"Stability Improvement:  {improvement_recall_std:+.4f} ({100*improvement_recall_std/baseline_recall_std:+.1f}%)")
    
    # Compute SHAP importance
    shap_df = compute_shap_importance(model, X, feature_cols, output_dir)
    
    # Save results
    save_results(output_dir, fold_metrics, summary, feature_cols, model)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print()

if __name__ == "__main__":
    main()
