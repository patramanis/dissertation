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
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "classification-model" / "multi-horizon"

# Configuration
SECTOR = "XLK"
HORIZONS = {
    '1d': 1,        # 1 day
    '1w': 5,        # 1 week (5 trading days)
    '1m': 21,       # 1 month (21 trading days)
    '2m': 42,       # 2 months (42 trading days)
    '6m': 126       # 6 months (126 trading days)
}
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

def prepare_targets(df):
    print(f"\n{'='*80}")
    print("PREPARING MULTI-HORIZON TARGETS")
    print(f"{'='*80}")
    
    # Calculate SPY proxy (market average)
    df['SPY_proxy'] = df[SECTORS].mean(axis=1)
    
    targets = {}
    target_stats = []
    
    for horizon_name, horizon_days in HORIZONS.items():
        # Calculate returns for sector and market
        sector_return = df[SECTOR].pct_change(horizon_days).shift(-horizon_days)
        spy_return = df['SPY_proxy'].pct_change(horizon_days).shift(-horizon_days)
        
        # Target: 1 if sector beats market, 0 otherwise
        target_col = f'target_{horizon_name}'
        df[target_col] = (sector_return > spy_return).astype(int)
        targets[horizon_name] = target_col
        
        # Stats
        non_null = df[target_col].notna().sum()
        if non_null > 0:
            positive_ratio = df[target_col].sum() / non_null
            target_stats.append({
                'horizon': horizon_name,
                'days': horizon_days,
                'samples': non_null,
                'positive': df[target_col].sum(),
                'negative': non_null - df[target_col].sum(),
                'positive_ratio': positive_ratio
            })
            
            print(f"\n{horizon_name} ({horizon_days} days):")
            print(f"  Samples: {non_null}")
            print(f"  Positive: {df[target_col].sum()} ({positive_ratio:.1%})")
            print(f"  Negative: {non_null - df[target_col].sum()}")
    
    return df, targets, pd.DataFrame(target_stats)

def prepare_features(df, target_col):
    # Feature columns: exclude Date, sectors, all targets, SPY_proxy, and 'target' column
    target_cols = [c for c in df.columns if c.startswith('target_')]
    exclude_cols = ['Date'] + SECTORS + target_cols + ['SPY_proxy', 'target']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Clean data - remove rows where target is NaN
    df_clean = df[df[target_col].notna()].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    dates = df_clean['Date']
    
    return X, y, dates, feature_cols

def train_and_evaluate(X, y, horizon_name):
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {horizon_name}")
    print(f"{'='*80}")
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    feature_importance = None
    
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
        
        print(f"  Class balance: {n_neg} neg, {n_pos} pos (scale={scale_pos_weight:.3f})")
        
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
            'horizon': horizon_name,
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
        
        print(f"  AUC: {metrics['auc']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Store feature importance from last fold
        if fold == N_SPLITS:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
    
    return fold_metrics, all_y_true, all_y_pred, all_y_proba, model, feature_importance

def calculate_summary_metrics(fold_metrics, all_y_true, all_y_pred, all_y_proba):
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
        
        # Overall metrics
        'overall_accuracy': accuracy_score(all_y_true, all_y_pred),
        'overall_auc': roc_auc_score(all_y_true, all_y_proba),
        'overall_f1': f1_score(all_y_true, all_y_pred),
        'overall_precision': precision_score(all_y_true, all_y_pred, zero_division=0),
        'overall_recall': recall_score(all_y_true, all_y_pred, zero_division=0),
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    summary['confusion_matrix'] = cm.tolist()
    
    return summary

def save_results(run_dir, horizon_name, fold_metrics, summary, feature_importance):
    horizon_dir = run_dir / horizon_name
    horizon_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fold metrics
    pd.DataFrame(fold_metrics).to_csv(horizon_dir / 'fold_metrics.csv', index=False)
    
    # Save summary
    with open(horizon_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save feature importance
    if feature_importance is not None:
        feature_importance.to_csv(horizon_dir / 'feature_importance.csv', index=False)
    
    print(f"\nResults saved to: {horizon_dir}")

def create_comparison_plots(all_results, run_dir):
    print(f"\n{'='*80}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*80}")
    
    horizons = list(all_results.keys())
    
    # Extract metrics
    metrics_data = []
    for horizon in horizons:
        summary = all_results[horizon]['summary']
        metrics_data.append({
            'horizon': horizon,
            'days': HORIZONS[horizon],
            'auc': summary['overall_auc'],
            'auc_std': summary['auc_std'],
            'recall': summary['overall_recall'],
            'recall_std': summary['recall_std'],
            'precision': summary['overall_precision'],
            'precision_std': summary['precision_std'],
            'f1': summary['overall_f1'],
            'samples': all_results[horizon]['fold_metrics'][0]['test_size'] * N_SPLITS
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Plot 1: AUC comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # AUC by horizon
    ax = axes[0, 0]
    ax.errorbar(df_metrics['days'], df_metrics['auc'], 
                yerr=df_metrics['auc_std'], marker='o', capsize=5, linewidth=2)
    ax.set_xlabel('Horizon (Trading Days)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    
    # Recall by horizon
    ax = axes[0, 1]
    ax.errorbar(df_metrics['days'], df_metrics['recall'], 
                yerr=df_metrics['recall_std'], marker='s', capsize=5, linewidth=2, color='green')
    ax.set_xlabel('Horizon (Trading Days)', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Precision by horizon
    ax = axes[1, 0]
    ax.errorbar(df_metrics['days'], df_metrics['precision'], 
                yerr=df_metrics['precision_std'], marker='^', capsize=5, linewidth=2, color='orange')
    ax.set_xlabel('Horizon (Trading Days)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # F1 by horizon
    ax = axes[1, 1]
    ax.plot(df_metrics['days'], df_metrics['f1'], marker='D', linewidth=2, color='purple')
    ax.set_xlabel('Horizon (Trading Days)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(run_dir / 'horizon_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics table
    df_metrics.to_csv(run_dir / 'horizon_metrics.csv', index=False)
    
    print(f"Comparison plots saved to: {run_dir}")
    
    return df_metrics

def analyze_feature_consistency(all_results, run_dir):
    print(f"\n{'='*80}")
    print("ANALYZING FEATURE CONSISTENCY")
    print(f"{'='*80}")
    
    # Collect top features for each horizon
    top_n = 20
    feature_ranks = {}
    
    for horizon, results in all_results.items():
        feat_imp = results['feature_importance']
        top_features = feat_imp.head(top_n)
        feature_ranks[horizon] = top_features.set_index('feature')['importance'].to_dict()
    
    # Find features that appear in top N for multiple horizons
    all_features = set()
    for horizon_features in feature_ranks.values():
        all_features.update(horizon_features.keys())
    
    feature_consistency = []
    for feature in all_features:
        appearances = sum(1 for h in feature_ranks if feature in feature_ranks[h])
        avg_importance = np.mean([feature_ranks[h].get(feature, 0) for h in feature_ranks])
        feature_consistency.append({
            'feature': feature,
            'horizons_in_top20': appearances,
            'avg_importance': avg_importance
        })
    
    df_consistency = pd.DataFrame(feature_consistency).sort_values(
        ['horizons_in_top20', 'avg_importance'], ascending=False
    )
    
    # Save
    df_consistency.to_csv(run_dir / 'feature_consistency.csv', index=False)
    
    # Print top consistent features
    print("\nMost Consistent Features (appear in top 20 for multiple horizons):")
    print(df_consistency.head(15).to_string(index=False))
    
    return df_consistency

def identify_missing_features(df_consistency, all_results):
    """
    Identify potential missing features to strengthen signal.
    """
    print(f"\n{'='*80}")
    print("FEATURE GAP ANALYSIS")
    print(f"{'='*80}")
    
    # Analyze current feature types
    top_features = df_consistency.head(30)['feature'].tolist()
    
    feature_types = {
        'price_based': 0,
        'volatility': 0,
        'momentum': 0,
        'technical': 0,
        'macro': 0,
        'sentiment': 0
    }
    
    for feat in top_features:
        feat_lower = feat.lower()
        if 'return' in feat_lower or 'pct' in feat_lower or '_lag' in feat_lower:
            feature_types['price_based'] += 1
        if 'rollstd' in feat_lower or 'vol' in feat_lower or 'atr' in feat_lower:
            feature_types['volatility'] += 1
        if 'rsi' in feat_lower or 'macd' in feat_lower or 'momentum' in feat_lower:
            feature_types['momentum'] += 1
        if 'adx' in feat_lower or 'cci' in feat_lower or 'bb' in feat_lower:
            feature_types['technical'] += 1
        if 'epu' in feat_lower or 'gpr' in feat_lower or 'spread' in feat_lower or 'tnx' in feat_lower:
            feature_types['macro'] += 1
        if 'sentiment' in feat_lower or 'news' in feat_lower:
            feature_types['sentiment'] += 1
    
    print("\nCurrent Feature Type Distribution (Top 30):")
    for feat_type, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat_type:15s}: {count:2d} features")
    
    # Identify gaps
    print("\n" + "="*80)
    print("MISSING FEATURE CATEGORIES (Recommendations)")
    print("="*80)
    
    recommendations = []
    
    # 1. Options data
    if feature_types['volatility'] < 10:
        recommendations.append({
            'category': 'Options & Implied Volatility',
            'missing': [
                'VIX term structure (VIX vs VIX3M)',
                'Put/Call ratios per sector',
                'Options volume and open interest',
                'Implied volatility skew',
                'Volatility risk premium (IV - RV)'
            ],
            'benefit': 'Captures market fear/greed and hedging demand',
            'horizon_benefit': 'Most useful for 1d-1w (short-term sentiment)'
        })
    
    # 2. Sentiment data
    if feature_types['sentiment'] < 2:
        recommendations.append({
            'category': 'Market Sentiment',
            'missing': [
                'AAII Sentiment Survey (bull/bear ratio)',
                'CNN Fear & Greed Index',
                'Twitter/X sentiment for tech stocks',
                'News sentiment (financial media)',
                'Google Trends for sector keywords'
            ],
            'benefit': 'Captures retail and institutional positioning',
            'horizon_benefit': 'Most useful for 1w-1m (medium-term flows)'
        })
    
    # 3. Flows data
    recommendations.append({
        'category': 'Fund Flows & Positioning',
        'missing': [
            'ETF fund flows (XLK vs SPY inflows)',
            'CFTC Commitment of Traders (positioning)',
            'Mutual fund flows by sector',
            'Institutional ownership changes',
            'Short interest ratio changes'
        ],
        'benefit': 'Captures actual money flows and positioning',
        'horizon_benefit': 'Most useful for 1m-6m (medium to long-term)'
    })
    
    # 4. Fundamental data
    recommendations.append({
        'category': 'Fundamental Indicators',
        'missing': [
            'Sector earnings revisions',
            'Forward P/E ratios (sector vs market)',
            'Revenue growth expectations',
            'Analyst upgrades/downgrades',
            'Earnings surprise history'
        ],
        'benefit': 'Captures fundamental value changes',
        'horizon_benefit': 'Most useful for 1m-6m (longer-term value)'
    })
    
    # 5. Relative strength
    recommendations.append({
        'category': 'Inter-Sector Dynamics',
        'missing': [
            'Sector rotation indicators',
            'Relative strength vs other sectors',
            'Correlation changes with SPY',
            'Beta evolution over time',
            'Sector pair spreads (XLK-XLF, XLK-XLE)'
        ],
        'benefit': 'Captures sector rotation and relative performance',
        'horizon_benefit': 'Most useful for 1w-2m (rotation cycles)'
    })
    
    # 6. Macro events
    recommendations.append({
        'category': 'Event-Based Features',
        'missing': [
            'Fed meeting dates (before/after)',
            'Earnings season indicators',
            'Product launch cycles (Apple, Microsoft)',
            'Major tech conferences',
            'Regulatory events'
        ],
        'benefit': 'Captures event-driven volatility',
        'horizon_benefit': 'Most useful for 1d-1w (event timing)'
    })
    
    # 7. Alternative data
    recommendations.append({
        'category': 'Alternative Data',
        'missing': [
            'Semiconductor sales data (for tech)',
            'Cloud revenue proxies (AWS, Azure)',
            'App download trends',
            'Web traffic to tech companies',
            'Job postings in tech sector'
        ],
        'benefit': 'Early indicators of sector health',
        'horizon_benefit': 'Most useful for 1m-6m (leading indicators)'
    })
    
    return recommendations, feature_types

def print_recommendations(recommendations, feature_types, df_metrics):
    print("\n" + "="*80)
    print("PRIORITY RECOMMENDATIONS TO STRENGTHEN SIGNAL")
    print("="*80)
    
    # Find weakest horizon
    weakest_horizon = df_metrics.loc[df_metrics['auc'].idxmin()]
    strongest_horizon = df_metrics.loc[df_metrics['auc'].idxmax()]
    
    print(f"\nCurrent Performance:")
    print(f"  Strongest: {strongest_horizon['horizon']} (AUC: {strongest_horizon['auc']:.4f})")
    print(f"  Weakest:   {weakest_horizon['horizon']} (AUC: {weakest_horizon['auc']:.4f})")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['category']}")
        print(f"   Benefit: {rec['benefit']}")
        print(f"   Best for: {rec['horizon_benefit']}")
        print(f"   Missing data:")
        for item in rec['missing'][:3]:  # Show top 3
            print(f"     • {item}")
        if len(rec['missing']) > 3:
            print(f"     ... and {len(rec['missing']) - 3} more")

def main():
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"MULTI-HORIZON CLASSIFICATION MODEL - {SECTOR}")
    print(f"Run: {timestamp}")
    print("="*80)
    
    # Load data
    df = load_data(use_selected=True)
    
    # Prepare all targets
    df, targets, target_stats = prepare_targets(df)
    
    # Save target statistics
    target_stats.to_csv(run_dir / 'target_statistics.csv', index=False)
    
    # Train model for each horizon
    all_results = {}
    
    for horizon_name, target_col in targets.items():
        print(f"\n{'#'*80}")
        print(f"PROCESSING HORIZON: {horizon_name} ({HORIZONS[horizon_name]} days)")
        print(f"{'#'*80}")
        
        # Prepare features
        X, y, dates, feature_cols = prepare_features(df, target_col)
        
        print(f"\nDataset info:")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {len(y)}")
        print(f"  Date range: {dates.min()} to {dates.max()}")
        print(f"  Positive rate: {y.mean():.1%}")
        
        # Train and evaluate
        fold_metrics, all_y_true, all_y_pred, all_y_proba, model, feature_importance = \
            train_and_evaluate(X, y, horizon_name)
        
        # Calculate summary
        summary = calculate_summary_metrics(fold_metrics, all_y_true, all_y_pred, all_y_proba)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {horizon_name}")
        print(f"{'='*80}")
        print(f"Overall AUC:       {summary['overall_auc']:.4f} (±{summary['auc_std']:.4f})")
        print(f"Overall Recall:    {summary['overall_recall']:.4f} (±{summary['recall_std']:.4f})")
        print(f"Overall Precision: {summary['overall_precision']:.4f} (±{summary['precision_std']:.4f})")
        print(f"Overall F1:        {summary['overall_f1']:.4f}")
        
        # Save results
        save_results(run_dir, horizon_name, fold_metrics, summary, feature_importance)
        
        # Store for comparison
        all_results[horizon_name] = {
            'fold_metrics': fold_metrics,
            'summary': summary,
            'feature_importance': feature_importance
        }
    
    # Create comparison plots
    print(f"\n{'#'*80}")
    print("CROSS-HORIZON ANALYSIS")
    print(f"{'#'*80}")
    
    df_metrics = create_comparison_plots(all_results, run_dir)
    df_consistency = analyze_feature_consistency(all_results, run_dir)
    recommendations, feature_types = identify_missing_features(df_consistency, all_results)
    
    # Print recommendations
    print_recommendations(recommendations, feature_types, df_metrics)
    
    # Save recommendations
    with open(run_dir / 'recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"\n{'='*80}")
    print("MULTI-HORIZON ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {run_dir}")
    print(f"\nFiles created:")
    print(f"  • horizon_metrics.csv - Performance comparison")
    print(f"  • horizon_comparison.png - Visual comparison")
    print(f"  • feature_consistency.csv - Feature importance across horizons")
    print(f"  • recommendations.json - Missing feature suggestions")
    print(f"  • [horizon]/summary.json - Detailed results per horizon")
    
    return all_results, df_metrics, recommendations

if __name__ == "__main__":
    all_results, df_metrics, recommendations = main()
