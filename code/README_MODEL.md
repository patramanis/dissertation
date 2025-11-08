# XGBoost Sector Rotation Model - Development Pipeline

## Overview
Solid foundation for building a **knowledge-driven** sector rotation model avoiding survivor bias and random noise. This pipeline supports future advanced features (GARCH-MIDAS, MRS, ROCKET, QXGBoost, Monte Carlo).

## Architecture

### 1. **model_xgb.py** - XGBoost Training
**Purpose:** Train regression/classification models with proper temporal integrity
- ✓ TimeSeriesSplit (5-fold) preserves temporal order
- ✓ Inner validation split (20% of train) prevents early stopping leakage  
- ✓ Anti-overfitting: `max_depth=3, reg_lambda=3.0, reg_alpha=0.5, subsample=0.7`
- ✓ No look-ahead bias: targets use future returns only

**Output:**
- `results/xgb_model.pkl` - Trained model for SHAP
- `results/xgb_config.json` - Training parameters
- `results/xgb_cv_metrics.json` - Out-of-sample fold metrics (MAE, R²)
- `results/xgb_oof_preds.csv` - Out-of-fold predictions

**Usage:**
```bash
python code/model_xgb.py --task regression --horizon 10 --splits 5
python code/model_xgb.py --task classification --horizon 5
```

### 2. **shap_analysis.py** - Feature Interpretability
**Purpose:** Explain which features drive predictions, detect harmful features
- SHAP TreeExplainer computes feature contributions
- Summary plot: SHAP vs feature values (scatter)
- Bar plot: Mean absolute SHAP per feature (ranking)

**Output:**
- `results/shap_summary_plot.png` - Interpretability scatter plot
- `results/shap_bar_plot.png` - Feature importance ranking
- `results/shap_feature_importance.csv` - Numerical rankings

**Usage:**
```bash
python code/shap_analysis.py
```
Loads the last trained model from `results/xgb_model.pkl`

### 3. **feature_backtest.py** - Single-Feature Validation
**Purpose:** Test each feature independently, rank by predictive power
- Trains tiny XGBoost per feature (only 1 column)
- 5-fold TSCV with inner validation
- Reports MAE, R² for each feature
- Detects true signal vs. lucky noise

**Output:**
- `results/feature_backtest_results.csv` - Ranked by R²

**Usage:**
```bash
python code/feature_backtest.py
```

## Data Pipeline

```
sector_data.csv (10 ETFs, 6307 days)
     |
     +---> model_xgb.py ---> [Training loop]
     |
features_data.csv (9 indicators: OIL, USD, GOLD, IRX, TNX, VIX, EPU, GPR, SPREAD)
     |
     +---> Align on Date ---> X matrix (6297 rows × 9 cols)
     |                            |
     +---> Build target --------> y (SPY returns over horizon)
            (log-returns, no leakage)

TimeSeriesSplit(5):
  [Train1] [Val1] [Test1]
           [Train2] [Val2] [Test2]
                    [Train3] [Val3] [Test3]
                             [Train4] [Val4] [Test4]
                                      [Train5] [Val5] [Test5]
```

## Key Design Decisions

### ✓ Temporal Integrity
- **No look-ahead bias**: Features from time `t` predict return from `t+horizon`
- **Inner validation**: Uses only training data (prevents test leakage on early stopping)
- **Forward-fill with limits**: Market data ffill(limit=5) avoids stale prices

### ✓ Survivor Bias Mitigation
- Using 9 base sector indicators (not position-weighted)
- Will detect weak features post-2017 when XLRE/XLC created
- Feature backtest ranks each independently

### ✓ Signal vs. Noise Detection
- R² metric: Negative R² = worse than mean (random noise)
- Single-feature backtest: Most features will fail, only strong ones remain
- Mean R² < 0.02 typical for daily returns (expected)

## Typical Workflow

### Step 1: Train baseline
```bash
cd code
python model_xgb.py --task regression --horizon 10
```
Check `results/xgb_cv_metrics.json` for mean MAE/R²

### Step 2: Interpret via SHAP
```bash
python shap_analysis.py
```
View `shap_bar_plot.png` to see feature importance

### Step 3: Backtest individual features
```bash
python feature_backtest.py
```
Check `feature_backtest_results.csv` - weak features = drop, strong = keep

### Step 4: Add new features, repeat Steps 1-3
Example flow:
1. Add GARCH-MIDAS volatility in `feature_data_load.py`
2. Rerun `python model_xgb.py`
3. `python shap_analysis.py` to see if volatility helps
4. `python feature_backtest.py` to verify independent signal

## Future Extensions

### Near-term (build on solid base)
- [ ] **Regime Detection**: Add MRS model output as feature
- [ ] **Volatility**: GARCH-MIDAS for conditioned volatility
- [ ] **Feature Scaling**: ROCKET time-series convolutions

### Medium-term (quantile + uncertainty)
- [ ] **QXGBoost**: Quantile regression for return intervals
- [ ] **Ensemble**: Multiple horizons (1d, 5d, 10d, 20d)

### Long-term (probabilistic)
- [ ] **Monte Carlo**: Generate distribution of outcomes
- [ ] **Risk Metrics**: VaR, CVaR, Sharpe from predictions

## Dependencies

```
xgboost>=2.0.3
shap>=0.45.1
scikit-learn>=1.4.2
matplotlib>=3.8.4
pandas>=2.3.3
numpy>=2.3.4
```

Install:
```bash
pip install -r requirements.txt
```

## Common Issues

**Issue:** SHAP fails because model not found
**Solution:** Run `python code/model_xgb.py` first to save model pickle

**Issue:** Features have R² < -0.5
**Solution:** Expected for daily returns. Increase horizon (e.g., `--horizon 20`) for weekly signal

**Issue:** All features rank equally in backtest
**Solution:** Target (returns) may be too noisy. Try classification (`--task classification`) instead

## References

- **SHAP**: https://github.com/shap/shap
- **TimeSeriesSplit**: sklearn.model_selection
- **XGBoost Anti-Overfit**: max_depth, reg_lambda, early_stopping
- **Survivor Bias**: Recognize 11 sectors post-2017 vs 9 before

---
**Status:** Ready for feature engineering and advanced models
