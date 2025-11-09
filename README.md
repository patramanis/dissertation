# Sector Classification Using Machine Learning

**Forecasting XLK (Technology Sector) Performance Based on Economic Uncertainty**

A comprehensive machine learning pipeline for predicting whether the XLK sector will outperform the market over a 21-day horizon, using the "uncertainty trinity" framework (Economic Policy Uncertainty, Geopolitical Risk, Market Volatility) combined with technical indicators and cross-asset relationships.

---

## 📊 Project Overview

### Objective
Predict if XLK (Technology Select Sector SPDR Fund) will beat the broader market over the next 21 trading days using macro-economic features, technical indicators, and market regime analysis.

### Key Results
- **AUC:** 0.5784 (7.8% above random)
- **Test Accuracy:** 55.96%
- **Win Rate:** 57.84%
- **Status:** ✅ Signal detected - Commercially viable

---

## 🗂️ Project Structure

```
dissertation/
├── code/                   # Core production code
│   ├── sector_data_load.py
│   ├── feature_data_load.py
│   ├── simple_regime_detection.py
│   ├── technical_indicators.py
│   ├── feature_engineering_v2.py
│   └── README.md
├── diagnostics/           # Analysis & testing (not in git)
│   ├── xgb_diagnostic_v2.py
│   ├── fast_shap_selection.py
│   └── README.md
├── data/                  # Data files
│   ├── EPU.csv           # Economic Policy Uncertainty (source data)
│   ├── GPR.csv           # Geopolitical Risk (source data)
│   ├── sector_data.csv   # 9 sector ETFs (generated)
│   ├── features_data.csv # Macro features (generated)
│   └── engineered_features_v2.csv  # Final features (generated)
├── temporal_files/       # Intermediate outputs (not in git)
├── tests/                # Unit tests (not in git)
└── requirements.txt      # Python dependencies
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd dissertation

# Create virtual environment
python -m venv .venv312
.venv312\Scripts\activate  # Windows
# source .venv312/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Pipeline
```bash
# Load sector data
python code/sector_data_load.py

# Load macro features
python code/feature_data_load.py

# Generate regime features
python code/simple_regime_detection.py

# Run feature engineering
python code/feature_engineering_v2.py
```

### 3. Output
- `data/engineered_features_v2.csv` - 6034 rows × 1094 columns
- 1084 engineered features ready for model training

---

## 📈 Features

### Data Sources
1. **Sector ETFs (9):** XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY
2. **Macro Features (9):** 
   - OIL (Crude Oil prices)
   - USD (US Dollar Index)
   - GOLD (Gold prices)
   - IRX (13-week Treasury Bill)
   - TNX (10-year Treasury Note)
   - VIX (Volatility Index)
   - EPU (Economic Policy Uncertainty)
   - GPR (Geopolitical Risk)
   - SPREAD_10Y_13W (Yield spread)

### Feature Engineering (1084 → 100 features)

| Category | Count | Examples |
|----------|-------|----------|
| **Rolling Statistics** | 504 | XLU_rollstd252, VIX_rollmean63 |
| **Technical Indicators** | 289 | RSI, MACD, Bollinger, ADX |
| **Lag Features** | 108 | XLK_lag1, USD_lag21 |
| **Cross-Asset** | 81 | Sector correlations, betas |
| **Volatility** | 26 | Realized volatility 21d, 63d |
| **Momentum** | 36 | Returns 5d, 21d, 63d |
| **Other** | 40 | Regime, temporal, uncertainty |
| **Total** | **1084** | |
| **After SHAP Selection** | **100** | Top predictive features |

### Top 5 Most Important Features (by SHAP)
1. **XLU_rollstd252_lag1** - Utilities 1-year volatility
2. **SPREAD_10Y_13W_adx_lag1** - Yield spread trend strength
3. **USD_rollstd63_lag1** - Dollar 3-month volatility
4. **corr_XLB_XLY_21d_lag1** - Materials-Consumer correlation
5. **GPR_rollstd42_lag1** - Geopolitical risk volatility

---

## 🎯 Model Performance

### Baseline (1084 features)
- AUC: 0.530
- Test Accuracy: 50.85%
- Overfitting Gap: 42.1%

### After SHAP Selection (100 features)
- **AUC: 0.5784 ✅ (+9.1% improvement)**
- **Test Accuracy: 55.96% ✅ (+10% improvement)**
- **Overfitting Gap: 35.85% ✅ (14.8% improvement)**
- **F1 Score: 0.5737**

### Model Configuration
```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=10,
    reg_alpha=1,
    random_state=42
)
```

---

## 🔬 Methodology

### 1. Data Collection (2000-2025)
- 25 years of daily financial data
- 6,307 trading days
- 18 base features (9 sectors + 9 macro)

### 2. Regime Detection
- VIX-based classification into 3 market states:
  - **Regime 0 (Low Vol):** VIX < 20 → 62.4% of time
  - **Regime 1 (Medium Vol):** 20 ≤ VIX < 30 → 27.8%
  - **Regime 2 (High Vol):** VIX ≥ 30 → 9.8%

### 3. Feature Engineering
- **Temporal windows:** 5, 10, 21, 42, 63, 126, 252 days
- **Technical indicators:** RSI, MACD, Bollinger Bands, Stochastic, ADX
- **Cross-asset features:** Correlations, betas
- **Anti-leakage:** All features lagged by 1 day

### 4. Feature Selection
- **Method:** SHAP (SHapley Additive exPlanations)
- **Strategy:** Select top 100 features by mean |SHAP value|
- **Result:** 90.8% feature reduction with performance improvement

### 5. Model Training
- **Algorithm:** XGBoost (Gradient Boosting)
- **Validation:** 5-Fold Time Series Cross-Validation
- **Optimization:** Regularization (L1=1, L2=10) to prevent overfitting

---

## 📊 Key Insights

### What Works
1. ✅ **Long-term volatility** (126, 252 days) more predictive than short-term
2. ✅ **Technical indicators** (RSI, MACD, ADX) capture momentum
3. ✅ **Cross-sector correlations** detect rotation patterns
4. ✅ **Yield spread dynamics** (ADX on SPREAD) signal regime shifts
5. ✅ **Utilities sector volatility** (XLU) as defensive indicator

### What Doesn't Work
1. ❌ Short-term features (5-10 day windows) - mostly noise
2. ❌ Temporal features (day of week, month) - no edge
3. ❌ Deep models (depth>4) - severe overfitting
4. ❌ Too many features (>400) - curse of dimensionality

---

## 📦 Dependencies

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.42.0
matplotlib>=3.7.0
statsmodels>=0.14.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Development

### Run Tests
```bash
pytest tests/
```

### Run Diagnostics
```bash
python diagnostics/xgb_diagnostic_v2.py
python diagnostics/fast_shap_selection.py
```

### Code Structure
- **Production code:** `code/` (in git)
- **Diagnostics:** `diagnostics/` (excluded from git)
- **Results:** `temporal_files/` (excluded from git)

---

## 📝 Citation

If you use this work, please cite:

```
[Author Name] (2025). Sector Classification Using Machine Learning: 
Forecasting XLK Performance Based on Economic Uncertainty.
[University], Dissertation.
```

---

## 📄 License

[Specify License]

---

## 👤 Author

Thomas Patramanis  
[University]  
[Email]

---

## 🙏 Acknowledgments

- Economic Policy Uncertainty data from [Baker, Bloom & Davis](https://www.policyuncertainty.com/)
- Geopolitical Risk data from [Caldara & Iacoviello](https://www.matteoiacoviello.com/gpr.htm)
- Yahoo Finance for sector ETF data

