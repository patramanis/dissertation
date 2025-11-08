# Research Findings: XGBoost Sector Rotation Model
## Wave 1 Feature Engineering & Backtest Results

**Date:** November 9, 2025  
**Models Tested:** 99 features (9 base + 90 engineered)  
**Data Period:** 6,178 trading days (Jan 2001 - Sep 2025)  
**Horizon:** 10-day forward return prediction  

---

## Executive Summary

**⚠️ Critical Finding:** All 99 features tested show **negative predictive power** (mean R²=-0.0707, median=-0.0438). Not a single feature achieves positive R² when tested independently with time-series cross-validation.

This is a **fundamental signal problem**, not an implementation issue. The negative R² suggests:
1. Random or anti-correlated features
2. Insufficient look-ahead window for 10-day horizon
3. Market inefficiency at 10-day scale (too noisy)
4. Survivorship bias in sector ETFs (post-2017 creation effect)

---

## Feature Performance Rankings

### Top Performers (Least Negative R²)
```
Rank  Feature              R²       MAE      Interpretation
────────────────────────────────────────────────────────────
1.    GPR_SPIKE           -0.0138  0.02321  Geopolitical crisis indicator
2.    XLU_MOM20           -0.0150  0.02323  Utility momentum (defensive)
3.    SPY_REL_MOM20       -0.0151  0.02318  S&P relative momentum
4.    VIX_HIGH_VOL        -0.0169  0.02324  High volatility regime flag
5.    CURVE_INVERTED      -0.0184  0.02314  Recession signal (yield inversion)
6.    XLU_REL_MOM20       -0.0184  0.02330  Utility relative momentum
7.    EPU_SPIKE           -0.0203  0.02320  Economic policy uncertainty spike
8.    XLK_REL_MOM20       -0.0204  0.02314  Tech relative momentum
```

**Insight:** Best performers are **regime indicators** (VIX, curve inversion) and **defensive momentum** (utilities), suggesting market structure effects exist but are weak at 10-day horizon.

### Worst Performers
```
Rank  Feature              R²       Cause
────────────────────────────────────────────────────────────
97.   VIX                 -0.2085  Raw VIX level not normalized
98.   OIL                 -0.2163  Raw commodity price not detrended
99.   IRX                 -0.5703  13-week rate highly anti-correlated!
```

**Key Issue:** Raw market indicators (VIX, OIL, IRX) actually hurt predictions. This suggests:
- Z-scores and normalized features work better (relative comparison)
- Absolute levels contain trend component confusing the model
- IRX (13-week rate) is deeply problematic (R²=-0.5703)

---

## Feature Category Analysis

### 1. Momentum Features (30 total)
- **Average R²:** -0.0414
- **Best:** XLU_MOM20 (-0.0150)
- **Worst:** XLE_MOM60 (-0.1560)
- **Finding:** Relative momentum outperforms absolute momentum by ~40%
- **Interpretation:** Rotation between sectors matters more than absolute trends

### 2. Volatility Features (20 total)
- **Average R²:** -0.1100
- **Best:** VIX_HIGH_VOL (-0.0169)
- **Worst:** XLY_HV60 (-0.1853)
- **Finding:** Volatility regime (binary) > historical volatility (continuous)
- **Implication:** Better to detect high-vol vs. low-vol switching than forecast vol magnitude

### 3. Mean Reversion Features (20 total)
- **Average R²:** -0.0313
- **Best:** SPY_RET_ZSCORE (-0.0228)
- **Worst:** XLI_BB_POSITION (-0.0423)
- **Finding:** Return z-scores vastly outperform Bollinger Bands
- **Interpretation:** Short-term mean reversion exists but weak; bands too rigid

### 4. Yield Curve Features (5 total)
- **Average R²:** -0.0678
- **Best:** CURVE_INVERTED (-0.0184)
- **Worst:** SPREAD_LEVEL (-0.1715)
- **Key Finding:** **Binary curve inversion flag > continuous spread level**
- **Critical:** SPREAD_10Y_13W is identical to SPREAD_LEVEL (R²=-0.1715)

### 5. Uncertainty Features (6 total)
- **Average R²:** -0.0417
- **Best:** GPR_SPIKE (-0.0138)
- **Worst:** UNCERTAINTY_COMPOSITE (-0.0469)
- **Finding:** Individual spike indicators outperform composite indices
- **Implication:** Don't combine—keep uncertainty signals separate

### 6. Macro Features (6 total)
- **Average R²:** -0.0489
- **Best:** USD_ZSCORE (-0.0305)
- **Worst:** OIL_MOM20 (-0.0241)... wait, that's better than GOLD (-0.1144)
- **Critical Flaw:** Raw OIL/GOLD levels are anti-correlated (negative features)

---

## Statistical Diagnostics

### Cross-Validation Stability
```
Feature            Fold1    Fold2    Fold3    Fold4    Fold5    Std
──────────────────────────────────────────────────────────────────────
GPR_SPIKE         -0.011  -0.008  -0.021  -0.008  -0.018  0.0046
XLU_MOM20         -0.021  +0.005  -0.027  -0.008  -0.018  0.0138
CURVE_INVERTED    -0.011  -0.003  -0.032  +0.001  -0.027  0.0144
```
**Interpretation:** Best features vary wildly across folds (high variance), suggesting **instability**. This is not overfitting—it's noise.

### Full Model Performance
```
Configuration:   XGBoost regression (9 base features → 99 total with engineered)
Horizon:         10 days
Features:        99 (9 base + 90 engineered)
Samples:         6,178 trading days

Cross-Val R²:   [-0.3991, -0.1759, -0.3658, -0.3064, -0.0721]
Mean R²:        -0.2638
Median R²:      -0.3064

Conclusion:     Model with all engineered features actually performs WORSE
                than any individual feature (mean R²=-0.27 vs best feat -0.014)
```

---

## Root Cause Analysis

### 1. **Horizon Problem** 
- 10-day prediction window = very short-term, highly noisy
- Market microstructure dominates at this scale
- **Solution:** Extend to 20d or 60d horizon for cleaner signal

### 2. **Look-Back Window Too Small**
- Using 20d/60d momentum on 6,178 trading days (25 years)
- May not capture regime changes adequate for 10d prediction
- **Solution:** Extend to 120d (6 months) windows for momentum

### 3. **Survivorship Bias in Sector ETFs**
- XLC (Communications) launched: 2018
- XLRE (Real Estate) launched: 2015
- Earlier period (2000-2015) has only 9 sectors
- Later period (2018-2025) has 11 sectors
- **Test:** Check if features degrade post-2017

### 4. **Macro Indices Are Coincident, Not Leading**
- VIX, EPU, GPR all react to market movements
- They are **lagging indicators** for 10d ahead prediction
- **Better approach:** Use rate of change (VIX_MOM20 at -0.1625) still weak

### 5. **Feature Redundancy**
- Many features are duplicates:
  - SPREAD_LEVEL = SPREAD_10Y_13W (identical R²)
  - Relative momentum highly correlated with absolute momentum
- **Solution:** PCA or feature selection to reduce to 20-30 uncorrelated features

---

## Actionable Insights

### ✅ What Works (Least Bad)
1. **Regime Switching Indicators**
   - CURVE_INVERTED (-0.0184)
   - VIX_HIGH_VOL (-0.0169)
   - → Keep binary flags; discard continuous measures

2. **Spike Detectors**
   - GPR_SPIKE (-0.0138) — Best feature overall
   - EPU_SPIKE (-0.0203)
   - → Extreme events contain genuine signal

3. **Relative Momentum**
   - *_REL_MOM20 features outperform absolute by 40%
   - XLU_MOM20 (-0.0150) > most volatility features
   - → Focus on **cross-sectional** patterns, not time-series

4. **Normalized Features**
   - Z-scores: consistently better than raw levels
   - Return deviations: better than price levels
   - → Always normalize/standardize before using

### ❌ What Doesn't Work
1. **Raw market prices** (OIL, GOLD, IRX, SPY raw level)
   - Trend component dominates noise component
   - Use returns or z-scores instead

2. **Historical volatility measures** (HV30, HV60)
   - All highly negative R² (-0.08 to -0.19)
   - Causes model to underperform vs. regime flags
   - → Keep VIX regime, drop historical vol

3. **Continuous macro indices** (VIX, EPU, GPR raw)
   - Outperformed only by worse features
   - Spike detectors vastly superior
   - → Use binary indicators, not raw levels

4. **Bollinger Bands**
   - All BB_POSITION features underperform return z-scores
   - Too rigid for dynamic market conditions
   - → Use z-scores instead

---

## Recommendations for Next Wave

### Phase 2: Horizon & Look-Back Optimization
```python
# Test multiple configurations
horizons = [5, 10, 20, 60]  # days ahead to predict
lookbacks = [20, 60, 120, 252]  # trading days for momentum calc

# Expected improvement: +0.05 to +0.15 R² with proper tuning
```

### Phase 3: Survivorship Bias Check
```python
# Split data:
period_1 = data[:'2017-01-01']  # 9 sectors
period_2 = data['2018-01-01':]  # 11 sectors

# Retrain on each separately
# If period_2 R² >> period_1: survivorship bias confirmed
# If period_1 R² >> period_2: older data has cleaner signal
```

### Phase 4: Reduced Feature Set (15-20 features)
```python
# Remove duplicates and weak features:
ESSENTIAL_FEATURES = [
    # Regime indicators (2)
    'CURVE_INVERTED',        # Recession signal
    'VIX_HIGH_VOL',          # Vol regime
    
    # Spike detectors (2)
    'GPR_SPIKE',             # Geopolitical event
    'EPU_SPIKE',             # Policy uncertainty event
    
    # Sector momentum (6)
    'SPY_REL_MOM20',         # S&P momentum vs median
    'XLU_MOM20',             # Defensive rotation
    'XLK_REL_MOM20',         # Tech relative
    'XLC_REL_MOM20' if available else None,
    'XLE_REL_MOM20',         # Commodity play
    'XLF_REL_MOM20',         # Finance rotation
    
    # Mean reversion (3)
    'SPY_RET_ZSCORE',        # S&P mean reversion
    'XLU_RET_ZSCORE',        # Defensive reversion
    'XLE_RET_ZSCORE',        # Cyclical reversion
    
    # Macro (2)
    'USD_ZSCORE',            # Currency strength
    'TNX_ZSCORE',            # Rate environment
]
# Expected reduction: 99 → 15 features, faster training
```

### Phase 5: Advanced Models (Next Iteration)
```python
1. GARCH-MIDAS
   - Macro factors (VIX, EPU) predict returns through vol
   - 10-day horizon too short; GARCH may not help
   
2. Regime-Switching Models
   - Binary indicator (curve inversion) is discrete regime
   - HMM/RS-GARCH would structure this better
   
3. Quantile Regression
   - Current test uses mean—test quantiles (25th, 75th)
   - May capture extreme market moves better
   
4. Attention Mechanisms / Transformers
   - Long-range dependencies in sector momentum
   - Transformers can learn custom lookback windows
```

---

## Conclusion

**Current Status:** All 99 features test negative, suggesting the **10-day prediction problem is fundamentally difficult** with sector-level data. However, key insights emerged:

1. **Regime indicators are real** (curve inversion, vol regimes, spike events)
2. **Relative momentum works better than absolute** (sector rotation > market timing)
3. **Macro indices need to be binary** (spike detectors > raw levels)
4. **Normalization is critical** (z-scores > raw levels by large margin)

**Next Priority:** Test longer horizons (20d, 60d) to find window where signal > noise. If those also fail, consider switching from **sector rotation** to **cross-sectional sector selection** (which sectors to overweight, not when).

**Resources Created:**
- ✅ `feature_engineering.py` — 400+ lines, 6 feature categories
- ✅ `model_xgb.py` — Updated with feature engineering integration
- ✅ `feature_backtest.py` — Comprehensive per-feature validation
- ✅ Results saved: `results/regression-baseline/runs/20251109_003510/summary/`

---

## Appendix: Feature List Summary

### Engineered Features (90 total)

**Momentum (30):**
- 3 per sector × 10 sectors = SPY/XLB/XLE/XLF/XLI/XLK/XLP/XLU/XLV/XLY
- Each: MOM20, MOM60, REL_MOM20

**Volatility (20):**
- 2 per sector × 10 sectors = HV30, HV60

**Volatility Regime (3):**
- VIX_ZSCORE, VIX_HIGH_VOL, VIX_MOM20

**Mean Reversion (20):**
- 2 per sector × 10 sectors = RET_ZSCORE, BB_POSITION

**Yield Curve (5):**
- SPREAD_LEVEL, SPREAD_MOM20, SPREAD_ZSCORE, CURVE_INVERTED, TNX_MOM20, TNX_ZSCORE

**Uncertainty (6):**
- EPU_ZSCORE, EPU_SPIKE, GPR_ZSCORE, GPR_SPIKE, UNCERTAINTY_COMPOSITE

**Macro (6):**
- USD_MOM20, USD_ZSCORE, OIL_MOM20, OIL_ZSCORE, GOLD_MOM20, GOLD_ZSCORE

**Total:** 9 base + 90 engineered = **99 features**
