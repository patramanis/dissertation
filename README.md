# Sector Rotation with Machine Learning and Risk Filtering Mechanisms

This repository contains the implementation of a dissertation that systematically investigates the effectiveness of sector rotation strategies combining machine learning techniques with risk filtering mechanisms, aiming to optimize the risk-return relationship in sector ETF investing.

## Research Overview

The central hypothesis examined is that machine learning models, despite their ability to recognize complex patterns of relative performance, require complementary exposure control mechanisms to stabilize returns under real market conditions. The theoretical foundation is grounded in four distinct streams of financial literature: sectoral heterogeneity and cross-sectional predictability, momentum strategies, regime-switching models, and modern applications of machine learning in finance.

## Methodology

The methodology comprises a ten-stage processing pipeline, designed for avoidance of look-ahead bias through Point-in-Time data alignment. The investment universe consists of nine SPDR sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) representing major S&P 500 sectors, with an analysis period spanning from September 2000 to September 2025. The target variable is defined as the excess logarithmic return of each sector versus the SPY benchmark. The final feature set comprises ~450 predictors combining technical measures, cross-sectional ranks for robustness, and rolling correlation exposures to macro/uncertainty drivers across multiple windows, with an explicit focus on capturing uncertainty and regime shifts. The evaluation spans 39 rolling folds per horizon (and 38 folds for h=63 due to endpoint data constraints).

For training the predictive model, the AutoGluon-Tabular framework was employed, which automatically selects, trains, and combines multiple machine learning algorithms (LightGBM, XGBoost, CatBoost, Random Forest) into ensemble models. Input features include technical indicators (relative momentum, rolling beta, idiosyncratic volatility, maximum drawdown, semi-variance), rolling correlations with seven macroeconomic drivers (interest rates, oil, dollar, bonds, VIX, credit spreads, gold) across six time windows, as well as cross-sectional ranks for outlier robustness. Cross-validation was implemented using a rolling, purged time-series scheme with five years of training, with 15% of them used as validation period, and one year of test per fold, with purge gaps to prevent leakage from overlapping labels.

The model is trained to predict sector excess log-returns versus SPY and is primarily evaluated as a cross-sectional ranking signal (which sector should outperform peers over the next h days). Portfolio construction maps predictions into periodic sector selection (top-ranked sectors) and applies a second layer of exposure control (filters/gates) that can reduce or scale market exposure under adverse regimes.

## Filtering Mechanisms

The central contribution lies in the comparative evaluation of fourteen different signal filtering mechanisms, categorized into four families: simple momentum filters (unfiltered, relative momentum, absolute momentum, dual momentum, pure momentum without ML), regime filters (three moving average voting system, regime with inverse volatility weighting), dynamic regime filters (three-zone VIX system, composite risk scalar, streak decay system, layered adaptive system), and hybrid filters (trend plus VIX combination, VIX adaptive, VIX z-score). Evaluation was conducted across three investment horizons (weekly h=5, monthly h=21, quarterly h=63), yielding 42 unique strategy-horizon combinations.
Filters operate as meta-layers on top of the ML ranking signal and serve two roles: (i) timing/gating, deciding whether the strategy is active or defensive, and (ii) exposure scaling, adjusting the magnitude of risk-taking. Regime-based variants identify risk regimes using moving-average voting rules and volatility conditions; “InVol” variants further apply inverse-volatility weighting to reduce concentration during stress. VIX-based filters include both absolute-threshold rules and adaptive (distribution-aware) variants such as rolling z-score normalizations and multi-zone risk states.

## Validation

To ensure objectivity, triple external validation was applied using three independent industry-standard tools: VectorBT for ground truth backtesting (Sharpe, Sortino, Calmar, Maximum Drawdown), Alphalens for factor quality analysis (Information Coefficient, Information Ratio), and QuantStats for risk tearsheet generation (Alpha, Beta, VaR, CVaR).

Backtests incorporate a 1-way transaction cost of 10 bps and a 2.5% annual risk-free rate when computing risk-adjusted metrics (e.g., Sharpe). Rebalancing is performed at the end of each holding period consistent with the prediction horizon (h ∈ {5, 21, 63} trading days).

## Model Evaluation

RMSE is reported per horizon for LightGBM, XGBoost, CatBoost, Random Forest, and the second-level weighted ensemble. While RMSE increases mechanically with horizon length, ranking diagnostics show that signal quality is strongly horizon-dependent: the long-horizon (h=63) exhibits statistically significant IC and exceeds the practical “investable” IR threshold, whereas the short-horizon (h=5) remains close to a random baseline.
IC mean increases from 0.0035 (t=0.52) at h=5 to 0.0109 (t=1.70) at h=21 and 0.0317 (t=5.09) at h=63, with IC IR rising to 0.081 at h=63.
Practical top-K selection: Hit@3 remains near the random baseline (~33.3%) but Lift@K turns consistently positive, peaking at h=63.
The dissertation documents that machine learning produces an exploitable sector selection signal, but its conversion into a sustainable investment strategy requires exposure control mechanisms based on regime recognition.

## Filter Evaluation

Results show strong horizon dependence: weekly horizons (h=5) are dominated by noise, while h=21 and h=63 deliver materially better risk-adjusted profiles. Average strategy Sharpe increases from 0.036 (h=5) to 0.313 (h=21) and 0.318 (h=63), while average alpha moves from −0.021 to +0.012/+0.011. Regime-based filters provide the strongest drawdown protection, with Regime achieving Calmar 0.437 versus 0.197 for SPY and reducing maximum drawdown to −14.7% versus −55.2%. In efficiency terms, Trend VIX attains the highest mean Sharpe (0.420), while VIX Adaptive is the most aggressive profile and reaches the highest Sharpe in the long horizon (h=63 Sharpe 0.52). The unfiltered strategy confirms the central hypothesis: without exposure control, the ML signal remains vulnerable to crash regimes and can exhibit worse drawdowns than the benchmark.

## Project Structure

The project is organized into the following directories: configs (configuration files for cross-validation, models, and default parameters), data (raw market data, processed features, labels, and panel datasets), scripts (pipeline execution scripts from data alignment to model training), src (source code for backtests, cross-validation, data processing, features, labels, and utilities), diagnostics (verification and analysis tools), and tests (leakage detection and validation tests).

## Pipeline Stages

The processing pipeline follows ten stages: raw data ingestion (SPY, SPDR, Futures, Macro), Point-in-time alignment enforcement via publication-lag policies and conservative forward-filling rules, label inputs extraction, base stationary transforms (log-returns, differences), label construction (excess returns), technical features computation, correlation features calculation, panel building with inner join, cross-validation fold generation with purge gaps, and AutoGluon training with backtest execution.

## Requirements

The implementation requires Python 3.12 with dependencies including AutoGluon-Tabular for automated machine learning, VectorBT for backtesting, Alphalens for factor analysis, QuantStats for performance analytics, and standard scientific computing libraries (pandas, numpy, scipy).
