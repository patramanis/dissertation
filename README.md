# Sector Rotation with Machine Learning and Risk Filtering Mechanisms

This repository contains the implementation of a dissertation that systematically investigates the effectiveness of sector rotation strategies combining machine learning techniques with risk filtering mechanisms, aiming to optimize the risk-return relationship in sector ETF investing.

## Research Overview

The central hypothesis examined is that machine learning models, despite their ability to recognize complex patterns of relative performance, require complementary exposure control mechanisms to stabilize returns under real market conditions. The theoretical foundation is grounded in four distinct streams of financial literature: sectoral heterogeneity and cross-sectional predictability, momentum strategies, regime-switching models, and modern applications of machine learning in finance.

## Methodology

The methodology comprises a ten-stage processing pipeline, designed for avoidance of look-ahead bias through Point-in-Time data alignment. The investment universe consists of nine SPDR sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) covering the S&P 500 index, with an analysis period spanning from September 2000 to September 2025. The target variable is defined as the excess logarithmic return of each sector versus the SPY benchmark.

For training the predictive model, the AutoGluon-Tabular framework was employed, which automatically selects, trains, and combines multiple machine learning algorithms (LightGBM, XGBoost, CatBoost, Random Forest) into ensemble models. Input features include technical indicators (relative momentum, rolling beta, idiosyncratic volatility, maximum drawdown, semi-variance), rolling correlations with seven macroeconomic drivers (interest rates, oil, dollar, bonds, VIX, credit spreads, gold) across six time windows, as well as cross-sectional ranks for outlier robustness. Cross-validation was implemented using rolling windows of five-year duration with purge gaps to prevent information leakage through overlapping labels.

## Filtering Mechanisms

The central contribution lies in the comparative evaluation of fourteen different signal filtering mechanisms, categorized into four families: simple momentum filters (unfiltered, relative momentum, absolute momentum, dual momentum, pure momentum without ML), regime filters (three moving average voting system, regime with inverse volatility weighting), dynamic regime filters (three-zone VIX system, composite risk scalar, streak decay system, layered adaptive system), and hybrid filters (trend plus VIX combination, VIX adaptive, VIX z-score). Evaluation was conducted across three investment horizons (weekly h=5, monthly h=21, quarterly h=63), yielding 42 unique strategy-horizon combinations.

## Validation

To ensure objectivity, triple external validation was applied using three independent industry-standard tools: VectorBT for ground truth backtesting (Sharpe, Sortino, Calmar, Maximum Drawdown), Alphalens for factor quality analysis (Information Coefficient, Information Ratio), and QuantStats for risk tearsheet generation (Alpha, Beta, VaR, CVaR).

## Key Findings

The results confirm that the investment horizon constitutes a structural parameter: the short-term horizon (h=5) proves unsuitable for sector rotation strategies, while the medium-term (h=21) and long-term (h=63) horizons converge to a superior performance profile. Regime filters dominate in risk management, with the Regime strategy achieving the highest Calmar Ratio (0.437), more than double that of the SPY benchmark (0.197), and dramatic Maximum Drawdown reduction to negative 14.7 percent versus negative 55.2 percent for SPY. The dissertation documents that machine learning produces an exploitable sector selection signal, but its conversion into a sustainable investment strategy requires exposure control mechanisms based on regime recognition.

## Project Structure

The project is organized into the following directories: configs (configuration files for cross-validation, models, and default parameters), data (raw market data, processed features, labels, and panel datasets), scripts (pipeline execution scripts from data alignment to model training), src (source code for backtests, cross-validation, data processing, features, labels, and utilities), diagnostics (verification and analysis tools), and tests (leakage detection and validation tests).

## Pipeline Stages

The processing pipeline follows ten stages: raw data ingestion (SPY, SPDR, Futures, Macro), Point-in-Time alignment with publication lags, label inputs extraction, base stationary transforms (log-returns, differences), label construction (excess returns), technical features computation, correlation features calculation, panel building with inner join, cross-validation fold generation with purge gaps, and AutoGluon training with backtest execution.

## Requirements

The implementation requires Python 3.12 with dependencies including AutoGluon-Tabular for automated machine learning, VectorBT for backtesting, Alphalens for factor analysis, QuantStats for performance analytics, and standard scientific computing libraries (pandas, numpy, scipy).