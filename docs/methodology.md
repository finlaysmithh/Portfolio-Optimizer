# Methodology

This project implements a practical factor-based portfolio optimizer with realistic constraints and costs. Key elements:

## Factors

- Value: proxies like EBIT/EV, B/P (offline default uses synthetic standings)
- Quality: ROIC, gross margins, accruals (offline default uses synthetic standings)
- Momentum: 6–12m excluding most recent month, based on price momentum
- Size: market cap proxy (offline via standing proxy)
- Low-Vol: inverse realized volatility

All factors are standardized to z-scores and then combined into a composite.

## Screening

- Liquidity: average daily dollar volume (ADV) over the past month must exceed threshold (default $10M)
- Composite: keep top X names
- Sector diversification: simple cap per sector (default ≤35%) when sector data is available; with offline sample data, we simulate sector tags

## Risk Model

- Covariance: EWMA with decay λ (default 0.94)
- Shrinkage: Ledoit-Wolf style shrinkage fallback when estimation is unstable (implemented in-house to avoid heavy deps)
- Metrics: annualized volatility, beta vs SPY, Sharpe (rf configurable), 1y parametric VaR (with Cornish-Fisher option), max drawdown, R² vs benchmark
- Regime overlay: VIX percentile (e.g., >80th) tightens max weight and raises min cash

## Optimization

- Objective: maximize expected Sharpe (proxy via mean-variance) or mean-variance with risk aversion
- Constraints: long-only, fully invested, 5–8 names; factor exposure minimums (e.g., w·value ≥ 0.30, w·momentum ≥ 0.20); sector and single-name caps; turnover penalty to reduce churn
- Implementation: attempts PyPortfolioOpt, falls back to deterministic heuristic satisfying constraints

## Backtesting

- Event-time monthly rebalances
- Transaction costs: default 5 bps per unit turnover
- Attribution: basic allocation/selection effects vs SPY
- MC Stress: block bootstrap of returns to estimate tail risk (drawdown, VaR/ES distribution)

## References

- EWMA (RiskMetrics): σ²_t = λσ²_{t-1} + (1-λ)r²_t
- Ledoit-Wolf shrinkage: Σ* = δT + (1-δ)S
- Cornish-Fisher expansion for VaR adjustment
