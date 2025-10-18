# portfolio-optimizer-pro Docs

This documentation covers methodology, API reference, system architecture, and interview-ready context. The project is designed to run offline by default with synthetic data and deterministic seeds, and to transparently upgrade to live providers when API keys are provided.

- See methodology: `docs/methodology.md`
- API reference: `docs/api_reference.md`
- Interview brief (one page): `docs/interview_brief.md`
- Images are generated to `docs/images/`

## Modules

- Data: Multi-provider fetch with failover + disk cache
- Factors: Value, Quality, Momentum, Size, Low-Vol (z-scores)
- Risk: EWMA covariance, shrinkage, metrics, regimes, liquidity
- Optimize: Mean-variance/Sharpe proxy with constraints and turnover penalty
- Backtest: Monthly rebalance, costs, performance attribution, MC stress
- Reporting: Tear-sheet PDF and email alerts
- UIs: Streamlit and Excel (xlwings)

## CLI

- `po fetch` — populate or refresh cached data
- `po screen` — screen universe by factors and liquidity
- `po optimize` — produce 5–8 name portfolio with constraints
- `po backtest` — run a basic backtest with costs
- `po tearsheet` — export PDF tear sheet

Offline defaults ensure CI runs cleanly without network.
