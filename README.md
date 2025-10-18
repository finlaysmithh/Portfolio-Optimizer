# portfolio-optimizer-pro
![Image 18-10-2025 at 15 42](https://github.com/user-attachments/assets/62167990-917a-47fe-9e5a-796840acdbf4)
![Image 18-10-2025 at 20 46](https://github.com/user-attachments/assets/cae8e93e-aee6-4375-9afa-8d0274e62ef3)

5–8 stock portfolio optimizer targeting ≥3% alpha vs S&P 500 over 1–2 years with factor constraints, regime overlay, liquidity guards, realistic costs, and backtesting. Production-grade repository with a Streamlit research app, Excel UI, robust Python library, tests, CI/CD, docs, and a marketing site.

- License: MIT
- Python: 3.10+
- Tech: pandas, numpy, scipy, statsmodels, scikit-learn, PyPortfolioOpt, yfinance, Alpha Vantage, FMP, Plotly/Matplotlib, xlwings, Streamlit

## Quickstart

- Prereqs: Python 3.10+, Node 18+ (for marketing site), Make, Git

1) Initialize environment

```
make init
```

2) Run tests

```
make test
```

3) Launch Streamlit app (local)

```
make app
```

4) Install/register Excel add-in (optional)

```
make excel
```

5) CLI examples

```
po fetch --universe sp500 --since 2018-01-01
po screen --top 30
po optimize --target-names 8 --min-value 0.3 --min-momo 0.2
po backtest --since 2018-01-01 --rebalance monthly --cost-bps 5
po tearsheet --out docs/images/tearsheet.pdf
```

## Repo Structure

See docs/index.md for full walkthrough. Key areas:

- `src/portfolio_opt/` — Python library (data, factors, risk, optimize, backtest, reporting, excel, cli)
- `streamlit_app/` — Research UI
- `site/` — Marketing site (Vite + React)
- `docs/` — Methodology, API reference, interview brief, and images
- `.github/workflows/` — CI/CD, monthly rebalance action, Docker

## Secrets & Config

- Copy `env/.env.example` to `.env` in the project root or `env/.env` and set:
  - `ALPHAVANTAGE_API_KEY`, `FMP_API_KEY`, `EMAIL_USER`, `EMAIL_APP_PASSWORD`
- The library loads env via `python-dotenv` and validates presence where required. All code handles missing keys gracefully (uses failover provider or offline synthetic sample data).

## Determinism and Offline Defaults

- The library contains a small, built-in synthetic dataset and caching to ensure the full pipeline runs offline and in CI without network.
- Default universe: a 30-name S&P-like subset. Optimizer outputs 5–8 names by default and satisfies factor constraints in tests.

## Makefile Targets

- `init` — Create venv, install package (dev extras), pre-commit style tools
- `test` — Run linters, type-checkers, and pytest with coverage
- `app` — Launch Streamlit app
- `excel` — Install xlwings add-in locally
- `format` — Run black + ruff
- `typecheck` — Run mypy
- `lint` — Run ruff and bandit

## Docker

- Build & run with docker-compose (library + streamlit). See `docker-compose.yml`.

## CI/CD

- GitHub Actions: lint, typecheck, tests, coverage; Docker image build.
- Scheduled monthly rebalance: runs CLI, uploads PDF tear sheet to release, optional email summary via yagmail.

## Documentation

- `docs/methodology.md` — formulas (EWMA, Ledoit-Wolf), constraint math, VaR, regime rules
- `docs/interview_brief.md` — one-pager with architecture diagram (mermaid) and data-flow
- `docs/api_reference.md` — public functions and CLI

## Status

This repo is interview-ready and runnable end-to-end on first run with offline data. Replace synthetic data with your provider keys when ready for production.
