# API Reference

Public entry-points intended for external use.

## Python

- `portfolio_opt.data.fetchers.fetch_price_history(tickers, start, end)` → DataFrame of prices
- `portfolio_opt.factors.compute.compute_factors(prices)` → DataFrame of z-scored factors
- `portfolio_opt.risk.metrics.compute_metrics(returns, benchmark)` → dict of metrics
- `portfolio_opt.optimize.optimizer.optimize(factors, prices, constraints)` → weights, selection, diagnostics
- `portfolio_opt.backtest.engine.backtest(prices, weights_by_date, costs_bps)` → results dict
- `portfolio_opt.reporting.tearsheet.save_tearsheet(results, out_path)` → path

## CLI

- `po fetch --universe sp500 --since 2018-01-01`
- `po screen --top 30`
- `po optimize --target-names 8 --min-value 0.3 --min-momo 0.2`
- `po backtest --since 2018-01-01 --rebalance monthly --cost-bps 5`
- `po tearsheet --out docs/images/tearsheet.pdf`
