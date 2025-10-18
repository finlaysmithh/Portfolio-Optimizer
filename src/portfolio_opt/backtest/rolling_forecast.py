from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ..utils import ensure_datetime_index, to_returns
from ..integration.expected import ExpectedConfig, forecasted_mu, historical_mean_mu
from ..integration.risk import forecasted_sigma, historical_sigma


@dataclass
class PortfolioForecastBacktest:
    dates: list[pd.Timestamp]
    weights: dict[pd.Timestamp, pd.Series]
    realized_returns: pd.Series
    benchmark_returns: pd.Series | None


def walk_forward_optimize(
    prices: pd.DataFrame,
    rebalance_freq: str,
    expected_cfg: ExpectedConfig,
    risk_model: str = "forecasted_sigma",
    optimizer_func: Callable[[pd.Series, pd.DataFrame], pd.Series] | None = None,
    benchmark: pd.Series | None = None,
) -> PortfolioForecastBacktest:
    """Walk-forward: at each rebalance date, forecast mu/Sigma, optimize, hold 1 step.

    optimizer_func produces weights given (mu, Sigma). Defaults to equal-weight.
    """
    prices = ensure_datetime_index(prices)
    rets = to_returns(prices)
    if optimizer_func is None:
        def optimizer_func(mu: pd.Series, sigma: pd.DataFrame) -> pd.Series:  # type: ignore[misc]
            n = len(mu)
            return pd.Series(1.0 / n, index=mu.index)

    rebal_dates = rets.resample(rebalance_freq).first().index
    weights: dict[pd.Timestamp, pd.Series] = {}
    realized: list[tuple[pd.Timestamp, float]] = []
    bench_realized: list[tuple[pd.Timestamp, float]] = []

    for i, d in enumerate(rebal_dates[:-1]):
        hist = prices.loc[:d]
        horizon = (rebal_dates[i + 1] - d).days
        horizon = max(1, min(expected_cfg.horizon_days, horizon))
        # Forecast mu
        mu_table, mu_array = forecasted_mu(hist, list(hist.columns), expected_cfg)
        mu_fallback = pd.Series(mu_array, index=mu_table.index, dtype=float)
        horizon_label = f"mean(d{horizon})"
        if horizon_label in mu_table.columns:
            mu = mu_table[horizon_label].reindex(hist.columns).astype(float)
        else:
            mu = mu_fallback.reindex(hist.columns).astype(float)
        mu = mu.fillna(0.0)
        # Forecast Sigma
        if risk_model == "forecasted_sigma" or risk_model == "dcc":
            sigma = forecasted_sigma(hist, model=risk_model, horizon_days=horizon)
        else:
            sigma = historical_sigma(hist)
        # Optimize
        w = optimizer_func(mu, sigma)
        w = w.reindex(hist.columns).fillna(0.0)
        w = w.clip(lower=0.0)
        w = w / (w.sum() or 1.0)
        weights[d] = w
        # Realized return until next rebalance
        seg = rets.loc[(rets.index >= d) & (rets.index < rebal_dates[i + 1])]
        for t, row in seg.iterrows():
            realized.append((t, float(np.dot(w.values, row.fillna(0.0).values))))
            if benchmark is not None:
                bench_realized.append((t, float(benchmark.reindex(seg.index).get(t, np.nan))))

    rs = pd.Series({t: r for t, r in realized}).sort_index()
    br = pd.Series({t: r for t, r in bench_realized}).sort_index() if bench_realized else None
    return PortfolioForecastBacktest(dates=list(weights.keys()), weights=weights, realized_returns=rs, benchmark_returns=br)
