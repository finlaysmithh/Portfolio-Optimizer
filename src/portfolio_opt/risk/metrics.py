from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import ensure_datetime_index


def annualize_vol(returns: pd.Series, periods: int = 252) -> float:
    return float(np.sqrt(periods) * np.nanstd(returns, ddof=1))


def beta_vs(returns: pd.Series, benchmark: pd.Series) -> float:
    x = np.cov(returns.dropna(), benchmark.dropna())
    if x.shape == (2, 2) and x[1, 1] != 0:
        return float(x[0, 1] / x[1, 1])
    return float("nan")


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    ex = returns - rf / periods
    mu = np.nanmean(ex)
    sd = np.nanstd(ex, ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(periods) * mu / sd)


def parametric_var(returns: pd.Series, level: float = 0.95, periods: int = 252) -> float:
    mu, sd = np.nanmean(returns), np.nanstd(returns, ddof=1)
    z = 1.64485  # ~N(0,1) 95%
    return float(-(mu * periods + z * sd * np.sqrt(periods)))


def max_drawdown(prices: pd.Series) -> float:
    prices = ensure_datetime_index(prices)
    peaks = prices.cummax()
    dd = (prices - peaks) / peaks
    return float(dd.min())


def r_squared(returns: pd.Series, benchmark: pd.Series) -> float:
    x = returns.dropna()
    y = benchmark.reindex_like(x).dropna()
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return float(c * c)


def compute_metrics(
    returns: pd.Series, benchmark: pd.Series | None = None, rf: float = 0.0
) -> dict:
    out = {
        "ann_vol": annualize_vol(returns),
        "sharpe": sharpe_ratio(returns, rf=rf),
        "var_95": parametric_var(returns),
    }
    if benchmark is not None:
        out["beta"] = beta_vs(returns, benchmark)
        out["r2"] = r_squared(returns, benchmark)
    return out
