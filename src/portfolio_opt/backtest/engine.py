from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from .costs import trading_cost, turnover as turnover_func
from ..utils import ensure_datetime_index, to_returns


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    weights_by_date: dict[pd.Timestamp, pd.Series]
    turnover: pd.Series


def backtest(
    prices: pd.DataFrame,
    rebalance_freq: str = "M",
    cost_bps: float = 5.0,
    weight_generator=None,
) -> BacktestResult:
    prices = ensure_datetime_index(prices)
    rets = to_returns(prices)
    if rets.empty:
        raise ValueError("Not enough data")

    if weight_generator is None:
        # Equal-weight all names as a placeholder
        def weight_generator(_date: pd.Timestamp) -> pd.Series:
            n = prices.shape[1]
            return pd.Series(1.0 / n, index=prices.columns)

    rebal_dates = rets.resample(rebalance_freq).first().index
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    port_ret = []
    t_over = []
    prev_w = None
    for i, d in enumerate(rebal_dates):
        w = weight_generator(d)
        w = w.clip(lower=0)
        w = w / w.sum()
        weights_by_date[d] = w
        # Apply returns until next rebalance
        if i < len(rebal_dates) - 1:
            seg = rets.loc[(rets.index >= d) & (rets.index < rebal_dates[i + 1])]
        else:
            seg = rets.loc[rets.index >= d]
        for t, row in seg.iterrows():
            gross = float(np.dot(w.values, row.fillna(0.0).values))
            if prev_w is None:
                tc = 0.0
            else:
                to = turnover_func(prev_w, w)
                tc = trading_cost(to, cost_bps)
            port_ret.append((t, gross - tc))
            prev_w = w
        t_over.append((d, 0.0 if prev_w is None else 0.0))

    pr = pd.Series({t: r for t, r in port_ret}).sort_index()
    eq = (1 + pr).cumprod()
    to_series = pd.Series({t: v for t, v in t_over})
    return BacktestResult(
        equity_curve=eq, returns=pr, weights_by_date=weights_by_date, turnover=to_series
    )
