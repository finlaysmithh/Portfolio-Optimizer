from __future__ import annotations

import pandas as pd


def simple_allocation_selection(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    # Very rough placeholder: split excess return into allocation/selection equally
    ex = (portfolio_returns - benchmark_returns.reindex_like(portfolio_returns).fillna(0)).dropna()
    total = float(ex.sum())
    return {"allocation": total / 2.0, "selection": total / 2.0}
