from __future__ import annotations

import json
from typing import Any

import pandas as pd
import xlwings as xw

from ..data.fetchers import fetch_price_history
from ..factors.compute import DEFAULT_UNIVERSE, compute_factors
from ..optimize.constraints import Constraints
from ..optimize.optimizer import optimize


@xw.func
def PO_METRICS(ticker: str) -> str:
    # Return a small JSON metrics blob for a single ticker
    px = fetch_price_history(
        [ticker], start="2022-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    f = compute_factors(px)
    if ticker not in f.index:
        return json.dumps({})
    row = f.loc[ticker].to_dict()
    return json.dumps(row)


@xw.func
def PO_OPTIMIZE(tickers_range, params_json: str = "{}") -> Any:
    try:
        tickers = [t for t in tickers_range if t]
        params = json.loads(params_json or "{}")
    except Exception:
        tickers = DEFAULT_UNIVERSE
        params = {}
    px = fetch_price_history(
        tickers, start="2022-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    f = compute_factors(px)
    cons = Constraints(
        target_names=int(params.get("target_names", 6)),
        min_value_exposure=float(params.get("min_value", 0.3)),
        min_momentum_exposure=float(params.get("min_momo", 0.2)),
        single_name_cap=float(params.get("single_cap", 0.25)),
        sector_cap=float(params.get("sector_cap", 0.35)),
    )
    res = optimize(f, px, cons)
    # Return as 2D array for Excel: ticker, weight
    return [[t, float(res.weights.loc[t])] for t in res.tickers]
