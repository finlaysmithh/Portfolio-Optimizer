from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ..utils import ensure_datetime_index, to_returns
from ..forecast.covariance import project_ewma_cov, dcc_garch_cov_forecast, DCCGarchConfig
from ..risk.covariance import robust_covariance


RiskModel = Literal["ewma", "shrinkage", "forecasted_sigma", "dcc"]


@dataclass
class RiskConfig:
    use_dcc: bool = False
    horizon_days: int = 21


def forecasted_sigma(
    prices: pd.DataFrame, model: RiskModel = "forecasted_sigma", horizon_days: int = 21
) -> pd.DataFrame:
    rets = to_returns(ensure_datetime_index(prices))
    if model == "dcc":
        return dcc_garch_cov_forecast(rets, horizon=horizon_days, config=DCCGarchConfig())
    # default: EWMA projection
    return project_ewma_cov(rets, lam=0.94, horizon=horizon_days)


def historical_sigma(prices: pd.DataFrame) -> pd.DataFrame:
    rets = to_returns(ensure_datetime_index(prices))
    return robust_covariance(rets)
