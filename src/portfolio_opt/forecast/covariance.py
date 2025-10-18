from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ..risk.ewma import ewma_cov


def project_ewma_cov(
    returns: pd.DataFrame, lam: float = 0.94, horizon: int = 21
) -> pd.DataFrame:
    """Project EWMA covariance to a horizon under iid daily returns assumption.

    For arithmetic returns, horizon aggregation approximates linearly: Var_T ≈ T * Var_1.
    Validates inputs and raises a clear error if returns are empty or EWMA fails.
    """
    if returns is None or returns.empty:
        raise ValueError("No returns available for the selected date range/tickers.")
    daily_cov = ewma_cov(returns, lam=lam)
    if daily_cov is None or getattr(daily_cov, "size", 0) == 0:
        raise ValueError("EWMA produced empty covariance; check data cleaning and date range.")
    n = int(daily_cov.shape[0])
    if daily_cov.shape != (n, n):
        raise ValueError(f"EWMA covariance shape mismatch: {daily_cov.shape}")
    return daily_cov * float(horizon)


def _try_import_arch():
    try:
        import arch  # type: ignore
        return arch
    except Exception:
        return None


@dataclass
class DCCGarchConfig:
    dist: Literal["normal", "t"] = "normal"
    p: int = 1
    q: int = 1


def dcc_garch_cov_forecast(
    returns: pd.DataFrame, horizon: int = 21, config: DCCGarchConfig | None = None
) -> pd.DataFrame:
    """Optional DCC-GARCH covariance forecast using arch.

    This is a light wrapper; if arch is not installed, raises ImportError.
    """
    arch = _try_import_arch()
    if arch is None:
        raise ImportError("arch package not installed; set use_dcc=false or install arch")

    from arch.univariate import ConstantMean, GARCH, StudentsT, Normal
    from arch.multivariate import DynamicConditionalCorrelation

    r = returns.dropna().astype(float)
    if r.shape[1] < 2:
        # Univariate GARCH then scale to covariance
        am = ConstantMean(r.iloc[:, 0] * 100)
        am.volatility = GARCH(1, 1)
        am.distribution = StudentsT() if (config and config.dist == "t") else Normal()
        res = am.fit(disp="off")
        forecasts = res.forecast(horizon=horizon)
        var = forecasts.variance.values[-1, -1] / (100 ** 2)
        return pd.DataFrame([[var]], index=r.columns, columns=r.columns)

    # Multivariate DCC
    vols = []
    for col in r.columns:
        am = ConstantMean(r[col] * 100)
        am.volatility = GARCH(1, 1)
        am.distribution = StudentsT() if (config and config.dist == "t") else Normal()
        vols.append(am)
    dcc = DynamicConditionalCorrelation(vols)
    res = dcc.fit(disp="off")
    f = res.forecast(horizon=horizon)
    # Use last horizon step
    cov = f.covariance.values[-1, -1] / (100 ** 2)
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)
