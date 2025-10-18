from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from .base import Forecaster, _ensure_series, _quantiles_from_normal


MIN_HISTORY = 50


@dataclass
class ETSForecaster(Forecaster):
    """Exponential smoothing forecaster (statsmodels Holt-Winters).

    Designed for daily returns where seasonality is typically absent. We try
    a small set of trend/no-trend configs and pick the one with lowest AIC.
    """

    add_trend: bool = False
    fitted_: Any | None = None
    resid_std_: float | None = None

    def fit(self, returns: pd.Series, exog: pd.DataFrame | None = None) -> "ETSForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        r = _ensure_series(returns)
        if len(r) < MIN_HISTORY:
            raise ValueError(f"Need at least {MIN_HISTORY} observations for ETS, got {len(r)}")
        candidates: list[tuple[str | None, float, Any]] = []
        for trend in (None, "add"):
            try:
                model = ExponentialSmoothing(r, trend=trend, seasonal=None, initialization_method="estimated")
                fitted = model.fit(optimized=True)
                aic = float(getattr(fitted, "aic", np.inf))
                resid = r - fitted.fittedvalues
                resid_std = float(np.nanstd(resid, ddof=1) or 1e-6)
                candidates.append((trend, aic, (fitted, resid_std)))
            except Exception:
                continue
        if not candidates:
            # Fallback: Gaussian white noise estimate
            self.fitted_ = None
            self.resid_std_ = float(np.nanstd(r, ddof=1) or 1e-6)
            return self
        best = sorted(candidates, key=lambda x: x[1])[0]
        self.add_trend = best[0] == "add"
        self.fitted_, self.resid_std_ = best[2]
        return self

    def predict(self, horizon: int, exog_future: pd.DataFrame | None = None) -> dict[str, Any]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.fitted_ is None:
            mean = np.zeros(horizon, dtype=float)
        else:
            # statsmodels returns in-sample/out-of-sample forecasts
            fcast = self.fitted_.forecast(steps=horizon)
            mean = np.asarray(fcast, dtype=float)
        std = np.full(horizon, float(self.resid_std_ or 1e-6), dtype=float)
        return {"mean": mean, "std": std, "quantiles": _quantiles_from_normal(mean, std)}
