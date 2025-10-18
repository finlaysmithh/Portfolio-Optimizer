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


def _try_import_pmdarima():
    try:
        import pmdarima as pm  # type: ignore
        return pm
    except Exception:
        return None


@dataclass
class ARIMAForecaster(Forecaster):
    """Auto-ARIMA forecaster (pmdarima if available) or statsmodels SARIMAX grid.

    For daily returns, seasonality is typically None. Supports optional exogenous
    features via SARIMAX when provided in ``fit``/``predict``.
    """

    order_: tuple[int, int, int] | None = None
    fitted_: Any | None = None
    resid_std_: float | None = None
    use_sarimax_: bool = False

    def fit(self, returns: pd.Series, exog: pd.DataFrame | None = None) -> "ARIMAForecaster":
        r = _ensure_series(returns)
        if len(r) < MIN_HISTORY:
            raise ValueError(f"Need at least {MIN_HISTORY} observations for ARIMA, got {len(r)}")
        pm = _try_import_pmdarima()
        if exog is not None:
            # Use SARIMAX for exogenous features
            self.use_sarimax_ = True
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            best_aic = np.inf
            best_fit: Any | None = None
            best_order: tuple[int, int, int] | None = None
            # Small grid for (p,d,q)
            for p in (0, 1, 2):
                for d in (0, 1):
                    for q in (0, 1, 2):
                        try:
                            model = SARIMAX(r, order=(p, d, q), exog=exog, enforce_stationarity=False,
                                            enforce_invertibility=False)
                            res = model.fit(disp=False)
                            aic = float(res.aic)
                            if aic < best_aic:
                                best_aic = aic
                                best_fit = res
                                best_order = (p, d, q)
                        except Exception:
                            continue
            if best_fit is None:
                # Fallback to white noise
                self.fitted_ = None
                self.order_ = (0, 0, 0)
                self.resid_std_ = float(np.nanstd(r, ddof=1) or 1e-6)
                return self
            resid = r - best_fit.fittedvalues
            self.fitted_ = best_fit
            self.order_ = best_order
            self.resid_std_ = float(np.nanstd(resid, ddof=1) or 1e-6)
            return self

        # No exogenous variables: use pmdarima if available; otherwise small SARIMAX grid
        if pm is not None:
            try:
                model = pm.auto_arima(
                    r.values, start_p=0, start_q=0, max_p=3, max_q=3, max_d=1,
                    seasonal=False, error_action="ignore", suppress_warnings=True, stepwise=True
                )
                self.fitted_ = model
                self.order_ = tuple(model.order)  # type: ignore[assignment]
                resid = r.values - model.predict_in_sample()
                self.resid_std_ = float(np.nanstd(resid, ddof=1) or 1e-6)
                return self
            except Exception:
                pass

        # statsmodels SARIMAX small grid fallback
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        best_aic = np.inf
        best_fit = None
        best_order: tuple[int, int, int] | None = None
        for p in (0, 1, 2, 3):
            for d in (0, 1):
                for q in (0, 1, 2, 3):
                    try:
                        model = SARIMAX(r, order=(p, d, q), enforce_stationarity=False,
                                        enforce_invertibility=False)
                        res = model.fit(disp=False)
                        aic = float(res.aic)
                        if aic < best_aic:
                            best_aic = aic
                            best_fit = res
                            best_order = (p, d, q)
                    except Exception:
                        continue
        if best_fit is None:
            self.fitted_ = None
            self.order_ = (0, 0, 0)
            self.resid_std_ = float(np.nanstd(r, ddof=1) or 1e-6)
            return self
        resid = r - best_fit.fittedvalues
        self.fitted_ = best_fit
        self.order_ = best_order
        self.resid_std_ = float(np.nanstd(resid, ddof=1) or 1e-6)
        return self

    def predict(self, horizon: int, exog_future: pd.DataFrame | None = None) -> dict[str, Any]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.fitted_ is None:
            mean = np.zeros(horizon, dtype=float)
            std = np.full(horizon, float(self.resid_std_ or 1e-6), dtype=float)
            return {"mean": mean, "std": std, "quantiles": _quantiles_from_normal(mean, std)}

        if self.use_sarimax_ and exog_future is not None:
            # statsmodels SARIMAX
            fcast = self.fitted_.get_forecast(steps=horizon, exog=exog_future)
            mean = np.asarray(fcast.predicted_mean, dtype=float)
            std = np.sqrt(np.clip(np.asarray(fcast.var_pred_mean, dtype=float), 1e-12, np.inf))
            return {"mean": mean, "std": std, "quantiles": _quantiles_from_normal(mean, std)}

        # pmdarima or SARIMAX without exog
        try:
            # pmdarima
            mean_arr = np.asarray(self.fitted_.predict(n_periods=horizon))  # type: ignore[attr-defined]
            mean = mean_arr.astype(float)
        except Exception:
            # statsmodels SARIMAX
            fcast = self.fitted_.get_forecast(steps=horizon)
            mean = np.asarray(fcast.predicted_mean, dtype=float)
        std = np.full(horizon, float(self.resid_std_ or 1e-6), dtype=float)
        return {"mean": mean, "std": std, "quantiles": _quantiles_from_normal(mean, std)}
