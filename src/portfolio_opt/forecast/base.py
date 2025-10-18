from __future__ import annotations

import os
import sys
from typing import Any, Protocol, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class Forecaster(Protocol):
    """Protocol for a univariate per-ticker return forecaster.

    Implementations should be stateless after prediction or support re-fitting
    via ``fit``. Predictions should be for arithmetic returns at the desired
    trading frequency (daily in this project).
    """

    def fit(self, returns: pd.Series, exog: pd.DataFrame | None = None) -> "Forecaster":
        """Fit the forecaster on a 1-D return series.

        Args:
            returns: pd.Series indexed by datetime, values as returns.
            exog: Optional exogenous features aligned to ``returns``.
        Returns:
            self for chaining.
        """

    def predict(
        self, horizon: int, exog_future: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """Predict multiple steps ahead.

        Returns a dict with keys:
            - "mean": np.ndarray[float] shape (horizon,)
            - "std": np.ndarray[float] shape (horizon,)
            - "quantiles": dict[float, np.ndarray] (e.g., {0.05, 0.5, 0.95})
        """


def _ensure_series(x: pd.Series) -> pd.Series:
    return ensure_series(x)


def _quantiles_from_normal(mean: np.ndarray, std: np.ndarray) -> dict[float, np.ndarray]:
    from scipy.stats import norm

    q05 = norm.ppf(0.05, loc=mean, scale=std)
    q50 = norm.ppf(0.50, loc=mean, scale=std)
    q95 = norm.ppf(0.95, loc=mean, scale=std)
    return {0.05: q05, 0.5: q50, 0.95: q95}


def ensure_series(x: pd.Series | pd.DataFrame | Any, name: str = "") -> pd.Series:
    """Ensure a 1D pandas Series with DatetimeIndex, drop NaNs, float dtype.

    - If given a single-column DataFrame, squeeze to Series.
    - Raises TypeError for multi-column DataFrames.
    - Coerces index to DatetimeIndex and casts values to float.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            raise TypeError("Expected pandas Series for returns, got DataFrame with multiple columns.")
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    x = x.dropna()
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index)
    x = x.astype(float)
    if x.empty:
        raise ValueError("Returns series is empty after cleaning")
    if name:
        x.name = name
    return x.sort_index()


def safe_fallback(series: pd.Series) -> Tuple[float, float]:
    """Return historical mean/std for fallback forecasts.

    Ensures finite floats; if data is degenerate or NaN, returns zeros.
    """
    if series is None or series.empty:
        return 0.0, 0.0
    mu = float(series.mean())
    sigma = float(series.std(ddof=1))
    if not np.isfinite(mu):
        mu = 0.0
    if not np.isfinite(sigma) or sigma < 0:
        sigma = 0.0
    return mu, sigma
