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

from ..features.pipeline import build_feature_matrix
from .base import Forecaster, _ensure_series, _quantiles_from_normal


MIN_HISTORY = 50


@dataclass
class LinearMLForecaster(Forecaster):
    """Simple linear ML baseline using Lasso/Ridge with engineered features.

    Trains a 1-step ahead model and rolls forward to produce a multi-step
    forecast by iterating. Residual std is used as homoscedastic uncertainty.
    """

    model_type: str = "lasso"  # or "ridge"
    alpha_: float | None = None
    coef_: np.ndarray | None = None
    intercept_: float | None = None
    resid_std_: float | None = None
    feature_config: dict[str, Any] | None = None
    last_features_: pd.DataFrame | None = None

    def fit(self, returns: pd.Series, exog: pd.DataFrame | None = None) -> "LinearMLForecaster":
        from sklearn.linear_model import LassoCV, RidgeCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        r = _ensure_series(returns)
        if len(r) < MIN_HISTORY:
            raise ValueError(f"Need at least {MIN_HISTORY} observations for linear model, got {len(r)}")
        X = build_feature_matrix(r, exog=exog, config=self.feature_config)
        # Align target as next-period return
        y = r.shift(-1).reindex(X.index)
        mask = y.notna() & np.isfinite(X).all(axis=1)
        X_train = X[mask]
        y_train = y[mask]
        if len(X_train) < 20:
            # Not enough data, fall back to zero-mean
            self.alpha_ = 0.0
            self.coef_ = np.zeros(X.shape[1], dtype=float)
            self.intercept_ = 0.0
            self.resid_std_ = float(np.nanstd(r, ddof=1) or 1e-6)
            self.last_features_ = X.tail(1)
            return self

        if self.model_type == "ridge":
            base = RidgeCV(alphas=(0.1, 1.0, 10.0))
        else:
            base = LassoCV(alphas=None, cv=5, n_alphas=50, random_state=123)

        pipe = Pipeline([("sc", StandardScaler()), ("lin", base)])
        pipe.fit(X_train.values, y_train.values)
        lin = pipe.named_steps["lin"]
        # Extract coefficients in standardized space via pipeline
        sc = pipe.named_steps["sc"]
        coefs = getattr(lin, "coef_", np.zeros(X.shape[1]))
        intercept = float(getattr(lin, "intercept_", 0.0))
        # Map back to original space: y = (X - mu)/sigma * beta + intercept
        beta = coefs / (sc.scale_ + 1e-12)
        bias = intercept - np.sum(sc.mean_ * beta)

        y_pred = (X_train.values * beta).sum(axis=1) + bias
        resid = y_train.values - y_pred
        self.alpha_ = float(getattr(lin, "alpha_", 0.0))
        self.coef_ = beta.astype(float)
        self.intercept_ = float(bias)
        self.resid_std_ = float(np.nanstd(resid, ddof=1) or 1e-6)
        self.last_features_ = X.tail(1)
        return self

    def predict(self, horizon: int, exog_future: pd.DataFrame | None = None) -> dict[str, Any]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.coef_ is None or self.last_features_ is None:
            mean = np.zeros(horizon, dtype=float)
            std = np.full(horizon, float(self.resid_std_ or 1e-6), dtype=float)
            return {"mean": mean, "std": std, "quantiles": _quantiles_from_normal(mean, std)}

        X_next = self.last_features_.copy()
        means: list[float] = []
        for _ in range(horizon):
            y_hat = float(np.dot(X_next.values.ravel(), self.coef_) + (self.intercept_ or 0.0))
            means.append(y_hat)
            # Iteratively update momentum-based features using predicted return
            # Simplified: shift returns by appending y_hat to last return in cache
            if "ret_1" in X_next.columns:
                X_next.loc[:, "ret_1"] = y_hat
        mean = np.asarray(means, dtype=float)
        std = np.full(horizon, float(self.resid_std_ or 1e-6), dtype=float)
        return {"mean": mean, "std": std, "quantiles": _quantiles_from_normal(mean, std)}
