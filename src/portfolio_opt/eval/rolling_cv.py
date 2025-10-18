from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..forecast.base import Forecaster
from .metrics import rmse, mae, hit_rate


@dataclass
class RollingCVResult:
    scores: dict[str, float]
    last_model: Any | None


def rolling_origin_cv(
    returns: pd.Series,
    forecaster_factory: Callable[[], Forecaster],
    horizon: int = 1,
    folds: int = 5,
    min_train: int = 100,
) -> RollingCVResult:
    r = returns.dropna()
    n = len(r)
    if n < min_train + folds:
        # Not enough data; return dummy scores
        return RollingCVResult(scores={"rmse": np.nan, "mae": np.nan, "hit": np.nan}, last_model=None)

    step = max(1, (n - min_train) // folds)
    preds: list[float] = []
    trues: list[float] = []
    model: Any | None = None
    for i in range(min_train, n - horizon + 1, step):
        train = r.iloc[:i]
        test_y = r.iloc[i : i + horizon]
        f = forecaster_factory().fit(train)
        pr = f.predict(horizon)
        yhat = float(np.asarray(pr["mean"])[-1])  # last step forecast
        # use the first test point for 1-step evaluation if horizon>1
        preds.append(yhat)
        trues.append(float(test_y.iloc[-1]))
        model = f
    y_true = np.asarray(trues)
    y_pred = np.asarray(preds)
    scores = {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "hit": hit_rate(y_true, y_pred)}
    return RollingCVResult(scores=scores, last_model=model)

