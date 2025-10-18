import numpy as np
import pandas as pd

from portfolio_opt.forecast.arima import ARIMAForecaster


def test_arima_shapes():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    # AR(1) process for returns
    rng = np.random.default_rng(0)
    e = rng.normal(0, 0.01, size=len(idx))
    r = np.zeros(len(idx))
    for i in range(1, len(idx)):
        r[i] = 0.3 * r[i - 1] + e[i]
    s = pd.Series(r, index=idx)
    f = ARIMAForecaster().fit(s)
    pred = f.predict(10)
    assert pred["mean"].shape == (10,)
    assert pred["std"].shape == (10,)

