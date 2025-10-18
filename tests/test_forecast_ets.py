import numpy as np
import pandas as pd

from portfolio_opt.forecast.ets import ETSForecaster


def test_ets_shapes():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, size=len(idx))
    s = pd.Series(r, index=idx)
    f = ETSForecaster().fit(s)
    pred = f.predict(5)
    assert pred["mean"].shape == (5,)
    assert pred["std"].shape == (5,)

