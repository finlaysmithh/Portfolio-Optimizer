import numpy as np
import pandas as pd

from portfolio_opt.forecast.ml_linear import LinearMLForecaster


def test_ml_linear_shapes():
    idx = pd.date_range("2020-01-01", periods=250, freq="B")
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, size=len(idx))
    s = pd.Series(r, index=idx)
    f = LinearMLForecaster(model_type="lasso").fit(s)
    pred = f.predict(7)
    assert pred["mean"].shape == (7,)
    assert pred["std"].shape == (7,)

