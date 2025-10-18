import numpy as np
import pandas as pd

from portfolio_opt.forecast.covariance import project_ewma_cov


def test_project_ewma_cov_dims():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    rng = np.random.default_rng(0)
    r1 = rng.normal(0, 0.01, size=len(idx))
    r2 = rng.normal(0, 0.02, size=len(idx))
    px = pd.DataFrame({"A": (1 + r1).cumprod(), "B": (1 + r2).cumprod()}, index=idx)
    rets = px.pct_change().dropna()
    cov = project_ewma_cov(rets, lam=0.94, horizon=21)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov.values))

