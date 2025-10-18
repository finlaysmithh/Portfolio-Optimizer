import numpy as np
import pandas as pd

from portfolio_opt.integration.expected import ExpectedConfig, forecasted_mu, historical_mean_mu
from portfolio_opt.integration.risk import forecasted_sigma, historical_sigma


def test_expected_and_risk_integration():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    r1 = rng.normal(0, 0.01, size=len(idx))
    r2 = rng.normal(0, 0.02, size=len(idx))
    px = pd.DataFrame({"A": (1 + r1).cumprod(), "B": (1 + r2).cumprod()}, index=idx)

    mu_hist = historical_mean_mu(px, horizon=5)
    assert mu_hist.shape[0] == 2

    table, mu_array = forecasted_mu(px, ["A", "B"], ExpectedConfig(model="ets", horizon_days=5, min_history=50))
    assert list(table.index) == ["A", "B"]
    assert mu_array.shape == (2,)

    sigma_hist = historical_sigma(px)
    sigma_fc = forecasted_sigma(px, model="forecasted_sigma", horizon_days=5)
    assert sigma_hist.shape == sigma_fc.shape


def test_forecasted_mu_dataframe_handling():
    idx = pd.date_range("2021-01-01", periods=180, freq="B")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {
            "AAA": (1 + rng.normal(0, 0.01, size=len(idx))).cumprod(),
            "BBB": (1 + rng.normal(0, 0.012, size=len(idx))).cumprod(),
            "CCC": (1 + rng.normal(0, 0.008, size=len(idx))).cumprod(),
        },
        index=idx,
    )

    table, mu_array = forecasted_mu(prices, list(prices.columns), ExpectedConfig(model="ets", horizon_days=3, min_history=50))
    assert table.shape[0] == 3
    assert mu_array.shape == (3,)


def test_forecasted_mu_single_column_dataframe():
    idx = pd.date_range("2021-01-01", periods=120, freq="B")
    rng = np.random.default_rng(2)
    prices = pd.DataFrame({"AAA": (1 + rng.normal(0, 0.01, size=len(idx))).cumprod()}, index=idx)

    table, mu_array = forecasted_mu(prices, ["AAA"], ExpectedConfig(model="ets", horizon_days=2, min_history=40))
    assert table.index.tolist() == ["AAA"]
    assert mu_array.shape == (1,)
