import numpy as np
import pandas as pd

from portfolio_opt.integration.expected import ExpectedConfig, forecasted_mu


def test_forecasted_mu_with_missing_data_falls_back():
    idx = pd.date_range("2021-01-01", periods=160, freq="B")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        {
            "AAA": (1 + rng.normal(0, 0.01, size=len(idx))).cumprod(),
            "BBB": (1 + rng.normal(0, 0.015, size=len(idx))).cumprod(),
        },
        index=idx,
    )
    # Inject NaNs and constant segments to trigger fallback
    prices.loc[idx[20:30], "AAA"] = np.nan
    prices.loc[idx[50:80], "BBB"] = prices.loc[idx[50], "BBB"]

    table, mu_array = forecasted_mu(prices, list(prices.columns), ExpectedConfig(model="ets", horizon_days=3, min_history=50))

    assert table.shape[0] == 2
    assert mu_array.shape == (2,)
    assert np.isfinite(table.values).all()
    assert np.isfinite(mu_array).all()
