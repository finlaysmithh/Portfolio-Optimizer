import pandas as pd

from portfolio_opt.data.fetchers import fetch_price_history
from portfolio_opt.factors.compute import DEFAULT_UNIVERSE, compute_factors
from portfolio_opt.optimize.constraints import Constraints
from portfolio_opt.optimize.optimizer import optimize


def test_optimize_basic_constraints():
    tickers = DEFAULT_UNIVERSE
    px = fetch_price_history(
        tickers, start="2020-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    f = compute_factors(px)
    cons = Constraints(target_names=6, min_value_exposure=0.3, min_momentum_exposure=0.2)
    res = optimize(f, px, cons)

    # 5-8 names
    assert 5 <= len(res.tickers) <= 8
    # Weights sum to 1
    assert abs(res.weights.sum() - 1.0) < 1e-6
    # Factor exposures satisfied
    assert res.exposures["value"] >= cons.min_value_exposure - 1e-3
    assert res.exposures["momentum"] >= cons.min_momentum_exposure - 1e-3
