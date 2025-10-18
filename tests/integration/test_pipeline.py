from pathlib import Path

import pandas as pd

from portfolio_opt.backtest.engine import backtest
from portfolio_opt.data.fetchers import fetch_price_history
from portfolio_opt.factors.compute import DEFAULT_UNIVERSE, compute_factors
from portfolio_opt.optimize.constraints import Constraints
from portfolio_opt.optimize.optimizer import optimize
from portfolio_opt.reporting.tearsheet import save_tearsheet


def test_end_to_end_tearsheet(tmp_path):
    tickers = DEFAULT_UNIVERSE
    px = fetch_price_history(
        tickers, start="2020-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    f = compute_factors(px)
    cons = Constraints(target_names=6)
    res = optimize(f, px, cons)
    sub_px = px[res.tickers]

    def gen(_):
        return res.weights

    bt = backtest(sub_px, rebalance_freq="M", cost_bps=5.0, weight_generator=gen)
    out = tmp_path / "tearsheet.pdf"
    path = save_tearsheet(
        {"equity_curve": bt.equity_curve, "returns": bt.returns}, res.exposures, out
    )
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
