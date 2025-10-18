import numpy as np
import pandas as pd

from portfolio_opt.sim.forward import SimConfig, simulate_paths, portfolio_paths, summarize_paths


def test_forward_sim_shapes_and_summary():
    mu = pd.Series([0.001, 0.0], index=["A", "B"])  # daily mean
    sigma = pd.DataFrame([[0.0004, 0.0], [0.0, 0.0009]], index=mu.index, columns=mu.index)
    sims = simulate_paths(mu, sigma, horizon=10, config=SimConfig(sim_paths=500, seed=123))
    assert sims.shape == (500, 10, 2)
    w = pd.Series([0.5, 0.5], index=mu.index)
    port = portfolio_paths(sims, w)
    assert port.shape == (500, 10)
    summary = summarize_paths(port)
    assert "expected_terminal_return" in summary

