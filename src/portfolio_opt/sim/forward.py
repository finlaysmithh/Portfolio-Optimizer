from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@dataclass
class SimConfig:
    distribution: Literal["gaussian", "student_t"] = "gaussian"
    student_t_df: int = 7
    sim_paths: int = 10000
    seed: int | None = 42


def simulate_paths(
    mu: pd.Series,
    sigma: pd.DataFrame,
    horizon: int,
    config: SimConfig | None = None,
) -> np.ndarray:
    """Simulate multivariate returns over horizon.

    Returns array with shape (sim_paths, horizon, n_assets).
    """
    cfg = config or SimConfig()
    mu_array = np.asarray(mu.values if isinstance(mu, pd.Series) else mu, dtype=float)
    if not np.isfinite(mu_array).all():
        print("[warn] NaN in mu — replaced with zeros for simulation")
        mu_array = np.nan_to_num(mu_array)
    n = int(len(mu_array))
    # Validate covariance shape vs. mu BEFORE any arithmetic
    if sigma is None or getattr(sigma, "size", 0) == 0 or sigma.shape != (n, n):
        raise ValueError(
            f"Empty or mismatched covariance matrix (got {getattr(sigma,'shape',None)}, expected ({n},{n})). "
            "Check date range or data availability."
        )
    if cfg.seed is not None:
        rng = np.random.default_rng(cfg.seed)
    else:
        rng = np.random.default_rng()
    # Safe Cholesky with tiny jitter added after the shape guard
    Sigma = np.asarray(getattr(sigma, "values", sigma), dtype=np.float64)
    Sigma = Sigma + 1e-12 * np.eye(n, dtype=np.float64)
    chol = np.linalg.cholesky(Sigma)
    sims = []
    for _ in range(cfg.sim_paths):
        if cfg.distribution == "student_t":
            z = rng.standard_t(cfg.student_t_df, size=(horizon, n))
            z = z / np.sqrt(cfg.student_t_df / (cfg.student_t_df - 2))
        else:
            z = rng.standard_normal(size=(horizon, n))
        shocks = z @ chol.T
        path = mu_array.reshape(1, -1) + shocks  # daily returns per step
        sims.append(path)
    return np.asarray(sims, dtype=float)


def portfolio_paths(
    sims: np.ndarray, weights: pd.Series, rebalance_inside_horizon: bool = False
) -> np.ndarray:
    """Map asset paths to portfolio return paths.

    If rebalance_inside_horizon=False, assumes weights fixed over horizon.
    Returns array shape (sim_paths, horizon).
    """
    w = weights.reindex(weights.index).fillna(0).values.astype(float)
    if not rebalance_inside_horizon:
        # Matrix multiply over last axis
        port = (sims * w.reshape(1, 1, -1)).sum(axis=2)
        return port
    # Rebalance: normalize weights each step (simple proportional scheme)
    port = []
    for path in sims:
        w_t = w.copy()
        pr: list[float] = []
        for step in path:
            r = float((step * w_t).sum())
            pr.append(r)
            equity = (1 + r)
            w_t = w_t * (1 + step)
            w_t = w_t / (w_t.sum() + 1e-12)
            w_t = w_t / equity  # keep exposure normalized
        port.append(pr)
    return np.asarray(port, dtype=float)


def summarize_paths(port_paths: np.ndarray) -> dict[str, pd.Series | float]:
    """Compute summary statistics across simulated portfolio paths.

    Returns dict with expected return, VaR/CVaR at 95/99, and quantile bands.
    """
    # Cumulative return per path
    cum = (1 + pd.DataFrame(port_paths.T)).cumprod().iloc[-1] - 1.0
    exp_ret = float(np.nanmean(cum))
    for_q = pd.Series(cum).quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    # Daily portfolio VaR/ES approximations over horizon (on terminal distribution)
    var95 = float(pd.Series(cum).quantile(0.05))
    var99 = float(pd.Series(cum).quantile(0.01))
    cvar95 = float(pd.Series(cum[cum <= var95]).mean()) if np.any(cum <= var95) else var95
    cvar99 = float(pd.Series(cum[cum <= var99]).mean()) if np.any(cum <= var99) else var99
    return {
        "expected_terminal_return": exp_ret,
        "quantiles": for_q,
        "var95": var95,
        "var99": var99,
        "cvar95": cvar95,
        "cvar99": cvar99,
    }
