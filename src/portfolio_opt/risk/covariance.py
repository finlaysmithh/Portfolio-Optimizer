from __future__ import annotations

import numpy as np
import pandas as pd

from .ewma import ewma_cov


def ledoit_wolf_shrinkage(
    sample_cov: pd.DataFrame, shrink_to: str = "identity", delta: float | None = None
) -> pd.DataFrame:
    s = sample_cov.values
    n = s.shape[0]
    if shrink_to == "identity":
        target = np.eye(n) * np.trace(s) / n
    else:
        target = np.diag(np.diag(s))
    if delta is None:
        # Simple heuristic shrinkage intensity
        var_s = np.var(s)
        var_t = np.var(target)
        delta = var_s / (var_s + var_t + 1e-9)
        delta = float(np.clip(delta, 0.05, 0.9))
    shrunk = delta * target + (1 - delta) * s
    return pd.DataFrame(shrunk, index=sample_cov.index, columns=sample_cov.columns)


def robust_covariance(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    ew = ewma_cov(returns, lam=lam)
    if not np.all(np.isfinite(ew.values)):
        ew = returns.cov()
    return ledoit_wolf_shrinkage(ew)
