from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    r = returns.dropna(how="all")
    if r.empty:
        return pd.DataFrame()
    r = r - r.mean()
    cov = np.zeros((r.shape[1], r.shape[1]))
    weights_list: list[float] = []
    w = 1.0
    for _ in range(len(r)):
        weights_list.append(w)
        w *= lam
    weights = np.array(list(reversed(weights_list)))
    weights = weights / float(weights.sum() or 1.0)
    for i in range(r.shape[0]):
        x = r.iloc[i].to_numpy().reshape(-1, 1)
        cov += weights[i] * (x @ x.T)
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)
