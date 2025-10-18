from __future__ import annotations

import numpy as np
import pandas as pd


def assert_aligned(mu: pd.Series, sigma: pd.DataFrame) -> None:
    if not all(mu.index == sigma.index):
        raise ValueError("mu and sigma must share the same index/tickers")
    if not all(sigma.columns == sigma.index):
        raise ValueError("sigma must be square with matching index/columns")


def safe_finite(df: pd.DataFrame | pd.Series, fill: float = 0.0):
    return df.replace([np.inf, -np.inf], np.nan).fillna(fill)

