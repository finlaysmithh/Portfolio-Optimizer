from __future__ import annotations

import numpy as np
import pandas as pd


def turnover(prev_w: pd.Series, next_w: pd.Series) -> float:
    prev = prev_w.reindex(next_w.index).fillna(0.0)
    return float(np.abs(next_w - prev).sum())


def trading_cost(turnover_rate: float, cost_bps: float = 5.0) -> float:
    return float(turnover_rate * (cost_bps / 10000.0))
