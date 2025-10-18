from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ..utils import ensure_datetime_index, to_returns


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    r = prices.diff()
    up = r.clip(lower=0).rolling(window).mean()
    down = (-r.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1 + rs)


def _macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "macd_signal": sig, "macd_hist": hist})


def build_feature_matrix(
    returns: pd.Series, exog: pd.DataFrame | None = None, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Builds a simple feature matrix from a single asset's price/returns.

    Args:
        returns: pd.Series of returns.
        exog: Optional exogenous features aligned on the same index.
        config: Optional dict to tune windows.
    Returns:
        pd.DataFrame with engineered features.
    """
    r = ensure_datetime_index(returns.to_frame(name="ret")).iloc[:, 0]
    # Construct synthetic price from returns for indicators
    price = (1 + r.fillna(0)).cumprod()
    w_short = int((config or {}).get("w_short", 5))
    w_med = int((config or {}).get("w_med", 21))
    w_long = int((config or {}).get("w_long", 63))

    feats = pd.DataFrame(index=r.index)
    feats["ret_1"] = r.shift(0)
    feats["ret_5"] = r.rolling(w_short).sum()
    feats["ret_21"] = r.rolling(w_med).sum()
    feats["vol_21"] = r.rolling(w_med).std()
    feats["skew_63"] = r.rolling(w_long).skew()
    feats["kurt_63"] = r.rolling(w_long).kurt()
    feats["mom_3m"] = price / price.shift(63) - 1
    feats["mom_6m"] = price / price.shift(126) - 1
    feats["mom_12m"] = price / price.shift(252) - 1
    feats["rsi_14"] = _rsi(price, 14)
    feats = feats.join(_macd(price), how="left")

    if exog is not None and not exog.empty:
        # Use forward-fill and align
        X = exog.reindex(r.index).ffill().copy()
        # Avoid column collisions
        X.columns = [f"exog__{c}" for c in X.columns]
        feats = feats.join(X, how="left")

    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats
