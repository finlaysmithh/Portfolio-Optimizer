"""Utility functions for date handling and return conversion."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


def hash_key(*parts: Any, prefix: str = "") -> str:
    """Create a stable hash key from arbitrary parts (JSON serialized)."""

    if len(parts) == 1:
        payload = parts[0]
    else:
        payload = parts
    data = json.dumps(payload, default=str, sort_keys=True)
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:" + digest if prefix else digest


def ensure_datetime_index(df_or_series: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Ensure the input has a DatetimeIndex, coercing when necessary."""

    if not isinstance(df_or_series.index, pd.DatetimeIndex):
        df_or_series.index = pd.to_datetime(df_or_series.index, errors="coerce")
    return df_or_series


def to_returns(prices: pd.DataFrame | pd.Series, method: str = "pct_change") -> pd.DataFrame:
    """Convert price levels to simple (or log) returns, dropping all-NaN rows."""

    if prices is None or len(prices) == 0:
        return pd.DataFrame()
    prices_df = pd.DataFrame(prices)
    if method == "pct_change":
        returns = prices_df.pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        returns = np.log(prices_df / prices_df.shift(1))
    return returns.dropna(how="all")


from .formatting import fmt_bp, fmt_float, fmt_pct  # noqa: E402, F401
from .validation import assert_aligned, safe_finite  # noqa: E402, F401

__all__ = [
    "fmt_bp",
    "fmt_float",
    "fmt_pct",
    "assert_aligned",
    "safe_finite",
    "hash_key",
    "ensure_datetime_index",
    "to_returns",
]
