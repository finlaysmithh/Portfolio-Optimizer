from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .caching import DiskCache
from ..utils import ensure_datetime_index


@dataclass
class PriceProviderResult:
    name: str
    prices: pd.DataFrame
    healthy: bool
    message: str = ""


def _synthetic_prices(tickers: Sequence[str], start: str | datetime, end: str | datetime) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="B")
    rng = np.random.default_rng(42)
    prices = {}
    for t in tickers:
        # Create synthetic lognormal price path
        rets = rng.normal(loc=0.0004, scale=0.01, size=len(idx))
        series = 100 * np.exp(np.cumsum(rets))
        prices[t] = series
    df = pd.DataFrame(prices, index=idx)
    return df


def _try_yfinance(tickers: Sequence[str], start: str | datetime, end: str | datetime) -> PriceProviderResult:
    try:
        import yfinance as yf

        data = yf.download(
            tickers=list(tickers), start=start, end=end, group_by="ticker", progress=False
        )
        # Normalize to Close prices DataFrame with shared index
        if isinstance(data.columns, pd.MultiIndex):
            close = pd.concat(
                {t: data[t]["Close"] for t in tickers if t in data.columns.get_level_values(0)},
                axis=1,
            )
            close.columns = [c[0] for c in close.columns]
        else:
            close = data["Close"].to_frame()
        close = ensure_datetime_index(close)
        if close.empty:
            return PriceProviderResult("yfinance", _synthetic_prices(tickers, start, end), False, "empty")
        return PriceProviderResult("yfinance", close, True, "ok")
    except Exception as e:  # noqa: BLE001
        return PriceProviderResult("yfinance", _synthetic_prices(tickers, start, end), False, str(e))


def _try_alpha_vantage(tickers: Sequence[str], start: str | datetime, end: str | datetime) -> PriceProviderResult:
    # Minimal stub: use synthetic offline for CI; implement actual API later
    return PriceProviderResult("alpha_vantage", _synthetic_prices(tickers, start, end), False, "stub")


def _try_fmp(tickers: Sequence[str], start: str | datetime, end: str | datetime) -> PriceProviderResult:
    # Minimal stub: use synthetic offline for CI; implement actual API later
    return PriceProviderResult("fmp", _synthetic_prices(tickers, start, end), False, "stub")


def fetch_price_history(
    tickers: Sequence[str], start: str | datetime, end: str | datetime
) -> pd.DataFrame:
    """Fetch prices with provider failover and disk cache.

    Order: Alpha Vantage → FMP → yfinance → synthetic fallback
    Caching: data/ with TTL
    """
    cache = DiskCache(namespace="prices")
    key = cache.key_from_params(tickers=",".join(sorted(tickers)), start=start, end=end)
    cached = cache.load_df(key)
    if cached is not None:
        return cached

    # Try providers
    tried: list[PriceProviderResult] = []
    tried.append(_try_alpha_vantage(tickers, start, end))
    if not tried[-1].healthy:
        tried.append(_try_fmp(tickers, start, end))
    if not tried[-1].healthy:
        tried.append(_try_yfinance(tickers, start, end))

    # Use first healthy, or last as fallback
    chosen = next((r for r in tried if r.healthy), tried[-1])
    prices = chosen.prices

    # Clean up any missing columns
    for t in tickers:
        if t not in prices.columns:
            prices[t] = np.nan
    prices = prices[tickers].dropna(how="all")

    cache.save_df(key, prices)
    return prices
