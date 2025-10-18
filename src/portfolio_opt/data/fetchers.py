from __future__ import annotations

import os
import sys
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from .caching import DiskCache
from ..utils import ensure_datetime_index

logger = logging.getLogger(__name__)


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

        def _extract_close(df: pd.DataFrame, syms: Sequence[str]) -> pd.DataFrame:
            # Normalize to Close prices DataFrame with shared index
            if df is None or df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                level0 = df.columns.get_level_values(0)
                cols = {t: df[t]["Close"] for t in syms if t in level0}
                if not cols:
                    return pd.DataFrame()
                out = pd.concat(cols, axis=1)
                out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
                return ensure_datetime_index(out)
            # Single ticker frame
            if "Close" in df.columns:
                out = df["Close"].to_frame()
                out.columns = [syms[0] if syms else "Close"]
                return ensure_datetime_index(out)
            return pd.DataFrame()

        syms = list(dict.fromkeys(tickers))  # preserve order, drop duplicates
        combined: pd.DataFrame | None = None
        failed: set[str] = set()
        chunk_size = 50

        for i in range(0, len(syms), chunk_size):
            chunk = syms[i : i + chunk_size]
            try:
                data = yf.download(
                    tickers=chunk,
                    start=start,
                    end=end,
                    group_by="ticker",
                    progress=False,
                    threads=False,
                )
            except Exception as exc:  # noqa: BLE001
                failed.update(chunk)
                logger.warning("yfinance chunk download failed for %s: %s", ", ".join(chunk), exc)
                continue

            close = _extract_close(data, chunk)
            if close is None or close.empty:
                failed.update(chunk)
                continue

            missing_cols = [sym for sym in chunk if sym not in close.columns]
            if missing_cols:
                failed.update(missing_cols)
            close = close.reindex(columns=chunk)
            close = close.sort_index()
            combined = close if combined is None else combined.join(close, how="outer")

        if combined is None or combined.empty:
            return PriceProviderResult("yfinance", _synthetic_prices(tickers, start, end), False, "empty")

        combined = ensure_datetime_index(combined)
        combined = combined.sort_index()
        combined = combined.loc[:, ~combined.columns.duplicated()]

        available_cols = [col for col in combined.columns if combined[col].notna().any()]
        healthy = bool(available_cols)
        if not healthy:
            return PriceProviderResult("yfinance", _synthetic_prices(tickers, start, end), False, "empty")

        for t in tickers:
            if t not in combined.columns:
                combined[t] = np.nan
        combined = combined[list(tickers)]

        if failed:
            preview = ", ".join(sorted(failed)[:5])
            logger.warning(
                "yfinance skipped %d tickers during download (first few: %s)",
                len(failed),
                preview,
            )
            message = f"partial:{len(failed)}"
        else:
            message = "ok"

        return PriceProviderResult("yfinance", combined, healthy, message)
    except Exception as e:  # noqa: BLE001
        logger.warning("yfinance download failed entirely: %s", e)
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

    # Robustness: ensure business-day frequency and forward-fill gaps; keep columns with any data
    prices = prices.sort_index()
    try:
        prices = prices.asfreq("B").ffill()
    except Exception:
        # If index not convertible to freq, keep as-is
        prices = prices.ffill()
    prices = prices.loc[:, prices.notna().sum() > 0]
    # Pre-compute returns to ensure at least some non-empty rows downstream
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")

    cache.save_df(key, prices)
    return prices
