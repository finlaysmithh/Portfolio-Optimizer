from __future__ import annotations

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

def avg_dollar_volume(ticker: str, days: int = 60, start=None, end=None) -> float:
    """
    Average dollar volume over the last `days`. Robust to yfinance API changes
    and avoids float-on-Series deprecation warnings.
    """
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,  # keep Close semantics stable
            threads=False,
        )
    except Exception as e:
        raise ValueError(f"download failed for {ticker}: {e}")

    if data is None or data.empty:
        raise ValueError(f"No price/volume data for {ticker}")
    if "Volume" not in data.columns:
        raise ValueError(f"No Volume column for {ticker}")

    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    dv = (data[price_col] * data["Volume"]).tail(days)
    avg = dv.mean()

    # avoid float() on a 1-element Series (future TypeError)
    if isinstance(avg, pd.Series):
        avg = avg.iloc[0]
    elif hasattr(avg, "item"):
        try:
            avg = avg.item()
        except Exception:
            pass

    return float(avg)

def liquidity_screen(tickers, min_adv: float = 1e7, days: int = 60, start=None, end=None):
    """
    Screen tickers by ADV, skipping any that fail to fetch (graceful fallback).
    """
    passed = {}
    for t in tickers:
        try:
            adv = avg_dollar_volume(t, days=days, start=start, end=end)
            if adv >= min_adv:
                passed[t] = adv
        except Exception as e:
            logger.warning("Skipping %s: %s", t, e)
            continue
    return passed
