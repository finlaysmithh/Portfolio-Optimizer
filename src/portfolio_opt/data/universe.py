from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import io
import logging

import pandas as pd
import requests
import certifi

logger = logging.getLogger(__name__)

# Common class-share / odd Yahoo mappings
YAHOO_FIXES = {
    # Accept either dotted or hyphenated input and normalize to Yahoo's hyphen form
    "BRK.B": "BRK-B",
    "BRK-B": "BRK-B",
    "BF.B": "BF-B",
    "BF-B": "BF-B",
}

def _to_yahoo(t: str) -> Optional[str]:
    """
    Normalize a raw ticker to a Yahoo-compatible symbol.
    - Uppercase and strip whitespace
    - Drop junk duplicates (e.g., 'T.1', 'V.1')
    - Convert '.' to '-' for class shares (e.g., 'BRK.B' -> 'BRK-B')
    - Apply manual fixes
    """
    t = (t or "").strip().upper()
    if not t:
        return None
    if t.endswith(".1"):  # junk/duplicate variants sometimes appear in lists
        return None
    t = t.replace(".", "-")
    return YAHOO_FIXES.get(t, t)

def _sanitize_universe(tickers: Iterable[str]) -> list[str]:
    """
    Apply Yahoo normalization and de-duplicate while preserving order.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        norm = _to_yahoo(raw)
        if not norm:
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out

def load_tickers_from_file(path: str | Path) -> list[str]:
    """
    Read tickers from a text file. Each line may contain a single ticker
    or a comma/semicolon-separated list. Order is preserved; duplicates removed.
    Output is sanitized for Yahoo Finance.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tickers file not found: {p}")

    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    raw: list[str] = []
    for ln in lines:
        # support comma- or semicolon-separated lists per line
        parts = [t.strip() for t in ln.replace(";", ",").split(",") if t.strip()]
        raw.extend(parts if parts else [ln])

    return _sanitize_universe(raw)

def _fetch_sp500_from_wikipedia() -> list[str]:
    """
    Fetch S&P 500 tickers from Wikipedia using requests+certifi (robust TLS),
    parse with pandas.read_html, and return a sanitized list.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "portfolio-optimizer/1.0 (+https://localhost)"}

    resp = requests.get(url, headers=headers, timeout=20, verify=certifi.where())
    resp.raise_for_status()

    # Parse the HTML content string (wrapped in StringIO to avoid FutureWarning)
    try:
        # Prefer the 'constituents' table; fallback to the first table found
        try:
            tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
        except Exception:
            tables = pd.read_html(io.StringIO(resp.text))
    except Exception as e:
        raise RuntimeError(f"Failed to parse S&P 500 table: {e}") from e

    if not tables:
        raise RuntimeError("No HTML tables found on Wikipedia page.")

    df = tables[0]
    sym_col = None
    for candidate in ("Symbol", "Ticker", "Ticker symbol"):
        if candidate in df.columns:
            sym_col = candidate
            break
    if sym_col is None:
        raise KeyError("Could not find a Symbol/Ticker column in the table.")

    syms = [str(s).strip().upper().replace(".", "-") for s in df[sym_col].tolist()]
    return _sanitize_universe(syms)

def _cache_path() -> Path:
    # Repo root / data / sp500_cache.csv
    return Path(__file__).resolve().parents[3] / "data" / "sp500_cache.csv"

def _read_cache() -> list[str]:
    p = _cache_path()
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        return _sanitize_universe(df[col].astype(str).tolist())
    except Exception as e:
        logger.warning("Failed to read local S&P 500 cache %s: %s", p, e)
        return []

def _write_cache(symbols: list[str]) -> None:
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Symbol": symbols}).to_csv(p, index=False)
    except Exception as e:
        logger.debug("Skipping cache write (%s)", e)

def sp500_tickers() -> list[str]:
    """
    Return the current S&P 500 constituents as Yahoo-compatible tickers.

    Order of attempts:
    1) Wikipedia via requests+certifi (fresh)
    2) Local cache file at data/sp500_cache.csv (if present)
    3) Empty list (caller can supply a custom universe)
    """
    # 1) Try live fetch
    try:
        syms = _fetch_sp500_from_wikipedia()
        if syms:
            _write_cache(syms)  # refresh local cache for next time
            return syms
    except Exception as e:
        logger.warning("Failed to fetch S&P 500 from Wikipedia: %s", e)

    # 2) Fallback to cache
    cached = _read_cache()
    if cached:
        return cached

    # 3) Give up gracefully
    return []
