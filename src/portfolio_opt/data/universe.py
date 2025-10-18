from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import io
import logging

import certifi
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Persistent CSV storing the latest S&P 500 universe.
_UNIVERSE_CSV = Path(__file__).resolve().parents[3] / "data" / "universe" / "sp500.csv"

# Lightweight fallback when both local disk and Wikipedia fail (10 tickers).
_DEMO_SP500 = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "JNJ",
    "V",
]

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
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse S&P 500 table: {exc}") from exc

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


def _read_local_sp500() -> list[str]:
    """
    Attempt to read the S&P 500 universe from the persisted CSV.
    """
    if not _UNIVERSE_CSV.exists():
        raise FileNotFoundError(_UNIVERSE_CSV)

    try:
        df = pd.read_csv(_UNIVERSE_CSV)
        if df.empty:
            return []
        if "Symbol" in df.columns:
            raw = df["Symbol"].astype(str).tolist()
        else:
            raw = df.iloc[:, 0].astype(str).tolist()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read {_UNIVERSE_CSV}: {exc}") from exc

    return _sanitize_universe(raw)


def _write_local_sp500(symbols: list[str]) -> None:
    """
    Persist the fetched S&P 500 universe to disk for offline reuse.
    """
    try:
        _UNIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Symbol": symbols}).to_csv(_UNIVERSE_CSV, index=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to write S&P 500 CSV cache %s: %s", _UNIVERSE_CSV, exc)


def sp500_tickers() -> list[str]:
    """
    Return the current S&P 500 constituents as Yahoo-compatible tickers.

    Order of attempts:
    1) Local CSV at data/universe/sp500.csv
    2) Wikipedia fetch (cached to the CSV above)
    3) Demo fallback (10 large-cap names) with a warning
    """
    # 1) Local CSV first to avoid network dependency on Streamlit Cloud
    try:
        local_syms = _read_local_sp500()
        if local_syms:
            logger.info("Loaded %d S&P 500 tickers from local CSV %s", len(local_syms), _UNIVERSE_CSV)
            return local_syms
        logger.warning("Local S&P 500 CSV %s is empty. Attempting Wikipedia fetch.", _UNIVERSE_CSV)
    except FileNotFoundError:
        logger.info("Local S&P 500 CSV %s not found. Attempting Wikipedia fetch.", _UNIVERSE_CSV)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read S&P 500 CSV %s: %s. Attempting Wikipedia fetch.", _UNIVERSE_CSV, exc)

    # 2) Fetch from Wikipedia and persist for next time
    try:
        wiki_syms = _fetch_sp500_from_wikipedia()
        if wiki_syms:
            logger.info(
                "Fetched %d S&P 500 tickers from Wikipedia and cached to %s",
                len(wiki_syms),
                _UNIVERSE_CSV,
            )
            _write_local_sp500(wiki_syms)
            return wiki_syms
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch S&P 500 from Wikipedia: %s", exc)

    # 3) Demo fallback
    logger.warning(
        "Falling back to demo S&P 500 subset (%d tickers). "
        "Check connectivity or ensure %s exists.",
        len(_DEMO_SP500),
        _UNIVERSE_CSV,
    )
    return _DEMO_SP500.copy()
