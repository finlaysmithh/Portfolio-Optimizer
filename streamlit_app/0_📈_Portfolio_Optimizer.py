# streamlit_app/app.py
from __future__ import annotations

import os
from datetime import date, timedelta

# Ensure local package (src/portfolio_opt) is importable when running without install
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st
st.set_page_config(page_title="Portfolio Optimizer App 📈", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as plx
import yfinance as yf

# Light theming: silver background and subtle cards
st.markdown(
    """
    <style>
    .stApp { background-color: #eef0f3; }
    /* Add extra top padding so the hero bar clears the sticky header */
    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
    /* Make the global Streamlit header/toolbar grey to match app */
    [data-testid="stHeader"] { background: #eef0f3; border-bottom: 1px solid #d4d4d4; }
    [data-testid="stHeader"] div { background: transparent; }
    [data-testid="stToolbar"] { background: #eef0f3 !important; }
    header { background: #eef0f3; }
    .opt-card {
        background: linear-gradient(180deg, #f7f7f7 0%, #e5e5e5 100%);
        padding: 1rem 1.25rem; border-radius: 10px;
        border: 1px solid #d4d4d4; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .opt-header { font-weight: 600; margin-bottom: .5rem; }
    /* Hero top bar */
    .hero-bar {
        background: #eaf4ff; /* light blue */
        border: 1px solid #cfe3ff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin: 0.5rem 0 1rem 0; /* nudge down from header */
    }
    .hero-title { font-size: 1.6rem; font-weight: 700; color: #0a2540; }
    .hero-sub { color: #29465b; margin-top: .25rem; }
    /* Floating Guide link near toolbar */
    .guide-link {
        position: fixed; top: 8px; right: 110px; z-index: 10000;
        background: #f0f2f5; color: #0a2540; text-decoration: none;
        border: 1px solid #d4d4d4; border-radius: 8px;
        padding: 6px 10px; font-weight: 600; font-size: 0.9rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .guide-link:hover { background: #e5e7eb; }
    .coded-by {
        position: fixed;
        top: 8px;
        left: 16px;
        z-index: 10000;
        font-size: 0.75rem;
        color: #394b59;
        opacity: 0.85;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="coded-by">coded by Finlay Smith</div>
    """,
    unsafe_allow_html=True,
)

# ---- Try to import your package pieces (fall back gracefully if any are missing) ----
try:
    from portfolio_opt.data.universe import sp500_tickers
except Exception:
    sp500_tickers = None

try:
    from portfolio_opt.data.fetchers import fetch_price_history  # optional
except Exception:
    fetch_price_history = None

try:
    from portfolio_opt.reporting.tearsheet import save_tearsheet  # optional
except Exception:
    save_tearsheet = None

# Optional PyPortfolioOpt for max-Sharpe weights
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
except Exception:
    EfficientFrontier = None
    risk_models = None
    expected_returns = None

# ---------- Sidebar controls ----------
with st.sidebar:
    guide_box = st.expander("ℹ️ Guide to Portfolio Optimizer", expanded=False)
    with guide_box:
        st.markdown(
            "\n".join(
                [
                    "- Universe: choose demo list, S&P 500 (via Wikipedia), or your custom tickers.",
                    "- Metrics: compute return, volatility, Sharpe, beta, VaR, drawdown, and more per ticker.",
                    "- Filters: screen on liquidity, Sharpe, momentum, beta range, and optional momentum exclusions.",
                    "- Ranking: order by composite score, Sharpe, or momentum before selecting target names.",
                    "- Optimization: run equal-weight or max-Sharpe (PyPortfolioOpt) while respecting constraints.",
                    "- Backtest: set horizon, rebalancing, and trading costs to evaluate performance.",
                    "- Outputs: review charts/tables and download CSVs or tear sheets for follow-up.",
                ]
            )
        )

    st.markdown("---")
    st.header("Universe & Filters")
    universe_choice = st.selectbox("Universe", ["Demo 30", "S&P 500", "Custom list"], index=1)
    custom_raw = st.text_area("Custom tickers (comma/space separated)", "")

    st.markdown("---")
    st.subheader("Selection")
    target_names = st.slider("Target # names", min_value=3, max_value=25, value=6, step=1)
    min_adv   = st.number_input("Min ADV ($)", min_value=0.0, value=0.0, step=1e6, format="%.0f")
    rf_pct    = st.number_input("Risk-free (annual %)", min_value=0.0, max_value=10.0, value=4.0, step=0.25)
    min_sharpe = st.slider("Min Sharpe", 0.0, 3.0, 0.8, 0.1)
    beta_range = st.slider("Beta range", 0.0, 2.5, (0.0, 1.5), 0.05)
    st.markdown("Momentum")
    mom_months = st.slider("Lookback (months)", 3, 18, 12, 1)
    mom_ex_recent = st.checkbox("Exclude most recent month", value=True)
    min_momentum = st.slider("Min momentum (L/S)", -1.0, 2.0, 0.0, 0.05,
                             help="Price change over the lookback horizon; e.g., 0.10 = +10%.")
    rank_method = st.selectbox("Rank by", ["Composite (Sharpe + Momentum)", "Sharpe", "Momentum"])
    use_monthly_sharpe = st.checkbox("Use monthly returns for Sharpe", value=False,
                                     help="Compute Sharpe from monthly returns (12/√12) instead of daily.")
    weight_method = st.selectbox("Weights", ["Equal-Weight", "Max Sharpe (PyPortfolioOpt)"])

    st.markdown("---")
    st.subheader("Backtest")
    start = st.date_input("Start", value=date.today() - timedelta(days=365*3))
    end   = st.date_input("End",   value=date.today())
    cost_bps = st.number_input("Trading cost (bps)", min_value=0.0, value=5.0, step=1.0)
    apply_costs = st.checkbox("Apply trading cost in curve", value=True)

    run_btn = st.button("Run Optimization")
    demo_btn = st.button("Demo Run (1y, no ADV)")
    st.markdown("---")

st.markdown(
    """
    <div class="hero-bar">
      <div class="hero-title">📈 Portfolio Optimizer</div>
      <div class="hero-sub">Compute risk metrics, screen by momentum and Sharpe, and optimize weights — with real S&P 500 data.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
def _parse_custom(raw: str) -> list[str]:
    import re
    toks = [t.strip().upper().replace(".", "-") for t in re.split(r"[,\s]+", raw) if t.strip()]
    # de-duplicate, keep order
    return list(dict.fromkeys(toks))

@st.cache_data(show_spinner=False)
def _get_sp500() -> list[str]:
    if sp500_tickers is None:
        # Minimal fallback demo universe
        return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","JPM","JNJ","V",
                "PEP","XOM","PG","AVGO","COST","HD","BAC","LLY","CVX","KO",
                "MRK","WMT","PFE","ORCL","ADBE","CSCO","DIS","NFLX","CRM","ABBV"]
    syms = sp500_tickers()
    return syms or ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","JPM","JNJ","V"]

def _resolve_universe() -> list[str]:
    if universe_choice == "S&P 500":
        syms = _get_sp500()
    elif universe_choice == "Custom list" and custom_raw.strip():
        syms = _parse_custom(custom_raw)
    else:
        syms = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","JPM","JNJ","V",
                "PEP","XOM","PG","AVGO","COST","HD","BAC","LLY","CVX",
                "KO","MRK","WMT","PFE","ORCL","ADBE","CSCO","DIS","NFLX","CRM"]
    return syms

def _yf_prices(tickers: list[str], start, end) -> pd.DataFrame:
    """
    Wide frame of adjusted prices (Adj Close if available, else Close).
    Downloads in chunks to be more reliable for large universes.
    """
    if not tickers:
        return pd.DataFrame()

    tickers = list(dict.fromkeys(tickers))  # de-duplicate, keep order
    chunks: list[list[str]] = [tickers[i : i + 100] for i in range(0, len(tickers), 100)]
    frames: list[pd.DataFrame] = []
    for ch in chunks:
        try:
            df = yf.download(
                ch,
                start=str(start), end=str(end),
                auto_adjust=False, progress=False, threads=False,
            )
        except Exception:
            df = None
        if df is None or df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = list(df.columns.levels[0])
            key = "Adj Close" if "Adj Close" in lvl0 else ("Close" if "Close" in lvl0 else None)
            if key is None:
                continue
            px = df[key]
        else:
            col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
            if col is None:
                continue
            sym = ch[0] if isinstance(ch, list) and ch else "SYMBOL"
            px = df[[col]].rename(columns={col: sym})
        frames.append(px)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    # Drop duplicate columns if any chunk overlap happened
    out = out.loc[:, ~out.columns.duplicated()]
    return out.sort_index().dropna(how="all")

def _yf_single_price(ticker: str, start, end) -> pd.Series | None:
    try:
        df = yf.download(
            [ticker], start=str(start), end=str(end), auto_adjust=False, progress=False, threads=False
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.levels[0])
        key = "Adj Close" if "Adj Close" in lvl0 else ("Close" if "Close" in lvl0 else None)
        if key is None:
            return None
        s = df[key].iloc[:, 0]
        s.name = ticker
        return s
    else:
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            return None
        s = df[col].rename(ticker)
        return s

def _improve_coverage(prices: pd.DataFrame, tickers: list[str], start, end) -> pd.DataFrame:
    if prices is None or prices.empty:
        prices = pd.DataFrame(index=pd.date_range(start=str(start), end=str(end), freq="B"))
    df = prices.copy()
    have = set(df.columns)
    need = [t for t in tickers if t not in have]
    # Also retry any columns with zero observations
    zero = [t for t in df.columns if df[t].notna().sum() == 0]
    to_retry = list(dict.fromkeys(need + zero))
    fixed = []
    for t in to_retry:
        s = _yf_single_price(t, start, end)
        if s is None or s.dropna().empty:
            continue
        df[t] = s
        fixed.append(t)
    if fixed:
        df = df.sort_index().loc[:, ~df.columns.duplicated()]
    return df

def _fetch_prices(tickers: list[str], start, end) -> pd.DataFrame:
    # Use yfinance directly to ensure real data
    return _yf_prices(tickers, start, end)

def _equity_from_weights(prices: pd.DataFrame, w: pd.Series | dict | float | int) -> pd.Series:
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    # Coerce weights to a Series aligned to columns
    if not isinstance(w, pd.Series):
        try:
            if np.isscalar(w):
                w = pd.Series([float(w)] * prices.shape[1], index=prices.columns)
            else:
                w = pd.Series(w)
        except Exception:
            w = pd.Series(1.0 / max(1, prices.shape[1]), index=prices.columns)
    w = w.reindex(prices.columns).fillna(0.0)
    if w.sum() == 0:
        w = pd.Series(1.0 / max(1, prices.shape[1]), index=prices.columns)
    else:
        w = w / w.sum()
    port = (rets @ w)  # daily return series
    eq = (1.0 + port).cumprod()
    eq.name = "equity"
    return eq

def _equity_with_cost(prices: pd.DataFrame, w_target: pd.Series | dict | float | int, cost_bps: float, freq: str = "M") -> pd.Series:
    """Monthly rebalanced equity curve with simple trading costs.

    - Rebalance at period end to target weights `w_target`.
    - Cost model: one-time deduction at each rebalance equal to 0.5 * sum(|w_target - w_drift|) * cost_bps.
      The 0.5 factor approximates that buys == sells.
    - Returns a daily equity series built from within-month static weights.
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    # Coerce weights to a Series aligned to columns
    if not isinstance(w_target, pd.Series):
        try:
            if np.isscalar(w_target):
                w_target = pd.Series([float(w_target)] * prices.shape[1], index=prices.columns)
            else:
                w_target = pd.Series(w_target)
        except Exception:
            w_target = pd.Series(1.0 / max(1, prices.shape[1]), index=prices.columns)
    w_tgt = w_target.reindex(prices.columns).fillna(0.0)
    if w_tgt.sum() <= 0:
        return pd.Series(dtype=float)
    w_tgt = w_tgt / w_tgt.sum()

    rets = prices.pct_change(fill_method=None).dropna(how="all")
    if rets.empty:
        return pd.Series(dtype=float)

    eq = 1.0
    out_parts: list[pd.Series] = []
    for _, r_month in rets.groupby(pd.Grouper(freq=freq)):
        if r_month is None or r_month.empty:
            continue
        # Within the month, use fixed start-of-month target weights
        port_daily = (r_month * w_tgt).sum(axis=1).fillna(0.0)
        path = (1.0 + port_daily).cumprod()
        path = eq * path
        eq = float(path.iloc[-1])

        # Compute drifted weights at month end, then turnover to rebalance to target
        g = (1.0 + r_month).prod().reindex(prices.columns).fillna(1.0)
        w_drift = w_tgt * g
        if w_drift.sum() > 0:
            w_drift = w_drift / w_drift.sum()
        # total traded fraction when rebalancing to target (approx one-sided)
        turnover = 0.5 * float(np.abs(w_tgt - w_drift).sum())
        if cost_bps and turnover > 0:
            eq = eq * (1.0 - (turnover * (cost_bps / 1e4)))
            path.iloc[-1] = eq

        out_parts.append(path)

    if not out_parts:
        return pd.Series(dtype=float)
    out = pd.concat(out_parts)
    # Ensure strictly increasing index with no duplicates
    out = out[~out.index.duplicated(keep="last")]
    out.name = "equity"
    return out

@st.cache_data(show_spinner=False)
def _market_series(start, end) -> pd.Series:
    m = _yf_prices(["^GSPC"], start, end)
    if m is None or m.empty:
        # Fallback to SPY proxy
        m = _yf_prices(["SPY"], start, end)
    if m is None or m.empty:
        return pd.Series(dtype=float)
    s = m.iloc[:, 0].dropna()
    s.name = "market"
    return s

@st.cache_data(show_spinner=False)
def _compute_metrics(prices: pd.DataFrame, rf_ann: float, market_px: pd.Series,
                     mom_months: int = 12, mom_ex_recent: bool = True,
                     monthly_sharpe: bool = False) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    # Ensure clean, unique, numeric columns
    prices = prices.sort_index().dropna(how="all")
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    rets = prices.pct_change(fill_method=None)
    tickers = list(prices.columns)
    if rets.empty or not tickers:
        return pd.DataFrame(index=list(prices.columns))

    # Annualized return and volatility
    if monthly_sharpe:
        px_m = prices.resample("M").last()
        rets_m = px_m.pct_change()
        mu_ann = rets_m.mean() * 12
        vol_ann = rets_m.std() * np.sqrt(12)
    else:
        log1p = np.log1p(rets)
        mu_ann = np.exp(log1p.mean() * 252) - 1.0
        vol_ann = rets.std() * np.sqrt(252)
    sharpe = (mu_ann - rf_ann) / vol_ann.replace(0, np.nan)

    # Market metrics (pairwise with market to avoid dropping all rows)
    beta = pd.Series(index=tickers, dtype=float)
    r2 = pd.Series(index=tickers, dtype=float)
    if market_px is not None and not market_px.empty:
        mret_all = market_px.pct_change(fill_method=None).dropna()
        for t in tickers:
            ri = rets[t].dropna()
            pair = pd.concat([ri.rename("ri"), mret_all.rename("m")], axis=1, join="inner").dropna()
            if pair.shape[0] < 20:
                beta[t] = np.nan
                r2[t] = np.nan
                continue
            m_c = pair["m"] - pair["m"].mean()
            ri_c = pair["ri"] - pair["ri"].mean()
            var_m = float((m_c ** 2).mean())
            cov_im = float((ri_c * m_c).mean())
            beta[t] = cov_im / var_m if var_m > 0 else np.nan
            corr = pair["ri"].corr(pair["m"]) if var_m > 0 else np.nan
            r2[t] = float(corr ** 2) if pd.notna(corr) else np.nan

    # Momentum: L-month price change, optionally excluding most recent month
    try:
        px_m = prices.resample("M").last()
        if mom_ex_recent and px_m.shape[0] >= (mom_months + 1):
            s = px_m.iloc[-(mom_months + 1):-1]
        else:
            s = px_m.iloc[-mom_months:]
        momentum = (s.iloc[-1] / s.iloc[0]) - 1.0
    except Exception:
        momentum = pd.Series(index=prices.columns, dtype=float)

    # Parametric 1y VaR at 95%
    mu_d = rets.mean()
    sd_d = rets.std()
    mu_1y = mu_d * 252
    sd_1y = sd_d * np.sqrt(252)
    z = 1.645
    var_95_1y = mu_1y - z * sd_1y

    # Max drawdown
    mdd = pd.Series(index=tickers, dtype=float)
    for t in tickers:
        st_col = prices[t]
        # If duplicate column labels slipped through, `prices[t]` can be a DataFrame
        if isinstance(st_col, pd.DataFrame):
            s = st_col.iloc[:, 0].dropna()
        else:
            s = st_col.dropna()
        if s.empty:
            mdd[t] = np.nan
            continue
        peak = s.cummax()
        dd = (s - peak) / peak
        try:
            mdd_val = float(dd.min())
        except Exception:
            # Last-resort scalarization if dd is not a simple Series
            mdd_val = float(pd.Series(dd).min())
        mdd[t] = mdd_val

    # Data coverage diagnostics
    obs = rets.notna().sum()
    coverage = obs / max(1, rets.shape[0])

    out = pd.DataFrame({
        "ann_return": mu_ann,
        "ann_vol": vol_ann,
        "sharpe": sharpe,
        "momentum": momentum,
        "beta": beta,
        "r2": r2,
        "var_95_1y": var_95_1y,
        "max_drawdown": mdd,
        "obs": obs,
        "coverage": coverage,
    })
    return out

@st.cache_data(show_spinner=False)
def _sector_map_from_wiki() -> dict[str, str]:
    """Fetch sector mapping for S&P 500 from Wikipedia with robust HTTP.

    Returns {symbol: sector} where symbol uses Yahoo-style hyphen for class shares.
    """
    import io
    import requests, certifi

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "portfolio-optimizer/1.0 (+https://localhost)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20, verify=certifi.where())
        resp.raise_for_status()
        # Parse HTML via pandas from in-memory string for reliability
        try:
            tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
        except Exception:
            tables = pd.read_html(io.StringIO(resp.text))
        if not tables:
            return {}
        df = tables[0]
        sym_col = next((c for c in ["Symbol", "Ticker", "Ticker symbol"] if c in df.columns), None)
        sec_col = next((c for c in ["GICS Sector", "Sector"] if c in df.columns), None)
        if sym_col is None or sec_col is None:
            return {}
        m: dict[str, str] = {}
        for sym, sec in zip(df[sym_col].astype(str), df[sec_col].astype(str)):
            s = str(sym).strip().upper().replace(".", "-")
            m[s] = str(sec).strip()
        return m
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def _yf_sector(ticker: str) -> str | None:
    """Fallback: fetch sector for a single ticker via yfinance (best-effort)."""
    try:
        info = yf.Ticker(ticker).get_info()
        sec = info.get("sector") if isinstance(info, dict) else None
        if sec:
            return str(sec)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def _sector_map_for(tickers: list[str]) -> dict[str, str]:
    base = _sector_map_from_wiki()
    out: dict[str, str] = {}
    for t in tickers:
        key = str(t).strip().upper()
        sec = base.get(key)
        if not sec:
            sec = _yf_sector(key)
        out[key] = sec or "Unknown"
    return out

def _yf_price_and_volume(tickers: list[str], start, end) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch price and volume via yfinance in chunks, returning (price, volume).

    - Price prefers Adj Close, else Close.
    - Returns (None, None) if nothing fetched.
    """
    if not tickers:
        return None, None
    tickers = list(dict.fromkeys(tickers))
    chunks: list[list[str]] = [tickers[i : i + 100] for i in range(0, len(tickers), 100)]
    px_parts: list[pd.DataFrame] = []
    vol_parts: list[pd.DataFrame] = []
    for ch in chunks:
        try:
            df = yf.download(
                ch,
                start=str(start), end=str(end),
                auto_adjust=False, progress=False, threads=False,
            )
        except Exception:
            df = None
        if df is None or df.empty:
            continue

        # Price slice
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = list(df.columns.levels[0])
            key = "Adj Close" if "Adj Close" in lvl0 else ("Close" if "Close" in lvl0 else None)
            px = df[key] if key else None
            vol = df["Volume"] if "Volume" in lvl0 else None
        else:
            col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
            sym = ch[0] if ch else "SYMBOL"
            px = df[[col]].rename(columns={col: sym}) if col else None
            vol = df[["Volume"]].rename(columns={"Volume": sym}) if "Volume" in df.columns else None

        if px is not None:
            px_parts.append(px)
        if vol is not None:
            vol_parts.append(vol)

    px = pd.concat(px_parts, axis=1) if px_parts else None
    vol = pd.concat(vol_parts, axis=1) if vol_parts else None
    if px is not None:
        px = px.loc[:, ~px.columns.duplicated()].sort_index().dropna(how="all")
    if vol is not None:
        vol = vol.loc[:, ~vol.columns.duplicated()].sort_index().dropna(how="all")
    return px, vol


# ---------- Main flow ----------
# Demo run preset: if pressed, override parameters and trigger run
if 'demo_trigger' not in st.session_state:
    st.session_state['demo_trigger'] = False

if demo_btn:
    # Set convenient demo defaults
    universe_choice = "Demo 30"
    custom_raw = ""
    min_adv = 0.0
    # Relax filters to ensure a result in demo
    min_sharpe = 0.0
    beta_range = (0.0, 2.0)
    weight_method = "Equal-Weight"
    start = date.today() - timedelta(days=365)
    end = date.today()
    cost_bps = 5.0
    run_btn = True
    st.session_state['demo_trigger'] = True

tickers = _resolve_universe()

# Sector filter in sidebar (after resolving universe)
try:
    sec_map_all = _sector_map_for(list(tickers))
    all_sectors = sorted(set(sec_map_all.values())) if sec_map_all else []
except Exception:
    sec_map_all, all_sectors = {}, []

with st.sidebar:
    st.subheader("Sectors")
    sector_mode = st.radio(
        "Universe sectors",
        ["All sectors", "Specific sector"],
        index=0,
        help="Default includes all sectors. Choose a specific sector to restrict the universe.",
    )
    selected_sector = None
    if all_sectors and sector_mode == "Specific sector":
        selected_sector = st.selectbox("Sector", options=all_sectors, index=0)

orig_count = len(tickers)
if all_sectors and selected_sector:
    tickers = [t for t in tickers if sec_map_all.get(str(t).upper(), "Unknown") == selected_sector]

if not tickers:
    st.error("No tickers after sector filter. Pick at least one sector.")
    st.stop()

st.caption(f"Universe: **{len(tickers)}** names selected (from {orig_count} after sector filter).")

if run_btn:
    try:
        # 1) Prices
        with st.spinner("Fetching prices..."):
            prices = _fetch_prices(tickers, start, end)
            # Improve coverage by retrying missing/empty tickers one-by-one
            prices = _improve_coverage(prices, tickers, start, end)
        if prices.empty:
            st.error("No price data found for the selected period/universe.")
            st.stop()

        # 2) Liquidity screen (ADV) — simple rolling ADV using Close/Adj Close * Volume
        with st.spinner("Screening liquidity..."):
            px_src, vol = _yf_price_and_volume(tickers, start, end)
            passed = tickers
            if vol is not None and px_src is not None:
                # Align columns intersection in case of partial fetch
                common = [t for t in tickers if t in vol.columns and t in px_src.columns]
                if common:
                    dv = (px_src[common] * vol[common]).rolling(60).mean().iloc[-1].dropna()
                    passed = [t for t, adv in dv.items() if adv >= float(min_adv)]
            st.caption(f"Liquidity screen (ADV ≥ ${min_adv:,.0f}): **{len(passed)}** names")

        if len(passed) < max(3, target_names):
            st.warning("Not enough names after liquidity screen; relax ADV or factor thresholds.")
            # keep going with what we have
            passed = passed[:max(3, len(passed))]

        # 3) Metrics
        with st.spinner("Computing metrics..."):
            market_px = _market_series(start, end)
            metrics = _compute_metrics(
                prices[passed],
                rf_ann=float(rf_pct)/100.0,
                market_px=market_px,
                mom_months=int(mom_months),
                mom_ex_recent=bool(mom_ex_recent),
                monthly_sharpe=bool(use_monthly_sharpe),
            )
        if metrics is None or metrics.empty:
            st.error("No metrics computed. Try changing dates or universe.")
            st.stop()
        # Quick summary so you can sanity-check distribution
        try:
            total_n = int(metrics.shape[0])
            sharpe_n = int((metrics["sharpe"] >= float(min_sharpe)).sum())
            mom_n = int((metrics["momentum"] >= float(min_momentum)).sum())
            st.caption(f"Sharpe ≥ {float(min_sharpe):.2f}: {sharpe_n} / {total_n} • Momentum ≥ {float(min_momentum):.2f}: {mom_n} / {total_n}")
        except Exception:
            pass

        # Filters: Sharpe, momentum, and beta range
        lo, hi = beta_range
        filt = metrics.copy()
        filt = filt[(filt["sharpe"] >= float(min_sharpe))]
        filt = filt[(filt["momentum"] >= float(min_momentum))]
        filt = filt[(filt["beta"].fillna(0.0) >= float(lo)) & (filt["beta"].fillna(0.0) <= float(hi))]
        st.caption(f"Metrics screen: **{len(filt)}** names after filters")

        # Sort by rank method and select top N (fallback to whole universe if filters empty)
        base = filt if not filt.empty else metrics
        if base is metrics:
            st.warning("No names after filters; relaxing thresholds and using top Sharpe from entire universe.")
        if rank_method == "Sharpe":
            ranked = base.sort_values(["sharpe", "ann_return"], ascending=False)
        elif rank_method == "Momentum":
            ranked = base.sort_values(["momentum", "ann_return"], ascending=False)
        else:
            # Composite: average normalized Sharpe and Momentum
            tmp = base[["sharpe", "momentum"]].copy()
            for c in tmp.columns:
                rng = tmp[c].max() - tmp[c].min()
                tmp[c] = (tmp[c] - tmp[c].min()) / (rng + 1e-9)
            comp = (tmp["sharpe"] + tmp["momentum"]) / 2.0
            ranked = base.assign(composite=comp).sort_values(["composite", "ann_return"], ascending=False)
        sel = ranked.index.tolist()[: int(target_names)]
        if not sel:
            st.error("No names after filters. Lower thresholds or widen beta range.")
            st.stop()

        # 4) Weights
        weights = None
        if weight_method == "Max Sharpe (PyPortfolioOpt)" and EfficientFrontier is not None:
            with st.spinner("Optimizing weights (max Sharpe)..."):
                try:
                    # Expected returns and covariance from price history
                    mu = expected_returns.mean_historical_return(prices[sel])
                    S = risk_models.sample_cov(prices[sel])
                    ef = EfficientFrontier(mu, S)
                    ef.max_sharpe(risk_free_rate=float(rf_pct)/100.0)
                    w = ef.clean_weights()
                    w = pd.Series(w, name="weight")
                    w = w[w > 0]
                    w = w / w.sum()
                    weights = w.to_frame()
                except Exception as e:
                    st.warning(f"PyPortfolioOpt failed ({e}); falling back to equal-weight.")

        if weights is None:
            w = pd.Series(1.0 / len(sel), index=sel, name="weight")
            weights = w.to_frame()

        # Show metrics table and allow download
        st.subheader("Metrics (filtered)")
        st.dataframe(ranked.loc[sel].style.format({
            "ann_return": "{:.2%}",
            "ann_vol": "{:.2%}",
            "sharpe": "{:.2f}",
            "beta": "{:.2f}",
            "r2": "{:.2f}",
            "var_95_1y": "{:.2%}",
            "max_drawdown": "{:.2%}",
        }))
        csv = metrics.sort_values("sharpe", ascending=False).to_csv()
        st.download_button("Download full metrics CSV", csv, file_name="metrics.csv", mime="text/csv")

        # Save weights.csv
        weights.to_csv("weights.csv", float_format="%.6f")
        st.success("✅ Optimization complete. Saved **weights.csv** in project root.")
        st.dataframe(weights.style.format({"weight": "{:.4%}"}))

        # 5) Simple equity curve using static weights (fallback if backtest engine unavailable)
        if apply_costs:
            eq = _equity_with_cost(prices[weights.index], weights["weight"], float(cost_bps), freq="M")
            title = f"Equity Curve (monthly rebalance, cost={cost_bps:.0f}bps)"
        else:
            eq = _equity_from_weights(prices[weights.index], weights["weight"]) 
            title = "Equity Curve (static weights)"
        eq.index.name = "Date"
        fig = plx.line(eq.reset_index(), x="Date", y="equity" if getattr(eq, 'name', None) == 'equity' else 0, title=title)
        fig.update_layout(yaxis_title="Equity")
        st.plotly_chart(fig, use_container_width=True)

        # 6) Optional tear sheet (only if available)
        if save_tearsheet is not None:
            try:
                out = "docs/images/tearsheet.pdf"
                os.makedirs(os.path.dirname(out), exist_ok=True)
                ret_series = eq.pct_change(fill_method=None).fillna(0.0)
                results = {"equity_curve": eq, "returns": ret_series}
                save_tearsheet(results, exposures=None, out_path=out)
                st.caption(f"Tear sheet saved: `{out}`")
            except Exception as e:
                st.info(f"Tear sheet export skipped: {e}")

        # 7) More charts
        st.subheader("More Charts")
        try:
            # Drawdown
            dd = (eq / eq.cummax()) - 1.0
            dd.index.name = "Date"
            fig_dd = plx.line(dd.reset_index(), x="Date", y="equity", title="Drawdown")
            fig_dd.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_dd, use_container_width=True)
        except Exception:
            pass

        try:
            # Rolling 60d Sharpe and Volatility
            ret = eq.pct_change(fill_method=None)
            roll_sharpe = (ret.rolling(60).mean() / (ret.rolling(60).std() + 1e-9)) * np.sqrt(252)
            roll_vol = ret.rolling(60).std() * np.sqrt(252)
            roll = pd.DataFrame({
                "Date": roll_sharpe.index,
                "Rolling Sharpe (60d)": roll_sharpe.values,
                "Rolling Vol (60d, ann)": roll_vol.values,
            }).dropna()
            fig_roll = plx.line(roll, x="Date", y=["Rolling Sharpe (60d)", "Rolling Vol (60d, ann)"], title="Rolling Metrics (60d)")
            st.plotly_chart(fig_roll, use_container_width=True)
        except Exception:
            pass

        try:
            # Weights bar chart
            wdf = weights.copy()
            wdf = wdf.reset_index().rename(columns={"index": "ticker"})
            fig_w = plx.bar(wdf, x="ticker", y="weight", title="Weights", labels={"weight": "Weight"})
            fig_w.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_w, use_container_width=True)
        except Exception:
            pass

        try:
            # Momentum vs Sharpe scatter for filtered universe
            rp = ranked.copy()
            rp = rp.assign(ticker=rp.index)
            fig_sc = plx.scatter(rp, x="momentum", y="sharpe", color="beta", hover_name="ticker", title="Momentum vs Sharpe (filtered)")
            st.plotly_chart(fig_sc, use_container_width=True)
        except Exception:
            pass

        try:
            # Benchmark overlay (^GSPC) vs Portfolio
            bench = _market_series(start, end)
            if bench is not None and not bench.empty:
                b = bench.loc[eq.index].dropna()
                if not b.empty:
                    b_eq = (b / b.iloc[0]).rename("S&P 500")
                    p = eq.rename("Portfolio")
                    both = pd.concat([p, b_eq], axis=1).dropna()
                    both.index.name = "Date"
                    fig_b = plx.line(both.reset_index(), x="Date", y=["Portfolio", "S&P 500"], title="Portfolio vs S&P 500")
                    st.plotly_chart(fig_b, use_container_width=True)
        except Exception:
            pass

        try:
            # Returns histogram (daily)
            r = eq.pct_change(fill_method=None).dropna()
            fig_hist = plx.histogram(r, x=r.name if isinstance(r, pd.Series) else 0, nbins=50, title="Daily Returns Distribution")
            fig_hist.update_layout(xaxis_tickformat=".1%", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception:
            pass

        try:
            # Correlation heatmap of selected tickers
            rets_sel = prices[weights.index].pct_change(fill_method=None).dropna(how="all")
            corr = rets_sel.corr().fillna(0.0)
            fig_heat = plx.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="RdBu_r", origin="lower", title="Correlation Heatmap (Selected)")
            fig_heat.update_coloraxes(cmin=-1, cmax=1)
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception:
            pass

        try:
            # Sector weights (from Wikipedia mapping with yfinance fallback)
            w = weights["weight"].copy()
            sec_map = _sector_map_for(list(w.index))
            sectors = [sec_map.get(str(t).upper(), "Unknown") for t in w.index]
            sw = pd.DataFrame({"sector": sectors, "weight": w.values}).groupby("sector").sum().sort_values("weight", ascending=False)
            fig_sec = plx.bar(sw.reset_index(), x="sector", y="weight", title="Sector Weights")
            fig_sec.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_sec, use_container_width=True)
        except Exception:
            pass

        try:
            # Rolling beta vs S&P 500 (60d)
            bench = _market_series(start, end)
            if bench is not None and not bench.empty:
                r_p = eq.pct_change(fill_method=None)
                r_m = bench.pct_change(fill_method=None)
                dfb = pd.concat([r_p.rename("p"), r_m.rename("m")], axis=1).dropna()
                cov_pm = dfb["p"].rolling(60).cov(dfb["m"]) 
                var_m = dfb["m"].rolling(60).var()
                beta_roll = (cov_pm / (var_m + 1e-12)).rename("Rolling Beta (60d)")
                beta_roll.index.name = "Date"
                fig_beta = plx.line(beta_roll.reset_index(), x="Date", y="Rolling Beta (60d)", title="Rolling Beta vs S&P 500 (60d)")
                st.plotly_chart(fig_beta, use_container_width=True)
        except Exception:
            pass

    except Exception as e:
        st.error(f"App error: {e}")
        st.exception(e)
else:
    st.info("Pick your universe & constraints in the sidebar, then hit **Run Optimization**.")
