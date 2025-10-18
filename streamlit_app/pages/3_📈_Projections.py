from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None
assert (px is None) or hasattr(px, "line"), "plotly.express alias 'px' was shadowed; rename local variables."
import streamlit as st

from scipy.stats import norm

try:
    from scipy.stats import t as tdist
except Exception:  # pragma: no cover
    tdist = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from portfolio_opt.integration.expected import ExpectedConfig, forecasted_mu
from portfolio_opt.integration.risk import forecasted_sigma
from portfolio_opt.sim.forward import SimConfig, simulate_paths, portfolio_paths, summarize_paths
from portfolio_opt.data.fetchers import fetch_price_history  # type: ignore
from portfolio_opt.utils import to_returns
from portfolio_opt.data.universe import sp500_tickers
from portfolio_opt.factors.compute import DEFAULT_UNIVERSE
from portfolio_opt.utils.formatting import fmt_pct


st.set_page_config(page_title="Projections App 📈", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #eef0f3; }
    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
    [data-testid="stHeader"], header { background: #eef0f3; border-bottom: 1px solid #d4d4d4; }
    [data-testid="stHeader"] div { background: transparent; }
    [data-testid="stToolbar"] { background: #eef0f3 !important; }
    .opt-card {
        background: linear-gradient(180deg, #f7f7f7 0%, #e5e5e5 100%);
        padding: 1rem 1.25rem; border-radius: 10px;
        border: 1px solid #d4d4d4; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .opt-header { font-weight: 600; margin-bottom: .5rem; }
    .hero-bar {
        background: #eaf4ff;
        border: 1px solid #cfe3ff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin: 0.5rem 0 1.2rem 0;
    }
    .hero-title { font-size: 1.6rem; font-weight: 700; color: #0a2540; }
    .hero-sub { color: #29465b; margin-top: .25rem; }
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
    <div class="hero-bar">
      <div class="hero-title">Projections App 📈</div>
      <div class="hero-sub">Forecast per-ticker returns, simulate forward paths, and benchmark your portfolio in one view.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="coded-by">coded by Finlay Smith</div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _fetch_prices(tickers: list[str], start: str | None, end: str | None) -> pd.DataFrame:
    return fetch_price_history(tickers, start=start, end=end)


@st.cache_data(show_spinner=True)
def _run_forecasts(px: pd.DataFrame, cfg: ExpectedConfig) -> tuple[pd.DataFrame, np.ndarray]:
    return forecasted_mu(px, list(px.columns), cfg)


def _get_quantiles(distribution: str, df: int) -> tuple[float, float, float]:
    if distribution == "student_t" and tdist is not None:
        try:
            lower = float(tdist.ppf(0.05, df))
            upper = float(tdist.ppf(0.95, df))
            return lower, 0.0, upper
        except Exception:  # pragma: no cover - guard only
            pass
    lower = float(norm.ppf(0.05))
    upper = float(norm.ppf(0.95))
    return lower, 0.0, upper


def _apply_cumulative(table: pd.DataFrame, horizon: int, distribution: str, df: int) -> pd.DataFrame:
    if "mean(d1)" not in table.columns or "std(d1)" not in table.columns:
        return table
    mean1 = table["mean(d1)"].fillna(0.0)
    std1 = table["std(d1)"].fillna(0.0)
    lower_q, _, upper_q = _get_quantiles(distribution, df)
    updated = table.copy()
    for step in range(1, horizon + 1):
        mean_col = f"mean(d{step})"
        std_col = f"std(d{step})"
        p05_col = f"p05(d{step})"
        p50_col = f"p50(d{step})"
        p95_col = f"p95(d{step})"
        if mean_col in updated.columns:
            updated[mean_col] = mean1 * step
        if std_col in updated.columns:
            updated[std_col] = std1 * np.sqrt(step)
        if mean_col in updated.columns and std_col in updated.columns:
            mean_k = updated[mean_col]
            std_k = updated[std_col]
            if p05_col in updated.columns:
                updated[p05_col] = mean_k + lower_q * std_k
            if p50_col in updated.columns:
                updated[p50_col] = mean_k
            if p95_col in updated.columns:
                updated[p95_col] = mean_k + upper_q * std_k
    return updated


def _format_table(table: pd.DataFrame, steps_to_show: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    column_map: dict[str, str] = {}
    ordered_cols: list[str] = []
    for step in steps_to_show:
        for metric, label in {"mean": "Mean", "std": "Std", "p05": "P05", "p50": "P50", "p95": "P95"}.items():
            col = f"{metric}(d{step})"
            if col in table.columns:
                ordered_cols.append(col)
                column_map[col] = f"{label} d{step}"
    table_shown = table.reindex(columns=ordered_cols)
    table_formatted = table_shown.applymap(lambda v: fmt_pct(v, 2))
    table_csv_ready = table_formatted.copy()
    table_formatted = table_formatted.rename(columns=column_map)
    return table_formatted, table_csv_ready.rename(columns=column_map), column_map


def _format_var(summary: dict[str, float | pd.Series]) -> tuple[list[str], pd.DataFrame]:
    lines = [
        f"**Expected terminal return:** {fmt_pct(summary.get('expected_terminal_return', np.nan))}",
        f"**VaR 95:** {fmt_pct(summary.get('var95', np.nan))}",
        f"**CVaR 95:** {fmt_pct(summary.get('cvar95', np.nan))}",
        f"**VaR 99:** {fmt_pct(summary.get('var99', np.nan))}",
        f"**CVaR 99:** {fmt_pct(summary.get('cvar99', np.nan))}",
    ]
    quantiles = summary.get("quantiles")
    if isinstance(quantiles, pd.Series):
        q_df = pd.DataFrame(
            {
                "Quantile": [f"{int(float(idx) * 100)}th" for idx in quantiles.index],
                "Value": [fmt_pct(val) for val in quantiles.values],
            }
        )
    else:
        q_df = pd.DataFrame(columns=["Quantile", "Value"])
    return lines, q_df


def _format_alpha(alpha_value: float, ci68: tuple[float, float], ci95: tuple[float, float]) -> list[str]:
    return [
        f"**Expected alpha:** {fmt_pct(alpha_value)}",
        f"68% CI: {fmt_pct(ci68[0])} → {fmt_pct(ci68[1])}",
        f"95% CI: {fmt_pct(ci95[0])} → {fmt_pct(ci95[1])}",
    ]


# Sidebar guide & controls (grey channel)
with st.sidebar:
    guide = st.expander("ℹ️ Guide to Projections", expanded=False)
    with guide:
        st.markdown(
            "\n".join(
                [
                    "- Forecasts per-ticker returns (μ) and joint risk (Σ), then simulates forward paths.",
                    "- Paths: number of simulated trajectories used to build outcome bands.",
                    "- Distribution: Gaussian draws are fast; Student-t adds fat tails via degrees of freedom.",
                    "- Horizon: number of trading days ahead for forecasts and simulations.",
                    "- Covariance: EWMA (daily), EWMA projected (scaled horizon), DCC (dynamic correlations).",
                    "- Benchmark & alpha: compare portfolio forecast to selected index and derive expected excess return.",
                    "- Daily vs cumulative toggle: view single-day stats or k-day compounded summaries.",
                ]
            )
        )

    st.subheader("Controls")
    universe = st.selectbox("Universe", ["Custom list", "S&P 500"], index=0)
    model = st.selectbox("Forecast model", ["auto_arima", "ets", "sarimax", "ml_lasso", "ml_ridge"], index=0)
    horizon = st.selectbox("Horizon (days)", [5, 10, 21, 63], index=2)
    show_cumulative = st.checkbox("Show cumulative stats (k-day)", value=False)
    use_exog = st.checkbox("Use exogenous features", value=False, help="Requires optional feature matrix; demo uses price-derived only")
    paths = st.number_input("Paths", min_value=100, max_value=200000, value=10000, step=1000)
    dist = st.selectbox("Distribution", ["gaussian", "student_t"], index=0)
    df = st.number_input("t degrees of freedom", min_value=3, max_value=50, value=7, step=1)
    cov_choice = st.selectbox("Covariance model", ["EWMA (projected)", "DCC-GARCH (slow)"], index=0)
    bench = st.selectbox("Benchmark", ["^GSPC", "SPY"], index=0)
    start_str = st.text_input("Start (YYYY-MM-DD)", value="")
    end_str = st.text_input("End (YYYY-MM-DD)", value="")
    show_debug = st.checkbox("Show debug shapes", value=False)

    if universe == "S&P 500":
        try:
            tickers = sp500_tickers()
        except Exception:
            tickers = list(DEFAULT_UNIVERSE)
        if not tickers:
            tickers = list(DEFAULT_UNIVERSE)
        tickers_display = ", ".join(tickers[:5]) + (" …" if len(tickers) > 5 else "")
        st.caption(f"Universe loaded: {len(tickers)} tickers (preview: {tickers_display})")
    else:
        tickers_input = st.text_area("Tickers (comma-separated)", value="AAPL, MSFT, AMZN, GOOGL, META").strip()
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) > 200 and paths > 5000:
        st.warning("For faster runs, reduce Paths or use a smaller universe.")

    run = st.button("Run Projections", use_container_width=True)


if not run:
    st.info("Configure inputs in the sidebar, then click **Run Projections**.")
    st.stop()

errors: list[str] = []
if not tickers:
    errors.append("Please provide at least one ticker.")

today = date.today()
if not start_str:
    start_str = (today - timedelta(days=365 * 3)).isoformat()
if not end_str:
    end_str = today.isoformat()
try:
    start = pd.to_datetime(start_str).tz_localize(None)
    end = pd.to_datetime(end_str).tz_localize(None)
    if start >= end:
        errors.append("Start date must be before End date.")
except Exception:
    errors.append("Start/End must be valid dates (YYYY-MM-DD).")

if errors:
    for msg in errors:
        st.error(msg)
    st.stop()

cov_model_key = "forecasted_sigma" if cov_choice.startswith("EWMA") else "dcc"

with st.spinner("Fetching prices..."):
    prices_df = _fetch_prices(tickers, start.isoformat(), end.isoformat())
    bench_px = _fetch_prices([bench], start.isoformat(), end.isoformat()).rename(columns={bench: "BENCH"})

if prices_df is None or prices_df.empty or prices_df.dropna(how="all").empty:
    st.error("No price data available for the selected tickers and date range. Try different dates or tickers.")
    st.stop()

cfg = ExpectedConfig(model=model, horizon_days=int(horizon), exogenous=use_exog)
with st.spinner("Running per-ticker forecasts..."):
    table_raw, mu_array = _run_forecasts(prices_df, cfg)

mu_series = pd.Series(mu_array, index=table_raw.index, dtype=float).reindex(prices_df.columns).fillna(0.0)

with st.spinner("Forecasting covariance and simulating portfolio paths..."):
    sigma = forecasted_sigma(prices_df, model=cov_model_key, horizon_days=1)
    weights = pd.Series(1.0 / len(prices_df.columns), index=prices_df.columns)
    sim_cfg = SimConfig(distribution=dist, student_t_df=int(df), sim_paths=int(paths), seed=42)
    try:
        sims = simulate_paths(mu_series.values, sigma, horizon=int(horizon), config=sim_cfg)
        port = portfolio_paths(sims, weights)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    summary = summarize_paths(port)

    # Adjust table for daily vs cumulative display
    display_table = table_raw.copy()
    if show_cumulative:
        display_table = _apply_cumulative(display_table, int(horizon), dist, int(df))

    steps_to_show = list(range(1, min(int(horizon), 6) + 1))
    table_formatted, table_for_csv, column_map = _format_table(display_table, steps_to_show)

    if "Mean d1" in table_formatted.columns and (table_formatted["Mean d1"] == "—").all():
        st.warning("⚠️ Forecasts returned all zeros — using historical means instead.")

    st.subheader("Ticker Forecasts")
    st.dataframe(table_formatted, use_container_width=True)
    st.caption(
        f"Values shown as {'cumulative' if show_cumulative else 'daily'} statistics. Displaying d1..d{steps_to_show[-1]}. "
        "Use the chart below for cumulative path bands."
    )
    st.download_button(
        "Download table CSV",
        table_for_csv.to_csv(index=True).encode("utf-8"),
        "ticker_forecasts.csv",
        "text/csv",
    )

    # Portfolio simulation chart with better aesthetics
    st.subheader("Portfolio Forward Simulation")
    cum_paths = (1 + pd.DataFrame(port.T)).cumprod()

    def _flat(arr):
        if isinstance(arr, (pd.Series, pd.Index)):
            return arr.to_numpy()
        return np.asarray(arr).reshape(-1,)

    timeline = np.arange(1, len(cum_paths) + 1, dtype=int)
    p05_values = cum_paths.quantile(0.05, axis=1).to_numpy()
    median_values = cum_paths.quantile(0.5, axis=1).to_numpy()
    p95_values = cum_paths.quantile(0.95, axis=1).to_numpy()
    timeline = _flat(timeline)
    p05_values = _flat(p05_values)
    median_values = _flat(median_values)
    p95_values = _flat(p95_values)
    L = min(len(timeline), len(p05_values), len(median_values), len(p95_values))
    plot_df = pd.DataFrame(
        {
            "Step": timeline[:L],
            "P05": p05_values[:L],
            "Median": median_values[:L],
            "P95": p95_values[:L],
        }
    )

    if plot_df.empty or plot_df[["P05", "Median", "P95"]].isna().all().all():
        st.warning("No simulation data to plot. Try adjusting the horizon or rerunning forecasts.")
    else:
        if px is None:
            import matplotlib.pyplot as plt

            fig_, ax = plt.subplots()
            ax.plot(plot_df["Step"], plot_df["Median"], label="Median")
            ax.plot(plot_df["Step"], plot_df["P05"], label="P05")
            ax.plot(plot_df["Step"], plot_df["P95"], label="P95")
            ax.set_title("Portfolio Forward Simulation (Median & 5–95% band)")
            ax.set_xlabel("Step")
            ax.set_ylabel("Cumulative return (start=1.00)")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig_)
        else:
            plot_long = plot_df.melt(id_vars=["Step"], var_name="Series", value_name="Value")
            fig = px.line(
                plot_long,
                x="Step",
                y="Value",
                color="Series",
                title="Portfolio Forward Simulation (Median & 5–95% band)",
            )
            fig.update_layout(
                xaxis_title="Step",
                yaxis_title="Cumulative return (start=1.00)",
                legend_title_text="",
                template="plotly_white",
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True, tickformat=".2f")
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Median = 50th percentile. P05/P95 = 5th/95th percentile across simulated paths.")

    # VaR / CVaR presentation
    st.subheader("VaR / CVaR")
    var_lines, quantile_df = _format_var(summary)
    st.markdown("\n".join(f"- {line}" for line in var_lines))
    if not quantile_df.empty:
        st.table(quantile_df)

    # Benchmark alpha display with formatted numbers
    b_table, b_mu_array = _run_forecasts(bench_px, cfg)
    bench_mu_series = pd.Series(b_mu_array, index=b_table.index, dtype=float)
    bench_mu_value = float(bench_mu_series.iloc[0]) if not bench_mu_series.empty else 0.0
    alpha_next = float(mu_series.mean() - bench_mu_value)
    port_std = float(np.nan_to_num(display_table.get("std(d1)", pd.Series(dtype=float)).mean(), nan=0.0))
    bench_std = float(np.nan_to_num(b_table.get("std(d1)", pd.Series(dtype=float)).mean(), nan=0.0))
    alpha_std = float(np.sqrt(max(port_std**2 + bench_std**2, 1e-12)))
    ci68 = (alpha_next - alpha_std, alpha_next + alpha_std)
    ci95 = (alpha_next - 1.96 * alpha_std, alpha_next + 1.96 * alpha_std)

    st.subheader("Benchmark Alpha")
    alpha_lines = _format_alpha(alpha_next, ci68, ci95)
    st.markdown("\n".join(f"- {line}" for line in alpha_lines))

    # Download helpers
summary_payload = {
    key: (float(val) if not isinstance(val, pd.Series) else {str(k): float(v) for k, v in val.items()})
    for key, val in summary.items()
}
st.download_button(
    "Download summary JSON",
    json.dumps(summary_payload, indent=2).encode("utf-8"),
    "simulation_summary.json",
    "application/json",
)

if show_debug:
    debug_df = pd.DataFrame(
        {
            "Item": ["Prices", "Returns", "Forecast table", "Sigma"],
            "Shape": [str(prices_df.shape), str(to_returns(prices_df).shape), str(table_raw.shape), str(sigma.shape)],
        }
    )
    st.caption("Debug shapes")
    st.table(debug_df)
