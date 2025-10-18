from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from ..utils import ensure_datetime_index, to_returns
except ImportError:  # pragma: no cover - fallback for segmented imports
    def ensure_datetime_index(df):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        return df

    def to_returns(prices, method: str = "pct_change"):
        if prices is None or len(prices) == 0:
            return pd.DataFrame()
        if method == "pct_change":
            returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
        else:
            returns = np.log(prices / prices.shift(1))
        return returns.dropna(how="all")
from ..forecast.arima import ARIMAForecaster
from ..forecast.ets import ETSForecaster
from ..forecast.ml_linear import LinearMLForecaster
from ..forecast.base import ensure_series, safe_fallback


ExpectedSource = Literal["historical_mean", "forecasted_mu"]

logger = logging.getLogger(__name__)


@dataclass
class ExpectedConfig:
    model: Literal["auto_arima", "ets", "sarimax", "ml_lasso", "ml_ridge"] = "auto_arima"
    horizon_days: int = 21
    exogenous: bool = False
    min_history: int = 60


def _forecaster_for(model: str):
    if model == "ets":
        return ETSForecaster()
    if model == "ml_lasso":
        return LinearMLForecaster(model_type="lasso")
    if model == "ml_ridge":
        return LinearMLForecaster(model_type="ridge")
    # "auto_arima" and "sarimax" both map to ARIMAForecaster; exog presence decides SARIMAX usage
    return ARIMAForecaster()


def historical_mean_mu(prices: pd.DataFrame, horizon: int = 21) -> pd.Series:
    rets = to_returns(ensure_datetime_index(prices))
    mu_daily = rets.mean(axis=0).fillna(0.0)
    # Aggregate horizon mean approximately linearly for arithmetic returns
    return mu_daily * float(horizon)


def forecasted_mu(
    prices: pd.DataFrame, tickers: list[str] | None, cfg: ExpectedConfig, exog: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Produce per-ticker forecasts over horizon.

    Returns:
        table: DataFrame indexed by ticker with per-step metrics (mean/std/p05/p50/p95).
        mu_series: Series of first-step means aligned to ``tickers`` order.
    """
    rets = to_returns(ensure_datetime_index(prices))
    use = list(tickers or rets.columns)
    horizon = max(int(cfg.horizon_days), 1)
    metrics = [f"{metric}(d{step})" for metric in ("mean", "std", "p05", "p50", "p95") for step in range(1, horizon + 1)]

    def _build_row(symbol: str, mean_arr: np.ndarray, std_arr: np.ndarray) -> dict[str, float | str]:
        row: dict[str, float | str] = {"ticker": symbol}
        mean_arr = np.asarray(mean_arr, dtype=float)
        std_arr = np.asarray(std_arr, dtype=float)
        for step in range(horizon):
            m = float(mean_arr[step]) if step < len(mean_arr) else 0.0
            s = float(std_arr[step]) if step < len(std_arr) else 0.0
            if not np.isfinite(m):
                m = 0.0
            if not np.isfinite(s) or s < 0:
                s = 0.0
            label = step + 1
            row[f"mean(d{label})"] = m
            row[f"std(d{label})"] = s
            row[f"p05(d{label})"] = m - 1.645 * s
            row[f"p50(d{label})"] = m
            row[f"p95(d{label})"] = m + 1.645 * s
        return row

    rows: list[dict[str, float | str]] = []

    for ticker in use:
        # Gracefully handle missing columns
        if ticker not in rets.columns:
            logger.warning("No returns column for ticker %s; filling NaNs", ticker)
            rows.append(_build_row(ticker, np.zeros(horizon), np.zeros(horizon)))
            continue
        try:
            series = ensure_series(rets[ticker], name=ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to prepare returns for %s: %s", ticker, exc)
            rows.append(_build_row(ticker, np.zeros(horizon), np.zeros(horizon)))
            continue

        hist_mean, hist_std = safe_fallback(series)
        fallback_mean_arr = np.full(horizon, hist_mean, dtype=float)
        fallback_std_arr = np.full(horizon, max(hist_std, 0.0), dtype=float)

        if len(series) < int(cfg.min_history):
            logger.warning(
                "Insufficient history for %s (need >=%d, got %d)", ticker, cfg.min_history, len(series)
            )
            rows.append(_build_row(ticker, fallback_mean_arr, fallback_std_arr))
            continue

        forecaster = _forecaster_for(cfg.model)
        ex = None
        ex_future = None
        if cfg.exogenous and exog is not None:
            ex = exog.reindex(series.index).dropna()
            series = series.reindex(ex.index).dropna()
            if len(series) < int(cfg.min_history):
                logger.warning(
                    "Exogenous alignment removed history for %s (need >=%d, got %d)",
                    ticker,
                    cfg.min_history,
                    len(series),
                )
                rows.append(_build_row(ticker, fallback_mean_arr, fallback_std_arr))
                continue
            ex_future = ex.tail(1).copy()

        try:
            model = forecaster.fit(series, ex)
            pred = model.predict(horizon, ex_future)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Forecast failed for %s: %s", ticker, exc)
            rows.append(_build_row(ticker, fallback_mean_arr, fallback_std_arr))
            continue

        mean_vals = np.asarray(pred.get("mean", []), dtype=float)
        std_vals = np.asarray(pred.get("std", []), dtype=float)
        def _pad(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            out = np.full(horizon, np.nan, dtype=float)
            steps = min(len(arr), horizon)
            if steps > 0:
                out[:steps] = arr[:steps]
            return out

        mean_padded = _pad(mean_vals)
        std_padded = _pad(std_vals) if std_vals.size else np.full(horizon, np.nan, dtype=float)
        mean_padded = np.where(np.isfinite(mean_padded), mean_padded, fallback_mean_arr)
        std_valid_mask = (np.isfinite(std_padded)) & (std_padded > 0)
        std_padded = np.where(std_valid_mask, std_padded, fallback_std_arr)

        rows.append(
            _build_row(
                ticker,
                mean_padded,
                std_padded,
            )
        )

    if not rows:
        table = pd.DataFrame(columns=["ticker", *metrics]).set_index("ticker")
        mu_daily = np.array([], dtype=float)
        return table, mu_daily

    table = pd.DataFrame(rows).set_index("ticker")
    table.index = table.index.astype(str)
    table = table.reindex(use)
    table = table.apply(pd.to_numeric, errors="coerce")

    fill_map: dict[str, float] = {}
    if "mean(d1)" in table.columns:
        mean_vals = table["mean(d1)"].dropna()
        mean_stat = mean_vals.mean() if not mean_vals.empty else np.nan
        mean_fill = float(mean_stat) if np.isfinite(mean_stat) else 0.0
        fill_map["mean(d1)"] = mean_fill
    if "std(d1)" in table.columns:
        std_vals = table["std(d1)"].dropna()
        std_stat = std_vals.mean() if not std_vals.empty else np.nan
        std_fill = float(std_stat) if np.isfinite(std_stat) else 0.0
        fill_map["std(d1)"] = std_fill
    if "p05(d1)" in table.columns:
        fill_map["p05(d1)"] = 0.0
    if "p50(d1)" in table.columns:
        fill_map["p50(d1)"] = 0.0
    if "p95(d1)" in table.columns:
        fill_map["p95(d1)"] = 0.0
    if fill_map:
        table = table.fillna(fill_map)
    table = table.fillna(0.0)

    mu_daily = table.get("mean(d1)")
    if mu_daily is None:
        mu_daily_arr = np.zeros(len(table), dtype=float)
    else:
        mu_daily_arr = mu_daily.to_numpy(dtype=float)
    return table, mu_daily_arr
