from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Regime:
    percentile: float
    is_stress: bool
    threshold: float
    latest: float


def vix_series() -> pd.Series:
    try:
        import yfinance as yf

        vix = yf.download("^VIX", period="2y", progress=False)["Close"]
        s = vix.astype(float)
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        idx = pd.date_range(end=pd.Timestamp.today(), periods=252 * 2, freq="B")
        rng = np.random.default_rng(123)
        s = pd.Series(rng.normal(20, 5, size=len(idx)), index=idx)
        return s.clip(lower=10)


def compute_regime(percentile: float = 0.8) -> Regime:
    s = vix_series().dropna()
    thr = float(np.quantile(s.values, percentile))
    latest = float(s.iloc[-1])
    return Regime(percentile=percentile, is_stress=latest >= thr, threshold=thr, latest=latest)


def regime_overlays(config: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = compute_regime(percentile=(config or {}).get("percentile", 0.8))
    # Example overlays
    overlays: dict[str, Any] = {
        "max_weight": 0.2 if reg.is_stress else 0.25,
        "min_cash": 0.05 if reg.is_stress else 0.0,
        "regime": reg.__dict__,
    }
    return overlays
