from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def iso_dt(dt: datetime | str | None) -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    return dt.strftime("%Y-%m-%d")


def hash_key(*parts: Iterable[str | int | float]) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode("utf-8"))
    return m.hexdigest()[:16]


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return df


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def monthly_periods(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.Index(pd.to_datetime(pd.Series(dates).dt.to_period("M").dt.to_timestamp("M")))


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = ensure_datetime_index(prices)
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
