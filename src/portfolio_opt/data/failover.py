from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

Provider = Callable[[Sequence[str], str, str], pd.DataFrame]


@dataclass
class ProviderInfo:
    name: str
    fn: Provider


def first_healthy(providers: list[ProviderInfo], tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    last: pd.DataFrame | None = None
    for p in providers:
        try:
            df = p.fn(tickers, start, end)
            if not df.empty:
                return df
            last = df
        except Exception:
            continue
    return last if last is not None else pd.DataFrame()
