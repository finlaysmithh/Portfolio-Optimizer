from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap(
    returns: pd.Series, block: int = 21, n_paths: int = 500, length: int | None = None
) -> pd.DataFrame:
    r = returns.dropna().to_numpy()
    if length is None:
        length = len(r)
    rng = np.random.default_rng(123)
    paths: list[list[float]] = []
    for _ in range(n_paths):
        out: list[float] = []
        while len(out) < length:
            start = rng.integers(0, max(1, len(r) - block))
            out.extend(r[start : start + block])
        paths.append(out[:length])
    idx = pd.RangeIndex(length)
    df = pd.DataFrame(paths, columns=idx)
    return df


def stress_drawdowns(paths: pd.DataFrame) -> pd.Series:
    dds = []
    for _, row in paths.iterrows():
        eq = (1 + pd.Series(row)).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        dds.append(dd)
    return pd.Series(dds)
