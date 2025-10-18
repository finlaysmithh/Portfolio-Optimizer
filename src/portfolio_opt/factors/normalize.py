from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    for c in out.columns:
        s = out[c]
        mu = np.nanmean(s)
        sd = np.nanstd(s, ddof=0)
        if sd == 0 or np.isnan(sd):
            out[c] = 0.0
        else:
            out[c] = (s - mu) / sd
    return out


def minmax01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi - lo == 0 or np.isnan(hi - lo):
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)
