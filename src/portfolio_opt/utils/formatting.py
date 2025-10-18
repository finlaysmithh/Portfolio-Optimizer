from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _is_num(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        if value.size != 1:
            return False
        value = float(value.reshape(-1)[0])
    return isinstance(value, (int, float, np.floating)) and np.isfinite(value)


def fmt_pct(value: Any, digits: int = 2) -> str:
    if not _is_num(value):
        return "—"
    return f"{float(value) * 100:+.{digits}f}%"


def fmt_bp(value: Any) -> str:
    if not _is_num(value):
        return "—"
    return f"{float(value) * 10000:.0f} bp"


def fmt_float(value: Any, digits: int = 4) -> str:
    if not _is_num(value):
        return "—"
    return f"{float(value):.{digits}f}"


__all__ = ["fmt_pct", "fmt_bp", "fmt_float"]
