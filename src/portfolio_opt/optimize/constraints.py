from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Constraints:
    target_names: int = 6  # between 5 and 8 by default
    min_value_exposure: float = 0.30
    min_momentum_exposure: float = 0.20
    single_name_cap: float = 0.25
    sector_cap: float = 0.35
    turnover_penalty: float = 0.0  # 0.0 for demo
