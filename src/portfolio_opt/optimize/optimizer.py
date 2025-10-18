from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..factors.compute import sector_map
from ..risk.liquidity import check as liquidity_check
from .constraints import Constraints


@dataclass
class OptimizeResult:
    tickers: list[str]
    weights: pd.Series
    exposures: dict[str, float]
    diagnostics: dict[str, float]


def _apply_sector_cap(selection: list[str], sectors: dict[str, str], cap: float) -> list[str]:
    from collections import defaultdict

    counts: dict[str, int] = defaultdict(int)
    result: list[str] = []
    max_per_sector = max(1, int(np.floor(cap * len(selection))))
    for t in selection:
        s = sectors.get(t, "Unknown")
        if counts[s] < max_per_sector:
            result.append(t)
            counts[s] += 1
    return result


def _tilt_for_exposure(
    weights: pd.Series, zscores: pd.DataFrame, key: str, target: float, alpha: float = 0.2
) -> pd.Series:
    exp = float(np.dot(weights.values, zscores.loc[weights.index, key].values))
    if exp >= target:
        return weights
    tilt = 1.0 + alpha * zscores.loc[weights.index, key]
    tilt = tilt.clip(lower=0.1)
    w = weights * tilt
    w = w.clip(lower=0)
    w = w / w.sum()
    return w


def optimize(
    zscores: pd.DataFrame,
    prices: pd.DataFrame,
    constraints: Constraints | None = None,
    prev_weights: pd.Series | None = None,
) -> OptimizeResult:
    constraints = constraints or Constraints()
    z = zscores.copy()
    tickers = list(z.index)

    # Liquidity screen (avg dollar volume heuristic)
    liquid = [t for t in tickers if liquidity_check(t)]
    z = z.loc[liquid]
    if z.empty:
        raise ValueError("No liquid tickers after screening")

    # Rank by composite and enforce sector cap at selection time
    z = z.sort_values("composite", ascending=False)
    sectors = sector_map(z.index)
    prelim = _apply_sector_cap(list(z.index), sectors, cap=constraints.sector_cap)

    k = int(np.clip(constraints.target_names, 5, 8))
    chosen = prelim[:k]
    # Ensure uniqueness
    chosen = list(dict.fromkeys(chosen))
    z_sel = z.loc[chosen]

    # Start equal-weighted
    w = pd.Series(1.0 / k, index=chosen)

    # Turnover penalty: tilt toward previous weights if provided
    if prev_weights is not None:
        prev = prev_weights.reindex(chosen).fillna(0)
        lam = float(np.clip(1.0 - constraints.turnover_penalty, 0.0, 1.0))
        w = lam * w + (1 - lam) * prev
        w = w / w.sum()

    # Single-name cap
    w = w.clip(upper=constraints.single_name_cap)
    w = w / w.sum()

    # Exposure tilts
    for _ in range(10):
        w_old = w.copy()
        w = _tilt_for_exposure(w, z_sel, "value", constraints.min_value_exposure, alpha=0.35)
        w = _tilt_for_exposure(w, z_sel, "momentum", constraints.min_momentum_exposure, alpha=0.3)
        # Enforce single-name cap and renormalize
        w = w.clip(upper=constraints.single_name_cap)
        w = w / w.sum()
        if np.allclose(w.values, w_old.values, atol=1e-6):
            break

    # If exposures are still short, do a simple greedy swap
    def exposure_for(key: str, weights: pd.Series, zframe: pd.DataFrame) -> float:
        return float(np.dot(weights.values, zframe[key].loc[weights.index].values))

    for key, target in [
        ("value", constraints.min_value_exposure),
        ("momentum", constraints.min_momentum_exposure),
    ]:
        exp = exposure_for(key, w, z_sel)
        if exp < target:
            # find replacement candidate
            remaining = z[~z.index.isin(chosen)]
            if not remaining.empty:
                # pick best candidate not already chosen
                best_new = None
                for cand in remaining.sort_values(key, ascending=False).index:
                    if cand not in chosen:
                        best_new = cand
                        break
                if best_new is None:
                    continue
                worst_old = z_sel[key].idxmin()
                if best_new != worst_old:
                    chosen = [t for t in chosen if t != worst_old] + [best_new]
                    chosen = list(dict.fromkeys(chosen))
                    z_sel = z.loc[chosen]
                    w = pd.Series(1.0 / len(chosen), index=chosen)
                    w = _tilt_for_exposure(w, z_sel, key, target, alpha=0.4)
                    w = w.clip(upper=constraints.single_name_cap)
                    w = w / w.sum()

    exposures = {
        "value": float(np.dot(w.values, z_sel["value"].values)),
        "momentum": float(np.dot(w.values, z_sel["momentum"].values)),
    }

    diagnostics = {
        "n_selected": float(len(chosen)),
        "single_name_cap": float(constraints.single_name_cap),
    }

    return OptimizeResult(tickers=chosen, weights=w, exposures=exposures, diagnostics=diagnostics)
