from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _drawdown(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    return (eq / peak) - 1.0


def save_tearsheet(results: dict, exposures: dict[str, float] | None, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    eq: pd.Series = results["equity_curve"]
    ret: pd.Series = results["returns"]

    with PdfPages(out) as pdf:
        # Page 1: Equity curve
        fig, ax = plt.subplots(figsize=(10, 6))
        eq.plot(ax=ax, color="navy", lw=2)
        ax.set_title("Portfolio Equity Curve")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Drawdown
        fig, ax = plt.subplots(figsize=(10, 4))
        dd = _drawdown(eq)
        dd.plot(ax=ax, color="crimson", lw=1.5)
        ax.set_title("Drawdown")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Rolling Sharpe (60d)
        fig, ax = plt.subplots(figsize=(10, 4))
        roll = ret.rolling(60).mean() / (ret.rolling(60).std() + 1e-9) * np.sqrt(252)
        roll.plot(ax=ax, color="darkgreen")
        ax.set_title("Rolling Sharpe (60d)")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Factor exposures
        if exposures:
            fig, ax = plt.subplots(figsize=(8, 4))
            items = list(exposures.items())
            keys, vals = zip(*items, strict=False)
            ax.bar(keys, vals, color="#1f77b4")
            ax.set_ylim(0, max(0.5, max(vals) * 1.2))
            ax.set_title("Factor Exposures")
            ax.grid(True, axis="y", alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)

    return out
