from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..integration.expected import ExpectedConfig, forecasted_mu
from ..integration.risk import forecasted_sigma
from ..sim.forward import SimConfig, simulate_paths, portfolio_paths, summarize_paths
from ..data.fetchers import fetch_price_history  # type: ignore


def _read_config_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser(description="Forecast CLI: per-ticker and portfolio projections")
    ap.add_argument("--tickers-file", required=True, help="CSV with tickers column or plain list")
    ap.add_argument("--config", default="config.yaml", help="YAML config path")
    ap.add_argument("--out", default=None, help="Output directory (defaults to artifacts/forecast_YYYYMMDD)")
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional)")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    args = ap.parse_args()

    cfg = _read_config_yaml(Path(args.config))
    fcfg = (cfg.get("forecasting") or {})
    model = str(fcfg.get("model", "auto_arima"))
    horizon = int(fcfg.get("horizon_days", 21))
    exogenous = bool(fcfg.get("exogenous", False))
    dist = str(fcfg.get("distribution", "gaussian"))
    df = int(fcfg.get("student_t_df", 7))
    paths = int(fcfg.get("sim_paths", 10000))
    benchmark = str(fcfg.get("benchmark", "^GSPC"))

    # Load tickers
    tf = Path(args.tickers_file)
    if tf.suffix.lower() == ".csv":
        tickers = pd.read_csv(tf)
        if tickers.shape[1] == 1:
            tickers = tickers.iloc[:, 0].astype(str).tolist()
        elif "ticker" in tickers.columns:
            tickers = tickers["ticker"].astype(str).tolist()
        else:
            tickers = tickers.columns.astype(str).tolist()
    else:
        tickers = [t.strip() for t in tf.read_text().replace("\n", ",").split(",") if t.strip()]

    start = args.start
    end = args.end
    px = fetch_price_history(tickers, start=start, end=end)
    bench_px = fetch_price_history([benchmark], start=start, end=end)
    bench_px = bench_px.rename(columns={benchmark: "BENCH"})

    expected_cfg = ExpectedConfig(model=model, horizon_days=horizon, exogenous=exogenous)
    mu_table, mu_array = forecasted_mu(px, tickers, expected_cfg)
    mu_series = pd.Series(mu_array, index=mu_table.index, dtype=float)
    # Use daily covariance for step-wise simulation
    sigma = forecasted_sigma(px, model="forecasted_sigma", horizon_days=1)

    # Portfolio sim: equal weights by default for CLI
    w = pd.Series(1.0 / len(tickers), index=tickers)
    mu_daily = mu_series.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    sim_cfg = SimConfig(distribution=dist if dist in ("gaussian", "student_t") else "gaussian",
                        student_t_df=df, sim_paths=paths, seed=42)
    sims = simulate_paths(mu_daily, sigma, horizon=horizon, config=sim_cfg)
    port_paths = portfolio_paths(sims, w)
    summary = summarize_paths(port_paths)

    # Benchmark forecast (same machinery on BENCH)
    b_mu_table, _ = forecasted_mu(bench_px, ["BENCH"], expected_cfg)

    now = datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.out) if args.out else Path("artifacts") / f"forecast_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_table.to_csv(out_dir / "forecasts_per_ticker.csv")
    np.save(out_dir / "sigma_forecast.npy", sigma.values)
    with (out_dir / "portfolio_sim_summary.json").open("w") as f:
        json.dump({k: (float(v) if not isinstance(v, pd.Series) else v.to_dict()) for k, v in summary.items()}, f)
    b_mu_table.to_csv(out_dir / "benchmark_forecast.csv")

    print(f"Wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
