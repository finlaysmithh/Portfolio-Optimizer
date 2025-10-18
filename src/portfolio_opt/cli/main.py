from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..backtest.engine import backtest
from ..data.fetchers import fetch_price_history
from ..data.universe import load_tickers_from_file, sp500_tickers
from ..factors.compute import DEFAULT_UNIVERSE, compute_factors
from ..optimize.constraints import Constraints
from ..optimize.optimizer import optimize
from ..reporting.tearsheet import save_tearsheet


def _resolve_tickers(args: argparse.Namespace) -> list[str]:
    # Priority: --tickers-file > --tickers > --universe (sp500) > default demo
    if getattr(args, "tickers_file", None):
        return load_tickers_from_file(args.tickers_file)
    if getattr(args, "tickers", None):
        return [t.strip().upper() for t in str(args.tickers).split(",") if t.strip()]
    if getattr(args, "universe", None) == "sp500":
        syms = sp500_tickers()
        if syms:
            return syms
    return DEFAULT_UNIVERSE


def cmd_fetch(args: argparse.Namespace) -> None:
    tickers = _resolve_tickers(args)
    px = fetch_price_history(tickers, start=args.since, end=args.until)
    print(f"fetched: {list(px.columns)} from {px.index.min().date()} to {px.index.max().date()}")


def cmd_screen(args: argparse.Namespace) -> None:
    tickers = _resolve_tickers(args)
    px = fetch_price_history(
        tickers, start="2020-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    f = compute_factors(px)
    topn = int(args.top)
    print(f.head(topn))


def cmd_optimize(args: argparse.Namespace) -> None:
    tickers = _resolve_tickers(args)
    px = fetch_price_history(
        tickers, start="2020-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    f = compute_factors(px)
    cons = Constraints(
        target_names=int(args.target_names),
        min_value_exposure=float(args.min_value),
        min_momentum_exposure=float(args.min_momo),
    )
    res = optimize(f, px, cons)
    df = pd.DataFrame(
        {"ticker": res.tickers, "weight": [float(res.weights[t]) for t in res.tickers]}
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"saved weights to {args.out}")
    else:
        print(df)


def cmd_backtest(args: argparse.Namespace) -> None:
    tickers = _resolve_tickers(args)
    px = fetch_price_history(
        tickers, start=args.since, end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    f = compute_factors(px)
    cons = Constraints(target_names=6)
    res = optimize(f, px, cons)
    # Simple backtest: equal-weight selected names, monthly rebal, with costs
    sub_px = px[res.tickers]

    def gen(_):
        return res.weights

    bt = backtest(sub_px, rebalance_freq="M", cost_bps=float(args.cost_bps), weight_generator=gen)
    print(
        {
            "start": str(bt.equity_curve.index.min().date()),
            "end": str(bt.equity_curve.index.max().date()),
            "final_equity": float(bt.equity_curve.iloc[-1]),
        }
    )


def cmd_tearsheet(args: argparse.Namespace) -> None:
    tickers = _resolve_tickers(args)
    px = fetch_price_history(
        tickers, start="2020-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    f = compute_factors(px)
    cons = Constraints(target_names=6)
    res = optimize(f, px, cons)
    sub_px = px[res.tickers]

    def gen(_):
        return res.weights

    bt = backtest(sub_px, rebalance_freq="M", cost_bps=5.0, weight_generator=gen)
    path = save_tearsheet(
        {"equity_curve": bt.equity_curve, "returns": bt.returns}, res.exposures, args.out
    )
    print(f"saved tear sheet: {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="po", description="portfolio-optimizer-pro CLI")
    sp = p.add_subparsers(dest="cmd")

    p_fetch = sp.add_parser("fetch", help="Fetch and cache data")
    p_fetch.add_argument("--universe", default=None, choices=["sp500"], help="Built-in universe")
    p_fetch.add_argument("--tickers", default=None, help="Comma-separated tickers")
    p_fetch.add_argument("--tickers-file", default=None, help="Path to newline/CSV list of tickers")
    p_fetch.add_argument("--since", default="2018-01-01")
    p_fetch.add_argument("--until", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p_fetch.set_defaults(func=cmd_fetch)

    p_screen = sp.add_parser("screen", help="Screen universe")
    p_screen.add_argument("--universe", default=None, choices=["sp500"], help="Built-in universe")
    p_screen.add_argument("--tickers", default=None, help="Comma-separated tickers")
    p_screen.add_argument(
        "--tickers-file", default=None, help="Path to newline/CSV list of tickers"
    )
    p_screen.add_argument("--top", default=30)
    p_screen.set_defaults(func=cmd_screen)

    p_opt = sp.add_parser("optimize", help="Optimize portfolio")
    p_opt.add_argument("--universe", default=None, choices=["sp500"], help="Built-in universe")
    p_opt.add_argument("--tickers", default=None, help="Comma-separated tickers")
    p_opt.add_argument("--tickers-file", default=None, help="Path to newline/CSV list of tickers")
    p_opt.add_argument("--target-names", default=6)
    p_opt.add_argument("--min-value", default=0.3)
    p_opt.add_argument("--min-momo", default=0.2)
    p_opt.add_argument("--out", default=None)
    p_opt.set_defaults(func=cmd_optimize)

    p_bt = sp.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("--universe", default=None, choices=["sp500"], help="Built-in universe")
    p_bt.add_argument("--tickers", default=None, help="Comma-separated tickers")
    p_bt.add_argument("--tickers-file", default=None, help="Path to newline/CSV list of tickers")
    p_bt.add_argument("--since", default="2018-01-01")
    p_bt.add_argument("--rebalance", default="monthly")
    p_bt.add_argument("--cost-bps", default=5)
    p_bt.set_defaults(func=cmd_backtest)

    p_ts = sp.add_parser("tearsheet", help="Export tear sheet PDF")
    p_ts.add_argument("--universe", default=None, choices=["sp500"], help="Built-in universe")
    p_ts.add_argument("--tickers", default=None, help="Comma-separated tickers")
    p_ts.add_argument("--tickers-file", default=None, help="Path to newline/CSV list of tickers")
    p_ts.add_argument("--out", default="docs/images/tearsheet.pdf")
    p_ts.set_defaults(func=cmd_tearsheet)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
