#!/usr/bin/env python3
from __future__ import annotations
import shutil
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import itertools
import math

import numpy as np
import pandas as pd

# Import from your main strategy file
import strategy_stack_no_regime as strat


DEFAULT_START_DATES = [
    "2026-01-01",
    "2025-06-01",
    "2025-01-01",
    "2024-06-01",
    "2024-01-01",
    "2023-06-01",
    "2023-01-01",
    "2022-06-01",
    "2022-01-01",
    "2021-06-01",
    "2021-01-01",
    "2020-06-01",
    "2020-01-01",
    "2019-06-01",
    "2019-01-01",
]

DEFAULT_TOP_KS = [3, 4, 5, 6]
DEFAULT_ALLOCATION_MODES = ["score_proportional"]
ALLOCATION_MODE_CHOICES = ["equal", "score_proportional", "momentum_proportional"]
DEFAULT_EXECUTION_MODES = ["overlay", "pure_topk"]
DEFAULT_SCHEDULES = ["weekly", "biweekly"]
SCHEDULE_CODE_MAP = {"weekly": "W", "biweekly": "BW"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast in-process Weekly-only grid backtest for strategy_stack_no_regime.py"
    )
    p.add_argument("--universe", required=True, help="Path to universe file")
    p.add_argument("--cash-ticker", default="SGOV")
    p.add_argument(
        "--max-alloc", type=float, default=strat.DEFAULT_MAX_ALLOC,
        help="Maximum allocation per non-cash ETF as a 0-1 fraction. Use 1.0 for no cap.",
    )
    p.add_argument("--hold-band", type=int, default=None)
    p.add_argument("--hold-bands", nargs="*", type=int, default=None, help="Optional list of hold-band values to sweep.")
    p.add_argument("--transaction-cost-bps", type=int, default=0)
    p.add_argument("--max-wait-pullback", type=int, default=3)
    p.add_argument(
        "--max-wait-pullbacks",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of max-wait-pullback values to sweep.",
    )
    p.add_argument("--price-cache-dir", default="price_cache")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--start-dates", nargs="*", default=DEFAULT_START_DATES)
    p.add_argument("--top-ks", nargs="*", type=int, default=DEFAULT_TOP_KS)
    p.add_argument(
        "--allocation-modes",
        nargs="*",
        default=DEFAULT_ALLOCATION_MODES,
        choices=ALLOCATION_MODE_CHOICES,
    )
    p.add_argument(
        "--execution-modes",
        nargs="*",
        default=DEFAULT_EXECUTION_MODES,
        choices=DEFAULT_EXECUTION_MODES,
    )
    p.add_argument(
        "--schedules",
        nargs="*",
        default=DEFAULT_SCHEDULES,
        choices=list(SCHEDULE_CODE_MAP.keys()),
        help="Rebalance schedules to test.",
    )
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--output-dir", default="grid_backtest_inprocess_runs")
    p.add_argument(
        "--save-equity-curves",
        action="store_true",
        help="Also save per-run weekly equity curves",
    )
    p.add_argument(
        "--clean-output-dir",
        action="store_true",
        help="Delete old files in output-dir before starting a new run.",
    )
    p.add_argument(
        "--clean-cache-dir",
        action="store_true",
        help="Delete old files in price-cache-dir before starting a new run.",
    )
    return p.parse_args()


def metrics_from_equity_curve(equity: pd.Series, turnover: pd.Series) -> Dict[str, float]:
    daily_ret = equity.pct_change().fillna(0.0)
    n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0)
    vol = float(daily_ret.std() * math.sqrt(252))
    sharpe = float((daily_ret.mean() * 252) / vol) if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    turnover = turnover.dropna() if len(turnover) else turnover

    return {
        "Total Return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "CAGR": cagr,
        "Annual Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Avg Turnover/Trade Date": float(turnover.mean()) if len(turnover) else 0.0,
        "Num Trade Dates": int((turnover > 0).sum()) if len(turnover) else 0,
    }


def run_schedule_backtest_fast(
    universe: List[str],
    ohlcv: Dict[str, pd.DataFrame],
    close_px: pd.DataFrame,
    feature_store: Dict[str, pd.DataFrame],
    cash_ticker: str,
    top_k: int,
    start: str,
    end: str,
    schedule_code: str,
    max_alloc: float,
    hold_band: Optional[int],
    transaction_cost_bps: int,
    allocation_mode: str,
    execution_mode: str,
    max_wait_pullback: int,
):
    """
    Schedule-based, drift-aware, MOC execution.
    Minimal memory version: returns equity + turnover only.
    """
    px = close_px.loc[start:end, universe].sort_index().ffill().dropna(how="all")
    if px.empty:
        raise ValueError(f"No price data in {start} → {end}")

    rebalance_dates = {d for d in px.resample(strat.REBALANCE_MAP[schedule_code]).last().index if d in px.index}
    if not rebalance_dates:
        raise ValueError(f"No {schedule_code} rebalance dates in {start} → {end}")

    weights = pd.Series(0.0, index=universe, dtype=float)
    weights[cash_ticker] = 1.0
    current_selected: Set[str] = set()

    equity = 1.0
    equity_vals = []
    turnover_rows = []

    for i, dt in enumerate(px.index):
        if i == 0:
            equity_vals.append(equity)
            continue

        prev_dt = px.index[i - 1]
        daily_ret_vec = (px.loc[dt] / px.loc[prev_dt] - 1.0).fillna(0.0)

        # earn daily return with starting weights
        port_ret = float((weights * daily_ret_vec).sum())
        equity *= (1.0 + port_ret)

        # drift weights to close
        gross = weights * (1.0 + daily_ret_vec)
        gross_sum = float(gross.sum())
        drifted_weights = gross / gross_sum if gross_sum > 1e-12 else weights.copy()

        new_weights = drifted_weights.copy()

        if dt in rebalance_dates:
            metrics = strat.build_snapshot_metrics(universe, feature_store, dt)

            alloc = strat.select_candidates(
                metrics=metrics,
                cash_ticker=cash_ticker,
                top_k=top_k,
                max_alloc=max_alloc,
                allocation_mode=allocation_mode,
                execution_mode=execution_mode,
                max_wait_pullback=max_wait_pullback,
                prev_holdings=current_selected,
                hold_band=hold_band,
            )

            current_selected = set(
                alloc.loc[alloc["selected"] & ~alloc["cash_like"], "ticker"].tolist()
            )

            target_w_map = alloc.set_index("ticker")["target_weight"]
            target_w = target_w_map.reindex(universe).fillna(0.0)
            turnover = float((target_w - drifted_weights).abs().sum() / 2.0)

            if transaction_cost_bps > 0 and turnover > 0:
                cost_frac = turnover * 2.0 * transaction_cost_bps / 10_000.0
                equity *= max(0.0, 1.0 - cost_frac)

            new_weights = target_w
            turnover_rows.append({"date": dt, "turnover": turnover})

        weights = new_weights
        equity_vals.append(equity)

    equity_series = pd.Series(equity_vals, index=px.index, name="equity")
    turnover_df = pd.DataFrame(turnover_rows)
    return equity_series, turnover_df

def reset_directory(path_str: str, label: str) -> None:
    path = Path(path_str)
    if path.exists():
        print(f"Cleaning {label}: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def main() -> None:
    args = parse_args()
    if not 0 < args.max_alloc <= 1:
        raise ValueError("max_alloc must be in the interval (0, 1].")
    hold_bands = args.hold_bands if args.hold_bands is not None else [args.hold_band]
    max_wait_pullbacks = (
        args.max_wait_pullbacks if args.max_wait_pullbacks is not None else [args.max_wait_pullback]
    )
    if args.clean_output_dir:
        reset_directory(args.output_dir, "output directory")
    else:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.clean_cache_dir:
        reset_directory(args.price_cache_dir, "price cache directory")
    else:
        Path(args.price_cache_dir).mkdir(parents=True, exist_ok=True)

    outdir = Path(args.output_dir)

    universe = strat.parse_universe_file(args.universe)
    if args.cash_ticker.upper() not in universe:
        raise ValueError(f"{args.cash_ticker.upper()} must exist in universe")

    tickers = sorted(set(universe) | strat.COMPARISON_BENCHMARK_TICKERS)

    earliest_start = min(args.start_dates)
    buffered_start = (datetime.fromisoformat(earliest_start) - timedelta(days=550)).strftime("%Y-%m-%d")

    print(f"Loading prices once for {len(tickers)} tickers")
    print(f"Date range: {buffered_start} → {args.end}")
    print(f"Investable universe: {len(universe)} ETFs from file")

    ohlcv = strat.download_ohlcv_history(
        tickers=tickers,
        start=buffered_start,
        end=args.end,
        cache_dir=args.price_cache_dir,
        refresh_cache=args.refresh_cache,
    )
    close_px = strat.get_close_series(ohlcv)

    print("Precomputing features once")
    feature_store = strat.precompute_feature_store(
        universe=universe,
        ohlcv=ohlcv,
        cash_ticker=args.cash_ticker.upper(),
        momentum_lookback_days=strat.DEFAULT_MOMENTUM_LOOKBACK_DAYS,
        breakout_days=strat.DEFAULT_BREAKOUT_DAYS,
        exit_days=strat.DEFAULT_EXIT_DAYS,
        atr_days=strat.DEFAULT_ATR_DAYS,
    )

    combos = list(itertools.product(
        args.start_dates,
        args.top_ks,
        hold_bands,
        max_wait_pullbacks,
        args.allocation_modes,
        args.execution_modes,
        args.schedules,
    ))

    print(f"Running {len(combos)} schedule combinations")

    results = []
    failure_rows = []

    for idx, (start_date, top_k, hold_band, max_wait_pullback, allocation_mode, execution_mode, schedule_name) in enumerate(combos, start=1):
        schedule_code = SCHEDULE_CODE_MAP[schedule_name]
        print(
            f"[{idx}/{len(combos)}] RUNNING | "
            f"start={start_date} | schedule={schedule_name} | top_k={top_k} | hold_band={hold_band} | "
            f"max_wait_pullback={max_wait_pullback} | alloc={allocation_mode} | exec={execution_mode}"
        )
        try:
            eq, to_df = run_schedule_backtest_fast(
                universe=universe,
                ohlcv=ohlcv,
                close_px=close_px,
                feature_store=feature_store,
                cash_ticker=args.cash_ticker.upper(),
                top_k=top_k,
                start=start_date,
                end=args.end,
                schedule_code=schedule_code,
                max_alloc=args.max_alloc,
                hold_band=hold_band,
                transaction_cost_bps=args.transaction_cost_bps,
                allocation_mode=allocation_mode,
                execution_mode=execution_mode,
                max_wait_pullback=max_wait_pullback,
            )

            stats = metrics_from_equity_curve(
                eq,
                to_df["turnover"] if not to_df.empty else pd.Series(dtype=float),
            )

            benchmark_stats = {}
            for benchmark in sorted(strat.COMPARISON_BENCHMARK_TICKERS):
                if benchmark in close_px.columns:
                    bench_eq = strat.benchmark_series(close_px, start_date, args.end, benchmark)
                    bench_metrics = metrics_from_equity_curve(bench_eq, pd.Series(dtype=float))
                    benchmark_stats.update({
                        f"{benchmark} Total Return": bench_metrics["Total Return"],
                        f"{benchmark} CAGR": bench_metrics["CAGR"],
                        f"{benchmark} Sharpe": bench_metrics["Sharpe"],
                        f"{benchmark} Max Drawdown": bench_metrics["Max Drawdown"],
                    })
                    benchmark_stats[f"Excess Total Return vs {benchmark}"] = (
                        stats["Total Return"] - bench_metrics["Total Return"]
                    )
                    benchmark_stats[f"Excess CAGR vs {benchmark}"] = (
                        stats["CAGR"] - bench_metrics["CAGR"]
                    )

            stats.update({
                "Schedule": schedule_name.title(),
                "start_date": start_date,
                "end_date": args.end,
                "top_k": top_k,
                "max_alloc": args.max_alloc,
                "allocation_mode": allocation_mode,
                "execution_mode": execution_mode,
                "hold_band": hold_band,
                "transaction_cost_bps": args.transaction_cost_bps,
                "max_wait_pullback": max_wait_pullback,
            })
            stats.update(benchmark_stats)
            results.append(stats)

            print(
                f"[{idx}/{len(combos)}] SUCCESS | "
                f"CAGR={stats['CAGR']:.2%} | Sharpe={stats['Sharpe']:.2f} | MaxDD={stats['Max Drawdown']:.2%}"
            )

            if args.save_equity_curves:
                run_name = (
                    f"start_{start_date}__schedule_{schedule_name}__topk_{top_k}"
                    f"__hold_{hold_band}__wait_{max_wait_pullback}"
                    f"__alloc_{allocation_mode}__exec_{execution_mode}"
                ).replace(":", "-")
                eq.to_csv(outdir / f"{run_name}__weekly_equity.csv", index=True)

        except Exception as e:
            failure_rows.append({
                "start_date": start_date,
                "schedule": schedule_name,
                "top_k": top_k,
                "hold_band": hold_band,
                "max_wait_pullback": max_wait_pullback,
                "allocation_mode": allocation_mode,
                "execution_mode": execution_mode,
                "error": str(e),
            })
            print(
                f"[{idx}/{len(combos)}] FAILED | "
                f"start={start_date} | schedule={schedule_name} | top_k={top_k} | hold_band={hold_band} | "
                f"max_wait_pullback={max_wait_pullback} | alloc={allocation_mode} | exec={execution_mode} | {e}"
            )

    if results:
        df = pd.DataFrame(results).sort_values(["CAGR", "Sharpe"], ascending=[False, False], na_position="last")
        combined_path = outdir / "weekly_grid_combined_results.csv"
        df.to_csv(combined_path, index=False)

        top10_path = outdir / "weekly_grid_top10.csv"
        df.head(10).to_csv(top10_path, index=False)

        print(f"Wrote: {combined_path}")
        print(f"Wrote: {top10_path}")

    if failure_rows:
        fail_df = pd.DataFrame(failure_rows)
        fail_path = outdir / "weekly_grid_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        print(f"Wrote: {fail_path}")


if __name__ == "__main__":
    main()
