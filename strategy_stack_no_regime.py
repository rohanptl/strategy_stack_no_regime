#!/usr/bin/env python3
"""
wealthfront_strategy_stack_backtest.py

Automates a rules-based ETF allocation process using only the tickers in
WealthfrontETFs.txt, and includes a backtest that compares weekly vs monthly
vs quarterly rebalancing side by side.

Key constraints
---------------
- Only use ETFs from WealthfrontETFs.txt
- Target portfolio always sums to 100%
- Maximum allocation per ETF = 20%
- SGOV is treated as cash / defensive sleeve and is NOT capped at 20%
- IBIT and ETHA are treated like any other ETF (no fixed 5% allocation)
- No ETF outside the Wealthfront ETF list is ever suggested

Book-derived indicator settings used
------------------------------------
- 10-day EMA, 50-day SMA, 200-day SMA applied at the ETF level
- MACD (12, 26, 9)
- 89-day breakout confirmation
- 13-day exit warning
- ATR(15)
- 6-month relative-strength ranking as a practical ETF-rotation assumption
  compatible with the books' multi-month trend orientation

Important model choice
----------------------
This version does NOT use a broad market regime filter such as SPY > 200-day SMA.
Selection is always driven by ETF-level strength, trend quality, breakout behavior,
and volatility-aware scoring. SGOV is only used as residual cash if not enough
qualified ETFs are available.

Why 6-month ranking is an assumption
------------------------------------
The books provide the regime/trend/breakout/risk framework, but they do not
give one exact Wealthfront-specific ETF allocation formula. The ranking horizon
for ETF rotation is therefore parameterized and easy to modify.

What is still missing from the source material in this conversation
-------------------------------------------------------------------
1) Exact PMO implementation:
   The Ord/DecisionPoint PMO is proprietary and not fully specified in the
   conversation. This script leaves a hook for a weekly proxy if you want to
   add one later.

2) One canonical book-defined target-weight optimizer:
   The books are rich in signal logic, but they do not define one exact
   portfolio-weight formula for ETF allocation. This script therefore uses a
   transparent score + cap + normalize process.

Run examples
------------
1) Generate today's target allocation and rebalance files:
python wealthfront_strategy_stack_backtest.py ^
  --mode allocate ^
  --universe ".\\WealthfrontETFs.txt" ^
  --holdings ".\\holdings.csv" ^
  --export-prefix wealthfront_book_stack

2) Backtest weekly / monthly / quarterly side by side:
python wealthfront_strategy_stack_backtest.py ^
  --mode backtest ^
  --universe ".\\WealthfrontETFs.txt" ^
  --start 2020-01-01 ^
  --end 2026-03-01 ^
  --top-k 10 ^
  --export-prefix wealthfront_book_stack_bt
"""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local execution expectation:
# install with:
#   pip install yfinance pandas numpy openpyxl
try:
    import yfinance as yf
except Exception as e:
    raise SystemExit(
        "This script requires yfinance for market data.\n"
        "Install it locally with: pip install yfinance\n"
        f"Original import error: {e}"
    )

DEFAULT_CASH_TICKER = "SGOV"
DEFAULT_TOP_K = 10
DEFAULT_MAX_ALLOC = 0.20

DEFAULT_EMA_FAST = 10
DEFAULT_SMA_MID = 50
DEFAULT_SMA_LONG = 200

DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9

DEFAULT_BREAKOUT_DAYS = 89
DEFAULT_EXIT_DAYS = 13
DEFAULT_ATR_DAYS = 15
DEFAULT_VOL_DAYS = 20
DEFAULT_MOMENTUM_LOOKBACK_DAYS = 126  # ~6 months

REBALANCE_MAP = {
    "W": "W-FRI",
    "M": "ME",
    "Q": "QE",
}


def print_progress(message: str) -> None:
    """Simple flushed terminal progress logger."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)



def parse_universe_file(path: str) -> List[str]:
    """
    WealthfrontETFs.txt is assumed to contain alternating fund names and tickers.
    We keep only lines that look like tickers.
    """
    tickers: List[str] = []
    seen = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip().upper()
            if not line:
                continue
            if re.fullmatch(r"[A-Z]{1,6}", line) or re.fullmatch(r"[A-Z]{1,5}[.-][A-Z]{1,3}", line):
                if line not in seen:
                    seen.add(line)
                    tickers.append(line)

    if not tickers:
        raise ValueError("No tickers found in universe file.")
    return tickers


def read_holdings(path: str) -> pd.DataFrame:
    """
    Flexible holdings parser.

    Supported input styles:
      ticker,shares
      ticker,allocation_pct
      ticker,current_alloc_%
      ticker,weight
      ticker,market_value
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("holdings.csv must contain a 'ticker' column")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"] != ""].copy()

    out = pd.DataFrame({"ticker": df["ticker"]})

    if "shares" in df.columns:
        out["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)

    weight_cols = ["allocation_pct", "current_alloc_%", "current_alloc_pct", "weight", "current_weight"]
    mv_cols = ["market_value", "value", "current_value"]

    found_weight = next((c for c in weight_cols if c in df.columns), None)
    if found_weight:
        vals = pd.to_numeric(df[found_weight], errors="coerce").fillna(0.0)
        out["current_weight"] = np.where(vals > 1.0, vals / 100.0, vals)
    else:
        found_mv = next((c for c in mv_cols if c in df.columns), None)
        if found_mv:
            mv = pd.to_numeric(df[found_mv], errors="coerce").fillna(0.0)
            total = float(mv.sum())
            out["current_weight"] = mv / total if total > 0 else 0.0

    return out.groupby("ticker", as_index=False).sum(numeric_only=True)


def download_ohlcv_history(tickers: List[str], start: Optional[str], end: Optional[str]) -> Dict[str, pd.DataFrame]:
    """
    Downloads OHLCV history per ticker so we can calculate ATR and avoid
    multi-index column hassles.
    """
    out: Dict[str, pd.DataFrame] = {}
    total = len(tickers)
    print_progress(f"Starting price download for {total} tickers from {start or 'max'} to {end or 'latest'}")

    for i, t in enumerate(tickers, start=1):
        if i == 1 or i == total or i % 10 == 0:
            print_progress(f"Downloading {i}/{total}: {t}")
        df = yf.download(
            t,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            interval="1d",
            threads=False,
        )
        if df is None or len(df) == 0:
            print_progress(f"  No data returned for {t}; skipping")
            continue
        df = df.sort_index()
        # normalize columns if yfinance returns multiindex for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        out[t] = df

    print_progress(f"Finished price download. Usable tickers: {len(out)}/{total}")
    return out


def get_close_series(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {}
    for t, df in ohlcv.items():
        if "Adj Close" in df.columns:
            closes[t] = df["Adj Close"].copy()
        elif "Close" in df.columns:
            closes[t] = df["Close"].copy()
    if not closes:
        raise ValueError("No close series available.")
    return pd.DataFrame(closes).sort_index()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def macd(series: pd.Series, fast: int = DEFAULT_MACD_FAST, slow: int = DEFAULT_MACD_SLOW, signal: int = DEFAULT_MACD_SIGNAL):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = DEFAULT_ATR_DAYS) -> pd.Series:
    return true_range(high, low, close).rolling(window).mean()


def realized_vol(series: pd.Series, window: int = DEFAULT_VOL_DAYS) -> pd.Series:
    return series.pct_change().rolling(window).std() * math.sqrt(252)


def weekly_proxy_macd_hist(series: pd.Series) -> pd.Series:
    """
    PMO exact implementation is not available here, so this is a hook using
    weekly-close MACD histogram as a slower momentum proxy.
    """
    weekly = series.resample("W-FRI").last().dropna()
    _, _, h = macd(weekly)
    return h.reindex(series.index, method="ffill")


def determine_market_regime(spy_close: pd.Series, asof: pd.Timestamp):
    """
    Regime filter intentionally disabled in this version.
    Kept as a stub so the rest of the script structure remains simple.
    """
    return None


def build_snapshot_metrics(
    universe: List[str],
    ohlcv: Dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    cash_ticker: str,
    momentum_lookback_days: int,
    breakout_days: int,
    exit_days: int,
    atr_days: int,
) -> pd.DataFrame:
    rows = []

    for ticker in universe:
        if ticker not in ohlcv:
            continue

        df = ohlcv[ticker].loc[:asof].copy()
        if len(df) < max(DEFAULT_SMA_LONG, breakout_days + 5, momentum_lookback_days + 5):
            continue

        close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        high = df["High"]
        low = df["Low"]

        sma50 = sma(close, DEFAULT_SMA_MID)
        sma200 = sma(close, DEFAULT_SMA_LONG)
        ema10 = ema(close, DEFAULT_EMA_FAST)
        macd_line, signal_line, hist = macd(close)
        atr15 = atr(high, low, close, atr_days)
        rv20 = realized_vol(close, DEFAULT_VOL_DAYS)
        weekly_hist = weekly_proxy_macd_hist(close)

        prior_high = close.shift(1).rolling(breakout_days).max()
        prior_low = close.shift(1).rolling(exit_days).min()

        latest = float(close.iloc[-1])
        ret_6m = float(latest / close.iloc[-momentum_lookback_days] - 1.0)

        rows.append(
            {
                "ticker": ticker,
                "last_close": latest,
                "sma50": float(sma50.iloc[-1]),
                "sma200": float(sma200.iloc[-1]),
                "ema10": float(ema10.iloc[-1]),
                "macd": float(macd_line.iloc[-1]),
                "macd_signal": float(signal_line.iloc[-1]),
                "macd_hist": float(hist.iloc[-1]),
                "weekly_macd_hist": float(weekly_hist.iloc[-1]) if not pd.isna(weekly_hist.iloc[-1]) else np.nan,
                "atr15": float(atr15.iloc[-1]) if not pd.isna(atr15.iloc[-1]) else np.nan,
                "realized_vol20": float(rv20.iloc[-1]) if not pd.isna(rv20.iloc[-1]) else np.nan,
                "ret_6m": ret_6m,
                "breakout_89d": bool(latest > prior_high.iloc[-1]) if not pd.isna(prior_high.iloc[-1]) else False,
                "exit_13d": bool(latest < prior_low.iloc[-1]) if not pd.isna(prior_low.iloc[-1]) else False,
                "above_ema10": bool(latest > float(ema10.iloc[-1])),
                "above_sma200": bool(latest > float(sma200.iloc[-1])),
                "sma50_gt_sma200": bool(float(sma50.iloc[-1]) > float(sma200.iloc[-1])),
                "cash_like": ticker == cash_ticker,
            }
        )

    if not rows:
        # Return an empty frame with expected metric columns so that downstream
        # merging and CSV exports can proceed without missing-column errors.
        return pd.DataFrame(
            columns=[
                "ticker", "last_close", "sma50", "sma200", "ema10",
                "macd", "macd_signal", "macd_hist", "weekly_macd_hist",
                "atr15", "realized_vol20", "ret_6m",
                "breakout_89d", "exit_13d",
                "above_ema10", "above_sma200", "sma50_gt_sma200",
                "cash_like",
            ]
        )
    return pd.DataFrame(rows)


def capped_normalize(weights: pd.Series, max_cap: float, uncapped_tickers: Optional[set] = None) -> pd.Series:
    """
    Normalize to 100% while capping all but uncapped tickers.
    """
    uncapped_tickers = uncapped_tickers or set()
    w = weights.copy().fillna(0.0).clip(lower=0.0)

    if w.sum() <= 0:
        return w

    w = w / w.sum()

    for _ in range(20):
        over = [t for t in w.index if t not in uncapped_tickers and w[t] > max_cap + 1e-12]
        if not over:
            break

        capped_sum = 0.0
        for t in over:
            w[t] = max_cap

        capped_mask = pd.Series(False, index=w.index)
        for t in w.index:
            if (t not in uncapped_tickers) and (w[t] >= max_cap - 1e-12):
                capped_mask[t] = True

        capped_sum = float(w[capped_mask].sum())
        free_mask = ~capped_mask
        free_sum = float(w[free_mask].sum())

        if free_sum > 0:
            w.loc[free_mask] = w.loc[free_mask] / free_sum * (1.0 - capped_sum)

    total = float(w.sum())
    return w / total if total > 0 else w


def score_and_allocate(
    metrics: pd.DataFrame,
    cash_ticker: str,
    top_k: int,
    max_alloc: float,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(
            {
                "ticker": [cash_ticker],
                "target_weight": [1.0],
                "raw_score": [0.0],
                "last_close": [np.nan],
                "sma50": [np.nan],
                "sma200": [np.nan],
                "ema10": [np.nan],
                "macd": [np.nan],
                "macd_signal": [np.nan],
                "macd_hist": [np.nan],
                "weekly_macd_hist": [np.nan],
                "atr15": [np.nan],
                "realized_vol20": [np.nan],
                "ret_6m": [np.nan],
                "breakout_89d": [False],
                "exit_13d": [False],
                "above_ema10": [False],
                "above_sma200": [False],
                "sma50_gt_sma200": [False],
                "cash_like": [True],
            }
        )
    
    df = metrics.copy()
    df["eligible"] = False

    for idx, row in df.iterrows():
        t = row["ticker"]
        if t == cash_ticker:
            df.at[idx, "eligible"] = True
            continue

        if row["above_sma200"] and not row["exit_13d"]:
            df.at[idx, "eligible"] = True

    score = np.zeros(len(df), dtype=float)
    score += df["above_sma200"].astype(float) * 3.0
    score += df["sma50_gt_sma200"].astype(float) * 2.0
    score += df["above_ema10"].astype(float) * 1.0
    score += (df["macd_hist"] > 0).astype(float) * 1.5
    score += (df["weekly_macd_hist"] > 0).astype(float) * 1.0
    score += df["breakout_89d"].astype(float) * 2.0

    if df["ret_6m"].notna().any():
        rs_scaled = (df["ret_6m"] - df["ret_6m"].min()) / (df["ret_6m"].max() - df["ret_6m"].min() + 1e-12)
        score += rs_scaled * 4.0

    if df["realized_vol20"].notna().any():
        inv_vol_scaled = 1.0 - (
            (df["realized_vol20"] - df["realized_vol20"].min()) /
            (df["realized_vol20"].max() - df["realized_vol20"].min() + 1e-12)
        )
        score += inv_vol_scaled * 1.5

    score -= df["exit_13d"].astype(float) * 4.0
    df["raw_score"] = score

    # zero out ineligible non-cash
    mask_zero = (~df["eligible"]) & (~df["cash_like"])
    df.loc[mask_zero, "raw_score"] = 0.0

    # select top-k non-cash names
    ranked = df.loc[~df["cash_like"]].sort_values("raw_score", ascending=False)
    allowed = set(ranked.head(top_k)["ticker"].tolist())

    df["selected"] = df["ticker"].isin(allowed) | df["cash_like"]

    for idx, row in df.iterrows():
        if row["ticker"] != cash_ticker and row["raw_score"] <= 0:
            df.at[idx, "selected"] = False

    df["target_weight"] = 0.0

    selected_non_cash = df[df["selected"] & (~df["cash_like"])].copy()

    if selected_non_cash.empty:
        df.loc[df["ticker"] == cash_ticker, "target_weight"] = 1.0
        return df

    w = selected_non_cash.set_index("ticker")["raw_score"].clip(lower=0.0)
    if w.sum() <= 0:
        df.loc[df["ticker"] == cash_ticker, "target_weight"] = 1.0
        return df

    w = capped_normalize(w, max_cap=max_alloc, uncapped_tickers=set())
    for t, wt in w.items():
        df.loc[df["ticker"] == t, "target_weight"] = float(wt)

    residual = 1.0 - float(df["target_weight"].sum())
    df.loc[df["ticker"] == cash_ticker, "target_weight"] += residual

    total = float(df["target_weight"].sum())
    df["target_weight"] = df["target_weight"] / total
    return df


def rationale_for_row(row: pd.Series, cash_ticker: str) -> str:
    t = row["ticker"]
    if t == cash_ticker:
        if row["target_weight"] > 0:
            return f"{cash_ticker} used as residual cash sleeve when not enough ETFs qualify; uncapped."
        return f"{cash_ticker} available as defensive sleeve but not currently needed."

    reasons = []
    if row["above_sma200"]:
        reasons.append("above 200SMA")
    if row["sma50_gt_sma200"]:
        reasons.append("50SMA above 200SMA")
    if row["above_ema10"]:
        reasons.append("holding above 10EMA")
    if row["macd_hist"] > 0:
        reasons.append("positive MACD histogram")
    if row["weekly_macd_hist"] > 0:
        reasons.append("positive weekly momentum proxy")
    if row["breakout_89d"]:
        reasons.append("at 89-day breakout")
    if pd.notna(row["ret_6m"]):
        reasons.append(f"6m return {row['ret_6m']:.1%}")
    if pd.notna(row["realized_vol20"]):
        reasons.append(f"20d vol {row['realized_vol20']:.1%}")

    if row["target_weight"] > 0:
        return "Selected because " + ", ".join(reasons) + "."
    fails = []
    if not row["above_sma200"]:
        fails.append("below 200SMA")
    if row["exit_13d"]:
        fails.append("13-day exit condition active")
    if row["raw_score"] <= 0:
        fails.append("composite score not strong enough")
    if not fails:
        fails.append("did not make final top-ranked set")
    return "Not selected because " + ", ".join(fails) + "."


def derive_current_weights(holdings: pd.DataFrame, close_px: pd.DataFrame) -> pd.DataFrame:
    h = holdings.copy()

    if "current_weight" in h.columns:
        total = float(h["current_weight"].sum())
        if total > 0:
            h["current_weight"] = h["current_weight"] / total
            return h[["ticker", "current_weight"]]

    if "shares" in h.columns:
        latest = []
        for _, row in h.iterrows():
            t = row["ticker"]
            p = float(close_px[t].dropna().iloc[-1]) if t in close_px.columns and len(close_px[t].dropna()) else np.nan
            latest.append(p)
        h["last_price"] = latest
        h["market_value"] = h["shares"] * h["last_price"]
        total = float(h["market_value"].sum())
        h["current_weight"] = h["market_value"] / total if total > 0 else 0.0
        return h[["ticker", "current_weight"]]

    h["current_weight"] = 0.0
    return h[["ticker", "current_weight"]]


def allocation_mode(
    universe_path: str,
    holdings_path: str,
    cash_ticker: str,
    top_k: int,
    export_prefix: str,
    start: Optional[str],
    end: Optional[str],
):
    print_progress("Loading universe and holdings")
    universe = parse_universe_file(universe_path)
    holdings = read_holdings(holdings_path)

    bad = sorted(set(holdings["ticker"]) - set(universe))
    if bad:
        raise ValueError(f"Holdings contain tickers not in Wealthfront universe: {bad}")

    if cash_ticker not in universe:
        raise ValueError(f"{cash_ticker} must exist in WealthfrontETFs.txt")

    tickers = sorted(set(universe) | {"SPY"})
    print_progress(f"Universe loaded: {len(universe)} ETFs")
    ohlcv = download_ohlcv_history(tickers, start=start, end=end)
    print_progress("Building close-price matrix")
    close_px = get_close_series(ohlcv)

    if "SPY" not in close_px.columns:
        raise ValueError("Could not download SPY data.")

    asof = close_px.dropna(how="all").index[-1]
    print_progress(f"Computing strategy snapshot as of {asof.date()}")

    metrics = build_snapshot_metrics(
        universe=universe,
        ohlcv=ohlcv,
        asof=asof,
        cash_ticker=cash_ticker,
        momentum_lookback_days=DEFAULT_MOMENTUM_LOOKBACK_DAYS,
        breakout_days=DEFAULT_BREAKOUT_DAYS,
        exit_days=DEFAULT_EXIT_DAYS,
        atr_days=DEFAULT_ATR_DAYS,
    )

    print_progress("Scoring ETFs and generating target allocation")
    alloc = score_and_allocate(metrics, cash_ticker, top_k, DEFAULT_MAX_ALLOC)

    print_progress("Reconciling current holdings against target weights")
    current = derive_current_weights(holdings, close_px)
    out = alloc.merge(current, on="ticker", how="left")
    
    out["current_weight"] = out["current_weight"].fillna(0.0)
    out["delta_weight"] = out["target_weight"] - out["current_weight"]
    out["current_alloc_pct"] = out["current_weight"] * 100.0
    out["target_alloc_pct"] = out["target_weight"] * 100.0
    out["delta_pct_points"] = out["delta_weight"] * 100.0
    out["action"] = np.where(out["delta_weight"] > 1e-6, "INCREASE",
                      np.where(out["delta_weight"] < -1e-6, "DECREASE", "HOLD"))
    out["rationale"] = out.apply(lambda r: rationale_for_row(r, cash_ticker), axis=1)

    action_view = out[
        ~((out["current_weight"].abs() < 1e-12) & (out["target_weight"].abs() < 1e-12))
    ].copy().sort_values(["target_weight", "raw_score", "ticker"], ascending=[False, False, True])

    metrics_cols = [
        "ticker", "last_close", "ret_6m", "realized_vol20", "atr15",
        "above_ema10", "above_sma200", "sma50_gt_sma200",
        "macd", "macd_signal", "macd_hist", "weekly_macd_hist",
        "breakout_89d", "exit_13d", "raw_score",
        "current_alloc_pct", "target_alloc_pct", "delta_pct_points",
        "action", "rationale",
    ]
    action_cols = ["ticker", "current_alloc_pct", "target_alloc_pct", "delta_pct_points", "action", "rationale"]

    print_progress("Writing output files")
    action_view[metrics_cols].to_csv(f"{export_prefix}_full_metrics.csv", index=False)
    action_view[action_cols].to_csv(f"{export_prefix}_rebalance_actions.csv", index=False)

    print("=" * 88)
    print("WEALTHFRONT ETF STRATEGY STACK")
    print("=" * 88)
    print(f"As of: {asof.date()}")
    print("Market regime filter: DISABLED")
    print("Selection is driven by ETF-level trend, momentum, breakout, and volatility scoring.")
    print()
    print(action_view[action_cols].to_string(index=False))
    print()
    print(f"Wrote: {export_prefix}_full_metrics.csv")
    print(f"Wrote: {export_prefix}_rebalance_actions.csv")


def metrics_from_equity_curve(equity: pd.Series, turnover: pd.Series) -> Dict[str, float]:
    daily_ret = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0)
    vol = float(daily_ret.std() * math.sqrt(252))
    sharpe = float((daily_ret.mean() * 252) / vol) if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Avg Turnover/Rebalance": float(turnover.mean()) if len(turnover) else 0.0,
        "Num Rebalances": int((turnover > 0).sum()),
    }


def run_schedule_backtest(
    universe: List[str],
    ohlcv: Dict[str, pd.DataFrame],
    close_px: pd.DataFrame,
    schedule_code: str,
    cash_ticker: str,
    top_k: int,
    start: str,
    end: str,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Rebalance at schedule dates using information available at each rebalance close.
    Apply new weights from next trading day open proxy => implemented as next close-to-close
    approximation for simplicity and reproducibility.
    """
    px = close_px.loc[start:end, universe].dropna(how="all")
    if px.empty:
        raise ValueError(f"No price data available in range {start} to {end}")

    rebalance_dates = px.resample(REBALANCE_MAP[schedule_code]).last().index
    rebalance_dates = [d for d in rebalance_dates if d in px.index]
    if len(rebalance_dates) < 2:
        raise ValueError(f"Not enough rebalance points for schedule {schedule_code}")

    schedule_name = {"W": "Weekly", "M": "Monthly", "Q": "Quarterly"}.get(schedule_code, schedule_code)
    print_progress(f"Starting {schedule_name} backtest with {len(rebalance_dates)} rebalance dates")

    weights = pd.Series(0.0, index=universe)
    weights[cash_ticker] = 1.0

    equity = pd.Series(index=px.index, dtype=float)
    equity.iloc[0] = 1.0
    turnovers = []
    weights_history = []

    pending_weights = weights.copy()

    for i, dt in enumerate(px.index):
        if i == 0:
            weights_history.append({"date": dt, **weights.to_dict()})
            continue

        prev_dt = px.index[i - 1]
        daily_ret_vec = px.loc[dt] / px.loc[prev_dt] - 1.0
        portfolio_ret = float((weights.fillna(0.0) * daily_ret_vec.fillna(0.0)).sum())
        equity.iloc[i] = equity.iloc[i - 1] * (1.0 + portfolio_ret)

        # Rebalance signal generated at today's close, effective for next day
        if dt in rebalance_dates:
            if len(turnovers) == 0 or (len(turnovers) + 1) % 10 == 0 or dt == rebalance_dates[-1]:
                print_progress(f"{schedule_name}: processing rebalance {len(turnovers)+1}/{len(rebalance_dates)} on {dt.date()}")
            metrics = build_snapshot_metrics(
                universe=universe,
                ohlcv=ohlcv,
                asof=dt,
                cash_ticker=cash_ticker,
                momentum_lookback_days=DEFAULT_MOMENTUM_LOOKBACK_DAYS,
                breakout_days=DEFAULT_BREAKOUT_DAYS,
                exit_days=DEFAULT_EXIT_DAYS,
                atr_days=DEFAULT_ATR_DAYS,
            )
            alloc = score_and_allocate(metrics, cash_ticker, top_k, DEFAULT_MAX_ALLOC)
            new_weights = alloc.set_index("ticker")["target_weight"].reindex(universe).fillna(0.0)
            turnover = float((new_weights - weights).abs().sum() / 2.0)
            turnovers.append({"date": dt, "turnover": turnover})
            pending_weights = new_weights
            weights_history.append({"date": dt, **new_weights.to_dict()})

        # Apply pending weights on next bar
        weights = pending_weights.copy()

    equity = equity.ffill()
    turnover_df = pd.DataFrame(turnovers)
    weights_df = pd.DataFrame(weights_history)
    print_progress(f"Finished {schedule_name} backtest")
    return equity, turnover_df, weights_df


def backtest_mode(
    universe_path: str,
    cash_ticker: str,
    top_k: int,
    export_prefix: str,
    start: str,
    end: str,
):
    print_progress("Loading ETF universe for backtest")
    universe = parse_universe_file(universe_path)
    if cash_ticker not in universe:
        raise ValueError(f"{cash_ticker} must exist in WealthfrontETFs.txt")

    tickers = sorted(set(universe) | {"SPY"})
    # Adjust start date to include buffer for lookback calculations
    start_dt = datetime.fromisoformat(start)
    buffer_days = 250
    adjusted_start = (start_dt - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    print_progress(f"Backtest window: {start} to {end} (using buffered download start {adjusted_start})")
    print_progress(f"Universe loaded: {len(universe)} ETFs")
    ohlcv = download_ohlcv_history(tickers, start=adjusted_start, end=end)
    print_progress("Building close-price matrix")
    close_px = get_close_series(ohlcv)

    schedules = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q"}

    equity_curves = {}
    turnover_tables = {}
    weight_tables = {}
    summary_rows = []

    for label, code in schedules.items():
        print_progress(f"Running {label.lower()} rebalance comparison")
        eq, to_df, w_df = run_schedule_backtest(
            universe=universe,
            ohlcv=ohlcv,
            close_px=close_px,
            schedule_code=code,
            cash_ticker=cash_ticker,
            top_k=top_k,
            start=start,
            end=end,
        )
        equity_curves[label] = eq
        turnover_tables[label] = to_df
        weight_tables[label] = w_df

        stats = metrics_from_equity_curve(eq, to_df["turnover"] if not to_df.empty else pd.Series(dtype=float))
        stats["Schedule"] = label
        summary_rows.append(stats)

    summary = pd.DataFrame(summary_rows)[
        ["Schedule", "Total Return", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Calmar", "Avg Turnover/Rebalance", "Num Rebalances"]
    ].sort_values("CAGR", ascending=False)

    equity_df = pd.DataFrame(equity_curves)

    print_progress("Writing backtest output files")
    summary.to_csv(f"{export_prefix}_backtest_summary.csv", index=False)
    equity_df.to_csv(f"{export_prefix}_equity_curves.csv", index=True)

    for label, df in turnover_tables.items():
        df.to_csv(f"{export_prefix}_{label.lower()}_turnover.csv", index=False)
    for label, df in weight_tables.items():
        df.to_csv(f"{export_prefix}_{label.lower()}_weights_history.csv", index=False)

    print("=" * 110)
    print("BACKTEST SUMMARY: WEEKLY vs MONTHLY vs QUARTERLY")
    print("=" * 110)
    display = summary.copy()
    for c in ["Total Return", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Calmar", "Avg Turnover/Rebalance"]:
        if c in display.columns:
            display[c] = display[c].map(lambda x: f"{x:.2%}" if isinstance(x, (float, np.floating)) and c != "Sharpe" and c != "Calmar" else x)
    # Sharpe & Calmar as ratios
    display["Sharpe"] = summary["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    display["Calmar"] = summary["Calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    print(display.to_string(index=False))
    print()
    print(f"Wrote: {export_prefix}_backtest_summary.csv")
    print(f"Wrote: {export_prefix}_equity_curves.csv")
    print(f"Wrote schedule turnover and weights-history CSVs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["allocate", "backtest"], required=True)
    p.add_argument("--universe", required=True, help="Path to WealthfrontETFs.txt")
    p.add_argument("--holdings", help="Path to holdings.csv (required for --mode allocate)")
    p.add_argument("--cash-ticker", default=DEFAULT_CASH_TICKER)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--start", help="Backtest / download start date, e.g. 2020-01-01")
    p.add_argument("--end", help="Backtest / download end date, e.g. 2026-03-01")
    p.add_argument("--export-prefix", default="wealthfront_book_stack")
    return p


def main():
    args = build_parser().parse_args()

    if args.mode == "allocate":
        if not args.holdings:
            raise SystemExit("--holdings is required for --mode allocate")
        allocation_mode(
            universe_path=args.universe,
            holdings_path=args.holdings,
            cash_ticker=args.cash_ticker.upper(),
            top_k=args.top_k,
            export_prefix=args.export_prefix,
            start=args.start,
            end=args.end,
        )
    else:
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required for --mode backtest")
        backtest_mode(
            universe_path=args.universe,
            cash_ticker=args.cash_ticker.upper(),
            top_k=args.top_k,
            export_prefix=args.export_prefix,
            start=args.start,
            end=args.end,
        )


if __name__ == "__main__":
    main()
