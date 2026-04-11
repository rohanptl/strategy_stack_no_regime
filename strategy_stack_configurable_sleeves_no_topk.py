#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Install yfinance pandas numpy openpyxl. " + str(e))


# ── constants ──────────────────────────────────────────────────────────────

DEFAULT_CASH_TICKER            = "SGOV"
DEFAULT_TOP_K                  = 5
DEFAULT_MAX_ALLOC              = 1.00
DEFAULT_EMA_FAST               = 10
DEFAULT_SMA_MID                = 50
DEFAULT_SMA_LONG               = 200
DEFAULT_MACD_FAST              = 12
DEFAULT_MACD_SLOW              = 26
DEFAULT_MACD_SIGNAL            = 9
DEFAULT_BREAKOUT_DAYS          = 89
DEFAULT_BREAKOUT_RECENT_DAYS   = 5
DEFAULT_EXIT_DAYS              = 13
DEFAULT_ATR_DAYS               = 15
DEFAULT_VOL_DAYS               = 20
DEFAULT_MOMENTUM_LOOKBACK_DAYS = 126
DEFAULT_PULLBACK_EMA_BUFFER    = 0.02
DEFAULT_EXTENDED_FROM_EMA      = 0.05
DEFAULT_WAIT_PULLBACK_NEAR_EMA = 0.08
DEFAULT_WAIT_PULLBACK_MID_MULTIPLIER = 0.75
DEFAULT_WAIT_PULLBACK_FAR_MULTIPLIER = 0.45
DEFAULT_WAIT_PULLBACK_MIN_MULTIPLIER = 0.25
DEFAULT_WAIT_PULLBACK_MAX_MULTIPLIER = 1.00
DEFAULT_BREADTH_FULLY_RISK_ON  = 0.90
DEFAULT_DEFENSE_BREADTH_THRESHOLD = 0.45
DEFAULT_DEFENSE_PARTICIPATION_THRESHOLD = 0.25
DEFAULT_DEFENSE_ACTIVE_MULTIPLIER = 0.75
DEFAULT_DEFENSE_WAIT_MULTIPLIER = 0.60
DEFAULT_WAIT_PULLBACK_BREADTH_BONUS_SLOTS = 3
DEFAULT_BUY_NOW_MIN_BUDGET = 0.50
DEFAULT_BUY_NOW_MAX_BUDGET = 0.70
DEFAULT_WAIT_PULLBACK_MIN_BUDGET = 0.20
DEFAULT_WAIT_PULLBACK_MAX_BUDGET = 0.40
DEFAULT_MIN_EXECUTABLE_FOR_CASH_CAP = 5
DEFAULT_MAX_CASH_WHEN_ENOUGH = 0.20
DEFAULT_VOL_TARGETING_EXPONENT = 0.50
DEFAULT_REBALANCE_INERTIA = 0.70
DEFAULT_WAIT_ONLY_ACTIVE_MULTIPLIER = 0.50
DEFAULT_WAIT_ONLY_MAX_ACTIVE_MULTIPLIER = 0.80
DEFAULT_RECOVERING_TOTAL_BUDGET = 0.10
DEFAULT_RECOVERING_MAX_NAMES = 2
DEFAULT_RECOVERING_MIN_IMPROVEMENT = 2.5
DEFAULT_HOLD_BAND_OFFSET       = 2
DEFAULT_TRANSACTION_COST_BPS   = 0

COMPARISON_BENCHMARK_TICKERS = {"SPY", "QQQ"}
REBALANCE_MAP                = {"W": "W-FRI", "BW": "2W-FRI", "M": "ME", "Q": "QE"}


# ── utilities ──────────────────────────────────────────────────────────────

def print_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def print_step(step_no: int, total_steps: int, message: str) -> None:
    print_progress(f"Step {step_no}/{total_steps}: {message}")


def parse_universe_file(path: str) -> List[str]:
    tickers, seen = [], set()
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


def parse_key_value_string(spec: Optional[str], cast=float) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not spec:
        return out
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid key=value item: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = cast(value.strip())
    return out


def derive_sleeve_topk(top_k: int, sleeve_targets: Dict[str, float]) -> Dict[str, int]:
    active = {k: float(v) for k, v in sleeve_targets.items() if float(v) > 0}
    if not active or top_k <= 0:
        return {}
    total = sum(active.values())
    norm = {k: v / total for k, v in active.items()}
    raw = {k: norm[k] * top_k for k in norm}
    base = {k: int(math.floor(v)) for k, v in raw.items()}

    # ensure each active sleeve gets at least one slot when possible
    for k in norm:
        if top_k >= len(norm) and base[k] == 0:
            base[k] = 1

    assigned = sum(base.values())
    if assigned > top_k:
        # trim from largest buckets first
        for k, _ in sorted(base.items(), key=lambda kv: (-kv[1], kv[0])):
            while assigned > top_k and base[k] > 0:
                base[k] -= 1
                assigned -= 1
    elif assigned < top_k:
        remainders = sorted(
            ((raw[k] - math.floor(raw[k]), k) for k in norm),
            reverse=True,
        )
        idx = 0
        while assigned < top_k and remainders:
            _, k = remainders[idx % len(remainders)]
            base[k] += 1
            assigned += 1
            idx += 1
    return base


def load_configurable_universes(
    combined_universe_path: Optional[str],
    universe_3x_path: Optional[str],
    universe_2x_path: Optional[str],
    universe_broad_path: Optional[str],
) -> Tuple[List[str], Dict[str, str]]:
    ticker_to_sleeve: Dict[str, str] = {}
    ordered: List[str] = []

    def add_items(path: Optional[str], sleeve: Optional[str]) -> None:
        nonlocal ordered, ticker_to_sleeve
        if not path:
            return
        for t in parse_universe_file(path):
            if t not in ticker_to_sleeve and sleeve:
                ticker_to_sleeve[t] = sleeve
            elif sleeve:
                ticker_to_sleeve[t] = sleeve
            if t not in ordered:
                ordered.append(t)

    add_items(universe_3x_path, "3x")
    add_items(universe_2x_path, "2x")
    add_items(universe_broad_path, "broad")

    if not ordered:
        if not combined_universe_path:
            raise ValueError("A universe file is required.")
        ordered = parse_universe_file(combined_universe_path)

    for t in ordered:
        ticker_to_sleeve.setdefault(t, "general")

    return ordered, ticker_to_sleeve

def _cache_key(start: Optional[str], end: Optional[str]) -> str:
    s = start or "max"
    e = end or "latest"
    return f"{s}__{e}"


def _cache_file_for_ticker(cache_dir: str, ticker: str, start: Optional[str], end: Optional[str]) -> Path:
    key = _cache_key(start, end)
    return Path(cache_dir) / f"{ticker.upper()}__{key}.parquet"


def load_ohlcv_from_cache(
    tickers: List[str],
    cache_dir: str,
    start: Optional[str],
    end: Optional[str],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        path = _cache_file_for_ticker(cache_dir, t, start, end)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                if len(df) > 0 and "Close" in df.columns:
                    out[t] = df
            except Exception:
                pass
    return out


def save_ohlcv_to_cache(
    ohlcv: Dict[str, pd.DataFrame],
    cache_dir: str,
    start: Optional[str],
    end: Optional[str],
) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    for t, df in ohlcv.items():
        try:
            path = _cache_file_for_ticker(cache_dir, t, start, end)
            df.to_parquet(path)
        except Exception:
            # cache failure should not break strategy execution
            pass

def read_holdings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("holdings.csv must contain a 'ticker' column")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"] != ""].copy()

    weight_cols = ["allocation_pct", "current_alloc_%", "current_alloc_pct", "weight", "current_weight"]
    mv_cols     = ["market_value", "value", "current_value"]

    has_shares   = "shares" in df.columns
    found_weight = next((c for c in weight_cols if c in df.columns), None)
    found_mv     = next((c for c in mv_cols if c in df.columns), None)

    modes = sum(bool(x) for x in [has_shares, found_weight is not None, found_mv is not None])
    if modes == 0:
        raise ValueError("holdings file must contain shares, or a weight column, or a market value column")
    if modes > 1:
        raise ValueError("holdings file should use only one input mode: shares OR weight OR market value")

    out = pd.DataFrame({"ticker": df["ticker"]})

    if has_shares:
        out["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    elif found_weight:
        vals = pd.to_numeric(df[found_weight], errors="coerce").fillna(0.0)
        out["current_weight"] = np.where(vals > 1.0, vals / 100.0, vals)
    elif found_mv:
        mv    = pd.to_numeric(df[found_mv], errors="coerce").fillna(0.0)
        total = float(mv.sum())
        out["current_weight"] = mv / total if total > 0 else 0.0

    return out.groupby("ticker", as_index=False).sum(numeric_only=True)


# ── market data ────────────────────────────────────────────────────────────

def download_ohlcv_history(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    cache_dir: Optional[str] = None,
    refresh_cache: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Cache-aware OHLCV loader:
    - if cache_dir is provided, load cached parquet files first
    - download only missing tickers (or all if refresh_cache=True)
    - save newly downloaded tickers back to cache
    """
    tickers = list(dict.fromkeys([t.upper().strip() for t in tickers if t and str(t).strip()]))
    total = len(tickers)

    cached: Dict[str, pd.DataFrame] = {}
    if cache_dir and not refresh_cache:
        cached = load_ohlcv_from_cache(tickers, cache_dir, start, end)
        if cached:
            print_progress(f"Loaded {len(cached)}/{total} tickers from cache")

    missing = [t for t in tickers if t not in cached]
    if not missing:
        return cached

    print_progress(
        f"Downloading {len(missing)} missing tickers  |  {start or 'max'} → {end or 'latest'}"
    )

    downloaded: Dict[str, pd.DataFrame] = {}

    # ---- bulk path first ----
    try:
        bulk = yf.download(
            tickers=missing,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            interval="1d",
            threads=True,
            group_by="ticker",
        )

        if bulk is not None and len(bulk) > 0:
            if isinstance(bulk.columns, pd.MultiIndex):
                lvl0 = set(map(str, bulk.columns.get_level_values(0)))
                lvl1 = set(map(str, bulk.columns.get_level_values(1)))

                if set(missing).intersection(lvl0):
                    for t in missing:
                        if t not in lvl0:
                            continue
                        try:
                            df = bulk[t].copy()
                            if df is not None and len(df) > 0 and "Close" in df.columns:
                                df = df.sort_index().dropna(how="all")
                                if len(df) > 0:
                                    downloaded[t] = df
                        except Exception:
                            pass

                elif "Close" in lvl0 or "Open" in lvl0:
                    for t in missing:
                        try:
                            df = bulk.xs(t, level=1, axis=1).copy()
                            if df is not None and len(df) > 0 and "Close" in df.columns:
                                df = df.sort_index().dropna(how="all")
                                if len(df) > 0:
                                    downloaded[t] = df
                        except Exception:
                            pass

                else:
                    for t in missing:
                        for level_first in [0, 1]:
                            try:
                                df = bulk.xs(t, level=level_first, axis=1).copy()
                                if df is not None and len(df) > 0 and "Close" in df.columns:
                                    df = df.sort_index().dropna(how="all")
                                    if len(df) > 0:
                                        downloaded[t] = df
                                        break
                            except Exception:
                                continue

            else:
                if len(missing) == 1 and "Close" in bulk.columns:
                    df = bulk.sort_index().dropna(how="all")
                    if len(df) > 0:
                        downloaded[missing[0]] = df

    except Exception as e:
        print_progress(f"Bulk download warning: {e}")

    # ---- fallback for remaining missing tickers ----
    still_missing = [t for t in missing if t not in downloaded]
    if still_missing:
        print_progress(f"Falling back to single-ticker download for {len(still_missing)} symbols")
        for i, t in enumerate(still_missing, start=1):
            if i == 1 or i == len(still_missing) or i % 10 == 0:
                print_progress(f"  fallback {i}/{len(still_missing)}: {t}")
            ok = False
            for _ in range(2):
                try:
                    df = yf.download(
                        t,
                        start=start,
                        end=end,
                        auto_adjust=True,
                        progress=False,
                        interval="1d",
                        threads=False,
                    )
                    if df is not None and len(df) > 0:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [c[0] for c in df.columns]
                        df = df.sort_index().dropna(how="all")
                        if len(df) > 0 and "Close" in df.columns:
                            downloaded[t] = df
                            ok = True
                            break
                except Exception:
                    continue
            if not ok:
                pass

    if cache_dir and downloaded:
        save_ohlcv_to_cache(downloaded, cache_dir, start, end)

    out = {**cached, **downloaded}
    print_progress(f"Usable tickers: {len(out)}/{total}")
    return out


def get_close_series(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {t: df["Close"] for t, df in ohlcv.items() if "Close" in df.columns}
    if not closes:
        raise ValueError("No close series available.")
    return pd.DataFrame(closes).sort_index()


# ── technical indicators ───────────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def macd(
    series: pd.Series,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
    signal: int = DEFAULT_MACD_SIGNAL,
):
    macd_line   = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = DEFAULT_ATR_DAYS) -> pd.Series:
    return true_range(high, low, close).rolling(window).mean()


def realized_vol(series: pd.Series, window: int = DEFAULT_VOL_DAYS) -> pd.Series:
    return series.pct_change().rolling(window).std() * math.sqrt(252)


def weekly_proxy_macd_hist(series: pd.Series) -> pd.Series:
    weekly = series.resample("W-FRI").last().dropna()
    _, _, h = macd(weekly)
    return h.reindex(series.index, method="ffill")


# ── benchmark hurdle ───────────────────────────────────────────────────────

# ── precomputed feature store ──────────────────────────────────────────────

def precompute_feature_store(
    universe: List[str],
    ohlcv: Dict[str, pd.DataFrame],
    cash_ticker: str,
    momentum_lookback_days: int,
    breakout_days: int,
    exit_days: int,
    atr_days: int,
) -> Dict[str, pd.DataFrame]:
    """
    Precompute all per-ticker indicators once.
    Returns: dict[ticker] -> DataFrame indexed by date with all feature columns.
    """
    store: Dict[str, pd.DataFrame] = {}

    for ticker in universe:
        if ticker not in ohlcv:
            continue

        df = ohlcv[ticker].copy().sort_index()
        df = df[~df.index.duplicated(keep='last')]
        if len(df) == 0 or "Close" not in df.columns or "High" not in df.columns or "Low" not in df.columns:
            continue

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        sma50     = sma(close, DEFAULT_SMA_MID)
        sma200    = sma(close, DEFAULT_SMA_LONG)
        ema10     = ema(close, DEFAULT_EMA_FAST)
        macd_l, sig_l, hist = macd(close)
        atr15     = atr(high, low, close, atr_days)
        rv20      = realized_vol(close, DEFAULT_VOL_DAYS)
        wkly_hist = weekly_proxy_macd_hist(close)

        prior_high = close.shift(1).rolling(breakout_days).max()
        prior_low  = close.shift(1).rolling(exit_days).min()
        recent_max = close.shift(1).rolling(DEFAULT_BREAKOUT_RECENT_DAYS).max()

        feat = pd.DataFrame(index=df.index)
        feat["ticker"]           = ticker
        feat["last_close"]       = close
        feat["sma50"]            = sma50
        feat["sma200"]           = sma200
        feat["ema10"]            = ema10
        feat["macd"]             = macd_l
        feat["macd_signal"]      = sig_l
        feat["macd_hist"]        = hist
        feat["weekly_macd_hist"] = wkly_hist
        feat["atr15"]            = atr15
        feat["realized_vol20"]   = rv20
        feat["ret_6m"]           = close / close.shift(momentum_lookback_days) - 1.0
        feat["dist_from_ema10"]  = close / ema10 - 1.0
        feat["trend_stack_persist_20d"] = ((close > sma200) & (sma50 > sma200)).astype(float).rolling(20).mean()
        feat["ema_persist_20d"]         = (close > ema10).astype(float).rolling(20).mean()
        feat["macd_persist_20d"]        = (hist > 0).astype(float).rolling(20).mean()
        feat["weekly_macd_persist_8w"]  = (wkly_hist > 0).astype(float).rolling(40).mean()
        feat["ret_6m_change_20d"]       = feat["ret_6m"] - feat["ret_6m"].shift(20)
        feat["macd_hist_change_10d"]    = hist - hist.shift(10)
        feat["weekly_macd_change_20d"]  = wkly_hist - wkly_hist.shift(20)
        feat["ema_persist_change_20d"]  = feat["ema_persist_20d"] - feat["ema_persist_20d"].shift(20)
        feat["breakout_89d"]     = close > prior_high
        feat["breakout_recent"]  = close > recent_max
        feat["exit_13d"]         = close < prior_low
        feat["above_ema10"]      = close > ema10
        feat["above_sma200"]     = close > sma200
        feat["sma50_gt_sma200"]  = sma50 > sma200
        feat["cash_like"]        = ticker == cash_ticker

        store[ticker] = feat

    return store


# ── metrics snapshot ───────────────────────────────────────────────────────

def build_snapshot_metrics(
    universe: List[str],
    feature_store: Dict[str, pd.DataFrame],
    asof: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a cross-sectional snapshot for `asof` by reading precomputed features.
    """
    frames = []
    required = ["last_close", "sma200", "ret_6m"]

    for ticker in universe:
        feat = feature_store.get(ticker)
        if feat is None:
            continue
        row = feat.reindex([asof])
        if row.empty:
            continue
        frames.append(row)

    if not frames:
        raise ValueError(f"No metrics at {asof.date()} from precomputed feature store.")

    out = pd.concat(frames, axis=0)
    out = out.dropna(subset=required).copy()
    if out.empty:
        raise ValueError(f"No metrics at {asof.date()} from precomputed feature store.")

    bool_cols = [
        "breakout_89d", "breakout_recent", "exit_13d",
        "above_ema10", "above_sma200", "sma50_gt_sma200", "cash_like",
    ]
    for col in bool_cols:
        out[col] = out[col].fillna(False).astype(bool)

    numeric_cols = [
        "last_close", "sma50", "sma200", "ema10", "macd", "macd_signal",
        "macd_hist", "weekly_macd_hist", "atr15", "realized_vol20", "ret_6m",
        "dist_from_ema10", "trend_stack_persist_20d", "ema_persist_20d",
        "macd_persist_20d", "weekly_macd_persist_8w", "ret_6m_change_20d",
        "macd_hist_change_10d", "weekly_macd_change_20d", "ema_persist_change_20d",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    cols = [
        "ticker", "last_close", "sma50", "sma200", "ema10", "macd",
        "macd_signal", "macd_hist", "weekly_macd_hist", "atr15",
        "realized_vol20", "ret_6m", "dist_from_ema10",
        "trend_stack_persist_20d", "ema_persist_20d",
        "macd_persist_20d", "weekly_macd_persist_8w",
        "ret_6m_change_20d", "macd_hist_change_10d",
        "weekly_macd_change_20d", "ema_persist_change_20d",
        "breakout_89d", "breakout_recent",
        "exit_13d", "above_ema10", "above_sma200", "sma50_gt_sma200", "cash_like",
    ]
    return out[cols].reset_index(drop=True)


# ── scoring & weight normalisation ─────────────────────────────────────────

def capped_normalize(
    weights: pd.Series,
    max_cap: float,
    uncapped_tickers: Optional[Set[str]] = None,
) -> pd.Series:
    """
    Cap non-uncapped tickers at max_cap without forcing the result to sum to 1.
    Any residual should be handled by the caller (typically sent to cash).
    """
    uncapped_tickers = uncapped_tickers or set()
    w = weights.copy().fillna(0.0).clip(lower=0.0)
    if w.sum() <= 0:
        return w

    w = w / w.sum()

    for _ in range(50):
        over = [t for t in w.index if t not in uncapped_tickers and w[t] > max_cap + 1e-12]
        if not over:
            break

        for t in over:
            w[t] = max_cap

        capped_mask = pd.Series(
            [t not in uncapped_tickers and w[t] >= max_cap - 1e-12 for t in w.index],
            index=w.index,
        )

        free_sum   = float(w.loc[~capped_mask].sum())
        capped_sum = float(w.loc[capped_mask].sum())
        residual   = max(0.0, 1.0 - capped_sum)

        if free_sum <= 1e-12:
            break

        w.loc[~capped_mask] = w.loc[~capped_mask] / free_sum * residual

    return w


def _add_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    non_cash = ~df["cash_like"]

    df["eligible"] = df["cash_like"] | (
        non_cash
        & df["above_sma200"].astype(bool)
    )
    return df


def _add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score = np.zeros(len(df), dtype=float)

    score += df["above_sma200"].astype(float)           * 3.0
    score += df["sma50_gt_sma200"].astype(float)        * 2.5
    score += df["above_ema10"].astype(float)            * 1.0
    score += (df["macd_hist"] > 0).astype(float)        * 2.0
    score += (df["weekly_macd_hist"] > 0).astype(float) * 1.5
    score += df["breakout_89d"].astype(float)           * 1.5
    score += df["breakout_recent"].astype(float)        * 2.0

    eligible_mask = df["eligible"] & ~df["cash_like"]

    if eligible_mask.any() and df.loc[eligible_mask, "ret_6m"].notna().any():
        mn = df.loc[eligible_mask, "ret_6m"].min()
        mx = df.loc[eligible_mask, "ret_6m"].max()
        idx = df.loc[eligible_mask].index
        score[idx] += (
            (df.loc[eligible_mask, "ret_6m"] - mn) / (mx - mn + 1e-12)
        ).to_numpy() * 5.0

    persistence_features = [
        ("trend_stack_persist_20d", 3.0),
        ("ema_persist_20d", 2.0),
        ("macd_persist_20d", 2.0),
        ("weekly_macd_persist_8w", 1.5),
    ]
    for col, weight in persistence_features:
        if eligible_mask.any() and df.loc[eligible_mask, col].notna().any():
            mn = df.loc[eligible_mask, col].min()
            mx = df.loc[eligible_mask, col].max()
            idx = df.loc[eligible_mask].index
            score[idx] += (
                (df.loc[eligible_mask, col] - mn) / (mx - mn + 1e-12)
            ).to_numpy() * weight

    if eligible_mask.any() and df.loc[eligible_mask, "realized_vol20"].notna().any():
        mn = df.loc[eligible_mask, "realized_vol20"].min()
        mx = df.loc[eligible_mask, "realized_vol20"].max()
        idx = df.loc[eligible_mask].index
        score[idx] += (
            1.0 - (df.loc[eligible_mask, "realized_vol20"] - mn) / (mx - mn + 1e-12)
        ).to_numpy() * 1.0

    score += (df["trend_stack_persist_20d"] >= 0.8).astype(float) * 1.5
    score += (df["ema_persist_20d"] >= 0.7).astype(float) * 1.0
    score += (df["macd_persist_20d"] >= 0.65).astype(float) * 1.0
    score -= (df["trend_stack_persist_20d"] < 0.4).astype(float) * 1.5
    score -= (df["macd_persist_20d"] < 0.35).astype(float) * 1.0
    score -= (df["dist_from_ema10"] > DEFAULT_WAIT_PULLBACK_NEAR_EMA).astype(float) * 1.5
    score -= df["exit_13d"].astype(float) * 4.0

    df["raw_score"] = score
    df.loc[~df["eligible"] & ~df["cash_like"], "raw_score"] = 0.0
    improvement = np.zeros(len(df), dtype=float)
    improvement += df["ret_6m_change_20d"].fillna(0.0).to_numpy() * 12.0
    improvement += df["macd_hist_change_10d"].fillna(0.0).to_numpy() * 8.0
    improvement += df["weekly_macd_change_20d"].fillna(0.0).to_numpy() * 6.0
    improvement += df["ema_persist_change_20d"].fillna(0.0).to_numpy() * 4.0
    improvement += (df["above_ema10"].astype(bool) & (df["macd_hist_change_10d"] > 0)).astype(float).to_numpy() * 1.0
    df["improvement_score"] = improvement
    return df

def build_model_weights(
    df: pd.DataFrame,
    cash_ticker: str,
    max_alloc: float,
    allocation_mode: str,
    ticker_to_sleeve: Optional[Dict[str, str]] = None,
    sleeve_targets: Optional[Dict[str, float]] = None,
    sleeve_max_allocs: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build model_target_weight globally across selected non-cash ETFs.
    """
    df = df.copy()
    df["target_weight"] = 0.0

    sel_nc = df[df["selected"] & ~df["cash_like"]].copy()
    if sel_nc.empty:
        return df

    def _base_weights(frame: pd.DataFrame) -> pd.Series:
        if allocation_mode == "equal":
            w = pd.Series(1.0, index=frame["ticker"])
        elif allocation_mode == "score_proportional":
            w = frame.set_index("ticker")["raw_score"].clip(lower=0.0)
            if float(w.sum()) <= 0:
                w = pd.Series(1.0, index=frame["ticker"])
        elif allocation_mode == "momentum_proportional":
            w = frame.set_index("ticker")["ret_6m"].clip(lower=0.0)
            if float(w.sum()) <= 0:
                w = pd.Series(1.0, index=frame["ticker"])
        else:
            raise ValueError(f"Unsupported allocation mode: {allocation_mode}")
        return w / float(w.sum())

    w = _base_weights(sel_nc)
    vol = sel_nc.set_index("ticker")["realized_vol20"].replace(0.0, np.nan)
    if vol.notna().any():
        inv_vol = (1.0 / vol.clip(lower=1e-6)) ** DEFAULT_VOL_TARGETING_EXPONENT
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol[inv_vol.notna()].median() if inv_vol.notna().any() else 1.0)
        w = (w * inv_vol).fillna(0.0)
        if float(w.sum()) > 0:
            w = w / float(w.sum())
    w = capped_normalize(w, max_cap=max_alloc)
    for t, wt in w.items():
        df.loc[df["ticker"] == t, "target_weight"] = float(wt)

    total = float(df["target_weight"].sum())
    if total > 1e-12:
        df["target_weight"] /= total

    return df

    if allocation_mode == "equal":
        w = pd.Series(1.0, index=sel_nc["ticker"])

    elif allocation_mode == "score_proportional":
        w = sel_nc.set_index("ticker")["raw_score"].clip(lower=0.0)
        if float(w.sum()) <= 0:
            w = pd.Series(1.0, index=sel_nc["ticker"])

    elif allocation_mode == "momentum_proportional":
        # Use only positive momentum as sizing input
        w = sel_nc.set_index("ticker")["ret_6m"].clip(lower=0.0)
        if float(w.sum()) <= 0:
            w = pd.Series(1.0, index=sel_nc["ticker"])

    else:
        raise ValueError(f"Unsupported allocation mode: {allocation_mode}")

    # Normalize to 1 first, then cap without re-expanding above caps
    w = w / float(w.sum())
    w = capped_normalize(w, max_cap=max_alloc)

    for t, wt in w.items():
        df.loc[df["ticker"] == t, "target_weight"] = float(wt)

    residual  = 1.0 - float(df["target_weight"].sum())
    cash_mask = df["ticker"] == cash_ticker
    if cash_mask.any():
        df.loc[cash_mask, "target_weight"] += residual

    total = float(df["target_weight"].sum())
    if total > 1e-12:
        df["target_weight"] /= total

    return df

def _assign_weights(
    df: pd.DataFrame,
    cash_ticker: str,
    max_alloc: float,
    score_col: str = "raw_score",
) -> pd.DataFrame:
    df = df.copy()
    df["target_weight"] = 0.0
    sel_nc = df[df["selected"] & ~df["cash_like"]]

    if sel_nc.empty:
        df.loc[df["ticker"] == cash_ticker, "target_weight"] = 1.0
        return df

    w = sel_nc.set_index("ticker")[score_col].clip(lower=0.0)
    if w.sum() <= 0:
        df.loc[df["ticker"] == cash_ticker, "target_weight"] = 1.0
        return df

    w = capped_normalize(w, max_cap=max_alloc)
    for t, wt in w.items():
        df.loc[df["ticker"] == t, "target_weight"] = float(wt)

    residual  = 1.0 - float(df["target_weight"].sum())
    cash_mask = df["ticker"] == cash_ticker
    if cash_mask.any():
        df.loc[cash_mask, "target_weight"] += residual

    total = float(df["target_weight"].sum())
    if total > 1e-12:
        df["target_weight"] /= total

    return df


# ── entry signal classification ────────────────────────────────────────────

def classify_entry_signal(row: pd.Series) -> str:
    if bool(row.get("cash_like", False)):
        return "BUY NOW"
    if not bool(row.get("above_sma200", False)) or bool(row.get("exit_13d", False)):
        return "DO NOT BUY"

    last_close = row.get("last_close", np.nan)
    ema10v     = row.get("ema10", np.nan)
    if pd.isna(last_close) or pd.isna(ema10v) or ema10v <= 0:
        return "WAIT FOR PULLBACK"

    dist         = float(last_close) / float(ema10v) - 1.0
    strong_trend = bool(row.get("above_sma200")) and bool(row.get("sma50_gt_sma200"))
    pos_mom      = (
        pd.notna(row.get("macd_hist")) and row.get("macd_hist") > 0
        and pd.notna(row.get("weekly_macd_hist")) and row.get("weekly_macd_hist") > 0
    )

    if strong_trend and pos_mom and bool(row.get("breakout_89d")) and bool(row.get("breakout_recent")) and dist <= DEFAULT_EXTENDED_FROM_EMA:
        return "BUY NOW"
    if strong_trend and pos_mom and bool(row.get("above_ema10")) and dist <= DEFAULT_PULLBACK_EMA_BUFFER:
        return "BUY NOW"
    if strong_trend and pos_mom and dist > DEFAULT_EXTENDED_FROM_EMA:
        return "WAIT FOR PULLBACK"
    if strong_trend:
        return "WAIT FOR PULLBACK"
    return "DO NOT BUY"


def apply_entry_labels_and_allocate(
    df: pd.DataFrame,
    cash_ticker: str,
    max_alloc: float,
    execution_mode: str,
    max_wait_pullback: int,
    min_executable_for_cash_cap: int = DEFAULT_MIN_EXECUTABLE_FOR_CASH_CAP,
    max_cash_when_enough: float = DEFAULT_MAX_CASH_WHEN_ENOUGH,
) -> pd.DataFrame:
    """
    execution_mode:
    - overlay: BUY NOW gets full model weight; WAIT FOR PULLBACK gets partial weight
      that scales up as trend breadth strengthens; residual stays in cash
    - pure_topk: ignore entry labels and use model_target_weight directly
    """
    df = df.copy()
    df["entry_label"] = "DO NOT BUY"

    sel_nc = df["selected"] & ~df["cash_like"]
    if sel_nc.any():
        selected = df.loc[sel_nc].copy()

        strong_trend = selected["above_sma200"].astype(bool) & selected["sma50_gt_sma200"].astype(bool)
        pos_mom = (selected["macd_hist"] > 0) & (selected["weekly_macd_hist"] > 0)
        valid_ema = selected["ema10"].notna() & (selected["ema10"] > 0)
        dist = pd.Series(np.inf, index=selected.index, dtype=float)
        dist.loc[valid_ema] = selected.loc[valid_ema, "last_close"] / selected.loc[valid_ema, "ema10"] - 1.0

        labels = pd.Series("DO NOT BUY", index=selected.index, dtype=object)
        invalid = ~selected["above_sma200"].astype(bool) | selected["exit_13d"].astype(bool)

        breakout_buy = (
            ~invalid
            & valid_ema
            & strong_trend
            & pos_mom
            & selected["breakout_89d"].astype(bool)
            & selected["breakout_recent"].astype(bool)
            & (dist <= DEFAULT_EXTENDED_FROM_EMA)
        )
        labels.loc[breakout_buy] = "BUY NOW"

        pullback_buy = (
            ~invalid
            & valid_ema
            & strong_trend
            & pos_mom
            & selected["above_ema10"].astype(bool)
            & (dist <= DEFAULT_PULLBACK_EMA_BUFFER)
        )
        labels.loc[pullback_buy] = "BUY NOW"

        labels.loc[~invalid & strong_trend & (labels != "BUY NOW")] = "WAIT FOR PULLBACK"
        recovering = (
            (labels == "DO NOT BUY")
            & ~invalid
            & valid_ema
            & (dist <= DEFAULT_WAIT_PULLBACK_NEAR_EMA)
            & (selected["improvement_score"] >= DEFAULT_RECOVERING_MIN_IMPROVEMENT)
            & (
                (selected["macd_hist_change_10d"] > 0)
                | (selected["weekly_macd_change_20d"] > 0)
                | (selected["ema_persist_change_20d"] > 0)
            )
        )
        labels.loc[recovering] = "RECOVERING"
        df.loc[selected.index, "entry_label"] = labels

    if execution_mode == "pure_topk":
        df["target_weight"] = df["model_target_weight"]
        total = float(df["target_weight"].sum())
        if total > 1e-12:
            df["target_weight"] /= total
        return df

    if execution_mode != "overlay":
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    df["target_weight"] = 0.0

    universe_non_cash = df[~df["cash_like"]].copy()
    if not universe_non_cash.empty:
        trend_breadth = float(
            (
                universe_non_cash["above_sma200"].astype(bool)
                & universe_non_cash["sma50_gt_sma200"].astype(bool)
            ).mean()
        )
        ema_breadth = float(universe_non_cash["above_ema10"].astype(bool).mean())
        momentum_breadth = float(
            (
                (universe_non_cash["macd_hist"] > 0)
                & (universe_non_cash["weekly_macd_hist"] > 0)
            ).mean()
        )
        strong_trend_breadth = float(np.mean([trend_breadth, ema_breadth, momentum_breadth]))
    else:
        strong_trend_breadth = 0.0
    breadth_strength = min(1.0, strong_trend_breadth / DEFAULT_BREADTH_FULLY_RISK_ON)
    weak_participation = min(ema_breadth, momentum_breadth) if not universe_non_cash.empty else 0.0
    defense_active = (
        strong_trend_breadth < DEFAULT_DEFENSE_BREADTH_THRESHOLD
        and weak_participation < DEFAULT_DEFENSE_PARTICIPATION_THRESHOLD
    )
    wait_multiplier = (
        DEFAULT_WAIT_PULLBACK_MIN_MULTIPLIER
        + (DEFAULT_WAIT_PULLBACK_MAX_MULTIPLIER - DEFAULT_WAIT_PULLBACK_MIN_MULTIPLIER) * breadth_strength
    )
    if defense_active:
        wait_multiplier *= DEFAULT_DEFENSE_WAIT_MULTIPLIER

    buy_now = df[df["selected"] & ~df["cash_like"] & (df["entry_label"] == "BUY NOW")]
    wait_pb = df[df["selected"] & ~df["cash_like"] & (df["entry_label"] == "WAIT FOR PULLBACK")].copy()
    recovering = df[df["selected"] & ~df["cash_like"] & (df["entry_label"] == "RECOVERING")].copy()
    if not wait_pb.empty:
        wait_pb = wait_pb.sort_values(
            ["model_target_weight", "raw_score", "ticker"],
            ascending=[False, False, True],
        )
        if max_wait_pullback >= 0:
            wait_limit = max_wait_pullback + int(round(breadth_strength * DEFAULT_WAIT_PULLBACK_BREADTH_BONUS_SLOTS))
            wait_pb = wait_pb.head(wait_limit)
    if not recovering.empty:
        recovering = recovering.sort_values(
            ["improvement_score", "raw_score", "ticker"],
            ascending=[False, False, True],
        ).head(DEFAULT_RECOVERING_MAX_NAMES)
    if buy_now.empty and wait_pb.empty and recovering.empty:
        fallback = df[df["selected"] & ~df["cash_like"]].copy()
        if fallback.empty:
            fallback = df[~df["cash_like"] & (df["raw_score"] > 0)].copy()
        if fallback.empty:
            fallback = df[~df["cash_like"]].copy()
        fallback = fallback.sort_values(["raw_score", "ticker"], ascending=[False, True]).head(max(1, max_wait_pullback))
        fw = fallback.set_index("ticker")["model_target_weight"].clip(lower=0.0)
        if float(fw.sum()) <= 0:
            fw = fallback.set_index("ticker")["raw_score"].clip(lower=0.0)
        if float(fw.sum()) <= 0:
            fw = pd.Series(1.0, index=fallback["ticker"])
        fw = capped_normalize(fw, max_cap=max_alloc)
        for t, wt in fw.items():
            df.loc[df["ticker"] == t, "target_weight"] = float(wt)
    else:
        component_weights = []

        if not buy_now.empty:
            bw = buy_now.set_index("ticker")["model_target_weight"].clip(lower=0.0)
            if float(bw.sum()) <= 0:
                bw = buy_now.set_index("ticker")["raw_score"].clip(lower=0.0)
            if float(bw.sum()) <= 0:
                bw = pd.Series(1.0, index=buy_now["ticker"])
            component_weights.append(bw)

        if not wait_pb.empty:
            wait_dist = pd.Series(np.inf, index=wait_pb.index, dtype=float)
            valid_wait_ema = wait_pb["ema10"].notna() & (wait_pb["ema10"] > 0)
            wait_dist.loc[valid_wait_ema] = (
                wait_pb.loc[valid_wait_ema, "last_close"] / wait_pb.loc[valid_wait_ema, "ema10"] - 1.0
            )
            proximity_scale = pd.Series(DEFAULT_WAIT_PULLBACK_FAR_MULTIPLIER, index=wait_pb.index, dtype=float)
            proximity_scale.loc[wait_dist <= DEFAULT_EXTENDED_FROM_EMA] = 1.0
            proximity_scale.loc[
                (wait_dist > DEFAULT_EXTENDED_FROM_EMA) & (wait_dist <= DEFAULT_WAIT_PULLBACK_NEAR_EMA)
            ] = DEFAULT_WAIT_PULLBACK_MID_MULTIPLIER
            proximity_scale.loc[wait_pb["above_ema10"].astype(bool) & (wait_dist <= DEFAULT_PULLBACK_EMA_BUFFER)] = 1.0

            ww = (
                wait_pb.set_index("ticker")["model_target_weight"].clip(lower=0.0)
                * wait_multiplier
                * proximity_scale.to_numpy()
            )
            if float(ww.sum()) <= 0:
                ww = pd.Series(1.0, index=wait_pb["ticker"])
            if defense_active:
                ww *= DEFAULT_DEFENSE_WAIT_MULTIPLIER
            component_weights.append(ww)

        if not recovering.empty:
            rw = recovering.set_index("ticker")["improvement_score"].clip(lower=0.0)
            if float(rw.sum()) <= 0:
                rw = recovering.set_index("ticker")["raw_score"].clip(lower=0.0)
            if float(rw.sum()) <= 0:
                rw = pd.Series(1.0, index=recovering["ticker"])
            component_weights.append(rw * DEFAULT_RECOVERING_TOTAL_BUDGET)

        combined = pd.concat(component_weights) if component_weights else pd.Series(dtype=float)
        combined = combined.groupby(level=0).sum()
        if float(combined.sum()) <= 0:
            combined = pd.Series(1.0, index=combined.index if len(combined) else buy_now["ticker"])
        combined = capped_normalize(combined, max_cap=max_alloc)
        if buy_now.empty and not wait_pb.empty:
            positive_improvement = wait_pb["improvement_score"].clip(lower=0.0)
            improvement_scale = min(
                1.0,
                float(positive_improvement.mean()) / max(DEFAULT_RECOVERING_MIN_IMPROVEMENT, 1e-6)
            ) if not positive_improvement.empty else 0.0
            wait_only_active_multiplier = (
                DEFAULT_WAIT_ONLY_ACTIVE_MULTIPLIER
                + (DEFAULT_WAIT_ONLY_MAX_ACTIVE_MULTIPLIER - DEFAULT_WAIT_ONLY_ACTIVE_MULTIPLIER) * improvement_scale
            )
            combined *= wait_only_active_multiplier
        for t, wt in combined.items():
            df.loc[df["ticker"] == t, "target_weight"] = float(wt)

    residual = max(0.0, 1.0 - float(df["target_weight"].sum()))
    cash_mask = df["ticker"] == cash_ticker
    if cash_mask.any() and residual > 0:
        df.loc[cash_mask, "target_weight"] += residual

    total = float(df["target_weight"].sum())
    if total > 1e-12:
        df["target_weight"] /= total

    return df


def rationale_for_row(row: pd.Series, cash_ticker: str) -> str:
    def yes_no(flag: object) -> str:
        return "yes" if bool(flag) else "no"

    def pct(value: object) -> str:
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "n/a"

    trend_ok = bool(row.get("above_sma200")) and bool(row.get("sma50_gt_sma200"))
    momentum_ok = float(row.get("macd_hist", 0.0)) > 0 and float(row.get("weekly_macd_hist", 0.0)) > 0
    breakout_ok = bool(row.get("breakout_89d")) and bool(row.get("breakout_recent"))
    above_ema10 = bool(row.get("above_ema10"))
    exit_flag = bool(row.get("exit_13d"))

    if row["ticker"] == cash_ticker:
        return (
            f"{cash_ticker} absorbs residual cash; final target {pct(row.get('target_weight', 0.0))} "
            f"because selected ETFs are only partially deployable today."
            if row.get("target_weight", 0) > 0
            else f"{cash_ticker} is the cash sleeve but no residual allocation is needed today."
        )
    if row.get("selected") and row.get("entry_label") == "BUY NOW":
        return (
            f"Selected and executable now: trend={yes_no(trend_ok)}, momentum={yes_no(momentum_ok)}, "
            f"breakout_fresh={yes_no(breakout_ok)}, above_ema10={yes_no(above_ema10)}."
        )
    if row.get("selected") and row.get("entry_label") == "WAIT FOR PULLBACK":
        return (
            f"Selected but partial-only entry: trend={yes_no(trend_ok)}, momentum={yes_no(momentum_ok)}, "
            f"breakout_fresh={yes_no(breakout_ok)}, above_ema10={yes_no(above_ema10)}. "
            f"Model target {pct(row.get('model_target_weight', 0.0))}, deployed target {pct(row.get('target_weight', 0.0))}."
        )
    if row.get("selected") and row.get("entry_label") == "DO NOT BUY":
        reasons = []
        if exit_flag:
            reasons.append("exit_13d triggered")
        if not bool(row.get("above_sma200")):
            reasons.append("below SMA200")
        if not bool(row.get("sma50_gt_sma200")):
            reasons.append("SMA50 <= SMA200")
        if float(row.get("macd_hist", 0.0)) <= 0:
            reasons.append("daily MACD histogram <= 0")
        if float(row.get("weekly_macd_hist", 0.0)) <= 0:
            reasons.append("weekly MACD histogram <= 0")
        if not above_ema10:
            reasons.append("below EMA10")
        detail = ", ".join(reasons) if reasons else "entry conditions not met"
        return f"Selected on ranking, but not executable today: {detail}."

    reasons = []
    if not bool(row.get("above_sma200")):
        reasons.append("below SMA200")
    if not bool(row.get("sma50_gt_sma200")):
        reasons.append("SMA50 <= SMA200")
    if float(row.get("raw_score", 0.0)) <= 0:
        reasons.append("raw score <= 0")
    if exit_flag:
        reasons.append("exit_13d triggered")
    detail = ", ".join(reasons) if reasons else "outranked by stronger candidates"
    return f"Not selected: {detail}."


# ── candidate selection ────────────────────────────────────────────────────


def select_candidates(
    metrics: pd.DataFrame,
    cash_ticker: str,
    max_alloc: float,
    allocation_mode: str,
    execution_mode: str,
    max_wait_pullback: int,
    min_executable_for_cash_cap: int = DEFAULT_MIN_EXECUTABLE_FOR_CASH_CAP,
    max_cash_when_enough: float = DEFAULT_MAX_CASH_WHEN_ENOUGH,
    ticker_to_sleeve: Optional[Dict[str, str]] = None,
    sleeve_targets: Optional[Dict[str, float]] = None,
    sleeve_max_allocs: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    df = _add_eligibility(metrics.copy())
    df = _add_scores(df)

    if ticker_to_sleeve and sleeve_targets:
        df["sleeve"] = df["ticker"].map(lambda t: ticker_to_sleeve.get(t, "general"))
        active_sleeves = {k for k, v in sleeve_targets.items() if float(v) > 0}
        df["selected"] = (
            (~df["cash_like"])
            & df["eligible"]
            & (df["raw_score"] > 0)
            & df["sleeve"].isin(active_sleeves)
        ) | df["cash_like"]
    else:
        df["selected"] = ((~df["cash_like"]) & df["eligible"] & (df["raw_score"] > 0)) | df["cash_like"]

    df = build_model_weights(
        df, cash_ticker, max_alloc, allocation_mode,
        ticker_to_sleeve, sleeve_targets, sleeve_max_allocs
    )
    df["model_target_weight"] = df["target_weight"].copy()

    return apply_entry_labels_and_allocate(
        df,
        cash_ticker=cash_ticker,
        max_alloc=max_alloc,
        execution_mode=execution_mode,
        max_wait_pullback=max_wait_pullback,
        min_executable_for_cash_cap=min_executable_for_cash_cap,
        max_cash_when_enough=max_cash_when_enough,
    )



def recheck_entry_signals(
    metrics: pd.DataFrame,
    cash_ticker: str,
    max_alloc: float,
    allocation_mode: str,
    execution_mode: str,
    max_wait_pullback: int,
    current_weights: Optional[pd.Series] = None,
    min_executable_for_cash_cap: int = DEFAULT_MIN_EXECUTABLE_FOR_CASH_CAP,
    max_cash_when_enough: float = DEFAULT_MAX_CASH_WHEN_ENOUGH,
    ticker_to_sleeve: Optional[Dict[str, str]] = None,
    sleeve_targets: Optional[Dict[str, float]] = None,
    sleeve_max_allocs: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    df = _add_eligibility(metrics.copy())
    df = _add_scores(df)

    held_non_cash: Set[str] = set()
    if current_weights is not None and len(current_weights):
        held_non_cash = {
            str(t)
            for t, wt in current_weights.items()
            if str(t) != cash_ticker and float(wt) > 1e-12
        }

    # Entry checks should only revisit currently held names, not reopen the
    # entire candidate set and effectively turn lower-frequency schedules into
    # weekly rebalances.
    df["selected"] = df["ticker"].isin(held_non_cash) | df["cash_like"]

    df = build_model_weights(
        df, cash_ticker, max_alloc, allocation_mode,
        ticker_to_sleeve, sleeve_targets, sleeve_max_allocs
    )
    df["model_target_weight"] = df["target_weight"].copy()

    return apply_entry_labels_and_allocate(
        df,
        cash_ticker=cash_ticker,
        max_alloc=max_alloc,
        execution_mode=execution_mode,
        max_wait_pullback=max_wait_pullback,
        min_executable_for_cash_cap=min_executable_for_cash_cap,
        max_cash_when_enough=max_cash_when_enough,
    )


# ── current holdings reconciliation ───────────────────────────────────────

def derive_current_weights(holdings: pd.DataFrame, close_px: pd.DataFrame) -> pd.DataFrame:
    h = holdings.copy()

    if "current_weight" in h.columns:
        total = float(h["current_weight"].sum())
        if total > 0:
            h["current_weight"] /= total
            return h[["ticker", "current_weight"]]

    if "shares" in h.columns:
        prices = []
        for _, row in h.iterrows():
            t = row["ticker"]
            p = float(close_px[t].dropna().iloc[-1]) if t in close_px.columns and len(close_px[t].dropna()) else np.nan
            prices.append(p)

        h["last_price"]   = prices
        h["market_value"] = h["shares"] * h["last_price"]

        missing_price = h["last_price"].isna()
        if missing_price.any():
            missing = h.loc[missing_price, "ticker"].tolist()
            print_progress(f"Warning: no recent price for held tickers: {missing}")

        total = float(h["market_value"].sum())
        h["current_weight"] = h["market_value"] / total if total > 0 else 0.0
        return h[["ticker", "current_weight"]]

    h["current_weight"] = 0.0
    return h[["ticker", "current_weight"]]


# ── performance analytics ──────────────────────────────────────────────────

def metrics_from_equity_curve(equity: pd.Series, turnover: pd.Series) -> Dict[str, float]:
    daily_ret = equity.pct_change().fillna(0.0)
    n_years   = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr      = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0)
    vol       = float(daily_ret.std() * math.sqrt(252))
    sharpe    = float((daily_ret.mean() * 252) / vol) if vol > 0 else np.nan
    dd        = equity / equity.cummax() - 1.0
    max_dd    = float(dd.min())
    calmar    = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

    turnover = turnover.dropna() if len(turnover) else turnover

    return {
        "Total Return":             float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "CAGR":                     cagr,
        "Annual Vol":               vol,
        "Sharpe":                   sharpe,
        "Max Drawdown":             max_dd,
        "Calmar":                   calmar,
        "Avg Turnover/Trade Date":  float(turnover.mean()) if len(turnover) else 0.0,
        "Num Trade Dates":          int((turnover > 0).sum()) if len(turnover) else 0,
    }


def benchmark_series(close_px: pd.DataFrame, start: str, end: str, ticker: str) -> pd.Series:
    s = close_px.loc[start:end, ticker].dropna()
    eq = s / s.iloc[0]
    eq.name = ticker
    return eq


# ── backtest engine ────────────────────────────────────────────────────────

def run_schedule_backtest(
    universe: List[str],
    ohlcv: Dict[str, pd.DataFrame],
    close_px: pd.DataFrame,
    feature_store: Dict[str, pd.DataFrame],
    schedule_code: str,
    cash_ticker: str,
    start: str,
    end: str,
    max_alloc: float,
    transaction_cost_bps: int,
    allocation_mode: str,
    execution_mode: str,
    max_wait_pullback: int,
    min_executable_for_cash_cap: int,
    max_cash_when_enough: float,
    ticker_to_sleeve: Optional[Dict[str, str]] = None,
    sleeve_targets: Optional[Dict[str, float]] = None,
    sleeve_max_allocs: Optional[Dict[str, float]] = None,
):
    """
    Drift-aware backtest with explicit MOC execution.
    """
    px = close_px.loc[start:end, universe].sort_index().ffill().dropna(how="all")
    if px.empty:
        raise ValueError(f"No price data in {start} → {end}")

    rebalance_dates   = {d for d in px.resample(REBALANCE_MAP[schedule_code]).last().index if d in px.index}
    entry_check_dates = {d for d in px.resample(REBALANCE_MAP["W"]).last().index if d in px.index}
    if not rebalance_dates:
        return None, None, None, None

    schedule_name = {"W": "Weekly", "BW": "Biweekly", "M": "Monthly", "Q": "Quarterly"}.get(schedule_code, schedule_code)
    print_progress(
        f"{schedule_name}: {len(rebalance_dates)} rebalances, "
        f"{len(entry_check_dates)} weekly entry checks"
    )

    weights = pd.Series(0.0, index=universe, dtype=float)
    equity_vals:     list = []
    turnover_rows:   list = []
    weights_history: list = []
    entry_lbl_hist:  list = []

    equity = 1.0

    for i, dt in enumerate(px.index):
        if i == 0:
            metrics = build_snapshot_metrics(universe, feature_store, dt)
            alloc = select_candidates(
                metrics, cash_ticker, max_alloc,
                allocation_mode, execution_mode, max_wait_pullback,
                min_executable_for_cash_cap, max_cash_when_enough,
                ticker_to_sleeve=ticker_to_sleeve,
                sleeve_targets=sleeve_targets, sleeve_max_allocs=sleeve_max_allocs,
            )
            init_w = alloc.set_index("ticker")["target_weight"].reindex(universe).fillna(0.0)
            init_total = float(init_w.sum())
            if init_total > 1e-12:
                weights = init_w / init_total
            equity_vals.append(equity)
            weights_history.append({"date": dt, **weights.to_dict()})
            continue

        prev_dt = px.index[i - 1]
        daily_ret_vec = (px.loc[dt] / px.loc[prev_dt] - 1.0).fillna(0.0)

        port_ret = float((weights * daily_ret_vec).sum())
        equity *= (1.0 + port_ret)

        gross = weights * (1.0 + daily_ret_vec)
        gross_sum = float(gross.sum())
        if gross_sum > 1e-12:
            drifted_weights = gross / gross_sum
        else:
            drifted_weights = weights.copy()

        need_rebalance   = dt in rebalance_dates
        need_entry_check = dt in entry_check_dates

        new_weights = drifted_weights.copy()

        if need_rebalance or need_entry_check:
            metrics = build_snapshot_metrics(universe, feature_store, dt)

            if need_rebalance:
                alloc = select_candidates(
                    metrics, cash_ticker, max_alloc,
                    allocation_mode, execution_mode, max_wait_pullback,
                    min_executable_for_cash_cap, max_cash_when_enough,
                    ticker_to_sleeve=ticker_to_sleeve,
                    sleeve_targets=sleeve_targets, sleeve_max_allocs=sleeve_max_allocs,
                )
            else:
                alloc = recheck_entry_signals(
                    metrics, cash_ticker, max_alloc,
                    allocation_mode, execution_mode, max_wait_pullback,
                    current_weights=drifted_weights,
                    min_executable_for_cash_cap=min_executable_for_cash_cap,
                    max_cash_when_enough=max_cash_when_enough,
                    ticker_to_sleeve=ticker_to_sleeve,
                    sleeve_targets=sleeve_targets, sleeve_max_allocs=sleeve_max_allocs,
                )

            target_w_map = alloc.set_index("ticker")["target_weight"]
            target_w = target_w_map.reindex(universe).fillna(0.0)
            target_w = (
                drifted_weights * (1.0 - DEFAULT_REBALANCE_INERTIA)
                + target_w * DEFAULT_REBALANCE_INERTIA
            )
            target_total = float(target_w.sum())
            if target_total > 1e-12:
                target_w = target_w / target_total
            turnover = float((target_w - drifted_weights).abs().sum() / 2.0)

            if transaction_cost_bps > 0 and turnover > 0:
                cost_frac = turnover * 2.0 * transaction_cost_bps / 10_000.0
                equity *= max(0.0, 1.0 - cost_frac)

            new_weights = target_w.copy()

            turnover_rows.append({
                "date": dt,
                "turnover": turnover,
                "event": "rebalance" if need_rebalance else "entry_check",
            })

            weights_history.append({"date": dt, **new_weights.to_dict()})

            sel_nc = alloc[alloc["selected"] & ~alloc["cash_like"]]
            ec     = sel_nc["entry_label"].value_counts()
            entry_lbl_hist.append({
                "date":                    dt,
                "selected_non_cash_count": int(len(sel_nc)),
                "buy_now_count":           int(ec.get("BUY NOW", 0)),
                "wait_for_pullback_count": int(ec.get("WAIT FOR PULLBACK", 0)),
                "do_not_buy_count":        int(ec.get("DO NOT BUY", 0)),
                "sgov_weight":             float(alloc.loc[alloc["ticker"] == cash_ticker, "target_weight"].sum()),
            })

        weights = new_weights
        equity_vals.append(equity)

    equity_series = pd.Series(equity_vals, index=px.index)
    return (
        equity_series,
        pd.DataFrame(turnover_rows),
        pd.DataFrame(weights_history),
        pd.DataFrame(entry_lbl_hist),
    )


# ── allocation mode ────────────────────────────────────────────────────────

def allocation_mode(
    universe_path: Optional[str],
    holdings_path: str,
    cash_ticker: str,
    export_prefix: str,
    start: Optional[str],
    end: Optional[str],
    max_alloc: float,
    allocation_mode: str,
    execution_mode: str,
    max_wait_pullback: int,
    min_executable_for_cash_cap: int,
    max_cash_when_enough: float,
    price_cache_dir: Optional[str],
    refresh_cache: bool,
    universe_3x_path: Optional[str] = None,
    universe_2x_path: Optional[str] = None,
    universe_broad_path: Optional[str] = None,
    sleeve_targets: Optional[Dict[str, float]] = None,
    sleeve_max_allocs: Optional[Dict[str, float]] = None,
):
    if not 0 < max_alloc <= 1:
        raise ValueError("max_alloc must be in the interval (0, 1].")
    total_steps = 6
    print_progress("=" * 100)
    print_progress("ALLOCATION RUN")
    print_progress("=" * 100)
    print_step(1, total_steps, "Loading universe and current holdings")
    universe, ticker_to_sleeve = load_configurable_universes(universe_path, universe_3x_path, universe_2x_path, universe_broad_path)
    holdings = read_holdings(holdings_path)

    bad = sorted(set(holdings["ticker"]) - set(universe))
    if bad:
        raise ValueError(f"Holdings contain tickers not in universe file: {bad}")
    if cash_ticker not in universe:
        universe.append(cash_ticker)
        ticker_to_sleeve.setdefault(cash_ticker, "cash")
    effective_start = start or (datetime.today() - timedelta(days=550)).strftime("%Y-%m-%d")
    if any([universe_3x_path, universe_2x_path, universe_broad_path]):
        sleeve_targets = sleeve_targets or {"broad": 0.70, "3x": 0.20, "2x": 0.10}
        sleeve_max_allocs = sleeve_max_allocs or {"broad": 0.20, "3x": 0.10, "2x": 0.08}
    else:
        sleeve_targets = None
        sleeve_max_allocs = None
    tickers = sorted(set(universe) | COMPARISON_BENCHMARK_TICKERS)
    print_progress(
        f"Universe loaded: {len(universe)} ETFs  |  Holdings rows: {len(holdings)}  |  "
        f"Price window: {effective_start} -> {end or 'latest'}"
    )

    print_step(2, total_steps, "Downloading or loading price history")
    ohlcv = download_ohlcv_history(
        tickers,
        start=effective_start,
        end=end,
        cache_dir=price_cache_dir,
        refresh_cache=refresh_cache,
    )
    close_px = get_close_series(ohlcv)
    asof     = close_px.dropna(how="all").index[-1]
    print_progress(f"Latest usable snapshot date: {asof.date()}")

    print_step(3, total_steps, "Building feature store")
    feature_store = precompute_feature_store(
        universe, ohlcv, cash_ticker,
        DEFAULT_MOMENTUM_LOOKBACK_DAYS, DEFAULT_BREAKOUT_DAYS,
        DEFAULT_EXIT_DAYS, DEFAULT_ATR_DAYS,
    )

    print_step(4, total_steps, "Scoring ETFs and selecting candidates")
    metrics = build_snapshot_metrics(universe, feature_store, asof)
    print_progress(f"ETFs with sufficient history: {len(metrics)}")

    current = derive_current_weights(holdings, close_px)

    alloc = select_candidates(
        metrics, cash_ticker, max_alloc,
        allocation_mode, execution_mode, max_wait_pullback,
        min_executable_for_cash_cap, max_cash_when_enough,
        ticker_to_sleeve=ticker_to_sleeve,
        sleeve_targets=sleeve_targets, sleeve_max_allocs=sleeve_max_allocs,
    )
    sel_nc = alloc[alloc["selected"] & ~alloc["cash_like"]].copy()
    entry_counts = sel_nc["entry_label"].value_counts()
    cash_target = float(alloc.loc[alloc["ticker"] == cash_ticker, "target_weight"].sum())
    print_progress(
        f"Selected non-cash ETFs: {len(sel_nc)}  |  "
        f"BUY NOW: {int(entry_counts.get('BUY NOW', 0))}  |  "
        f"WAIT FOR PULLBACK: {int(entry_counts.get('WAIT FOR PULLBACK', 0))}  |  "
        f"DO NOT BUY: {int(entry_counts.get('DO NOT BUY', 0))}  |  "
        f"{cash_ticker}: {cash_target:.1%}"
    )

    print_step(5, total_steps, "Comparing target allocation vs current holdings")
    out = alloc.merge(current, on="ticker", how="left")
    if "sleeve" not in out.columns:
        out["sleeve"] = out["ticker"].map(lambda t: ticker_to_sleeve.get(t, "general"))
    out["current_weight"]         = out["current_weight"].fillna(0.0)
    out["delta_weight"]           = out["target_weight"] - out["current_weight"]
    out["current_alloc_pct"]      = out["current_weight"] * 100.0
    out["target_alloc_pct"]       = out["target_weight"]  * 100.0
    out["model_target_alloc_pct"] = out["model_target_weight"] * 100.0
    out["delta_pct_points"]       = out["delta_weight"] * 100.0
    out["action"]    = np.where(out["delta_weight"] >  1e-6, "INCREASE",
                       np.where(out["delta_weight"] < -1e-6, "DECREASE", "HOLD"))
    out["rationale"] = out.apply(lambda r: rationale_for_row(r, cash_ticker), axis=1)

    view = out.sort_values(["target_weight", "raw_score", "ticker"], ascending=[False, False, True])

    metrics_cols = [
        "ticker","sleeve","last_close","ret_6m","realized_vol20","atr15",
        "above_ema10","above_sma200","sma50_gt_sma200",
        "macd","macd_signal","macd_hist","weekly_macd_hist",
        "breakout_89d","breakout_recent","exit_13d",
        "raw_score","selected","entry_label",
        "model_target_alloc_pct","current_alloc_pct","target_alloc_pct",
        "delta_pct_points","action","rationale",
    ]
    action_cols = [
        "ticker","sleeve","entry_label","current_alloc_pct",
        "target_alloc_pct","delta_pct_points","action",
    ]

    print_step(6, total_steps, "Writing output files")
    view[metrics_cols].to_csv(f"{export_prefix}_full_metrics.csv", index=False)

    actions_view = view[action_cols].copy()
    actions_view = actions_view[
        ~((actions_view["current_alloc_pct"].abs() < 1e-9) &
          (actions_view["target_alloc_pct"].abs() < 1e-9))
    ]
    actions_view.to_csv(f"{export_prefix}_rebalance_actions.csv", index=False)

    print()
    print("=" * 100)
    print("ALLOCATION SUMMARY")
    print("=" * 100)
    print(f"As of: {asof.date()}")
    print(
        f"Allocation mode: {allocation_mode}  |  Execution mode: {execution_mode}"
    )
    if sleeve_targets:
        print(f"Sleeve targets: {sleeve_targets}")
        print(f"Sleeve max allocs: {sleeve_max_allocs}")
    print(f"Max alloc: {max_alloc:.0%}  |  Max WAIT FOR PULLBACK: {max_wait_pullback}  |  Cash ticker: {cash_ticker}")
    print()
    print("Recommended rebalance actions")
    print(actions_view.to_string(index=False))
    print()
    print(f"Wrote: {export_prefix}_full_metrics.csv")
    print(f"Wrote: {export_prefix}_rebalance_actions.csv")


# ── backtest mode ──────────────────────────────────────────────────────────

def backtest_mode(
    universe_path: Optional[str],
    cash_ticker: str,
    export_prefix: str,
    start: str,
    end: str,
    max_alloc: float,
    transaction_cost_bps: int,
    allocation_mode: str,
    execution_mode: str,
    max_wait_pullback: int,
    min_executable_for_cash_cap: int,
    max_cash_when_enough: float,
    price_cache_dir: Optional[str],
    refresh_cache: bool,
    universe_3x_path: Optional[str] = None,
    universe_2x_path: Optional[str] = None,
    universe_broad_path: Optional[str] = None,
    sleeve_targets: Optional[Dict[str, float]] = None,
    sleeve_max_allocs: Optional[Dict[str, float]] = None,
):
    if not 0 < max_alloc <= 1:
        raise ValueError("max_alloc must be in the interval (0, 1].")
    print_progress("Loading ETF universe")
    universe, ticker_to_sleeve = load_configurable_universes(universe_path, universe_3x_path, universe_2x_path, universe_broad_path)

    if cash_ticker not in universe:
        universe.append(cash_ticker)
        ticker_to_sleeve.setdefault(cash_ticker, "cash")

    if any([universe_3x_path, universe_2x_path, universe_broad_path]):
        sleeve_targets = sleeve_targets or {"broad": 0.70, "3x": 0.20, "2x": 0.10}
        sleeve_max_allocs = sleeve_max_allocs or {"broad": 0.20, "3x": 0.10, "2x": 0.08}
    else:
        sleeve_targets = None
        sleeve_max_allocs = None

    tickers        = sorted(set(universe) | COMPARISON_BENCHMARK_TICKERS)
    buffered_start = (datetime.fromisoformat(start) - timedelta(days=550)).strftime("%Y-%m-%d")
    print_progress(
        f"Universe: {len(universe)} ETFs from file  |  "
        f"Backtest: {start} → {end}  |  Download from: {buffered_start}"
    )
    if transaction_cost_bps:
        print_progress(f"Transaction cost: {transaction_cost_bps} bps/side ({transaction_cost_bps * 2} bps round-trip)")

    ohlcv = download_ohlcv_history(
        tickers,
        start=buffered_start,
        end=end,
        cache_dir=price_cache_dir,
        refresh_cache=refresh_cache,
    )
    close_px = get_close_series(ohlcv)

    feature_store = precompute_feature_store(
        universe, ohlcv, cash_ticker,
        DEFAULT_MOMENTUM_LOOKBACK_DAYS, DEFAULT_BREAKOUT_DAYS,
        DEFAULT_EXIT_DAYS, DEFAULT_ATR_DAYS,
    )

    schedules    = {"Weekly": "W", "Biweekly": "BW", "Monthly": "M", "Quarterly": "Q"}
    equity_curves, turnover_tables, weight_tables, entry_tables = {}, {}, {}, {}
    summary_rows: list = []

    for label, code in schedules.items():
        result = run_schedule_backtest(
            universe, ohlcv, close_px, feature_store, code, cash_ticker, start, end,
            max_alloc, transaction_cost_bps, allocation_mode, execution_mode, max_wait_pullback,
            min_executable_for_cash_cap, max_cash_when_enough,
            ticker_to_sleeve=ticker_to_sleeve,
            sleeve_targets=sleeve_targets, sleeve_max_allocs=sleeve_max_allocs,
        )
        if result[0] is None:
            print_progress(f"Skipping {label}: not enough rebalance points in range")
            continue

        eq, to_df, w_df, e_df = result
        equity_curves[label]   = eq
        turnover_tables[label] = to_df
        weight_tables[label]   = w_df
        entry_tables[label]    = e_df

        stats = metrics_from_equity_curve(
            eq, to_df["turnover"] if not to_df.empty else pd.Series(dtype=float)
        )
        stats.update({
            "Avg BUY NOW Count": float(e_df["buy_now_count"].mean())            if not e_df.empty else 0.0,
            "Avg WAIT Count":    float(e_df["wait_for_pullback_count"].mean())  if not e_df.empty else 0.0,
            "Avg DO NOT BUY":    float(e_df["do_not_buy_count"].mean())         if not e_df.empty else 0.0,
            "Avg SGOV Weight":   float(e_df["sgov_weight"].mean())              if not e_df.empty else 0.0,
            "Schedule":          label,
        })
        summary_rows.append(stats)

    for b in sorted(COMPARISON_BENCHMARK_TICKERS):
        if b in close_px.columns:
            eq    = benchmark_series(close_px, start, end, b)
            stats = metrics_from_equity_curve(eq, pd.Series(dtype=float))
            stats.update({
                "Schedule": b,
                "Avg BUY NOW Count": np.nan,
                "Avg WAIT Count": np.nan,
                "Avg DO NOT BUY": np.nan,
                "Avg SGOV Weight": np.nan,
            })
            summary_rows.append(stats)
            equity_curves[b] = eq

    summary = pd.DataFrame(summary_rows)[[
        "Schedule","Total Return","CAGR","Annual Vol","Sharpe","Max Drawdown","Calmar",
        "Avg Turnover/Trade Date","Num Trade Dates",
        "Avg BUY NOW Count","Avg WAIT Count","Avg DO NOT BUY","Avg SGOV Weight",
    ]].sort_values("CAGR", ascending=False)

    equity_df = pd.DataFrame(equity_curves)

    print_progress("Writing output files")
    summary.to_csv(f"{export_prefix}_backtest_summary.csv", index=False)
    equity_df.to_csv(f"{export_prefix}_equity_curves.csv", index=True)

    for label, df in turnover_tables.items():
        df.to_csv(f"{export_prefix}_{label.lower()}_turnover.csv", index=False)
    for label, df in weight_tables.items():
        df.to_csv(f"{export_prefix}_{label.lower()}_weights_history.csv", index=False)
    for label, df in entry_tables.items():
        df.to_csv(f"{export_prefix}_{label.lower()}_entry_labels.csv", index=False)

    print("=" * 126)
    print("BACKTEST SUMMARY: DRIFT-AWARE MOC | WEEKLY vs BIWEEKLY vs MONTHLY vs QUARTERLY vs BENCHMARKS (SPY, QQQ)")
    print("=" * 126)

    disp = summary.copy()
    for c in ["Total Return","CAGR","Annual Vol","Max Drawdown","Avg Turnover/Trade Date","Avg SGOV Weight"]:
        disp[c] = summary[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    disp["Sharpe"] = summary["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    disp["Calmar"] = summary["Calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    for c in ["Avg BUY NOW Count","Avg WAIT Count","Avg DO NOT BUY"]:
        disp[c] = summary[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    print(disp.to_string(index=False))
    print()
    print(f"Wrote: {export_prefix}_backtest_summary.csv")
    print(f"Wrote: {export_prefix}_equity_curves.csv")
    print("Wrote schedule turnover, weights-history, and entry-label CSVs")


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ETF Strategy Stack — allocation & backtest tool")
    p.add_argument("--mode", choices=["allocate", "backtest"], required=True)
    p.add_argument("--universe", help="Path to combined universe file (one ticker per line)")
    p.add_argument("--holdings", help="Path to holdings CSV (required for --mode allocate)")

    p.add_argument("--universe-3x", help="Optional 3x sleeve universe file")
    p.add_argument("--universe-2x", help="Optional 2x single-name sleeve universe file")
    p.add_argument("--universe-broad", help="Optional broad/index/macro sleeve universe file")
    p.add_argument(
        "--sleeve-targets",
        default=None,
        help='Comma-separated sleeve target weights, e.g. "broad=0.7,3x=0.2,2x=0.1"',
    )
    p.add_argument(
        "--sleeve-max-allocs",
        default=None,
        help='Comma-separated max allocation caps by sleeve, e.g. "broad=0.2,3x=0.1,2x=0.08"',
    )
    p.add_argument("--cash-ticker", default=DEFAULT_CASH_TICKER)
    p.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD")
    p.add_argument("--export-prefix", default="etf_strategy")
    p.add_argument(
        "--max-alloc", type=float, default=DEFAULT_MAX_ALLOC,
        help="Maximum allocation per non-cash ETF as a 0-1 fraction. Use 1.0 for no cap.",
    )
    p.add_argument(
        "--transaction-cost-bps", type=int, default=DEFAULT_TRANSACTION_COST_BPS,
        help="One-way transaction cost in basis points on trade dates (default: 0).",
    )
    p.add_argument(
        "--allocation-mode",
        default="score_proportional",
        choices=["equal", "score_proportional", "momentum_proportional"],
        help="How selected ETFs are weighted before residual goes to cash.",
    )
    p.add_argument(
        "--execution-mode",
        default="overlay",
        choices=["overlay", "pure_topk"],
        help="overlay = obey BUY NOW / WAIT / DO NOT BUY labels; pure_topk = ignore entry overlay and use model weights directly.",
    )
    p.add_argument(
        "--max-wait-pullback",
        type=int,
        default=3,
        help="Maximum number of WAIT FOR PULLBACK ETFs allowed to receive partial allocation in overlay mode.",
    )
    p.add_argument(
        "--min-executable-for-cash-cap",
        type=int,
        default=DEFAULT_MIN_EXECUTABLE_FOR_CASH_CAP,
        help="If at least this many ETFs are executable, cap cash at --max-cash-when-enough.",
    )
    p.add_argument(
        "--max-cash-when-enough",
        type=float,
        default=DEFAULT_MAX_CASH_WHEN_ENOUGH,
        help="Maximum cash weight allowed when enough ETFs are executable.",
    )
    p.add_argument(
        "--price-cache-dir",
        default=None,
        help="Optional directory for OHLCV parquet cache. If provided, cached files are reused across runs.",
    )
    p.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Redownload OHLCV even if cache files exist.",
    )
    return p


def main():
    args = build_parser().parse_args()
    sleeve_targets = parse_key_value_string(args.sleeve_targets, float) if args.sleeve_targets else None
    sleeve_max_allocs = parse_key_value_string(args.sleeve_max_allocs, float) if args.sleeve_max_allocs else None

    if args.mode == "allocate":
        if not args.holdings:
            raise SystemExit("--holdings is required for --mode allocate")
        allocation_mode(
            args.universe,
            args.holdings,
            args.cash_ticker.upper(),
            args.export_prefix,
            args.start,
            args.end,
            args.max_alloc,
            args.allocation_mode,
            args.execution_mode,
            args.max_wait_pullback,
            args.min_executable_for_cash_cap,
            args.max_cash_when_enough,
            args.price_cache_dir,
            args.refresh_cache,
            universe_3x_path=args.universe_3x,
            universe_2x_path=args.universe_2x,
            universe_broad_path=args.universe_broad,
            sleeve_targets=sleeve_targets,
            sleeve_max_allocs=sleeve_max_allocs,
        )
    else:
        if not args.start or not args.end:
            raise SystemExit("--start and --end are required for --mode backtest")
        backtest_mode(
            args.universe,
            args.cash_ticker.upper(),
            args.export_prefix,
            args.start,
            args.end,
            args.max_alloc,
            args.transaction_cost_bps,
            args.allocation_mode,
            args.execution_mode,
            args.max_wait_pullback,
            args.min_executable_for_cash_cap,
            args.max_cash_when_enough,
            args.price_cache_dir,
            args.refresh_cache,
            universe_3x_path=args.universe_3x,
            universe_2x_path=args.universe_2x,
            universe_broad_path=args.universe_broad,
            sleeve_targets=sleeve_targets,
            sleeve_max_allocs=sleeve_max_allocs,
        )


if __name__ == "__main__":
    main()
