# #!/usr/bin/env python3
“””
ETF Momentum Strategy – Allocation & Backtest Tool

## Modes

allocate  – Score all ETFs in your universe, reconcile against current
holdings, and emit rebalance instructions.
backtest  – Simulate the strategy over Weekly / Monthly / Quarterly
rebalance schedules and compare against SPY, QQQ, DIA.

## Changes vs original

- Benchmark-hurdle logic fully removed.
- Batch yfinance downloads (50 tickers per call) with per-ticker fallback
  and configurable retries + back-off.
- Scoring in select_candidates fully vectorised (no iterrows).
- Eligibility filter vectorised.
- Standard-library `logging` replaces ad-hoc print_progress().
- Constants consolidated; OFFENSIVE / DEFENSIVE sets at module level.
- nlargest() replaces sort_values().head() where applicable.
- Graceful KeyboardInterrupt / unhandled-exception handling in main().
- Type hints and docstrings throughout.
  “””
  from **future** import annotations

import argparse
import logging
import math
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
import yfinance as yf
except ImportError as exc:
sys.exit(
f”Missing dependency: {exc}.\n”
“Install with:  pip install yfinance pandas numpy openpyxl”
)

# —————————————————————————

# Logging

# —————————————————————————

logging.basicConfig(
level=logging.INFO,
format=”[%(asctime)s] %(levelname)s %(message)s”,
datefmt=”%H:%M:%S”,
stream=sys.stdout,
)
log = logging.getLogger(**name**)

# —————————————————————————

# Strategy constants

# —————————————————————————

DEFAULT_CASH_TICKER            = “SGOV”
DEFAULT_TOP_K                  = 5
DEFAULT_MAX_ALLOC              = 0.20
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
DEFAULT_HOLD_BUFFER_MULTIPLIER = 2
DEFAULT_OFFENSIVE_TARGET       = 0.80

MIN_HISTORY_BARS = max(
DEFAULT_SMA_LONG,
DEFAULT_BREAKOUT_DAYS + 5,
DEFAULT_MOMENTUM_LOOKBACK_DAYS + 5,
)

# Download settings

DOWNLOAD_BATCH_SIZE  = 50
DOWNLOAD_RETRIES     = 3
DOWNLOAD_RETRY_DELAY = 2  # seconds between retries

REBALANCE_MAP: Dict[str, str] = {“W”: “W-FRI”, “M”: “ME”, “Q”: “QE”}

BENCHMARK_TICKERS: Tuple[str, …] = (“SPY”, “QQQ”, “DIA”)

OFFENSIVE_TICKERS: Set[str] = {
“QQQ”,“VGT”,“IGV”,“SOXX”,“SMH”,“ARKK”,“AIQ”,“BOTZ”,“CLOU”,“WCLD”,“FDN”,“IYW”,
“IWF”,“MTUM”,“JMOM”,“QCLN”,“TAN”,“PBW”,“CIBR”,“HACK”,“XLK”,“XLY”,“XLC”,“IWM”,
“IJR”,“VB”,“VUG”,“SCHG”,“MGK”,“IBIT”,“ETHA”,“KWEB”,“EMQQ”,“ARKW”,“ARKG”,“ROBT”,
“XT”,“FTEC”,“IWO”,“VONG”,“SPYG”,“VTI”,“VOO”,“SPY”,“DIA”,
}
DEFENSIVE_TICKERS: Set[str] = {
“SGOV”,“SHY”,“IEF”,“TLT”,“BIL”,“TIP”,“GLD”,“IAU”,“UUP”,“USMV”,“SPLV”,“QUAL”,
“SCHD”,“VTV”,“IVE”,“IWD”,“XLU”,“XLP”,“VYM”,“DVY”,“JEPI”,“JEPQ”,“BND”,“AGG”,
“LQD”,“DBC”,“GSG”,“REET”,“VNQ”,
}

# Output column sets

METRICS_COLS: List[str] = [
“ticker”,“sleeve”,“last_close”,“ret_6m”,“realized_vol20”,“atr15”,
“above_ema10”,“above_sma200”,“sma50_gt_sma200”,“macd”,“macd_signal”,“macd_hist”,
“weekly_macd_hist”,“breakout_89d”,“breakout_recent”,“exit_13d”,“raw_score”,
“selected”,“entry_label”,“model_target_alloc_pct”,“current_alloc_pct”,
“target_alloc_pct”,“delta_pct_points”,“action”,“rationale”,
]
ACTION_COLS: List[str] = [
“ticker”,“sleeve”,“entry_label”,“current_alloc_pct”,“model_target_alloc_pct”,
“target_alloc_pct”,“delta_pct_points”,“action”,“rationale”,
]
SUMMARY_COLS: List[str] = [
“Schedule”,“Total Return”,“CAGR”,“Annual Vol”,“Sharpe”,“Max Drawdown”,“Calmar”,
“Avg Turnover/Rebalance”,“Num Rebalances”,
“Avg BUY NOW Count”,“Avg WAIT Count”,“Avg DO NOT BUY Count”,“Avg SGOV Weight”,
]

# —————————————————————————

# I/O helpers

# —————————————————————————

_TICKER_RE = re.compile(r”[A-Z]{1,6}|[A-Z]{1,5}[.-][A-Z]{1,3}”)

def parse_universe_file(path: str) -> List[str]:
“”“Read a plain-text file of ticker symbols (one per line, any order).”””
tickers: List[str] = []
seen:    Set[str]  = set()
with open(path, encoding=“utf-8”, errors=“ignore”) as fh:
for raw in fh:
line = raw.strip().upper()
if line and _TICKER_RE.fullmatch(line) and line not in seen:
seen.add(line)
tickers.append(line)
if not tickers:
raise ValueError(f”No valid tickers found in universe file: {path}”)
return tickers

def read_holdings(path: str) -> pd.DataFrame:
“””
Parse a holdings CSV into a DataFrame with columns:
ticker, [shares], [current_weight]
Accepts allocation as pct (0-100) or fraction (0-1) and normalises.
“””
df = pd.read_csv(path)
df.columns = [c.strip().lower() for c in df.columns]
if “ticker” not in df.columns:
raise ValueError(“holdings.csv must contain a ‘ticker’ column.”)

```
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
df = df[df["ticker"].str.len() > 0].copy()
out = pd.DataFrame({"ticker": df["ticker"]})

if "shares" in df.columns:
    out["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)

WEIGHT_COLS = ["allocation_pct","current_alloc_%","current_alloc_pct","weight","current_weight"]
MV_COLS     = ["market_value","value","current_value"]

weight_col = next((c for c in WEIGHT_COLS if c in df.columns), None)
if weight_col:
    vals = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    out["current_weight"] = np.where(vals > 1.0, vals / 100.0, vals)
else:
    mv_col = next((c for c in MV_COLS if c in df.columns), None)
    if mv_col:
        mv    = pd.to_numeric(df[mv_col], errors="coerce").fillna(0.0)
        total = float(mv.sum())
        out["current_weight"] = mv / total if total > 0 else 0.0

return out.groupby("ticker", as_index=False).sum(numeric_only=True)
```

# —————————————————————————

# Price download

# —————————————————————————

def download_ohlcv_history(
tickers: List[str],
start:   Optional[str],
end:     Optional[str],
) -> Dict[str, pd.DataFrame]:
“””
Download daily OHLCV in batches of DOWNLOAD_BATCH_SIZE.
On batch failure falls back to individual downloads with retry + back-off.
“””
out:     Dict[str, pd.DataFrame] = {}
total   = len(tickers)
batches = [tickers[i:i + DOWNLOAD_BATCH_SIZE] for i in range(0, total, DOWNLOAD_BATCH_SIZE)]
log.info(
“Downloading price history for %d tickers (%s → %s) in %d batches”,
total, start or “max”, end or “latest”, len(batches),
)

```
for batch_idx, batch in enumerate(batches, 1):
    log.info("Batch %d/%d (%d tickers)", batch_idx, len(batches), len(batch))
    raw = _batch_download(batch, start, end)

    if raw is None:
        # Batch completely failed – fall back ticker by ticker
        for t in batch:
            out.update(_single_download(t, start, end))
        continue

    if len(batch) == 1:
        if not raw.empty:
            out[batch[0]] = raw.sort_index()
    else:
        for t in batch:
            try:
                df = raw[t].dropna(how="all").sort_index()
                if not df.empty:
                    out[t] = df
            except KeyError:
                pass

log.info("Download complete. Usable tickers: %d / %d", len(out), total)
return out
```

def _batch_download(
batch: List[str],
start: Optional[str],
end:   Optional[str],
) -> Optional[pd.DataFrame]:
for attempt in range(1, DOWNLOAD_RETRIES + 1):
try:
raw = yf.download(
batch,
start=start,
end=end,
auto_adjust=False,
progress=False,
interval=“1d”,
threads=True,
group_by=“ticker”,
)
if raw is not None and not raw.empty:
return raw
except Exception as exc:
log.warning(“Batch attempt %d/%d failed: %s”, attempt, DOWNLOAD_RETRIES, exc)
if attempt < DOWNLOAD_RETRIES:
time.sleep(DOWNLOAD_RETRY_DELAY)
return None

def _single_download(
ticker: str,
start:  Optional[str],
end:    Optional[str],
) -> Dict[str, pd.DataFrame]:
for attempt in range(1, DOWNLOAD_RETRIES + 1):
try:
df = yf.download(
ticker, start=start, end=end,
auto_adjust=False, progress=False,
interval=“1d”, threads=False,
)
if df is not None and not df.empty:
df = df.sort_index()
if isinstance(df.columns, pd.MultiIndex):
df.columns = [c[0] for c in df.columns]
return {ticker: df}
except Exception as exc:
log.warning(“Single download attempt %d/%d for %s failed: %s”, attempt, DOWNLOAD_RETRIES, ticker, exc)
if attempt < DOWNLOAD_RETRIES:
time.sleep(DOWNLOAD_RETRY_DELAY)
log.warning(“Permanently failed to download %s”, ticker)
return {}

def get_close_series(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
“”“Assemble a (date × ticker) DataFrame of adjusted close prices.”””
closes = {
t: (df[“Adj Close”] if “Adj Close” in df.columns else df[“Close”])
for t, df in ohlcv.items()
}
if not closes:
raise ValueError(“No close price series available.”)
return pd.DataFrame(closes).sort_index()

# —————————————————————————

# Technical indicators

# —————————————————————————

def ema(series: pd.Series, span: int) -> pd.Series:
return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
return series.rolling(window).mean()

def macd(
series: pd.Series,
fast:   int = DEFAULT_MACD_FAST,
slow:   int = DEFAULT_MACD_SLOW,
signal: int = DEFAULT_MACD_SIGNAL,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
macd_line   = ema(series, fast) - ema(series, slow)
signal_line = ema(macd_line, signal)
return macd_line, signal_line, macd_line - signal_line

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
prev_close = close.shift(1)
return pd.concat(
[high - low, (high - prev_close).abs(), (low - prev_close).abs()],
axis=1,
).max(axis=1)

def atr(
high:  pd.Series,
low:   pd.Series,
close: pd.Series,
window: int = DEFAULT_ATR_DAYS,
) -> pd.Series:
return true_range(high, low, close).rolling(window).mean()

def realized_vol(series: pd.Series, window: int = DEFAULT_VOL_DAYS) -> pd.Series:
return series.pct_change().rolling(window).std() * math.sqrt(252)

def weekly_proxy_macd_hist(series: pd.Series) -> pd.Series:
weekly = series.resample(“W-FRI”).last().dropna()
_, _, hist = macd(weekly)
return hist.reindex(series.index, method=“ffill”)

# —————————————————————————

# Sleeve classification

# —————————————————————————

def classify_sleeve(ticker: str) -> str:
if ticker in OFFENSIVE_TICKERS:
return “offensive”
if ticker in DEFENSIVE_TICKERS:
return “defensive”
return “offensive”   # unknown tickers default to offensive

# —————————————————————————

# Snapshot metrics

# —————————————————————————

def _scalar(s: pd.Series) -> float:
“”“Return the last value of a series as float, or NaN if missing.”””
v = s.iloc[-1]
return float(v) if pd.notna(v) else float(“nan”)

def build_snapshot_metrics(
universe:              List[str],
ohlcv:                 Dict[str, pd.DataFrame],
asof:                  pd.Timestamp,
cash_ticker:           str,
momentum_lookback_days: int = DEFAULT_MOMENTUM_LOOKBACK_DAYS,
breakout_days:         int = DEFAULT_BREAKOUT_DAYS,
exit_days:             int = DEFAULT_EXIT_DAYS,
atr_days:              int = DEFAULT_ATR_DAYS,
) -> pd.DataFrame:
“””
Compute a cross-sectional snapshot of technical indicators for every
ticker in *universe* using data up to and including *asof*.
“””
required_bars = max(DEFAULT_SMA_LONG, breakout_days + 5, momentum_lookback_days + 5)
rows: List[Dict] = []

```
for ticker in universe:
    if ticker not in ohlcv:
        continue
    df = ohlcv[ticker].loc[:asof]
    if len(df) < required_bars:
        continue

    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    high, low = df["High"], df["Low"]

    _sma50  = sma(close, DEFAULT_SMA_MID)
    _sma200 = sma(close, DEFAULT_SMA_LONG)
    _ema10  = ema(close, DEFAULT_EMA_FAST)
    ml, sl, hist = macd(close)
    _atr15  = atr(high, low, close, atr_days)
    _rv20   = realized_vol(close, DEFAULT_VOL_DAYS)
    _w_hist = weekly_proxy_macd_hist(close)

    prior_high  = close.shift(1).rolling(breakout_days).max()
    prior_low   = close.shift(1).rolling(exit_days).min()
    recent_high = close.shift(1).rolling(DEFAULT_BREAKOUT_RECENT_DAYS).max()

    latest = float(close.iloc[-1])
    ret_6m = float(latest / close.iloc[-momentum_lookback_days] - 1.0)

    sma50_last  = _scalar(_sma50)
    sma200_last = _scalar(_sma200)
    ema10_last  = _scalar(_ema10)

    rows.append({
        "ticker":           ticker,
        "last_close":       latest,
        "sma50":            sma50_last,
        "sma200":           sma200_last,
        "ema10":            ema10_last,
        "macd":             _scalar(ml),
        "macd_signal":      _scalar(sl),
        "macd_hist":        _scalar(hist),
        "weekly_macd_hist": _scalar(_w_hist),
        "atr15":            _scalar(_atr15),
        "realized_vol20":   _scalar(_rv20),
        "ret_6m":           ret_6m,
        "breakout_89d":     bool(latest > float(prior_high.iloc[-1]))  if pd.notna(prior_high.iloc[-1])  else False,
        "breakout_recent":  bool(float(recent_high.iloc[-1]) < latest) if pd.notna(recent_high.iloc[-1]) else False,
        "exit_13d":         bool(latest < float(prior_low.iloc[-1]))   if pd.notna(prior_low.iloc[-1])   else False,
        "above_ema10":      bool(latest > ema10_last),
        "above_sma200":     bool(latest > sma200_last),
        "sma50_gt_sma200":  bool(sma50_last > sma200_last),
        "cash_like":        ticker == cash_ticker,
        "sleeve":           classify_sleeve(ticker),
    })

if not rows:
    lengths = [len(ohlcv[t].loc[:asof]) for t in universe if t in ohlcv]
    raise ValueError(
        f"No metrics available at {asof.date()}. "
        f"Required bars: {required_bars}. "
        f"Max bars found: {max(lengths, default=0)}. "
        f"Try --start with an earlier date."
    )

return pd.DataFrame(rows)
```

# —————————————————————————

# Weight normalisation (cap-aware)

# —————————————————————————

def capped_normalize(
weights:          pd.Series,
max_cap:          float,
uncapped_tickers: Optional[Set[str]] = None,
) -> pd.Series:
“””
Normalise *weights* so they sum to 1, with no single ticker exceeding
*max_cap* (unless it is in *uncapped_tickers*).
“””
uncapped_tickers = uncapped_tickers or set()
w = weights.clip(lower=0.0).fillna(0.0)
if w.sum() <= 0:
return w
w /= w.sum()

```
for _ in range(20):
    over = [t for t in w.index if t not in uncapped_tickers and w[t] > max_cap + 1e-12]
    if not over:
        break
    for t in over:
        w[t] = max_cap
    capped_mask = pd.Series(
        [t not in uncapped_tickers and w[t] >= max_cap - 1e-12 for t in w.index],
        index=w.index,
    )
    free_sum   = float(w[~capped_mask].sum())
    capped_sum = float(w[capped_mask].sum())
    if free_sum > 0:
        w.loc[~capped_mask] = w.loc[~capped_mask] / free_sum * (1.0 - capped_sum)

total = float(w.sum())
return w / total if total > 0 else w
```

# —————————————————————————

# Entry signal classification

# —————————————————————————

def classify_entry_signal(row: pd.Series) -> str:
“”“Return BUY NOW / WAIT FOR PULLBACK / DO NOT BUY for a single ETF.”””
if bool(row.get(“cash_like”, False)):
return “BUY NOW”
if not bool(row.get(“above_sma200”, False)) or bool(row.get(“exit_13d”, False)):
return “DO NOT BUY”

```
last_close = row.get("last_close", float("nan"))
ema10v     = row.get("ema10",      float("nan"))
if pd.isna(last_close) or pd.isna(ema10v) or ema10v <= 0:
    return "WAIT FOR PULLBACK"

dist         = float(last_close) / float(ema10v) - 1.0
strong_trend = bool(row.get("above_sma200")) and bool(row.get("sma50_gt_sma200"))
positive_mom = (
    pd.notna(row.get("macd_hist"))        and row["macd_hist"]        > 0
    and pd.notna(row.get("weekly_macd_hist")) and row["weekly_macd_hist"] > 0
)

if strong_trend and positive_mom:
    if bool(row.get("breakout_89d")) and bool(row.get("breakout_recent")) and dist <= DEFAULT_EXTENDED_FROM_EMA:
        return "BUY NOW"
    if bool(row.get("above_ema10")) and dist <= DEFAULT_PULLBACK_EMA_BUFFER:
        return "BUY NOW"
    if dist > DEFAULT_EXTENDED_FROM_EMA:
        return "WAIT FOR PULLBACK"

if strong_trend:
    return "WAIT FOR PULLBACK"

return "DO NOT BUY"
```

def apply_entry_labels_and_allocate(
df:          pd.DataFrame,
cash_ticker: str,
max_alloc:   float,
) -> pd.DataFrame:
“””
Stamp entry labels onto currently-selected ETFs, then translate
‘BUY NOW’ tickers into actual target weights.  Any residual goes to cash.
“””
df = df.copy()
df[“entry_label”] = “DO NOT BUY”
sel_nc = df[“selected”] & ~df[“cash_like”]
if sel_nc.any():
df.loc[sel_nc, “entry_label”] = df.loc[sel_nc].apply(classify_entry_signal, axis=1)

```
df["model_target_weight"] = df["target_weight"]
df["target_weight"]       = 0.0

buy_now = df[df["selected"] & ~df["cash_like"] & (df["entry_label"] == "BUY NOW")]
if buy_now.empty:
    df.loc[df["ticker"] == cash_ticker, "target_weight"] = 1.0
else:
    w = capped_normalize(
        buy_now.set_index("ticker")["model_target_weight"].clip(lower=0.0),
        max_cap=max_alloc,
    )
    for t, wt in w.items():
        df.loc[df["ticker"] == t, "target_weight"] = float(wt)
    residual = 1.0 - float(df["target_weight"].sum())
    df.loc[df["ticker"] == cash_ticker, "target_weight"] += residual

total = float(df["target_weight"].sum())
if total > 0:
    df["target_weight"] /= total
return df
```

# —————————————————————————

# Candidate selection  (fully vectorised scoring)

# —————————————————————————

def select_candidates(
metrics:               pd.DataFrame,
cash_ticker:           str,
top_k:                 int,
max_alloc:             float,
prev_holdings:         Optional[Set[str]] = None,
hold_buffer_multiplier: int = DEFAULT_HOLD_BUFFER_MULTIPLIER,
) -> pd.DataFrame:
“””
Score every ETF, select the top-K split across offensive/defensive sleeves,
apply persistence (hold-buffer), then hand off to entry-signal gating.
“””
df = metrics.copy()

```
# --- Eligibility (vectorised) ---
df["eligible"] = df["ticker"].eq(cash_ticker) | (df["above_sma200"] & ~df["exit_13d"])

# --- Composite score (vectorised) ---
r6_min, r6_max = df["ret_6m"].min(),        df["ret_6m"].max()
rv_min, rv_max = df["realized_vol20"].min(), df["realized_vol20"].max()

rs_scaled = (df["ret_6m"]        - r6_min) / (r6_max - r6_min + 1e-12)
iv_scaled = 1.0 - (df["realized_vol20"] - rv_min) / (rv_max - rv_min + 1e-12)

score = (
      df["above_sma200"].astype(float)          * 3.0
    + df["sma50_gt_sma200"].astype(float)        * 2.5
    + df["above_ema10"].astype(float)            * 1.0
    + (df["macd_hist"] > 0).astype(float)        * 2.0
    + (df["weekly_macd_hist"] > 0).astype(float) * 1.5
    + df["breakout_89d"].astype(float)           * 1.5
    + df["breakout_recent"].astype(float)        * 2.0
    + rs_scaled.fillna(0.0)                      * 5.0
    + iv_scaled.fillna(0.0)                      * 1.0
    + df["sleeve"].eq("offensive").astype(float) * 0.5
    - df["exit_13d"].astype(float)               * 4.0
)
# Zero-out ineligible non-cash tickers
df["raw_score"] = score.where(df["eligible"] | df["cash_like"], other=0.0)

# --- Sleeve-aware top-K selection ---
non_cash     = df[~df["cash_like"]]
offensive_k  = max(1, min(top_k, math.ceil(top_k * DEFAULT_OFFENSIVE_TARGET)))
defensive_k  = max(0, top_k - offensive_k)

off_picks = set(
    non_cash[non_cash["sleeve"] == "offensive"]
    .nlargest(offensive_k, "raw_score")["ticker"]
)
def_picks = set(
    non_cash[non_cash["sleeve"] == "defensive"]
    .nlargest(defensive_k, "raw_score")["ticker"]
)
selected = off_picks | def_picks

# --- Persistence: hold buffer ---
if prev_holdings:
    keep_threshold = top_k * hold_buffer_multiplier
    ranked = (
        non_cash.sort_values("raw_score", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    for t in prev_holdings:
        row = ranked[ranked["ticker"] == t]
        if not row.empty:
            if int(row["rank"].iloc[0]) <= keep_threshold and float(row["raw_score"].iloc[0]) > 0:
                selected.add(t)

# Trim to top_k and remove zero-score tickers
selected = set(
    non_cash[non_cash["ticker"].isin(selected)]
    .nlargest(top_k, "raw_score")["ticker"]
)
zero_score = set(df.loc[~df["cash_like"] & (df["raw_score"] <= 0), "ticker"])
selected  -= zero_score

df["selected"]      = df["ticker"].isin(selected) | df["cash_like"]
df["target_weight"] = 0.0

sel_non_cash = df[df["selected"] & ~df["cash_like"]]
if sel_non_cash.empty or sel_non_cash["raw_score"].clip(lower=0).sum() <= 0:
    df.loc[df["ticker"] == cash_ticker, "target_weight"] = 1.0
    df["entry_label"]        = np.where(df["cash_like"], "BUY NOW", "DO NOT BUY")
    df["model_target_weight"] = df["target_weight"]
    return df

w = capped_normalize(
    sel_non_cash.set_index("ticker")["raw_score"].clip(lower=0.0),
    max_cap=max_alloc,
)
for t, wt in w.items():
    df.loc[df["ticker"] == t, "target_weight"] = float(wt)

residual = 1.0 - float(df["target_weight"].sum())
df.loc[df["ticker"] == cash_ticker, "target_weight"] += residual
df["target_weight"] /= float(df["target_weight"].sum())

return apply_entry_labels_and_allocate(df, cash_ticker=cash_ticker, max_alloc=max_alloc)
```

# —————————————————————————

# Rationale text

# —————————————————————————

def rationale_for_row(row: pd.Series, cash_ticker: str) -> str:
t, label, sleeve = row[“ticker”], row.get(“entry_label”, “”), row.get(“sleeve”, “”)
if t == cash_ticker:
return (
f”{cash_ticker}: holds residual cash while selected ETFs await valid entry signals.”
if row.get(“target_weight”, 0) > 0
else f”{cash_ticker}: available as cash sleeve but not currently needed.”
)
if row.get(“selected”) and label == “BUY NOW”:
return f”Selected & executable. Sleeve={sleeve}; momentum/trend confirmed.”
if row.get(“selected”) and label == “WAIT FOR PULLBACK”:
return f”Selected – wait for pullback. Sleeve={sleeve}; trend ok but entry extended or not fresh.”
if row.get(“selected”) and label == “DO NOT BUY”:
return f”Selected on ranking but do not buy. Sleeve={sleeve}; entry signal weak or invalid.”
return “Not selected or failed filter rules.”

# —————————————————————————

# Current-weight derivation

# —————————————————————————

def derive_current_weights(
holdings: pd.DataFrame,
close_px: pd.DataFrame,
) -> pd.DataFrame:
h = holdings.copy()
if “current_weight” in h.columns:
total = float(h[“current_weight”].sum())
if total > 0:
h[“current_weight”] /= total
return h[[“ticker”, “current_weight”]]

```
if "shares" in h.columns:
    prices = {
        t: float(close_px[t].dropna().iloc[-1])
           if t in close_px.columns and len(close_px[t].dropna()) > 0
           else float("nan")
        for t in h["ticker"]
    }
    h["last_price"]    = h["ticker"].map(prices)
    h["market_value"]  = h["shares"] * h["last_price"]
    total              = float(h["market_value"].sum())
    h["current_weight"] = h["market_value"] / total if total > 0 else 0.0
    return h[["ticker", "current_weight"]]

h["current_weight"] = 0.0
return h[["ticker", "current_weight"]]
```

# —————————————————————————

# Backtest helpers

# —————————————————————————

def metrics_from_equity_curve(
equity:   pd.Series,
turnover: pd.Series,
) -> Dict[str, float]:
daily_ret = equity.pct_change().fillna(0.0)
n_years   = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
cagr      = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0)
vol       = float(daily_ret.std() * math.sqrt(252))
sharpe    = float((daily_ret.mean() * 252) / vol) if vol > 0 else float(“nan”)
dd        = equity / equity.cummax() - 1.0
max_dd    = float(dd.min())
calmar    = float(cagr / abs(max_dd)) if max_dd < 0 else float(“nan”)
return {
“Total Return”:           float(equity.iloc[-1] / equity.iloc[0] - 1.0),
“CAGR”:                   cagr,
“Annual Vol”:             vol,
“Sharpe”:                 sharpe,
“Max Drawdown”:           max_dd,
“Calmar”:                 calmar,
“Avg Turnover/Rebalance”: float(turnover.mean()) if len(turnover) else 0.0,
“Num Rebalances”:         int((turnover > 0).sum()),
}

def benchmark_series(
close_px: pd.DataFrame,
start:    str,
end:      str,
ticker:   str,
) -> pd.Series:
s  = close_px.loc[start:end, ticker].dropna()
eq = s / s.iloc[0]
eq.name = ticker
return eq

def run_schedule_backtest(
universe:      List[str],
ohlcv:         Dict[str, pd.DataFrame],
close_px:      pd.DataFrame,
schedule_code: str,
cash_ticker:   str,
top_k:         int,
start:         str,
end:           str,
) -> Tuple[
Optional[pd.Series],
Optional[pd.DataFrame],
Optional[pd.DataFrame],
Optional[pd.DataFrame],
]:
px = close_px.loc[start:end, universe].dropna(how=“all”)
if px.empty:
raise ValueError(f”No price data in range {start} – {end}.”)

```
rebalance_dates   = {d for d in px.resample(REBALANCE_MAP[schedule_code]).last().index if d in px.index}
entry_check_dates = {d for d in px.resample(REBALANCE_MAP["W"]).last().index          if d in px.index}

if not rebalance_dates:
    return None, None, None, None

label = {"W": "Weekly", "M": "Monthly", "Q": "Quarterly"}[schedule_code]
log.info(
    "%s backtest: %d rebalances, %d weekly entry checks",
    label, len(rebalance_dates), len(entry_check_dates),
)

weights              = pd.Series(0.0, index=universe)
weights[cash_ticker] = 1.0
equity               = pd.Series(index=px.index, dtype=float)
equity.iloc[0]       = 1.0
turnovers:           List[Dict] = []
weights_history:     List[Dict] = [{"date": px.index[0], **weights.to_dict()}]
entry_label_history: List[Dict] = []
current_selected     = set()
pending_weights      = weights.copy()

for i in range(1, len(px)):
    dt, prev_dt = px.index[i], px.index[i - 1]
    daily_rets  = (px.loc[dt] / px.loc[prev_dt] - 1.0).fillna(0.0)
    port_ret    = float((weights.fillna(0.0) * daily_rets).sum())
    equity.iloc[i] = equity.iloc[i - 1] * (1.0 + port_ret)

    need_rebalance   = dt in rebalance_dates
    need_entry_check = dt in entry_check_dates

    if need_rebalance or need_entry_check:
        snap = build_snapshot_metrics(
            universe, ohlcv, dt, cash_ticker,
            DEFAULT_MOMENTUM_LOOKBACK_DAYS, DEFAULT_BREAKOUT_DAYS,
            DEFAULT_EXIT_DAYS, DEFAULT_ATR_DAYS,
        )

        if need_rebalance:
            alloc            = select_candidates(snap, cash_ticker, top_k, DEFAULT_MAX_ALLOC, prev_holdings=current_selected)
            current_selected = set(alloc.loc[alloc["selected"] & ~alloc["cash_like"], "ticker"])
        else:
            alloc             = select_candidates(snap, cash_ticker, top_k, DEFAULT_MAX_ALLOC, prev_holdings=current_selected)
            alloc["selected"] = alloc["ticker"].isin(current_selected) | alloc["cash_like"]
            alloc             = apply_entry_labels_and_allocate(alloc, cash_ticker=cash_ticker, max_alloc=DEFAULT_MAX_ALLOC)

        new_weights = alloc.set_index("ticker")["target_weight"].reindex(universe).fillna(0.0)
        turnover    = float((new_weights - weights).abs().sum() / 2.0) if need_rebalance else 0.0

        turnovers.append({"date": dt, "turnover": turnover})
        pending_weights = new_weights
        weights_history.append({"date": dt, **new_weights.to_dict()})

        sel_nc = alloc[alloc["selected"] & ~alloc["cash_like"]]
        ec     = sel_nc["entry_label"].value_counts()
        entry_label_history.append({
            "date":                    dt,
            "selected_non_cash_count": len(sel_nc),
            "buy_now_count":           int(ec.get("BUY NOW",          0)),
            "wait_for_pullback_count": int(ec.get("WAIT FOR PULLBACK", 0)),
            "do_not_buy_count":        int(ec.get("DO NOT BUY",        0)),
            "sgov_weight":             float(alloc.loc[alloc["ticker"] == cash_ticker, "target_weight"].sum()),
        })

    weights = pending_weights.copy()

return (
    equity.ffill(),
    pd.DataFrame(turnovers),
    pd.DataFrame(weights_history),
    pd.DataFrame(entry_label_history),
)
```

# —————————————————————————

# Mode: allocate

# —————————————————————————

def allocation_mode(
universe_path: str,
holdings_path: str,
cash_ticker:   str,
top_k:         int,
export_prefix: str,
start:         Optional[str],
end:           Optional[str],
) -> None:
log.info(“Loading universe and holdings”)
universe = parse_universe_file(universe_path)
holdings = read_holdings(holdings_path)

```
bad = sorted(set(holdings["ticker"]) - set(universe))
if bad:
    raise ValueError(f"Holdings contain tickers not in universe file: {bad}")
if cash_ticker not in universe:
    raise ValueError(f"{cash_ticker} must be present in the universe file.")

effective_start = start or (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
tickers = sorted(set(universe) | set(BENCHMARK_TICKERS))
log.info("Universe: %d ETFs | Window: %s → %s", len(universe), effective_start, end or "latest")

ohlcv    = download_ohlcv_history(tickers, start=effective_start, end=end)
close_px = get_close_series(ohlcv)
asof     = close_px.dropna(how="all").index[-1]

log.info("Computing strategy snapshot as of %s", asof.date())
metrics = build_snapshot_metrics(
    universe, ohlcv, asof, cash_ticker,
    DEFAULT_MOMENTUM_LOOKBACK_DAYS, DEFAULT_BREAKOUT_DAYS,
    DEFAULT_EXIT_DAYS, DEFAULT_ATR_DAYS,
)
log.info("ETFs with sufficient history: %d", len(metrics))

current       = derive_current_weights(holdings, close_px)
prev_holdings = set(current.loc[current["current_weight"] > 0, "ticker"])

log.info("Scoring ETFs and generating target allocation")
alloc = select_candidates(metrics, cash_ticker, top_k, DEFAULT_MAX_ALLOC, prev_holdings=prev_holdings)

log.info("Reconciling current vs target weights")
out = alloc.merge(current, on="ticker", how="left")
out["current_weight"]         = out["current_weight"].fillna(0.0)
out["delta_weight"]           = out["target_weight"] - out["current_weight"]
out["current_alloc_pct"]      = out["current_weight"]        * 100.0
out["target_alloc_pct"]       = out["target_weight"]         * 100.0
out["model_target_alloc_pct"] = out["model_target_weight"]   * 100.0
out["delta_pct_points"]       = out["delta_weight"]          * 100.0
out["action"] = np.where(
    out["delta_weight"] >  1e-6, "INCREASE",
    np.where(out["delta_weight"] < -1e-6, "DECREASE", "HOLD"),
)
out["rationale"] = out.apply(lambda r: rationale_for_row(r, cash_ticker), axis=1)

view = out.sort_values(["target_weight","raw_score","ticker"], ascending=[False,False,True])

view[METRICS_COLS].to_csv(f"{export_prefix}_full_metrics.csv",     index=False)
view[ACTION_COLS].to_csv(f"{export_prefix}_rebalance_actions.csv", index=False)

sep = "=" * 100
print(f"\n{sep}\nENHANCED ETF STRATEGY STACK\n{sep}")
print(f"As of : {asof.date()}")
print("Selection : concentrated, sleeve-aware, persistence-aware")
print("Execution : only BUY NOW names receive capital; WAIT / DO NOT BUY remain in SGOV\n")
print(view[ACTION_COLS].to_string(index=False))
print(f"\nWrote: {export_prefix}_full_metrics.csv")
print(f"Wrote: {export_prefix}_rebalance_actions.csv")
```

# —————————————————————————

# Mode: backtest

# —————————————————————————

def backtest_mode(
universe_path: str,
cash_ticker:   str,
top_k:         int,
export_prefix: str,
start:         str,
end:           str,
) -> None:
log.info(“Loading ETF universe for backtest”)
universe = parse_universe_file(universe_path)
if cash_ticker not in universe:
raise ValueError(f”{cash_ticker} must be present in the universe file.”)

```
tickers        = sorted(set(universe) | set(BENCHMARK_TICKERS))
adj_start      = (datetime.fromisoformat(start) - timedelta(days=400)).strftime("%Y-%m-%d")
log.info(
    "Universe: %d ETFs | Backtest: %s → %s (buffered download start: %s)",
    len(universe), start, end, adj_start,
)

ohlcv    = download_ohlcv_history(tickers, start=adj_start, end=end)
close_px = get_close_series(ohlcv)

schedules           = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
equity_curves:       Dict[str, pd.Series]     = {}
turnover_tables:     Dict[str, pd.DataFrame]  = {}
weight_tables:       Dict[str, pd.DataFrame]  = {}
entry_label_tables:  Dict[str, pd.DataFrame]  = {}
summary_rows:        List[Dict]               = []

for label, code in schedules.items():
    log.info("Running %s rebalance comparison", label.lower())
    eq, to_df, w_df, e_df = run_schedule_backtest(
        universe, ohlcv, close_px, code, cash_ticker, top_k, start, end,
    )
    if eq is None:
        log.warning("Skipping %s: not enough rebalance points in date range", label.lower())
        continue

    equity_curves[label]      = eq
    turnover_tables[label]    = to_df
    weight_tables[label]      = w_df
    entry_label_tables[label] = e_df

    stats = metrics_from_equity_curve(
        eq, to_df["turnover"] if not to_df.empty else pd.Series(dtype=float)
    )
    if not e_df.empty:
        stats.update({
            "Avg BUY NOW Count":    float(e_df["buy_now_count"].mean()),
            "Avg WAIT Count":       float(e_df["wait_for_pullback_count"].mean()),
            "Avg DO NOT BUY Count": float(e_df["do_not_buy_count"].mean()),
            "Avg SGOV Weight":      float(e_df["sgov_weight"].mean()),
        })
    else:
        stats.update({
            "Avg BUY NOW Count": 0.0, "Avg WAIT Count": 0.0,
            "Avg DO NOT BUY Count": 0.0, "Avg SGOV Weight": 0.0,
        })
    stats["Schedule"] = label
    summary_rows.append(stats)

for b in BENCHMARK_TICKERS:
    if b in close_px.columns:
        eq    = benchmark_series(close_px, start, end, b)
        stats = metrics_from_equity_curve(eq, pd.Series(dtype=float))
        stats.update({
            "Schedule": b,
            "Avg BUY NOW Count": float("nan"), "Avg WAIT Count": float("nan"),
            "Avg DO NOT BUY Count": float("nan"), "Avg SGOV Weight": float("nan"),
        })
        summary_rows.append(stats)
        equity_curves[b] = eq

summary   = pd.DataFrame(summary_rows)[SUMMARY_COLS].sort_values("CAGR", ascending=False)
equity_df = pd.DataFrame(equity_curves)

log.info("Writing backtest output files")
summary.to_csv(f"{export_prefix}_backtest_summary.csv", index=False)
equity_df.to_csv(f"{export_prefix}_equity_curves.csv",  index=True)
for lbl, df in turnover_tables.items():    df.to_csv(f"{export_prefix}_{lbl.lower()}_turnover.csv",        index=False)
for lbl, df in weight_tables.items():      df.to_csv(f"{export_prefix}_{lbl.lower()}_weights_history.csv", index=False)
for lbl, df in entry_label_tables.items(): df.to_csv(f"{export_prefix}_{lbl.lower()}_entry_labels.csv",    index=False)

# --- Pretty-print summary ---
sep = "=" * 126
print(f"\n{sep}")
print("BACKTEST SUMMARY: WEEKLY vs MONTHLY vs QUARTERLY vs BENCHMARKS (SPY, QQQ, DIA)")
print(f"{sep}")
display = summary.copy()
pct_cols = ["Total Return","CAGR","Annual Vol","Max Drawdown","Avg Turnover/Rebalance","Avg SGOV Weight"]
for c in pct_cols:
    display[c] = display[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
display["Sharpe"] = summary["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
display["Calmar"] = summary["Calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
for c in ["Avg BUY NOW Count","Avg WAIT Count","Avg DO NOT BUY Count"]:
    display[c] = summary[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
print(display.to_string(index=False))
print(f"\nWrote: {export_prefix}_backtest_summary.csv")
print(f"Wrote: {export_prefix}_equity_curves.csv")
print("Wrote schedule turnover, weights-history, and entry-label diagnostic CSVs")
```

# —————————————————————————

# CLI

# —————————————————————————

def build_parser() -> argparse.ArgumentParser:
p = argparse.ArgumentParser(
description=“ETF Momentum Strategy – Allocation & Backtest Tool”,
formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
p.add_argument(”–mode”,          choices=[“allocate”,“backtest”], required=True,
help=“allocate: generate rebalance instructions; backtest: historical simulation”)
p.add_argument(”–universe”,      required=True,
help=“Path to universe file (one ticker per line)”)
p.add_argument(”–holdings”,
help=“Path to holdings CSV (required for allocate mode)”)
p.add_argument(”–cash-ticker”,   default=DEFAULT_CASH_TICKER,
help=“Ticker used as the cash/defensive sleeve”)
p.add_argument(”–top-k”,         type=int, default=DEFAULT_TOP_K,
help=“Maximum number of non-cash ETFs to hold”)
p.add_argument(”–start”,         help=“Download start date YYYY-MM-DD”)
p.add_argument(”–end”,           help=“Download end date  YYYY-MM-DD”)
p.add_argument(”–export-prefix”, default=“etf_strategy”,
help=“Prefix for all output CSV files”)
return p

def main() -> None:
args = build_parser().parse_args()
try:
if args.mode == “allocate”:
if not args.holdings:
raise SystemExit(”–holdings is required for –mode allocate”)
allocation_mode(
args.universe, args.holdings,
args.cash_ticker.upper(), args.top_k,
args.export_prefix, args.start, args.end,
)
else:
if not args.start or not args.end:
raise SystemExit(”–start and –end are required for –mode backtest”)
backtest_mode(
args.universe, args.cash_ticker.upper(), args.top_k,
args.export_prefix, args.start, args.end,
)
except KeyboardInterrupt:
log.warning(“Interrupted by user.”)
sys.exit(130)
except SystemExit:
raise
except Exception:
log.exception(“Fatal error – aborting.”)
sys.exit(1)

if **name** == “**main**”:
main()