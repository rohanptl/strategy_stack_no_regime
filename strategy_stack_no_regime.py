#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Install yfinance pandas numpy openpyxl. " + str(e))

DEFAULT_CASH_TICKER = "SGOV"
DEFAULT_TOP_K = 5
DEFAULT_MAX_ALLOC = 0.20
DEFAULT_EMA_FAST = 10
DEFAULT_SMA_MID = 50
DEFAULT_SMA_LONG = 200
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BREAKOUT_DAYS = 89
DEFAULT_BREAKOUT_RECENT_DAYS = 5
DEFAULT_EXIT_DAYS = 13
DEFAULT_ATR_DAYS = 15
DEFAULT_VOL_DAYS = 20
DEFAULT_MOMENTUM_LOOKBACK_DAYS = 126
DEFAULT_PULLBACK_EMA_BUFFER = 0.02
DEFAULT_EXTENDED_FROM_EMA = 0.05
DEFAULT_HOLD_BUFFER_MULTIPLIER = 2
DEFAULT_OFFENSIVE_TARGET = 0.80
DEFAULT_BENCHMARK_HURDLE = "spy"
REBALANCE_MAP = {"W": "W-FRI", "M": "ME", "Q": "QE"}

def print_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

def parse_universe_file(path: str) -> List[str]:
    tickers, seen = [], set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip().upper()
            if not line:
                continue
            if re.fullmatch(r"[A-Z]{1,6}", line) or re.fullmatch(r"[A-Z]{1,5}[.-][A-Z]{1,3}", line):
                if line not in seen:
                    seen.add(line); tickers.append(line)
    if not tickers:
        raise ValueError("No tickers found in universe file.")
    return tickers

def read_holdings(path: str) -> pd.DataFrame:
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
    out: Dict[str, pd.DataFrame] = {}
    total = len(tickers)
    print_progress(f"Starting price download for {total} tickers from {start or 'max'} to {end or 'latest'}")
    for i, t in enumerate(tickers, start=1):
        if i == 1 or i == total or i % 10 == 0:
            print_progress(f"Downloading {i}/{total}: {t}")
        df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False, interval="1d", threads=False)
        if df is None or len(df) == 0:
            continue
        df = df.sort_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        out[t] = df
    print_progress(f"Finished price download. Usable tickers: {len(out)}/{total}")
    return out

def get_close_series(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {}
    for t, df in ohlcv.items():
        closes[t] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
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
    return pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = DEFAULT_ATR_DAYS) -> pd.Series:
    return true_range(high, low, close).rolling(window).mean()

def realized_vol(series: pd.Series, window: int = DEFAULT_VOL_DAYS) -> pd.Series:
    return series.pct_change().rolling(window).std() * math.sqrt(252)

def weekly_proxy_macd_hist(series: pd.Series) -> pd.Series:
    weekly = series.resample("W-FRI").last().dropna()
    _, _, h = macd(weekly)
    return h.reindex(series.index, method="ffill")

def classify_sleeve(ticker: str) -> str:
    offensive = {"QQQ","VGT","IGV","SOXX","SMH","ARKK","AIQ","BOTZ","CLOU","WCLD","FDN","IYW","IWF","MTUM","JMOM",
                 "QCLN","TAN","PBW","CIBR","HACK","XLK","XLY","XLC","IWM","IJR","VB","VUG","SCHG","MGK","IBIT","ETHA",
                 "KWEB","EMQQ","ARKW","ARKG","ROBT","XT","FTEC","IWO","VONG","SPYG","VTI","VOO","SPY","DIA"}
    defensive = {"SGOV","SHY","IEF","TLT","BIL","TIP","GLD","IAU","UUP","USMV","SPLV","QUAL","SCHD","VTV","IVE","IWD",
                 "XLU","XLP","VYM","DVY","JEPI","JEPQ","BND","AGG","LQD","DBC","GSG","REET","VNQ"}
    if ticker in offensive: return "offensive"
    if ticker in defensive: return "defensive"
    return "offensive"

def benchmark_hurdle_return(metrics: pd.DataFrame, hurdle_mode: str) -> float:
    hurdle_mode = hurdle_mode.lower()
    benchmark_returns = {}
    for b in ["SPY","QQQ","DIA"]:
        row = metrics.loc[metrics["ticker"] == b]
        if not row.empty and pd.notna(row.iloc[0]["ret_6m"]):
            benchmark_returns[b] = float(row.iloc[0]["ret_6m"])
    if hurdle_mode == "none": return -np.inf
    if hurdle_mode == "spy": return benchmark_returns.get("SPY", -np.inf)
    if hurdle_mode == "qqq": return benchmark_returns.get("QQQ", -np.inf)
    if hurdle_mode == "dia": return benchmark_returns.get("DIA", -np.inf)
    if hurdle_mode == "best_of_3": return max(benchmark_returns.values()) if benchmark_returns else -np.inf
    raise ValueError(f"Unsupported benchmark hurdle mode: {hurdle_mode}")

def build_snapshot_metrics(universe, ohlcv, asof, cash_ticker, momentum_lookback_days, breakout_days, exit_days, atr_days):
    rows = []
    for ticker in universe:
        if ticker not in ohlcv: continue
        df = ohlcv[ticker].loc[:asof].copy()
        if len(df) < max(DEFAULT_SMA_LONG, breakout_days + 5, momentum_lookback_days + 5): continue
        close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        high, low = df["High"], df["Low"]
        sma50, sma200, ema10 = sma(close, DEFAULT_SMA_MID), sma(close, DEFAULT_SMA_LONG), ema(close, DEFAULT_EMA_FAST)
        macd_line, signal_line, hist = macd(close)
        atr15 = atr(high, low, close, atr_days)
        rv20 = realized_vol(close, DEFAULT_VOL_DAYS)
        weekly_hist = weekly_proxy_macd_hist(close)
        prior_high = close.shift(1).rolling(breakout_days).max()
        prior_low = close.shift(1).rolling(exit_days).min()
        breakout_recent = (close.shift(1).rolling(DEFAULT_BREAKOUT_RECENT_DAYS).max() < close).iloc[-1]
        latest = float(close.iloc[-1])
        ret_6m = float(latest / close.iloc[-momentum_lookback_days] - 1.0)
        rows.append({
            "ticker": ticker, "last_close": latest, "sma50": float(sma50.iloc[-1]), "sma200": float(sma200.iloc[-1]),
            "ema10": float(ema10.iloc[-1]), "macd": float(macd_line.iloc[-1]), "macd_signal": float(signal_line.iloc[-1]),
            "macd_hist": float(hist.iloc[-1]), "weekly_macd_hist": float(weekly_hist.iloc[-1]) if not pd.isna(weekly_hist.iloc[-1]) else np.nan,
            "atr15": float(atr15.iloc[-1]) if not pd.isna(atr15.iloc[-1]) else np.nan,
            "realized_vol20": float(rv20.iloc[-1]) if not pd.isna(rv20.iloc[-1]) else np.nan,
            "ret_6m": ret_6m, "breakout_89d": bool(latest > prior_high.iloc[-1]) if not pd.isna(prior_high.iloc[-1]) else False,
            "breakout_recent": bool(breakout_recent) if not pd.isna(breakout_recent) else False,
            "exit_13d": bool(latest < prior_low.iloc[-1]) if not pd.isna(prior_low.iloc[-1]) else False,
            "above_ema10": bool(latest > float(ema10.iloc[-1])), "above_sma200": bool(latest > float(sma200.iloc[-1])),
            "sma50_gt_sma200": bool(float(sma50.iloc[-1]) > float(sma200.iloc[-1])),
            "cash_like": ticker == cash_ticker, "sleeve": classify_sleeve(ticker)
        })
    if not rows: raise ValueError("No metrics available at snapshot date.")
    return pd.DataFrame(rows)

def capped_normalize(weights: pd.Series, max_cap: float, uncapped_tickers: Optional[Set[str]] = None) -> pd.Series:
    uncapped_tickers = uncapped_tickers or set()
    w = weights.copy().fillna(0.0).clip(lower=0.0)
    if w.sum() <= 0: return w
    w = w / w.sum()
    for _ in range(20):
        over = [t for t in w.index if t not in uncapped_tickers and w[t] > max_cap + 1e-12]
        if not over: break
        for t in over: w[t] = max_cap
        capped_mask = pd.Series(False, index=w.index)
        for t in w.index:
            if t not in uncapped_tickers and w[t] >= max_cap - 1e-12: capped_mask[t] = True
        capped_sum = float(w[capped_mask].sum())
        free_mask = ~capped_mask
        free_sum = float(w[free_mask].sum())
        if free_sum > 0:
            w.loc[free_mask] = w.loc[free_mask] / free_sum * (1.0 - capped_sum)
    total = float(w.sum())
    return w / total if total > 0 else w

def classify_entry_signal(row: pd.Series) -> str:
    if bool(row.get("cash_like", False)): return "BUY NOW"
    if (not bool(row.get("above_sma200", False))) or bool(row.get("exit_13d", False)): return "DO NOT BUY"
    last_close, ema10v = row.get("last_close", np.nan), row.get("ema10", np.nan)
    if pd.isna(last_close) or pd.isna(ema10v) or ema10v <= 0: return "WAIT FOR PULLBACK"
    dist_from_ema10 = float(last_close) / float(ema10v) - 1.0
    strong_trend = bool(row.get("above_sma200", False)) and bool(row.get("sma50_gt_sma200", False))
    positive_momentum = (pd.notna(row.get("macd_hist", np.nan)) and row.get("macd_hist", np.nan) > 0) and (pd.notna(row.get("weekly_macd_hist", np.nan)) and row.get("weekly_macd_hist", np.nan) > 0)
    if strong_trend and positive_momentum and bool(row.get("breakout_89d", False)) and bool(row.get("breakout_recent", False)) and dist_from_ema10 <= DEFAULT_EXTENDED_FROM_EMA:
        return "BUY NOW"
    if strong_trend and positive_momentum and bool(row.get("above_ema10", False)) and dist_from_ema10 <= DEFAULT_PULLBACK_EMA_BUFFER:
        return "BUY NOW"
    if strong_trend and positive_momentum and dist_from_ema10 > DEFAULT_EXTENDED_FROM_EMA:
        return "WAIT FOR PULLBACK"
    if strong_trend: return "WAIT FOR PULLBACK"
    return "DO NOT BUY"

def apply_entry_labels_and_allocate(df: pd.DataFrame, cash_ticker: str, max_alloc: float) -> pd.DataFrame:
    df = df.copy()
    df["entry_label"] = "DO NOT BUY"
    selected_mask = df["selected"] & (~df["cash_like"])
    if selected_mask.any():
        df.loc[selected_mask, "entry_label"] = df.loc[selected_mask].apply(classify_entry_signal, axis=1)
    df["model_target_weight"] = df["target_weight"]
    df["target_weight"] = 0.0
    buy_now = df[(df["selected"]) & (~df["cash_like"]) & (df["entry_label"] == "BUY NOW")].copy()
    if buy_now.empty:
        df.loc[df["ticker"] == cash_ticker, "target_weight"] = 1.0
    else:
        w = buy_now.set_index("ticker")["model_target_weight"].clip(lower=0.0)
        if w.sum() > 0:
            w = capped_normalize(w, max_cap=max_alloc, uncapped_tickers=set())
            for t, wt in w.items():
                df.loc[df["ticker"] == t, "target_weight"] = float(wt)
        residual = 1.0 - float(df["target_weight"].sum())
        df.loc[df["ticker"] == cash_ticker, "target_weight"] += residual
    total = float(df["target_weight"].sum())
    if total > 0: df["target_weight"] = df["target_weight"] / total
    return df

def select_candidates(metrics: pd.DataFrame, cash_ticker: str, top_k: int, max_alloc: float, hurdle_mode: str, prev_holdings: Optional[Set[str]] = None, hold_buffer_multiplier: int = DEFAULT_HOLD_BUFFER_MULTIPLIER) -> pd.DataFrame:
    df = metrics.copy()
    hurdle = benchmark_hurdle_return(df, hurdle_mode)
    df["benchmark_hurdle"] = hurdle
    df["eligible"] = False
    for idx, row in df.iterrows():
        if row["ticker"] == cash_ticker:
            df.at[idx, "eligible"] = True; continue
        ok = bool(row["above_sma200"]) and not bool(row["exit_13d"])
        if ok and pd.notna(row["ret_6m"]): ok = float(row["ret_6m"]) > hurdle
        df.at[idx, "eligible"] = ok
    score = np.zeros(len(df), dtype=float)
    score += df["above_sma200"].astype(float) * 3.0
    score += df["sma50_gt_sma200"].astype(float) * 2.5
    score += df["above_ema10"].astype(float) * 1.0
    score += (df["macd_hist"] > 0).astype(float) * 2.0
    score += (df["weekly_macd_hist"] > 0).astype(float) * 1.5
    score += df["breakout_89d"].astype(float) * 1.5
    score += df["breakout_recent"].astype(float) * 2.0
    if df["ret_6m"].notna().any():
        rs_scaled = (df["ret_6m"] - df["ret_6m"].min()) / (df["ret_6m"].max() - df["ret_6m"].min() + 1e-12)
        score += rs_scaled * 5.0
    if df["realized_vol20"].notna().any():
        inv_vol_scaled = 1.0 - ((df["realized_vol20"] - df["realized_vol20"].min()) / (df["realized_vol20"].max() - df["realized_vol20"].min() + 1e-12))
        score += inv_vol_scaled * 1.0
    score += (df["sleeve"] == "offensive").astype(float) * 0.5
    score -= df["exit_13d"].astype(float) * 4.0
    df["raw_score"] = score
    df.loc[(~df["eligible"]) & (~df["cash_like"]), "raw_score"] = 0.0
    offensive = df[(~df["cash_like"]) & (df["sleeve"] == "offensive")].sort_values("raw_score", ascending=False).copy()
    defensive = df[(~df["cash_like"]) & (df["sleeve"] == "defensive")].sort_values("raw_score", ascending=False).copy()
    offensive_k = max(1, min(top_k, math.ceil(top_k * DEFAULT_OFFENSIVE_TARGET)))
    defensive_k = max(0, top_k - offensive_k)
    selected = set(offensive.head(offensive_k)["ticker"].tolist()) | set(defensive.head(defensive_k)["ticker"].tolist())
    if prev_holdings:
        all_ranked = df[(~df["cash_like"])].sort_values("raw_score", ascending=False).copy()
        all_ranked["rank"] = np.arange(1, len(all_ranked) + 1)
        keep_threshold = top_k * hold_buffer_multiplier
        for t in prev_holdings:
            if t in all_ranked["ticker"].values:
                rank = int(all_ranked.loc[all_ranked["ticker"] == t, "rank"].iloc[0])
                score_t = float(all_ranked.loc[all_ranked["ticker"] == t, "raw_score"].iloc[0])
                if rank <= keep_threshold and score_t > 0: selected.add(t)
    ranked_selected = df[(~df["cash_like"]) & (df["ticker"].isin(selected))].sort_values("raw_score", ascending=False)
    selected = set(ranked_selected.head(top_k)["ticker"].tolist())
    df["selected"] = df["ticker"].isin(selected) | df["cash_like"]
    for idx, row in df.iterrows():
        if row["ticker"] != cash_ticker and row["raw_score"] <= 0: df.at[idx, "selected"] = False
    df["target_weight"] = 0.0
    selected_non_cash = df[df["selected"] & (~df["cash_like"])].copy()
    if selected_non_cash.empty:
        df.loc[df["ticker"] == cash_ticker, "target_weight"] = 1.0
        df["entry_label"] = np.where(df["cash_like"], "BUY NOW", "DO NOT BUY")
        df["model_target_weight"] = df["target_weight"]
        return df
    w = selected_non_cash.set_index("ticker")["raw_score"].clip(lower=0.0)
    if w.sum() <= 0:
        df.loc[df["ticker"] == cash_ticker, "target_weight"] = 1.0
        df["entry_label"] = np.where(df["cash_like"], "BUY NOW", "DO NOT BUY")
        df["model_target_weight"] = df["target_weight"]
        return df
    w = capped_normalize(w, max_cap=max_alloc, uncapped_tickers=set())
    for t, wt in w.items(): df.loc[df["ticker"] == t, "target_weight"] = float(wt)
    residual = 1.0 - float(df["target_weight"].sum())
    df.loc[df["ticker"] == cash_ticker, "target_weight"] += residual
    total = float(df["target_weight"].sum())
    df["target_weight"] = df["target_weight"] / total
    return apply_entry_labels_and_allocate(df, cash_ticker=cash_ticker, max_alloc=max_alloc)

def rationale_for_row(row: pd.Series, cash_ticker: str) -> str:
    if row["ticker"] == cash_ticker:
        return f"{cash_ticker} holds residual cash while selected ETFs wait for valid entry signals; uncapped." if row.get("target_weight",0)>0 else f"{cash_ticker} available as cash sleeve but not currently needed."
    if bool(row.get("selected", False)) and row.get("entry_label","") == "BUY NOW":
        return f"Selected and executable now. Sleeve={row.get('sleeve','')}, momentum/trend confirmed."
    if bool(row.get("selected", False)) and row.get("entry_label","") == "WAIT FOR PULLBACK":
        return f"Selected, but wait for pullback. Sleeve={row.get('sleeve','')}, trend acceptable but entry extended or not fresh."
    if bool(row.get("selected", False)) and row.get("entry_label","") == "DO NOT BUY":
        return f"Selected on ranking, but do not buy. Sleeve={row.get('sleeve','')}, current entry signal weak or invalid."
    return "Not selected or failed hurdle/filter rules."

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

def allocation_mode(universe_path, holdings_path, cash_ticker, top_k, export_prefix, start, end, hurdle_mode):
    print_progress("Loading universe and holdings")
    universe = parse_universe_file(universe_path)
    holdings = read_holdings(holdings_path)
    bad = sorted(set(holdings["ticker"]) - set(universe))
    if bad: raise ValueError(f"Holdings contain tickers not in Wealthfront universe: {bad}")
    if cash_ticker not in universe: raise ValueError(f"{cash_ticker} must exist in WealthfrontETFs.txt")
    tickers = sorted(set(universe) | {"SPY","QQQ","DIA"})
    print_progress(f"Universe loaded: {len(universe)} ETFs")
    ohlcv = download_ohlcv_history(tickers, start=start, end=end)
    print_progress("Building close-price matrix")
    close_px = get_close_series(ohlcv)
    asof = close_px.dropna(how="all").index[-1]
    print_progress(f"Computing strategy snapshot as of {asof.date()}")
    metrics = build_snapshot_metrics(universe, ohlcv, asof, cash_ticker, DEFAULT_MOMENTUM_LOOKBACK_DAYS, DEFAULT_BREAKOUT_DAYS, DEFAULT_EXIT_DAYS, DEFAULT_ATR_DAYS)
    current = derive_current_weights(holdings, close_px)
    prev_holdings = set(current.loc[current["current_weight"] > 0, "ticker"].tolist())
    print_progress("Scoring ETFs and generating target allocation")
    alloc = select_candidates(metrics, cash_ticker, top_k, DEFAULT_MAX_ALLOC, hurdle_mode, prev_holdings=prev_holdings)
    print_progress("Reconciling current holdings against target weights")
    out = alloc.merge(current, on="ticker", how="left")
    out["current_weight"] = out["current_weight"].fillna(0.0)
    out["delta_weight"] = out["target_weight"] - out["current_weight"]
    out["current_alloc_pct"] = out["current_weight"] * 100.0
    out["target_alloc_pct"] = out["target_weight"] * 100.0
    out["model_target_alloc_pct"] = out["model_target_weight"] * 100.0
    out["delta_pct_points"] = out["delta_weight"] * 100.0
    out["action"] = np.where(out["delta_weight"] > 1e-6, "INCREASE", np.where(out["delta_weight"] < -1e-6, "DECREASE", "HOLD"))
    out["rationale"] = out.apply(lambda r: rationale_for_row(r, cash_ticker), axis=1)
    action_view = out.sort_values(["target_weight","raw_score","ticker"], ascending=[False,False,True])
    metrics_cols = ["ticker","sleeve","last_close","ret_6m","benchmark_hurdle","realized_vol20","atr15","above_ema10","above_sma200","sma50_gt_sma200","macd","macd_signal","macd_hist","weekly_macd_hist","breakout_89d","breakout_recent","exit_13d","raw_score","selected","entry_label","model_target_alloc_pct","current_alloc_pct","target_alloc_pct","delta_pct_points","action","rationale"]
    action_cols = ["ticker","sleeve","entry_label","current_alloc_pct","model_target_alloc_pct","target_alloc_pct","delta_pct_points","action","rationale"]
    print_progress("Writing output files")
    action_view[metrics_cols].to_csv(f"{export_prefix}_full_metrics.csv", index=False)
    action_view[action_cols].to_csv(f"{export_prefix}_rebalance_actions.csv", index=False)
    print("="*100); print("ENHANCED ETF STRATEGY STACK"); print("="*100)
    print(f"As of: {asof.date()}"); print(f"Benchmark hurdle mode: {hurdle_mode}")
    print("Selection: concentrated, sleeve-aware, benchmark-relative, persistence-aware")
    print("Execution: only BUY NOW names receive capital; WAIT FOR PULLBACK / DO NOT BUY stay in SGOV")
    print(); print(action_view[action_cols].to_string(index=False)); print()
    print(f"Wrote: {export_prefix}_full_metrics.csv"); print(f"Wrote: {export_prefix}_rebalance_actions.csv")

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
    return {"Total Return": total_return,"CAGR": cagr,"Annual Vol": vol,"Sharpe": sharpe,"Max Drawdown": max_dd,"Calmar": calmar,"Avg Turnover/Rebalance": float(turnover.mean()) if len(turnover) else 0.0,"Num Rebalances": int((turnover > 0).sum())}

def benchmark_series(close_px: pd.DataFrame, start: str, end: str, ticker: str) -> pd.Series:
    s = close_px.loc[start:end, ticker].dropna(); eq = s / s.iloc[0]; eq.name = ticker; return eq

def run_schedule_backtest(universe, ohlcv, close_px, schedule_code, cash_ticker, top_k, start, end, hurdle_mode):
    px = close_px.loc[start:end, universe].dropna(how="all")
    if px.empty: raise ValueError(f"No price data available in range {start} to {end}")
    rebalance_dates = [d for d in px.resample(REBALANCE_MAP[schedule_code]).last().index if d in px.index]
    if len(rebalance_dates) < 2: raise ValueError(f"Not enough rebalance points for schedule {schedule_code}")
    entry_check_dates = [d for d in px.resample(REBALANCE_MAP["W"]).last().index if d in px.index]
    schedule_name = {"W":"Weekly","M":"Monthly","Q":"Quarterly"}.get(schedule_code, schedule_code)
    print_progress(f"Starting {schedule_name} backtest with {len(rebalance_dates)} portfolio rebalances and {len(entry_check_dates)} weekly entry checks")
    weights = pd.Series(0.0, index=universe); weights[cash_ticker] = 1.0
    equity = pd.Series(index=px.index, dtype=float); equity.iloc[0] = 1.0
    turnovers, weights_history, entry_label_history = [], [], []
    current_selected, pending_weights = set(), weights.copy()
    for i, dt in enumerate(px.index):
        if i == 0:
            weights_history.append({"date": dt, **weights.to_dict()}); continue
        prev_dt = px.index[i-1]
        daily_ret_vec = px.loc[dt] / px.loc[prev_dt] - 1.0
        portfolio_ret = float((weights.fillna(0.0) * daily_ret_vec.fillna(0.0)).sum())
        equity.iloc[i] = equity.iloc[i-1] * (1.0 + portfolio_ret)
        need_full_rebalance = dt in rebalance_dates
        need_entry_check = dt in entry_check_dates
        if need_full_rebalance or need_entry_check:
            metrics = build_snapshot_metrics(universe, ohlcv, dt, cash_ticker, DEFAULT_MOMENTUM_LOOKBACK_DAYS, DEFAULT_BREAKOUT_DAYS, DEFAULT_EXIT_DAYS, DEFAULT_ATR_DAYS)
            if need_full_rebalance:
                alloc = select_candidates(metrics, cash_ticker, top_k, DEFAULT_MAX_ALLOC, hurdle_mode, prev_holdings=current_selected)
                current_selected = set(alloc.loc[(alloc["selected"]) & (~alloc["cash_like"]), "ticker"].tolist())
            else:
                alloc = select_candidates(metrics, cash_ticker, top_k, DEFAULT_MAX_ALLOC, hurdle_mode, prev_holdings=current_selected)
                alloc["selected"] = alloc["ticker"].isin(current_selected) | alloc["cash_like"]
                alloc = apply_entry_labels_and_allocate(alloc, cash_ticker=cash_ticker, max_alloc=DEFAULT_MAX_ALLOC)
            new_weights = alloc.set_index("ticker")["target_weight"].reindex(universe).fillna(0.0)
            turnover = float((new_weights - weights).abs().sum() / 2.0) if need_full_rebalance else 0.0
            turnovers.append({"date": dt, "turnover": turnover})
            pending_weights = new_weights
            weights_history.append({"date": dt, **new_weights.to_dict()})
            selected_non_cash = alloc[(alloc["selected"]) & (~alloc["cash_like"])].copy()
            entry_counts = selected_non_cash["entry_label"].value_counts()
            entry_label_history.append({"date": dt,"selected_non_cash_count": int(len(selected_non_cash)),"buy_now_count": int(entry_counts.get("BUY NOW", 0)),"wait_for_pullback_count": int(entry_counts.get("WAIT FOR PULLBACK", 0)),"do_not_buy_count": int(entry_counts.get("DO NOT BUY", 0)),"sgov_weight": float(alloc.loc[alloc["ticker"] == cash_ticker, "target_weight"].sum())})
        weights = pending_weights.copy()
    equity = equity.ffill()
    return equity, pd.DataFrame(turnovers), pd.DataFrame(weights_history), pd.DataFrame(entry_label_history)

def backtest_mode(universe_path, cash_ticker, top_k, export_prefix, start, end, hurdle_mode):
    print_progress("Loading ETF universe for backtest")
    universe = parse_universe_file(universe_path)
    if cash_ticker not in universe: raise ValueError(f"{cash_ticker} must exist in WealthfrontETFs.txt")
    tickers = sorted(set(universe) | {"SPY","QQQ","DIA"})
    adjusted_start = (datetime.fromisoformat(start) - timedelta(days=400)).strftime("%Y-%m-%d")
    print_progress(f"Backtest window: {start} to {end} (using buffered download start {adjusted_start})")
    print_progress(f"Universe loaded: {len(universe)} ETFs")
    ohlcv = download_ohlcv_history(tickers, start=adjusted_start, end=end)
    print_progress("Building close-price matrix")
    close_px = get_close_series(ohlcv)
    schedules = {"Weekly":"W","Monthly":"M","Quarterly":"Q"}
    equity_curves, turnover_tables, weight_tables, entry_label_tables, summary_rows = {}, {}, {}, {}, []
    for label, code in schedules.items():
        print_progress(f"Running {label.lower()} rebalance comparison")
        eq, to_df, w_df, e_df = run_schedule_backtest(universe, ohlcv, close_px, code, cash_ticker, top_k, start, end, hurdle_mode)
        equity_curves[label], turnover_tables[label], weight_tables[label], entry_label_tables[label] = eq, to_df, w_df, e_df
        stats = metrics_from_equity_curve(eq, to_df["turnover"] if not to_df.empty else pd.Series(dtype=float))
        stats["Avg BUY NOW Count"] = float(e_df["buy_now_count"].mean()) if not e_df.empty else 0.0
        stats["Avg WAIT Count"] = float(e_df["wait_for_pullback_count"].mean()) if not e_df.empty else 0.0
        stats["Avg DO NOT BUY Count"] = float(e_df["do_not_buy_count"].mean()) if not e_df.empty else 0.0
        stats["Avg SGOV Weight"] = float(e_df["sgov_weight"].mean()) if not e_df.empty else 0.0
        stats["Schedule"] = label
        summary_rows.append(stats)
    for b in ["QQQ","SPY","DIA"]:
        if b in close_px.columns:
            eq = benchmark_series(close_px, start, end, b)
            stats = metrics_from_equity_curve(eq, pd.Series(dtype=float))
            stats.update({"Schedule": b, "Avg BUY NOW Count": np.nan, "Avg WAIT Count": np.nan, "Avg DO NOT BUY Count": np.nan, "Avg SGOV Weight": np.nan})
            summary_rows.append(stats); equity_curves[b] = eq
    summary = pd.DataFrame(summary_rows)[["Schedule","Total Return","CAGR","Annual Vol","Sharpe","Max Drawdown","Calmar","Avg Turnover/Rebalance","Num Rebalances","Avg BUY NOW Count","Avg WAIT Count","Avg DO NOT BUY Count","Avg SGOV Weight"]].sort_values("CAGR", ascending=False)
    equity_df = pd.DataFrame(equity_curves)
    print_progress("Writing backtest output files")
    summary.to_csv(f"{export_prefix}_backtest_summary.csv", index=False)
    equity_df.to_csv(f"{export_prefix}_equity_curves.csv", index=True)
    for label, df in turnover_tables.items(): df.to_csv(f"{export_prefix}_{label.lower()}_turnover.csv", index=False)
    for label, df in weight_tables.items(): df.to_csv(f"{export_prefix}_{label.lower()}_weights_history.csv", index=False)
    for label, df in entry_label_tables.items(): df.to_csv(f"{export_prefix}_{label.lower()}_entry_labels.csv", index=False)
    print("="*126); print("BACKTEST SUMMARY: WEEKLY vs MONTHLY vs QUARTERLY vs BENCHMARKS (SPY, QQQ, DIA)"); print("="*126)
    display = summary.copy()
    for c in ["Total Return","CAGR","Annual Vol","Max Drawdown","Avg Turnover/Rebalance","Avg SGOV Weight"]:
        display[c] = display[c].map(lambda x: f"{x:.2%}" if isinstance(x, (float, np.floating)) and pd.notna(x) else x)
    display["Sharpe"] = summary["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    display["Calmar"] = summary["Calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    for c in ["Avg BUY NOW Count","Avg WAIT Count","Avg DO NOT BUY Count"]:
        display[c] = summary[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    print(display.to_string(index=False)); print()
    print(f"Wrote: {export_prefix}_backtest_summary.csv"); print(f"Wrote: {export_prefix}_equity_curves.csv"); print("Wrote schedule turnover, weights-history, and entry-label diagnostic CSVs")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["allocate","backtest"], required=True)
    p.add_argument("--universe", required=True)
    p.add_argument("--holdings")
    p.add_argument("--cash-ticker", default=DEFAULT_CASH_TICKER)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--export-prefix", default="wealthfront_v2")
    p.add_argument("--benchmark-hurdle", default=DEFAULT_BENCHMARK_HURDLE, choices=["none","spy","qqq","dia","best_of_3"])
    return p

def main():
    args = build_parser().parse_args()
    if args.mode == "allocate":
        if not args.holdings: raise SystemExit("--holdings is required for --mode allocate")
        allocation_mode(args.universe, args.holdings, args.cash_ticker.upper(), args.top_k, args.export_prefix, args.start, args.end, args.benchmark_hurdle)
    else:
        if not args.start or not args.end: raise SystemExit("--start and --end are required for --mode backtest")
        backtest_mode(args.universe, args.cash_ticker.upper(), args.top_k, args.export_prefix, args.start, args.end, args.benchmark_hurdle)

if __name__ == "__main__":
    main()
