#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot a schedule equity curve with ETF price lines highlighted only when positions are materially held."
    )
    p.add_argument("--equity-csv", required=True, help="Path to equity curves CSV")
    p.add_argument("--weights-csv", required=True, help="Path to schedule weights history CSV")
    p.add_argument("--schedule", required=True, help="Column name in equity CSV, e.g. Weekly or Biweekly")
    p.add_argument("--price-cache-dir", required=True, help="Directory containing cached parquet OHLCV files")
    p.add_argument("--output", required=True, help="Output PNG path")
    p.add_argument("--start-date", help="Optional chart start date YYYY-MM-DD")
    p.add_argument("--end-date", help="Optional chart end date YYYY-MM-DD")
    p.add_argument(
        "--annotate-top-n",
        type=int,
        default=0,
        help="Number of holdings to show in each annotation box. Use 0 to show all holdings above --min-weight.",
    )
    p.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
        help="Minimum portfolio weight to consider a position highlighted in an interval (default: 0.05 = 5%%).",
    )
    p.add_argument(
        "--include-cash",
        action="store_true",
        help="Include SGOV in highlighted ETF price lines.",
    )
    return p.parse_args()


def _apply_date_window(
    frame: pd.DataFrame | pd.Series,
    start_date: Optional[str],
    end_date: Optional[str],
    date_col: Optional[str] = None,
) -> pd.DataFrame | pd.Series:
    out = frame
    if date_col is not None:
        if start_date:
            out = out[out[date_col] >= pd.Timestamp(start_date)]
        if end_date:
            out = out[out[date_col] <= pd.Timestamp(end_date)]
        return out

    if start_date:
        out = out.loc[out.index >= pd.Timestamp(start_date)]
    if end_date:
        out = out.loc[out.index <= pd.Timestamp(end_date)]
    return out


def load_equity_curve(path: Path, schedule: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"Date": "date"}).sort_values("date")
    if schedule not in df.columns:
        raise ValueError(f"Schedule '{schedule}' not found in {path}")
    return df.set_index("date")[schedule].astype(float)


def load_weights(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


def pick_latest_cache_file(cache_dir: Path, ticker: str) -> Optional[Path]:
    candidates = sorted(cache_dir.glob(f"{ticker.upper()}__*.parquet"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def load_close_series(cache_dir: Path, tickers: List[str]) -> pd.DataFrame:
    closes: Dict[str, pd.Series] = {}
    for ticker in tickers:
        parquet_path = pick_latest_cache_file(cache_dir, ticker)
        if parquet_path is None:
            continue
        try:
            df = pd.read_parquet(parquet_path)
            if "Close" not in df.columns:
                continue
            series = df["Close"].copy()
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)
            closes[ticker] = series.sort_index()
        except Exception:
            continue
    if not closes:
        raise ValueError("No cached close series could be loaded.")
    return pd.DataFrame(closes).sort_index()


def build_active_segments(
    weights: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    min_weight: float,
    include_cash: bool,
) -> Dict[str, List[tuple[pd.Timestamp, pd.Timestamp, float]]]:
    tickers = [c for c in weights.columns if c != "date"]
    if not include_cash:
        tickers = [t for t in tickers if t != "SGOV"]

    segments: Dict[str, List[tuple[pd.Timestamp, pd.Timestamp, float]]] = {
        ticker: []
        for ticker in tickers
    }

    rows = weights.reset_index(drop=True)
    for i, row in rows.iterrows():
        start = row["date"]
        end = rows.loc[i + 1, "date"] if i + 1 < len(rows) else all_dates.max() + pd.Timedelta(days=1)
        for ticker in tickers:
            weight = float(row.get(ticker, 0.0))
            if weight >= min_weight:
                segments[ticker].append((start, end, weight))

    return segments


def _build_annotation_text(
    row: pd.Series,
    min_weight: float,
    include_cash: bool,
    annotate_top_n: int,
) -> Optional[str]:
    weights = row.drop(labels=["date"]).astype(float)
    if not include_cash:
        weights = weights.drop(labels=["SGOV"], errors="ignore")
    active = weights[weights >= min_weight].sort_values(ascending=False)
    if annotate_top_n > 0:
        active = active.head(annotate_top_n)
    if active.empty:
        return None
    body = "\n".join(f"{ticker} {weight:.1%}" for ticker, weight in active.items())
    return f"{row['date'].strftime('%Y-%m-%d')}\n{body}"


def plot_schedule_chart(
    equity: pd.Series,
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    schedule: str,
    output_path: Path,
    min_weight: float,
    include_cash: bool,
    annotate_top_n: int,
) -> None:
    common_dates = equity.index.intersection(closes.index)
    equity = equity.reindex(common_dates).dropna()
    closes = closes.reindex(common_dates).ffill()
    weights = weights[(weights["date"] >= common_dates.min()) & (weights["date"] <= common_dates.max())].copy()
    active_segments = build_active_segments(weights, common_dates, min_weight=min_weight, include_cash=include_cash)

    fig, (ax_eq, ax_px) = plt.subplots(
        2, 1,
        figsize=(20, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.6]},
    )

    portfolio_index = equity / float(equity.iloc[0]) * 100.0
    ax_eq.plot(portfolio_index.index, portfolio_index.values, color="black", linewidth=2.2, label=f"{schedule} portfolio")
    ax_eq.set_ylabel("Portfolio Value\n(start=100)")
    ax_eq.set_title(
        f"{schedule} Portfolio And Highlighted Holdings\n"
        f"ETF lines appear only during intervals where weight >= {min_weight:.0%}"
    )
    ax_eq.grid(True, alpha=0.25)
    ax_eq.legend(loc="upper left")

    event_dates = pd.to_datetime(weights["date"]).drop_duplicates().sort_values()
    text_transform = blended_transform_factory(ax_eq.transData, ax_eq.transAxes)
    annotation_levels = [0.98, 0.80, 0.62]
    for event_date in event_dates:
        ax_eq.axvline(event_date, color="#b8b8b8", linestyle=":", linewidth=0.8, alpha=0.45, zorder=0)
        ax_px.axvline(event_date, color="#b8b8b8", linestyle=":", linewidth=0.8, alpha=0.45, zorder=0)
    for idx, row in weights.reset_index(drop=True).iterrows():
        text = _build_annotation_text(row, min_weight=min_weight, include_cash=include_cash, annotate_top_n=annotate_top_n)
        if not text:
            continue
        ax_eq.text(
            row["date"],
            annotation_levels[idx % len(annotation_levels)],
            text,
            transform=text_transform,
            fontsize=6.2,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#c0c0c0", alpha=0.78),
        )

    plotted_labels = set()
    for ticker, segments in active_segments.items():
        if ticker not in closes.columns:
            continue
        for start, end, _weight in segments:
            segment = closes.loc[(closes.index >= start) & (closes.index < end), ticker].dropna()
            if segment.empty:
                continue
            segment_return = (segment / float(segment.iloc[0]) - 1.0) * 100.0
            label = ticker if ticker not in plotted_labels else None
            ax_px.plot(segment_return.index, segment_return.values, linewidth=1.6, alpha=0.95, label=label)
            plotted_labels.add(ticker)

    ax_px.axhline(0.0, color="#808080", linewidth=1.0, alpha=0.5)
    ax_px.set_ylabel("ETF Return Since\nInterval Start (%)")
    ax_px.set_xlabel("Trade Date")
    ax_px.grid(True, alpha=0.25)
    if plotted_labels:
        ax_px.legend(loc="upper left", ncol=6, fontsize=8)

    num_ticks = min(6, len(common_dates))
    tick_idx = np.unique(np.linspace(0, len(common_dates) - 1, num=num_ticks, dtype=int))
    tick_dates = common_dates[tick_idx]
    ax_px.set_xticks(tick_dates)
    ax_px.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax_px.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    equity = load_equity_curve(Path(args.equity_csv), args.schedule)
    weights = load_weights(Path(args.weights_csv))
    equity = _apply_date_window(equity, args.start_date, args.end_date)
    weights = _apply_date_window(weights, args.start_date, args.end_date, date_col="date")
    tickers = [c for c in weights.columns if c != "date"]
    if not args.include_cash:
        tickers = [t for t in tickers if t != "SGOV"]
    closes = load_close_series(Path(args.price_cache_dir), tickers)
    plot_schedule_chart(
        equity=equity,
        weights=weights,
        closes=closes,
        schedule=args.schedule,
        output_path=Path(args.output),
        min_weight=args.min_weight,
        include_cash=args.include_cash,
        annotate_top_n=args.annotate_top_n,
    )


if __name__ == "__main__":
    main()
