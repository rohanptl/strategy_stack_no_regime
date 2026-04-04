# strategy_stack_no_regime
python strategy_stack_no_regime.py   --mode allocate   --universe "./WealthfrontETFs.txt"   --holdings "./holdings.csv"   --top-k 5   --hold-band 7   --export-prefix wealthfront_v2_alloc --allocation-mode momentum_proportional --execution-mode overlay

python strategy_stack_no_regime.py   --mode backtest   --universe "./WealthfrontETFs.txt"   --top-k 5  --hold-band 7   --start 2023-01-01  --end 2026-03-22  --export-prefix wealthfront_v2  --allocation-mode momentum_proportional --execution-mode overlay

--benchmark-hurdle spy

Example 1
python strategy_stack_no_regime.py \
  --mode allocate \
  --universe "./WealthfrontETFs.txt" \
  --holdings "./holdings.csv" \
  --top-k 5 \
  --allocation-mode equal \
  --execution-mode pure_topk

Meaning:

pick top 5
equal weight them
ignore entry labels
only use SGOV if fewer than 5 qualify or caps leave residual
Example 2
python strategy_stack_no_regime.py \
  --mode allocate \
  --universe "./WealthfrontETFs.txt" \
  --holdings "./holdings.csv" \
  --top-k 5 \
  --allocation-mode momentum_proportional \
  --execution-mode overlay

Meaning:

pick top 5
weight by 6-month momentum
then reduce or zero weights using BUY NOW / WAIT / DO NOT BUY
residual to SGOV

options:
  -h, --help            show this help message and exit
  --mode {allocate,backtest}
  --universe UNIVERSE   Path to universe file (one ticker per line)
  --holdings HOLDINGS   Path to holdings CSV (required for --mode allocate)
  --cash-ticker CASH_TICKER
  --top-k TOP_K
  --start START         Start date YYYY-MM-DD
  --end END             End date YYYY-MM-DD
  --export-prefix EXPORT_PREFIX
  --benchmark-hurdle {none,spy,qqq,dia,best_of_3}
  --hold-band HOLD_BAND
                        Rank threshold for retaining existing holdings. Default: top-k + 2.
  --transaction-cost-bps TRANSACTION_COST_BPS
                        One-way transaction cost in basis points on trade dates (default: 0).
  --allocation-mode {equal,score_proportional,momentum_proportional}
                        How selected ETFs are weighted before residual goes to cash.
  --execution-mode {overlay,pure_topk}
                        overlay = obey BUY NOW / WAIT / DO NOT BUY labels; pure_topk = ignore entry overlay and use model weights directly.



--------------------------------------
BACK TEST
--------------------------------------
How to use it
First grid run with one-time shared download
python grid_backtest_strategy_stack.py \
  --universe "./WealthfrontETFs.txt" \
  --strategy-script "./strategy_stack_no_regime.py" \
  --end "2026-04-03" \
  --price-cache-dir "./price_cache" \
  --prefetch-history

That will:

download once into ./price_cache
then every combination reuses local parquet files
Force a fresh redownload
python grid_backtest_strategy_stack.py \
  --universe "./WealthfrontETFs.txt" \
  --strategy-script "./strategy_stack_no_regime.py" \
  --end "2026-04-03" \
  --price-cache-dir "./price_cache" \
  --prefetch-history \
  --refresh-cache
Result

This gives you:

one shared OHLCV cache
much faster grid runs
lower yfinance rate-limit risk
no need to re-download identical history for every parameter combination
