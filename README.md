# strategy_stack_no_regime
python strategy_stack_no_regime.py --mode backtest  --universe "./WealthfrontETFs.txt"  --start 2024-01-01  --end 2026-03-21  --top-k 5  --benchmark-hurdle spy  --export-prefix wealthfront_v2

python strategy_stack_no_regime.py  --mode allocate  --universe "./WealthfrontETFs.txt"  --holdings "./holdings.csv"  --top-k 5  --export-prefix wealthfront_v2_alloc

--benchmark-hurdle spy