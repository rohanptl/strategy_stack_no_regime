# strategy_stack_no_regime
python strategy_stack_no_regime.py   --mode allocate   --universe "./WealthfrontETFs.txt"   --holdings "./holdings.csv"   --top-k 5   --hold-band 7   --export-prefix wealthfront_v2_alloc

python strategy_stack_no_regime.py   --mode backtest   --universe "./WealthfrontETFs.txt"   --top-k 5  --hold-band 7   --start 2023-01-01  --end 2026-03-22  --export-prefix wealthfront_v2

--benchmark-hurdle spy