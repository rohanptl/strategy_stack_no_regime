# strategy_stack_no_regime
python strategy_stack_no_regime.py   --mode allocate   --universe "./WealthfrontETFs.txt"   --holdings "./holdings.csv"   --export-prefix wealthfront_live

python strategy_stack_no_regime.py   --mode backtest   --universe "./WealthfrontETFs.txt"   --start 2026-01-01   --end 2026-03-21   --top-k 10 --export-prefix wealthfront_bt