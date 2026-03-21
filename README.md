# strategy_stack_no_regime
python strategy_stack_no_regime.py   --mode allocate   --universe "./WealthfrontETFs.txt"   --holdings "./holdings.csv"   --export-prefix wealthfront_no_regime --start=2025-01-01

python strategy_stack_no_regime.py   --mode backtest   --universe "./WealthfrontETFs.txt"   --start 2026-01-01   --end 2026-03-01   --top-k 10   --export-prefix wealthfront_no_regime_bt