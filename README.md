# strategy_stack_no_regime

ETF allocation and backtest toolkit for a fixed ETF universe such as [WealthfrontETFs.txt](/c:/Rohan/strategy_stack_no_regime/WealthfrontETFs.txt).

There are two main scripts:

- [strategy_stack_no_regime.py](/c:/Rohan/strategy_stack_no_regime/strategy_stack_no_regime.py)
  Current allocation run or multi-schedule backtest run.
- [grid_backtest_strategy_stack.py](/c:/Rohan/strategy_stack_no_regime/grid_backtest_strategy_stack.py)
  Parameter sweep runner for repeated backtests.

## Setup

Create and activate the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Main Strategy Script

### Allocation Mode

Use this for the current recommended allocation versus an existing holdings file.

Example:

```powershell
python strategy_stack_no_regime.py --mode allocate --universe WealthfrontETFs.txt --holdings holdings.csv --top-k 5 --hold-band 7 --max-alloc 0.35 --allocation-mode score_proportional --execution-mode overlay --max-wait-pullback 1 --price-cache-dir price_cache --refresh-cache
```

Useful output files:

- `etf_strategy_full_metrics.csv`
- `etf_strategy_rebalance_actions.csv`

What the terminal shows:

- step-by-step progress
- snapshot date used
- how many ETFs were selected
- counts of `BUY NOW`, `WAIT FOR PULLBACK`, `DO NOT BUY`
- recommended rebalance actions

### Backtest Mode

Use this to compare `Weekly`, `Biweekly`, `Monthly`, and `Quarterly` schedules over one date range.

Example:

```powershell
python strategy_stack_no_regime.py --mode backtest --universe WealthfrontETFs.txt --top-k 5 --hold-band 7 --max-alloc 0.35 --allocation-mode score_proportional --execution-mode overlay --max-wait-pullback 1 --start 2023-01-01 --end 2026-04-02 --export-prefix wealthfront_v3
```

Useful output files:

- `wealthfront_v3_backtest_summary.csv`
- `wealthfront_v3_equity_curves.csv`
- `wealthfront_v3_weekly_weights_history.csv`
- `wealthfront_v3_weekly_turnover.csv`
- `wealthfront_v3_weekly_entry_labels.csv`
- similar files for biweekly, monthly, and quarterly

## Grid Backtest Script

Use this when you want to sweep parameters across many start dates and compare results in one run.

### Minimal Weekly Run

```powershell
python grid_backtest_strategy_stack.py --universe WealthfrontETFs.txt --schedules weekly
```

### Recommended Weekly Date-Sweep Run

```powershell
python grid_backtest_strategy_stack.py --universe WealthfrontETFs.txt --top-ks 5 --allocation-modes score_proportional --execution-modes overlay --hold-band 7 --max-alloc 0.35 --schedules weekly --end 2026-04-02 --output-dir backtest_output
```

### Parameter Search Example

```powershell
python grid_backtest_strategy_stack.py --universe WealthfrontETFs.txt --top-ks 3 4 5 6 7 --hold-bands 5 6 7 8 9 --max-wait-pullbacks 1 2 3 4 --allocation-modes score_proportional --execution-modes overlay --schedules weekly --max-alloc 0.35 --end 2026-04-02 --output-dir backtest_output_param_search
```

Latest reference result from the last sweep in `backtest_output_param_search`:

- best balanced combination:
  `top_k=5`, `hold_band=7`, `max_wait_pullback=1`
- best median CAGR combination:
  `top_k=3`, `hold_band=6`, `max_wait_pullback=2`

Reference metrics aggregated across the 15 start dates in that sweep:

| Goal | top_k | hold_band | max_wait_pullback | Median CAGR | Median Sharpe | Median Max Drawdown | Median Calmar | Median Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best balanced | 5 | 7 | 1 | 22.16% | 1.425 | -13.03% | 1.598 | 67.94% |
| Best CAGR | 3 | 6 | 2 | 23.91% | 1.229 | -15.96% | 1.402 | 68.19% |

Useful output files:

- `weekly_grid_combined_results.csv`
- `weekly_grid_top10.csv`
- optional per-run equity curves if `--save-equity-curves` is used

## Important Defaults

Defaults in [strategy_stack_no_regime.py](/c:/Rohan/strategy_stack_no_regime/strategy_stack_no_regime.py):

- `cash_ticker = SGOV`
- `top_k = 5`
- `hold_band = top_k + 2` when not provided explicitly
- `max_alloc = 1.0`
- `allocation_mode = score_proportional`
- `execution_mode = overlay`
- `max_wait_pullback = 3`
- `transaction_cost_bps = 0`
- `price_cache_dir = None`
- `refresh_cache = False`

Allocation mode default history window:

- if `--start` is omitted in allocation mode, the script uses about `today - 550 days`
- if `--end` is omitted, it uses the latest available data

Defaults in [grid_backtest_strategy_stack.py](/c:/Rohan/strategy_stack_no_regime/grid_backtest_strategy_stack.py):

- default start-date sweep is `DEFAULT_START_DATES`
- default `top_ks = [3, 4, 5, 6]`
- default `allocation_modes = [score_proportional]`
- default `execution_modes = [overlay, pure_topk]`
- default `schedules = [weekly, biweekly]`
- default `price_cache_dir = price_cache`

## Key Concepts

### `top-k`

Target number of non-cash ETFs the strategy tries to hold.

### `hold-band`

Existing holdings can stay even after slipping out of `top-k`, as long as they remain within the wider keep threshold.

Example:

- `top-k = 5`
- `hold-band = 7`

New entrants must break into the top 5, but an existing holding can remain until it falls below rank 7.

### `allocation-mode`

Controls how selected ETFs are weighted before entry overlay and before residual cash goes to `SGOV`:

- `equal`
  every selected non-cash ETF starts with the same model weight
- `score_proportional`
  selected ETFs get larger model weights when their `raw_score` is higher
- `momentum_proportional`
  selected ETFs get larger model weights when their positive 6-month momentum is higher

Important:

- `allocation-mode` does not decide which ETFs are selected
- it only decides how weight is split across the selected ETFs
- after this step, `max_alloc` is applied
- then `execution-mode` may reduce or zero some of those model weights
- any leftover weight goes to `SGOV`

### `execution-mode`

- `overlay`
  applies entry labels after model weights are built
- `pure_topk`
  ignores entry labels and uses model weights directly

In `overlay` mode:

- `BUY NOW`
  keeps the full model weight
- `WAIT FOR PULLBACK`
  gets only a partial allocation
- `DO NOT BUY`
  gets zero allocation

This means an ETF can be selected by ranking but still end up with little or no final allocation if the entry signal is not currently favorable.

### `max-wait-pullback`

In `overlay` mode, this is the maximum number of `WAIT FOR PULLBACK` ETFs that may receive partial allocation.

Example:

- if `max-wait-pullback = 1`, only the top `WAIT FOR PULLBACK` ETF gets partial weight
- if `max-wait-pullback = 3`, up to three `WAIT FOR PULLBACK` ETFs may get partial weight
- any additional `WAIT FOR PULLBACK` ETFs get zero final allocation

Practical effect:

- lower values make the strategy more conservative and push more residual weight into `SGOV`
- higher values allow more partial deployment even when entry timing is not ideal

## Cache Usage

If you want fast reruns, use a populated cache directory.

First run or forced refresh:

```powershell
python strategy_stack_no_regime.py --mode allocate --universe WealthfrontETFs.txt --holdings holdings.csv --price-cache-dir price_cache --refresh-cache
```

Later reruns:

```powershell
python strategy_stack_no_regime.py --mode allocate --universe WealthfrontETFs.txt --holdings holdings.csv --price-cache-dir price_cache
```

Meaning:

- `--price-cache-dir price_cache`
  stores and reuses downloaded OHLCV history
- `--refresh-cache`
  forces a fresh download even if cache files exist

## Output Interpretation

### Backtest Metrics

- `CAGR`
  Compound annual growth rate
- `Sharpe`
  Return relative to volatility
- `Max Drawdown`
  Worst peak-to-trough decline
- `Calmar`
  `CAGR / |Max Drawdown|`
- `Avg Turnover/Trade Date`
  Average portfolio reshuffle size on dates where trades occurred
- `Num Trade Dates`
  Count of dates with non-zero turnover

### Allocation Outputs

In `etf_strategy_full_metrics.csv`:

- `raw_score`
  ranking score used for selection
- `selected`
  whether the ETF made the selected set
- `entry_label`
  `BUY NOW`, `WAIT FOR PULLBACK`, or `DO NOT BUY`
- `model_target_alloc_pct`
  pre-overlay model allocation
- `target_alloc_pct`
  final allocation after overlay and residual cash
- `rationale`
  short explanation of why the ETF is selected, blocked, or not selected

In `etf_strategy_rebalance_actions.csv`:

- `current_alloc_pct`
- `target_alloc_pct`
- `delta_pct_points`
- `action`

This is the leaner file to use for actual rebalance decisions.

## Current CLI Reference

### `strategy_stack_no_regime.py`

```text
--mode {allocate,backtest}
--universe UNIVERSE
--holdings HOLDINGS
--cash-ticker CASH_TICKER
--top-k TOP_K
--start START
--end END
--export-prefix EXPORT_PREFIX
--hold-band HOLD_BAND
--max-alloc MAX_ALLOC
--transaction-cost-bps TRANSACTION_COST_BPS
--allocation-mode {equal,score_proportional,momentum_proportional}
--execution-mode {overlay,pure_topk}
--max-wait-pullback MAX_WAIT_PULLBACK
--price-cache-dir PRICE_CACHE_DIR
--refresh-cache
```

### `grid_backtest_strategy_stack.py`

```text
--universe UNIVERSE
--cash-ticker CASH_TICKER
--max-alloc MAX_ALLOC
--hold-band HOLD_BAND
--hold-bands [HOLD_BANDS ...]
--transaction-cost-bps TRANSACTION_COST_BPS
--max-wait-pullback MAX_WAIT_PULLBACK
--max-wait-pullbacks [MAX_WAIT_PULLBACKS ...]
--price-cache-dir PRICE_CACHE_DIR
--refresh-cache
--start-dates [START_DATES ...]
--top-ks [TOP_KS ...]
--allocation-modes [{equal,score_proportional,momentum_proportional} ...]
--execution-modes [{overlay,pure_topk} ...]
--schedules [{weekly,biweekly} ...]
--end END
--output-dir OUTPUT_DIR
--save-equity-curves
--clean-output-dir
--clean-cache-dir
```
