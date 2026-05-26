# Architecture

## Overview

The engine has four distinct layers. Each layer has a single responsibility
and communicates with adjacent layers through typed dataclasses. No layer
imports from a layer above it.

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLI / orchestration  (main.py, multi_asset.py)                     │
│  User-facing entry points. Parses arguments, wires together the     │
│  layers below, and calls build_dashboard().                         │
├─────────────────────────────────────────────────────────────────────┤
│  Walk-forward framework  (walk_forward.py)                          │
│  Splits data into rolling train/test windows, calls strategy.fit()  │
│  on each training period, runs the simulation on each test period,  │
│  aggregates per-window MetricsResult into a BacktestResult.         │
├─────────────────────────────────────────────────────────────────────┤
│  Strategies  (strategy/)            Execution  (execution.py)       │
│  BaseStrategy: fit() + signals()    ExecutionConfig: cost, slippage │
│  MovingAverageStrategy (grid)       delay. run_simulation_with_     │
│  KalmanFilterStrategy (MLE)         execution() fills at close      │
│  MomentumStrategy (grid)            ± slippage_factor × range.      │
├─────────────────────────────────────────────────────────────────────┤
│  Data  (data/)                    Statistical testing (metrics.py,  │
│  ingestion.py: yfinance + cache   reality_check.py, benchmark.py)   │
│  validator.py: schema + range     Sharpe, Sortino, Omega, Calmar,   │
│  checks, adjusted H/L repair      block bootstrap p, White's RC.    │
└─────────────────────────────────────────────────────────────────────┘
```

## Data flow through a single walk-forward window

```
load_data() → validate_data()
     │
     ▼
walk_forward()
  ├─ slice train data
  ├─ strategy.fit(train_data)          # calibrate parameters
  │       └─ for MA: grid search over (short, long) pairs
  │       └─ for Kalman: Nelder-Mead MLE of (Q, R) in log-space
  │       └─ for Momentum: grid search over lookback periods
  ├─ strategy.generate_signals_with_context() # signals with context warmup
  ├─ run_simulation_with_execution()   # fills with slippage, delay, costs
  │       └─ returns SimulationResult (trades, portfolio_values)
  ├─ calculate_metrics()               # Sharpe, Sortino, max DD, p-value
  │       └─ _monte_carlo_p_value()    # block bootstrap
  ├─ candidate_test_returns()          # all candidates (for RC)
  └─ WindowResult(train_start, train_end, test_start, test_end,
                  simulation_result, metrics_result, active_params, ...)

     │ (repeat for each window)
     ▼

_build_summary_metrics()
  ├─ Fisher combined p (all per-window p-values)
  ├─ white_reality_check() over all windows' candidate matrices
  ├─ mean Sharpe / Sortino / Omega / trade count across windows
  ├─ min max_drawdown (worst window, not mean)
  └─ _calmar() on stitched return series

BacktestResult → compute_benchmark() → build_dashboard()
```

## Key design decisions

### Walk-forward, not in-sample

In-sample backtesting calibrates and tests on the same data. It reliably
over-fits. Walk-forward calibrates on a training period then tests on held-out
data. The test windows are stitched together to produce a performance curve
that reflects what a live trader would have seen.

### Why Fisher combination, not a single p-value

Each test window produces one p-value. The p-values are combined using
Fisher's method (−2 Σ ln pᵢ ~ χ²(2k)). This tests the joint null that
all windows show no skill, giving power even when individual windows are
underpowered. Fisher's method is appropriate when the test windows are
approximately independent (they don't overlap).

### Why White's Reality Check

Standard backtesting picks the best-performing parameter set and reports its
Sharpe. This is data snooping: you haven't tested one strategy, you've tested
the whole grid and reported the winner. White's RC corrects for this: the null
is that no strategy in the candidate universe has positive expected performance
relative to the benchmark (cash), after accounting for the search across all
candidates. A small RC p-value means the best strategy is
significantly better than chance even after the multiple-comparison correction.

This is also why Sharpe-based p-value and RC p-value can diverge: the former
tests whether the selected strategy beats zero; the latter tests whether the
best candidate beats zero after correcting for the number of candidates tried.

### Why per-window Fisher AND RC (not just one)

Fisher p and RC p test different things:
- Fisher p: "did the selected strategy beat cash, across all windows jointly?"
- RC p: "is the selected strategy better than its competitors, after
  multiple-comparison correction?"

A strategy can have Fisher p < 0.05 but RC p > 0.05: it beats cash but not
the rest of the grid. That is the most common failure mode: a strategy that
looks significant on its own is actually just one of many that happen to work
on SPY in a bull market.

Both statistics use the same set of windows (including flat-cash windows, which
contribute p=1.0 to Fisher and zero-return arrays to the RC candidate matrix).
This parity is required for the two statistics to be comparable.

### Block bootstrap, not i.i.d. bootstrap

Financial returns are autocorrelated. An i.i.d. bootstrap destroys this
structure and produces p-values that are too small (underestimates variance).
The stationary block bootstrap (Politis & Romano 1994) resamples contiguous
blocks of returns, preserving autocorrelation up to the block length. Block
length is set to √n (the conventional heuristic). The optimal block length
from Politis & White (2004) would be slightly better but adds ~100 lines
of spectral density estimation.

### Why adjust high/low for dividends

The yfinance `Adj Close` is dividend- and split-adjusted but `High` and `Low`
are raw. This creates a problem: on ex-dividend dates, the adjusted close can
sit below the unadjusted low, making `fill = close + slippage × (high - low)`
nonsensical. We apply the adjustment factor to high and low as well:

```
factor = Adj Close / Close
adj_high = High × factor
adj_low  = Low  × factor
```

Small remaining discrepancies (< 0.5%) are clipped. Larger ones raise, since
they indicate data errors rather than rounding.

### Flat-cash window semantics

When a strategy generates no signals in a test window, the portfolio holds
cash and earns zero return. This is a valid result, not an error to skip.
Skipping it would cause Fisher combination and the RC matrix to test different
hypotheses (different number of windows), breaking the p-value comparison.

Flat-cash windows contribute:
- `p_value = 1.0` to Fisher (no evidence of skill)
- `sharpe = 0.0` to the mean Sharpe (not NaN, which would inflate the mean)
- `sortino = 0.0` (not ∞, which would be silently excluded from nanmean)
- `omega = 1.0` (not ∞, same reason)
- A zero-return array to the RC candidate matrix (not absent, for parity)

### Why no leverage, no short-selling

This is an equity trend-following backtest, not a long-short hedge fund.
Adding leverage would obscure whether the strategy has skill or just amplifies
returns. Short-selling would require a separate short constraint model (locate
costs, borrow fees). The existing framework is intentionally limited to long-only
to keep the risk model simple and the results interpretable.

### Execution model: close-of-day fills

Signals generated at bar t are filled at bar t+1's close (with `signal_delay=1`,
the default). This is realistic for daily equity strategies: a signal computed
from day t data is acted on the next trading day. `signal_delay=0` fills at
the signal bar's close and is available for comparison but represents an
optimistic upper bound (assumes perfect same-day execution).

## What each module owns

| Module | Owns | Does NOT own |
|--------|------|--------------|
| `config.py` | All numeric constants | Logic |
| `models.py` | All data container types | Business logic |
| `data/` | Download, clean, cache, validate | Strategy or simulation |
| `strategy/` | Signal generation, parameter calibration | Execution, metrics |
| `execution.py` | Order filling, slippage, cost, delay | Strategy logic |
| `metrics.py` | All performance metrics, bootstrap p | Walk-forward orchestration |
| `reality_check.py` | White's RC only | Any other statistic |
| `benchmark.py` | B&H comparison, IR, paired t-test | Strategy p-values |
| `walk_forward.py` | Window orchestration, Fisher combination | Metric formulas |
| `dashboard.py` | Rendering only | Metric computation |
| `main.py` | CLI wiring | Any computation |
| `multi_asset.py` | Cross-asset loop, comparison table, graceful ticker failure | Per-ticker metric computation (delegates to walk_forward + benchmark) |

## Test organisation

Tests mirror the module structure. One test file per source module.
Shared infrastructure lives in `conftest.py` (fixtures) and `helpers.py`
(pure data-generation functions).

`conftest.py` does two critical things:
1. Patches `N_PERMUTATIONS` to 200 for the full test session (production = 10,000).
2. Provides module-scoped cached `wf_result_504` and `wf_result_756` fixtures
   so walk_forward() runs at most twice per test module, not once per test.

See [docs/performance.md](performance.md) for why this matters for CI runtime.
