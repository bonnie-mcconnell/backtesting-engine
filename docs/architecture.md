# Architecture

## Overview

Four layers, each with a single responsibility, communicating through typed dataclasses.
No layer imports from a layer above it.

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLI / orchestration  (main.py, multi_asset.py)                     │
│  Parses arguments, wires the layers together, calls build_dashboard │
├─────────────────────────────────────────────────────────────────────┤
│  Walk-forward framework  (walk_forward.py)                          │
│  Splits data into rolling train/test windows, calls strategy.fit()  │
│  on each training period, runs simulation, aggregates results.      │
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
  ├─ strategy.fit(train_data)
  │       └─ MA: grid search over (short, long) pairs
  │       └─ Kalman: Nelder-Mead MLE of (Q, R) in log-space
  │       └─ Momentum: grid search over lookback periods
  ├─ strategy.generate_signals_with_context()
  ├─ run_simulation_with_execution()   → SimulationResult (trades, portfolio_values)
  ├─ calculate_metrics()               → Sharpe, Sortino, max DD, p-value
  ├─ candidate_test_returns()          → all candidates (for RC)
  └─ WindowResult(train_start, train_end, test_start, test_end, ...)

     │ (repeat for each window)
     ▼

_build_summary_metrics()
  ├─ Fisher combined p across all windows
  ├─ white_reality_check() over all windows' candidate matrices
  ├─ mean Sharpe / Sortino / Omega across windows
  ├─ worst-case max_drawdown (not mean)
  └─ Calmar from stitched return series (concatenated across windows, not mean of per-window Calmars)

BacktestResult → compute_benchmark() → build_dashboard()
```

## Key design decisions

**Walk-forward rather than in-sample.** In-sample backtesting calibrates and tests on the same data and reliably overfits. Walk-forward calibrates on a training period then tests on held-out data. The ~26 test windows produce a performance curve that reflects what a live trader would have seen.

**Fisher combination over per-window p-values.** −2 Σ ln pᵢ ~ χ²(2k) gives more power than averaging p-values and makes regime-dependent performance visible. Fisher assumes approximate window independence, which is labelled in the output.

**White's RC runs on the full candidate matrix, not just the winner.** Standard backtesting picks the best parameter set and reports its Sharpe - you tested the whole grid but reported one result. RC bootstraps all candidates simultaneously, preserving cross-candidate correlation, and asks whether the best candidate beats the benchmark better than chance would predict after that search.

**Flat-cash windows are not skipped.** A window where the strategy makes no trades is a valid result - the strategy held cash. Excluding these from the aggregate Sharpe biases the summary upward by removing a real outcome. Flat-cash windows contribute p=1.0 to Fisher and zero-return arrays to the RC candidate matrix. This is why Fisher and RC must cover the same window set - if one excludes flat-cash and the other doesn't, they're testing different hypotheses.

**Adjusted high and low, not just adjusted close.** The slippage model fills at `close ± slippage × (high − low)`. On ex-dividend dates, adjusted close can fall outside the unadjusted [low, high] band, making fill prices nonsensical. `ingestion.py` applies the same adjustment factor to all three columns.

**Cost-inclusive position sizing.** `position_value = cash × fraction / (1 + cost_rate)` so that `position_value + buy_cost = cash × fraction` exactly. The naive version (`cash × fraction` then subtract cost) creates a small negative cash balance after every trade, which compounds across windows.

## Where things get complicated

**The Kalman filter prior.** The filter initialises with `μ₀ = log_prices[0]` and `P₀ = 1.0`. This is weakly informative - the 95% prior interval in log-price space corresponds to roughly [0.13×, 7.4×] the starting price, which is essentially uninformative. In practice the filter converges within 5–20 bars regardless of the prior. The 50-bar context window eliminates any prior effect for test signals entirely. The alternative (true diffuse prior, P→∞) gives identical results to within floating point noise.

**Two RC nulls are reported.** The cash null (White 2000) tests whether the best candidate beats zero return after data-snooping correction. The buy-and-hold null resamples active returns (strategy minus B&H) rather than raw returns - a strategy that passes the cash null but fails the B&H null is capturing beta, not generating alpha. Both p-values are computed and reported in all strategy runs.

**Fisher independence assumption.** Walk-forward windows share regime exposure (adjacent windows overlap by two years of training data). Fisher's method assumes independence and is slightly anti-conservative in this setting. The output labels the Fisher p as "(approx: windows not fully independent)". Treat it as directional evidence, not a precise threshold.
