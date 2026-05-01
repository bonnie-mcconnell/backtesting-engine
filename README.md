# backtesting-engine

![CI](https://github.com/bonnie-mcconnell/backtesting-engine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Version](https://img.shields.io/badge/version-0.5.5-blue)
![Tests](https://img.shields.io/badge/tests-243%20passing-brightgreen)

A backtesting engine that answers the question most backtesting tutorials skip: did the strategy have a genuine statistical edge, or did it just happen to work on this particular slice of history?

The short answer, for SPY trend-following with realistic execution costs, is no. None of the three strategies produce a statistically significant result after walk-forward validation. That finding is more useful than a cherry-picked Sharpe from a single historical run - and the infrastructure to reach it honestly is what this project is actually about.

```bash
git clone https://github.com/bonnie-mcconnell/backtesting-engine.git
cd backtesting-engine
poetry install
make run          # downloads 30yr SPY, runs all three strategies, saves four HTML dashboards
```

First run ~4–6 minutes (data downloads and caches). Each subsequent run uses the cache.

---

## Why I built this

Every backtesting tutorial I found had the same structure: run a strategy, compute a Sharpe ratio, declare it good. That workflow has a specific problem - it's measuring how well you can fit the past in hindsight, not how the strategy would have performed if you'd run it live.

Walk-forward validation is the obvious fix, but it creates two harder problems. The first is that a Sharpe ratio from a walk-forward run still has no null distribution. You need block bootstrap to build one - but block bootstrap, not simple shuffling. Shuffling individual returns doesn't change the mean or standard deviation, so every permutation gives the identical Sharpe. You need consecutive blocks to preserve the autocorrelation structure that trend-following strategies exploit. I found a lot of implementations that shuffled individual bars and called it done. They're testing nothing.

The second problem is data snooping. The moving average grid tests 112 (short, long) pairs per training window. Reporting only the best pair's out-of-sample performance inflates apparent significance even when no pair has genuine edge - the multiple comparisons problem. White's Reality Check (2000) fixes this by testing the correct null: does *any* strategy in the full search universe beat the benchmark out-of-sample? The tricky part, which almost every implementation I found got wrong, is that the candidate return matrix must use test-period returns, not training returns. The winning pair was selected by training Sharpe - so of course it looks best in-sample. Test-period returns for all candidates are the correct input.

The Kalman filter became more interesting than I expected. A fixed moving average is implicitly a Kalman filter with untuned noise parameters. The local-level model makes those parameters explicit (process noise Q, observation noise R) and lets you calibrate them per training window by maximising the log-likelihood of the one-step prediction errors. The signal-to-noise ratio Q/R visibly shifts across the 30-year SPY window in a way that roughly tracks market regimes. That's not something a fixed MA can surface.

---

## What it does

### Three strategies, each mechanically distinct

**Moving average crossover.** Grid-searched over short ∈ [20, 80] and long ∈ [100, 250] per training window. The calibrated windows vary across market regimes, which is more honest than fixed 50/200 days - and the White's Reality Check corrects for the 112 comparisons made to find them.

**Kalman filter trend following.** The local-level state-space model:

```
trend[t]     = trend[t-1] + w[t],   w[t] ~ N(0, Q)
log_price[t] = trend[t]   + v[t],   v[t] ~ N(0, R)
```

Q and R are calibrated per training window by maximising the exact Gaussian log-likelihood over the innovation sequence. Optimised in log-space using Nelder-Mead with `adaptive=True` - log reparameterisation enforces positivity without constraints and keeps the search space symmetric when Q and R span orders of magnitude, which they do.

**Time-series momentum.** Buy when the T-day log-return is positive; sell when negative. Lookback T is grid-searched over {20, 40, 60, 90, 120, 180, 250} trading days per training window. This is the Moskowitz, Ooi & Pedersen (2012) time-series momentum signal. It's mechanically distinct from moving average crossover: instead of comparing two smoothed prices, it uses a single price ratio over a fixed window. The two signals are correlated but diverge in fast-reversing markets, where MA crossovers can lag.

### Four layers of significance testing

**Per-window block-bootstrap p-value.** Circular block bootstrap, block size √n, 10,000 resamplings. Circular prevents boundary truncation bias. Block-structured preserves autocorrelation.

**Fisher combined p-value.** `-2 Σ ln(pᵢ) ~ χ²(2k)` across ~29 windows (3+1 year windows on 30 years of SPY). More sensitive than averaging p-values because a single window with strong evidence dominates - correct when you want to detect whether *any* window shows genuine edge.

**White's Reality Check.** Data-snooping corrected p-value for strategies with parameter grids (MA and momentum). Uses stationary bootstrap (variable block lengths from Geometric(1/b)) on test-period returns for all candidates across all windows. Block size b = √T per White's recommendation. Not applicable to the Kalman filter, which has no parameter grid - Q and R are found by MLE on each training window, not selected from a discrete search universe.

**Buy-and-hold benchmark comparison.** Information ratio, paired t-test on per-window Sharpe differences, and beats-benchmark fraction. The information ratio is the right metric for "did the strategy add value over passive" - raw Sharpe ignores the correlation between strategy and benchmark returns.

### Realistic execution

Three configurable frictions, all of which move the result:

- **Slippage**: fills at `close ± factor × (high − low)`. Default 5% of daily range.
- **Signal delay**: signals fire at bar t, fill at bar t+1. Minimum realistic assumption - close prices aren't actionable until after close.
- **Transaction costs**: 0.1% per side.

`cost_sensitivity_sweep()` runs the full walk-forward at each point in a (cost_rate × slippage) grid. The `n_workers` parameter parallelises across CPU cores - on an 8-core machine a 5×5 grid that takes ~12 minutes serially finishes in ~2 minutes.

### Interactive dashboards

Self-contained HTML, no server required. Six panels per strategy:

| Panel | What it shows |
|---|---|
| Equity curve | Stitched out-of-sample portfolio vs buy-and-hold; window shading; range selector |
| Drawdown | Rolling peak-to-trough; max drawdown annotated |
| Rolling Sharpe | 63-day rolling annualised Sharpe |
| Per-window Sharpe | Bars coloured green/red relative to buy-and-hold Sharpe; IR and beats-% annotated |
| Return distribution | Histogram with normal overlay; skew and excess kurtosis |
| Parameter evolution | MA: short/long window drift. Kalman: Q/R signal-to-noise. Momentum: lookback drift |

---

## Results

All three strategies were evaluated on 30 years of SPY (1993–2024) with realistic execution: 0.1% fee per side, 5% of daily range slippage, and one-day signal delay. None produced a statistically significant result after walk-forward validation.

| Strategy | Sharpe | Sortino | Max DD | Fisher p | RC p | Beats B&H |
|---|---|---|---|---|---|---|
| MA Crossover | 0.31 | 0.42 | −18.4% | 0.41 | 0.58 | 38% |
| Kalman Filter | 0.28 | 0.38 | −19.1% | 0.49 | N/A | 35% |
| Momentum | 0.33 | 0.45 | −17.9% | 0.39 | 0.61 | 42% |
| Buy & Hold | 0.54 | 0.71 | −50.8% | - | - | - |

**Fisher p** is the combined significance across all ~29 walk-forward windows. Values above 0.05 mean the performance is consistent with noise. **RC p** is White's Reality Check p-value - the data-snooping corrected version that accounts for testing 112 (MA) or 7 (momentum) parameter combinations. All RC p-values are above 0.05. The Kalman filter has no RC p-value because MLE calibration is not a search over a discrete candidate universe.

**Note on the Reality Check null:** RC p tests whether any strategy beats cash (zero return), not whether it beats buy-and-hold. A small RC p would mean "the strategy adds value over doing nothing" - a weaker claim than beating the index. The information ratio and paired t-test in the benchmark comparison are the correct metrics for the harder question. All three strategies fail both tests.

The Kalman filter's calibrated signal-to-noise ratio (Q/R) is the most interpretable output beyond the headline metrics. It shifts visibly across the 30-year window: SNR rises during the 2000–2002 and 2008–2009 drawdowns (the filter becomes more reactive, tracking price more closely) and compresses during the low-volatility 2012–2019 expansion (the filter smooths more aggressively). This regime-dependence is what the parameter evolution panel in the dashboard surfaces - and it is the specific property that a fixed 50/200 MA cannot provide.

The result itself is the point. SPY buy-and-hold has a higher Sharpe than any of these trend-following strategies under realistic execution. The infrastructure for reaching that conclusion honestly - with no look-ahead bias, proper multiple-comparison correction, and cost sensitivity analysis - is what this project is actually about.

---

## Commands

```bash
make run           # all strategies + cost sensitivity sweep
make run-ma        # moving average strategy only
make run-kalman    # Kalman filter strategy only
make run-momentum  # momentum strategy only
make run-costs     # cost sensitivity sweep only
make test          # 243 tests
make check         # lint + typecheck + tests (mirrors CI exactly)
make clean         # remove generated dashboards and caches

# Custom ticker and date range:
make run-custom TICKER=QQQ START=2000-01-01

# Or directly:
poetry run backtesting-engine --strategy ma --ticker QQQ --start 2000-01-01
poetry run backtesting-engine --help
```

---

## Architecture

```
src/backtesting_engine/
├── config.py                All constants, each with a one-line justification
├── models.py                Frozen dataclass contracts
├── data/
│   ├── ingestion.py         yfinance download + Parquet cache + ex-div reconciliation
│   └── validator.py         Structural checks before data enters the pipeline
├── strategy/
│   ├── base.py              fit() / generate_signals() / candidate_test_returns()
│   ├── moving_average.py    Grid search calibration; stores all candidate test returns
│   ├── kalman_filter.py     MLE via Kalman innovation log-likelihood
│   └── momentum.py          Time-series momentum; lookback grid search
├── walk_forward.py          Rolling window orchestrator; Fisher + Reality Check
├── execution.py             Slippage + delay model; parallel cost_sensitivity_sweep()
├── metrics.py               Sharpe/Sortino/Drawdown/Calmar/Omega + block bootstrap
├── reality_check.py         White's Reality Check; stationary bootstrap
├── benchmark.py             Buy-and-hold comparison; information ratio; paired t-test
├── dashboard.py             Six-panel Plotly HTML dashboard
└── simulator.py             Explicit bar-by-bar simulator (readable reference baseline;
                             production entry point is execution.py)
```

**Strategy interface.** `fit(train_data)` calibrates in-sample. `generate_signals(test_data)` produces signals out-of-sample. `candidate_test_returns(test_data, context)` returns test-period returns for every parameter candidate - used to build the White's Reality Check candidate matrix. The orchestrator calls all three uniformly without knowing what's inside each strategy.

**Test-period returns in the Reality Check.** `candidate_test_returns()` is called *after* `fit()` but receives only test data. Training returns are the wrong input because the winning candidate was selected by training Sharpe. Using test returns means the Reality Check tests genuine out-of-sample performance across the full search universe.

**Parallelism.** `cost_sensitivity_sweep(n_workers=N)` runs N independent walk-forwards concurrently using `ProcessPoolExecutor`. Each (cost, slippage) combination is fully independent, so this is embarrassingly parallel. Pass `n_workers=-1` to use all available CPUs. The inner walk-forward window loop is sequential by design - windows share a mutable strategy object that is re-fitted on each window.

**Data caching.** `load_data()` writes Parquet files to `~/.cache/backtesting-engine/`. Files are considered stale after 24 hours. Pass `use_cache=False` to force a fresh download.

---

## Tests

243 tests across twelve modules. All expected values are derived independently of the implementation - never by calling the function under test.

| Module | What's tested |
|---|---|
| `test_metrics.py` | Sharpe/Sortino/drawdown/Calmar/Omega; NaN guards; edge cases |
| `test_kalman.py` | Filter recursion; likelihood ordering; fit() convergence; active_params() |
| `test_walk_forward.py` | Window count; no look-ahead; Fisher p; active_params storage; RC p-values |
| `test_simulator.py` | Trade cycle; PnL; signal validation; portfolio consistency |
| `test_strategy.py` | MA signals; fit(); context_window_size(); candidate_test_returns(); OCP compliance |
| `test_momentum.py` | Signal logic; uptrend/downtrend detection; fit(); context; candidate returns |
| `test_reality_check.py` | p-value bounds; H0 centering correctness; alpha detection; matrix assembly |
| `test_execution.py` | Slippage fills; signal delay; _OpenPosition correctness; config validation |
| `test_benchmark.py` | Buy-and-hold returns; cost deduction; information ratio; compute_benchmark() |
| `test_cli.py` | Argument parser defaults, all flags, invalid input rejection, flag combinations |
| `test_validator.py` | All structural data checks |
| `test_data/` | OHLCV ingestion; ex-dividend reconciliation; validation integration |

---

## Design decisions

**Why three strategies instead of one?** Moving average crossover and Kalman filter are both trend-following, so adding a second MA variant adds code without adding depth. Time-series momentum is mechanically distinct - it uses a single log-return over a fixed window rather than comparing two smoothed prices. The three strategies answer the same underlying question (is the trend up or down?) through different lenses, which makes their failure to beat buy-and-hold more informative than a single strategy's failure.

**Why Kalman filter over a second MA variant?** The Kalman filter is a different model class with a generative probabilistic structure, parameters with statistical meaning (Q = process noise variance, R = observation noise variance), and MLE calibration rather than grid search. A fixed MA is a special case of a Kalman filter with implicit, fixed Q/R. The comparison between a technical indicator and a properly specified state-space model is more interesting than between two technical indicators.

**Why MLE in log-space with Nelder-Mead?** Q and R are strictly positive and span several orders of magnitude. Optimising log Q and log R enforces positivity without constraints and makes the search space symmetric on the scale where the likelihood changes meaningfully. Nelder-Mead with `adaptive=True` handles the flat likelihood surface near Q → 0 better than gradient-based methods, which can stall there.

**Why stationary bootstrap for the Reality Check?** White (2000) recommends it because financial return series have heterogeneous autocorrelation. Fixed-block bootstrap underweights positions near array boundaries. Variable block lengths from Geometric(1/b) ensure stationarity of the resampled series.

**Why Fisher's method and not averaged p-values?** Averaging p-values has no sampling distribution justification. Fisher's `-2 Σ ln(pᵢ) ~ χ²(2k)` is derived from the fact that `-2 ln(p)` is χ²(2) under the null. It's also more sensitive: a single window with strong evidence dominates, which is the right behaviour when testing whether *any* window shows genuine edge.

**Why signal delay = 1 as the default?** Close prices aren't actionable until after the close. Acting on a signal at the same bar requires knowing the closing price before close - impossible in practice. One-day delay is the minimum realistic assumption.

**Why carry position state across window boundaries?** The naive approach resets every strategy to flat at the start of each test window. If the strategy was long at the training/test boundary - common in trending markets - it starts flat, misses the opening bars, and systematically understates returns. The correct approach detects the boundary position state from the context window and injects a buy signal at test bar 0 when entering long. All three strategies implement this.

**Why circular block bootstrap for Sharpe?** Standard block bootstrap clips blocks at array boundaries, so positions near the end of the series are underrepresented. Circular bootstrap wraps around so every block has exactly √n bars and every position is equally likely as a block start.

---

## Known limitations

**Single asset.** No multi-asset portfolio construction. The natural extension is an ensemble layer that allocates across strategies and assets based on regime detection.

**Approximate independence across windows.** Fisher's combined p-value assumes walk-forward windows are independent. Adjacent windows share training data and experience the same macro events. The p-value is an approximation - the output says so explicitly.

**Block bootstrap power.** The per-window bootstrap p-value has no power against iid alternatives. For strategies with independently distributed returns, p ≈ 0.5 regardless of Sharpe. It gains power specifically when the strategy exploits autocorrelation that block shuffling destroys. Documented in both code and tests.

**Metric aggregation across windows.** Sharpe, Sortino, and Omega are averaged across walk-forward windows - the standard walk-forward protocol where each window is an independent evaluation. Max drawdown reports the worst single window (not the mean - a mean drawdown is not meaningful). Calmar ratio is computed from the stitched portfolio across all windows so that cross-window compound drawdowns are captured correctly.

**Sortino ratio uses downside deviation.** The denominator is `sqrt(mean(min(r,0)²))` - the RMS of returns below threshold - not `std(downside_returns)`. Using std would inflate the ratio for strategies with consistent small losses (std→0 as losses become uniform). The downside deviation definition matches Sortino & van der Meer (1991).

**Close-to-close execution.** Fills at the close plus slippage based on the daily range. Real institutional execution uses intraday data and VWAP benchmarking. Reasonable for daily strategies on liquid ETFs; not suitable for intraday strategies or thinly traded names.

---

## What I learned building this

The most important correctness constraint in the Reality Check is that `candidate_test_returns()` must receive test data, not training data. The bug is easy to introduce: the function runs after `fit()`, and `fit()` selects the best parameter pair by training Sharpe - so if you evaluate all candidates on training data, the winner is always ranked first by construction. Every bootstrap replication will confirm it. The p-values look low and definitive, but the test is circular - you are measuring how well the optimiser found the best in-sample pair, not whether any pair has genuine out-of-sample edge. The fix is one argument change, but understanding why it matters requires being precise about what the Reality Check null actually states: that no strategy in the search universe beats the benchmark in the out-of-sample period. That question has no meaning if the returns are from the training period.

The circular block bootstrap exists to fix a boundary problem in the standard implementation. If you draw blocks starting at random positions and clip at the array boundary, blocks near the end of the series are shorter than the target block size - a block starting at position 240 in a 252-bar window gets truncated to 12 bars instead of √252 ≈ 16. Positions near the end of the series are underrepresented in the null distribution. The effect is subtle because the p-values still look plausible - the bias only becomes visible when you check whether the bootstrap correctly fails to reject a null that should hold, or correctly rejects one that shouldn't. Tiling the array twice and using modular indexing so blocks wrap around costs nothing and removes the bias entirely.

The `_OpenPosition` dataclass replaced two local variables inside the simulation loop - `entry_price: float | None` and `entry_date: pd.Timestamp | None` - where the None-narrowing was done with `assert entry_price is not None` guards. The problem is that Python strips assert statements when run with the `-O` flag, which means in optimised mode the guard disappears and a None entry price reaches the arithmetic that computes P&L, producing a `TypeError` several frames from the actual mistake with no useful context. Grouping both fields into a single dataclass means the position state is either `None` (flat) or an `_OpenPosition` instance with both fields guaranteed present at construction time. The narrowing moves from a runtime assertion to a type-system guarantee.

If I started over, the main thing I would change is parallelising the inner walk-forward window loop, not just the cost sensitivity sweep. The orchestrator passes the same strategy instance through all windows sequentially and calls `fit()` on each one. The windows are logically independent in terms of data, but parallelising requires spawning independent strategy copies per worker. That is straightforward for the MA and momentum strategies, but the Kalman filter's internal scipy optimiser state caused pickling errors across `ProcessPoolExecutor` workers, so I left the inner loop sequential and only parallelised the cost sweep, where each worker receives a fresh strategy instance. Fixing this properly would cut the per-strategy runtime from roughly two minutes to around fifteen seconds on an eight-core machine.

---

## What I'd build next

**Regime-conditioned strategy selection.** A two-state HMM trained on rolling volatility could gate strategy selection: Kalman filter in trending regimes, a mean-reversion variant in sideways markets. The walk-forward infrastructure already handles this - a regime classifier would just be a new strategy wrapping two sub-strategies.

**ML-based strategy.** Feature engineering (returns, volatility ratios, cross-sectional momentum) feeding a gradient-boosted classifier, wrapped as a `BaseStrategy` subclass. The hard part - rigorous out-of-sample evaluation - is already done.

**Intraday execution model.** Replace close-to-close fills with a VWAP participation model on 5-minute bar data, making cost estimates comparable to what an institutional desk would report.

**Parallel walk-forward windows.** The orchestrator passes the same strategy instance through all ~29 windows sequentially, calling `fit()` on each. Parallelising requires spawning independent strategy copies per worker - straightforward for MA and momentum, but the Kalman filter's `scipy.optimize.OptimizeResult` state caused pickling errors across `ProcessPoolExecutor` workers during development. Fixing this properly would reduce per-strategy runtime from ~2 minutes to ~15 seconds on an 8-core machine.
