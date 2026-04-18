# backtesting-engine

![CI](https://github.com/bonnie-mcconnell/backtesting-engine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-91%20passing-brightgreen)

A backtesting engine built to answer the question: **did this strategy actually have an edge, or did it just get lucky on this particular slice of history?**

---

## The problem with standard backtests

The typical backtesting tutorial looks like this: run a strategy over historical data, compute a Sharpe ratio, feel good about it. Two things go wrong.

**Overfitting to history.** If you test a strategy on the same data you used to design it, of course it works - you already saw the answers. Even without deliberate cheating, trying enough indicator combinations on a single dataset guarantees some will appear predictive purely by chance.

**No significance testing.** A Sharpe ratio is a number. Without asking *how often would a random strategy produce this result?*, you cannot tell whether an observed Sharpe of 0.8 reflects genuine edge or noise. The number means nothing without a claim attached to it.

This engine addresses both problems directly, which most publicly available backtesting frameworks do not.

---

## What it does differently

### Walk-forward validation

Rather than a single train/test split, the engine uses rolling windows: train on years 1–3, test on year 4, advance by one year, repeat. Every test period is strictly out-of-sample relative to the training window that preceded it.

On 30 years of SPY data this produces ~26 independent out-of-sample evaluations. Consistency across those windows is evidence of robustness. Concentration in one or two windows is evidence of luck.

**The training window is actually used.** During each window, the engine runs a grid search over (short\_window, long\_window) pairs on the training data and selects the pair with the highest Sharpe. The test period is then evaluated with those parameters - parameters chosen without ever looking at test data. This is what walk-forward validation is supposed to mean.

**Warmup context eliminates NaN signals.** A 200-day moving average needs 200 bars of history before it produces a valid value. Without this fix, the first 200 bars of every test window are treated as no-signal regardless of what prices are doing. The engine passes the tail of the training window as context when computing test-period signals, so MA values are valid from day one of the test.

### Block-bootstrap Monte Carlo significance testing

The engine builds a null distribution by resampling blocks of returns 10,000 times and computing the Sharpe of each. The p-value is the fraction of bootstrapped strategies that matched or exceeded the observed Sharpe.

**Why block bootstrap, not simple shuffling?** The Sharpe ratio is order-invariant - shuffling individual returns leaves mean and standard deviation unchanged, producing identical Sharpe values across all permutations. Block bootstrapping preserves local autocorrelation structure while randomising the global sequence. This produces a genuine null distribution that tests whether the strategy exploits return autocorrelation, rather than producing a noise floor of all-identical values.

**Why circular blocks?** Standard block bootstrap draws blocks starting at random positions and clips them at the array boundary. This underrepresents tail behaviour because blocks near the end are systematically shorter. Circular bootstrap wraps around the end of the array so every position is equally represented and every block is exactly `sqrt(n)` long.

**Why Fisher's method to combine p-values?** Averaging p-values across windows is not a valid statistical operation. Fisher's method (`-2 × Σ ln(pᵢ)`) follows a chi-squared distribution with 2k degrees of freedom under the joint null, and is more sensitive than averaging because it is dominated by windows with very strong evidence rather than giving equal weight to everything.

### All metrics implemented from scratch in NumPy

Sharpe, Sortino, maximum drawdown, Calmar, Omega - no TA-Lib, no quantstats. Each formula is unit-tested against hand-calculated expected values that were derived independently of the implementation.

One non-obvious fix: the standard `max_drawdown` formula computes `cumprod(1 + r)` starting from the first return, which means a loss on the very first bar shows up as zero drawdown (the rolling max starts at the post-loss value). The implementation prepends `1.0` to represent the pre-return portfolio value, anchoring the peak at the start of the period.

---

## Sample output

```
Loading SPY from 1993-01-01...
  8084 trading days loaded (1993-01-29 – 2025-04-17)

Running walk-forward validation (3yr train / 1yr test windows)...

Train period                    Test period                     Sharpe  Sortino   Max DD  Calmar   p-val    Trades
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1993-01-29 → 1996-01-25      1996-01-26 → 1997-01-23          1.11     0.75   -4.93%    2.31  0.5356   2 trades
  1994-01-27 → 1997-01-23      1997-01-24 → 1998-01-23          0.32     0.22   -6.33%    0.43  0.4792   2 trades
  ...

Summary across walk-forward windows
  Windows evaluated : 26  (0 skipped)
  Sharpe ratio      : 0.631
  Sortino ratio     : 0.472
  Max drawdown      : -4.16%
  Calmar ratio      : 1.328
  Omega ratio       : 1.342
  Mean p-value      : 0.5111  (per-window bootstrap mean)
  Fisher p-value    : 0.4823  (combined across all windows)

  NOT SIGNIFICANT: Fisher p = 0.4823 >= 0.05. The observed performance across
  walk-forward windows is consistent with noise. The golden cross on SPY does
  not have a statistically significant edge over this period.
```

The result is honest: positive returns across 26 windows, but no statistically significant edge. That is a more useful finding than a cherry-picked Sharpe from a single historical run.

The engine also produces a four-panel visualisation:

![backtest_results](backtest_results.png)

---

## Architecture

```
src/backtesting_engine/
├── config.py          # All constants with named justifications - no magic numbers
├── models.py          # Frozen dataclass contracts between pipeline stages
├── main.py            # Entry point: load → validate → fit → evaluate → report
├── walk_forward.py    # Rolling window orchestration and Fisher aggregation
├── simulator.py       # Bar-by-bar trade execution with signal validation
├── metrics.py         # Performance metrics and block-bootstrap p-value
├── visualisation.py   # Four-panel dashboard: equity, drawdown, Sharpe, returns
├── data/
│   ├── ingestion.py   # yfinance download, adjusted close prices
│   └── validator.py   # Structural checks before data enters the pipeline
└── strategy/
    ├── base.py        # Abstract interface: fit(train_data) + generate_signals(data)
    └── moving_average.py  # Grid-search calibration + warmup-aware signal generation
```

The strategy layer uses an abstract base class with two methods. `fit(train_data)` is called once per window to calibrate parameters in-sample. `generate_signals(test_data)` is called after fit to produce the out-of-sample signals. The orchestrator calls these uniformly without knowing which strategies are parameter-free and which are not.

Adding a new strategy - momentum, mean reversion, ML-based - requires implementing two methods. Nothing else changes.

---

## Running it

```bash
git clone https://github.com/bonnie-mcconnell/backtesting-engine.git
cd backtesting-engine
poetry install
make run
```

Downloads SPY from 1993, runs walk-forward validation, prints per-window metrics, reports the Fisher combined significance result, and saves `backtest_results.png`. First run takes ~45 seconds due to Monte Carlo permutations (10,000 per window × 26 windows).

---

## Tests

```bash
make test
```

91 tests across five modules:

- **`test_metrics.py`** - each metric verified against hand-calculated expected values; edge cases including zero std, single negative return (previously produced NaN Sortino - now fixed), zero drawdown, first-bar losses (previously not captured by max drawdown - now fixed with `1.0` prepend)
- **`test_simulator.py`** - complete buy/sell cycle with explicit PnL arithmetic; signal validation (invalid values raise); end-of-window position close; portfolio/trade consistency
- **`test_strategy.py`** - golden cross and death cross signal detection; fit() parameter update and validity; warmup context produces signals from test-period day one
- **`test_walk_forward.py`** - window count for known data lengths; no look-ahead bias; correct window advancement; Fisher combined p-value ordering properties
- **`test_validator.py`** - all structural checks: non-DatetimeIndex, duplicate timestamps, non-monotonic index, missing values, negative prices, insufficient rows

---

## Design decisions and trade-offs

**Walk-forward over a single split.** A single 80/20 split produces one test result. That result might be lucky or unlucky - there is no way to tell. 26 independent evaluations make consistency (or its absence) visible.

**Coarse parameter grid.** The grid search uses 10-day steps over a bounded range. Fine-grained optimisation on training data would overfit to training-period noise rather than pick parameters with genuine predictive content. The grid is intentionally coarse as a regularisation choice.

**Full portfolio allocation.** 100% position sizing means all returns are attributable to the strategy rather than to uninvested cash. Partial allocation dilutes metrics with cash drag, making the strategy appear more stable than it is.

**Explicit bar-by-bar simulation loop.** A vectorised implementation would be faster. An explicit loop is transparent and testable - every execution decision can be traced to a single line. For daily data on a single asset, the performance is acceptable.

**Zero risk-free rate default.** Subtracting a positive risk-free rate reduces the apparent Sharpe. Zero is the most strategy-favourable assumption and produces an upper bound on performance. The constant is named (`RISK_FREE_RATE`) and can be changed in one place.

---

## Known limitations

**Single asset.** No portfolio construction, cross-asset correlation, or capital rotation between strategies.

**Simplified execution model.** Trades fill at closing price with a fixed percentage fee. No slippage, market impact, bid-ask spread, or partial fills.

**Block bootstrap p-values test autocorrelation exploitation, not raw alpha.** For strategies with independent daily returns, every resampled block preserves the mean, so the null Sharpe distribution is centred near the observed Sharpe and p ≈ 0.5 regardless of actual alpha. The test gains power specifically when the strategy exploits return autocorrelation that block shuffling destroys. This is documented in the code and in the test suite.

**Fixed training/test window ratio.** The 3:1 train/test ratio is not optimised. Different ratios would produce different window counts and potentially different aggregated results.

---

## What I would build next

The evaluation framework is complete. The natural extension is a strategy discovery layer: a feature engineering pipeline that derives candidate predictors from price and volume data, trains a model to predict next-day returns, wraps the output as a `BaseStrategy` subclass, and evaluates it through this engine. The hard part is already done - generating a strategy that looks good is easy; generating one that survives 26 independent out-of-sample evaluations is not.