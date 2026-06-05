# backtesting-engine
[![CI](https://github.com/bonnie-mcconnell/backtesting-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/bonnie-mcconnell/backtesting-engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A walk-forward backtesting framework for testing whether systematic trading strategies survive realistic execution costs and multiple-comparison correction.

The short answer for SPY 1993–2024: none of the three strategies tested (moving average crossover, Kalman filter trend-following, time-series momentum) produce statistically significant excess returns over buy-and-hold after 0.1% transaction costs and 5% slippage.

That result is the point. Rigorous negative results - reproducible, inspectable, statistically corrected - are more useful than strategies that look good in backtests and fall apart live.

---

## Why I built this

I kept running into the same problem: a strategy looks profitable until you add transaction costs, or until you realise the parameter you optimised was the best of 20 candidates and you never corrected for that search. Either of those kills most retail strategies. I wanted a framework that made both problems visible and hard to accidentally hide.

The Kalman filter was the most technically involved component. Getting MLE calibration right in log-space with a diffuse prior at each window boundary required careful handling of the cold-start problem - the filter diverges on short training windows if the prior variance is set naively. The calibrated SNR shifts noticeably across walk-forward windows in a pattern that tracks the 2000–2002 and 2007–2009 drawdown periods, which is consistent with documented equity market regime changes. It doesn't translate into tradeable alpha, but the structural interpretation is worth examining.

I also wanted to understand White's Reality Check from an implementation standpoint rather than just the paper. Seeing the RC p-value move from significant to non-significant as the parameter grid is widened makes the data-snooping correction concrete in a way that reading the theory doesn't.

---

## Quickstart

```bash
git clone https://github.com/bonnie-mcconnell/backtesting-engine
cd backtesting-engine
poetry install
make run
```

Reproduce the exact README results (frozen to 2024-12-31):

```bash
make run-frozen
```

Test whether the null result holds across asset classes (SPY, QQQ, TLT, GLD):

```bash
make run-multi
```

Outputs `results/dashboard_ma.html`, `results/dashboard_kalman.html`, `results/dashboard_momentum.html`, `results/cost_sensitivity.html`. Open any in a browser - they're self-contained HTML with embedded Plotly JS, no server required.

---

## Results

Run `make run-frozen` to generate current results. The command is:

```bash
backtesting-engine \
  --ticker SPY --start 1993-01-29 --end 2024-12-31 \
  --cost 0.001 --slippage 0.05 --delay 1 \
  --train-years 3 --test-years 1 \
  --output-dir results/
```

This produces `results/dashboard_ma.html`, `results/dashboard_kalman.html`, `results/dashboard_momentum.html`, and `results/cost_sensitivity.html`. Each dashboard is a self-contained HTML file (~4.5MB) with embedded Plotly JS. Open in any browser, no server required.

**Dashboard panels (6 per strategy):**

1. **Equity curve**: per-window portfolio values stitched together vs buy-and-hold benchmark across the full SPY history
2. **Per-window Sharpe**: bar chart comparing strategy Sharpe vs benchmark Sharpe for each of the ~26 test windows, coloured green/red vs that window's benchmark (not the aggregate)
3. **Rolling drawdown**: maximum drawdown at each bar across the full test period
4. **Parameter evolution**: how calibrated parameters drift across windows (MA short/long windows, Kalman SNR, momentum lookback)
5. **Returns distribution**: histogram of daily returns vs Normal fit
6. **Trade diagnostics**: per-trade P&L, holding period distribution, win rate across windows

The headline result for all three strategies is a Fisher combined p-value well above 0.05. For MA crossover and momentum (both of which search a parameter grid), the Reality Check p-value is also non-significant after multiple-comparison correction. For the Kalman filter, which has no parameter grid, Reality Check does not apply. The cost sensitivity sweep shows that at zero cost and zero slippage some strategies approach significance; it disappears as you raise costs toward the 0.1%/5% baseline, which is the point - the apparent edge is friction-dependent.

See [docs/reproducibility.md](docs/reproducibility.md) for environment details,
[docs/methodology.md](docs/methodology.md) for statistical methodology,
[docs/architecture.md](docs/architecture.md) for data flow and design decisions,
and [docs/performance.md](docs/performance.md) for expected runtimes.

---

## Architecture

```
src/backtesting_engine/
├── strategy/
│   ├── base.py              BaseStrategy interface + returns_from_signals
│   ├── moving_average.py    Grid-search calibrated MA crossover
│   ├── kalman_filter.py     MLE-calibrated local level Kalman filter
│   └── momentum.py          Lookback grid-search time-series momentum
├── data/
│   ├── ingestion.py         yfinance + split/div-adjusted H/L + Parquet cache
│   └── validator.py         Schema, missing-value, and range checks
├── execution.py             Slippage, delay, cost model + cost sensitivity sweep
├── walk_forward.py          Rolling train/test with position carry-over
├── reality_check.py         White's (2000) Reality Check
├── metrics.py               Sharpe, Sortino, Calmar, Omega, block bootstrap p
├── benchmark.py             Buy-and-hold comparison, Information Ratio, paired t-test
├── multi_asset.py           Cross-asset validation: same strategy across ticker universe
├── simulator.py             Reference simulator (readable baseline, zero-friction)
├── models.py                Dataclasses: Trade, SimulationResult, WindowResult, etc.
├── dashboard.py             6-panel interactive Plotly dashboard
├── config.py                All constants in one place
└── main.py                  CLI entry point
```

### The strategy interface

Every strategy implements four methods:

```python
strategy.fit(train_data)                          # calibrate parameters in-sample
strategy.generate_signals(data)                   # pd.Series of {-1, 0, 1}
strategy.candidate_test_returns(test, context)    # dict[param → return series] for RC
strategy.active_params()                          # dict of calibrated parameter values
```

Adding a new strategy means implementing this interface. The walk-forward runner, Reality Check, and dashboard work without modification.

---

## Key design decisions and tradeoffs

**`ExecutionConfig` defaults match the CLI.** Calling `walk_forward(data, strategy)` without an explicit `ExecutionConfig` uses cost=0.1%, slippage=5% of daily range, delay=1 bar - the same conservative model the CLI uses. There is no hidden "optimistic" mode when using the library programmatically. Zero-friction runs (for verifying strategy logic in isolation) require an explicit `ExecutionConfig(transaction_cost_rate=0, slippage_factor=0, signal_delay=0)`.

**Walk-forward, not a single in/out split.** A single optimisation followed by one out-of-sample test gives you one data point. Walk-forward gives ~26 independent test windows. Fisher combination across those windows is more informative than a single aggregate Sharpe, and you can see whether performance is consistent or just driven by one lucky window.

**Block bootstrap with centred null.** Return series from trend-following have serial correlation that violates the iid assumption underlying parametric Sharpe tests. Block bootstrap preserves autocorrelation by resampling contiguous blocks. The critical implementation detail: returns are centred (mean subtracted) before resampling. Without centring, the bootstrap distribution inherits the strategy's observed drift, and p(boot_sharpe ≥ observed_sharpe) ≈ 0.5 for any positive-drift strategy regardless of signal quality. Centring anchors H₀ at zero mean explicitly.

**White's Reality Check for grid search correction.** MA crossover has ~15 candidate (short, long) combinations. Picking the best performer without accounting for the search gives a biased result. RC bootstraps the full candidate return matrix simultaneously, preserving cross-candidate correlations, so the p-value accounts for the number of combinations tried.

**Adjusted high/low, not just adjusted close.** The execution model fills at `close ± slippage_factor × (high - low)`. If close is dividend-adjusted but high/low are not, the close can sit outside the [low, high] band on ex-dividend dates, making the fill price nonsensical. All three price columns use the same adjustment factor.

**No-trade windows as flat-cash, not excluded.** A window where the strategy makes no trades is a valid outcome - the strategy held cash. The old code excluded these windows from the aggregate Sharpe, which biased the summary upward by removing a real outcome from the record. No-trade windows now contribute Sharpe = 0 and p = 1.0.

**Cost-inclusive position sizing.** `position_value = cash × fraction / (1 + cost_rate)` so that `position_value + buy_cost = cash × fraction` exactly. The intuitive formula (`position_value = cash × fraction`, then subtract cost) creates a small negative cash balance after every trade.

**Benchmark cost and slippage parity.** The buy-and-hold benchmark applies the same `transaction_cost_rate` and `slippage_factor` from `ExecutionConfig` as the strategy. The strategy pays both frictions on every fill; the benchmark pays them on its one round-trip entry and exit per window. This makes the comparison consistent across cost sensitivity sweeps.

---

## Statistical test hierarchy

Ordered from weakest to strongest:

1. **Block bootstrap p (per window)** - does one window beat the zero-mean null?
2. **Fisher combined p** - do the windows collectively beat the zero-mean null? (approximate: windows not independent)
3. **White's RC p** - does the best parameter combination survive multiple-comparison correction?
4. **Beats B&H fraction** - in what fraction of windows does strategy Sharpe exceed buy-and-hold?
5. **Information Ratio + paired t-test** - does the strategy add consistent risk-adjusted value over buy-and-hold?

Tests 1–3 answer "is there any signal at all?" Tests 4–5 answer "does it matter in practice?" The headline claim uses 4 and 5.

---

## CLI reference

```
backtesting-engine [options]

--strategy {ma,kalman,momentum,all}   Strategy to run (default: all)
--ticker SYMBOL                        Ticker symbol (default: SPY)
--start YYYY-MM-DD                     Start date (default: 1993-01-29)
--end YYYY-MM-DD                       End date, inclusive (default: today).
                                       --end 2024-12-31 includes December 31.
                                       Set this for reproducible results.
--cost RATE                            Transaction cost per side (default: 0.001)
--slippage FACTOR                      Fraction of daily range (default: 0.05)
--delay BARS                           Signal execution delay in bars (default: 1)
--train-years N                        Training window in years (default: 3)
--test-years N                         Test window in years (default: 1)
--output-dir DIR                       Output directory (default: .)
--seed N                               Bootstrap random seed (default: 42)
                                       Set explicitly for fully reproducible results.
--no-cache                             Force fresh data download
--costs-only                           Run cost sensitivity sweep only
```

Cross-asset validation runs as a separate command:

```
backtesting-multi [options]

--tickers TICKER [TICKER ...]          Tickers to test (default: SPY QQQ TLT GLD)
--start YYYY-MM-DD                     Start date (default: 2005-01-01)
--end YYYY-MM-DD                       End date, inclusive (default: today)
--cost RATE                            Transaction cost per side (default: 0.001)
--slippage FACTOR                      Fraction of daily range (default: 0.05)
--delay BARS                           Signal execution delay in bars (default: 1)
--train-years N                        Training window in years (default: 3)
--test-years N                         Test window in years (default: 1)
--output-dir DIR                       Output directory (default: .)
--seed N                               Bootstrap random seed (default: 42)
```

---

## Tests

```bash
make test     # full suite
make check    # lint + typecheck + tests
make run-multi  # cross-asset comparison (SPY, QQQ, TLT, GLD)
```

Test coverage (413 tests, `make test`) includes: execution model correctness (slippage, delay, backward compat), position sizing invariant (no negative cash), block bootstrap null centring, RC flat-cash window parity (Fisher and RC cover the same windows), RC boundary carry-over parity, benchmark cost and slippage parity, per-window benchmark Sharpe accuracy, cross-asset validation (graceful ticker failure, result types, comparison table), `_fmt_metric` infinite-value safety, `--end` inclusive date offset, runtime `_min_rows` validation, yfinance retry handling, and Windows UTF-8 portability.

---

## Known limitations

- Single-asset focus for the main CLI. `make run-multi` tests four asset classes (SPY, QQQ, TLT, GLD) but only with MA crossover - Kalman and momentum cross-asset aren't implemented.
- **Reality Check null is cash, not buy-and-hold.** White's RC tests whether the best strategy beats zero return. A strategy can produce a low RC p-value while still underperforming B&H. Resampling active returns (strategy minus B&H) rather than raw returns would directly address this - it's not implemented here.
- Fisher combination is approximate because walk-forward windows are not fully independent.
- Bootstrap block length is fixed at √n. Optimal length depends on autocorrelation structure (Politis & White 2004).
- yfinance data can have revision errors the validator does not catch.

Full discussion in [docs/methodology.md](docs/methodology.md).

---

## What I would do next

This project is frozen at v0.8.0. Items I'd pursue with more time:

**Benchmark-relative Reality Check** is the biggest methodological gap. The RC null is cash. Resampling active returns (strategy minus B&H) rather than raw returns would directly test whether the strategy adds value over passive - that's the right question for an equity strategy.

**EM algorithm for Kalman MLE** would make `make run-kalman` significantly faster. Nelder-Mead re-runs the full filter on every function evaluation. The EM algorithm has a closed-form E-step for the local level model; it would cut the ~2,000 filter passes per window down to maybe 30.

**Full cross-asset validation** for all three strategies. Right now `make run-multi` only runs MA crossover across SPY/QQQ/TLT/GLD. If the null holds for Kalman and momentum too, that's a meaningfully stronger conclusion.

**Optimal block length** via the Politis-White (2004) spectral method. The √n heuristic is reasonable but not calibrated to the actual autocorrelation structure of each series.

---

## References

- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097–1126.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. Chapter 2.
- Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303–1313.
- Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H. (2012). Time Series Momentum. *Journal of Financial Economics*, 104(2), 228–250.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Politis, D.N. & White, H. (2004). Automatic Block-Length Selection for the Dependent Bootstrap. *Econometric Reviews*, 23(1), 53–70.
- Lesmond, D.A., Ogden, J.P. & Trzcinka, C. (1999). A New Estimate of Transaction Costs. *Review of Financial Studies*, 12(5), 1113–1141. The 0.1% per side default is consistent with their estimates for liquid US equity ETFs.

---

## License

MIT - see [LICENSE](LICENSE).
