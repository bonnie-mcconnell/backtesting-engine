# backtesting-engine
[![CI](https://github.com/bonnie-mcconnell/backtesting-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/bonnie-mcconnell/backtesting-engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A walk-forward backtesting framework for testing whether systematic trading strategies survive realistic execution costs and multiple-comparison correction.

The short answer for SPY 1993–2024: none of the three strategies tested (moving average crossover, Kalman filter trend-following, time-series momentum) produce statistically significant excess returns over buy-and-hold after 0.1% transaction costs and 5% slippage.

That result is the point. The framework is designed to make that kind of rigorous negative result reproducible and inspectable, not to find strategies that look good in backtests.

---

## Why I built this

I wanted to understand why most published backtests are wrong. The specific problem I kept running into: a strategy looks profitable until you add transaction costs, or until you realise the parameter you optimised was selected from a grid of 20 candidates and you never corrected for that search. Both of these kill most retail strategies.

The Kalman filter was the hardest part to get right. Fitting it by maximum likelihood in log-space with Nelder-Mead, then using a diffuse prior to handle the cold-start at each window boundary, took longer than everything else combined. The result is interesting: the Kalman SNR adapts across walk-forward windows in a way that does correspond to detectable regime shifts - it just doesn't translate into tradeable alpha after costs.

---

## Quickstart

```bash
git clone https://github.com/you/backtesting-engine
cd backtesting-engine
poetry install
make run
```

Reproduce the exact README results (frozen to 2024-12-31):

```bash
make run-frozen
```

Outputs `results/dashboard_ma.html`, `results/dashboard_kalman.html`, `results/dashboard_momentum.html`, `results/cost_sensitivity.html`. Open any in a browser - they are self-contained HTML with embedded Plotly JS, no server required.

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

The table below intentionally does not hardcode numbers. Hardcoded numbers become stale within weeks as data revisions accumulate, and a mismatch between the README and actual output is a credibility problem. Run the command and the dashboards will show every metric with full context.

See [docs/reproducibility.md](docs/reproducibility.md) for environment details.

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
├── simulator.py             Reference simulator (readable baseline)
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

**Benchmark cost parity.** The buy-and-hold benchmark applies the same `transaction_cost_rate` from `ExecutionConfig` as the strategy. Using a global default constant for the benchmark but a custom rate for the strategy makes the comparison non-apples-to-apples in cost sensitivity sweeps.

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
--end YYYY-MM-DD                       End date (default: today)
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

---

## Tests

```bash
make test     # full suite
make check    # lint + typecheck + tests
```

Test coverage includes: execution model correctness (slippage, delay, backward compat), position sizing invariant (no negative cash), block bootstrap null centring, momentum RC candidate independence (each candidate uses its own lookback, not the fitted winner), flat-cash window handling, benchmark cost parity, walk-forward window mechanics, and Reality Check implementation.

---

## Known limitations

- Single asset (SPY). Cross-asset robustness is untested.
- Fisher combination is approximate because walk-forward windows are not fully independent.
- Reality Check null is cash, not buy-and-hold. Significant under RC ≠ significant vs B&H.
- Bootstrap block length is fixed at √n. Optimal length depends on autocorrelation structure.
- yfinance data can have revision errors the validator does not catch.

Full discussion in [docs/methodology.md](docs/methodology.md).

---

## What I would do next

The biggest gap is benchmark-relative Reality Check: resample active returns (strategy minus buy-and-hold) rather than raw returns. That would directly test the claim that matters - the strategy adds value over passive - rather than the weaker claim that it beats cash.

The second gap is cross-asset validation. Running the same framework over a basket of ETFs (SPY, QQQ, EEM, EFA, TLT, GLD) would test whether the null result is specific to SPY or robust across asset classes.

The Kalman filter MLE is slow because it re-runs the full filter on every optimiser call. The EM algorithm would be faster and has a natural stopping criterion.

---

## References

- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097–1126.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. Chapter 2.
- Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303–1313.
- Welch, G. & Bishop, G. (1995). An Introduction to the Kalman Filter. UNC TR 95-041.

---

## License

MIT - see [LICENSE](LICENSE).
