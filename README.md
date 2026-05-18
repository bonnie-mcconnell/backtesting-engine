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

The Kalman filter was the hardest part to get right. Fitting it by maximum likelihood in log-space with Nelder-Mead, then using a diffuse prior to handle the cold-start at each window boundary, took longer than everything else combined. The result is interesting: the Kalman SNR shifts noticeably across walk-forward windows in a pattern that appears correlated with documented equity market regime changes (2000–2002 dot-com, 2007–2009 financial crisis). Whether that correlation is structural or coincidental is an open question. It does not translate into tradeable alpha after costs, which is the honest answer.

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

This produces `results/dashboard_ma.html`, `results/dashboard_kalman.html`, `results/dashboard_momentum.html`, and `results/cost_sensitivity.html`. Each dashboard is a self-contained HTML file (~4.5MB) with embedded Plotly JS. Open in any browser, no server required.

**Dashboard panels (6 per strategy):**

1. **Equity curve**: per-window portfolio values stitched together vs buy-and-hold benchmark across the full SPY history
2. **Per-window Sharpe**: bar chart comparing strategy Sharpe vs benchmark Sharpe for each of the ~26 test windows, coloured green/red vs that window's benchmark (not the aggregate)
3. **Rolling drawdown**: maximum drawdown at each bar across the full test period
4. **Parameter evolution**: how calibrated parameters drift across windows (MA short/long windows, Kalman SNR, momentum lookback)
5. **Returns distribution**: histogram of daily returns vs Normal fit
6. **Trade diagnostics**: per-trade P&L, holding period distribution, win rate across windows

The headline result for all three strategies is a Fisher combined p-value well above 0.05 and a Reality Check p-value similarly non-significant after multiple-comparison correction. The cost sensitivity sweep shows that at zero cost and zero slippage, some strategies look significant; the significance disappears as execution costs are raised toward the 0.1%/5% baseline, confirming that the apparent edge is mostly friction-dependent.

The table below intentionally does not hardcode numbers. Hardcoded numbers become stale within weeks as data revisions accumulate, and a mismatch between the README and actual output is a credibility problem. Run the command and the dashboards will show every metric with full context.

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

**Benchmark cost and slippage parity.** The buy-and-hold benchmark applies the same `transaction_cost_rate` and `slippage_factor` from `ExecutionConfig` as the strategy. The strategy pays both frictions on every fill; the benchmark pays them on its one round-trip entry and exit per window. This makes the comparison genuinely apples-to-apples across cost sensitivity sweeps.

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

- Single-asset focus for the main CLI. `make run-multi` tests four asset classes (SPY, QQQ, TLT, GLD) but only with the MA crossover strategy, not Kalman or momentum. This is the primary scope limitation of this frozen version.
- **Reality Check null is cash, not buy-and-hold.** White's RC tests whether the best strategy beats zero return. It does not test whether the strategy beats a passive buy-and-hold benchmark. A strategy can produce a low RC p-value while still underperforming B&H. The correct null for that question is to resample active returns (strategy minus B&H), which is not implemented here. This is the primary methodological gap in this frozen version.
- Fisher combination is approximate because walk-forward windows are not fully independent.
- Bootstrap block length is fixed at √n. Optimal length depends on autocorrelation structure (see Politis & White 2004 in References).
- yfinance data can have revision errors the validator does not catch.

Full discussion in [docs/methodology.md](docs/methodology.md).

---

## What I would do next

This project is frozen at v0.8.0. The items below are extensions that were scoped but not implemented. The first two overlap with Known Limitations above and are included here because they have a clear implementation path.

**Cross-asset validation** is partially implemented. `make run-multi` runs MA crossover on SPY, QQQ, TLT, and GLD and prints a comparison table. The obvious next step is running all three strategies across the full basket and surfacing the results in the dashboard, but it was not completed. If the null result holds across all four asset classes for all three strategies, that is a meaningfully stronger conclusion than the current single-strategy cross-asset check.

**Benchmark-relative Reality Check** is the biggest methodological gap in the current implementation. The RC null is cash (zero return). The right null for an active equity strategy is buy-and-hold. Resampling active returns (strategy minus B&H) rather than raw returns would directly test whether the strategy adds value over passive. This would require restructuring how `build_candidate_return_matrix()` receives benchmark returns.

**EM algorithm for Kalman MLE** would make `make run-kalman` 3–5× faster. The Nelder-Mead optimiser re-runs the full Kalman filter on every function evaluation. The EM algorithm has a closed-form E-step for the local level model and a natural stopping criterion; it would eliminate the ~2,000 filter passes per window.

**Optimal block length** via the Politis-White (2004) spectral method would replace the fixed √n heuristic. The optimal block length depends on the autocorrelation structure of each return series, which differs across assets and regimes. The current heuristic is reasonable but not calibrated to the data.

---

## References

- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097–1126.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. Chapter 2.
- Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303–1313.
- Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H. (2012). Time Series Momentum. *Journal of Financial Economics*, 104(2), 228–250.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Politis, D.N. & White, H. (2004). Automatic Block-Length Selection for the Dependent Bootstrap. *Econometric Reviews*, 23(1), 53–70.
- Lesmond, D.A., Ogden, J.P. & Trzcinka, C. (1999). A New Estimate of Transaction Costs. *Review of Financial Studies*, 12(5), 1113–1141.

---

## License

MIT - see [LICENSE](LICENSE).
