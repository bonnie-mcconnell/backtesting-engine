# backtesting-engine

![CI](https://github.com/bonnie-mcconnell/backtesting-engine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)

A backtesting engine that tests whether a trading strategy has a statistically significant edge — not just whether it made money on historical data.

Built as a quant finance portfolio project by Bonnie McConnell, second-year CS + Statistics, Massey University NZ.

---

## The problem with simple backtests

Most backtesting tutorials run a strategy on historical data, compute a Sharpe ratio, and call it done. This produces numbers that feel meaningful but aren't. Two things go wrong:

**Look-ahead bias and overfitting.** If you test a strategy on the same data you used to design it, of course it looks good — you've already seen the answers. Even without deliberate cheating, trying enough indicators on one dataset guarantees some will appear predictive purely by chance.

**No significance testing.** A Sharpe ratio without a p-value is a number without a claim. An observed Sharpe of 0.8 on one historical run could reflect genuine edge or could be noise. Without asking "how often would a random strategy produce this result?", you cannot distinguish the two.

This engine addresses both problems directly.

---

## What this engine does differently

**Walk-forward validation** rather than a single train/test split. The engine slices data into rolling windows — train on years 1–3, test on year 4, advance by one year, repeat. Every test period is strictly out-of-sample relative to its training window. The result is ~26 independent out-of-sample evaluations rather than one, which gives a realistic picture of whether strategy performance is consistent across changing market regimes or concentrated in a lucky slice of history.

**Block-bootstrap Monte Carlo significance testing** of the Sharpe ratio. The engine builds a null distribution by resampling consecutive blocks of returns 10,000 times and computing the Sharpe of each. Simple shuffling is invalid here — the Sharpe ratio is order-invariant, so shuffling individual returns produces identical Sharpe values across all permutations. Block bootstrapping preserves local autocorrelation structure while randomising the global sequence, producing a genuine null distribution. The reported p-value is the fraction of bootstrapped strategies that matched or exceeded the observed Sharpe.

**All metrics implemented from scratch in NumPy** — Sharpe, Sortino, maximum drawdown, Calmar, Omega. No TA-Lib, no quantstats. Each formula is unit-tested against hand-calculated values.

---

## Sample output

Running the engine on SPY from 1993 to present with a 50/200-day moving average crossover strategy, 3-year rolling training windows, 1-year test windows:

```
Walk-forward analysis completed. Results:
Train: 1993-01-29 to 1996-01-25, Test: 1996-01-26 to 1997-01-23, Sharpe: 1.11, Sortino: 0.75, Max DD: -4.93%, P-value: 0.5356
Train: 2009-02-02 to 2012-01-31, Test: 2012-02-01 to 2013-02-01, Sharpe: 2.22, Sortino: 2.58, Max DD: -2.99%, P-value: 0.5372
...

Summary Metrics:
  Sharpe Ratio:  0.63
  Sortino Ratio: 0.47
  Max Drawdown:  -4.16%
  Calmar Ratio:  0.48
  Omega Ratio:   1.21
  P-value:       0.51

Strategy is not statistically significant (p >= 0.05).
```

The result is honest: the golden cross on SPY shows modest positive returns across 26 walk-forward windows, but no statistically significant edge over the test period. That is a more useful finding than a cherry-picked Sharpe from a single historical run.

---

## Architecture

The engine separates five concerns, each independently testable:

| Component | Responsibility |
|---|---|
| `data/ingestion.py` | Download adjusted close prices via yfinance |
| `data/validator.py` | Enforce structural assumptions before data enters the pipeline |
| `strategy/` | Abstract interface + moving average crossover implementation |
| `simulator.py` | Day-by-day trade execution with transaction costs and position tracking |
| `metrics.py` | Returns-based performance metrics and Monte Carlo significance test |
| `walk_forward.py` | Rolling window orchestration and cross-window aggregation |
| `models.py` | Typed dataclass contracts between pipeline stages |

The strategy layer uses an abstract base class so new strategies can be added without touching any other component. The orchestrator depends only on the `BaseStrategy` interface, not any concrete implementation.

---

## Running it

```bash
git clone https://github.com/bonnie-mcconnell/backtesting-engine.git
cd backtesting-engine
poetry install
poetry run backtesting-engine
```

Downloads SPY data from 1993, runs rolling walk-forward evaluation, prints per-window metrics and a summary significance conclusion. First run takes ~30 seconds due to Monte Carlo permutations.

---

## Tests

```bash
poetry run pytest -v
```

30 tests across three modules:

- **`test_metrics.py`** — each metric function verified against hand-calculated expected values; edge cases including zero standard deviation, no downside returns, and zero drawdown
- **`test_simulator.py`** — complete buy/sell cycle with explicit PnL and cost verification; no-signal flat portfolio; end-of-window position closing; data/signal mismatch guard
- **`test_walk_forward.py`** — correct window count for known data lengths; no look-ahead bias (test start always after train end); correct window advancement; insufficient data raises

---

## Design decisions

**Why walk-forward over a single split?** A single 80/20 split produces one test result. That result might be lucky or unlucky — there is no way to tell. Walk-forward produces ~26 independent evaluations. Consistency across windows is evidence of robustness; inconsistency is evidence of luck.

**Why block bootstrap over signal shuffling?** Signal shuffling requires re-running the simulator 10,000 times, which is computationally prohibitive. Simple return shuffling is mathematically invalid for Sharpe (order-invariant). Block bootstrapping is the standard solution from the time-series literature (Politis & Romano, 1994) — it respects autocorrelation structure while producing a genuine null distribution.

**Why full portfolio allocation?** The engine uses 100% position sizing so that all returns are attributable to the strategy rather than to uninvested cash. Partial allocation dilutes metrics with cash drag, making the strategy appear more stable than it is.

**Why a day-by-day simulation loop?** Explicit iteration makes the execution logic transparent and easy to test. A vectorised implementation would be faster but harder to verify. For a single-asset engine evaluated over decades of daily data, the performance is acceptable.

---

## Known limitations

- **Single asset.** No portfolio allocation, cross-asset correlation, or capital rotation between strategies.
- **Simplified execution.** Trades fill at closing price with a fixed fee. No slippage, market impact, or bid-ask spread modelling.
- **Block bootstrap p-values are noisy for sparse return series.** When the strategy is only invested for a small fraction of the test window, most daily returns are zero, which limits the bootstrap's discriminating power. This is acknowledged as a known limitation of applying block bootstrapping to a strategy with rare signals.
- **Fixed strategy parameters.** Moving average windows are fixed at 50/200. No parameter search or optimisation.

---

## What I would build next

The natural extension is a strategy discovery layer — a feature engineering pipeline that derives predictors from price and volume data, trains a gradient boosting model to predict next-day returns, and evaluates the resulting strategy through this engine. The evaluation framework is already in place; the missing piece is a systematic way to generate and filter candidate strategies.