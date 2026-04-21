# backtesting-engine

![CI](https://github.com/bonnie-mcconnell/backtesting-engine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-154%20passing-brightgreen)

A backtesting engine that goes beyond "did the strategy make money?" to answer "did the strategy have a genuine statistical edge, or did it just get lucky on this particular slice of history?"

Includes: Two strategies, four layers of significance testing, realistic execution, interactive dashboards showing parameter evolution across market regimes.

---

## Why I built this

Every backtesting tutorial I could find had the same structure: run a strategy on historical data, compute a Sharpe ratio, call it done. That produces numbers that feel meaningful but aren't. I wanted to build something that reports a result you could actually defend - which meant walk-forward validation so each test period is strictly out-of-sample, statistical significance testing with a proper null distribution, and a correction for the data snooping that happens when you search over parameter combinations.

The hard parts turned out to be: understanding why you can't just shuffle returns to test Sharpe significance (the Sharpe ratio is order-invariant - shuffling changes nothing), correctly implementing White's Reality Check using test-period returns rather than training returns, and building a Kalman filter that calibrates its own parameters by maximising the likelihood of the prediction errors rather than by grid search.

---

## What it does

### Walk-forward validation with real in-window calibration

The engine slices historical data into rolling windows: calibrate on years 1–3, evaluate on year 4, advance by one year, repeat. On 30 years of SPY data this produces approximately 26 independent out-of-sample evaluations.

The training window is actually used. For the moving average strategy, a grid search over `(short_window, long_window)` pairs finds the highest-Sharpe pair on training data before the test period is touched. For the Kalman filter, maximum likelihood estimation calibrates the model parameters on training data. Each test period is evaluated with parameters chosen without looking at test data - and the calibrated parameters for each window are stored so you can see how they drift across market regimes.

### Two strategies that answer different questions

**Moving average crossover.** The 50/200-day golden cross is the most-documented strategy in existence. It is included as a benchmark - widely studied, intuitively clear, and representative of its strategy class. The walk-forward grid search selects different window pairs in different training regimes, which is more interesting than fixed parameters.

**Kalman filter trend following.** The local-level state-space model treats log-price as a latent trend evolving as a random walk plus observation noise:

```
trend[t]     = trend[t-1] + w[t],   w[t] ~ N(0, Q)
log_price[t] = trend[t]   + v[t],   v[t] ~ N(0, R)
```

The two parameters - process noise variance Q and observation noise variance R - are calibrated per training window by **maximum likelihood over the Kalman innovation sequence**. The innovations are the one-step-ahead prediction errors `e[t] = log_price[t] - E[log_price[t] | history]`. Under the model these are Gaussian with variance `S[t]` computed by the filter, so the exact log-likelihood is `ℓ = -½ Σ [ln(2π S[t]) + e[t]²/S[t]]`. Maximising this over (Q, R) with Nelder-Mead in log-space gives parameters that fit the actual return structure of the training window, not a grid.

A fixed moving average is a special case of this model with implicit, fixed Q/R. The Kalman filter is the version that learns the right Q/R from the data: high-variance training periods produce a more conservative filter; low-variance periods produce a more responsive one. The signal-to-noise ratio Q/R tracks which market regimes the filter identified as trending versus mean-reverting - and the dashboard shows how it changes across the 30-year evaluation period.

### Four layers of significance testing

**Per-window block-bootstrap p-value.** For each test window, 10,000 circular-block resamplings of the return sequence build a null distribution of Sharpe ratios. The p-value is the fraction of bootstrapped strategies that matched or exceeded the observed Sharpe. Circular blocks prevent boundary truncation bias that standard block bootstrap introduces.

**Fisher combined p-value across windows.** Averaging p-values across windows has no statistical justification - a p-value is not an effect size, and the average of k uniform random variables is not the relevant test statistic. Fisher's method derives from the fact that `-2 ln(p)` is chi-squared(2) under the null, so `-2 Σ ln(pᵢ)` is chi-squared(2k). The combined p-value answers whether the joint evidence across all 26 windows is too strong to be noise.

**White's Reality Check.** The moving average grid search evaluates 112 parameter pairs per training window. Reporting only the best pair's test-period performance inflates apparent significance - even if no pair has genuine edge, the best of 112 will look good by chance. White's Reality Check tests the correct null: does any strategy in the full search universe beat the benchmark in the out-of-sample period? The implementation uses the stationary bootstrap (variable block lengths from Geometric(1/b)) and builds the candidate matrix from test-period returns for every grid candidate in every window. Training returns would be wrong here - the winner was selected by training Sharpe, so its training performance is guaranteed to look best by construction.

**Strategy comparison.** After running both strategies through the same walk-forward engine with the same execution costs, the comparison is apples-to-apples: same data, same windows, same slippage, same delay.

### Realistic execution model

Three configurable execution frictions that change the answer significantly:

- **Slippage.** Fills at `close ± slippage_factor × (high - low)`. Requires OHLCV data. Default 5% of daily range - conservative for SPY.
- **Signal delay.** With `signal_delay=1`, signals fire at bar t but fill at bar t+1. Close prices are not actionable until after the market closes, so a one-day delay is the minimum realistic assumption. Many strategies that look good at zero delay become marginal with one day.
- **Transaction costs.** Symmetric fee on entry and exit. Default 0.1% per side.

### Interactive HTML dashboards

Self-contained HTML - no server, open in any browser. Six panels per strategy:

| Panel | What it shows |
|---|---|
| Equity curve | Stitched out-of-sample portfolio vs buy-and-hold; alternate windows shaded; 1Y/3Y/5Y/All range selector |
| Drawdown | Rolling peak-to-trough with maximum annotated |
| Rolling Sharpe | 63-day rolling annualised Sharpe - how performance evolves through market regimes |
| Per-window Sharpe | Bar chart coloured green/red; hover shows full metrics and trade count per window |
| Return distribution | Histogram with normal overlay; skewness and excess kurtosis annotated |
| Parameter evolution | MA: short/long window drift across training windows. Kalman: Q/R signal-to-noise ratio showing regime adaptation |

### Cost sensitivity analysis

`cost_sensitivity_sweep()` runs the full walk-forward analysis at each point in a 5×5 grid of (transaction\_cost\_rate, slippage\_factor) combinations and returns the Fisher p-value at each. The result is a heatmap saved as `cost_sensitivity.html` showing where the strategy loses statistical significance. The breakeven cost - the point at which p crosses 0.05 - is the most practically important number for deciding whether to pursue a strategy live.

---

## Sample output

```
════════════════════════════════════════════════════════════════════
  Backtesting Engine  ·  Walk-Forward  ·  Monte Carlo  ·  Reality Check
════════════════════════════════════════════════════════════════════

Loading SPY from 1993-01-01...
  8,084 trading days  (1993-01-29 – 2025-04-17)

──────────────────────────────────────────────────────────────────
  Strategy 1: Moving Average Crossover  (grid-search calibrated)
──────────────────────────────────────────────────────────────────
  Test window               Sharpe  Sortino    Max DD   p-val  Trades  Params
  ─────────────────────────────────────────────────────────────────────────────
  1996-01-26 → 1997-01-23     0.87     0.61    -5.21%  0.5241       2  MA(50/200)
  1997-01-24 → 1998-01-23     0.24     0.17    -6.89%  0.5031       2  MA(60/180)
  ...

  Windows                        26  (0 skipped)
  Sharpe ratio                   0.541
  Sortino ratio                  0.389
  Max drawdown                  -5.83%
  Fisher combined p              0.4912
  White Reality Check p          0.5731   ← data-snooping corrected

  ✗  NOT SIGNIFICANT: Fisher p = 0.4912 ≥ 0.05.
     Performance is consistent with noise - an honest result.

──────────────────────────────────────────────────────────────────
  Strategy 2: Kalman Filter Trend Following  (MLE calibrated)
──────────────────────────────────────────────────────────────────
  Test window               Sharpe  Sortino    Max DD   p-val  Trades  Params
  ─────────────────────────────────────────────────────────────────────────────
  1996-01-26 → 1997-01-23     1.12     0.81    -3.44%  0.4892       8  SNR=8.2e+03
  1997-01-24 → 1998-01-23     0.61     0.44    -4.11%  0.4701       6  SNR=1.1e+04
  ...

  Windows                        26  (0 skipped)
  Sharpe ratio                   0.718
  Sortino ratio                  0.531
  Max drawdown                  -4.21%
  Fisher combined p              0.3847

  ✗  NOT SIGNIFICANT: Fisher p = 0.3847 ≥ 0.05.
```

Neither strategy is significant at 0.05 after realistic execution costs. This is the correct result - the golden cross on SPY does not have a statistically significant edge at retail costs, and neither does the Kalman filter. The Kalman filter shows consistently better risk-adjusted metrics and a lower Fisher p (more consistent evidence across windows), but falls short of conventional significance. That finding is more useful than a cherry-picked Sharpe from a single historical run.

---

## Architecture

```
src/backtesting_engine/
├── __init__.py              Public API - all key symbols exported
├── config.py                All constants with names and justifications
├── models.py                Frozen dataclass contracts; WindowResult stores active_params
│
├── data/
│   ├── ingestion.py         yfinance OHLCV download (close + high + low + volume)
│   └── validator.py         Structural checks before data enters the pipeline
│
├── strategy/
│   ├── base.py              fit() + generate_signals() + candidate_test_returns()
│   ├── moving_average.py    Grid search calibration; stores all candidate test returns
│   └── kalman_filter.py     MLE via Kalman innovation log-likelihood; active_params()
│
├── walk_forward.py          Rolling window orchestrator; Fisher + Reality Check aggregation
├── simulator.py             Original bar-by-bar simulator (zero slippage, zero delay)
├── execution.py             Slippage + delay model; cost_sensitivity_sweep()
├── metrics.py               Sharpe/Sortino/Drawdown/Calmar/Omega + circular block bootstrap
├── reality_check.py         White's Reality Check; stationary bootstrap
└── dashboard.py             Six-panel interactive Plotly HTML dashboard
```

**The strategy interface has three methods.** `fit(train_data)` calibrates parameters in-sample. `generate_signals(test_data)` produces signals out-of-sample. `candidate_test_returns(test_data, context_data)` returns test-period returns for every candidate in the parameter search - empty dict for strategies with no grid search. The orchestrator calls all three uniformly without knowing what's inside.

**WindowResult stores calibrated parameters.** Every walk-forward window records the active parameters that were used - MA windows or Kalman Q/R - in `active_params`. `BacktestResult.param_evolution` returns the full ordered history, which the dashboard uses to render the parameter drift panel.

**The Reality Check uses test returns, not training returns.** `candidate_test_returns()` is called after `fit()` (so calibration is done) but receives only test data. This is the correct data for the Reality Check: we're asking whether any candidate beats the benchmark out-of-sample, not whether the winner looks good on the same data used to select it.

---

## Running it

```bash
git clone https://github.com/bonnie-mcconnell/backtesting-engine.git
cd backtesting-engine
poetry install
make run
```

Downloads 30 years of SPY OHLCV, runs both strategies with walk-forward validation, prints a comparative table, and saves three HTML files. Open any of them in a browser - no server required.

```bash
make run-ma      # Moving average only
make run-kalman  # Kalman filter only
make test        # 153 tests
make check       # lint + typecheck + tests
```

First run: approximately 3–5 minutes. Kalman MLE optimisation runs once per training window (~26 times). Bootstrap draws 10,000 block resamplings per window per strategy.

---

## Tests

153 tests across nine modules. Every numeric test derives its expected value independently of the implementation - expected values are never computed by calling the function under test.

| Module | Tests | Coverage |
|---|---|---|
| `test_metrics.py` | 27 | Each metric against hand-calculated values; NaN guards; edge cases |
| `test_simulator.py` | 17 | Trade cycle; signal validation; portfolio/trade consistency |
| `test_strategy.py` | 19 | MA signals; fit(); warmup context; `candidate_test_returns()` |
| `test_kalman.py` | 23 | Filter recursion; MLE likelihood; fit() convergence; `active_params()` |
| `test_reality_check.py` | 9 | p-value bounds; H0 behaviour; candidate matrix assembly |
| `test_execution.py` | 15 | Slippage fills; signal delay; config validation; backward compatibility |
| `test_walk_forward.py` | 21 | Window count; no look-ahead bias; Fisher p; active\_params storage; RC p-values |
| `test_validator.py` | 11 | All structural checks |
| `test_data/` | 11 | OHLCV ingestion; column presence; validation integration |

---

## Design decisions

**Why Kalman filter over a second moving average variant?** A second MA variant adds code without adding depth. The Kalman filter is a different model class: it has a generative probabilistic model, its parameters have statistical meaning (variance of process noise and observation noise), and calibration is by maximum likelihood rather than grid search. The comparison between a technical indicator and a properly specified statistical model is more interesting than between two technical indicators.

**Why MLE in log-space with Nelder-Mead?** Q and R are variances - strictly positive and potentially spanning several orders of magnitude. Optimising log(Q) and log(R) enforces positivity without constraints and makes the search space roughly symmetric on the scale where the likelihood is meaningful. Nelder-Mead with `adaptive=True` handles the flat likelihood surface near `Q → 0` better than gradient-based methods.

**Why stationary bootstrap for the Reality Check?** White (2000) recommends the stationary bootstrap specifically because financial return series have heterogeneous autocorrelation. Fixed-block bootstrap underweights positions near array boundaries; variable block lengths from Geometric(1/b) ensure stationarity of the resampled series. Block size b = sqrt(T) is the standard recommendation.

**Why Fisher's method and not averaged p-values?** Averaging p-values has no sampling distribution justification. Fisher's `-2 Σ ln(pᵢ) ~ χ²(2k)` is more sensitive because a single window with very strong evidence dominates the statistic - which is correct when you want to detect whether any window shows genuine edge.

**Why signal delay = 1 as the default?** Close prices are not actionable until after the market closes. Acting on a signal at the same bar that generated it requires knowing the closing price before close - which is impossible. A one-day delay is the minimum realistic assumption and is the standard in academic backtesting.

**Why circular block bootstrap for Sharpe?** Standard block bootstrap clips blocks at the array boundary, so blocks starting near the end are shorter than blocks starting elsewhere. This underrepresents tail behaviour. Circular bootstrap wraps around, so every block is exactly sqrt(n) and every position is equally likely as a block start.

---

## Known limitations

**Single asset.** No multi-asset portfolio construction or cross-asset correlation. The natural extension is an ensemble layer that allocates across strategies based on regime detection.

**Approximate independence.** Fisher's combined p-value assumes walk-forward windows are independent. Adjacent windows share training data and both experience the same macro events. The p-value is approximate, not exact - and this is acknowledged in the output.

**Block bootstrap power.** The block bootstrap Sharpe test has no power against iid alternatives. For strategies with independently distributed daily returns, p ≈ 0.5 regardless of Sharpe. It gains power specifically when the strategy exploits autocorrelation that block shuffling destroys. This is documented in both the code and test suite.

**Close-to-close execution.** Trades fill at day's close plus slippage based on the daily range. Real institutional execution uses intraday data and VWAP as the benchmark. The slippage model is a reasonable approximation for daily strategies on liquid ETFs.

---

## What I would build next

**Regime-conditioned strategy selection.** A hidden Markov model with two states (trending/mean-reverting) trained on rolling windows could gate strategy selection: Kalman filter in trending regimes, a mean-reversion variant in sideways markets. The walk-forward evaluation infrastructure is already in place - adding a regime classifier would be a new strategy wrapping two sub-strategies.

**ML-based strategy.** Feature engineering (returns, volatility, volume ratios, cross-sectional momentum) feeding a gradient-boosted classifier predicting next-day return direction, wrapped as a `BaseStrategy` subclass with `fit()` as training and `generate_signals()` as inference. The hard part - rigorous out-of-sample evaluation - is already done.

**Intraday execution model.** Replace close-to-close fills with a VWAP participation model using 5-minute bar data, making execution cost estimates directly comparable to what an institutional desk would report.
