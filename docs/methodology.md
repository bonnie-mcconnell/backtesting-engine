# Methodology

This document explains the statistical tests used, the exact hypotheses being tested, and the known limitations of this backtesting framework. It is written to be honest rather than impressive.

---

## The primary question

> Does this strategy produce statistically significant risk-adjusted returns over buy-and-hold after realistic execution costs?

Every metric in this codebase is subordinate to that question. Some tests answer it more directly than others. This document explains which is which.

---

## Return layers

There are two distinct return layers in this framework. Conflating them is the most common source of misleading backtests.

**Signal returns** (`returns_from_signals`)  
Theoretical returns computed by multiplying close-to-close price changes by the lagged position signal. No transaction costs, no slippage, no execution delay. Used for White's Reality Check candidate evaluation - they represent the gross potential of each parameter combination before execution friction.

**Executed returns** (from `run_simulation_with_execution`)  
Returns after applying transaction costs, slippage (as a fraction of the daily high-low range), and signal execution delay. These are the numbers used for the headline Sharpe, Sortino, Fisher p-value, and benchmark comparison. If the strategy has edge, it must survive here.

The Reality Check is run on signal returns by deliberate design choice, not because it is technically required. The alternative - running RC on executed returns for every candidate - would also be statistically valid and would give a more conservative (harder to pass) test. This framework uses signal returns for RC because:

1. **It tests pre-cost alpha capacity**: RC on signal returns asks "is there any genuine signal here before execution friction?" If a strategy fails even this weaker test, adding costs can only make it worse.
2. **Consistency with the literature**: White (2000) and most academic RC applications use pre-cost returns.
3. **Separation of concerns**: the grid search (parameter selection) and the execution model (cost/slippage/delay) are separate sources of bias; testing them separately makes it easier to diagnose where edge is lost.

The Fisher combined p-value is run on *executed* returns (costs, slippage, delay applied). A strategy can be significant under Fisher-on-signal-returns but not under RC, or significant under both but still underperform buy-and-hold - each layer tests a different question. The gap between RC and Fisher measures how much of the gross edge is absorbed by execution costs.

---

## Per-window block bootstrap

**What it tests:** Whether the observed Sharpe in one test window is consistent with a zero-mean return process with similar autocorrelation structure.

**How it works:** Returns are centred (mean subtracted) before resampling, so the bootstrap distribution represents H₀: true mean = 0. Blocks of length √n are sampled with replacement and concatenated to form surrogate return series. The p-value is the fraction of surrogate Sharpes exceeding the observed.

**Why centred:** Without centring, the bootstrap distribution inherits the observed mean return, so p(boot_sharpe ≥ observed_sharpe) ≈ 0.5 for any positive-mean strategy regardless of signal quality. That makes the p-value uninformative. Centring anchors the null at zero explicitly.

**Known limitation:** Block bootstrap gains most of its power against serially correlated returns (e.g. momentum-style strategies that exploit autocorrelation). For iid strategies with genuine alpha, the test has correct Type I error but may have lower power than a parametric test would. The Fisher combination across windows compensates for this.

---

## Fisher combination across windows

**What it tests:** Whether the aggregate of per-window p-values is consistent with all windows being drawn from H₀.

**Formula:** χ² = -2 × Σ ln(pᵢ), distributed as χ²(2k) under H₀ where k is the number of windows.

**Why this rather than a single aggregate Sharpe:** Walk-forward windows are not independent - adjacent windows share training data and market regime exposure. Fisher combination is approximate under this non-independence, but it aggregates evidence across windows in a way that a single Sharpe over the full period does not. It is honest about this: the output is labelled "(approx: windows not fully independent)".

**Known limitation:** Fisher is biased toward rejection when windows are positively correlated (overlapping regimes). Use the per-window p-values as the primary diagnostic; Fisher as a summary signal only.

---

## White's Reality Check

**What it tests:** Whether the best parameter combination in a grid search beats the benchmark better than chance would, after correcting for the fact that you searched over multiple combinations.

**How it works:** The null distribution is built by block-bootstrapping the full matrix of candidate return series simultaneously - preserving the cross-candidate correlation. The RC p-value is the fraction of surrogate "best candidate" performances that exceed the actual best.

**Null benchmark:** Cash (zero return). This is weaker than a buy-and-hold null. A strategy that beats cash but not buy-and-hold would look significant under RC but not under the Information Ratio test. Both are reported. The headline claim ("none of the strategies beat buy-and-hold") comes from the IR test, not the RC.

**Known limitation:** RC p-values depend on the candidate grid. A very narrow grid (few candidates) inflates the RC p-value; a very wide grid deflates it. The grids used here are fixed in the strategy class and documented.

---

## Benchmark comparison (Information Ratio)

**What it tests:** Whether the strategy adds risk-adjusted value over buy-and-hold.

**Metrics:**
- **Information Ratio:** mean(active return) / std(active return) × √252, where active return = strategy return − B&H return per bar. This is the Grinold & Kahn (2000) definition.
- **Beats fraction:** Fraction of windows where the strategy Sharpe exceeds the buy-and-hold Sharpe on the same window.
- **Sharpe diff t-test:** Paired t-test on per-window Sharpe differences. Tests whether the strategy *consistently* differs from buy-and-hold across windows, not just on average.

**Cost parity:** The benchmark applies the same transaction cost rate as the strategy (passed through `ExecutionConfig`). A benchmark with lower costs than the strategy would make the strategy look worse than it is, and vice versa. Both pay one round-trip cost (entry + exit) per window.

---

## Execution model

The execution model has three parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `transaction_cost_rate` | 0.001 | Fee per side (0.1%). Typical for a retail ETF trade on a major broker. |
| `slippage_factor` | 0.05 | Fill = close ± 5% of daily high-low range. On a 0.5% intraday range, this is 2.5 bps. |
| `signal_delay` | 1 | Signal generated on day t, executed at day t+1 open. Prevents look-ahead. |

The 0.1% fee is conservative for institutional access but realistic for retail. Use `--cost 0.0001 --slippage 0.01` to model near-zero-cost execution. The cost sensitivity sweep shows how significance degrades across the full parameter space.

---

## Verifying results are not seed-dependent

Both the block bootstrap and White's Reality Check use random resampling, which means results vary slightly across runs unless the seed is fixed. To verify that a result is genuine and not an artefact of a particular random seed:

```bash
# Run with three different seeds and compare Fisher p-values
backtesting-engine --strategy ma --seed 42 --end 2024-12-31
backtesting-engine --strategy ma --seed 99 --end 2024-12-31
backtesting-engine --strategy ma --seed 137 --end 2024-12-31
```

**What to look for:**

| Scenario | Example | Interpretation |
|---|---|---|
| Stable and significant | p=0.02, p=0.03, p=0.02 | Robust signal. Confidence in the result. |
| Stable and not significant | p=0.18, p=0.21, p=0.17 | Robust null. The strategy has no detectable edge. |
| Marginal and seed-sensitive | p=0.03, p=0.08, p=0.12 | **Do not claim significance.** The result depends on the bootstrap draw, not the data. Report it as marginal. |
| Inconsistent direction | p=0.04, p=0.09, p=0.04 | Marginal. Run more seeds or increase `N_PERMUTATIONS` in config. |

**What seed-dependence means statistically:** Bootstrap p-values have sampling variance of approximately `sqrt(p(1-p)/N)` where N is the number of permutations. With N=10,000 and p=0.05, the standard error is ~0.002, so p-values within ±0.005 of each other across seeds are consistent. Variation larger than ±0.01 at the 5% threshold indicates the result is marginal. In that case, increasing `N_PERMUTATIONS` to 50,000 in `config.py` will reduce variance but will not change a marginal result into a significant one.

The default seed is 42 (set in `config.py` as `BLOCK_BOOTSTRAP_SEED`). All results in the README and `docs/reproducibility.md` use `--seed 42`. Setting `--seed` also affects the cost sensitivity sweep bootstraps, so a full reproducible run requires specifying seed, end date, and all execution parameters.

---

## Cross-asset validation

`make run-multi` runs MA crossover walk-forward independently on SPY, QQQ, TLT, and GLD
using the same execution config and window parameters as the single-asset run. This addresses
the most common critique of single-asset backtests: a strategy that works on SPY but not on
QQQ, TLT, or GLD is probably fitting to a US equity bull-market regime, not detecting a
genuine pattern.

The comparison table reports Sharpe, Fisher p, RC p, beats-B&H fraction, and Information
Ratio for each ticker. A null result that holds consistently across all four asset classes
is a stronger negative finding than the single-asset SPY result alone.

**Current scope:** MA crossover only. Kalman and momentum cross-asset validation are not
yet implemented. Running Kalman on GLD with a 3-year training window takes ~5 minutes
per ticker due to the MLE optimisation on every window; this is a known constraint.

**Interpretation note:** Each ticker is tested independently. There is no correction for
testing four assets (no Bonferroni or Holm step-down). A single ticker appearing significant
at p<0.05 while three others are null should not be treated as a significant finding without
further investigation.

---

## `N_PERMUTATIONS`: production vs test

`config.py` sets `N_PERMUTATIONS = 10_000`. This is the value used by `make run` and
`make run-frozen`. It gives bootstrap p-value estimates accurate to ±0.003 at p=0.05
(standard error = √(p(1-p)/N) ≈ 0.002).

The test suite patches this to 200 via a session-scoped fixture in `conftest.py`. 200
permutations is sufficient to verify statistical correctness properties (p-values in [0,1],
directional response to drift, seed reproducibility) without running 90+ minutes in CI.

If you want to run the full-precision bootstrap in the test suite (e.g. to verify a marginal
p-value is stable), comment out the `_patch_n_permutations` fixture in `conftest.py` and
re-run. The suite will take approximately 5–10 minutes per Python version.

---


## Known limitations and failure modes

These are the ways this framework can give misleading results even when used correctly:

**Single-asset overinterpretation.** Results on SPY from 1993–2024 reflect US large-cap equity in a predominantly bull market. The framework includes no guarantee that results generalise to other assets, regimes, or time periods.

**Benchmark survivorship.** SPY tracks the S&P 500, which has survivorship bias (failed companies are removed). The benchmark is not truly passive - it benefits from index reconstitution.

**Non-independent walk-forward windows.** Adjacent windows share regime exposure. Statistical tests assume approximate independence, which is only approximate.

**Bootstrap block length.** Block size is fixed at √n. The optimal block length depends on the autocorrelation structure of the returns, which is unknown. Sensitivity to block length is not tested.

**Close-to-close execution.** All fills are at the closing price (adjusted for slippage). Strategies executed on real opening auctions or intraday would face different friction.

**yfinance data quality.** Prices are downloaded from Yahoo Finance. Split and dividend adjustments can have errors on specific dates. The validator checks for obvious anomalies but cannot catch all data issues.

**Parameter instability.** MA windows and Kalman SNR are calibrated in-sample and held fixed for the out-of-sample test window. In real deployment, parameters would be recalibrated continuously, and the optimal parameter may change in ways not captured by yearly re-fitting.

**Fisher independence assumption.** Fisher combination is approximate because walk-forward windows are not fully independent. The reported p-values should be treated as directional indicators, not precise significance thresholds.
