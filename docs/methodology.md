# Methodology

What each statistical test measures, what it doesn't, and where the framework
can mislead even when used correctly.

---

## The question

> Does this strategy produce statistically significant risk-adjusted returns over buy-and-hold after realistic execution costs?

Every metric is subordinate to that question.

---

## Two return layers

Two return layers exist, and conflating them is the most common source of misleading backtests.

**Signal returns** (`returns_from_signals`) - theoretical returns from multiplying close-to-close price changes by the lagged position signal. No costs, no slippage, no delay. Used for White's Reality Check candidate evaluation.

**Executed returns** (from `run_simulation_with_execution`) - returns after transaction costs, slippage, and execution delay. These drive the headline Sharpe, Fisher p, and benchmark comparison. If a strategy has edge, it must survive here.

The RC runs on signal returns by design. This tests pre-cost alpha - if a strategy fails this weaker test, costs can only make it worse. It also follows White (2000) and most academic RC implementations. The Fisher p uses executed returns. The gap between RC p and Fisher p shows how much gross edge execution friction absorbs.

---

## Per-window block bootstrap

Each test window produces a p-value. Returns are centred (mean subtracted) before resampling - without this, p(boot_sharpe ≥ observed_sharpe) ≈ 0.5 for any positive-drift strategy regardless of signal quality. Centring anchors H₀ at zero explicitly. Blocks of length √n are resampled to preserve autocorrelation structure.

Two different block bootstrap variants are used:

- **Per-window Sharpe p-value** (`metrics.py`): circular block bootstrap (Kunsch 1989). Fixed block length ⌊√n⌋, circular wrapping to avoid edge effects. Simple and fast for a single series.
- **White's RC** (`reality_check.py`): stationary bootstrap (Politis & Romano 1994). Geometrically distributed block lengths with mean √n. This is the method used in White (2000) and is preferred for the RC candidate matrix because it preserves the joint distribution of candidates more faithfully when block lengths vary across columns.

The practical implication: Fisher p-values (from the circular block bootstrap) and RC p-values (from the stationary bootstrap) are not computed by the same method, so they are not strictly comparable at the level of individual draws. Both are valid tests of their respective H₀; the choice of which variant to use for each test follows the established literature.

Power per window is limited - 252 daily returns isn't much data. Fisher combination is what gives the test meaningful power.

---

## Fisher combination

χ² = −2 × Σ ln(pᵢ), distributed as χ²(2k) under H₀. Aggregates evidence across windows without requiring a single long return series.

Fisher assumes independence. Walk-forward windows share regime exposure, so this is only approximate. Treat it as directional evidence, not a precise significance threshold. The output labels it "(approx: windows not fully independent)".

**Statistical power.** Power per window is limited (252 daily returns isn't much data - single-window power to detect Sharpe 0.5 is only ~13%). Fisher combination is what gives the test meaningful power: with 26 windows of 252 bars each, the combined test has approximately 76% power to detect a consistent annualised Sharpe of 0.5 at α=0.05 (iid approximation, one-sided). A consistent Sharpe of 0.7 gives ~95% power. This means the null result on the MA and momentum strategies is informative - if either had a genuine edge comparable to buy-and-hold (~0.52 Sharpe over the test period), the test would detect it with high probability. What the test cannot rule out is a weaker or intermittent edge (Sharpe < 0.3, or an edge that only appeared in a few windows).

---

## White's Reality Check

When you grid-search parameters and report the best Sharpe, the naive p-value is misleading - you've effectively run many tests and reported the best one. RC corrects for this by building a null distribution from the full candidate matrix simultaneously, preserving cross-candidate correlation.

Two nulls are reported:

**Cash null (RC p vs cash):** The original White (2000) formulation. Tests whether the best candidate beats zero return after data-snooping correction. A strategy can have a low RC p vs cash and still underperform buy-and-hold.

**Buy-and-hold null (RC p vs B&H):** Resamples active returns (candidate returns minus buy-and-hold returns) rather than raw returns. Tests whether the best candidate beats a passive investor after data-snooping correction. This is the more relevant test for an active equity strategy. A strategy that has low RC p vs cash but high RC p vs B&H is capturing beta rather than generating alpha.

RC p-values depend on the candidate grid. A wider grid inflates the RC p-value - more candidates means the bootstrap null distribution of the maximum is larger, so the observed max more easily falls within it; a narrower grid deflates it. The grids are fixed in the strategy classes and documented there.

---

## Benchmark comparison

Buy-and-hold on the same test windows, with the same transaction cost and slippage applied (one round-trip per window). Without cost parity the comparison isn't fair.

Three metrics: Information Ratio (`mean(active_returns) / std(active_returns) × √252`, per Grinold & Kahn 2000), beats-fraction (fraction of windows where strategy Sharpe exceeds B&H Sharpe), and a paired t-test on per-window Sharpe differences.

---

## Execution model

| Parameter | Default | Meaning |
|---|---|---|
| `transaction_cost_rate` | 0.001 | Fee per side (0.1%). Realistic for retail ETF trading. |
| `slippage_factor` | 0.05 | Fill = close ± 5% of daily high-low range. |
| `signal_delay` | 1 | Signal on day t filled at day t+1 close. Prevents look-ahead. |

---

## Verifying results are not seed-dependent

Both bootstrap and RC use random resampling. To verify a result isn't a bootstrap artefact:

```bash
backtesting-engine --strategy ma --seed 42 --end 2024-12-31
backtesting-engine --strategy ma --seed 99 --end 2024-12-31
backtesting-engine --strategy ma --seed 137 --end 2024-12-31
```

Stable p-values across seeds (within ~0.005 at p=0.05) indicate a robust result.
Variation larger than ±0.01 indicates a marginal one. More permutations reduce sampling
variance but won't turn a marginal result into a significant one.

---

## Cross-asset validation

`make run-multi` runs MA crossover on SPY, QQQ, TLT, and GLD independently with the same parameters. `make run-multi-all` runs all three strategies across the same tickers. A null result across all three strategies and all four asset classes is a materially stronger finding than the single-SPY result.

The comparison table doesn't apply a multiple-testing correction across tickers; a single ticker appearing significant while three are null shouldn't be treated as a significant result.

---

## `N_PERMUTATIONS`: production vs test

`config.py` sets `N_PERMUTATIONS = 10_000` for production. The test suite patches this to 200 via a session-scoped fixture to keep CI under 20 minutes. To run full-precision bootstrap in tests, comment out `_patch_n_permutations` in `conftest.py`.

---

## Known limitations

**Parameter instability.** The best in-sample MA pair and momentum lookback vary substantially across walk-forward windows (see RESULTS.md). If the strategy had a genuine stable edge, optimal parameters would cluster. The observed variation is consistent with fitting noise rather than signal - which is one reason the out-of-sample results are poor.

**Single-asset results.** SPY 1993–2024 is predominantly a bull market. Results may not generalise.

**Benchmark survivorship.** SPY has survivorship bias from index reconstitution. The benchmark is not truly passive.

**Non-independent windows.** Fisher combination assumes approximate independence.

**Bootstrap block length.** Fixed at √n. Optimal length depends on autocorrelation structure.

**Close-of-day fills.** All fills at adjusted close ± slippage. Real execution on opening auctions differs.

**yfinance data quality.** Split and dividend adjustments can have errors. The validator catches obvious ones but not all.
