# Results

> **These numbers are from a representative run.** Bootstrap p-values have sampling
> variance of ±0.02–0.04 at p = 0.5 due to finite permutations (10,000). The ratios
> and fractions are stable across runs; the p-values vary slightly. Run
> `make run-frozen` to get exact numbers from your environment.

Conditions: SPY 1993-01-29 to 2024-12-31, 0.1% transaction cost per side, 5%
slippage (fraction of daily high-low range), 1-bar signal delay, 3yr training
/ 1yr test windows, seed 42. 26 walk-forward windows.

```bash
make run-frozen
```

---

## Moving Average Crossover

26 walk-forward windows. Best in-sample pair varies by window; most common across windows is (40–60, 150–200).

| Metric | Value |
|---|---|
| Sharpe ratio (annualised) | 0.31 |
| Sortino ratio | 0.44 |
| Max drawdown | −47.3% |
| Calmar ratio | 0.07 |
| Omega ratio | 1.09 |
| Win rate | 51.2% |
| Avg holding (days) | 18.4 |
| Exposure | 68% |
| Fisher combined p | 0.41 |
| White RC p (vs cash) | 0.78 |
| White RC p (vs B&H) | 0.91 |

**Buy-and-hold comparison**

| Metric | Strategy | Buy-and-hold |
|---|---|---|
| Sharpe ratio | 0.31 | 0.52 |
| Max drawdown | −47.3% | −56.8% |
| Information ratio | −0.31 | - |
| Beats B&H (% of windows) | 38% | - |
| Paired t-test p | 0.19 | - |

---

## Kalman Filter Trend-Following

26 walk-forward windows. MLE-calibrated (Q, R) per window.

| Metric | Value |
|---|---|
| Sharpe ratio (annualised) | 0.18 |
| Sortino ratio | 0.26 |
| Max drawdown | −52.1% |
| Calmar ratio | 0.04 |
| Omega ratio | 1.05 |
| Win rate | 49.7% |
| Avg holding (days) | 24.1 |
| Exposure | 73% |
| Fisher combined p | 0.58 |
| White RC p (vs cash) | N/A - Kalman has no candidate grid |
| White RC p (vs B&H) | N/A - Kalman has no candidate grid |

**Buy-and-hold comparison**

| Metric | Strategy | Buy-and-hold |
|---|---|---|
| Sharpe ratio | 0.18 | 0.52 |
| Max drawdown | −52.1% | −56.8% |
| Information ratio | −0.47 | - |
| Beats B&H (% of windows) | 31% | - |
| Paired t-test p | 0.06 | - |

---

## Time-Series Momentum

26 walk-forward windows. Best in-sample lookback varies by window; longer lookbacks (120–252 days) dominate in trending regimes.

| Metric | Value |
|---|---|
| Sharpe ratio (annualised) | 0.24 |
| Sortino ratio | 0.34 |
| Max drawdown | −44.8% |
| Calmar ratio | 0.05 |
| Omega ratio | 1.07 |
| Win rate | 50.4% |
| Avg holding (days) | 31.6 |
| Exposure | 81% |
| Fisher combined p | 0.47 |
| White RC p (vs cash) | 0.82 |
| White RC p (vs B&H) | 0.94 |

**Buy-and-hold comparison**

| Metric | Strategy | Buy-and-hold |
|---|---|---|
| Sharpe ratio | 0.24 | 0.52 |
| Max drawdown | −44.8% | −56.8% |
| Information ratio | −0.39 | - |
| Beats B&H (% of windows) | 35% | - |
| Paired t-test p | 0.13 | - |

---

## Summary

All three strategies fail to reject the null at p < 0.05 on every test applied.

| Strategy | Fisher p | RC p (cash) | RC p (B&H) | IR | Beats B&H |
|---|---|---|---|---|---|
| MA crossover | 0.41 | 0.78 | 0.91 | −0.31 | 38% |
| Kalman filter | 0.58 | - | - | −0.47 | 31% |
| Momentum | 0.47 | 0.82 | 0.94 | −0.39 | 35% |

RC p is omitted for Kalman because it has no parameter grid: MLE calibrates (Q, R)
jointly from the training window, leaving exactly one candidate per window. White's RC
corrects for selection across a grid of candidates; with one candidate it reduces to
the univariate bootstrap p-value, which Fisher's method already captures. The omission
is deliberate, not a limitation of the tool.

For MA and Momentum, RC p (vs B&H) is higher than RC p (vs cash), as expected - the
B&H null is harder to reject than the cash null when the market drifts upward. None of
these strategies clear either bar.

At zero transaction cost and zero slippage, MA crossover and momentum approach
significance (Fisher p ~0.08–0.12). The significance disappears as costs increase
toward the 0.1%/5% baseline. The apparent edge is friction-dependent, which is
consistent with the existing literature on these strategies in liquid US equity markets.

---

## Notes

These numbers are rounded to two decimal places and were generated from a single run.
The bootstrap p-values have sampling variance of roughly ±0.02–0.03 at p = 0.5
(10,000 permutations). Run `make run-frozen` to reproduce exactly.

The buy-and-hold benchmark uses cost parity with the strategy: one round-trip at the
same transaction cost and slippage rates as the strategy, applied at the start of the
first walk-forward window. This is deliberate - comparing a cost-burdened strategy
against a frictionless benchmark would misattribute the cost drag to strategy skill.
