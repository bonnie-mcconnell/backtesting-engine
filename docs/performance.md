# Performance Guide

Expected runtimes, bottlenecks, and how to make runs faster.

---

## Expected runtimes

All timings are from a 2023 MacBook Pro (M2, 8 cores). Windows/Linux on equivalent
hardware is within 20–30%.

| Command | First run (download) | Subsequent runs (cached) |
|---|---|---|
| `make run-ma` | ~2–3 min | ~1.5–2 min |
| `make run-kalman` | ~4–6 min | ~3–5 min |
| `make run-momentum` | ~1–2 min | ~45 s – 1.5 min |
| `make run` (all three) | ~10–15 min | ~7–10 min |
| `make run-costs` | ~15–25 min | ~12–20 min |
| `make test` | - | ~3–4 min |

---

## What takes the most time

### 1. Kalman filter MLE (dominant cost)

The Kalman filter calibrates `(Q, R)` by maximum likelihood on each training window.
It uses Nelder-Mead (gradient-free), running up to 2,000 iterations per window.
With ~26 walk-forward windows, this is ~52,000 optimizer evaluations, each running
the full Kalman filter forward pass on 756 bars.

This is the correct tradeoff: Nelder-Mead is robust to the curvature discontinuities
near `Q→0` that break gradient-based methods. Faster alternatives (L-BFGS-B, gradient
descent) are unstable at the boundary. The EM algorithm would be faster and is worth
exploring for a future version.

**Rough breakdown for `make run-kalman`:**
- Kalman MLE across ~26 windows: ~80% of total time
- Signal generation + execution: ~5%
- Metrics + bootstrap (10,000 permutations × 26 windows): ~10%
- Dashboard rendering: ~5%

### 2. Block bootstrap (significant for all strategies)

`calculate_metrics()` runs 10,000 bootstrap permutations per window. With ~26 windows,
that is 260,000 permutations per strategy. The inner loop is pure NumPy and runs in
~2 ms per window on modern hardware (total: ~0.5 s per strategy). This is fast enough
that increasing `N_PERMUTATIONS` to 50,000 for more accurate p-values adds only ~2 s.

### 3. MA grid search (fast but visible)

The MA grid search evaluates ~112 candidate `(short, long)` pairs per training window.
Each evaluation runs `returns_from_signals()` - a vectorised NumPy operation on ~756
bars. Total: ~112 × 26 ≈ 2,900 evaluations per full run. This takes about 15 s.

---

## How to make it faster

### Short test runs

Use `--train-years 2 --test-years 1` and a shorter date range:

```bash
backtesting-engine --strategy ma --start 2010-01-01 --end 2020-12-31 --train-years 2 --test-years 1
```

This produces ~8 walk-forward windows instead of ~26 - roughly 3× faster.

### Reduce bootstrap permutations

In `config.py`, reduce `N_PERMUTATIONS` from 10,000 to 1,000 for exploration runs.
The p-value estimate is noisier (+/- 0.01 at p=0.05) but the run is ~10× faster.
Always use 10,000 for final results you intend to report.

### Parallel cost sensitivity sweep

`cost_sensitivity_sweep()` supports `n_workers > 1` for parallel execution:

```python
from backtesting_engine.execution import cost_sensitivity_sweep
results = cost_sensitivity_sweep(
    data, strategy,
    cost_rates=[0.001, 0.005],
    slippage_factors=[0.0, 0.05, 0.10],
    n_workers=-1,   # use all available CPUs
)
```

On an 8-core machine, a 5×5 sweep that takes 20 minutes serially completes in
~3 minutes. Note: parallelism uses `ProcessPoolExecutor` (spawns Python subprocesses)
so startup overhead is ~2–3 seconds per worker.

### Skip cost sensitivity

`--costs-only` runs only the sweep. Omit it and skip the sweep entirely with
`--strategy ma` (or `kalman`/`momentum`) without `--costs-only`.

---

## Profiling a run

To see exactly where time is spent:

```bash
poetry run python -m cProfile -s cumulative -m backtesting_engine.main --strategy kalman 2>&1 | head -40
```

Or with `py-spy` (lower overhead, no code changes needed):

```bash
pip install py-spy
py-spy record -o profile.svg -- python -m backtesting_engine.main --strategy kalman
open profile.svg  # flame graph in browser
```

The top entries will be `_kalman_filter_log_likelihood` and `scipy.optimize.minimize`.

---

## Memory usage

Memory is not a concern for single-asset daily data:

- SPY 1993–2024 (~8,000 rows × 4 columns): ~250 KB as float64
- Full `BacktestResult` with 26 windows: ~5–10 MB
- Dashboard HTML (embedded Plotly JS ~3 MB + data): ~4–8 MB

For multi-asset runs (not yet implemented), memory scales linearly with the number
of assets. A 100-asset universe over 30 years would be ~25 MB of price data, still
well within typical RAM limits.

---

## CI runtime

`make test` runs 413 tests in a few minutes on a modern machine. On this Windows
development machine, `poetry run pytest -q` completed in 2:14.

`N_PERMUTATIONS=10_000` in `config.py` is the production value used by `make run`.
Tests use a session-scoped fixture in `conftest.py` that patches this to 200 for the duration of
the test run. 200 permutations is sufficient to verify that p-values fall in [0,1], respond
correctly to high/low-drift inputs, and are reproducible. That is all tests need.

There is a subtle implementation detail: `white_reality_check` originally had
`n_bootstrap: int = N_PERMUTATIONS` as a default argument. Python evaluates default arguments
at function-definition time (import), not at call time, so patching the module attribute
after import had no effect. The fix: `n_bootstrap: int | None = None` with `N_PERMUTATIONS`
read inside the function body. `_monte_carlo_p_value` in `metrics.py` already used this
pattern correctly.

Without both patches the suite can take 90+ minutes across two Python versions in CI. With both
patches it completes comfortably inside the 20-minute CI timeout with no reduction in test correctness.

The CI workflow has `timeout-minutes: 20` as a safety net. If the suite ever approaches this
limit, identify slow tests with `pytest --durations=10` and promote expensive fixtures to
`scope="session"` in `conftest.py`.
