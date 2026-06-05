# Changelog

## [0.8.0] - 2026-05-12

> No 0.7.x release - the fixes below were developed incrementally after 0.6.2 and
> shipped together once cross-asset validation was done.

### Added

- **Cross-asset validation** (`multi_asset.py`, `tests/test_multi_asset.py`).
  `make run-multi` runs MA crossover on SPY, QQQ, TLT, and GLD (2005–2024), prints a
  comparison table with Sharpe, Fisher p, RC p, and IR. Tickers that fail data loading
  are skipped with a warning rather than crashing. `run_multi_asset()` added to public API.

- **`py.typed` PEP 561 marker** - signals to mypy that inline type hints are available
  for downstream consumers.

- **`backtesting-multi` CLI entry point** registered in `pyproject.toml`. Running
  `python -m backtesting_engine.multi_asset` previously emitted a RuntimeWarning;
  `backtesting-multi` eliminates that.

- `BenchmarkResult.per_window_benchmark_sharpes: list[float]` - per-window B&H Sharpe
  values so the dashboard can colour bars against the correct per-window benchmark, not
  the aggregate mean.
- `tests/test_fixes.py` - 26 new tests covering correctness fixes in this release.
- `tests/test_integration.py` - end-to-end coverage for walk-forward + benchmark + dashboard.
- `docs/performance.md` - runtime expectations and profiling notes.

### Fixed - CI and test infrastructure

- **CI was timing out after 6 hours.** Root cause: `N_PERMUTATIONS=10_000` in every
  test calling `walk_forward()`, with 49+ such calls in the suite. ~1.47M bootstrap
  iterations per Python version. Fix: `conftest.py` patches `N_PERMUTATIONS` to 200
  for the full session via an `autouse=True, scope="session"` fixture. Production still
  uses 10,000. Added `timeout-minutes: 20` to CI as a safety net.

- **Redundant `walk_forward` calls across the test suite.** `test_integration.py` was
  re-running `walk_forward` for every test method in each pipeline class. Refactored to
  class-scoped fixtures. `test_walk_forward.py` went from 28 independent walk_forward
  calls to 5 using module-scoped fixtures from `conftest.py`. Total integration suite
  time ~24s vs 120+s before.

- **CDN self-containment assertion was wrong** (`test_integration.py`). The string
  `cdn.plot.ly` appears inside the embedded Plotly JS bundle (as a topojson URL), not
  as a CDN load instruction. The old assertion `assert "cdn.plot.ly" not in html` always
  failed. Fixed to check for `src="https://cdn.plot.ly` - the actual script-tag pattern.

- **`white_reality_check` bootstrap patch was silently ineffective.** The function had
  `n_bootstrap: int = N_PERMUTATIONS` as a default argument. Python evaluates default
  arguments at import time, not call time, so patching the module attribute in
  `conftest.py` did nothing - RC still ran 10,000 iterations regardless of the patch.
  Changed to `n_bootstrap: int | None = None` with the module attribute read in the
  function body. `_monte_carlo_p_value` in `metrics.py` already did this correctly.

### Removed

- `_LOOKBACK_GRID` alias in `momentum.py`. `fit()` now references `MOMENTUM_LOOKBACKS`
  from `config.py` directly.

### Changed

- `pyproject.toml` now includes `[project.urls]`; version bumped to 0.8.0.
- `docs/methodology.md`: cross-asset section added; N_PERMUTATIONS production vs test
  distinction documented.
- `walk_forward.py`: renamed `valid = window_results` to `all_windows` - too easy to
  confuse with `BacktestResult.valid_windows`.
- `simulator.py`: removed duplicate `_OpenPosition` dataclass (was identical to the one
  in `execution.py`). Now imports from there.
- `CONTRIBUTING.md`: fixed the `MomentumStrategy` example - it was showing `fit()` as a
  no-op and omitting `candidate_test_returns()`, which would have silently broken RC for
  anyone who copied it.

### Fixed - crash bugs

- **`_fmt_metric` crashed with `±inf` values** - was passing `"∞"` as a format spec to
  `format()`, which raises `ValueError: Unknown format code`. Changed to return the
  `"∞ (unbounded)"` string directly.
- **`_min_rows` used module-level defaults, not runtime values** - `_min_rows` in
  `main.py` read `TRAINING_WINDOW_YEARS` and `TESTING_WINDOW_YEARS` at definition time.
  If `--train-years` or `--test-years` were passed on the CLI, the minimum row check
  used the wrong window sizes. Fixed: `_min_rows` now takes `train_years` and
  `test_years` as arguments.
- **`--end` was not inclusive** - `yfinance` treats the end date as exclusive, so
  `--end 2024-12-31` returned data through 2024-12-30. Added a one-calendar-day offset
  internally so the CLI contract matches user expectation.
- **RC flat-cash parity** - flat-cash windows contributed `p=1.0` to Fisher but were
  omitted from the RC candidate matrix. Fisher and RC were testing different window sets.
  Fixed: flat-cash windows now contribute zero-return candidates to the matrix.
- **Benchmark slippage missing** - `compute_benchmark()` applied transaction costs but
  not slippage when `ExecutionConfig` was passed. The benchmark entry/exit now applies
  `slippage_factor` on the same terms as the strategy.

---

## [0.6.2] - 2026-05-06

### Fixed
- `test_final_fixes.py` docstring referenced "v0.6.1 → v0.6.2" but version was never
  bumped. Now 0.6.2.
- Last hardcoded `42` in `execution.py` docstring replaced with `BLOCK_BOOTSTRAP_SEED`.

### Added
- `format_params()` on `BaseStrategy` - each strategy now owns its own parameter
  formatting. MA returns `MA(50/200)`, Kalman returns `SNR=1.23e-03`, Momentum returns
  `MOM(90)`. Eliminates strategy-specific knowledge from `main.py`.

### Documentation
- CI badge added to README.
- `docs/methodology.md`: new section on verifying results are not seed-dependent.

---

## [0.6.1] - 2026-05-06

### Fixed
- `_flat_cash_metrics()` was returning `sortino=inf` and `omega=inf`. Since `mean_metric`
  silently skips inf values, flat-cash windows were being dropped from aggregate
  Sortino/Omega, overstating both. Changed to `sortino=0.0` and `omega=1.0` so they
  participate correctly.
- `cost_sensitivity_sweep()` was ignoring `--seed`. The `_sweep_worker` had no
  `bootstrap_seed` parameter, so every sweep always used the config default regardless
  of what was passed on the CLI.

### Changed
- `_fmt` nested function inside `_print_summary()` promoted to module-level `_fmt_metric`
  and shared with `_print_comparison()`, which had its own duplicate `_fmt_val`.

---

## [0.6.0] - 2026-05-06

### Fixed
- `--seed` flag was parsed but never forwarded to `walk_forward`,
  `calculate_metrics`, or `white_reality_check`. Results were never seed-controlled
  regardless of what was passed.
- `ExecutionConfig` defaults were misaligned with the CLI: the dataclass defaulted to
  `slippage_factor=0.0, signal_delay=0` while the CLI defaulted to `--slippage 0.05
  --delay 1`. Programmatic use got a different (optimistic) execution model.
- Exposure fraction used `abs(portfolio_change) > threshold` to detect in-market bars.
  On a quiet SPY day (0.08% move) a fully-invested portfolio would fall below any
  reasonable threshold. Now computes exact exposure from trade entry/exit dates.

### Added
- `docs/architecture.md` - data flow diagram and design decision writeups.
- `docs/methodology.md` - what each test measures, what it doesn't, where it can mislead.
- `docs/reproducibility.md` - how to reproduce the frozen README results exactly.

---

## [0.5.4] - 2026-04-26

### Fixed
- `test_data/test_ingestion.py` mock functions were building `pd.MultiIndex` incorrectly
  (dict-of-tuple-keys syntax produces a plain Index of tuples, not a real MultiIndex).
  `get_level_values(0)` was returning tuples instead of field names, causing wrong column
  mapping silently.
- Cache poisoning in ingestion tests - `load_data()` calls now pass `use_cache=False` so
  a stale `~/.cache/backtesting-engine/` parquet can't bypass the mock between runs.
- Duplicate `_oscillating_data()` across three test files consolidated into `helpers.py`.

---

## [0.5.3] - 2026-04-22

### Fixed
- `data/ingestion.py` `_save_to_cache` was catching `except Exception: pass` silently.
  Changed to `except OSError` with `warnings.warn()`. A full disk now surfaces an
  explanation instead of just making every subsequent run 30 seconds slower.

### Added
- `TestReturnsFromSignals` (5 tests) - specifically exercises the hold-state semantics
  where `signal=0` must inherit the previous position, not reset to flat. Uncovered a
  latent off-by-one not covered by the integration tests.
- `TestWalkForwardInputValidation` (5 tests) - verifies `ValueError` is raised with a
  message matching "positive" for zero and negative window year values.

---

## [0.5.2] - 2026-04-21

### Fixed
- Summary `max_drawdown` was averaged across windows. Now uses the worst (minimum) value.
  A strategy with -30% dd in one window and -5% in four others was reporting -10%.
- Summary `calmar_ratio` was averaged per-window. Now computed from the stitched return
  series across all windows. Per-window averaging misses cross-window drawdowns - 10%
  loss at the end of window N followed by 10% at the start of N+1 is a -19% compound
  drawdown, not -10%.
- `metrics.py` Calmar now uses geometric compounding `prod(1+r)^(252/n) - 1` rather than
  `(1+mean(r))^252 - 1`. The arithmetic version overstates annualised return by 2–5pp at
  typical daily volatility (Jensen's inequality).
- `dashboard.py` buy-and-hold equity curve was flat in no-trade windows. Added `price_data`
  parameter so the B&H line uses actual price changes on every bar.

---

## [0.5.1] - 2026-04-20

### Fixed
- Sortino ratio was using `std(downside_returns, ddof=1)` - the dispersion *among*
  negative returns. Corrected to downside deviation `sqrt(mean(min(r-T, 0)²))` per
  Sortino & van der Meer (1991). For a strategy with small consistent losses, the old
  formula had std→0 and Sortino→∞. All previously computed Sortino values were wrong.
- Information ratio in `benchmark.py` was computing `mean(sharpe_diffs)/std(sharpe_diffs)`
  - a dimensionless "Sharpe of Sharpe differences". Corrected to
  `mean(active_returns)/std(active_returns)*sqrt(252)` per Grinold & Kahn (2000).
- Position carry-over at window boundaries: all three strategies were resetting to flat
  at the start of each test window. If the strategy was long at the training/test
  boundary it missed the opening position. Fixed by detecting boundary state from
  context window and injecting a buy at test bar 0 when entering long.

---

## [0.5.0] - 2026-04-20

### Added
- `MomentumStrategy` - time-series momentum with calibrated lookback. Grid-searches
  T ∈ {20, 40, 60, 90, 120, 180, 250} trading days per training window. Implements
  `candidate_test_returns()` so White's RC covers the full search universe.
  Signal: buy when T-day log-return crosses positive, sell when it crosses negative.
  Reference: Moskowitz, Ooi & Pedersen (2012).
- `cost_sensitivity_sweep()` parallelism - `n_workers > 1` runs each (cost, slippage)
  pair concurrently via `ProcessPoolExecutor`. Pass `n_workers=-1` for all CPUs.
  A 5×5 grid goes from ~12 minutes to ~2 minutes on an 8-core machine.
- Local Parquet caching in `~/.cache/backtesting-engine/`. First call downloads from
  yfinance; subsequent calls within 24 hours read from cache.
- `build_dashboard()` now accepts `benchmark: BenchmarkResult` - adds B&H reference
  line to the Sharpe bar panel with IR and beats-% annotation; bars coloured green/red
  vs per-window benchmark.

---

## [0.4.0] - 2026-04-19

### Added
- `benchmark.py` - `compute_benchmark()` computes buy-and-hold metrics over the same
  walk-forward windows as the strategy. Information ratio, paired t-test on per-window
  Sharpe differences, beats-benchmark fraction.

### Fixed
- Reality Check bootstrap comparison was written as
  `boot_stat >= observed_max - mean_returns.max()`, which always evaluates to `>= 0.0`
  (since `observed_max = mean_returns.max()`). Corrected to `boot_stat >= observed_max`
  per White (2000) eq. 3.3.

---

## [0.3.0] - 2026-04-19

### Added
- Interactive HTML dashboards (Plotly): equity curve, drawdown, rolling Sharpe,
  per-window Sharpe, return distribution, parameter evolution.
- `cost_sensitivity_sweep()` and cost heatmap.
- `ExecutionConfig`: configurable slippage, signal delay, transaction costs.
- White's Reality Check - stationary bootstrap corrects for data snooping across the
  MA parameter grid.
- Fisher combined p-value replaces averaged per-window p-values.

---

## [0.2.0] - 2026-04-17

### Added
- Kalman filter strategy with MLE calibration of Q and R per training window.
- Walk-forward validation orchestrator.
- Block-bootstrap Sharpe p-value.
- `pyproject.toml` with mypy strict, ruff, pytest. GitHub Actions CI.

---

## [0.1.0] - 2026-03-29

Initial version. Moving average crossover, bar-by-bar simulator with transaction costs,
Sharpe/Sortino/max drawdown/Calmar/Omega metrics, basic test suite.
