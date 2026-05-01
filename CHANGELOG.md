# Changelog

## [0.5.5] - 2026-05-01

### Fixed
- `tests/test_strategy.py` (`TestContextWindowSize::test_walk_forward_does_not_need_isinstance_for_context`):
  Used a relative `pathlib.Path("src/backtesting_engine/walk_forward.py")` that resolves
  to the working directory at test-run time. When pytest is invoked from any directory
  other than the repo root (as GitHub Actions does on a fresh checkout), the path does
  not exist and the test raises `FileNotFoundError` instead of testing anything.
  Fixed by resolving relative to `__file__`:
  `Path(__file__).parent.parent / "src/backtesting_engine/walk_forward.py"`.
  All 243 tests now pass regardless of the working directory pytest is launched from.

## [0.5.4] - 2026-04-26

### Removed
- `pyproject.toml`: Removed `matplotlib` from dependencies - it was never imported
  anywhere in `src/`. Only Plotly is used for visualisation. Eliminating the dead
  dependency reduces install size and removes a version constraint with no function.

### Documentation
- `README.md`: Filled the `## What I learned building this` section with a first-person
  account of three specific correctness bugs found during development: the circular
  boundary error in block bootstrap, the candidate_test_returns() training/test data
  conflation bug, and the _OpenPosition assert/-O flag issue.
- Architecture table: `simulator.py` entry now explicitly labelled "readable reference
  baseline" to distinguish it from the production entry point (`execution.py`).
- `## What I'd build next`: added clarifying sentence that `cost_sensitivity_sweep`
  already parallelises over (cost, slippage) pairs; the future work is parallelising
  the inner walk-forward window loop specifically.

## [0.5.0] - 2026-04-24

### Added
- `strategy/momentum.py`: `MomentumStrategy` - time-series momentum with calibrated
  lookback. Grid-searches T ∈ {20, 40, 60, 90, 120, 180, 250} trading days per training
  window. Implements `candidate_test_returns()` so White's Reality Check covers the
  full search universe. Signal: buy when T-day log-return crosses positive, sell when
  it crosses negative. Documented against Moskowitz, Ooi & Pedersen (2012).
- `execution.py`: `cost_sensitivity_sweep()` now accepts `n_workers` parameter.
  `n_workers > 1` runs each (cost, slippage) pair concurrently via
  `ProcessPoolExecutor`. Each combination is fully independent so this is
  embarrassingly parallel. Pass `n_workers=-1` to use all available CPUs.
  On an 8-core machine a 5×5 grid reduces from ~12 minutes to ~2 minutes.
- `data/ingestion.py`: Local Parquet caching in `~/.cache/backtesting-engine/`.
  First call downloads from yfinance and writes cache. Subsequent calls within
  24 hours read from cache. `use_cache=False` forces a fresh download.
  Cache is keyed by (ticker, start_date). Corrupt cache files are ignored.
- `dashboard.py`: `build_dashboard()` now accepts optional `benchmark: BenchmarkResult`.
  When provided: (1) information ratio and beats-benchmark fraction appear in the
  dashboard title, (2) a buy-and-hold reference line is added to the per-window
  Sharpe bars panel with IR and beats-% annotation, (3) bars are coloured green
  when the strategy beats the benchmark, red when it does not.
- `tests/test_momentum.py`: 30 tests covering signal logic, fit(), warmup context,
  candidate_test_returns(), and active_params().

### Changed
- `main.py`: Added Strategy 3 (time-series momentum). `_print_comparison()` updated
  to show all three strategies side by side with ✓ marking the best value per metric.
  Cost sensitivity sweep and heatmap extended to include momentum. `build_dashboard()`
  calls now pass `benchmark` argument.
- `CONTRIBUTING.md`: Updated project scope section.
- `README.md`: Removed fabricated sample output. Added momentum strategy, caching,
  and parallelism documentation. Accurate test count (204).

## [0.5.3] - 2026-04-22

### Fixed (code correctness)
- `walk_forward.py`, `main.py`: Replaced `v != v` / `v == v` NaN idioms with
  `math.isnan()` throughout. Both are correct under IEEE 754, but mixing idioms
  in the same codebase without explanation reads as inconsistency rather than
  intent.
- `walk_forward.py`, `dashboard.py`: Replaced `assert x is not None` guards
  with explicit `ValueError`. Asserts are stripped when Python runs with `-O`;
  a None here would cause a confusing AttributeError several frames away rather
  than a clear error at the source.
- `walk_forward.py`: Added input validation guard - `training_window_years` and
  `testing_window_years` must both be positive. Without this, `0` produces
  `train_days=0` and crashes differently in each strategy's `fit()`.
- `execution.py`: Changed `strategy: object` to `strategy: BaseStrategy` in
  `cost_sensitivity_sweep()`, eliminating the `# type: ignore[arg-type]`
  suppression in the inner `_run_one` closure.
- `execution.py`: Fixed circular import - `BaseStrategy` is now imported under
  `TYPE_CHECKING` with `from __future__ import annotations`, so the import is
  resolved at type-check time only. At runtime, `strategy/__init__.py` imports
  from `execution.py`, which would have created a cycle.
- `main.py` (`_save_cost_heatmap`): Split bare `except Exception` into
  `except ImportError` (plotly not installed, gives actionable message) and
  `except Exception` (unexpected failure, surfaces the type and message).

### Improved (design and documentation)
- `base.py` (`returns_from_signals`): Vectorised the position-tracking loop
  with pandas `ffill`. The stateful hold rule is equivalent to replacing 0
  (hold) with NaN and forward-filling the last buy/sell signal. Roughly 10–20×
  faster on typical window sizes; the correctness equivalence is documented
  inline with a worked example.
- `walk_forward.py` (`_fisher_combined_p`): Rewrote docstring to note the
  sensitivity to single-window strong signals and the approximate independence
  assumption.
- `execution.py`, `simulator.py`: Corrected stale docstring references to
  `run_simulation()` - the function still exists in `simulator.py` but
  `run_simulation_with_execution()` is the production entry point.
- `simulator.py`: Replaced "kept for backward compatibility" with accurate
  description - the module is retained because its explicit loop makes the
  execution model readable and because `test_simulator.py` exercises it directly.
- `strategy/moving_average.py`, `strategy/momentum.py`: Documented the
  mixed leading/trailing underscore naming convention on fitted attributes
  (`_all_candidate_pairs_`, `_all_lookbacks_`).
- `config.py`: Removed unused `FIGURE_DPI` constant (never referenced).

### Type annotations
- `strategy/kalman_filter.py`: Added type annotations to four module-level
  constants (`_MIN_VARIANCE: float`, `_OPTIM_XATOL: float`, etc.).
- `strategy/momentum.py`: Added `_LOOKBACK_GRID: list[int]` annotation.

### Error handling
- `data/ingestion.py` (`_save_to_cache`): Changed silent `except Exception: pass`
  to `except OSError` with `warnings.warn()`. Cache write failure is non-fatal
  but should be visible - a user with a full disk would otherwise see every run
  take 30+ extra seconds with no explanation.

### Testing (243 tests, +10 from previous release)
- Added `TestReturnsFromSignals` (5 tests) in `test_strategy.py`: directly
  exercises the hold-state semantics of `returns_from_signals()` - specifically
  the case where `signal=0` must inherit the previous position, not reset to flat.
- Added `TestWalkForwardInputValidation` (5 tests) in `test_walk_forward.py`:
  verifies the new input guard raises `ValueError` with a message matching
  "positive" for zero and negative window year values.
- Fixed pre-existing mock bug in `test_data/test_ingestion.py`: all three mock
  functions now build DataFrames with `pd.MultiIndex.from_tuples()` instead of
  dict-of-tuple-keys syntax. The original construction produced a plain Index of
  tuples rather than a true MultiIndex, so `get_level_values(0)` returned tuples
  instead of field names, causing wrong column mapping and silently incorrect
  test data.
- Fixed `test_data/test_ingestion.py` cache poisoning: all `load_data()` calls
  now pass `use_cache=False`, preventing a stale `~/.cache/backtesting-engine/`
  parquet from bypassing the `yf.download` mock between runs.
- Eliminated three duplicate `_oscillating_data()` definitions across
  `test_walk_forward.py`, `test_benchmark.py`, and `test_strategy.py`. Moved
  into `tests/helpers.py` (importable module) with a `with_high_low` parameter;
  pytest fixtures in `tests/conftest.py` wrap it.
- Added `pythonpath = ["tests"]` to `[tool.pytest.ini_options]` so `helpers.py`
  is importable without manipulating `sys.path` manually.

### Maintenance
- `Makefile` (`clean`): Added `rm -rf ~/.cache/backtesting-engine/` so
  `make clean` provides a complete reset including downloaded market data.

## [0.5.2] - 2026-04-21

### Fixed (correctness bugs in summary metric aggregation)
- `walk_forward.py`: **max_drawdown** in summary metrics was averaged across
  walk-forward windows. It is now the worst-case (minimum) across windows.
  A strategy with -30% dd in one window and -5% in others now correctly reports
  -30%, not the misleading average of -10%.
- `walk_forward.py`: **calmar_ratio** in summary metrics was averaged per-window.
  It is now computed from the stitched portfolio returns across all windows.
  Per-window averaging misses cross-window drawdowns - if a strategy loses 10%
  at the end of window N and another 10% at the start of window N+1, the compound
  drawdown is -19%, not -10%. The stitched Calmar captures this correctly.
- `dashboard.py`: Added `price_data` parameter to `build_dashboard()`. When the
  original close price Series is passed, the buy-and-hold equity curve uses actual
  price changes on every bar rather than approximating from trade entry/exit prices.
  Previously, no-trade windows showed a flat buy-and-hold line.
- `metrics.py`: **calmar_ratio** annualised return now uses geometric compounding
  `prod(1+r)^(252/n) - 1` rather than `(1+mean(r))^252 - 1`. Due to Jensen's
  inequality, the arithmetic compound overstates annualised return by 2-5
  percentage points annually at typical daily volatility.
- `walk_forward.py`: inf values (Calmar/Sortino when no drawdown/downside) are
  now excluded from metric averages rather than contaminating the mean.
  Display changed from bare `inf` to `∞ (no downside)` in console output.

### Tests added
- `tests/test_walk_forward.py`: `TestSummaryMetricAggregation` - 3 tests verifying
  max_dd is worst-case, Calmar is from stitched returns, Sharpe is mean.
- `tests/test_metrics.py`: `test_uses_geometric_not_arithmetic_compounding` and
  `test_geometric_vs_arithmetic_divergence` - regression tests for Calmar formula.

## [0.5.1] - 2026-04-20

### Fixed (correctness bugs - all output affected)
- `metrics.py`: **Sortino ratio** used `std(downside_returns, ddof=1)` - the standard
  deviation *among* negative returns. This inflates the ratio for strategies with
  small, consistent losses (std→0 as losses become uniform). Corrected to downside
  deviation: `sqrt(mean(min(r-T, 0)²))` per Sortino & van der Meer (1991). All
  previously computed Sortino values were inflated.
- `benchmark.py`: **Information ratio** computed `mean(sharpe_diffs)/std(sharpe_diffs)`,
  which is a dimensionless "Sharpe of Sharpe differences", not an IR. Corrected to
  `mean(active_returns)/std(active_returns)*sqrt(252)` where active return at bar t
  is `strategy_return[t] - bh_return[t]`, per Grinold & Kahn (2000). IR sign was
  correct but magnitude was not interpretable.
- `strategy/moving_average.py`, `strategy/kalman_filter.py`, `strategy/momentum.py`:
  **Position carry-over at window boundaries.** All strategies reset to flat at the
  start of each test window. If the strategy was long at the training/test boundary,
  it missed the opening position and understated returns. Fixed: each strategy detects
  its boundary state from the context window and injects a buy signal at test bar 0
  when entering long. For MA: checks if short MA > long MA at last context bar. For
  Kalman: checks if filtered velocity is positive. For momentum: checks if T-day
  log-return is positive.
- `main.py`: Added explicit caveat to Fisher combined p-value output: "approx: windows
  not fully independent". Walk-forward test windows share macro shocks and are
  positively correlated; Fisher's method assumes independence and over-rejects (gives
  p-values that are too small) in this setting.
- `reality_check.py`: Documented that the Reality Check null benchmarks against cash
  (zero return), not buy-and-hold. A strategy that beats cash but underperforms BH
  can still produce a small RC p-value. Added explanation of when you would switch
  to the buy-and-hold null.
- `strategy/moving_average.py`: Documented that grid search uses pre-cost, pre-delay
  returns (standard practice - costs unknown during optimisation).

### Tests updated
- `tests/test_metrics.py`: Updated Sortino tests to match corrected formula. Added
  `test_consistent_small_losses_not_inflated` and `test_downside_deviation_not_std`
  which specifically catch the regression.

## [0.4.0] - 2026-04-19

### Added
- `benchmark.py`: `compute_benchmark()` computes buy-and-hold metrics over the same
  walk-forward windows as the strategy. Returns `BenchmarkResult` with information
  ratio, paired t-test on per-window Sharpe differences, and beats-benchmark fraction.
- `BenchmarkResult` exposed in the public API.

### Fixed
- `reality_check.py`: Bootstrap comparison threshold was written as
  `boot_stat >= observed_max - mean_returns.max()` (always evaluates to `>= 0.0`
  because `observed_max = mean_returns.max()`). Corrected to `boot_stat >= observed_max`
  per White (2000) eq. 3.3. Under H0 with zero-mean strategies this still gives
  p ≈ 0.5 for k=1, which is correct - the equivalence only holds when observed_max ≈ 0.
  With k > 1 strategies the max-of-k order statistic makes the exact p-value data-
  dependent, which is also correct.
- `execution.py`: Removed dead variables `done = 0` and bare expression
  `len(cost_rates) * len(slippage_factors)` left from a removed progress bar.
- `walk_forward.py`: Replaced `except (ValueError, Exception)` (bare `except Exception`)
  with `except ValueError`, which is the only documented raise from
  `build_candidate_return_matrix()`.
- `data/ingestion.py`: Added `_reconcile_adjusted_close()` to handle ex-dividend
  dates where adjusted close falls slightly outside the unadjusted [low, high] band.
  Clips discrepancies < 0.5% silently; raises on larger ones.
- `config.py`: Corrected `RISK_FREE_RATE` comment - zero is appropriate for a
  long-only trend-following strategy that is frequently in cash, not because it
  "produces an upper bound" (the original comment was wrong about the direction).
- `strategy/moving_average.py`: Removed `_sharpe_of_returns` static method that
  duplicated `_sharpe` from `metrics.py`.

### Tests added
- `tests/test_benchmark.py`: 10 tests.
- `tests/test_reality_check.py`: 3 new tests in `TestRealityCheckCentering`.
- `tests/test_data/test_ingestion.py`: 3 new tests for ex-dividend reconciliation.

## [0.3.0] - 2026-04-19

### Added
- Interactive HTML dashboards (Plotly): equity curve, drawdown, rolling Sharpe,
  per-window Sharpe, return distribution, parameter evolution.
- `cost_sensitivity_sweep()`: full walk-forward at each (cost, slippage) grid point.
- `ExecutionConfig`: configurable slippage, signal delay, transaction costs.
- `active_params` on `WindowResult` for parameter drift visualisation.
- White's Reality Check (`reality_check.py`): stationary bootstrap corrects for
  data snooping across the MA parameter grid.
- Fisher combined p-value replaces averaged per-window p-values.

## [0.2.0] - 2026-04-17

### Added
- Kalman filter strategy with MLE calibration of Q and R per training window.
- Walk-forward validation orchestrator.
- Block-bootstrap Sharpe p-value.
- `pyproject.toml` with mypy strict, ruff, pytest. GitHub Actions CI.

## [0.1.0] - 2026-03-29

### Added
- Moving average crossover strategy.
- Bar-by-bar trade simulator with transaction costs.
- Sharpe, Sortino, max drawdown, Calmar, Omega metrics.
- Basic test suite.
