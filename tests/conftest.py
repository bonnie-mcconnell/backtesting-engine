"""
Shared pytest fixtures for the backtesting-engine test suite.

Fixtures defined here are auto-discovered by pytest and injected into any
test module that declares them as parameters - no import required.

Shared data-generation helpers (make_oscillating_data) live in helpers.py,
which is a regular importable module. They are imported here for use in
fixtures, and imported directly by test modules that call them inline.

Bootstrap permutation count
----------------------------
N_PERMUTATIONS=10_000 in config.py is correct for production runs: it gives
p-value estimates accurate to ±0.003 at p=0.05. In tests, we only need to
verify that p-values fall in [0,1], respond correctly to high/low-drift
inputs, and are reproducible given a fixed seed. 200 permutations is more
than sufficient for all of these properties while being ~50× faster.

The autouse session-scoped fixture below patches N_PERMUTATIONS globally for
the entire test session. It targets both sites where the value is read at
call time:
  - backtesting_engine.metrics   (_monte_carlo_p_value reads N_PERMUTATIONS
    as a bare name in the function body - evaluated at call time, not import)
  - backtesting_engine.reality_check (white_reality_check reads N_PERMUTATIONS
    in the function body via `n_iters = n_bootstrap if n_bootstrap is not None
    else N_PERMUTATIONS` - also at call time)

IMPORTANT: white_reality_check formerly had `n_bootstrap: int = N_PERMUTATIONS`
as a default argument, which Python evaluates once at function-definition time
(module import), not at call time. That meant patching the module attribute had
no effect. The signature was changed to `n_bootstrap: int | None = None` with
the module attribute read in the function body, so the patch now works.

Without this patch, 49 walk_forward() calls × ~3 windows × 10,000 iterations
each (both bootstrap AND RC) = ~1.47M bootstrap iterations, taking 90+ minutes
across two Python versions in CI. With 200 permutations the full suite runs in
a few minutes locally and comfortably inside the CI timeout.

Cached walk_forward results
----------------------------
walk_forward() is the most expensive operation in the test suite. Even with
pre-fixed MA windows (no grid search) and 200 bootstrap permutations, each
call takes ~1-2s. Many test classes merely inspect properties of a result
rather than verifying the call itself - they can share a cached result.

wf_result_504 and wf_result_756 are module-scoped fixtures that run
walk_forward once per test module that requests them. Tests in the same module
that receive the same fixture instance get a shared result - no redundant calls.

module scope (not session scope) because:
  1. session-scoped fixtures must be defined before all tests run, and
     some tests (test_correctness_invariants.py) intentionally run different
     walk_forward variants for correctness checks.
  2. module scope still eliminates the vast majority of redundant calls:
     test_walk_forward.py has 28 tests that all use the same two results.
"""

import pandas as pd
import pytest
from helpers import make_oscillating_data

import backtesting_engine.metrics as _metrics_module
import backtesting_engine.reality_check as _rc_module
from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.models import BacktestResult
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward

# Number of bootstrap permutations used during tests.
# Production value is 10_000 (config.py). Tests need ~200 to verify
# statistical correctness (direction, range, reproducibility) without
# spending 90+ minutes on Monte Carlo sampling.
_TEST_N_PERMUTATIONS = 200

# Zero-friction execution config for tests using close-only synthetic data.
# Slippage requires high/low columns; delay shifts signals by N bars.
# Correctness of those features is tested in test_execution.py with OHLCV data.
_ZERO_FRICTION = ExecutionConfig(slippage_factor=0.0, signal_delay=0)


@pytest.fixture(autouse=True, scope="session")
def _patch_n_permutations() -> None:
    """
    Globally reduce bootstrap permutations for the test session.

    autouse=True: applied to every test automatically with no opt-in required.
    scope="session": patched once for the full run, not reset between tests.

    Both functions read N_PERMUTATIONS as a module-level name in the function
    body (not as a frozen default argument), so patching the module attribute
    takes effect on every subsequent call.

    The _monte_carlo_p_value function in metrics.py always used this pattern.
    white_reality_check in reality_check.py originally had N_PERMUTATIONS as a
    default argument (evaluated at import time), which made patching ineffective.
    That was fixed by changing the signature to `n_bootstrap: int | None = None`
    with the module attribute read inside the function body.

    The original value is not restored because the test process exits after
    the session; restoring it would add complexity with no benefit.
    """
    _metrics_module.N_PERMUTATIONS = _TEST_N_PERMUTATIONS
    _rc_module.N_PERMUTATIONS = _TEST_N_PERMUTATIONS


@pytest.fixture
def oscillating_504() -> pd.DataFrame:
    """504 business days - fits exactly one 1+1yr walk-forward window."""
    return make_oscillating_data(504)


@pytest.fixture
def oscillating_756() -> pd.DataFrame:
    """756 business days - fits exactly two 1+1yr walk-forward windows."""
    return make_oscillating_data(756)


@pytest.fixture
def oscillating_756_ohlc() -> pd.DataFrame:
    """756 business days with high/low columns - used by benchmark tests."""
    return make_oscillating_data(756, with_high_low=True)


@pytest.fixture
def strategy() -> MovingAverageStrategy:
    """
    Fixed-window MovingAverageStrategy for tests that need a strategy instance.

    Short/long windows are set explicitly so fit() skips the full grid search
    and tests run in seconds rather than minutes. Only use this fixture when
    testing the walk-forward orchestrator or benchmark - tests that exercise
    fit() itself should construct MovingAverageStrategy() with no arguments.
    """
    return MovingAverageStrategy(short_window=20, long_window=50)


@pytest.fixture(scope="module")
def wf_result_504() -> BacktestResult:
    """
    Cached walk_forward result for 504-day data with fixed MA(20,50) strategy.

    scope="module": computed once when the first test in a module requests it,
    then reused for all subsequent tests in that module. This eliminates ~15
    redundant walk_forward calls in test_walk_forward.py that all inspect
    properties of the same result.

    Fixed MA windows: bypasses grid search so fit() is ~instant.
    Zero friction: synthetic data has only 'close', no high/low for slippage.
    Seed=42: deterministic bootstrap p-values across all test runs.

    Do NOT use this fixture in tests that are specifically testing the
    walk_forward call itself (bad-input validation, seed reproducibility,
    strategy-name storage) - those need their own calls to verify behaviour.
    """
    data = make_oscillating_data(504)
    strat = MovingAverageStrategy(short_window=20, long_window=50)
    return walk_forward(
        data, strat,
        training_window_years=1,
        testing_window_years=1,
        execution=_ZERO_FRICTION,
        bootstrap_seed=42,
    )


@pytest.fixture(scope="module")
def wf_result_756() -> BacktestResult:
    """
    Cached walk_forward result for 756-day data with fixed MA(20,50) strategy.

    Same rationale as wf_result_504. 756 days produces two walk-forward
    windows, which is the minimum needed to test:
      - window advancement (test_start[1] == test_end[0] + 1 BDay)
      - summary aggregation (max DD is min of per-window DDs, not mean)
      - Calmar stitching vs per-window mean distinction
      - Fisher combined p across 2 windows
    """
    data = make_oscillating_data(756)
    strat = MovingAverageStrategy(short_window=20, long_window=50)
    return walk_forward(
        data, strat,
        training_window_years=1,
        testing_window_years=1,
        execution=_ZERO_FRICTION,
        bootstrap_seed=42,
    )
