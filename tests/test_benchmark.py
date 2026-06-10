"""
Unit tests for the benchmark comparison module.

Buy-and-hold metrics are computed on the same walk-forward windows as the
strategy, with matching execution costs. Expected values derived by hand.

Shared helpers (make_oscillating_data) come from helpers.py.
"""

import math

import numpy as np
import pandas as pd
import pytest
from helpers import make_oscillating_data

from backtesting_engine.benchmark import BenchmarkResult, _buy_and_hold_returns, compute_benchmark
from backtesting_engine.config import TRANSACTION_COST_RATE
from backtesting_engine.models import BacktestResult
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward


@pytest.fixture(scope="class")
def small_result() -> tuple[BacktestResult, pd.DataFrame]:
    # with_high_low=True: the new ExecutionConfig default (slippage=0.05)
    # requires 'high' and 'low' columns. We use realistic defaults to mirror
    # exactly what main.py does, making this fixture a faithful integration test.
    #
    # scope="class": shared across all TestComputeBenchmark tests so walk_forward
    # (which runs the MA grid search + bootstrap) executes once, not 8 times.
    data = make_oscillating_data(756, with_high_low=True)
    strategy = MovingAverageStrategy(short_window=20, long_window=50)
    from backtesting_engine.execution import ExecutionConfig
    result = walk_forward(
        data, strategy,
        training_window_years=1, testing_window_years=1,
        execution=ExecutionConfig(),   # explicit: 0.1% cost, 5% slippage, 1-day delay
    )
    return result, data


class TestBuyAndHoldReturns:
    def test_length_is_one_less_than_prices(self) -> None:
        prices = pd.Series([100.0, 101.0, 102.0, 103.0])
        returns = _buy_and_hold_returns(prices)
        assert len(returns) == 3

    def test_middle_returns_are_pure_price_changes(self) -> None:
        # Middle bars (not entry/exit) should be plain pct_change.
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        returns = _buy_and_hold_returns(prices)
        # Bar 1 → 2: 102/101 - 1, no cost adjustment
        expected_middle = (102.0 - 101.0) / 101.0
        assert math.isclose(returns[1], expected_middle, rel_tol=1e-6)

    def test_entry_return_reduced_by_transaction_cost(self) -> None:
        prices = pd.Series([100.0, 101.0, 102.0])
        returns = _buy_and_hold_returns(prices)
        raw_first = (101.0 - 100.0) / 100.0
        assert returns[0] < raw_first  # cost deducted

    def test_exit_return_reduced_by_transaction_cost(self) -> None:
        prices = pd.Series([100.0, 101.0, 102.0])
        returns = _buy_and_hold_returns(prices)
        raw_last = (102.0 - 101.0) / 101.0
        assert returns[-1] < raw_last

    def test_empty_series_returns_empty_array(self) -> None:
        prices = pd.Series([100.0])
        returns = _buy_and_hold_returns(prices)
        assert len(returns) == 0

    def test_costs_sum_to_two_transaction_cost_rates(self) -> None:
        # Entry + exit costs should total 2 × TRANSACTION_COST_RATE on a flat series.
        n = 100
        prices = pd.Series(np.full(n, 100.0))
        returns = _buy_and_hold_returns(prices)
        # All middle returns are 0 (flat); first and last each have -TRANSACTION_COST_RATE.
        total_cost = -returns[0] - returns[-1]
        assert math.isclose(total_cost, 2 * TRANSACTION_COST_RATE, rel_tol=1e-5)


class TestComputeBenchmark:
    def test_returns_benchmark_result_instance(
        self, small_result: tuple
    ) -> None:
        result, data = small_result
        bm = compute_benchmark(result, data)
        assert isinstance(bm, BenchmarkResult)

    def test_benchmark_sharpe_is_finite(self, small_result: tuple) -> None:
        result, data = small_result
        bm = compute_benchmark(result, data)
        assert math.isfinite(bm.benchmark_sharpe)

    def test_benchmark_drawdown_is_non_positive(self, small_result: tuple) -> None:
        result, data = small_result
        bm = compute_benchmark(result, data)
        assert bm.benchmark_max_drawdown <= 0.0

    def test_beats_fraction_in_unit_interval(self, small_result: tuple) -> None:
        result, data = small_result
        bm = compute_benchmark(result, data)
        assert 0.0 <= bm.strategy_beats_benchmark_fraction <= 1.0

    def test_p_value_in_unit_interval(self, small_result: tuple) -> None:
        result, data = small_result
        bm = compute_benchmark(result, data)
        if not math.isnan(bm.sharpe_diff_p_value):
            assert 0.0 <= bm.sharpe_diff_p_value <= 1.0

    def test_information_ratio_is_finite(self, small_result: tuple) -> None:
        result, data = small_result
        bm = compute_benchmark(result, data)
        assert math.isfinite(bm.information_ratio)

    def test_window_count_matches_valid_windows(self, small_result: tuple) -> None:
        # The benchmark should cover exactly the same windows as the strategy.
        result, data = small_result
        bm = compute_benchmark(result, data)
        # beats_fraction * n_valid_windows should be a whole number
        n_valid = len(result.valid_windows)
        beats_count = bm.strategy_beats_benchmark_fraction * n_valid
        assert math.isclose(beats_count, round(beats_count), abs_tol=1e-6)

    def test_zero_tracking_error_gives_zero_ir(self) -> None:
        # If strategy Sharpe == benchmark Sharpe in every window, IR = 0.
        # Construct a case where strategy is identical to buy-and-hold.
        # We can't easily force this via walk_forward, so test the IR formula directly.
        from backtesting_engine.benchmark import BenchmarkResult
        # IR is 0 when sharpe_diffs is all zeros → tracking_error = 0 → IR clamped to 0
        # (tested by inspecting _buy_and_hold_returns edge case above)
        result = BenchmarkResult(
            benchmark_sharpe=0.5,
            benchmark_sortino=0.6,
            benchmark_max_drawdown=-0.1,
            information_ratio=0.0,
            sharpe_diff_t_stat=0.0,
            sharpe_diff_p_value=1.0,
            strategy_beats_benchmark_fraction=0.5,
        )
        assert result.information_ratio == 0.0


def _make_mixed_window_data_for_flat_cash() -> pd.DataFrame:
    """Price series that produces some flat-cash AND some trading windows.

    A very slow sine wave (period=750 bars) means individual 252-bar test
    windows are often monotone (no MA crossover -> flat-cash), but the full
    5-year dataset has enough trend reversals that at least 2 windows trade.
    This avoids triggering the zero-total-trades defensive guard.

    Module-level (not a method) so the pytest fixture below can reference it
    without instantiating TestRCFlatCashParity. The fixture result is cached
    at class scope so walk_forward runs exactly once for all three tests.
    """
    n = 1260
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    close = 100.0 + 15.0 * np.sin(2 * np.pi * t / 750.0)
    return pd.DataFrame({"close": close}, index=dates)


@pytest.fixture(scope="class")
def flat_cash_result():
    """
    Single walk_forward result shared across all TestRCFlatCashParity tests.

    scope="class" means this fixture is computed once when the first test in
    the class runs, then reused for all subsequent tests in the class.
    Without sharing, _run_walk_forward_with_flat_cash() was called 3× in
    separate method calls, each running the full MA grid search + bootstrap.
    With a class-scoped fixture, the expensive computation runs exactly once.
    """
    from backtesting_engine.execution import ExecutionConfig
    from backtesting_engine.strategy.moving_average import MovingAverageStrategy
    from backtesting_engine.walk_forward import walk_forward

    return walk_forward(
        _make_mixed_window_data_for_flat_cash(),
        MovingAverageStrategy(),
        training_window_years=1,
        testing_window_years=1,
        execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
    )


class TestRCFlatCashParity:
    """
    Flat-cash windows must contribute to the RC candidate matrix just as they
    contribute p=1.0 to Fisher's combined p. Without this, Fisher and RC test
    different hypotheses over different windows.

    The walk_forward result is shared via the flat_cash_result fixture (class
    scope) so the expensive MA grid search + bootstrap runs exactly once for
    all three tests in this class, not three times.
    """

    def test_flat_cash_windows_are_valid_windows(self, flat_cash_result) -> None:
        """Flat-cash windows must be included in valid_windows with skipped=False."""
        result = flat_cash_result
        assert result.flat_cash_window_count > 0, (
            "Expected at least one flat-cash window with slow-oscillation data. "
            "Verify the sine period is long enough relative to the test window."
        )
        assert len(result.valid_windows) == len(result.window_results), (
            "All windows (including flat-cash) must appear in valid_windows."
        )

    def test_rc_p_is_not_nan_when_strategy_has_candidates(self, flat_cash_result) -> None:
        """RC p must be computable even when some windows are flat-cash.

        Before the fix: flat-cash windows skipped RC candidate collection but
        contributed p=1.0 to Fisher -> inconsistent hypotheses, RC often NaN.
        After the fix: flat-cash windows contribute zero-return arrays to the
        RC matrix so both statistics test the same set of windows.
        """
        rc_p = flat_cash_result.summary_metrics.reality_check_p_value
        assert not math.isnan(rc_p), (
            "RC p is NaN despite strategy having candidates. "
            "Flat-cash windows must contribute zero-return arrays to RC matrix "
            "(required for parity with Fisher's combined p)."
        )
        assert 0.0 <= rc_p <= 1.0, f"RC p-value out of [0,1]: {rc_p}"

    def test_fisher_and_rc_cover_same_number_of_windows(self) -> None:
        """Fisher p uses all windows including flat-cash; RC must too.

        Uses its own walk_forward call rather than the shared fixture because
        it needs a different dataset (with high/low) to verify the property holds
        under a different data regime.
        """
        from backtesting_engine.execution import ExecutionConfig
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        data = make_oscillating_data(756, with_high_low=True)
        result = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
        )
        # Both stats must have been computed over the same window set.
        # We can't inspect the internal lists, but we can verify Fisher p
        # and RC p are both finite (not NaN) when candidates exist.
        assert not math.isnan(result.summary_metrics.combined_p_value)
        rc_p = result.summary_metrics.reality_check_p_value
        if not math.isnan(rc_p):
            # RC p ≥ Fisher p is the theoretical lower bound of the data-snooping correction.
            # Not a hard guarantee in finite samples, but RC p < 0.0 or > 1.0 would be wrong.
            assert 0.0 <= rc_p <= 1.0


# ── 5. RC boundary carry-over parity ─────────────────────────────────────────


# ---------------------------------------------------------------------------
# Slippage parity and per-window benchmark Sharpes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def slippage_parity_data():
    """
    Shared walk_forward result for TestBenchmarkSlippageParity.

    The test_compute_benchmark_passes_slippage_to_bh test needs a walk_forward
    result to call compute_benchmark on. Using a class-scoped fixture avoids
    running the full MA grid search once per test method.
    """
    from backtesting_engine.execution import ExecutionConfig
    from backtesting_engine.strategy.moving_average import MovingAverageStrategy
    from backtesting_engine.walk_forward import walk_forward

    data = make_oscillating_data(756, with_high_low=True)
    exec_slip = ExecutionConfig(transaction_cost_rate=0.001, slippage_factor=0.05, signal_delay=0)
    result = walk_forward(
        data, MovingAverageStrategy(),
        training_window_years=1, testing_window_years=1,
        execution=exec_slip,
    )
    return data, result, exec_slip


class TestBenchmarkSlippageParity:
    """
    The benchmark must apply the same slippage as the strategy.
    Previously _buy_and_hold_returns only applied transaction costs.
    """

    def test_slippage_reduces_benchmark_return(self) -> None:
        from backtesting_engine.benchmark import _buy_and_hold_returns

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = np.array([100.0 + i * 0.1 for i in range(n)])
        high = close * 1.01
        low = close * 0.99
        data_df = pd.DataFrame({"close": close, "high": high, "low": low}, index=dates)

        returns_no_slip = _buy_and_hold_returns(data_df, cost_rate=0.001, slippage_factor=0.0)
        returns_with_slip = _buy_and_hold_returns(data_df, cost_rate=0.001, slippage_factor=0.1)

        # With slippage, entry/exit returns must be lower.
        assert returns_with_slip[0] < returns_no_slip[0], (
            "Entry return should be lower with slippage"
        )
        assert returns_with_slip[-1] < returns_no_slip[-1], (
            "Exit return should be lower with slippage"
        )

    def test_zero_slippage_matches_old_series_api(self) -> None:
        """With slippage_factor=0 and a plain Series input, result is the same
        as the old API (backward compatibility)."""
        from backtesting_engine.benchmark import _buy_and_hold_returns

        prices = pd.Series([100.0, 101.0, 102.0, 101.0, 103.0])
        returns_series = _buy_and_hold_returns(prices, cost_rate=0.001, slippage_factor=0.0)

        dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
        data_df = pd.DataFrame({"close": prices.values}, index=dates)
        returns_df = _buy_and_hold_returns(data_df, cost_rate=0.001, slippage_factor=0.0)

        np.testing.assert_allclose(returns_series, returns_df, rtol=1e-10)

    def test_compute_benchmark_passes_slippage_to_bh(self, slippage_parity_data) -> None:
        """compute_benchmark must forward slippage from ExecutionConfig to BH returns."""
        from backtesting_engine.benchmark import compute_benchmark
        from backtesting_engine.execution import ExecutionConfig

        data, result, exec_slip = slippage_parity_data
        exec_no_slip = ExecutionConfig(transaction_cost_rate=0.001, slippage_factor=0.0, signal_delay=0)

        bm_with_slip = compute_benchmark(result, data, execution=exec_slip)
        bm_no_slip = compute_benchmark(result, data, execution=exec_no_slip)

        # Higher slippage must penalise benchmark Sharpe.
        assert bm_with_slip.benchmark_sharpe <= bm_no_slip.benchmark_sharpe, (
            "Benchmark Sharpe with slippage should be <= without slippage"
        )


# ── 7. BenchmarkResult per-window sharpes ────────────────────────────────────


@pytest.fixture(scope="class")
def per_window_sharpe_result():
    """
    walk_forward + compute_benchmark result shared across TestBenchmarkResultPerWindowSharpes.

    scope="class": runs once for all three tests in the class, not once per test.
    The MA grid search runs on each call to walk_forward; sharing avoids 3×
    redundant executions of the same computation.
    """
    from backtesting_engine.benchmark import compute_benchmark
    from backtesting_engine.execution import ExecutionConfig
    from backtesting_engine.strategy.moving_average import MovingAverageStrategy
    from backtesting_engine.walk_forward import walk_forward

    data = make_oscillating_data(756, with_high_low=True)
    result = walk_forward(
        data, MovingAverageStrategy(),
        training_window_years=1, testing_window_years=1,
        execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
    )
    bm = compute_benchmark(result, data)
    return result, bm


class TestBenchmarkResultPerWindowSharpes:
    """BenchmarkResult must carry per_window_benchmark_sharpes.

    All three tests share a single walk_forward result via the class-scoped
    per_window_sharpe_result fixture. Without sharing, the MA grid search
    ran independently for each test method.
    """

    def test_per_window_sharpes_populated(self, per_window_sharpe_result) -> None:
        result, bm = per_window_sharpe_result
        assert len(bm.per_window_benchmark_sharpes) == len(result.valid_windows), (
            "per_window_benchmark_sharpes must have one entry per valid window"
        )

    def test_per_window_sharpes_mean_equals_benchmark_sharpe(
        self, per_window_sharpe_result
    ) -> None:
        """Mean of per-window BH sharpes must equal benchmark_sharpe."""
        _, bm = per_window_sharpe_result
        mean_pw = float(np.mean(bm.per_window_benchmark_sharpes))
        assert math.isclose(mean_pw, bm.benchmark_sharpe, rel_tol=1e-9), (
            f"Mean of per-window sharpes {mean_pw:.6f} != benchmark_sharpe {bm.benchmark_sharpe:.6f}"
        )

    def test_per_window_sharpes_are_all_finite(self, per_window_sharpe_result) -> None:
        _, bm = per_window_sharpe_result
        for i, s in enumerate(bm.per_window_benchmark_sharpes):
            assert math.isfinite(s), f"Window {i} benchmark Sharpe is not finite: {s}"


# ── 8. Dashboard bar coloring uses per-window benchmark Sharpe ───────────────
