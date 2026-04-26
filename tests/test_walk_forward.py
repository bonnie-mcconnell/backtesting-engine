"""
Unit tests for the walk-forward orchestrator.

Tests use synthetic price data with controlled properties so that window
counts, date boundaries, and statistical properties can be verified by hand.
All tests pass window sizes explicitly rather than relying on config defaults,
so they remain valid even if config constants change.

Shared fixtures (oscillating_504, oscillating_756, strategy) and the
make_oscillating_data() helper are defined in conftest.py and auto-injected
by pytest - no import required.
"""

import numpy as np
import pandas as pd
import pytest
from helpers import make_oscillating_data

from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import _fisher_combined_p, walk_forward

# ---------------------------------------------------------------------------
# Window count
# ---------------------------------------------------------------------------

class TestWindowCount:
    def test_one_window_from_504_days(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        # 504 days → 1 window (train 0–252, test 252–504). No room for a second.
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        assert len(result.window_results) == 1

    def test_two_windows_from_756_days(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_756, strategy,
                              training_window_years=1, testing_window_years=1)
        assert len(result.valid_windows) == 2

    def test_insufficient_data_raises(self, strategy: MovingAverageStrategy) -> None:
        # 100 days cannot fit even one 1+1 year window (needs 504).
        dates = pd.date_range("2010-01-01", periods=100, freq="B")
        data = pd.DataFrame({"close": np.linspace(100.0, 110.0, 100)}, index=dates)
        with pytest.raises(ValueError):
            walk_forward(data, strategy,
                         training_window_years=1, testing_window_years=1)


# ---------------------------------------------------------------------------
# No look-ahead bias
# ---------------------------------------------------------------------------

class TestNoLookaheadBias:
    def test_test_start_always_after_train_end(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        for w in result.valid_windows:
            assert w.test_start > w.train_end

    def test_no_date_overlap_between_train_and_test(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_756, strategy,
                              training_window_years=1, testing_window_years=1)
        for w in result.valid_windows:
            assert w.test_start > w.train_end


# ---------------------------------------------------------------------------
# Window advancement
# ---------------------------------------------------------------------------

class TestWindowAdvancement:
    def test_each_window_advances_by_test_days(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_756, strategy,
                              training_window_years=1, testing_window_years=1)
        valid = result.valid_windows
        if len(valid) >= 2:
            # Second test window starts the business day after first test window ends.
            assert valid[1].test_start == valid[0].test_end + pd.offsets.BDay(1)


# ---------------------------------------------------------------------------
# BacktestResult structure
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_strategy_name(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        assert result.strategy_name == "MovingAverageStrategy"

    def test_summary_metrics_is_not_none(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        assert result.summary_metrics is not None

    def test_summary_sharpe_is_float(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        assert isinstance(result.summary_metrics.sharpe_ratio, float)

    def test_skipped_window_count_is_non_negative(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        assert result.skipped_window_count >= 0

    def test_valid_windows_property(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        for w in result.valid_windows:
            assert not w.skipped

    def test_combined_p_value_in_unit_interval(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        p = result.summary_metrics.combined_p_value
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Fisher combined p-value (unit tests, no walk_forward call needed)
# ---------------------------------------------------------------------------

class TestFisherCombinedP:
    def test_all_significant_p_values_produce_small_combined_p(self) -> None:
        # Very small per-window p-values → highly significant combined result.
        combined = _fisher_combined_p([0.01, 0.02, 0.01, 0.03])
        assert combined < 0.001

    def test_all_large_p_values_produce_large_combined_p(self) -> None:
        # High per-window p-values → combined p should also be large.
        combined = _fisher_combined_p([0.5, 0.6, 0.7, 0.5])
        assert combined > 0.05

    def test_combined_p_in_unit_interval(self) -> None:
        combined = _fisher_combined_p([0.1, 0.3, 0.05, 0.8])
        assert 0.0 <= combined <= 1.0

    def test_single_p_value(self) -> None:
        # Fisher with k=1: valid, should return a value in [0, 1].
        combined = _fisher_combined_p([0.05])
        assert 0.0 <= combined <= 1.0

    def test_more_windows_amplifies_weak_signal(self) -> None:
        # Same per-window p=0.1 becomes more significant as k grows.
        p4 = _fisher_combined_p([0.1] * 4)
        p16 = _fisher_combined_p([0.1] * 16)
        assert p16 < p4


# ---------------------------------------------------------------------------
# active_params stored on WindowResult
# ---------------------------------------------------------------------------

class TestActiveParamsStorage:
    def test_ma_windows_store_short_and_long(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        for w in result.valid_windows:
            assert "short_window" in w.active_params
            assert "long_window" in w.active_params

    def test_params_are_valid_window_sizes(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        for w in result.valid_windows:
            s = w.active_params["short_window"]
            long_w = w.active_params["long_window"]
            assert isinstance(s, int) and isinstance(long_w, int)
            assert s < long_w
            assert s > 0 and long_w > 0

    def test_param_evolution_property(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        evolution = result.param_evolution
        assert len(evolution) == len(result.valid_windows)
        for p in evolution:
            assert "short_window" in p


class TestRealityCheckWithTestReturns:
    def test_rc_p_value_is_finite_for_ma_strategy(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        import math
        assert not math.isnan(result.summary_metrics.reality_check_p_value)

    def test_rc_p_value_in_unit_interval(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(oscillating_504, strategy,
                              training_window_years=1, testing_window_years=1)
        p = result.summary_metrics.reality_check_p_value
        assert 0.0 <= p <= 1.0

    def test_rc_p_nan_for_kalman_strategy(
        self, oscillating_504: pd.DataFrame
    ) -> None:
        import math

        from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
        result = walk_forward(oscillating_504, KalmanFilterStrategy(),
                              training_window_years=1, testing_window_years=1)
        assert math.isnan(result.summary_metrics.reality_check_p_value)


# ---------------------------------------------------------------------------
# Summary metric aggregation correctness
# ---------------------------------------------------------------------------

class TestSummaryMetricAggregation:
    """
    Verify that summary metrics use the correct aggregation method for each metric.

    Key design decisions:
    - Sharpe/Sortino/Omega/p_value: mean across windows (standard walk-forward)
    - max_drawdown: worst-case (minimum) across windows, not mean
    - calmar_ratio: computed from stitched portfolio returns, not per-window mean
    """

    def test_max_drawdown_is_worst_not_mean(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_756, strategy,
            training_window_years=1, testing_window_years=1
        )
        valid = result.valid_windows
        if len(valid) < 2:
            pytest.skip("Need at least 2 windows")

        per_window_dds = [w.metrics_result.max_drawdown for w in valid]
        worst = min(per_window_dds)
        mean = float(sum(per_window_dds) / len(per_window_dds))

        # Summary must be worst-case, not mean
        assert abs(result.summary_metrics.max_drawdown - worst) < 1e-8, (
            f"Summary max_dd={result.summary_metrics.max_drawdown:.4f} "
            f"should equal worst={worst:.4f}, not mean={mean:.4f}"
        )

    def test_calmar_is_stitched_not_per_window_mean(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_756, strategy,
            training_window_years=1, testing_window_years=1
        )
        valid = result.valid_windows
        if not valid:
            pytest.skip("No valid windows")

        # Compute expected: Calmar from stitched returns
        import numpy as np

        from backtesting_engine.metrics import _calmar
        all_rets = []
        for w in valid:
            pv = w.simulation_result.portfolio_values
            if pv is not None and len(pv) > 1:
                all_rets.append(pv.pct_change().dropna().to_numpy())

        if not all_rets:
            pytest.skip("No portfolio returns")

        stitched = np.concatenate(all_rets)
        expected_calmar = _calmar(stitched)

        assert abs(result.summary_metrics.calmar_ratio - expected_calmar) < 1e-6, (
            f"Summary Calmar={result.summary_metrics.calmar_ratio:.4f} "
            f"should equal stitched={expected_calmar:.4f}"
        )

    def test_sharpe_is_mean_not_stitched(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        # Sharpe SHOULD be mean of per-window Sharpes (standard walk-forward protocol)
        result = walk_forward(
            oscillating_756, strategy,
            training_window_years=1, testing_window_years=1
        )
        valid = result.valid_windows
        if not valid:
            pytest.skip("No valid windows")

        import numpy as np
        per_sharpes = [w.metrics_result.sharpe_ratio for w in valid]
        expected = float(np.mean(per_sharpes))
        assert abs(result.summary_metrics.sharpe_ratio - expected) < 1e-8


# ---------------------------------------------------------------------------
# Input validation - window year parameters
# ---------------------------------------------------------------------------

class TestWalkForwardInputValidation:
    """
    Verify that walk_forward raises early with a clear error message when
    called with invalid window year parameters.

    Motivation: without this guard, training_window_years=0 produces
    train_days=0, strategy.fit() receives an empty DataFrame, and each
    strategy crashes with a different, confusing error deep in its own code.
    A single ValueError at the entry point is much easier to debug.
    """

    def test_zero_training_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=0, testing_window_years=1)

    def test_zero_testing_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=1, testing_window_years=0)

    def test_negative_training_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=-1, testing_window_years=1)

    def test_negative_testing_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=1, testing_window_years=-1)

    def test_valid_years_do_not_raise(self, strategy: MovingAverageStrategy) -> None:
        # Sanity check: the guard must not fire on legitimate inputs.
        data = make_oscillating_data(756)
        # Should not raise - may produce no valid windows but must not error on the guard.
        try:
            walk_forward(data, strategy, training_window_years=1, testing_window_years=1)
        except ValueError as e:
            assert "positive" not in str(e), (
                f"Guard fired on valid inputs: {e}"
            )
