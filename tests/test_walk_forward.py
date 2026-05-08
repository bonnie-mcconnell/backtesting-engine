"""
Unit tests for the walk-forward orchestrator.

All walk_forward calls use execution=_ZERO_FRICTION (slippage=0, delay=0)
because the synthetic data only has a 'close' column.  Tests that specifically
exercise the execution model are in test_execution.py which uses OHLCV fixtures.

Shared fixtures (oscillating_504, oscillating_756, strategy) and the
make_oscillating_data() helper are defined in conftest.py and auto-injected
by pytest - no import required.
"""

import math

import numpy as np
import pandas as pd
import pytest
from helpers import make_oscillating_data

from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import (
    _fisher_combined_p,
    _flat_cash_metrics,
    walk_forward,
)

# Zero-friction config for unit tests using close-only synthetic data.
_ZERO_FRICTION = ExecutionConfig(slippage_factor=0.0, signal_delay=0)


class TestWindowCount:
    def test_one_window_from_504_days(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        assert len(result.window_results) == 1

    def test_two_windows_from_756_days(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_756, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        assert len(result.valid_windows) == 2

    def test_insufficient_data_raises(self, strategy: MovingAverageStrategy) -> None:
        dates = pd.date_range("2010-01-01", periods=100, freq="B")
        data = pd.DataFrame({"close": np.linspace(100.0, 110.0, 100)}, index=dates)
        with pytest.raises(ValueError):
            walk_forward(
                data, strategy,
                training_window_years=1, testing_window_years=1,
                execution=_ZERO_FRICTION,
            )


class TestNoLookaheadBias:
    def test_test_start_always_after_train_end(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        for w in result.valid_windows:
            assert w.test_start > w.train_end

    def test_no_date_overlap_between_train_and_test(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_756, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        for w in result.valid_windows:
            assert w.test_start > w.train_end


class TestWindowAdvancement:
    def test_each_window_advances_by_test_days(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_756, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        valid = result.valid_windows
        if len(valid) >= 2:
            assert valid[1].test_start == valid[0].test_end + pd.offsets.BDay(1)


class TestBacktestResult:
    def test_strategy_name(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        assert result.strategy_name == "MovingAverageStrategy"

    def test_summary_metrics_is_not_none(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        assert result.summary_metrics is not None

    def test_summary_sharpe_is_float(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        assert isinstance(result.summary_metrics.sharpe_ratio, float)

    def test_flat_cash_window_count_is_non_negative(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        # Renamed from skipped_window_count: flat-cash windows are valid results.
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        assert result.flat_cash_window_count >= 0

    def test_valid_windows_property(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        for w in result.valid_windows:
            assert not w.skipped

    def test_combined_p_value_in_unit_interval(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        p = result.summary_metrics.combined_p_value
        assert 0.0 <= p <= 1.0


class TestFisherCombinedP:
    def test_all_significant_p_values_produce_small_combined_p(self) -> None:
        combined = _fisher_combined_p([0.01, 0.02, 0.01, 0.03])
        assert combined < 0.001

    def test_all_large_p_values_produce_large_combined_p(self) -> None:
        combined = _fisher_combined_p([0.5, 0.6, 0.7, 0.5])
        assert combined > 0.05

    def test_combined_p_in_unit_interval(self) -> None:
        combined = _fisher_combined_p([0.1, 0.3, 0.05, 0.8])
        assert 0.0 <= combined <= 1.0

    def test_single_p_value(self) -> None:
        combined = _fisher_combined_p([0.05])
        assert 0.0 <= combined <= 1.0

    def test_more_windows_amplifies_weak_signal(self) -> None:
        p4 = _fisher_combined_p([0.1] * 4)
        p16 = _fisher_combined_p([0.1] * 16)
        assert p16 < p4


class TestFlatCashMetrics:
    """A no-trade window is valid cash-holding, not a skipped window."""

    def test_sharpe_is_zero(self) -> None:
        assert _flat_cash_metrics().sharpe_ratio == 0.0

    def test_drawdown_is_zero(self) -> None:
        assert _flat_cash_metrics().max_drawdown == 0.0

    def test_p_value_is_one(self) -> None:
        assert _flat_cash_metrics().p_value == 1.0

    def test_exposure_is_zero(self) -> None:
        assert _flat_cash_metrics().exposure_fraction == 0.0

    def test_trade_count_is_zero(self) -> None:
        assert _flat_cash_metrics().trade_count == 0

    def test_sortino_is_zero_not_inf(self) -> None:
        # Must be 0.0, not inf. inf is excluded from summary means, which would
        # silently drop flat-cash windows from aggregate Sortino - overstating it.
        assert _flat_cash_metrics().sortino_ratio == 0.0

    def test_omega_is_one_not_inf(self) -> None:
        # 1.0 = neutral (no gains, no losses). inf causes same silent exclusion problem.
        assert _flat_cash_metrics().omega_ratio == 1.0


class TestActiveParamsStorage:
    def test_ma_windows_store_short_and_long(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        for w in result.valid_windows:
            assert "short_window" in w.active_params
            assert "long_window" in w.active_params

    def test_params_are_valid_window_sizes(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        for w in result.valid_windows:
            s = w.active_params["short_window"]
            long_w = w.active_params["long_window"]
            assert isinstance(s, int) and isinstance(long_w, int)
            assert s < long_w
            assert s > 0 and long_w > 0

    def test_param_evolution_property(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        evolution = result.param_evolution
        assert len(evolution) == len(result.valid_windows)
        for p in evolution:
            assert "short_window" in p


class TestRealityCheckWithTestReturns:
    def test_rc_p_value_is_finite_for_ma_strategy(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        assert not math.isnan(result.summary_metrics.reality_check_p_value)

    def test_rc_p_value_in_unit_interval(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        p = result.summary_metrics.reality_check_p_value
        assert 0.0 <= p <= 1.0

    def test_rc_p_nan_for_kalman_strategy(
        self, oscillating_504: pd.DataFrame
    ) -> None:
        from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
        result = walk_forward(
            oscillating_504, KalmanFilterStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        assert math.isnan(result.summary_metrics.reality_check_p_value)

    def test_bootstrap_seed_produces_deterministic_results(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        """Same seed must produce identical p-values every run."""
        r1 = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
            bootstrap_seed=42,
        )
        r2 = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
            bootstrap_seed=42,
        )
        assert r1.summary_metrics.combined_p_value == r2.summary_metrics.combined_p_value
        assert r1.summary_metrics.reality_check_p_value == r2.summary_metrics.reality_check_p_value


class TestSummaryMetricAggregation:
    def test_max_drawdown_is_worst_not_mean(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_756, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        valid = result.valid_windows
        if len(valid) < 2:
            pytest.skip("Need at least 2 windows")
        per_window_dds = [w.metrics_result.max_drawdown for w in valid]
        worst = min(per_window_dds)
        mean = float(sum(per_window_dds) / len(per_window_dds))
        assert abs(result.summary_metrics.max_drawdown - worst) < 1e-8, (
            f"max_dd={result.summary_metrics.max_drawdown:.4f} should equal worst={worst:.4f}, not mean={mean:.4f}"
        )

    def test_calmar_is_stitched_not_per_window_mean(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_756, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        valid = result.valid_windows
        if not valid:
            pytest.skip("No valid windows")
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
        assert abs(result.summary_metrics.calmar_ratio - expected_calmar) < 1e-6

    def test_sharpe_is_mean_not_stitched(
        self, oscillating_756: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        result = walk_forward(
            oscillating_756, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        valid = result.valid_windows
        if not valid:
            pytest.skip("No valid windows")
        per_sharpes = [w.metrics_result.sharpe_ratio for w in valid]
        expected = float(np.mean(per_sharpes))
        assert abs(result.summary_metrics.sharpe_ratio - expected) < 1e-8


class TestWalkForwardInputValidation:
    def test_zero_training_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=0, testing_window_years=1,
                         execution=_ZERO_FRICTION)

    def test_zero_testing_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=1, testing_window_years=0,
                         execution=_ZERO_FRICTION)

    def test_negative_training_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=-1, testing_window_years=1,
                         execution=_ZERO_FRICTION)

    def test_negative_testing_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=1, testing_window_years=-1,
                         execution=_ZERO_FRICTION)

    def test_valid_years_do_not_raise(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        try:
            walk_forward(data, strategy, training_window_years=1, testing_window_years=1,
                         execution=_ZERO_FRICTION)
        except ValueError as e:
            assert "positive" not in str(e)
