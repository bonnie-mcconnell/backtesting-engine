"""
Unit tests for the walk-forward orchestrator.

Tests use synthetic price data with controlled properties so that window
counts, date boundaries, and statistical properties can be verified by hand.
All tests pass window sizes explicitly rather than relying on config defaults,
so they remain valid even if config constants change.

Fixture design note:
    Monotonically trending data is NOT used here. A perfectly linear price
    series has exactly one golden cross in its entire history. If that crossover
    falls in the training period, the test window produces zero signals and the
    window is skipped. Oscillating data (sinusoidal with slight upward drift)
    guarantees multiple crossovers across the full history, ensuring at least
    one crossover fires in each test window.
"""

import numpy as np
import pandas as pd
import pytest

from backtesting_engine.walk_forward import walk_forward, _fisher_combined_p
from backtesting_engine.strategy.moving_average import MovingAverageStrategy


def _oscillating_data(n: int, start: str = "2010-01-01") -> pd.DataFrame:
    """
    Sinusoidal prices with a slight upward trend.

    Period chosen so multiple golden/death crosses appear in every 252-day
    test window regardless of where the window starts.
    """
    dates = pd.date_range(start, periods=n, freq="B")
    t = np.linspace(0, 20 * np.pi, n)  # dense enough that fit()-selected windows still cross
    prices = 100.0 + 20.0 * np.sin(t) + 0.05 * np.arange(n)
    return pd.DataFrame({"close": prices}, index=dates)


@pytest.fixture
def oscillating_504() -> pd.DataFrame:
    """504 business days of oscillating prices - fits exactly one 1+1yr window."""
    return _oscillating_data(504)


@pytest.fixture
def oscillating_756() -> pd.DataFrame:
    """756 business days of oscillating prices - fits exactly two 1+1yr windows."""
    return _oscillating_data(756)


@pytest.fixture
def strategy() -> MovingAverageStrategy:
    """Fixed short/long windows so fit() grid search doesn't dominate test time."""
    return MovingAverageStrategy(short_window=20, long_window=50)


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
            l = w.active_params["long_window"]
            assert isinstance(s, int) and isinstance(l, int)
            assert s < l
            assert s > 0 and l > 0

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
        from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
        import math
        result = walk_forward(oscillating_504, KalmanFilterStrategy(),
                              training_window_years=1, testing_window_years=1)
        assert math.isnan(result.summary_metrics.reality_check_p_value)