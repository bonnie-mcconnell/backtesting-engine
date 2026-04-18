"""
Unit tests for the MovingAverageStrategy.

Tests cover signal generation correctness, fit() parameter calibration,
warmup context, and the BaseStrategy interface contract.
"""

import numpy as np
import pandas as pd
import pytest

from backtesting_engine.strategy.moving_average import MovingAverageStrategy


def _make_data(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices}, index=dates)


def _oscillating_data(n: int = 1260, start: str = "2010-01-01") -> pd.DataFrame:
    """
    Sinusoidal prices with a slight upward trend.
    Guarantees multiple golden/death crosses, making it suitable for
    walk_forward and signal-detection tests.
    """
    dates = pd.date_range(start, periods=n, freq="B")
    t = np.linspace(0, 10 * np.pi, n)
    prices = 100.0 + 20.0 * np.sin(t) + 0.05 * np.arange(n)
    return pd.DataFrame({"close": prices}, index=dates)


def _trending_data(n: int = 504, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    prices = np.linspace(start_price, start_price * 1.5, n)
    return pd.DataFrame({"close": prices}, index=dates)


class TestSignalGeneration:
    def test_golden_cross_produces_buy_signal(self) -> None:
        # Phase 1 (60 bars): flat at 100 - short and long MA converge near 100.
        # Phase 2 (60 bars): sharp rise to 130 - short MA (10) rises faster than
        # long MA (30), crossing above it and producing a buy signal.
        prices = list(np.linspace(100.0, 100.0, 60)) + list(np.linspace(100.0, 130.0, 60))
        data = _make_data(prices)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert 1 in signals.values

    def test_death_cross_produces_sell_signal(self) -> None:
        # Phase 1 (80 bars): rising prices - golden cross fires, short above long.
        # Phase 2 (80 bars): sharp drop - short MA falls back below long MA (death cross).
        prices = list(np.linspace(100.0, 130.0, 80)) + list(np.linspace(130.0, 80.0, 80))
        data = _make_data(prices)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert -1 in signals.values

    def test_signals_only_contain_valid_values(self) -> None:
        data = _oscillating_data(300)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_aligned_to_data_index(self) -> None:
        data = _oscillating_data(300)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert signals.index.equals(data.index)

    def test_signals_are_integers(self) -> None:
        data = _oscillating_data(300)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert signals.dtype == int or np.issubdtype(signals.dtype, np.integer)

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="less than"):
            MovingAverageStrategy(short_window=200, long_window=50)


class TestFit:
    def test_fit_returns_self(self) -> None:
        strategy = MovingAverageStrategy()
        data = _oscillating_data(504)
        result = strategy.fit(data)
        assert result is strategy

    def test_fit_updates_windows(self) -> None:
        strategy = MovingAverageStrategy()
        data = _oscillating_data(504)
        strategy.fit(data)
        # After fit, windows must still be valid (short < long, both positive).
        assert strategy.short_window_ > 0
        assert strategy.long_window_ > 0

    def test_fit_maintains_short_less_than_long(self) -> None:
        strategy = MovingAverageStrategy()
        data = _oscillating_data(504)
        strategy.fit(data)
        assert strategy.short_window_ < strategy.long_window_

    def test_fit_with_insufficient_data_retains_defaults(self) -> None:
        strategy = MovingAverageStrategy(short_window=50, long_window=200)
        # Only 50 rows - no grid pair has enough history to compute valid MAs.
        data = _make_data([float(i) for i in range(100, 150)])
        original_short = strategy.short_window_
        original_long = strategy.long_window_
        strategy.fit(data)
        # Should retain defaults rather than raising.
        assert strategy.short_window_ == original_short
        assert strategy.long_window_ == original_long


class TestWarmupContext:
    def test_context_signals_only_cover_test_index(self) -> None:
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        all_data = _oscillating_data(200)
        context = all_data.iloc[:50]
        test_data = all_data.iloc[50:]
        signals = strategy.generate_signals_with_context(context, test_data)
        assert signals.index.equals(test_data.index)

    def test_context_signals_contain_only_valid_values(self) -> None:
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        all_data = _oscillating_data(300)
        context = all_data.iloc[:50]
        test_data = all_data.iloc[50:]
        signals = strategy.generate_signals_with_context(context, test_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_context_produces_more_signals_than_without(self) -> None:
        # With context, the MA is warmed up from bar 1 of test data.
        # Without context, the first long_window bars are always 0.
        # On oscillating data, context should yield >= as many non-zero signals.
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        all_data = _oscillating_data(400)
        context = all_data.iloc[:50]
        test_data = all_data.iloc[50:]

        with_context = strategy.generate_signals_with_context(context, test_data)
        without_context = strategy.generate_signals(test_data)

        n_with = (with_context != 0).sum()
        n_without = (without_context != 0).sum()
        assert n_with >= n_without