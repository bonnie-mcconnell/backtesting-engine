"""
Unit tests for the walk-forward orchestrator.

Tests use synthetic data with known length and small window sizes (1 year train,
1 year test) so that expected window counts and date boundaries can be computed
by hand without relying on real market data or config constants.
"""
import pandas as pd
import numpy as np
import pytest

from backtesting_engine.walk_forward import walk_forward
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.config import ANNUALISATION_FACTOR


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_data() -> pd.DataFrame:
    """504 business days of smoothly increasing prices.

    Smoothly increasing prices guarantee the moving average strategy produces
    exactly one golden cross, ensuring trades are generated in every test window.
    504 days = 2 * 252, which fits exactly two 1-year windows with no remainder.
    """
    dates = pd.date_range("2010-01-01", periods=504, freq="B")
    prices = np.linspace(100.0, 150.0, 504)
    return pd.DataFrame({"close": prices}, index=dates)


@pytest.fixture
def strategy() -> MovingAverageStrategy:
    """Moving average strategy instance for use across tests."""
    return MovingAverageStrategy()


# ---------------------------------------------------------------------------
# Window count
# ---------------------------------------------------------------------------

def test_correct_number_of_windows(
    trending_data: pd.DataFrame, strategy: MovingAverageStrategy
) -> None:
    # 504 days, train=252, test=252
    # window 1: rows 0-252 train, 252-504 test
    # window 2: rows 252-504 train, 504-756 test — does not fit (504 < 756)
    # so exactly 1 window should be produced
    result = walk_forward(
        trending_data, strategy,
        training_window_years=1,
        testing_window_years=1,
    )
    assert len(result.window_results) == 1


def test_two_windows_with_sufficient_data(strategy: MovingAverageStrategy) -> None:
    # 756 days fits train=252, test=252, advance=252, then train=252-504, test=504-756
    dates = pd.date_range("2010-01-01", periods=756, freq="B")
    prices = np.linspace(100.0, 180.0, 756)
    data = pd.DataFrame({"close": prices}, index=dates)
    result = walk_forward(
        data, strategy,
        training_window_years=1,
        testing_window_years=1,
    )
    assert len(result.window_results) == 2


# ---------------------------------------------------------------------------
# No look-ahead bias — test start always after train end
# ---------------------------------------------------------------------------

def test_no_lookahead_bias(
    trending_data: pd.DataFrame, strategy: MovingAverageStrategy
) -> None:
    result = walk_forward(
        trending_data, strategy,
        training_window_years=1,
        testing_window_years=1,
    )
    for window in result.window_results:
        assert window.test_start > window.train_end


# ---------------------------------------------------------------------------
# Window advancement — each window advances by exactly test_days
# ---------------------------------------------------------------------------

def test_window_advances_by_test_days(strategy: MovingAverageStrategy) -> None:
    dates = pd.date_range("2010-01-01", periods=756, freq="B")
    prices = np.linspace(100.0, 180.0, 756)
    data = pd.DataFrame({"close": prices}, index=dates)
    result = walk_forward(
        data, strategy,
        training_window_years=1,
        testing_window_years=1,
    )
    if len(result.window_results) >= 2:
        w1 = result.window_results[0]
        w2 = result.window_results[1]
        # second window's test start should be 252 trading days after first
        assert w2.test_start == w1.test_end + pd.offsets.BDay(1)


# ---------------------------------------------------------------------------
# Insufficient data raises ValueError
# ---------------------------------------------------------------------------

def test_insufficient_data_raises(strategy: MovingAverageStrategy) -> None:
    # 100 days is far less than the minimum 504 required for 1+1 year windows
    dates = pd.date_range("2010-01-01", periods=100, freq="B")
    data = pd.DataFrame({"close": np.linspace(100.0, 110.0, 100)}, index=dates)
    with pytest.raises(ValueError):
        walk_forward(
            data, strategy,
            training_window_years=1,
            testing_window_years=1,
        )


# ---------------------------------------------------------------------------
# BacktestResult structure
# ---------------------------------------------------------------------------

def test_backtest_result_strategy_name(
    trending_data: pd.DataFrame, strategy: MovingAverageStrategy
) -> None:
    result = walk_forward(
        trending_data, strategy,
        training_window_years=1,
        testing_window_years=1,
    )
    assert result.strategy_name == "MovingAverageStrategy"


def test_backtest_result_has_summary_metrics(
    trending_data: pd.DataFrame, strategy: MovingAverageStrategy
) -> None:
    result = walk_forward(
        trending_data, strategy,
        training_window_years=1,
        testing_window_years=1,
    )
    assert result.summary_metrics is not None
    assert isinstance(result.summary_metrics.sharpe_ratio, float)