"""
Shared pytest fixtures for the backtesting-engine test suite.

Fixtures defined here are auto-discovered by pytest and injected into any
test module that declares them as parameters - no import required.

Shared data-generation helpers (make_oscillating_data) live in helpers.py,
which is a regular importable module. They are imported here for use in
fixtures, and imported directly by test modules that call them inline.
"""

import pandas as pd
import pytest
from helpers import make_oscillating_data

from backtesting_engine.strategy.moving_average import MovingAverageStrategy


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
