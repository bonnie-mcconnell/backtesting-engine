"""
Shared test helpers for the backtesting-engine test suite.

This module contains pure functions (no pytest dependency) that generate
synthetic data used across multiple test modules. It is a regular importable
module - unlike conftest.py, which is auto-discovered by pytest and cannot
be reliably imported with `from conftest import ...`.

Fixtures that wrap these helpers (oscillating_504, oscillating_756, strategy)
live in conftest.py and are injected by pytest automatically.
"""

import numpy as np
import pandas as pd


def make_oscillating_data(
    n: int,
    start: str = "2010-01-01",
    with_high_low: bool = False,
) -> pd.DataFrame:
    """
    Generate synthetic oscillating price data for testing.

    Prices follow a sine wave (period = n/10π bars) with a slight upward
    drift. The frequency is dense enough that multiple golden/death cross
    events occur in every 252-bar window regardless of where that window
    starts - making this suitable for walk-forward and signal-detection tests.

    A monotonically trending series is NOT used because a perfectly linear
    price produces exactly one golden cross in its entire history. If that
    crossover falls in the training period, every test window produces zero
    signals and the window is skipped, making test assertions vacuous.

    Args:
        n: Number of business days.
        start: ISO date string for the first index entry.
        with_high_low: If True, append 'high' (close + 0.5) and 'low'
            (close − 0.5) columns. Required by benchmark tests that pass the
            full OHLC frame to compute_benchmark(), and by walk_forward tests
            that call data["high"] / data["low"] for slippage estimation.

    Returns:
        DataFrame with a business-day DatetimeIndex and at minimum a
        'close' column. Values are strictly positive (minimum ≈ 80).
    """
    dates = pd.date_range(start, periods=n, freq="B")
    t = np.linspace(0, 20 * np.pi, n)
    close = 100.0 + 20.0 * np.sin(t) + 0.05 * np.arange(n)
    df = pd.DataFrame({"close": close}, index=dates)
    if with_high_low:
        df["high"] = close + 0.5
        df["low"] = close - 0.5
    return df
