"""
Abstract base class for all trading strategies.

Every strategy must implement fit(train_data) and generate_signals(data).
fit() calibrates parameters on training data only and must return self.
generate_signals() produces a signal Series with values in {-1, 0, 1}.

Strategies with a lookback window should override generate_signals_with_context()
so rolling indicators are initialised at the start of each test window rather
than warming up on test data.

Strategies that perform a parameter grid search during fit() should implement
candidate_test_returns() to supply out-of-sample returns for every candidate.
The orchestrator uses these to build the Reality Check candidate matrix.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


def returns_from_signals(close: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """
    Compute daily strategy returns from a price array and a signal array.

    Position on day t is determined by the last active signal up to day t.
    The daily return is position[t] * price_return(t→t+1), giving an array
    one element shorter than the inputs.

    Signals: 1 = buy (go long), -1 = sell (go flat), 0 = hold current position.

    Returns are computed without transaction costs or slippage - the execution
    model applies those during walk-forward simulation. Optimising parameters
    over post-cost returns would introduce an extra look-ahead layer since costs
    are configuration, not market data.

    Example - signals = [0, 1, 0, 0, -1, 0]:
        replace 0→NaN  → [NaN, 1, NaN, NaN, -1, NaN]
        ffill          → [NaN,  1,   1,   1,  -1,  -1]
        fillna(0)      → [  0,  1,   1,   1,  -1,  -1]
        == 1           → [  0,  1,   1,   1,   0,   0]  flat, long×3, flat×2 ✓
    """
    signal_series = pd.Series(signals, dtype=float)
    active = (
        signal_series
        .replace(0.0, pd.NA)
        .ffill()
        .fillna(0.0)
    )
    position = (active == 1.0).to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        price_returns = np.diff(close) / close[:-1]
    price_returns = np.nan_to_num(price_returns, nan=0.0, posinf=0.0, neginf=0.0)

    result: np.ndarray = position[:-1] * price_returns
    return result


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> "BaseStrategy":
        """
        Calibrate strategy parameters on in-sample training data.

        Must not access any test-period data. Must return self.
        """
        ...

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from price data.

        Returns pd.Series of integer signals aligned to data.index.
        Values must be exactly in {-1, 0, 1}.
        """
        ...

    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        """
        Generate signals with warmup history from context_data.

        Default ignores context and delegates to generate_signals(test_data).
        Override when the strategy has a lookback period that requires prior
        history to produce valid signals at the start of each test window.
        """
        return self.generate_signals(test_data)

    def candidate_test_returns(
        self,
        test_data: pd.DataFrame,
        context_data: pd.DataFrame | None = None,
    ) -> dict[Any, "pd.Series"]:
        """
        Evaluate ALL candidate parameter sets on test_data.

        Called by the walk-forward orchestrator after fit() to collect
        out-of-sample returns for every candidate in the parameter search.
        Used to build the Reality Check candidate return matrix.

        Returns an empty dict by default (no parameter search). Strategies
        with parameter grids must override this to return test-period returns
        for every candidate.
        """
        return {}

    def active_params(self) -> "dict[str, object]":
        """Calibrated parameters as a plain dict for WindowResult storage.

        Override in strategies that calibrate parameters during fit().
        """
        return {}

    def format_params(self) -> str:
        """Human-readable string of current parameters, e.g. 'MA(50/200)'."""
        params = self.active_params()
        if not params:
            return ""
        return str(params)

    def param_evolution_spec(self) -> list[tuple[str, str]]:
        """
        List of (y_axis_label, active_params_key) pairs for the dashboard panel.

        Override when the strategy has calibrated parameters to plot over time.
        """
        params = self.active_params()
        if not params:
            return []
        return [(str(k), str(k)) for k in params]

    def context_window_size(self) -> int:
        """
        Number of training-tail bars passed as warmup context before each test window.

        Override when the strategy needs prior history for signals to be valid at
        test bar 0. Default is 50 - covers most short-window indicators.
        """
        return 50
