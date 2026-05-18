"""
Abstract base class for all trading strategies.

Every strategy must implement fit(train_data) and generate_signals(data).
fit() calibrates parameters on training data only and must return self.
generate_signals() produces a signal Series with values in {-1, 0, 1}.

Strategies with a lookback window should also implement
generate_signals_with_context(context_data, test_data), which prepends
warmup history so rolling indicators are initialised at the start of each
test window.

Strategies that perform a parameter grid search during fit() should implement
candidate_test_returns(test_data, context_data), which returns a dict mapping
each candidate parameter set to its daily return series on the TEST data.
The orchestrator collects these across windows to build the Reality Check
candidate matrix. Return an empty dict if there is no parameter search.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


def returns_from_signals(close: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """
    Compute daily strategy returns from a price array and a signal array.

    Position on day t (0 = flat, 1 = long) is determined by the last active
    signal up to and including day t. The daily return is position[t] multiplied
    by the price return from t to t+1, giving an array one element shorter than
    the inputs.

    This function is shared by all strategies that perform in-sample grid search
    (MovingAverageStrategy, MomentumStrategy) and by candidate_test_returns() for
    the Reality Check matrix. It lives here in base.py rather than being duplicated
    in each strategy module.

    Args:
        close: Price array of length n. Must not contain zeros (returns NaN otherwise).
        signals: Integer signal array of length n with values in {-1, 0, 1}.
                 1 = buy (go long), -1 = sell (go flat), 0 = hold current position.

    Returns:
        Float array of length n-1. Entry i is the strategy return from bar i to bar i+1.

    Notes:
        - Returns are computed without transaction costs or slippage. These frictions
          are applied by the execution model during the walk-forward simulation.
          Optimising parameters over post-cost returns would introduce an additional
          look-ahead layer (costs are configuration, not market data).
        - A zero in the close array would produce an infinite return. Real price
          data never has this; the guard in momentum.py (np.maximum(close, 1e-10))
          handles synthetic test data.
    """
    # replace 0 (hold) with NaN, forward-fill the last buy/sell, map to position.
    # Correctness sketch for signals = [0, 1, 0, 0, -1, 0]:
    #   replace 0→NaN  → [NaN, 1, NaN, NaN, -1, NaN]
    #   ffill          → [NaN,  1,   1,   1,  -1,  -1]
    #   fillna(0)      → [  0,  1,   1,   1,  -1,  -1]
    #   == 1           → [  0,  1,   1,   1,   0,   0]  ✓ flat, long×3, flat×2
    signal_series = pd.Series(signals, dtype=float)
    active = (
        signal_series
        .replace(0.0, pd.NA)
        .ffill()
        .fillna(0.0)
    )
    position = (active == 1.0).to_numpy(dtype=float)

    # Guard against zero prices in synthetic test data.
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

        Args:
            train_data: DataFrame with DatetimeIndex and 'close' column.

        Returns:
            self, after updating any internal parameter state.
        """
        ...

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from price data.

        Args:
            data: DataFrame with DatetimeIndex and 'close' column.

        Returns:
            pd.Series of integer signals aligned to data.index.
            Values must be exactly in {-1, 0, 1}.
        """
        ...

    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        """
        Generate signals with warmup history from context_data.

        Default implementation ignores context and delegates to
        generate_signals(test_data). Override when the strategy has a
        lookback period that requires prior history to produce valid signals.

        Args:
            context_data: Tail of training data for warmup.
            test_data: Out-of-sample test period.

        Returns:
            pd.Series of signals aligned to test_data.index only.
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
        out-of-sample returns for every candidate in the parameter search
        universe. Used to build the Reality Check candidate return matrix.

        Default implementation returns an empty dict (no parameter search).
        Strategies with parameter grids (e.g. MovingAverageStrategy) must
        override this to return test-period returns for every candidate.

        Args:
            test_data: Out-of-sample test period.
            context_data: Optional warmup history for signal computation.

        Returns:
            Dict mapping parameter identifier → pd.Series of daily returns
            on test_data. Empty dict means Reality Check is not applicable.
        """
        return {}

    def active_params(self) -> "dict[str, object]":
        """
        Return calibrated parameters as a plain dict for WindowResult storage.

        Default returns an empty dict. Strategies that calibrate parameters
        during fit() should override this to return their active parameter values.
        Called by walk_forward after fit() to record parameter evolution.

        Returns:
            Dict mapping parameter name to value. Empty dict for parameter-free
            strategies. Example: {'short_window': 50, 'long_window': 200}.
        """
        return {}

    def format_params(self) -> str:
        """
        Compact human-readable string of current parameters, e.g. 'MA(50/200)'.

        Override to return a readable format. Default calls str(active_params()).
        Empty string for parameter-free strategies.
        """
        params = self.active_params()
        if not params:
            return ""
        return str(params)

    def param_evolution_spec(self) -> list[tuple[str, str]]:
        """
        List of (y_axis_label, active_params_key) pairs for the dashboard panel.

        Override when the strategy has calibrated parameters to plot over time.
        Default returns one entry per active_params() key with the key as label.
        """
        params = self.active_params()
        if not params:
            return []
        # Generic fallback: one line per key, label is the key name.
        return [(str(k), str(k)) for k in params]

    def context_window_size(self) -> int:
        """
        Number of training-tail bars passed as warmup context before each test window.

        Override when the strategy needs prior history for signals to be valid at
        test bar 0. Default is 50, which covers most short-window indicators.
        Strategies with long lookbacks (e.g. 200-day MA) must override this.
        """
        return 50
