"""
Abstract base class for all trading strategies.

The two-method contract
-----------------------
Every strategy must implement:

  fit(train_data) -> self
    Calibrate any learned parameters using only the training window.
    Must not access or store information about the test period.
    Parameter-free strategies implement this as a no-op returning self.

  generate_signals(data) -> pd.Series
    Produce a signal Series for the given price data.
    Values must be exactly: 1 (buy), -1 (sell), 0 (hold).

Optionally, strategies that benefit from MA-style warmup history implement:

  generate_signals_with_context(context_data, test_data) -> pd.Series
    Generate signals for test_data using context_data as history warmup.
    The orchestrator calls this when the strategy has a lookback period
    that would otherwise produce NaN values at the start of the test window.

The Reality Check interface
---------------------------
Strategies that perform a parameter search during fit() should also expose:

  candidate_test_returns(test_data, context_data) -> dict[Any, np.ndarray]
    Return a dict mapping each candidate parameter set to its daily return
    series on the TEST data (not training data). The orchestrator collects
    these across windows to build the Reality Check candidate matrix.

    This is separate from fit() so that:
    - fit() calibrates on training data (parameter selection)
    - candidate_test_returns() evaluates ALL candidates on test data (Reality Check)
    These are two different operations and must not be conflated.

    Return an empty dict if the strategy has no parameter search.
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
        - Position is computed via vectorised forward-fill rather than a Python loop.
          The stateful hold rule (0 = hold, 1 = buy, -1 = sell) is equivalent to
          replacing 0s with NaN and forward-filling the last non-zero signal. This
          avoids per-bar Python overhead; for the MA grid search (112 pairs × 756
          training bars per window) it is roughly 10–20× faster than a loop.
    """
    # Vectorised hold logic: replace 0 (hold) with NaN, forward-fill the most
    # recent buy (+1) or sell (-1), then map to {+1 → long, else → flat}.
    #
    # This is semantically identical to the naive Python loop — the loop simply
    # inspects each bar and carries the last non-zero signal forward. Pandas
    # ffill does the same operation in C, roughly 10–20× faster on typical
    # window sizes. For the MA grid search (112 candidate pairs × ~756-bar
    # training windows) the speedup is measurable across the full run.
    #
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

    def context_window_size(self) -> int:
        """
        Return the number of bars of warmup history needed before signals are valid.

        The walk-forward orchestrator slices the last context_window_size() bars
        from the training window and passes them to generate_signals_with_context()
        so that rolling indicators (moving averages, lookback windows, Kalman state)
        are fully initialised at the start of each test period.

        Override this method whenever generate_signals_with_context() needs history
        to produce valid signals at bar 0 of the test window. Returning 0 means
        the strategy requires no warmup context (e.g. it is stateless or handles
        initialisation internally).

        Default: 50 bars — conservative for most state-space models and short-window
        indicators. Strategies with long lookback periods (e.g. a 200-day MA) must
        override this to return a value at least as large as their longest window.

        Returns:
            Number of training-tail bars to pass as context. Non-negative integer.
        """
        return 50
