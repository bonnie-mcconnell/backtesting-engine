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

import pandas as pd


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