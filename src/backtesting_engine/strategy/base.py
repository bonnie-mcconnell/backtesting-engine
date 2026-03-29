"""
Abstract base class for trading strategies.

Subclasses must implement generate_signals(), which receives a price DataFrame
and returns a signal Series with values: 1 (buy), -1 (sell), 0 (hold).
"""

from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    Defines the interface that all concrete strategy implementations must follow.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from price data.

        Args:
            data: DataFrame with DatetimeIndex and 'close' column.

        Returns:
            Series of signals aligned to data's index: 1 (buy), -1 (sell), 0 (hold).
        """
        ...