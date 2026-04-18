"""
Moving average crossover strategy with walk-forward parameter calibration.

The strategy generates a buy signal when the short-term moving average crosses
above the long-term moving average (golden cross), and a sell signal when it
crosses below (death cross).

Default parameters (50/200-day) are used as the starting point. When fit() is
called during walk-forward validation, the engine searches a coarse grid over
(short_window, long_window) pairs and selects the pair with the highest
annualised Sharpe ratio on the training window. This means each test period is
evaluated with parameters that were chosen without ever looking at test data.

Grid search is intentionally coarse (10-day steps) to reduce the risk of
in-sample overfitting. The training Sharpe is a noisy signal; fine-grained
optimisation would fit to noise rather than to genuine predictive structure.
"""

import numpy as np
import pandas as pd

from backtesting_engine.strategy.base import BaseStrategy
from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    MA_LONG_RANGE,
    MA_SHORT_RANGE,
    MA_STEP,
    MOVING_AVERAGE_LONG_DAYS,
    MOVING_AVERAGE_SHORT_DAYS,
    RISK_FREE_RATE,
)


class MovingAverageStrategy(BaseStrategy):
    """
    Moving average crossover strategy with optional grid-search calibration.

    Parameters
    ----------
    short_window : int
        Initial short MA window in trading days. Overwritten by fit() if called.
    long_window : int
        Initial long MA window in trading days. Overwritten by fit() if called.

    Attributes
    ----------
    short_window_ : int
        Active short window (may differ from constructor arg after fit()).
    long_window_ : int
        Active long window (may differ from constructor arg after fit()).
    """

    def __init__(
        self,
        short_window: int = MOVING_AVERAGE_SHORT_DAYS,
        long_window: int = MOVING_AVERAGE_LONG_DAYS,
    ) -> None:
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be less than long_window ({long_window})."
            )
        self.short_window_ = short_window
        self.long_window_ = long_window

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame) -> "MovingAverageStrategy":
        """
        Grid-search (short_window, long_window) pairs on training data.

        Evaluates every pair in the configured grid, computes the Sharpe ratio
        of the resulting signals on training data, and stores the best-performing
        pair. Pairs with fewer than long_window rows of valid MA history are
        skipped. If no pair produces any trades, the constructor defaults are
        retained.

        The Sharpe here is an in-sample training Sharpe, used only for parameter
        selection - not reported as a performance estimate.

        Args:
            train_data: In-sample price DataFrame (DatetimeIndex, 'close' column).

        Returns:
            self, with short_window_ and long_window_ updated.
        """
        best_sharpe = -np.inf
        best_short = self.short_window_
        best_long = self.long_window_

        short_min, short_max = MA_SHORT_RANGE
        long_min, long_max = MA_LONG_RANGE

        for short in range(short_min, short_max + 1, MA_STEP):
            for long in range(long_min, long_max + 1, MA_STEP):
                if short >= long:
                    continue
                if len(train_data) < long + 1:
                    # Not enough history to compute even one valid MA value.
                    continue

                signals = self._compute_signals(train_data, short, long)
                sharpe = self._sharpe_from_signals(train_data, signals)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_short = short
                    best_long = long

        self.short_window_ = best_short
        self.long_window_ = best_long
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate crossover signals using the currently active window parameters.

        To eliminate warmup NaN at the start of the test window, the caller
        should pass context_data (the tail of the training window) via
        generate_signals_with_context(). This method is provided for
        compatibility and direct use when context is not available.

        Args:
            data: Out-of-sample price DataFrame (DatetimeIndex, 'close' column).

        Returns:
            pd.Series of integer signals (1, -1, 0) aligned to data.index.
        """
        return self._compute_signals(data, self.short_window_, self.long_window_)

    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        """
        Generate signals for test_data using context_data as MA warmup history.

        The long MA requires long_window days of history before producing a
        valid value. Without warmup context, the first long_window rows of
        every test window produce NaN signals - meaning those days are always
        treated as "hold" regardless of actual price behaviour. This biases
        results for short test windows.

        By prepending the tail of the training window (context_data) when
        computing MAs, then slicing off context rows before returning, signals
        are valid from the first day of the test window.

        Args:
            context_data: Tail of training data (>= long_window rows recommended).
            test_data: Out-of-sample test period.

        Returns:
            pd.Series of integer signals aligned to test_data.index only.
        """
        combined = pd.concat([context_data, test_data])
        all_signals = self._compute_signals(combined, self.short_window_, self.long_window_)
        # Return only test-period signals - context rows are discarded.
        return all_signals.loc[test_data.index]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_signals(
        data: pd.DataFrame, short_window: int, long_window: int
    ) -> pd.Series:
        """
        Compute crossover signals for given window sizes.

        A buy signal (1) fires on the bar where the short MA crosses above the
        long MA. A sell signal (-1) fires on the bar where it crosses below.
        All other bars are 0 (hold). Moving averages are lagging indicators -
        the signal fires after the crossover has already occurred, which is a
        known and accepted trade-off for this strategy class.

        Args:
            data: DataFrame with 'close' column.
            short_window: Short MA period in days.
            long_window: Long MA period in days.

        Returns:
            pd.Series of signals (1, -1, 0) with same index as data.
        """
        close = data["close"]
        short_ma = close.rolling(short_window).mean()
        long_ma = close.rolling(long_window).mean()

        # True where short is above long; NaN rows (warmup) become False.
        above = (short_ma > long_ma).fillna(False)

        # diff() on {0,1} gives: +1 on crossover up, -1 on crossover down, 0 otherwise.
        signals = above.astype(int).diff().fillna(0).astype(int)
        return signals

    @staticmethod
    def _sharpe_from_signals(data: pd.DataFrame, signals: pd.Series) -> float:
        """
        Compute annualised Sharpe ratio implied by a signal series on price data.

        Used internally during grid search to score candidate parameter pairs.
        Returns -inf if no trades are generated or returns have zero variance.

        Args:
            data: Price DataFrame with 'close' column.
            signals: Signal series (1, -1, 0) aligned to data.index.

        Returns:
            Annualised Sharpe ratio, or -inf if undefined.
        """
        close = data["close"].to_numpy()
        sig = signals.to_numpy()

        # Build a position series: 1 when holding, 0 when flat.
        position = np.zeros(len(sig))
        held = 0
        for i, s in enumerate(sig):
            if s == 1:
                held = 1
            elif s == -1:
                held = 0
            position[i] = held

        # Strategy daily returns: position[t-1] * price_return[t]
        price_returns = np.diff(close) / close[:-1]
        strategy_returns = position[:-1] * price_returns - RISK_FREE_RATE

        if len(strategy_returns) == 0:
            return -np.inf

        std = strategy_returns.std(ddof=1)
        if std < 1e-10:
            return -np.inf

        return float(strategy_returns.mean() / std * np.sqrt(ANNUALISATION_FACTOR))