"""
Moving average crossover strategy with walk-forward parameter calibration.

Generates a buy signal when the short MA crosses above the long MA (golden
cross) and a sell signal when it crosses below (death cross).

Parameter calibration
---------------------
During fit(), a grid search over (short_window, long_window) pairs selects
the pair with the highest in-sample Sharpe on the training window.

Reality Check interface
-----------------------
candidate_test_returns() runs every grid candidate on the TEST data and
returns their daily return series. This is the correct data for White's
Reality Check: we are testing whether any candidate beats the zero-return
benchmark in the OUT-OF-SAMPLE period, not the training period.

The distinction matters: training returns will always show the winning pair
performing best by construction (that is how it was selected). Test-period
returns for all candidates are the correct input for the Reality Check null
distribution.
"""

import numpy as np
import pandas as pd

from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    MA_LONG_RANGE,
    MA_SHORT_RANGE,
    MA_STEP,
    MOVING_AVERAGE_LONG_DAYS,
    MOVING_AVERAGE_SHORT_DAYS,
    RISK_FREE_RATE,
)
from backtesting_engine.strategy.base import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """
    Moving average crossover with grid-search calibration.

    Parameters
    ----------
    short_window, long_window : int
        Initial window sizes. Overwritten by fit() on each walk-forward window.

    Attributes set by fit()
    -----------------------
    short_window_, long_window_ : int
        Calibrated window sizes for the current walk-forward window.
    _all_candidate_pairs_ : list[tuple[int, int]]
        All (short, long) pairs evaluated during the most recent fit().
        Used by candidate_test_returns() to avoid recomputing the grid.
    """

    def __init__(
        self,
        short_window: int = MOVING_AVERAGE_SHORT_DAYS,
        long_window: int = MOVING_AVERAGE_LONG_DAYS,
    ) -> None:
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be less than "
                f"long_window ({long_window})."
            )
        self.short_window_ = short_window
        self.long_window_ = long_window
        self._all_candidate_pairs_: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame) -> "MovingAverageStrategy":
        """
        Grid-search (short_window, long_window) on training data.

        Selects the pair with the highest in-sample Sharpe. Stores all
        evaluated pairs in _all_candidate_pairs_ so that
        candidate_test_returns() can evaluate the same universe on test data.

        Args:
            train_data: In-sample DataFrame with 'close' column.

        Returns:
            self, with short_window_ and long_window_ updated.
        """
        best_sharpe = -np.inf
        best_short = self.short_window_
        best_long = self.long_window_
        evaluated: list[tuple[int, int]] = []

        short_min, short_max = MA_SHORT_RANGE
        long_min, long_max = MA_LONG_RANGE
        close = train_data["close"].to_numpy(dtype=float)

        for short in range(short_min, short_max + 1, MA_STEP):
            for long in range(long_min, long_max + 1, MA_STEP):
                if short >= long:
                    continue
                if len(train_data) < long + 1:
                    continue

                evaluated.append((short, long))
                signals = self._compute_signals(train_data, short, long)
                returns = self._returns_from_signals(close, signals.to_numpy())
                sharpe = self._sharpe_of_returns(returns)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_short = short
                    best_long = long

        self.short_window_ = best_short
        self.long_window_ = best_long
        self._all_candidate_pairs_ = evaluated
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate crossover signals using calibrated window parameters."""
        return self._compute_signals(data, self.short_window_, self.long_window_)

    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        """
        Generate signals with training-tail warmup to eliminate NaN MAs.

        Prepends context_data before computing rolling MAs, then returns
        only signals aligned to test_data.index.
        """
        combined = pd.concat([context_data, test_data])
        all_signals = self._compute_signals(combined, self.short_window_, self.long_window_)
        return all_signals.loc[test_data.index]

    def candidate_test_returns(
        self,
        test_data: pd.DataFrame,
        context_data: pd.DataFrame | None = None,
    ) -> dict[tuple[int, int], "pd.Series"]:
        """
        Run every candidate (short, long) pair on the test data.

        Returns out-of-sample daily return series for each candidate.
        This is the correct input for White's Reality Check: we need
        test-period performance of the full candidate universe, not
        training-period performance of the winner.

        Args:
            test_data: Out-of-sample test period.
            context_data: Optional warmup tail from training window.

        Returns:
            Dict mapping (short, long) → pd.Series of daily returns
            on test_data. Only candidates evaluated during fit() are included.
        """
        results: dict[tuple[int, int], pd.Series] = {}
        close = test_data["close"].to_numpy(dtype=float)

        for short, long in self._all_candidate_pairs_:
            if context_data is not None:
                combined = pd.concat([context_data, test_data])
                all_sig = self._compute_signals(combined, short, long)
                signals = all_sig.loc[test_data.index]
            else:
                signals = self._compute_signals(test_data, short, long)

            raw_returns = self._returns_from_signals(close, signals.to_numpy())

            # Align to test_data index (raw_returns has length len(close)-1)
            returns_series = pd.Series(
                raw_returns,
                index=test_data.index[1:],  # one shorter due to differencing
                dtype=float,
            )
            results[(short, long)] = returns_series

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_signals(
        data: pd.DataFrame, short_window: int, long_window: int
    ) -> pd.Series:
        """
        MA crossover signals for given windows.

        Buy (1) when short MA crosses above long MA (golden cross).
        Sell (-1) when short MA crosses below long MA (death cross).
        Hold (0) otherwise. NaN warmup rows become False → 0.
        """
        close = data["close"]
        above = (
            close.rolling(short_window).mean() > close.rolling(long_window).mean()
        ).fillna(False)
        return above.astype(int).diff().fillna(0).astype(int)

    @staticmethod
    def _returns_from_signals(close: np.ndarray, signals: np.ndarray) -> np.ndarray:
        """
        Daily strategy returns implied by a signal array.

        Position on day t (0 or 1) × price return from t to t+1.
        One bar shorter than close due to differencing.
        """
        position = np.zeros(len(signals))
        held = 0
        for i, s in enumerate(signals):
            if s == 1:
                held = 1
            elif s == -1:
                held = 0
            position[i] = held

        price_returns = np.diff(close) / close[:-1]
        return position[:-1] * price_returns

    @staticmethod
    def _sharpe_of_returns(returns: np.ndarray) -> float:
        """Annualised Sharpe. Returns -inf if undefined."""
        if len(returns) == 0:
            return -np.inf
        excess = returns - RISK_FREE_RATE
        std = excess.std(ddof=1)
        if std < 1e-10:
            return -np.inf
        return float(excess.mean() / std * np.sqrt(ANNUALISATION_FACTOR))
