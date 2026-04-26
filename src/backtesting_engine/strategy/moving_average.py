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
    MA_LONG_RANGE,
    MA_SHORT_RANGE,
    MA_STEP,
    MOVING_AVERAGE_LONG_DAYS,
    MOVING_AVERAGE_SHORT_DAYS,
)
from backtesting_engine.metrics import _sharpe as _sharpe_annualised
from backtesting_engine.strategy.base import BaseStrategy, returns_from_signals


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
        Trailing underscore follows sklearn convention: attribute exists only
        after fit() has been called.
    _all_candidate_pairs_ : list[tuple[int, int]]
        All (short, long) pairs evaluated during the most recent fit().
        Used by candidate_test_returns() to avoid recomputing the grid.
        Leading underscore marks this as an implementation detail of the
        Reality Check interface — not part of the strategy's primary API.
        Trailing underscore marks it as fitted state (populated by fit()).
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

        Important: the grid search uses pre-cost, pre-slippage, zero-delay
        returns (via _returns_from_signals). This is standard practice — you
        cannot know execution costs with certainty during parameter selection,
        and optimising over costs would introduce another layer of look-ahead.
        The selected parameters are then evaluated out-of-sample with the full
        execution model (costs, slippage, delay) via walk_forward. The gap
        between pre-cost and post-cost performance is captured by comparing
        the training Sharpe to the test Sharpe in each WindowResult.

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
                returns = returns_from_signals(close, signals.to_numpy())
                sharpe = _sharpe_annualised(returns)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_short = short
                    best_long = long

        self.short_window_ = best_short
        self.long_window_ = best_long
        self._all_candidate_pairs_ = evaluated
        return self

    def context_window_size(self) -> int:
        """
        Return the number of warmup bars needed to fully initialise the long MA.

        The long moving average requires long_window_ bars of history before it
        produces a valid (non-NaN) value. Returning long_window_ + 1 guarantees
        that both MAs are warm and the crossover signal is valid at test bar 0.
        """
        return self.long_window_ + 1

    def active_params(self) -> dict[str, object]:
        """Return calibrated short/long windows for parameter evolution tracking."""
        return {
            "short_window": self.short_window_,
            "long_window": self.long_window_,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate crossover signals using calibrated window parameters."""
        return self._compute_signals(data, self.short_window_, self.long_window_)

    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        """
        Generate signals with training-tail warmup, preserving position carry-over.

        The plain diff() approach emits signals only on crossover bars. If the
        short MA is already above the long MA at the start of the test window
        (i.e. the strategy is already long from the training period), no buy
        signal is emitted for bar 0 of the test — the simulator starts flat.

        Fix: detect whether the strategy would be in a long position at the
        transition from context to test data. If yes, inject a buy signal at
        test_data.index[0] so the simulator opens the position immediately.

        This ensures the test window reflects the actual strategy state, not
        a reset to cash at every window boundary.

        Args:
            context_data: Tail of training data for rolling-average warmup.
            test_data: Out-of-sample test period.

        Returns:
            pd.Series of signals aligned to test_data.index.
        """
        combined = pd.concat([context_data, test_data])
        all_signals = self._compute_signals(combined, self.short_window_, self.long_window_)
        test_signals = all_signals.loc[test_data.index].copy()

        # Detect position at the context/test boundary.
        # If short MA > long MA on the last context bar, the strategy is long.
        # In that case, inject a buy signal at the first test bar so the simulator
        # opens the position rather than starting flat.
        close = combined["close"]
        short_ma = close.rolling(self.short_window_).mean()
        long_ma = close.rolling(self.long_window_).mean()

        last_context_idx = context_data.index[-1]
        if (
            last_context_idx in short_ma.index
            and not pd.isna(short_ma.loc[last_context_idx])
            and not pd.isna(long_ma.loc[last_context_idx])
            and short_ma.loc[last_context_idx] > long_ma.loc[last_context_idx]
            and test_signals.iloc[0] == 0  # no signal already at bar 0
        ):
            test_signals.iloc[0] = 1  # carry long position forward

        return test_signals

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

            raw_returns = returns_from_signals(close, signals.to_numpy())

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
