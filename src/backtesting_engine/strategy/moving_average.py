"""
Moving average crossover strategy with walk-forward parameter calibration.

Generates a buy signal when the short MA crosses above the long MA (golden
cross) and a sell signal when it crosses below (death cross).

During fit(), a grid search over (short_window, long_window) pairs selects
the pair with the highest in-sample Sharpe on the training window.

candidate_test_returns() runs every grid candidate on the TEST data and
returns their daily return series. This is the correct data for White's
Reality Check: we are testing whether any candidate beats the zero-return
benchmark out-of-sample, not in-sample. Training returns will always show
the winning pair performing best by construction; test-period returns for
all candidates are the correct input for the RC null distribution.
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

    Args:
        short_window: Initial short window size. Overwritten by fit() each window.
        long_window: Initial long window size. Overwritten by fit() each window.

    Attributes set by fit():
        short_window_: Calibrated short window (trailing underscore = sklearn convention,
            attribute exists only after fit() has been called).
        long_window_: Calibrated long window.
        _all_candidate_pairs_: All (short, long) pairs evaluated during the most recent
            fit(). Leading underscore = implementation detail of the RC interface, not
            part of the primary strategy API. Trailing underscore = fitted state.
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
        evaluated pairs in _all_candidate_pairs_ for candidate_test_returns().
        Grid search uses pre-cost returns - see architecture.md for why.
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

    def format_params(self) -> str:
        """Return compact string e.g. 'MA(50/200)'."""
        return f"MA({self.short_window_}/{self.long_window_})"

    def param_evolution_spec(self) -> list[tuple[str, str]]:
        """Two lines: short window and long window over time."""
        return [
            ("Short MA window (days)", "short_window"),
            ("Long MA window (days)", "long_window"),
        ]

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
        signal is emitted for bar 0 of the test - the simulator starts flat.

        Fix: detect whether the strategy would be in a long position at the
        transition from context to test data. If yes, inject a buy signal at
        test_data.index[0] so the simulator opens the position immediately.

        Ensures the test window reflects the actual strategy state rather
        than resetting to cash at every window boundary.

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
        RC requires test-period performance of the full candidate universe,
        not training-period performance of the winner.

        Boundary carry-over is applied identically to generate_signals_with_context:
        if a candidate would be long at the context/test boundary, a buy signal
        is injected at test bar 0. Without this, the RC candidates and the selected
        strategy are evaluated under different state assumptions - the selected
        strategy carries its position forward but candidate returns start flat.

        Args:
            test_data: Out-of-sample test period.
            context_data: Optional warmup tail from training window.

        Returns:
            Dict mapping (short, long) -> pd.Series of daily returns
            on test_data. Only candidates evaluated during fit() are included.
        """
        results: dict[tuple[int, int], pd.Series] = {}
        close = test_data["close"].to_numpy(dtype=float)

        for short, long in self._all_candidate_pairs_:
            if context_data is not None:
                combined = pd.concat([context_data, test_data])
                all_sig = self._compute_signals(combined, short, long)
                signals = all_sig.loc[test_data.index].copy()

                # Apply the same boundary carry-over logic as
                # generate_signals_with_context so RC candidates and the
                # selected strategy are evaluated under identical state.
                c = combined["close"]
                short_ma = c.rolling(short).mean()
                long_ma = c.rolling(long).mean()
                last_ctx_idx = context_data.index[-1]
                if (
                    last_ctx_idx in short_ma.index
                    and not pd.isna(short_ma.loc[last_ctx_idx])
                    and not pd.isna(long_ma.loc[last_ctx_idx])
                    and short_ma.loc[last_ctx_idx] > long_ma.loc[last_ctx_idx]
                    and signals.iloc[0] == 0
                ):
                    signals.iloc[0] = 1
            else:
                signals = self._compute_signals(test_data, short, long)

            raw_returns = returns_from_signals(close, signals.to_numpy())

            returns_series = pd.Series(
                raw_returns,
                index=test_data.index[1:],
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
