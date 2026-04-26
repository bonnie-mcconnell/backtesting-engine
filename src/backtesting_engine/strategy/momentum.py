"""
Time-series momentum strategy with walk-forward lookback calibration.

Strategy
--------
Buy when the T-day return (log(P[t] / P[t-T])) is positive; sell when negative.
This is the simplest version of the time-series momentum effect documented by
Moskowitz, Ooi & Pedersen (2012): assets that have been going up tend to continue
going up over the next month.

The lookback T controls the signal horizon. Short lookbacks (20–40 days) produce
noisy, high-turnover signals. Long lookbacks (120–250 days) are smoother but
slower to respond to reversals. The right T depends on the asset and the period.

Calibration
-----------
During fit(), a grid search over T ∈ {20, 40, 60, 90, 120, 180, 250} selects
the lookback with the highest in-sample Sharpe ratio. This is the same grid-search
pattern as MovingAverageStrategy, which means:

  1. The walk-forward engine calls candidate_test_returns() to collect test-period
     returns for all 7 candidates - feeding them into White's Reality Check.
  2. The calibrated lookback is stored in active_params() for the parameter
     evolution panel.

Why time-series momentum rather than cross-sectional?
Cross-sectional momentum ranks assets against each other and goes long the top
decile, short the bottom. That requires multiple assets. Time-series momentum
asks only whether a single asset is trending up or down - the same question as
a moving average crossover, but answered differently. Instead of comparing two
moving averages (which is a ratio of two smoothed prices), momentum uses the raw
return over a fixed window (which is a single price ratio). The two signals are
correlated but not identical.

The practical difference shows up in fast-reversing markets: a moving average
crossover can stay long through a sharp reversal if the long MA hasn't caught up.
A momentum signal flips the same day the lookback return turns negative.

References
----------
Moskowitz, T.J., Ooi, Y.H., & Pedersen, L.H. (2012). Time Series Momentum.
  Journal of Financial Economics, 104(2), 228-250.
"""

import numpy as np
import pandas as pd

from backtesting_engine.metrics import _sharpe as _sharpe_annualised
from backtesting_engine.strategy.base import BaseStrategy, returns_from_signals

# Lookback periods tested during grid search (in trading days).
# Covers the range documented in Moskowitz et al.: 1 month through 12 months.
_LOOKBACK_GRID: list[int] = [20, 40, 60, 90, 120, 180, 250]


class MomentumStrategy(BaseStrategy):
    """
    Time-series momentum with calibrated lookback.

    Parameters
    ----------
    lookback : int
        Initial lookback in trading days. Overwritten by fit() on each
        walk-forward window.

    Attributes set by fit()
    -----------------------
    lookback_ : int
        Calibrated lookback for the current walk-forward window.
        Trailing underscore follows sklearn convention: set only after fit().
    _all_lookbacks_ : list[int]
        All lookbacks evaluated during the most recent fit(). Used by
        candidate_test_returns() to build the Reality Check matrix.
        Leading underscore marks this as an implementation detail of the
        Reality Check interface. Trailing underscore marks it as fitted state.
    """

    def __init__(self, lookback: int = 120) -> None:
        if lookback <= 0:
            raise ValueError(f"lookback must be positive, got {lookback}.")
        self.lookback_ = lookback
        self._all_lookbacks_: list[int] = []

    def context_window_size(self) -> int:
        """
        Return the lookback period as the warmup window size.

        The momentum signal at bar t requires close[t - lookback_], so the
        first valid signal occurs at bar lookback_. Providing lookback_ context
        bars ensures the signal is defined at test bar 0.
        """
        return self.lookback_

    def fit(self, train_data: pd.DataFrame) -> "MomentumStrategy":
        """
        Grid-search lookback period on training data.

        Selects the lookback with the highest in-sample Sharpe. Uses
        the same coarse grid (_LOOKBACK_GRID) for all training windows
        to avoid overfitting.

        Args:
            train_data: In-sample DataFrame with 'close' column.

        Returns:
            self, with lookback_ updated.
        """
        close = train_data["close"].to_numpy(dtype=float)
        best_sharpe = -np.inf
        best_lookback = self.lookback_
        evaluated: list[int] = []

        for lb in _LOOKBACK_GRID:
            if len(close) < lb + 2:
                continue
            evaluated.append(lb)
            signals = _momentum_signals(close, lb)
            returns = returns_from_signals(close, signals)
            sharpe = _sharpe_annualised(returns)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_lookback = lb

        self.lookback_ = best_lookback
        self._all_lookbacks_ = evaluated
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals using the calibrated lookback."""
        close = data["close"].to_numpy(dtype=float)
        raw = _momentum_signals(close, self.lookback_)
        return pd.Series(raw, index=data.index, dtype=int)

    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        """
        Generate signals with training-tail warmup, preserving position carry-over.

        Prepends context_data so the lookback window is fully populated at the
        start of the test period. Also injects a buy signal at the first test bar
        when the strategy would already be long at the context/test boundary -
        preventing the false position reset that occurs when the buy signal fired
        inside the context window rather than the test window.

        Args:
            context_data: Tail of training data (last lookback_ bars).
            test_data: Out-of-sample test period.

        Returns:
            pd.Series of signals aligned to test_data.index.
        """
        combined = pd.concat([context_data, test_data])
        all_signals = self.generate_signals(combined)
        test_signals = all_signals.loc[test_data.index].copy()

        # Detect whether momentum was positive at the last context bar.
        # If yes, the strategy is long entering the test window - carry it forward.
        close = combined["close"].to_numpy(dtype=float)
        n_context = len(context_data)
        lb = self.lookback_

        if n_context >= lb:
            last_context_pos = n_context - 1  # index into combined
            if last_context_pos >= lb:
                log_close = np.log(np.maximum(close, 1e-10))
                momentum_at_boundary = log_close[last_context_pos] - log_close[last_context_pos - lb]
                if momentum_at_boundary > 0 and test_signals.iloc[0] == 0:
                    test_signals.iloc[0] = 1  # carry long position forward

        return test_signals

    def candidate_test_returns(
        self,
        test_data: pd.DataFrame,
        context_data: pd.DataFrame | None = None,
    ) -> dict[int, pd.Series]:
        """
        Run every lookback candidate on the test data.

        Returns out-of-sample daily returns for each lookback in the
        grid search universe. Used by walk_forward to build the
        Reality Check candidate matrix.

        Args:
            test_data: Out-of-sample test period.
            context_data: Optional warmup tail from training window.

        Returns:
            Dict mapping lookback → pd.Series of daily returns on test_data.
        """
        results: dict[int, pd.Series] = {}

        for lb in self._all_lookbacks_:
            if context_data is not None:
                combined = pd.concat([context_data, test_data])
                all_sig = self.generate_signals(combined)
                signals = all_sig.loc[test_data.index].to_numpy()
            else:
                signals = _momentum_signals(test_data["close"].to_numpy(dtype=float), lb)

            close = test_data["close"].to_numpy(dtype=float)
            raw_returns = returns_from_signals(close, signals)
            returns_series = pd.Series(
                raw_returns,
                index=test_data.index[1:],
                dtype=float,
            )
            results[lb] = returns_series

        return results

    def active_params(self) -> dict[str, object]:
        """Return calibrated lookback for parameter evolution tracking."""
        return {"lookback": self.lookback_}


# ---------------------------------------------------------------------------
# Internal helpers - pure NumPy, no side effects
# ---------------------------------------------------------------------------

def _momentum_signals(close: np.ndarray, lookback: int) -> np.ndarray:
    """
    Compute time-series momentum signals from a price array.

    Signal at bar t:
      momentum[t] = log(close[t] / close[t - lookback])
      signal[t]   = +1 if momentum[t] > 0 (upward trend, hold long)
                  = -1 if momentum transitions from positive to negative (sell)
                  =  0 otherwise (hold or out of market)

    Buy fires on the bar when momentum crosses from ≤0 to >0.
    Sell fires on the bar when momentum crosses from >0 to ≤0.
    The first `lookback` bars have no signal (warmup).

    Args:
        close: Price array.
        lookback: Lookback period in bars.

    Returns:
        Integer signal array of same length as close.
    """
    n = len(close)
    signals = np.zeros(n, dtype=int)

    if n <= lookback:
        return signals

    # Log return over lookback window, vectorised.
    # Avoid log(0) by clamping - prices should always be positive.
    log_close = np.log(np.maximum(close, 1e-10))
    momentum = log_close[lookback:] - log_close[:-lookback]

    # Position: 1 when momentum > 0, 0 otherwise.
    # Signal: +1 on transition 0→1, -1 on transition 1→0.
    position = (momentum > 0).astype(int)
    # Prepend current position to compute diff from bar lookback-1 forward.
    # Use 0 as the assumed prior position (out of market before warmup).
    prev = np.concatenate([[0], position[:-1]])
    delta = position - prev

    # delta[i] corresponds to close[i + lookback]
    signals[lookback:] = delta
    return signals
