"""
Time-series momentum strategy with walk-forward lookback calibration.

Buy when the T-day return (log(P[t] / P[t-T])) is positive; sell when negative.
This is the simplest version of the time-series momentum effect documented by
Moskowitz, Ooi & Pedersen (2012): assets that have been going up tend to continue
going up over the next month.

The lookback T controls the signal horizon. Short lookbacks (20–40 days) produce
noisy, high-turnover signals. Long lookbacks (120–250 days) are smoother but
slower to respond to reversals.

During fit(), a grid search over T ∈ {20, 40, 60, 90, 120, 180, 250} selects
the lookback with the highest in-sample Sharpe. Same grid-search pattern as
MovingAverageStrategy: candidate_test_returns() feeds all 7 candidates into
White's Reality Check, and the calibrated lookback is stored in active_params()
for the parameter evolution panel.

Cross-sectional momentum ranks assets against each other and goes long the top
decile, short the bottom. Time-series momentum asks only whether a single asset
is trending up or down: the same question as a moving average crossover, but
answered differently. Instead of comparing two smoothed prices, momentum uses the
raw return over a fixed window. The two signals are correlated but not identical.
The practical difference: a MA crossover can stay long through a sharp reversal
if the long MA hasn't caught up; a momentum signal flips the same day the lookback
return turns negative.

Moskowitz, T.J., Ooi, Y.H., & Pedersen, L.H. (2012). Time Series Momentum.
Journal of Financial Economics, 104(2), 228-250.
"""

import numpy as np
import pandas as pd

from backtesting_engine.config import MOMENTUM_LOOKBACKS
from backtesting_engine.metrics import _sharpe as _sharpe_annualised
from backtesting_engine.strategy.base import BaseStrategy, returns_from_signals


class MomentumStrategy(BaseStrategy):
    """
    Time-series momentum with calibrated lookback.

    Args:
        lookback: Initial lookback in trading days. Overwritten by fit() each window.

    Attributes set by fit():
        lookback_: Calibrated lookback (trailing underscore = sklearn convention).
        _all_lookbacks_: All lookbacks evaluated during the most recent fit(), used
            by candidate_test_returns() for the Reality Check matrix (leading
            underscore = RC implementation detail, trailing = fitted state).
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

        for lb in MOMENTUM_LOOKBACKS:
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
        if not evaluated:
            raise ValueError(
                f"No lookback could be evaluated: training data has {len(close)} bars "
                f"but the smallest lookback in the grid requires at least "
                f"{min(MOMENTUM_LOOKBACKS) + 2} bars. "
                "Pass more training data or reduce MOMENTUM_LOOKBACKS in config.py."
            )
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
        Generate signals with training-tail warmup from context_data.

        context_window_size() returns exactly lookback_ bars, which is enough
        to compute one momentum value - the one at test bar 0 (comparing
        test_data.index[0]'s close to context_data.index[0]'s close, lookback_
        bars apart). No explicit boundary carry-over is needed here, unlike
        MovingAverageStrategy and KalmanFilterStrategy: momentum's signal is a
        direct function of the current bar's T-day return, not of a transition
        from a prior state, so "is momentum positive right now" is the whole
        answer - there's no separate "was the position already open" question
        to carry forward. (Verified: test bar 0's signal here always agrees
        with momentum computed directly from the unwindowed series. See
        TestMomentumBoundaryCorrectness in test_strategy.py.)

        Args:
            context_data: Tail of training data (last lookback_ bars).
            test_data: Out-of-sample test period.

        Returns:
            pd.Series of signals aligned to test_data.index.
        """
        combined = pd.concat([context_data, test_data])
        all_signals = self.generate_signals(combined)
        return all_signals.loc[test_data.index].copy()

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

        No boundary carry-over is needed here - see generate_signals_with_context
        for why momentum's test-bar-0 signal is already correct without it.
        All candidates receive the same context_data (length = lookback_, the
        calibrated lookback). Candidates with lb > lookback_ have fewer than lb
        warmup bars available, so their signals are zero for the first
        (lb - lookback_) test bars. This understates larger-lb candidates
        slightly, making RC conservative for those candidates. Candidates with
        lb <= lookback_ are unaffected.

        Args:
            test_data: Out-of-sample test period.
            context_data: Optional warmup tail from training window.

        Returns:
            Dict mapping lookback -> pd.Series of daily returns on test_data.
        """
        results: dict[int, pd.Series] = {}

        for lb in self._all_lookbacks_:
            close_test = test_data["close"].to_numpy(dtype=float)
            if context_data is not None:
                combined_close = np.concatenate([
                    context_data["close"].to_numpy(dtype=float),
                    close_test,
                ])
                all_signals_raw = _momentum_signals(combined_close, lb)
                signals = all_signals_raw[len(context_data):]
            else:
                signals = _momentum_signals(close_test, lb)

            raw_returns = returns_from_signals(close_test, signals)
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

    def format_params(self) -> str:
        """Return compact string e.g. 'MOM(90)'."""
        return f"MOM({self.lookback_})"

    def param_evolution_spec(self) -> list[tuple[str, str]]:
        """One line: fitted lookback period over time."""
        return [("Fitted lookback (days)", "lookback")]


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
