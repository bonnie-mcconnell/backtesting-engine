"""
Kalman filter trend-following strategy with MLE parameter calibration.

Model
-----
We treat the log-price as the sum of a latent trend component and observation
noise. The trend itself evolves as a random walk:

    trend[t] = trend[t-1] + w[t],    w[t] ~ N(0, Q)   (process noise)
    log_price[t] = trend[t] + v[t],  v[t] ~ N(0, R)   (observation noise)

This is a local-level model (also called the random walk plus noise model).
It is the simplest nontrivial state-space model and has a closed-form
Kalman filter recursion - no approximations required.

The two parameters are:
  Q  process noise variance: how fast the latent trend is allowed to change.
     Large Q → the filter reacts quickly to price moves (noisy signals).
     Small Q → the filter reacts slowly, smoothing over fluctuations.
  R  observation noise variance: how much of the price move is considered
     noise around a stable trend.
     Large R → the filter trusts the model over the data (slow adaptation).
     Small R → the filter trusts the data over the model (fast adaptation).

The ratio Q/R is what actually determines filter behaviour. We call it the
signal-to-noise ratio (SNR). SNR = 0 → no trend detected. SNR → ∞ → the
filtered trend tracks price exactly.

Calibration
-----------
Parameters (Q, R) are chosen on each training window by maximising the
log-likelihood of the one-step-ahead prediction errors (innovations).
The innovation at time t is:

    e[t] = log_price[t] - E[log_price[t] | log_price[1:t-1]]

Under the model, innovations are Gaussian with variance S[t] (the innovation
covariance, computed by the filter). The log-likelihood is:

    ℓ(Q, R) = -½ Σ [ln(2π S[t]) + e[t]² / S[t]]

We maximise this over (Q, R) using scipy.optimize.minimize with the
Nelder-Mead method (gradient-free, robust to the curvature discontinuities
that appear at the boundary Q→0).

Signal generation
-----------------
After running the filter forward through test data (using training-calibrated
Q, R):
  - Buy  (1) when the filtered trend velocity (Δtrend) crosses above zero.
  - Sell (-1) when the filtered trend velocity crosses below zero.
  - Hold (0) otherwise.

The filtered velocity is the difference between consecutive filtered state
estimates. It is a smoothed, forward-looking estimate of trend direction -
less noisy than raw price differences, more responsive than a fixed moving
average because Q and R adapt to the return variance of each training window.

Why this is better than a fixed moving average
-----------------------------------------------
A 50/200-day moving average is a special case of a Kalman filter with fixed,
implicit Q and R. The difference is that a moving average uses a hard window
(equal weights over N days, zero before that), while the Kalman filter uses
exponentially decaying weights whose decay rate is determined by the
calibrated Q/R ratio. When the training data shows high return variance, the
filter learns a smaller Q/R and becomes more conservative. When return
variance is low, Q/R increases and the filter reacts faster. This adaptation
is the property that a fixed MA cannot provide.

References
----------
Harvey, A.C. (1989). Forecasting, Structural Time Series Models and the
    Kalman Filter. Cambridge University Press.
Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by State Space
    Methods (2nd ed.). Oxford University Press.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize  # type: ignore[import-untyped]

from backtesting_engine.strategy.base import BaseStrategy


# Numerical floor for variances - prevents degenerate filters where Q or R
# collapse to zero and the Kalman gain becomes undefined.
_MIN_VARIANCE = 1e-8

# Nelder-Mead convergence tolerances. Tighter than scipy defaults to avoid
# premature termination on flat likelihood surfaces.
_OPTIM_XATOL = 1e-6
_OPTIM_FATOL = 1e-6
_OPTIM_MAXITER = 2000


class KalmanFilterStrategy(BaseStrategy):
    """
    Kalman filter trend-following strategy.

    Parameters
    ----------
    q_init : float
        Initial process noise variance for optimisation warm-start.
        Default chosen to be in a reasonable range for daily log-returns.
    r_init : float
        Initial observation noise variance for optimisation warm-start.

    Attributes (set by fit)
    -----------------------
    q_ : float
        Calibrated process noise variance.
    r_ : float
        Calibrated observation noise variance.
    log_likelihood_ : float
        Log-likelihood of the training data under calibrated (q_, r_).
    """

    def __init__(
        self,
        q_init: float = 1e-4,
        r_init: float = 1e-2,
    ) -> None:
        if q_init <= 0 or r_init <= 0:
            raise ValueError("q_init and r_init must be strictly positive.")
        self.q_init = q_init
        self.r_init = r_init

        # Calibrated values - set by fit(), read by generate_signals().
        self.q_: float = q_init
        self.r_: float = r_init
        self.log_likelihood_: float = float("-inf")

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame) -> "KalmanFilterStrategy":
        """
        Calibrate Q and R by maximising the Kalman filter log-likelihood
        on the training window.

        The optimisation is over log(Q) and log(R) rather than Q and R
        directly. This reparameterisation has two advantages:
          1. It enforces positivity without an explicit constraint.
          2. It makes the search space more symmetric - Q and R can span
             many orders of magnitude, and the log space keeps Nelder-Mead
             from spending most of its budget in unproductive regions.

        Args:
            train_data: DataFrame with DatetimeIndex and 'close' column.

        Returns:
            self, with q_ and r_ updated.
        """
        log_prices = np.log(train_data["close"].to_numpy(dtype=float))

        def neg_log_likelihood(log_params: np.ndarray) -> float:
            q = float(np.exp(log_params[0]))
            r = float(np.exp(log_params[1]))
            ll = _kalman_log_likelihood(log_prices, q, r)
            # Return large positive value (not nan/inf) so optimiser stays stable.
            return -ll if np.isfinite(ll) else 1e10

        x0 = np.array([np.log(self.q_init), np.log(self.r_init)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_log_likelihood,
                x0,
                method="Nelder-Mead",
                options={
                    "xatol": _OPTIM_XATOL,
                    "fatol": _OPTIM_FATOL,
                    "maxiter": _OPTIM_MAXITER,
                    "adaptive": True,   # adaptive Nelder-Mead for robustness
                },
            )

        self.q_ = float(np.exp(result.x[0]))
        self.r_ = float(np.exp(result.x[1]))
        self.log_likelihood_ = float(-result.fun)
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Run the Kalman filter forward and generate trend-velocity signals.

        Args:
            data: DataFrame with DatetimeIndex and 'close' column.

        Returns:
            pd.Series of integer signals (1, -1, 0) aligned to data.index.
        """
        log_prices = np.log(data["close"].to_numpy(dtype=float))
        filtered_trend = _kalman_filter(log_prices, self.q_, self.r_)

        # Trend velocity: difference of consecutive filtered state estimates.
        velocity = np.diff(filtered_trend, prepend=filtered_trend[0])

        # Signal = sign change in velocity (trend reversal).
        # positive velocity: trend accelerating upward → hold long.
        # velocity crosses zero downward: trend reversing → sell.
        # velocity crosses zero upward: trend reversing → buy.
        trending_up = velocity > 0
        signal_raw = np.diff(trending_up.astype(int), prepend=int(trending_up[0]))
        # +1 = crossed up (buy), -1 = crossed down (sell), 0 = no change.
        signals = pd.Series(signal_raw.astype(int), index=data.index)
        return signals

    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        """
        Run the filter on context + test data, return only test signals.

        Unlike the moving average strategy, the Kalman filter naturally
        handles warmup because the filter state initialises from the first
        observation and converges within a few bars. Context data still
        improves initialisation by providing a better prior for the filter
        state at the start of the test window.

        Args:
            context_data: Tail of training data for filter initialisation.
            test_data: Out-of-sample test period.

        Returns:
            pd.Series of signals aligned to test_data.index only.
        """
        combined = pd.concat([context_data, test_data])
        all_signals = self.generate_signals(combined)
        return all_signals.loc[test_data.index]

    def active_params(self) -> dict[str, float]:
        """
        Return calibrated parameters as a plain dict for WindowResult storage.

        The signal-to-noise ratio (Q/R) is the interpretable quantity:
        high SNR → filter tracks price closely (trending regime);
        low SNR → filter smooths aggressively (mean-reverting regime).

        Returns:
            Dict with keys 'q', 'r', 'snr', 'log_likelihood'.
        """
        return {
            "q": self.q_,
            "r": self.r_,
            "snr": self.q_ / max(self.r_, 1e-300),
            "log_likelihood": self.log_likelihood_,
        }


# ---------------------------------------------------------------------------
# Core Kalman recursions - pure NumPy, no dependencies
# ---------------------------------------------------------------------------

def _kalman_filter(log_prices: np.ndarray, q: float, r: float) -> np.ndarray:
    """
    Run the Kalman filter forward and return filtered state estimates.

    State: latent trend level μ[t].
    Observation: y[t] = log_price[t] = μ[t] + v[t], v[t] ~ N(0, R).
    Transition: μ[t] = μ[t-1] + w[t], w[t] ~ N(0, Q).

    Kalman recursion (predict → update):
        Predict:  μ[t|t-1] = μ[t-1|t-1]
                  P[t|t-1] = P[t-1|t-1] + Q
        Update:   S[t]     = P[t|t-1] + R          (innovation variance)
                  K[t]     = P[t|t-1] / S[t]       (Kalman gain)
                  μ[t|t]   = μ[t|t-1] + K[t] * e[t]  (e[t] = y[t] - μ[t|t-1])
                  P[t|t]   = (1 - K[t]) * P[t|t-1]

    Args:
        log_prices: Log-price series as a NumPy array.
        q: Process noise variance (Q).
        r: Observation noise variance (R).

    Returns:
        Array of filtered state estimates μ[t|t], same length as log_prices.
    """
    n = len(log_prices)
    filtered = np.empty(n)

    # Initialise: diffuse prior - state mean = first observation, large variance.
    mu = log_prices[0]
    p = 1.0   # prior variance; large relative to typical Q values

    for t in range(n):
        # Predict step
        p_pred = p + q

        # Update step
        s = p_pred + r                   # innovation variance
        k = p_pred / s                   # Kalman gain ∈ (0, 1)
        innovation = log_prices[t] - mu  # one-step prediction error
        mu = mu + k * innovation         # updated state estimate
        p = (1.0 - k) * p_pred          # updated state variance

        filtered[t] = mu

    return filtered


def _kalman_log_likelihood(log_prices: np.ndarray, q: float, r: float) -> float:
    """
    Compute the log-likelihood of log_prices under the local-level model.

    The log-likelihood is the sum of log predictive densities:
        ℓ = Σ_t log N(y[t]; μ[t|t-1], S[t])
          = -½ Σ_t [ln(2π) + ln(S[t]) + e[t]² / S[t]]

    where e[t] = y[t] - μ[t|t-1] is the innovation and S[t] is the
    innovation variance.

    This is the exact Gaussian log-likelihood - no approximations.

    Args:
        log_prices: Log-price series.
        q: Process noise variance.
        r: Observation noise variance.

    Returns:
        Total log-likelihood (higher is better).
    """
    q = max(q, _MIN_VARIANCE)
    r = max(r, _MIN_VARIANCE)

    n = len(log_prices)
    mu = log_prices[0]
    p = 1.0
    ll = 0.0

    for t in range(n):
        p_pred = p + q
        s = p_pred + r

        if s < _MIN_VARIANCE:
            return float("-inf")

        innovation = log_prices[t] - mu
        ll -= 0.5 * (np.log(2 * np.pi * s) + innovation ** 2 / s)

        k = p_pred / s
        mu = mu + k * innovation
        p = (1.0 - k) * p_pred

    return ll