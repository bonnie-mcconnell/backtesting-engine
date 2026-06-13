"""
Kalman filter trend-following strategy with MLE parameter calibration.

Local-level model: log-price = latent trend + observation noise.

    trend[t] = trend[t-1] + w[t],    w[t] ~ N(0, Q)   (process noise)
    log_price[t] = trend[t] + v[t],  v[t] ~ N(0, R)   (observation noise)

Q and R are calibrated per training window by maximising the Kalman filter
log-likelihood. Optimisation runs over log(Q) and log(R) - enforces positivity
and keeps Nelder-Mead in a symmetric search space since Q and R can span
many orders of magnitude. Gradient methods don't work well here because the
likelihood surface has curvature discontinuities near Q→0.

The signal is the sign change in filtered trend velocity (Δtrend): buy when
velocity crosses above zero, sell when it crosses below. This adapts to each
training window's return variance in a way a fixed MA can't - when volatility
is high the filter calibrates a smaller Q/R and smooths more aggressively.

Harvey, A.C. (1989). Forecasting, Structural Time Series Models and the Kalman
Filter. Cambridge University Press.
Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by State Space Methods
(2nd ed.). Oxford University Press.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backtesting_engine.strategy.base import BaseStrategy

_MIN_VARIANCE: float = 1e-8    # floor for Q and R; prevents degenerate Kalman gain
_OPTIM_XATOL: float  = 1e-6   # Nelder-Mead tolerances - tighter than scipy defaults
_OPTIM_FATOL: float  = 1e-6   # to avoid premature convergence on flat surfaces
_OPTIM_MAXITER: int  = 2000


class KalmanFilterStrategy(BaseStrategy):
    """
    Kalman filter trend-following strategy.

    Args:
        q_init: Initial process noise variance (warm-start for optimisation).
        r_init: Initial observation noise variance.

    Attributes set by fit():
        q_: Calibrated process noise variance.
        r_: Calibrated observation noise variance.
        log_likelihood_: Log-likelihood of training data under (q_, r_).
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

        self.q_: float = q_init
        self.r_: float = r_init
        self.log_likelihood_: float = float("-inf")

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def context_window_size(self) -> int:
        """50-bar warmup so the filter state is stable at the test window start."""
        return 50

    def fit(self, train_data: pd.DataFrame) -> "KalmanFilterStrategy":
        """
        Calibrate Q and R by maximising the Kalman filter log-likelihood.

        Runs Nelder-Mead in log(Q), log(R) space. The reparameterisation
        enforces positivity and makes the search space roughly symmetric -
        Q and R span many orders of magnitude in practice.
        """
        log_prices = np.log(train_data["close"].to_numpy(dtype=float))

        def neg_log_likelihood(log_params: np.ndarray) -> float:
            q = float(np.exp(log_params[0]))
            r = float(np.exp(log_params[1]))
            ll = _kalman_log_likelihood(log_prices, q, r)
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
                    "adaptive": True,
                },
            )

        # Floor to _MIN_VARIANCE: _kalman_log_likelihood applies this floor
        # internally, so the objective is flat for any (q, r) below it and
        # Nelder-Mead can wander arbitrarily far into that flat region with
        # no signal to stop. Flooring here keeps the stored parameters
        # consistent with what was actually optimised, and keeps
        # _kalman_filter's s = p_pred + r safely away from zero at inference.
        self.q_ = max(float(np.exp(result.x[0])), _MIN_VARIANCE)
        self.r_ = max(float(np.exp(result.x[1])), _MIN_VARIANCE)
        self.log_likelihood_ = float(-result.fun)
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Run the filter forward and return trend-velocity crossing signals."""
        log_prices = np.log(data["close"].to_numpy(dtype=float))
        filtered_trend = _kalman_filter(log_prices, self.q_, self.r_)

        velocity = np.diff(filtered_trend, prepend=filtered_trend[0])
        trending_up = velocity > 0
        # +1 where velocity crossed up (buy), -1 where it crossed down (sell)
        signal_raw = np.diff(trending_up.astype(int), prepend=int(trending_up[0]))
        return pd.Series(signal_raw.astype(int), index=data.index)

    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        """
        Run the filter on context + test data, return only test signals.

        Position carry-over: if filter velocity is positive at the last context
        bar, inject a buy signal at test bar 0. Without this, a long position
        established during warmup would be silently dropped.
        """
        combined = pd.concat([context_data, test_data])
        all_signals = self.generate_signals(combined)
        test_signals = all_signals.loc[test_data.index].copy()

        log_prices_context = np.log(context_data["close"].to_numpy(dtype=float))
        filtered_context = _kalman_filter(log_prices_context, self.q_, self.r_)
        if len(filtered_context) >= 2:
            velocity_at_boundary = filtered_context[-1] - filtered_context[-2]
            if velocity_at_boundary > 0 and test_signals.iloc[0] == 0:
                test_signals.iloc[0] = 1

        return test_signals

    def active_params(self) -> dict[str, object]:
        """Calibrated Q, R, SNR, and log-likelihood for WindowResult storage."""
        return {
            "q": self.q_,
            "r": self.r_,
            "snr": self.q_ / max(self.r_, 1e-300),
            "log_likelihood": self.log_likelihood_,
        }

    def format_params(self) -> str:
        snr = self.q_ / max(self.r_, 1e-300)
        return f"SNR={snr:.2e}"

    def param_evolution_spec(self) -> list[tuple[str, str]]:
        return [
            ("Q/R signal-to-noise ratio", "snr"),
            ("Log-likelihood", "log_likelihood"),
        ]


# ---------------------------------------------------------------------------
# Core Kalman recursions
# ---------------------------------------------------------------------------

def _kalman_filter(log_prices: np.ndarray, q: float, r: float) -> np.ndarray:
    """
    Run the Kalman filter forward and return filtered state estimates.

    State-space model:
        Transition:   μ[t] = μ[t-1] + w[t],    w ~ N(0, Q)
        Observation:  y[t] = μ[t]   + v[t],    v ~ N(0, R)

    Standard predict-update recursion:
        Predict:  P[t|t-1] = P[t-1] + Q
        Update:   K = P[t|t-1] / (P[t|t-1] + R)
                  μ[t] = μ[t-1] + K * (y[t] - μ[t-1])
                  P[t] = (1 - K) * P[t|t-1]

    Prior: μ₀ = log_prices[0], P₀ = 1.0 (weakly informative; filter converges
    within ~5 bars for typical Q/R values, eliminated entirely by the 50-bar
    context window).
    """
    n = len(log_prices)
    filtered = np.empty(n)

    mu = log_prices[0]
    p = 1.0

    for t in range(n):
        p_pred = p + q
        s = p_pred + r
        k = p_pred / s
        innovation = log_prices[t] - mu
        mu = mu + k * innovation
        p = (1.0 - k) * p_pred
        filtered[t] = mu

    return filtered


def _kalman_log_likelihood(log_prices: np.ndarray, q: float, r: float) -> float:
    """
    Gaussian log-likelihood of log_prices under the local-level model.

        ℓ = Σ_t log N(y[t]; μ[t|t-1], S[t])
          = -½ Σ_t [ln(2π) + ln(S[t]) + e[t]² / S[t]]

    where e[t] = y[t] - μ[t|t-1] is the one-step prediction error and
    S[t] = P[t|t-1] + R is the innovation variance.
    """
    q = max(q, _MIN_VARIANCE)
    r = max(r, _MIN_VARIANCE)

    mu = log_prices[0]
    p = 1.0
    ll = 0.0

    for t in range(len(log_prices)):
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
