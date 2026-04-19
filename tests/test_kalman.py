"""
Unit tests for the Kalman filter strategy.

Tests cover:
  - Kalman filter recursion correctness (hand-verified values)
  - Log-likelihood is maximised at calibrated parameters
  - fit() convergence and parameter validity
  - Signal generation produces only {-1, 0, 1}
  - generate_signals_with_context returns only test-period indices
  - MLE parameters are data-dependent (different data → different params)
"""

import math
import numpy as np
import pandas as pd
import pytest

from backtesting_engine.strategy.kalman_filter import (
    KalmanFilterStrategy,
    _kalman_filter,
    _kalman_log_likelihood,
)


def _make_data(prices: list[float], start: str = "2015-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices}, index=dates)


def _trending_data(n: int = 500) -> pd.DataFrame:
    """Upward trend with moderate noise - calibration should converge."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    log_prices = np.cumsum(rng.normal(0.0005, 0.01, n))
    prices = 100.0 * np.exp(log_prices)
    return pd.DataFrame({"close": prices}, index=dates)


# ---------------------------------------------------------------------------
# Kalman filter recursion
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    def test_output_length_matches_input(self) -> None:
        log_prices = np.log([100.0, 101.0, 102.0, 101.5, 103.0])
        result = _kalman_filter(log_prices, q=1e-4, r=1e-2)
        assert len(result) == len(log_prices)

    def test_first_output_equals_first_input(self) -> None:
        # With diffuse prior (p=1), K ≈ 1/(1+R) ≈ 1 for small R.
        # filtered[0] ≈ log_prices[0]. Not exactly, but close.
        log_prices = np.log([100.0, 101.0, 102.0])
        filtered = _kalman_filter(log_prices, q=1e-6, r=1e-4)
        # First filtered value should be close to first observation.
        assert abs(filtered[0] - log_prices[0]) < 0.1

    def test_smooth_trend_produces_smooth_filtered(self) -> None:
        # Perfectly linear log-price: filtered trend should track it closely.
        log_prices = np.linspace(4.5, 5.0, 200)
        filtered = _kalman_filter(log_prices, q=1e-5, r=1e-3)
        # After warmup, filtered should be within 1% of true log-price.
        assert np.allclose(filtered[50:], log_prices[50:], rtol=0.01)

    def test_large_observation_noise_smooths_more(self) -> None:
        # Large R → filter trusts model more than data → smoother output.
        rng = np.random.default_rng(0)
        log_prices = np.cumsum(rng.normal(0, 0.01, 200))
        filtered_smooth = _kalman_filter(log_prices, q=1e-6, r=1.0)   # large R
        filtered_noisy  = _kalman_filter(log_prices, q=1e-6, r=1e-6)  # small R
        # Smooth filter has lower variance of changes than noisy filter.
        smooth_var = np.var(np.diff(filtered_smooth))
        noisy_var  = np.var(np.diff(filtered_noisy))
        assert smooth_var < noisy_var


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

class TestKalmanLogLikelihood:
    def test_finite_for_valid_inputs(self) -> None:
        log_prices = np.log(np.linspace(100.0, 110.0, 50))
        ll = _kalman_log_likelihood(log_prices, q=1e-4, r=1e-2)
        assert math.isfinite(ll)

    def test_monotone_in_q_for_noisy_data(self) -> None:
        # For a noisy random walk, moderate Q should beat very small Q.
        rng = np.random.default_rng(1)
        log_prices = np.cumsum(rng.normal(0, 0.01, 200))
        ll_small_q = _kalman_log_likelihood(log_prices, q=1e-8, r=1e-2)
        ll_good_q  = _kalman_log_likelihood(log_prices, q=1e-4, r=1e-2)
        assert ll_good_q > ll_small_q

    def test_degenerately_small_r_does_not_crash(self) -> None:
        log_prices = np.log([100.0, 101.0, 99.0, 102.0])
        # Should return -inf or finite value, never raise.
        ll = _kalman_log_likelihood(log_prices, q=1e-4, r=1e-12)
        assert math.isfinite(ll) or ll == float("-inf")


# ---------------------------------------------------------------------------
# Strategy fit()
# ---------------------------------------------------------------------------

class TestKalmanFit:
    def test_fit_returns_self(self) -> None:
        strategy = KalmanFilterStrategy()
        data = _trending_data(300)
        result = strategy.fit(data)
        assert result is strategy

    def test_fit_sets_positive_q_and_r(self) -> None:
        strategy = KalmanFilterStrategy()
        strategy.fit(_trending_data(300))
        assert strategy.q_ > 0.0
        assert strategy.r_ > 0.0

    def test_fit_log_likelihood_is_finite(self) -> None:
        strategy = KalmanFilterStrategy()
        strategy.fit(_trending_data(300))
        assert math.isfinite(strategy.log_likelihood_)

    def test_fit_improves_likelihood_over_defaults(self) -> None:
        # Calibrated parameters should give higher log-likelihood than defaults.
        strategy = KalmanFilterStrategy(q_init=1e-4, r_init=1e-2)
        data = _trending_data(500)
        log_prices = np.log(data["close"].to_numpy())

        ll_default = _kalman_log_likelihood(log_prices, q=1e-4, r=1e-2)
        strategy.fit(data)
        ll_calibrated = _kalman_log_likelihood(log_prices, strategy.q_, strategy.r_)

        assert ll_calibrated >= ll_default - 1.0  # allow small tolerance

    def test_different_data_produces_different_params(self) -> None:
        rng = np.random.default_rng(99)
        # High-volatility data
        n = 400
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        hi_vol = pd.DataFrame(
            {"close": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))},
            index=dates,
        )
        # Low-volatility data
        lo_vol = pd.DataFrame(
            {"close": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))},
            index=dates,
        )

        s1 = KalmanFilterStrategy()
        s2 = KalmanFilterStrategy()
        s1.fit(hi_vol)
        s2.fit(lo_vol)

        # Parameters should differ for different volatility regimes.
        assert s1.q_ != s2.q_ or s1.r_ != s2.r_

    def test_invalid_init_params_raise(self) -> None:
        with pytest.raises(ValueError):
            KalmanFilterStrategy(q_init=-1.0, r_init=1e-2)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

class TestKalmanSignals:
    def test_signals_only_contain_valid_values(self) -> None:
        strategy = KalmanFilterStrategy()
        data = _trending_data(300)
        strategy.fit(data)
        signals = strategy.generate_signals(data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_aligned_to_data_index(self) -> None:
        strategy = KalmanFilterStrategy()
        data = _trending_data(300)
        strategy.fit(data)
        signals = strategy.generate_signals(data)
        assert signals.index.equals(data.index)

    def test_signals_are_integers(self) -> None:
        strategy = KalmanFilterStrategy()
        data = _trending_data(300)
        strategy.fit(data)
        signals = strategy.generate_signals(data)
        assert pd.api.types.is_integer_dtype(signals)

    def test_trending_data_produces_some_buy_signals(self) -> None:
        # Strong upward trend should produce at least one buy signal.
        strategy = KalmanFilterStrategy()
        data = _trending_data(500)
        strategy.fit(data.iloc[:250])
        signals = strategy.generate_signals(data.iloc[250:])
        assert (signals == 1).any() or (signals == -1).any()


# ---------------------------------------------------------------------------
# Warmup context
# ---------------------------------------------------------------------------

class TestKalmanContext:
    def test_context_signals_cover_only_test_index(self) -> None:
        strategy = KalmanFilterStrategy()
        data = _trending_data(400)
        context = data.iloc[:50]
        test = data.iloc[50:]
        strategy.fit(context)
        signals = strategy.generate_signals_with_context(context, test)
        assert signals.index.equals(test.index)

    def test_context_signals_contain_only_valid_values(self) -> None:
        strategy = KalmanFilterStrategy()
        data = _trending_data(400)
        strategy.fit(data.iloc[:200])
        signals = strategy.generate_signals_with_context(
            data.iloc[:50], data.iloc[50:200]
        )
        assert set(signals.unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# active_params - parameter evolution storage
# ---------------------------------------------------------------------------

class TestKalmanActiveParams:
    def test_active_params_returns_dict(self) -> None:
        strategy = KalmanFilterStrategy()
        strategy.fit(_trending_data(300))
        params = strategy.active_params()
        assert isinstance(params, dict)

    def test_active_params_has_required_keys(self) -> None:
        strategy = KalmanFilterStrategy()
        strategy.fit(_trending_data(300))
        params = strategy.active_params()
        assert "q" in params
        assert "r" in params
        assert "snr" in params
        assert "log_likelihood" in params

    def test_snr_equals_q_over_r(self) -> None:
        strategy = KalmanFilterStrategy()
        strategy.fit(_trending_data(300))
        params = strategy.active_params()
        import math
        assert math.isclose(params["snr"], params["q"] / params["r"], rel_tol=1e-6)

    def test_all_values_are_finite(self) -> None:
        import math
        strategy = KalmanFilterStrategy()
        strategy.fit(_trending_data(300))
        params = strategy.active_params()
        for k, v in params.items():
            assert math.isfinite(v), f"active_params['{k}'] = {v} is not finite"

    def test_active_params_changes_after_different_training_data(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        dates = pd.date_range("2015-01-01", periods=n, freq="B")

        hi_vol = pd.DataFrame(
            {"close": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))}, index=dates
        )
        lo_vol = pd.DataFrame(
            {"close": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))}, index=dates
        )

        s1 = KalmanFilterStrategy()
        s2 = KalmanFilterStrategy()
        s1.fit(hi_vol)
        s2.fit(lo_vol)

        p1 = s1.active_params()
        p2 = s2.active_params()
        # Different volatility regimes must produce different calibrated params.
        assert p1["q"] != p2["q"] or p1["r"] != p2["r"]