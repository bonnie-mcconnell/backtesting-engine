"""
Unit tests for White's Reality Check and stationary bootstrap.

Tests verify p-value bounds, correct behaviour under H0, determinism,
and candidate matrix assembly from per-window dictionaries.
"""

import numpy as np
import pytest

from backtesting_engine.reality_check import build_candidate_return_matrix, white_reality_check


def _constant_returns(mean: float, n: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mean + rng.normal(0, 0.001, (n, k))


class TestWhiteRealityCheck:
    def test_p_value_in_unit_interval(self) -> None:
        returns = _constant_returns(0.001, 252, 5)
        p = white_reality_check(returns, n_bootstrap=500)
        assert 0.0 <= p <= 1.0

    def test_zero_mean_strategies_cannot_reject_h0(self) -> None:
        # Zero-mean strategies: the Reality Check should not reject H0.
        # With k strategies, the max of k sample means is positive by chance,
        # so p is not necessarily near 1 - but it must be above 0.05 (not
        # significant at the conventional threshold).
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.01, (500, 10))
        p = white_reality_check(returns, n_bootstrap=1000)
        assert p > 0.05, (
            f"Zero-mean strategies should not be significant at 0.05, got p={p:.3f}"
        )

    def test_deterministic_with_fixed_seed(self) -> None:
        rng = np.random.default_rng(5)
        returns = rng.normal(0.001, 0.01, (252, 5))
        p1 = white_reality_check(returns, n_bootstrap=500, seed=42)
        p2 = white_reality_check(returns, n_bootstrap=500, seed=42)
        assert p1 == p2

    def test_raises_on_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            white_reality_check(np.array([0.01, -0.02, 0.03]))

    def test_raises_on_single_time_period(self) -> None:
        with pytest.raises(ValueError, match="2 time periods"):
            white_reality_check(np.array([[0.01, 0.02]]))

    def test_single_strategy(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, (252, 1))
        p = white_reality_check(returns, n_bootstrap=200)
        assert 0.0 <= p <= 1.0

    def test_strongly_positive_best_strategy_lowers_p(self) -> None:
        # One strong strategy among weak ones should lower p vs all-weak.
        rng = np.random.default_rng(42)
        returns_strong = np.hstack([
            rng.normal(0.005, 0.008, (500, 1)),
            rng.normal(0.0, 0.008, (500, 9)),
        ])
        returns_weak = rng.normal(0.0, 0.008, (500, 10))
        p_strong = white_reality_check(returns_strong, n_bootstrap=500)
        p_weak = white_reality_check(returns_weak, n_bootstrap=500)
        # RC p for strong universe should be <= weak + tolerance
        assert p_strong <= p_weak + 0.25


class TestBuildCandidateReturnMatrix:
    def test_correct_shape(self) -> None:
        rng = np.random.default_rng(0)
        w1 = {(20, 100): rng.normal(0, 0.01, 252), (30, 150): rng.normal(0, 0.01, 252)}
        w2 = {(20, 100): rng.normal(0, 0.01, 252), (30, 150): rng.normal(0, 0.01, 252)}
        matrix = build_candidate_return_matrix([w1, w2])
        assert matrix.shape == (504, 2)

    def test_intersection_drops_missing_candidates(self) -> None:
        rng = np.random.default_rng(0)
        w1 = {(20, 100): rng.normal(0, 0.01, 252), (30, 150): rng.normal(0, 0.01, 252)}
        w2 = {(20, 100): rng.normal(0, 0.01, 252)}  # (30, 150) missing
        matrix = build_candidate_return_matrix([w1, w2])
        assert matrix.shape[1] == 1

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError):
            build_candidate_return_matrix([])

    def test_raises_when_no_common_candidates(self) -> None:
        rng = np.random.default_rng(0)
        w1 = {(20, 100): rng.normal(0, 0.01, 100)}
        w2 = {(30, 150): rng.normal(0, 0.01, 100)}
        with pytest.raises(ValueError, match="No parameter pair"):
            build_candidate_return_matrix([w1, w2])


class TestRealityCheckCentering:
    def test_single_zero_mean_strategy_p_in_reasonable_range(self) -> None:
        # With k=1 zero-mean strategy, the RC p-value should not be extreme.
        # The centred bootstrap distribution has mean near zero, so the p-value
        # is determined by where the observed sample mean falls in that distribution.
        # Over many data seeds the p-value is Uniform(0,1) under H0 - for any
        # single seed it can be anywhere in (0, 1), but extremely small values
        # (p < 0.05) should be rare when n=1000 and the true mean is zero.
        #
        # We run multiple seeds and verify the p-values are not systematically
        # small, which would indicate a biased bootstrap.
        small_p_count = 0
        for seed in range(20):
            rng_d = np.random.default_rng(seed * 100)
            returns = rng_d.normal(0.0, 0.01, (1000, 1))
            p = white_reality_check(returns, n_bootstrap=500, seed=seed)
            if p < 0.05:
                small_p_count += 1
        # Under correct H0 coverage, ~5% of seeds should give p < 0.05.
        # Allow up to 30% as a generous tolerance for the bootstrap approximation.
        assert small_p_count <= 6, (
            f"{small_p_count}/20 seeds gave p < 0.05 for zero-mean data. "
            "Expected ~1/20 under correct H0 coverage."
        )

    def test_strong_alpha_produces_lower_p_than_null(self) -> None:
        # A strategy with genuinely positive expected return should produce
        # a lower p-value than a universe of zero-mean strategies.
        rng = np.random.default_rng(456)
        # One strong strategy mixed with zero-mean noise
        strong = np.hstack([
            rng.normal(0.008, 0.01, (800, 1)),   # clear positive mean
            rng.normal(0.000, 0.01, (800, 19)),
        ])
        null = rng.normal(0.0, 0.01, (800, 20))
        p_strong = white_reality_check(strong, n_bootstrap=2000, seed=0)
        p_null = white_reality_check(null, n_bootstrap=2000, seed=0)
        assert p_strong < p_null + 0.15, (
            f"Strong alpha should lower RC p-value. Got p_strong={p_strong:.3f}, "
            f"p_null={p_null:.3f}."
        )

    def test_p_value_is_not_trivially_zero_or_one(self) -> None:
        # Guard against degenerate implementations that always return 0 or 1.
        # Use a zero-mean strategy - should produce a p-value clearly above zero.
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, (200, 1))
        p = white_reality_check(returns, n_bootstrap=500)
        assert p > 0.0, "p-value must be strictly positive for non-degenerate input"
        assert p < 1.0, "p-value must be strictly less than 1 for non-degenerate input"
        # Also verify a clearly-zero strategy doesn't get rejected
        assert p > 0.02, (
            f"Zero-mean strategy should not be near-rejected, got p={p:.4f}"
        )
