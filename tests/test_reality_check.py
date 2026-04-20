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
        # Zero-mean strategies: cannot reject H0 that none beats benchmark.
        # p should be high (we fail to reject, so p near 1).
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.01, (500, 10))
        p = white_reality_check(returns, n_bootstrap=1000)
        assert p > 0.5

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
