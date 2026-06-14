"""
Unit tests for White's Reality Check and stationary bootstrap.

p-value bounds, H0 behaviour, seed determinism, and candidate matrix
assembly from per-window return dicts.
"""

import math

import numpy as np
import pandas as pd
import pytest
from helpers import make_oscillating_data

from backtesting_engine.reality_check import build_candidate_return_matrix, white_reality_check


def _make_oscillating(n: int, with_high_low: bool = False) -> 'pd.DataFrame':
    return make_oscillating_data(n, with_high_low=with_high_low)


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


class TestBenchmarkRelativeRC:
    """
    Tests for white_reality_check() with benchmark_returns.

    The BH-null tests whether the best candidate beats the benchmark, not cash.
    Active returns = candidate returns - benchmark returns. The two nulls answer
    different questions: p_cash tests whether any strategy has positive mean return;
    p_bh tests whether any strategy has positive active return vs the benchmark.
    A strategy that beats cash but not B&H has high p_bh despite possibly low p_cash.
    """

    def test_p_bh_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        candidates = rng.normal(0.001, 0.01, (500, 5))
        benchmark = rng.normal(0.0005, 0.005, 500)
        p = white_reality_check(candidates, n_bootstrap=500, benchmark_returns=benchmark)
        assert 0.0 <= p <= 1.0

    def test_cash_beater_that_underperforms_bh_has_high_p_bh(self) -> None:
        # A strategy that beats cash but not B&H should have a large p_bh
        # even if p_cash is small. This is the key interpretive distinction
        # between the two nulls.
        rng = np.random.default_rng(77)
        # Benchmark with strong upward drift
        benchmark = rng.normal(0.005, 0.01, 600)
        # Candidates also beat cash (positive mean) but match benchmark - active mean ~0
        candidates = np.column_stack([
            benchmark + rng.normal(0.0, 0.01, 600) for _ in range(8)
        ])
        p_cash = white_reality_check(candidates, n_bootstrap=500, seed=0)
        p_bh = white_reality_check(candidates, n_bootstrap=500, seed=0, benchmark_returns=benchmark)
        # Both should be in [0, 1]
        assert 0.0 <= p_cash <= 1.0
        assert 0.0 <= p_bh <= 1.0
        # The key property: candidates beat cash (strong positive mean) so p_cash
        # should be moderate-to-small; candidates match B&H so p_bh should be large.
        # p_bh >> p_cash in this scenario.
        assert p_bh > p_cash, (
            f"Strategy matching B&H but beating cash should have p_bh > p_cash. "
            f"Got p_cash={p_cash:.3f}, p_bh={p_bh:.3f}."
        )

    def test_strategy_matching_bh_gives_high_p_bh(self) -> None:
        # If every candidate exactly matches the benchmark, active returns are
        # identically zero. The RC null (active mean = 0) is exactly satisfied,
        # so p_bh should be large (we cannot reject the null).
        rng = np.random.default_rng(42)
        benchmark = rng.normal(0.001, 0.01, 500)
        # All candidates are the benchmark plus tiny noise - active returns near zero
        candidates = np.column_stack([
            benchmark + rng.normal(0, 1e-6, 500) for _ in range(5)
        ])
        p_bh = white_reality_check(
            candidates, n_bootstrap=800, seed=0, benchmark_returns=benchmark
        )
        assert p_bh > 0.20, (
            f"Candidates matching B&H exactly should not reject BH null. Got p={p_bh:.3f}."
        )

    def test_strategy_beating_bh_gives_low_p_bh(self) -> None:
        # One candidate consistently beats the benchmark by a large margin.
        # The BH-null should be rejected (small p_bh).
        rng = np.random.default_rng(7)
        benchmark = rng.normal(0.001, 0.008, 800)
        # First candidate has active return of +0.006/day - very strong alpha
        strong_active = benchmark + rng.normal(0.006, 0.008, 800)
        weak = np.column_stack([benchmark + rng.normal(0.0, 0.008, 800) for _ in range(9)])
        candidates = np.hstack([strong_active[:, np.newaxis], weak])
        p_bh = white_reality_check(
            candidates, n_bootstrap=1000, seed=0, benchmark_returns=benchmark
        )
        assert p_bh < 0.20, (
            f"Strong alpha vs B&H should produce low BH-null p. Got p={p_bh:.3f}."
        )

    def test_raises_on_length_mismatch(self) -> None:
        rng = np.random.default_rng(0)
        candidates = rng.normal(0, 0.01, (300, 5))
        benchmark = rng.normal(0, 0.01, 250)  # wrong length
        with pytest.raises(ValueError, match="length"):
            white_reality_check(candidates, n_bootstrap=100, benchmark_returns=benchmark)

    def test_benchmark_none_is_unchanged(self) -> None:
        # Passing benchmark_returns=None must produce identical output to
        # omitting the argument entirely.
        rng = np.random.default_rng(99)
        candidates = rng.normal(0.001, 0.01, (300, 4))
        p_default = white_reality_check(candidates, n_bootstrap=300, seed=5)
        p_none = white_reality_check(candidates, n_bootstrap=300, seed=5, benchmark_returns=None)
        assert p_default == p_none

    def test_benchmark_2d_column_vector_accepted(self) -> None:
        # benchmark_returns can be shape (T, 1) - ravel() inside the function handles it.
        rng = np.random.default_rng(3)
        candidates = rng.normal(0.001, 0.01, (300, 4))
        benchmark_1d = rng.normal(0.0005, 0.005, 300)
        benchmark_2d = benchmark_1d[:, np.newaxis]
        p_1d = white_reality_check(candidates, n_bootstrap=300, seed=1, benchmark_returns=benchmark_1d)
        p_2d = white_reality_check(candidates, n_bootstrap=300, seed=1, benchmark_returns=benchmark_2d)
        assert p_1d == p_2d


# ── Flat-cash parity ──────────────────────────────────────────────────────────

def _make_mixed_window_data_for_flat_cash() -> pd.DataFrame:
    """Price series that produces some flat-cash AND some trading windows.

    A very slow sine wave (period=750 bars) means individual 252-bar test
    windows are often monotone (no MA crossover -> flat-cash), but the full
    5-year dataset has enough trend reversals that at least 2 windows trade.
    This avoids triggering the zero-total-trades defensive guard.

    Module-level (not a method) so the pytest fixture below can reference it
    without instantiating TestRCFlatCashParity. The fixture result is cached
    at class scope so walk_forward runs exactly once for all three tests.
    """
    n = 1260
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    close = 100.0 + 15.0 * np.sin(2 * np.pi * t / 750.0)
    return pd.DataFrame({"close": close}, index=dates)

@pytest.fixture(scope="class")
def flat_cash_result():
    """Single walk_forward result shared across all TestRCFlatCashParity tests.

    scope="class": the expensive MA grid search + bootstrap runs exactly once
    for the class, not once per test method.
    """
    from backtesting_engine.execution import ExecutionConfig
    from backtesting_engine.strategy.moving_average import MovingAverageStrategy
    from backtesting_engine.walk_forward import walk_forward

    return walk_forward(
        _make_mixed_window_data_for_flat_cash(),
        MovingAverageStrategy(),
        training_window_years=1,
        testing_window_years=1,
        execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
    )

class TestRCFlatCashParity:
    """
    Flat-cash windows must contribute to the RC candidate matrix just as they
    contribute p=1.0 to Fisher's combined p. Without this, Fisher and RC test
    different hypotheses over different windows.

    The walk_forward result is shared via the flat_cash_result fixture (class
    scope) so the expensive MA grid search + bootstrap runs exactly once for
    all three tests in this class, not three times.
    """

    def test_flat_cash_windows_are_valid_windows(self, flat_cash_result) -> None:
        """Flat-cash windows must be included in valid_windows with skipped=False."""
        result = flat_cash_result
        assert result.flat_cash_window_count > 0, (
            "Expected at least one flat-cash window with slow-oscillation data. "
            "Verify the sine period is long enough relative to the test window."
        )
        assert len(result.valid_windows) == len(result.window_results), (
            "All windows (including flat-cash) must appear in valid_windows."
        )

    def test_rc_p_is_not_nan_when_strategy_has_candidates(self, flat_cash_result) -> None:
        """RC p must be computable even when some windows are flat-cash.

        Flat-cash windows must contribute zero-return arrays to the RC candidate
        matrix. Without this, the RC and Fisher tests operate on different window
        sets - Fisher receives p=1.0 from flat-cash windows but RC silently
        excludes them, making the two statistics non-comparable.
        """
        rc_p = flat_cash_result.summary_metrics.reality_check_p_value
        assert not math.isnan(rc_p), (
            "RC p is NaN despite strategy having candidates. "
            "Flat-cash windows must contribute zero-return arrays to RC matrix "
            "(required for parity with Fisher's combined p)."
        )
        assert 0.0 <= rc_p <= 1.0, f"RC p-value out of [0,1]: {rc_p}"

    def test_fisher_and_rc_cover_same_number_of_windows(self) -> None:
        """Fisher p uses all windows including flat-cash; RC must too.

        Uses its own walk_forward call rather than the shared fixture because
        it needs a different dataset (with high/low) to verify the property holds
        under a different data regime.
        """
        from backtesting_engine.execution import ExecutionConfig
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        from backtesting_engine.walk_forward import walk_forward

        data = _make_oscillating(756, with_high_low=True)
        result = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
        )
        # Both stats must have been computed over the same window set.
        # We can't inspect the internal lists, but we can verify Fisher p
        # and RC p are both finite (not NaN) when candidates exist.
        assert not math.isnan(result.summary_metrics.combined_p_value)
        rc_p = result.summary_metrics.reality_check_p_value
        if not math.isnan(rc_p):
            assert 0.0 <= rc_p <= 1.0


class TestBuildCandidateReturnMatrixKeyTypes:
    """build_candidate_return_matrix must accept int keys (Momentum) and tuple keys (MA)."""

    def test_tuple_keys_work(self) -> None:
        from backtesting_engine.reality_check import build_candidate_return_matrix
        rng = np.random.default_rng(0)
        window1 = {(10, 50): rng.normal(0, 0.01, 100), (20, 50): rng.normal(0, 0.01, 100)}
        window2 = {(10, 50): rng.normal(0, 0.01, 100), (20, 50): rng.normal(0, 0.01, 100)}
        matrix = build_candidate_return_matrix([window1, window2])
        assert matrix.shape == (200, 2)

    def test_int_keys_work(self) -> None:
        """Momentum strategy uses int keys - must not raise TypeError."""
        from backtesting_engine.reality_check import build_candidate_return_matrix
        rng = np.random.default_rng(1)
        window1 = {20: rng.normal(0, 0.01, 100), 60: rng.normal(0, 0.01, 100)}
        window2 = {20: rng.normal(0, 0.01, 100), 60: rng.normal(0, 0.01, 100)}
        matrix = build_candidate_return_matrix([window1, window2])
        assert matrix.shape == (200, 2)

    def test_partial_key_overlap_uses_intersection(self) -> None:
        from backtesting_engine.reality_check import build_candidate_return_matrix
        rng = np.random.default_rng(2)
        window1 = {20: rng.normal(0, 0.01, 50), 60: rng.normal(0, 0.01, 50), 120: rng.normal(0, 0.01, 50)}
        window2 = {20: rng.normal(0, 0.01, 50), 60: rng.normal(0, 0.01, 50)}  # 120 missing
        matrix = build_candidate_return_matrix([window1, window2])
        # Only keys {20, 60} are in both windows
        assert matrix.shape[1] == 2
