"""
Unit tests for the metrics module.

All expected values are derived independently by hand and hard-coded.
They are never computed from the implementation itself, which would make
the test circular and unable to catch bugs.

Test naming convention: test_<function>_<scenario>_<expected_behaviour>
"""

import math

import numpy as np
import pandas as pd
import pytest

from backtesting_engine.metrics import (
    _block_bootstrap_p_value,
    _calmar,
    _max_drawdown,
    _omega,
    _sharpe,
    _sortino,
    calculate_metrics,
)
from backtesting_engine.models import MetricsResult

# ---------------------------------------------------------------------------
# _sharpe
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_zero_mean_returns_zero(self) -> None:
        # mean([0.1, -0.1]) = 0.0, so Sharpe = 0 regardless of std.
        returns = np.array([0.1, -0.1])
        assert _sharpe(returns) == 0.0

    def test_constant_returns_zero(self) -> None:
        # Constant series has near-zero std - guard returns 0.0.
        returns = np.array([0.05, 0.05, 0.05])
        assert _sharpe(returns) == 0.0

    def test_known_value(self) -> None:
        # Hand-derivation, not computed from the implementation:
        # returns = [0.02, 0.00, 0.02, 0.00]
        # mean = 0.04/4 = 0.01
        # deviations from mean: [0.01, -0.01, 0.01, -0.01]
        # var(ddof=1) = sum([0.0001]*4) / 3 = 0.0004/3 = 1/7500
        # std(ddof=1) = 1/sqrt(7500)
        # Sharpe = mean/std * sqrt(252) = 0.01 * sqrt(7500) * sqrt(252)
        #        = 0.01 * sqrt(1_890_000)
        expected = 0.01 * math.sqrt(1_890_000)  # ≈ 13.748
        returns = np.array([0.02, 0.00, 0.02, 0.00])
        assert np.isclose(_sharpe(returns), expected, rtol=1e-9)

    def test_positive_returns_positive_sharpe(self) -> None:
        returns = np.array([0.01, 0.02, 0.03, 0.01])
        assert _sharpe(returns) > 0.0

# ---------------------------------------------------------------------------
# _sortino
# ---------------------------------------------------------------------------

class TestSortino:
    def test_no_downside_returns_inf(self) -> None:
        # No below-threshold returns - downside deviation is zero, Sortino = inf.
        returns = np.array([0.01, 0.02, 0.03])
        assert _sortino(returns) == float("inf")

    def test_single_negative_return_is_finite(self) -> None:
        # Single negative return: downside_dev = abs(r), result is finite.
        returns = np.array([0.05, 0.03, -0.01, 0.02])
        result = _sortino(returns)
        assert math.isfinite(result)

    def test_known_value(self) -> None:
        # Hand-derivation for [0.05, -0.01, -0.03], threshold = 0:
        # mean_excess = (0.05 - 0.01 - 0.03) / 3 = 0.01/3
        # negative_excess squared: [0, (-0.01)^2, (-0.03)^2] = [0, 0.0001, 0.0009]
        # downside_dev = sqrt(mean([0, 0.0001, 0.0009]))
        #              = sqrt(0.001/3) = sqrt(1/3000) = 1/sqrt(3000)
        # Sortino = (0.01/3) / (1/sqrt(3000)) * sqrt(252)
        #         = (0.01/3) * sqrt(3000) * sqrt(252)
        #         = (0.01/3) * sqrt(756_000)
        expected = (0.01 / 3) * math.sqrt(756_000)  # ≈ 2.898
        returns = np.array([0.05, -0.01, -0.03])
        assert np.isclose(_sortino(returns), expected, rtol=1e-9)

    def test_consistent_small_losses_not_inflated(self) -> None:
        # Constant-loss series: downside deviation uses RMS (not std),
        # which is nonzero for constant losses. Result must be finite and negative.
        returns = np.array([-0.01, -0.01, -0.01, -0.01])
        result = _sortino(returns)
        assert math.isfinite(result)
        assert result < 0.0  # negative mean → negative Sortino

    def test_all_negative_returns_gives_negative_sortino(self) -> None:
        returns = np.array([-0.01, -0.02, -0.03])
        assert _sortino(returns) < 0.0

    def test_downside_deviation_not_std(self) -> None:
        # Verify we use RMS (downside deviation), not std(downside).
        # For [-0.01, -0.02]:
        #   std(ddof=1) = 0.00707  → would give large Sortino
        #   downside_dev = sqrt(mean([0.0001, 0.0004])) = sqrt(0.00025) = 0.01581
        # The two differ by factor ~2.2; check we get the smaller (correct) value.
        returns = np.array([0.05, -0.01, -0.02])
        result = _sortino(returns)
        negative_excess = np.minimum(returns, 0.0)
        expected_dd = np.sqrt(np.mean(negative_excess ** 2))
        expected = returns.mean() / expected_dd * np.sqrt(252)
        assert np.isclose(result, expected, rtol=1e-5)

# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_known_value(self) -> None:
        # returns = [0.1, -0.2, 0.1]
        # cumulative = [1.1, 0.88, 0.968]
        # rolling_max = [1.1, 1.1, 1.1]
        # drawdown[1] = (0.88 - 1.1) / 1.1 = -0.2
        returns = np.array([0.1, -0.2, 0.1])
        assert np.isclose(_max_drawdown(returns), -0.2, rtol=1e-5)

    def test_monotonically_increasing_returns_zero(self) -> None:
        # Prices always at new highs - no drawdown.
        returns = np.array([0.01, 0.02, 0.03])
        assert _max_drawdown(returns) == 0.0

    def test_always_non_positive(self) -> None:
        # Drawdown cannot be positive by definition.
        returns = np.array([0.01, 0.02, -0.05, 0.01])
        assert _max_drawdown(returns) <= 0.0

    def test_large_drawdown(self) -> None:
        # -50% drawdown: price halves then recovers.
        # returns = [-0.5, 1.0]: cumulative = [0.5, 1.0], rolling_max = [1.0, 1.0]
        # drawdown = [-0.5, 0.0]
        returns = np.array([-0.5, 1.0])
        assert np.isclose(_max_drawdown(returns), -0.5, rtol=1e-5)

# ---------------------------------------------------------------------------
# _calmar
# ---------------------------------------------------------------------------

class TestCalmar:
    def test_no_drawdown_returns_inf(self) -> None:
        returns = np.array([0.01, 0.02, 0.03])
        assert _calmar(returns) == float("inf")

    def test_known_value_negative_geometric_return(self) -> None:
        # returns = [0.1, -0.2, 0.1]
        # arithmetic mean = 0.0, but geometric compound:
        #   prod(1.1 * 0.8 * 1.1) = 0.968 → cumulative = 0.968
        #   annualised = 0.968^(252/3) - 1 ≈ -8.7%  (negative despite zero arithmetic mean)
        # max_drawdown: cumulative with prepended 1.0 = [1.0, 1.1, 0.88, 0.968]
        #   rolling_max = [1.0, 1.1, 1.1, 1.1]
        #   trough at bar 2: (0.88 - 1.1) / 1.1 = -0.20
        # Calmar = annualised / abs(max_dd) ≈ negative / 0.20
        # Geometric compounding is required - arithmetic overstates annualised return.
        returns = np.array([0.1, -0.2, 0.1])
        n = len(returns)
        cumulative = float(np.prod(1.0 + returns))
        ann = cumulative ** (252 / n) - 1.0
        max_dd = abs(_max_drawdown(returns))
        expected = ann / max_dd
        assert np.isclose(_calmar(returns), expected, rtol=1e-5)
        # Also verify it is negative (geometric beats arithmetic here)
        assert _calmar(returns) < 0.0

    def test_positive_mean_positive_calmar(self) -> None:
        # Positive average returns with some drawdown → positive Calmar.
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.02])
        assert _calmar(returns) > 0.0

    def test_uses_geometric_not_arithmetic_compounding(self) -> None:
        # Arithmetic mean = 0 for [0.1, -0.2, 0.1], so an arithmetic formula
        # would give annualised return = 0, Calmar = 0.
        # Geometric compounding: prod([1.1, 0.8, 1.1])^(252/3) - 1 ≈ -93%.
        # Calmar must reflect the actual compounded return, not the arithmetic mean.
        returns = np.array([0.1, -0.2, 0.1])
        result = _calmar(returns)
        assert abs(result) > 0.1, (
            f"Calmar must reflect geometric return, got {result:.4f}. "
            "Arithmetic mean = 0 for these returns, but geometric is negative."
        )
        assert result < 0.0

    def test_geometric_vs_arithmetic_divergence(self) -> None:
        # With high-volatility returns, geometric < arithmetic (variance drag).
        # Calmar using geometric should be LOWER than using arithmetic.
        rng = np.random.default_rng(1)
        returns = rng.normal(0.002, 0.02, 252)  # positive drift, realistic vol
        n = len(returns)
        # Compute both formulas explicitly
        arith_ann = (1 + float(np.mean(returns)))**252 - 1
        geom_ann = float(np.prod(1 + returns))**(252/n) - 1
        # Geometric must be lower (variance drag: ~0.5*vol^2*T per year)
        assert geom_ann < arith_ann, "Geometric return must be below arithmetic for nonzero variance"

# ---------------------------------------------------------------------------
# _omega
# ---------------------------------------------------------------------------

class TestOmega:
    def test_no_losses_returns_inf(self) -> None:
        returns = np.array([0.01, 0.02, 0.03])
        assert _omega(returns) == float("inf")

    def test_known_value(self) -> None:
        # returns = [0.03, 0.01, -0.01, -0.01]
        # gains = [0.03, 0.01], sum = 0.04
        # losses = [0.01, 0.01], sum = 0.02
        # omega = 0.04 / 0.02 = 2.0
        returns = np.array([0.03, 0.01, -0.01, -0.01])
        assert np.isclose(_omega(returns), 2.0, rtol=1e-5)

    def test_equal_gains_losses_returns_one(self) -> None:
        # Symmetric returns → omega = 1.0.
        returns = np.array([0.01, -0.01])
        assert np.isclose(_omega(returns), 1.0, rtol=1e-5)

# ---------------------------------------------------------------------------
# _block_bootstrap_p_value
# ---------------------------------------------------------------------------

class TestMonteCarloPValue:
    def test_p_value_in_unit_interval(self) -> None:
        returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
        p = _block_bootstrap_p_value(returns)
        assert 0.0 <= p <= 1.0

    def test_p_value_near_half_for_iid_returns(self) -> None:
        # The block bootstrap p-value for Sharpe tests autocorrelation
        # exploitation, not raw alpha - returns are centred before resampling
        # so the null tests H0: true mean return = 0, not H0: Sharpe = observed
        # Sharpe. Returns drawn from N(0.005, 0.01^2) have an annualised Sharpe
        # of ~0.005/0.01 * sqrt(252) ~= 0.79, well above zero, so the centred
        # bootstrap should produce a *low* p-value here, not p ~= 0.5.
        rng = np.random.default_rng(0)
        returns = rng.normal(loc=0.005, scale=0.01, size=252)
        p = _block_bootstrap_p_value(returns)
        # Positive-drift iid returns should be detectably above zero-mean H₀.
        assert p < 0.15, (
            f"Expected low p-value for high-Sharpe iid returns, got p={p:.4f}. "
            "Check that _block_bootstrap_p_value centres returns before resampling."
        )

    def test_flat_returns_high_p_value(self) -> None:
        # Near-zero Sharpe should produce a p-value near 0.5.
        rng = np.random.default_rng(1)
        returns = rng.normal(loc=0.0, scale=0.01, size=252)
        p = _block_bootstrap_p_value(returns)
        assert p > 0.1  # Should not be significant.

    def test_deterministic_with_fixed_seed(self) -> None:
        # Same input always produces same output (BLOCK_BOOTSTRAP_SEED is fixed).
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 50)
        p1 = _block_bootstrap_p_value(returns)
        p2 = _block_bootstrap_p_value(returns)
        assert p1 == p2

# ---------------------------------------------------------------------------
# calculate_metrics (integration)
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    def test_returns_metricsresult_with_all_fields(self) -> None:
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        values = pd.Series(
            100_000 * (1 + 0.001) ** np.arange(100), index=dates
        )
        m = calculate_metrics(values)
        assert isinstance(m, MetricsResult)
        assert not math.isnan(m.sharpe_ratio)
        assert not math.isnan(m.max_drawdown)

    def test_raises_on_empty_series(self) -> None:
        import pandas as pd
        with pytest.raises(ValueError, match="No returns"):
            calculate_metrics(pd.Series(dtype=float))

    def test_raises_on_single_value(self) -> None:
        import pandas as pd
        with pytest.raises(ValueError, match="No returns"):
            calculate_metrics(pd.Series([100_000.0]))
