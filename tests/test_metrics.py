"""
Unit tests for the metrics module.

All expected values are derived independently by hand and hard-coded.
They are never computed from the implementation itself, which would make
the test circular and unable to catch bugs.

Test naming convention: test_<function>_<scenario>_<expected_behaviour>
"""

import math
import numpy as np
import pytest

from backtesting_engine.metrics import (
    _calmar,
    _max_drawdown,
    _monte_carlo_p_value,
    _omega,
    _sharpe,
    _sortino,
    calculate_metrics,
)


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
        # returns = [0.02, 0.00, 0.02, 0.00]
        # mean = 0.01, std(ddof=1) = 0.011547005383792515
        # Sharpe = 0.01 / 0.011547... * sqrt(252) = 13.744...
        returns = np.array([0.02, 0.00, 0.02, 0.00])
        expected = 0.01 / 0.011547005383792515 * np.sqrt(252)
        assert np.isclose(_sharpe(returns), expected, rtol=1e-5)

    def test_positive_returns_positive_sharpe(self) -> None:
        returns = np.array([0.01, 0.02, 0.03, 0.01])
        assert _sharpe(returns) > 0.0


# ---------------------------------------------------------------------------
# _sortino
# ---------------------------------------------------------------------------

class TestSortino:
    def test_no_downside_returns_inf(self) -> None:
        # No negative returns - Sortino is undefined, returns inf.
        returns = np.array([0.01, 0.02, 0.03])
        assert _sortino(returns) == float("inf")

    def test_single_negative_return_returns_float_not_nan(self) -> None:
        # Previously: ddof=1 with one downside observation → NaN.
        # Fixed: single downside returns 0.0 (variance undefined, not infinite).
        returns = np.array([0.05, 0.03, -0.01, 0.02])
        result = _sortino(returns)
        assert not math.isnan(result)
        assert math.isfinite(result) or result == float("inf")

    def test_known_value(self) -> None:
        # returns = [0.05, -0.01, -0.03]
        # mean = 0.003333, downside = [-0.01, -0.03]
        # downside std(ddof=1) = 0.014142...
        # Sortino = 0.003333 / 0.014142 * sqrt(252) = 3.742...
        returns = np.array([0.05, -0.01, -0.03])
        mean = returns.mean()
        downside_std = np.std([-0.01, -0.03], ddof=1)
        expected = mean / downside_std * np.sqrt(252)
        assert np.isclose(_sortino(returns), expected, rtol=1e-5)

    def test_all_negative_returns_positive_sortino_or_zero(self) -> None:
        # Mean is negative - Sortino should also be negative (or zero if mean=0).
        returns = np.array([-0.01, -0.02, -0.03])
        assert _sortino(returns) < 0.0


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

    def test_known_value_zero_mean(self) -> None:
        # returns = [0.1, -0.2, 0.1]  → mean = 0.0, annualised return = 0.0
        # Calmar = 0.0 / 0.2 = 0.0
        returns = np.array([0.1, -0.2, 0.1])
        assert np.isclose(_calmar(returns), 0.0, atol=1e-5)

    def test_positive_mean_positive_calmar(self) -> None:
        # Positive average returns with some drawdown → positive Calmar.
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.02])
        assert _calmar(returns) > 0.0


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
# _monte_carlo_p_value
# ---------------------------------------------------------------------------

class TestMonteCarloPValue:
    def test_p_value_in_unit_interval(self) -> None:
        returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
        p = _monte_carlo_p_value(returns)
        assert 0.0 <= p <= 1.0

    def test_p_value_near_half_for_iid_returns(self) -> None:
        # The block bootstrap p-value for Sharpe tests autocorrelation exploitation,
        # NOT raw alpha. For iid returns, each resampled block preserves the mean,
        # so the null Sharpe distribution is centred near the observed Sharpe and
        # p ~ 0.5 regardless of the strategy actual mean return.
        #
        # This is a known and documented limitation: the test has no power against
        # iid alternatives. It gains power when the strategy exploits return
        # autocorrelation (e.g. momentum) that block shuffling destroys.
        #
        # Asserting p in [0.2, 0.8] verifies the bootstrap behaves correctly
        # for iid data, not that the strategy is uninformative.
        rng = np.random.default_rng(0)
        returns = rng.normal(loc=0.005, scale=0.01, size=252)
        p = _monte_carlo_p_value(returns)
        assert 0.2 <= p <= 0.8

    def test_flat_returns_high_p_value(self) -> None:
        # Near-zero Sharpe should produce a p-value near 0.5.
        rng = np.random.default_rng(1)
        returns = rng.normal(loc=0.0, scale=0.01, size=252)
        p = _monte_carlo_p_value(returns)
        assert p > 0.1  # Should not be significant.

    def test_deterministic_with_fixed_seed(self) -> None:
        # Same input always produces same output (BLOCK_BOOTSTRAP_SEED is fixed).
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 50)
        p1 = _monte_carlo_p_value(returns)
        p2 = _monte_carlo_p_value(returns)
        assert p1 == p2


# ---------------------------------------------------------------------------
# calculate_metrics (integration)
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    def test_returns_metricsresult_with_all_fields(self) -> None:
        import pandas as pd
        from backtesting_engine.models import MetricsResult

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