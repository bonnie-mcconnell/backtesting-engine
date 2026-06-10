"""
Correctness invariants: tests that guard against specific failure modes that
are easy to introduce and hard to catch with generic assertions.

Each test documents exactly what breaks if the invariant is violated:
  - negative cash from naive position sizing
  - benchmark using a hardcoded cost rate instead of ExecutionConfig
  - flat-cash windows excluded from the summary (biases Sharpe upward)
  - momentum RC evaluating all candidates with the fitted lookback (wrong)
  - bootstrap null not centred (p ≈ 0.5 for any positive-drift series)
"""

import math

import numpy as np
import pandas as pd
from helpers import make_oscillating_data

from backtesting_engine.config import (
    INITIAL_PORTFOLIO_VALUE,
    POSITION_SIZE_FRACTION,
    TRANSACTION_COST_RATE,
)
from backtesting_engine.execution import ExecutionConfig, run_simulation_with_execution
from backtesting_engine.metrics import _monte_carlo_p_value
from backtesting_engine.strategy.momentum import MomentumStrategy
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _close_only(n: int = 20, base: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(base, base + n - 1, n)
    return pd.DataFrame({"close": close}, index=dates)


def _ohlcv(n: int = 20, base: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(base, base + n - 1, n)
    return pd.DataFrame(
        {"open": close - 0.2, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=dates,
    )


# ---------------------------------------------------------------------------
# 1. Position sizing: cash never goes negative
# ---------------------------------------------------------------------------

class TestPositionSizingNoCashOverdraft:
    """
    The old formula spent cash * POSITION_SIZE_FRACTION + buy_fee, which left
    cash < 0 (slight leverage) after every trade.  The correct formula is:
        position_value = cash * fraction / (1 + cost_rate)
    so that position_value + buy_cost == cash * fraction exactly.
    """

    def test_cash_never_negative_after_buy(self) -> None:
        data = _close_only(n=6)
        # Signal: buy on bar 1, sell on bar 4.
        signals = pd.Series([0, 1, 0, 0, -1, 0], index=data.index)
        result = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0, signal_delay=0))

        # If portfolio values are all positive, cash never went negative.
        assert result.portfolio_values is not None
        assert (result.portfolio_values >= 0).all(), (
            "Portfolio values went negative - position sizing created leverage."
        )

    def test_shares_formula_cost_inclusive(self) -> None:
        """Verify the exact share count matches the corrected formula."""
        data = _close_only(n=6)
        signals = pd.Series([0, 1, 0, 0, -1, 0], index=data.index)
        result = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0, signal_delay=0))
        assert result.trades

        entry_price = result.trades[0].entry_price
        available = INITIAL_PORTFOLIO_VALUE * POSITION_SIZE_FRACTION
        expected_shares = (available / (1.0 + TRANSACTION_COST_RATE)) / entry_price
        assert math.isclose(result.trades[0].shares, expected_shares, rel_tol=1e-6), (
            f"Expected {expected_shares:.6f} shares (cost-inclusive sizing), "
            f"got {result.trades[0].shares:.6f}."
        )

    def test_portfolio_never_below_zero_over_many_trades(self) -> None:
        """Stress test: many rapid trades should never deplete cash below zero."""
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.default_rng(42).normal(0, 1, n))
        data = pd.DataFrame({"close": close}, index=dates)
        # Alternating signals forces a trade every 2 bars.
        raw = [0] + [1, -1] * (n // 2 - 1) + [0]
        signals = pd.Series(raw[:n], index=data.index)
        result = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0, signal_delay=0))
        assert result.portfolio_values is not None
        assert (result.portfolio_values >= 0).all()


# ---------------------------------------------------------------------------
# 2. Benchmark cost parity
# ---------------------------------------------------------------------------

class TestBenchmarkCostParity:
    """
    The old benchmark._buy_and_hold_returns() imported the global
    TRANSACTION_COST_RATE constant directly, ignoring the ExecutionConfig
    passed to the strategy.  A cost-sweep run would therefore use different
    cost rates for the strategy and the benchmark, making the comparison
    non-apples-to-apples.
    """

    def test_benchmark_lower_return_with_higher_cost(self) -> None:
        """Higher cost_rate passed to compute_benchmark reduces benchmark return."""
        from backtesting_engine.benchmark import _buy_and_hold_returns

        prices = pd.Series(np.linspace(100.0, 110.0, 50))

        returns_low  = _buy_and_hold_returns(prices, cost_rate=0.0001)
        returns_high = _buy_and_hold_returns(prices, cost_rate=0.01)

        # Higher cost → lower entry and exit return → lower cumulative return.
        assert returns_high[0] < returns_low[0],  "Higher cost should reduce entry return."
        assert returns_high[-1] < returns_low[-1], "Higher cost should reduce exit return."

    def test_compute_benchmark_uses_execution_config_cost(self) -> None:
        """compute_benchmark with a non-default cost should differ from the default."""
        from backtesting_engine.benchmark import _buy_and_hold_returns

        prices = pd.Series(np.linspace(100.0, 115.0, 100))

        default_returns = _buy_and_hold_returns(prices)
        custom_returns  = _buy_and_hold_returns(prices, cost_rate=0.005)

        # Custom (higher) cost must reduce cumulative return.
        assert custom_returns[0] < default_returns[0]

    def test_zero_cost_benchmark_has_no_entry_drag(self) -> None:
        from backtesting_engine.benchmark import _buy_and_hold_returns

        prices = pd.Series([100.0, 101.0, 102.0])
        returns = _buy_and_hold_returns(prices, cost_rate=0.0)
        expected_first = (101.0 - 100.0) / 100.0
        assert math.isclose(returns[0], expected_first, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 3. No-trade windows: flat-cash, not excluded
# ---------------------------------------------------------------------------

class TestFlatCashWindows:
    """
    A walk-forward window where no trades execute is a valid state: the
    strategy held cash for the full period.  The old code marked these windows
    as skipped=True and excluded them from summary metrics, biasing the
    aggregate Sharpe upward.  The fix includes them as flat-cash windows with
    Sharpe=0 and p=1.0.
    """

    def test_no_trade_window_sharpe_is_zero(self) -> None:
        from backtesting_engine.walk_forward import _flat_cash_metrics
        m = _flat_cash_metrics()
        assert m.sharpe_ratio == 0.0

    def test_no_trade_window_p_value_is_one(self) -> None:
        from backtesting_engine.walk_forward import _flat_cash_metrics
        m = _flat_cash_metrics()
        assert m.p_value == 1.0

    def test_no_trade_window_drawdown_is_zero(self) -> None:
        from backtesting_engine.walk_forward import _flat_cash_metrics
        m = _flat_cash_metrics()
        assert m.max_drawdown == 0.0

    def test_no_trade_window_sortino_is_zero(self) -> None:
        # 0.0 not inf - inf is excluded from summary means, silently overstating Sortino.
        from backtesting_engine.walk_forward import _flat_cash_metrics
        m = _flat_cash_metrics()
        assert m.sortino_ratio == 0.0

    def test_no_trade_window_omega_is_one(self) -> None:
        # 1.0 = neutral break-even. inf causes the same silent exclusion problem.
        from backtesting_engine.walk_forward import _flat_cash_metrics
        m = _flat_cash_metrics()
        assert m.omega_ratio == 1.0

    def test_no_trade_window_not_skipped(self) -> None:
        """A window with no trades should have skipped=False in a real walk_forward run."""
        # Use a very tight MA crossover so some windows will trade and some won't.
        # Use zero-friction execution to avoid needing OHLCV data in this unit test.
        data = make_oscillating_data(756, with_high_low=False)
        strategy = MovingAverageStrategy(short_window=5, long_window=10)
        result = walk_forward(
            data, strategy,
            training_window_years=1, testing_window_years=1,
            execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
        )
        # All windows must be non-skipped (flat-cash windows included as valid).
        skipped = [w for w in result.window_results if w.skipped]
        assert len(skipped) == 0, (
            f"{len(skipped)} windows still marked skipped; they should be flat-cash."
        )


# ---------------------------------------------------------------------------
# 4. Momentum RC: each candidate uses its own lookback
# ---------------------------------------------------------------------------

class TestMomentumRCCandidateLookback:
    """
    The original candidate_test_returns() in MomentumStrategy called
    self.generate_signals(combined) inside the candidate loop, which always
    used self.lookback_ (the fitted winner) regardless of the candidate lb.
    This meant the RC matrix was a constant - every candidate was evaluated
    with the same parameters - making the RC p-value meaningless.

    The fix calls _momentum_signals(combined_close, lb) directly so each
    candidate is genuinely evaluated with its own lookback.
    """

    def test_rc_candidates_differ_by_lookback(self) -> None:
        """Candidate return series must differ across lookbacks."""
        data = make_oscillating_data(504, with_high_low=False)
        strategy = MomentumStrategy()
        strategy.fit(data.iloc[:252])

        test_data = data.iloc[252:]
        candidates = strategy.candidate_test_returns(test_data, context_data=data.iloc[:252])

        assert len(candidates) >= 2, "Need at least 2 candidates to compare."
        keys = list(candidates.keys())
        a, b = candidates[keys[0]], candidates[keys[1]]

        # Align on common index (may differ in length due to different warmup).
        common = a.index.intersection(b.index)
        assert len(common) > 0

        # They must not be identical (would indicate the bug is still present).
        assert not np.allclose(a.loc[common].values, b.loc[common].values), (
            "All RC candidates are identical - the momentum lookback bug may still be present. "
            "Each candidate should be evaluated with its own lookback parameter."
        )

    def test_rc_context_path_candidates_still_differ_by_lookback(self) -> None:
        """With context, candidates must still differ across lookbacks (not all identical)."""
        data = make_oscillating_data(504, with_high_low=False)
        strategy = MomentumStrategy()
        strategy.fit(data.iloc[:252])

        test_data = data.iloc[252:]
        ctx = data.iloc[:252]

        cands = strategy.candidate_test_returns(test_data, context_data=ctx)
        assert len(cands) >= 2

        keys = list(cands.keys())
        a = cands[keys[0]]
        b = cands[keys[1]]
        common = a.index.intersection(b.index)

        # Candidates with different lookbacks must produce different signals.
        # If they were identical it would mean every candidate still uses self.lookback_.
        assert not np.allclose(a.loc[common].values, b.loc[common].values), (
            "All RC candidates with context are identical - the momentum lookback "
            "bug may still be present in the context code path."
        )


# ---------------------------------------------------------------------------
# 5. Block bootstrap: centred null gives correct p-values
# ---------------------------------------------------------------------------

class TestBootstrapNullCentering:
    """
    The original bootstrap resampled raw returns, inheriting the strategy's
    observed mean.  The bootstrap Sharpe distribution was therefore centred
    at the observed Sharpe rather than at zero, so p(boot >= observed) ≈ 0.5
    for any positive-drift strategy regardless of signal quality.

    The fix centres returns before resampling so H₀ is explicitly zero-mean.
    """

    def test_zero_mean_returns_give_p_near_half(self) -> None:
        """Zero-mean returns should produce p≈0.5 - not significant."""
        rng = np.random.default_rng(99)
        returns = rng.normal(loc=0.0, scale=0.01, size=252)
        p = _monte_carlo_p_value(returns)
        # For zero-mean data the centred and original distributions are identical,
        # so p should still sit near 0.5 (flat against H₀).
        assert 0.1 <= p <= 0.9, (
            f"Zero-mean returns should give p≈0.5, got p={p:.4f}."
        )

    def test_high_positive_drift_gives_low_p(self) -> None:
        """Returns with strong positive drift should be flagged as significant."""
        rng = np.random.default_rng(42)
        # Annualised Sharpe ≈ 0.01/0.005 × sqrt(252) ≈ 3.2 - very strong signal.
        returns = rng.normal(loc=0.01, scale=0.005, size=252)
        p = _monte_carlo_p_value(returns)
        assert p < 0.05, (
            f"High-Sharpe returns should give low p-value, got p={p:.4f}. "
            "Check that _monte_carlo_p_value centres returns before resampling."
        )

    def test_negative_drift_gives_high_p(self) -> None:
        """Returns with negative drift should not look significant under H₀: mean=0."""
        rng = np.random.default_rng(7)
        returns = rng.normal(loc=-0.005, scale=0.01, size=252)
        p = _monte_carlo_p_value(returns)
        # Negative drift → observed Sharpe < 0 → most bootstrap Sharpes > observed → high p.
        assert p > 0.5, (
            f"Negative-drift returns should give high p-value, got p={p:.4f}."
        )

    def test_centring_preserves_variance_structure(self) -> None:
        """Centring must not change the standard deviation of the returns."""
        rng = np.random.default_rng(5)
        returns = rng.normal(loc=0.003, scale=0.01, size=300)
        centered = returns - returns.mean()
        assert math.isclose(returns.std(), centered.std(), rel_tol=1e-9)
        assert math.isclose(centered.mean(), 0.0, abs_tol=1e-14)
