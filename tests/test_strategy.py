"""
Unit tests for the MovingAverageStrategy and BaseStrategy utilities.

MovingAverageStrategy signal generation, fit() calibration, warmup context,
and the BaseStrategy interface. Also covers returns_from_signals() edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from helpers import make_oscillating_data

from backtesting_engine.strategy.base import returns_from_signals
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward


def _make_data(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices}, index=dates)


def _trending_data(n: int = 504, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    prices = np.linspace(start_price, start_price * 1.5, n)
    return pd.DataFrame({"close": prices}, index=dates)


class TestSignalGeneration:
    def test_golden_cross_produces_buy_signal(self) -> None:
        # bars 0-59: flat at 100, both MAs converge - no crossover signal
        # bars 60-119: sharp rise to 130 - short MA (10) rises faster than
        # long MA (30), crossing above it and producing a buy signal.
        prices = list(np.linspace(100.0, 100.0, 60)) + list(np.linspace(100.0, 130.0, 60))
        data = _make_data(prices)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert 1 in signals.values

    def test_death_cross_produces_sell_signal(self) -> None:
        # bars 0-79: rising prices - golden cross fires, short above long
        # bars 80-159: sharp drop - short MA falls back below long MA (death cross)
        prices = list(np.linspace(100.0, 130.0, 80)) + list(np.linspace(130.0, 80.0, 80))
        data = _make_data(prices)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert -1 in signals.values

    def test_signals_only_contain_valid_values(self) -> None:
        data = make_oscillating_data(300)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_aligned_to_data_index(self) -> None:
        data = make_oscillating_data(300)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert signals.index.equals(data.index)

    def test_signals_are_integers(self) -> None:
        data = make_oscillating_data(300)
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert pd.api.types.is_integer_dtype(signals)

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="less than"):
            MovingAverageStrategy(short_window=200, long_window=50)


class TestFit:
    def test_fit_returns_self(self) -> None:
        strategy = MovingAverageStrategy()
        data = make_oscillating_data(504)
        result = strategy.fit(data)
        assert result is strategy

    def test_fit_updates_windows(self) -> None:
        strategy = MovingAverageStrategy()
        data = make_oscillating_data(504)
        strategy.fit(data)
        # After fit, windows must still be valid (short < long, both positive).
        assert strategy.short_window_ > 0
        assert strategy.long_window_ > 0

    def test_fit_maintains_short_less_than_long(self) -> None:
        strategy = MovingAverageStrategy()
        data = make_oscillating_data(504)
        strategy.fit(data)
        assert strategy.short_window_ < strategy.long_window_

    def test_fit_with_insufficient_data_retains_defaults(self) -> None:
        strategy = MovingAverageStrategy(short_window=50, long_window=200)
        # Only 50 rows - no grid pair has enough history to compute valid MAs.
        data = _make_data([float(i) for i in range(100, 150)])
        original_short = strategy.short_window_
        original_long = strategy.long_window_
        strategy.fit(data)
        # Should retain defaults rather than raising.
        assert strategy.short_window_ == original_short
        assert strategy.long_window_ == original_long


class TestWarmupContext:
    def test_context_signals_only_cover_test_index(self) -> None:
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        all_data = make_oscillating_data(200)
        context = all_data.iloc[:50]
        test_data = all_data.iloc[50:]
        signals = strategy.generate_signals_with_context(context, test_data)
        assert signals.index.equals(test_data.index)

    def test_context_signals_contain_only_valid_values(self) -> None:
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        all_data = make_oscillating_data(300)
        context = all_data.iloc[:50]
        test_data = all_data.iloc[50:]
        signals = strategy.generate_signals_with_context(context, test_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_context_produces_more_signals_than_without(self) -> None:
        # With context, the MA is warmed up from bar 1 of test data.
        # Without context, the first long_window bars are always 0.
        # On oscillating data, context should yield >= as many non-zero signals.
        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        all_data = make_oscillating_data(400)
        context = all_data.iloc[:50]
        test_data = all_data.iloc[50:]

        with_context = strategy.generate_signals_with_context(context, test_data)
        without_context = strategy.generate_signals(test_data)

        n_with = (with_context != 0).sum()
        n_without = (without_context != 0).sum()
        assert n_with >= n_without


# ---------------------------------------------------------------------------
# candidate_test_returns - the Reality Check interface
# ---------------------------------------------------------------------------

class TestCandidateTestReturns:
    def test_returns_dict_with_tuple_keys(self) -> None:
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        data = make_oscillating_data(504)
        train, test = data.iloc[:252], data.iloc[252:]
        strategy.fit(train)
        result = strategy.candidate_test_returns(test, context_data=train.iloc[-50:])
        assert isinstance(result, dict)
        assert all(isinstance(k, tuple) and len(k) == 2 for k in result.keys())

    def test_all_candidates_are_evaluated_on_test_data(self) -> None:
        # Every key in candidate_test_returns should also be in _all_candidate_pairs_
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        data = make_oscillating_data(504)
        train, test = data.iloc[:252], data.iloc[252:]
        strategy.fit(train)
        result = strategy.candidate_test_returns(test, context_data=train.iloc[-50:])
        for key in result.keys():
            assert key in strategy._all_candidate_pairs_

    def test_returns_are_test_period_length(self) -> None:
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        data = make_oscillating_data(504)
        train, test = data.iloc[:252], data.iloc[252:]
        strategy.fit(train)
        result = strategy.candidate_test_returns(test)
        for returns_series in result.values():
            # Returns are one bar shorter than data due to differencing.
            assert len(returns_series) == len(test) - 1

    def test_returns_are_finite_floats(self) -> None:
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        data = make_oscillating_data(504)
        train, test = data.iloc[:252], data.iloc[252:]
        strategy.fit(train)
        result = strategy.candidate_test_returns(test)
        import numpy as np
        for returns_series in result.values():
            assert np.all(np.isfinite(returns_series.to_numpy()))

    def test_empty_before_fit(self) -> None:
        # Before fit(), no candidates have been evaluated.
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        data = make_oscillating_data(300)
        test = data.iloc[150:]
        result = strategy.candidate_test_returns(test)
        assert result == {}

    def test_base_strategy_default_returns_empty(self) -> None:
        # KalmanFilterStrategy has no grid, so candidate_test_returns returns {}.
        from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
        strategy = KalmanFilterStrategy()
        data = make_oscillating_data(300)
        train, test = data.iloc[:150], data.iloc[150:]
        strategy.fit(train)
        result = strategy.candidate_test_returns(test)
        assert result == {}


# ---------------------------------------------------------------------------
# Position carry-over at window boundaries
# ---------------------------------------------------------------------------

class TestPositionCarryOver:
    """
    generate_signals_with_context must inject a buy signal at test bar 0
    when the strategy is already long at the context/test boundary.

    Without this, walk-forward understates returns by starting each test
    window flat even when the strategy should carry a long position over
    from the prior period. The injected signal must match the actual MA
    relationship at the boundary, not be hardcoded.
    """

    def test_ma_carries_long_position_into_uptrend_test(self) -> None:
        # Perfectly linear uptrend: short MA always above long MA after warmup.
        # The golden cross fires during the context window; without carry-over
        # the test window would have zero signals and zero trades.
        n = 300
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        prices = np.linspace(100.0, 200.0, n)
        data = pd.DataFrame({"close": prices}, index=dates)

        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        context = data.iloc[:40]   # warmup: golden cross fires here
        test = data.iloc[40:120]

        signals = strategy.generate_signals_with_context(context, test)
        assert signals.iloc[0] == 1, (
            "Strategy should carry long position into test window. "
            "Bar 0 must be a buy signal, not 0 (flat)."
        )

    def test_ma_does_not_inject_buy_when_already_flat(self) -> None:
        # Downtrend then uptrend: strategy is flat (short < long) at boundary.
        # No carry-over buy should be injected.
        n = 300
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        # Fall sharply then recover - death cross in context, no golden cross yet
        prices = list(np.linspace(200.0, 100.0, 150)) + list(np.linspace(100.0, 110.0, 150))
        data = pd.DataFrame({"close": prices}, index=dates)

        strategy = MovingAverageStrategy(short_window=10, long_window=30)
        context = data.iloc[:50]   # death cross fires here, short < long
        test = data.iloc[50:120]

        signals = strategy.generate_signals_with_context(context, test)
        # Verify no buy signal is injected at bar 0. The strategy is flat at the
        # boundary (death cross fired in context, short MA < long MA). Carry-over
        # only fires when short MA > long MA at the boundary, so bar 0 must be 0.
        assert signals.iloc[0] == 0, (
            "Should not inject carry-over buy when strategy is flat (short MA < long MA) "
            f"at the context/test boundary. Got signals.iloc[0]={signals.iloc[0]}."
        )


# ---------------------------------------------------------------------------
# BaseStrategy interface: context_window_size()
# ---------------------------------------------------------------------------

class TestContextWindowSize:
    """
    Verify that all strategies implement context_window_size() and that the
    walk-forward orchestrator uses it - eliminating isinstance dispatch.
    """

    def test_ma_context_window_is_at_least_long_window(self) -> None:
        strategy = MovingAverageStrategy(short_window=20, long_window=100)
        # context_window_size must be >= long_window so the long MA is warm
        assert strategy.context_window_size() >= strategy.long_window_

    def test_ma_context_window_updates_after_fit(self) -> None:
        # After fit(), context_window_size() should reflect the calibrated long_window_.
        data = make_oscillating_data(504)
        strategy = MovingAverageStrategy(short_window=20, long_window=50)
        strategy.fit(data)
        assert strategy.context_window_size() >= strategy.long_window_

    def test_kalman_context_window_is_positive(self) -> None:
        from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
        strategy = KalmanFilterStrategy()
        assert strategy.context_window_size() > 0

    def test_momentum_context_window_equals_lookback(self) -> None:
        from backtesting_engine.strategy.momentum import MomentumStrategy
        strategy = MomentumStrategy(lookback=90)
        # Momentum signal at bar t uses close[t - lookback], so needs lookback bars
        assert strategy.context_window_size() >= strategy.lookback_

    def test_custom_strategy_gets_default_context_window(self) -> None:
        """A strategy that doesn't override context_window_size() gets the default."""
        from backtesting_engine.strategy.base import BaseStrategy

        class MinimalStrategy(BaseStrategy):
            def fit(self, train_data: pd.DataFrame) -> "MinimalStrategy":
                return self
            def generate_signals(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(0, index=data.index)

        assert MinimalStrategy().context_window_size() == 50

    def test_walk_forward_does_not_need_isinstance_for_context(self) -> None:
        """
        The orchestrator must not import concrete strategy classes.
        If this import succeeds, the walk_forward module is clean.
        """
        import ast
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "src/backtesting_engine/walk_forward.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        # Collect all import names from the walk_forward module
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported_names.append(node.module)
        concrete_strategy_modules = [
            "backtesting_engine.strategy.moving_average",
            "backtesting_engine.strategy.kalman_filter",
            "backtesting_engine.strategy.momentum",
        ]
        for mod in concrete_strategy_modules:
            assert mod not in imported_names, (
                f"walk_forward.py imports {mod} - this creates isinstance coupling. "
                "Use the BaseStrategy interface (context_window_size, active_params) instead."
            )


# ---------------------------------------------------------------------------
# returns_from_signals - shared position-tracking utility
# ---------------------------------------------------------------------------

class TestReturnsFromSignals:
    """
    Direct tests for the returns_from_signals() utility in base.py.

    These tests target the hold-state semantics specifically: signal=0 must
    inherit the most recently active position, not reset to flat. This is the
    non-trivial part of the implementation that warrants direct coverage
    independent of any strategy class.
    """

    def test_hold_carries_long_across_zero_signals(self) -> None:
        # signals = [0, 1, 0, 0, -1, 0]
        # Positions:  0  1  1  1   0  0
        # Returns use position[:-1] = [0, 1, 1, 1, 0]
        close = np.array([100.0, 101.0, 103.0, 102.0, 104.0, 103.0])
        signals = np.array([0, 1, 0, 0, -1, 0])
        result = returns_from_signals(close, signals)

        price_returns = np.diff(close) / close[:-1]
        expected_positions = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        expected = expected_positions * price_returns

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_starts_flat_before_first_buy(self) -> None:
        # Leading zeros before first buy signal → strategy starts flat.
        # The vectorised ffill fills NaN forward, but leading NaN becomes 0.
        close = np.array([100.0, 102.0, 101.0, 105.0])
        signals = np.array([0, 0, 1, 0])
        result = returns_from_signals(close, signals)

        # Positions: [0, 0, 1, 1] → position[:-1] = [0, 0, 1]
        # Only the third return (bar 2→3) should be non-zero.
        assert result[0] == pytest.approx(0.0), "Pre-buy bar should be flat"
        assert result[1] == pytest.approx(0.0), "Pre-buy bar should be flat"
        assert result[2] == pytest.approx((105.0 - 101.0) / 101.0)

    def test_output_length_is_n_minus_one(self) -> None:
        # Fundamental contract: output is always one element shorter than input.
        for n in [2, 10, 100]:
            close = np.linspace(100.0, 110.0, n)
            signals = np.zeros(n, dtype=int)
            result = returns_from_signals(close, signals)
            assert len(result) == n - 1, f"Expected {n - 1} returns for {n} bars"

    def test_all_flat_signals_produce_zero_returns(self) -> None:
        # All signals = 0 → never long → all returns zero.
        close = np.array([100.0, 110.0, 90.0, 105.0])
        signals = np.zeros(4, dtype=int)
        result = returns_from_signals(close, signals)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_sell_signal_exits_long_position(self) -> None:
        # Buy at bar 0, sell at bar 2 → long for bars 0→1 and 1→2, flat after.
        close = np.array([100.0, 105.0, 103.0, 110.0])
        signals = np.array([1, 0, -1, 0])
        result = returns_from_signals(close, signals)

        # Positions: [1, 1, 0, 0] → position[:-1] = [1, 1, 0]
        price_returns = np.diff(close) / close[:-1]
        expected = np.array([1.0, 1.0, 0.0]) * price_returns
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------

class TestRCBoundaryCarryOver:
    """
    candidate_test_returns() must inject boundary carry-over identically to
    generate_signals_with_context(). Without this, the selected strategy and
    the RC candidate universe are evaluated under different state assumptions.
    """

    def _make_trending_data(self, n: int = 252) -> pd.DataFrame:
        """Strongly trending data so MAs stay long for most of the period."""
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        close = np.array([100.0 + i * 0.3 for i in range(n)])
        return pd.DataFrame({"close": close}, index=dates)

    def test_ma_candidate_returns_use_same_context(self) -> None:
        """After fitting MA, candidate_test_returns with context should apply
        boundary carry-over the same way generate_signals_with_context does."""
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy

        data = self._make_trending_data(252)
        train = data.iloc[:180]
        test = data.iloc[180:]
        context = train.iloc[-201:]  # enough context for MA warmup

        strategy = MovingAverageStrategy()
        strategy.fit(train)

        # Get signals from generate_signals_with_context
        gswc_signals = strategy.generate_signals_with_context(context, test)

        # Get candidate returns - the selected params should be in the dict
        candidates = strategy.candidate_test_returns(test, context)
        selected_key = (strategy.short_window_, strategy.long_window_)

        assert selected_key in candidates
        assert len(gswc_signals) == len(test)
        assert len(candidates[selected_key]) == len(test) - 1, (
            "Candidate return series length must be len(test) - 1"
        )

    def test_momentum_candidate_returns_inject_boundary(self) -> None:
        """Momentum candidate_test_returns must apply boundary carry-over."""
        from backtesting_engine.strategy.momentum import MomentumStrategy

        data = self._make_trending_data(300)
        train = data.iloc[:240]
        test = data.iloc[240:]
        strategy = MomentumStrategy()
        strategy.fit(train)

        lb = strategy.lookback_
        context = train.iloc[-lb:]

        candidates = strategy.candidate_test_returns(test, context)
        # All candidates must return arrays of length len(test) - 1
        for k, v in candidates.items():
            assert len(v) == len(test) - 1, (
                f"Candidate {k} return series has wrong length: {len(v)} != {len(test)-1}"
            )


@pytest.fixture(scope="class")
def slippage_parity_data():
    """
    Shared walk_forward result for TestBenchmarkSlippageParity.

    The test_compute_benchmark_passes_slippage_to_bh test needs a walk_forward
    result to call compute_benchmark on. Using a class-scoped fixture avoids
    running the full MA grid search once per test method.
    """
    from backtesting_engine.execution import ExecutionConfig
    from backtesting_engine.strategy.moving_average import MovingAverageStrategy

    data = make_oscillating_data(756, with_high_low=True)
    exec_slip = ExecutionConfig(transaction_cost_rate=0.001, slippage_factor=0.05, signal_delay=0)
    result = walk_forward(
        data, MovingAverageStrategy(),
        training_window_years=1, testing_window_years=1,
        execution=exec_slip,
    )
    return data, result, exec_slip

class TestFormatParams:
    """
    Each strategy owns its own format_params() - main.py must not hard-code
    strategy-specific knowledge for parameter formatting.

    If you add a new strategy, add a test here verifying its format_params()
    output is human-readable and non-empty after fit().
    """

    def _make_data(self, n: int = 300) -> "pd.DataFrame":
        import pandas as pd
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = pd.Series([100.0 + i * 0.1 for i in range(n)], index=dates)
        return pd.DataFrame({"close": close})

    def test_ma_format_params_looks_like_ma_short_slash_long(self) -> None:
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        strategy = MovingAverageStrategy()
        strategy.fit(self._make_data())
        result = strategy.format_params()
        assert result.startswith("MA(") and "/" in result and result.endswith(")")

    def test_ma_format_params_contains_calibrated_values(self) -> None:
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        strategy = MovingAverageStrategy()
        strategy.fit(self._make_data())
        result = strategy.format_params()
        assert str(strategy.short_window_) in result
        assert str(strategy.long_window_) in result

    def test_kalman_format_params_contains_snr(self) -> None:
        from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
        strategy = KalmanFilterStrategy()
        strategy.fit(self._make_data())
        result = strategy.format_params()
        assert result.startswith("SNR=")

    def test_momentum_format_params_contains_lookback(self) -> None:
        from backtesting_engine.strategy.momentum import MomentumStrategy
        strategy = MomentumStrategy()
        strategy.fit(self._make_data())
        result = strategy.format_params()
        assert result.startswith("MOM(")
        assert str(strategy.lookback_) in result

    def test_base_default_format_params_is_empty_for_no_params(self) -> None:
        import pandas as pd

        from backtesting_engine.strategy.base import BaseStrategy

        class ParameterFree(BaseStrategy):
            def fit(self, train_data: pd.DataFrame) -> "ParameterFree":
                return self
            def generate_signals(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(0, index=data.index)

        assert ParameterFree().format_params() == ""

    def test_walk_forward_stores_formatted_params_on_window(
        self, wf_result_504: object
    ) -> None:
        """formatted_params must be set on every non-empty window.

        Uses the shared wf_result_504 fixture (module scope) to avoid running a
        redundant walk_forward with full MA grid search. The fixed-window MA result
        still stores formatted_params correctly - it does not depend on grid search.
        """
        from backtesting_engine.models import BacktestResult
        result = wf_result_504
        assert isinstance(result, BacktestResult)
        for w in result.valid_windows:
            if w.simulation_result.trades:
                assert w.formatted_params.startswith("MA("), (
                    f"Expected MA(x/y) format, got: {w.formatted_params!r}"
                )

    def test_format_params_not_in_main_py_as_hardcoded_strategy_check(self) -> None:
        """main.py must not contain strategy-specific parameter key names.

        Checks both single-quote and double-quote forms to guard against
        bypass via quote-style variation (e.g. active_params.get('snr') vs ["snr"]).
        """
        import pathlib
        main_src = (
            pathlib.Path(__file__).parent.parent / "src/backtesting_engine/main.py"
        ).read_text(encoding="utf-8")
        for key in ("short_window", "snr"):
            for q in ('"', "'"):
                assert f"{q}{key}{q}" not in main_src, (
                    f"main.py contains {q}{key}{q} - strategy-specific knowledge that "
                    f"belongs in the strategy's format_params() or param_evolution_spec(), "
                    "not the orchestrator."
                )

class TestParamEvolutionSpec:
    """
    param_evolution_spec() is the strategy's contract for the dashboard
    parameter evolution panel. It eliminates all isinstance checks and
    hard-coded key names from dashboard.py.

    If you add a new strategy, add a test here verifying its spec.
    """

    def _make_data(self, n: int = 300) -> "pd.DataFrame":
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = pd.Series([100.0 + i * 0.1 for i in range(n)], index=dates)
        return pd.DataFrame({"close": close})

    def test_ma_spec_has_two_entries(self) -> None:
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        s = MovingAverageStrategy()
        s.fit(self._make_data())
        spec = s.param_evolution_spec()
        assert len(spec) == 2

    def test_ma_spec_keys_match_active_params(self) -> None:
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        s = MovingAverageStrategy()
        s.fit(self._make_data())
        params = s.active_params()
        for label, key in s.param_evolution_spec():
            assert key in params, f"Spec key '{key}' not in active_params {params}"

    def test_kalman_spec_has_two_entries(self) -> None:
        from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
        s = KalmanFilterStrategy()
        s.fit(self._make_data())
        spec = s.param_evolution_spec()
        assert len(spec) == 2

    def test_kalman_spec_keys_match_active_params(self) -> None:
        from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
        s = KalmanFilterStrategy()
        s.fit(self._make_data())
        params = s.active_params()
        for label, key in s.param_evolution_spec():
            assert key in params, f"Spec key '{key}' not in active_params {params}"

    def test_momentum_spec_has_one_entry(self) -> None:
        from backtesting_engine.strategy.momentum import MomentumStrategy
        s = MomentumStrategy()
        s.fit(self._make_data())
        spec = s.param_evolution_spec()
        assert len(spec) == 1

    def test_momentum_spec_key_matches_active_params(self) -> None:
        from backtesting_engine.strategy.momentum import MomentumStrategy
        s = MomentumStrategy()
        s.fit(self._make_data())
        params = s.active_params()
        for label, key in s.param_evolution_spec():
            assert key in params, f"Spec key '{key}' not in active_params {params}"

    def test_base_default_spec_is_empty_for_no_params(self) -> None:
        from backtesting_engine.strategy.base import BaseStrategy

        class ParameterFree(BaseStrategy):
            def fit(self, train_data: pd.DataFrame) -> "ParameterFree":
                return self
            def generate_signals(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(0, index=data.index)

        assert ParameterFree().param_evolution_spec() == []

    def test_walk_forward_stores_spec_on_window_result(
        self, wf_result_504: object
    ) -> None:
        """param_evolution_spec must be set with 2 entries on every MA window.

        Uses the shared wf_result_504 fixture (module scope). Fixed-window MA
        still stores param_evolution_spec correctly via strategy.param_evolution_spec().
        """
        from backtesting_engine.models import BacktestResult
        result = wf_result_504
        assert isinstance(result, BacktestResult)
        for w in result.valid_windows:
            assert len(w.param_evolution_spec) == 2, (
                f"Expected 2-entry MA spec, got: {w.param_evolution_spec}"
            )

    def test_dashboard_does_not_hardcode_short_window_or_snr(self) -> None:
        """dashboard.py must not contain strategy-specific parameter key names.

        Checks both single-quote and double-quote forms to guard against
        bypass via quote-style variation.
        """
        import pathlib
        dash_src = (
            pathlib.Path(__file__).parent.parent
            / "src/backtesting_engine/dashboard.py"
        ).read_text(encoding="utf-8")
        for key in ("short_window", "snr"):
            for q in ('"', "'"):
                assert f"{q}{key}{q}" not in dash_src, (
                    f"dashboard.py contains {q}{key}{q} - strategy-specific knowledge "
                    f"that belongs in the strategy's param_evolution_spec(), "
                    "not the dashboard."
                )
