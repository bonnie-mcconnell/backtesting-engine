"""
Tests added in the final fix pass (v0.6.1 → v0.6.2).

Covers:
  1. WindowResult.skipped=True fires DeprecationWarning.
  2. MOMENTUM_LOOKBACKS in config drives _LOOKBACK_GRID in momentum.py.
  3. build_candidate_return_matrix accepts both tuple and int keys (not just tuples).
  4. _extract_active_params no longer exists in walk_forward (was dead code).
  5. Each strategy exposes format_params() - no strategy-specific knowledge in main.py.
"""

import warnings

import numpy as np
import pandas as pd


class TestWindowResultDeprecationWarning:
    """WindowResult.skipped=True must fire a DeprecationWarning."""

    def _make_window_result(self, skipped: bool) -> object:
        from backtesting_engine.models import MetricsResult, SimulationResult, WindowResult
        sim = SimulationResult(trades=[])
        m = MetricsResult(
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
            calmar_ratio=0.0, omega_ratio=0.0, p_value=0.0,
        )
        return WindowResult(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-12-31"),
            test_start=pd.Timestamp("2021-01-01"),
            test_end=pd.Timestamp("2021-12-31"),
            simulation_result=sim,
            metrics_result=m,
            skipped=skipped,
        )

    def test_skipped_false_no_warning(self) -> None:
        """Normal construction (skipped=False) must not emit any warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_window_result(skipped=False)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0

    def test_skipped_true_fires_deprecation_warning(self) -> None:
        """Explicit skipped=True must emit exactly one DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_window_result(skipped=True)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1

    def test_skipped_true_warning_mentions_flat_cash(self) -> None:
        """Warning message should explain the replacement concept (flat-cash)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_window_result(skipped=True)
        msg = str(w[0].message)
        assert "flat-cash" in msg.lower() or "flat cash" in msg.lower()

    def test_skipped_true_warning_is_deprecation_not_runtime(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_window_result(skipped=True)
        assert issubclass(w[0].category, DeprecationWarning)
        assert not issubclass(w[0].category, RuntimeWarning)


class TestMomentumLookbackConfig:
    """MOMENTUM_LOOKBACKS in config.py must drive the momentum strategy grid."""

    def test_config_lookbacks_match_strategy_grid(self) -> None:
        from backtesting_engine.config import MOMENTUM_LOOKBACKS
        from backtesting_engine.strategy.momentum import _LOOKBACK_GRID
        assert MOMENTUM_LOOKBACKS == _LOOKBACK_GRID, (
            "Momentum strategy _LOOKBACK_GRID has diverged from config.MOMENTUM_LOOKBACKS. "
            "Edit config.py only - never edit the grid directly in momentum.py."
        )

    def test_config_lookbacks_are_sorted_ascending(self) -> None:
        from backtesting_engine.config import MOMENTUM_LOOKBACKS
        assert MOMENTUM_LOOKBACKS == sorted(MOMENTUM_LOOKBACKS), (
            "MOMENTUM_LOOKBACKS should be in ascending order for deterministic candidate key ordering."
        )

    def test_config_lookbacks_all_positive(self) -> None:
        from backtesting_engine.config import MOMENTUM_LOOKBACKS
        for lb in MOMENTUM_LOOKBACKS:
            assert lb > 0, f"Lookback {lb} must be positive."

    def test_config_lookbacks_exported_from_config_module(self) -> None:
        """Verify the constant is accessible from the config module."""
        import importlib
        config = importlib.import_module("backtesting_engine.config")
        assert hasattr(config, "MOMENTUM_LOOKBACKS")
        assert isinstance(config.MOMENTUM_LOOKBACKS, list)

    def test_strategy_uses_config_lookbacks_after_fit(self) -> None:
        """After fit(), _all_lookbacks_ must equal MOMENTUM_LOOKBACKS."""
        import pandas as pd

        from backtesting_engine.config import MOMENTUM_LOOKBACKS
        from backtesting_engine.strategy.momentum import MomentumStrategy
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        close = pd.Series([100 + i for i in range(300)], index=dates, dtype=float)
        data = pd.DataFrame({"close": close})
        strategy = MomentumStrategy()
        strategy.fit(data)
        assert strategy._all_lookbacks_ == MOMENTUM_LOOKBACKS


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


class TestExtractActiveParamsRemoved:
    """_extract_active_params was a dead one-liner wrapper that has been removed.

    This test ensures it is gone and the orchestrator calls strategy.active_params() directly.
    Prevents it from being accidentally re-added.
    """

    def test_extract_active_params_not_in_walk_forward_module(self) -> None:
        import inspect

        import backtesting_engine.walk_forward as wf
        members = [name for name, _ in inspect.getmembers(wf)]
        assert "_extract_active_params" not in members, (
            "_extract_active_params was a dead one-liner wrapper that should not exist. "
            "The orchestrator calls strategy.active_params() directly."
        )


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

    def test_walk_forward_stores_formatted_params_on_window(self) -> None:
        from helpers import make_oscillating_data

        from backtesting_engine.execution import ExecutionConfig
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        from backtesting_engine.walk_forward import walk_forward

        data = make_oscillating_data(504, with_high_low=False)
        result = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
        )
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
        ).read_text()
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

    def test_walk_forward_stores_spec_on_window_result(self) -> None:
        from helpers import make_oscillating_data

        from backtesting_engine.execution import ExecutionConfig
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        from backtesting_engine.walk_forward import walk_forward

        data = make_oscillating_data(504, with_high_low=False)
        result = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
        )
        for w in result.valid_windows:
            # MA strategy should have 2-entry spec on every window
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
        ).read_text()
        for key in ("short_window", "snr"):
            for q in ('"', "'"):
                assert f"{q}{key}{q}" not in dash_src, (
                    f"dashboard.py contains {q}{key}{q} - strategy-specific knowledge "
                    f"that belongs in the strategy's param_evolution_spec(), "
                    "not the dashboard."
                )
