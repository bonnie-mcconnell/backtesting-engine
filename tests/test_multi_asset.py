"""
Tests for cross-asset walk-forward validation (multi_asset.py).

Output structure, graceful skipping of tickers that fail data loading,
comparison table rendering, and CLI argument parsing.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import backtesting_engine.multi_asset as _ma_module
from backtesting_engine.benchmark import BenchmarkResult
from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.models import (
    BacktestResult,
    MetricsResult,
    SimulationResult,
    WindowResult,
)
from backtesting_engine.multi_asset import (
    _STRATEGY_MAP,
    _min_rows,
    _print_comparison_table,
    run_multi_asset,
)
from backtesting_engine.multi_asset import (
    _parse_args as _multi_parse_args,
)


def _make_ohlc(n: int = 756, seed: int = 0) -> pd.DataFrame:
    """Oscillating OHLC data suitable for walk_forward (needs close, high, low)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-01", periods=n, freq="B")
    t = np.linspace(0, 20 * np.pi, n)
    close = 100.0 + 20.0 * np.sin(t) + 0.05 * np.arange(n)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=dates)

def _mock_load_data(ticker: str, start: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Return synthetic data instead of hitting the network."""
    seed = hash(ticker) % 100
    return _make_ohlc(n=756, seed=seed)

class TestRunMultiAsset:
    """
    Tests for run_multi_asset() with mocked data loading.

    All tests mock load_data to avoid network access.  The walk_forward
    runs on synthetic oscillating data that is large enough for 1+1yr windows.
    """

    def test_returns_dict_with_ticker_keys(self, tmp_path: Path) -> None:
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY", "QQQ"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
            )
        assert set(results.keys()) == {"SPY", "QQQ"}, (
            "run_multi_asset must return a result for each successfully processed ticker"
        )

    def test_each_result_is_correct_types(self, tmp_path: Path) -> None:
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
            )
        result, benchmark = results["SPY"]
        assert isinstance(result, BacktestResult)
        assert isinstance(benchmark, BenchmarkResult)

    def test_failed_ticker_skipped_gracefully(self, tmp_path: Path) -> None:
        """A ticker that raises on data load must be omitted, not crash the run."""
        def _failing_load(ticker: str, *args: object, **kwargs: object) -> pd.DataFrame:
            if ticker == "FAIL":
                raise ValueError("Synthetic download error")
            return _mock_load_data(ticker, "", None)

        with patch("backtesting_engine.multi_asset.load_data", side_effect=_failing_load), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY", "FAIL"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
            )
        assert "SPY" in results, "Successful ticker must be present"
        assert "FAIL" not in results, "Failed ticker must be omitted, not crash"

    def test_empty_tickers_returns_empty_dict(self, tmp_path: Path) -> None:
        results = run_multi_asset(
            tickers=[],
            start="2005-01-01",
            end="2020-12-31",
            execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
            train_years=1,
            test_years=1,
            bootstrap_seed=42,
            output_dir=tmp_path,
        )
        assert results == {}

    def test_result_metrics_are_finite(self, tmp_path: Path) -> None:
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
            )
        result, _ = results["SPY"]
        m = result.summary_metrics
        assert math.isfinite(m.sharpe_ratio), "Sharpe must be finite for valid run"
        assert 0.0 <= m.combined_p_value <= 1.0, "Fisher p must be in [0, 1]"

class TestMinRows:
    """
    _min_rows(train_years, test_years) must reflect the actual requested
    window sizes, not a hardcoded default.

    Found via end-to-end CLI testing (not unit tests): every existing test in
    TestRunMultiAsset patches validate_data into a no-op, so none of them ever
    exercised the real validation logic this class tests directly. The
    original bug - a module-level _MIN_ROWS constant computed once from the
    default TRAINING_WINDOW_YEARS/TESTING_WINDOW_YEARS - had two failure
    modes: `backtesting-multi --train-years 1 --test-years 1` rejected data
    that was actually long enough (504 rows needed, but it demanded 1008),
    and `--train-years 5 --test-years 2` accepted data that was too short
    (1764 rows needed, but it only checked for 1008), pushing the failure
    into a less specific error message deeper inside walk_forward().
    """

    def test_min_rows_scales_with_requested_years(self) -> None:
        from backtesting_engine.config import ANNUALISATION_FACTOR
        assert _min_rows(1, 1) == (1 + 1) * ANNUALISATION_FACTOR
        assert _min_rows(5, 2) == (5 + 2) * ANNUALISATION_FACTOR
        assert _min_rows(1, 1) < _min_rows(5, 2)

    def test_min_rows_consistent_with_main_module(self) -> None:
        """multi_asset._min_rows and main._min_rows must agree for all inputs.

        Both modules compute the same formula. If they diverge, a ticker that
        passes validation in one CLI entry point fails in the other for the same
        parameters, which would be silently wrong in cross-asset runs.
        """
        from backtesting_engine.main import _min_rows as main_min_rows
        for train in (1, 2, 3, 5):
            for test in (1, 2):
                assert _min_rows(train, test) == main_min_rows(train, test), (
                    f"_min_rows({train}, {test}): "
                    f"multi_asset={_min_rows(train, test)}, "
                    f"main={main_min_rows(train, test)}"
                )

    def test_short_custom_window_accepts_data_too_short_for_default(
        self, tmp_path: Path
    ) -> None:
        """756 rows is enough for 1yr/1yr (needs 504) but not the default
        3yr/1yr (needs 1008). Uses the REAL validate_data, not mocked away."""
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data):
            results = run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
            )
        assert "SPY" in results, (
            "756-row data should pass validation for explicitly requested "
            "1yr/1yr windows (needs 704 rows), even though it's short of "
            "the unrelated 3yr/1yr default (1008 rows)."
        )

    def test_long_custom_window_rejects_data_long_enough_for_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """1300 rows clears the default 3yr/1yr requirement (1008) but is too
        short for an explicitly requested 5yr/2yr (needs 1764). Must be
        rejected by validate_data itself, not by a less specific failure
        further inside walk_forward() - checking only that the ticker is
        absent from results isn't enough, since walk_forward's own internal
        validation would also reject 1300 rows for 5yr/2yr windows (just with
        a vaguer message), which would make this test pass even with the
        original bug. Check the actual warning text instead.
        """
        def _load_1300_rows(ticker: str, start: str, end_date: str | None = None, use_cache: bool = True) -> pd.DataFrame:
            return _make_ohlc(n=1300, seed=1)

        with patch("backtesting_engine.multi_asset.load_data", side_effect=_load_1300_rows):
            results = run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=5,
                test_years=2,
                bootstrap_seed=42,
                output_dir=tmp_path,
            )
        assert "SPY" not in results
        out = capsys.readouterr().out
        assert "1764" in out, (
            f"Expected validate_data's own message citing the correct "
            f"requirement (1764 rows for 5yr/2yr), got: {out!r}"
        )

class TestPrintComparisonTable:
    """
    Tests that _print_comparison_table runs without error on various inputs.
    We do not test stdout content - that is a rendering detail.  We test that
    the function handles edge cases without raising exceptions.
    """

    def _make_fake_results(
        self, tickers: list[str]
    ) -> dict[str, tuple[BacktestResult, BenchmarkResult]]:
        """Build minimal fake results without running walk_forward."""
        fake_results: dict[str, tuple[BacktestResult, BenchmarkResult]] = {}
        for i, ticker in enumerate(tickers):
            dates = pd.date_range("2010-01-01", periods=2, freq="B")
            pv = pd.Series([100000.0, 101000.0], index=dates)
            sim = SimulationResult(trades=[], portfolio_values=pv, message="")
            m = MetricsResult(
                sharpe_ratio=0.5 + i * 0.1,
                sortino_ratio=0.6,
                max_drawdown=-0.1,
                calmar_ratio=1.0,
                omega_ratio=1.1,
                p_value=0.3,
                combined_p_value=0.25,
                reality_check_p_value=float("nan"),
                trade_count=5,
            )
            w = WindowResult(
                train_start=dates[0], train_end=dates[0],
                test_start=dates[0], test_end=dates[1],
                simulation_result=sim, metrics_result=m,
            )
            br = BacktestResult(
                strategy_name="MovingAverageStrategy",
                window_results=[w],
                summary_metrics=m,
            )
            bm = BenchmarkResult(
                benchmark_sharpe=0.4,
                benchmark_sortino=0.5,
                benchmark_max_drawdown=-0.12,
                information_ratio=0.2,
                sharpe_diff_t_stat=1.1,
                sharpe_diff_p_value=0.3,
                strategy_beats_benchmark_fraction=0.5,
                per_window_benchmark_sharpes=[0.4],
            )
            fake_results[ticker] = (br, bm)
        return fake_results

    def test_print_does_not_raise_on_normal_input(self) -> None:
        results = self._make_fake_results(["SPY", "QQQ", "TLT"])
        _print_comparison_table(results)  # must not raise

    def test_print_does_not_raise_on_single_ticker(self) -> None:
        results = self._make_fake_results(["SPY"])
        _print_comparison_table(results)  # must not raise

    def test_print_does_not_raise_on_empty(self) -> None:
        _print_comparison_table({})  # must not raise

    def test_print_does_not_raise_with_nan_rc(self) -> None:
        """RC p = nan (Kalman-style, no grid) must not cause a format error."""
        results = self._make_fake_results(["TLT"])
        _print_comparison_table(results)  # must not raise

class TestMultiAssetModule:
    """Verify the module is importable as __main__ and has a valid CLI."""

    def test_module_importable(self) -> None:
        assert hasattr(_ma_module, "run_multi_asset")
        assert hasattr(_ma_module, "main")
        assert hasattr(_ma_module, "_parse_args")

    def test_help_does_not_crash(self) -> None:
        """--help should print and raise SystemExit(0)."""
        old_argv = sys.argv
        sys.argv = ["multi_asset", "--help"]
        with pytest.raises(SystemExit) as exc_info:
            _multi_parse_args()
        assert exc_info.value.code == 0
        sys.argv = old_argv

class TestStrategyParameter:
    """
    Tests for the --strategy flag: ma, kalman, momentum, all.

    All tests mock load_data to avoid network access. Walk-forward runs on
    synthetic oscillating data (same fixture as above).
    """

    def test_default_strategy_is_ma(self, tmp_path: Path) -> None:
        # No --strategy arg should still produce MA results, not error out.
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
            )
        assert "SPY" in results

    def test_strategy_ma_keys_by_ticker(self, tmp_path: Path) -> None:
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY", "QQQ"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
                strategy="ma",
            )
        # Single strategy: keys are plain tickers.
        assert set(results.keys()) == {"SPY", "QQQ"}

    def test_strategy_momentum_runs(self, tmp_path: Path) -> None:
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
                strategy="momentum",
            )
        assert "SPY" in results
        result, benchmark = results["SPY"]
        assert isinstance(result, BacktestResult)
        assert isinstance(benchmark, BenchmarkResult)

    def test_strategy_kalman_runs(self, tmp_path: Path) -> None:
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
                strategy="kalman",
            )
        assert "SPY" in results
        result, _ = results["SPY"]
        assert math.isfinite(result.summary_metrics.sharpe_ratio)

    def test_strategy_all_keys_by_ticker_colon_strategy(self, tmp_path: Path) -> None:
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
                strategy="all",
            )
        expected_keys = {"SPY:ma", "SPY:kalman", "SPY:momentum"}
        assert set(results.keys()) == expected_keys, (
            f"strategy='all' must produce keys {{ticker}}:{{strategy_short}}. "
            f"Got: {set(results.keys())}"
        )

    def test_strategy_all_result_count(self, tmp_path: Path) -> None:
        # Two tickers × three strategies = six result entries.
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY", "QQQ"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
                strategy="all",
            )
        assert len(results) == 6

    def test_strategy_all_results_are_correct_types(self, tmp_path: Path) -> None:
        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"):
            results = run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
                strategy="all",
            )
        for key, (result, benchmark) in results.items():
            assert isinstance(result, BacktestResult), f"{key}: expected BacktestResult"
            assert isinstance(benchmark, BenchmarkResult), f"{key}: expected BenchmarkResult"

    def test_invalid_strategy_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
                strategy="bad_strategy",
            )

    def test_strategy_map_contains_all_three(self) -> None:
        # Regression guard: _STRATEGY_MAP must contain exactly the three documented strategies.
        assert set(_STRATEGY_MAP.keys()) == {"ma", "kalman", "momentum"}

    def test_dashboard_filename_includes_strategy(self, tmp_path: Path) -> None:
        # Dashboards are mocked out (build_dashboard may fail without real data
        # in some test environments), but the filename is derived before the call.
        # Verify naming by checking what gets written when build_dashboard is patched.
        written_paths: list[str] = []

        def _capture_dash(result, path, **kwargs):  # type: ignore[no-untyped-def]
            written_paths.append(str(path))

        with patch("backtesting_engine.multi_asset.load_data", side_effect=_mock_load_data), \
             patch("backtesting_engine.multi_asset.validate_data"), \
             patch("backtesting_engine.multi_asset.build_dashboard", side_effect=_capture_dash):
            run_multi_asset(
                tickers=["SPY"],
                start="2005-01-01",
                end="2020-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
                strategy="momentum",
            )

        assert len(written_paths) == 1
        assert "momentum" in written_paths[0]
        assert "spy" in written_paths[0].lower()

    def test_comparison_table_with_strategy_label(self, capsys: pytest.CaptureFixture) -> None:
        # _print_comparison_table must render the strategy_label in the header.
        dates = pd.date_range("2010-01-01", periods=2, freq="B")
        pv = pd.Series([100000.0, 101000.0], index=dates)
        sim = SimulationResult(trades=[], portfolio_values=pv, message="")
        m = MetricsResult(
            sharpe_ratio=0.5, sortino_ratio=0.6, max_drawdown=-0.1,
            calmar_ratio=1.0, omega_ratio=1.1, p_value=0.3,
            combined_p_value=0.25, reality_check_p_value=float("nan"),
            trade_count=5,
        )
        w = WindowResult(
            train_start=dates[0], train_end=dates[0],
            test_start=dates[0], test_end=dates[1],
            simulation_result=sim, metrics_result=m,
        )
        br = BacktestResult(
            strategy_name="MomentumStrategy",
            window_results=[w],
            summary_metrics=m,
        )
        bm = BenchmarkResult(
            benchmark_sharpe=0.4, benchmark_sortino=0.5,
            benchmark_max_drawdown=-0.12, information_ratio=0.2,
            sharpe_diff_t_stat=1.1, sharpe_diff_p_value=0.3,
            strategy_beats_benchmark_fraction=0.5,
            per_window_benchmark_sharpes=[0.4],
        )
        _print_comparison_table({"SPY": (br, bm)}, strategy_label="Momentum")
        captured = capsys.readouterr()
        assert "Momentum" in captured.out

    def test_help_shows_strategy_choices(self) -> None:
        old_argv = sys.argv
        sys.argv = ["backtesting-multi", "--help"]
        with pytest.raises(SystemExit) as exc_info:
            _multi_parse_args()
        assert exc_info.value.code == 0
        sys.argv = old_argv

    def test_no_cache_flag_default_false(self) -> None:
        old_argv = sys.argv
        sys.argv = ["backtesting-multi", "--tickers", "SPY"]
        args = _multi_parse_args()
        assert args.no_cache is False
        sys.argv = old_argv

    def test_no_cache_flag_sets_true(self) -> None:
        old_argv = sys.argv
        sys.argv = ["backtesting-multi", "--tickers", "SPY", "--no-cache"]
        args = _multi_parse_args()
        assert args.no_cache is True
        sys.argv = old_argv

class TestNoCacheFlag:
    """--no-cache must thread use_cache=False through to load_data."""

    def test_no_cache_threads_to_load_data(self, tmp_path: Path) -> None:
        calls: list[bool] = []

        def _capturing_load(
            ticker: str,
            start: str,
            end_date: str | None = None,
            use_cache: bool = True,
        ) -> pd.DataFrame:
            calls.append(use_cache)
            return _mock_load_data(ticker, start, end_date=end_date, use_cache=use_cache)

        with patch("backtesting_engine.multi_asset.load_data", side_effect=_capturing_load):
            run_multi_asset(
                tickers=["SPY"],
                start="2010-01-01",
                end=None,
                execution=ExecutionConfig(),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
                use_cache=False,
            )

        assert calls, "load_data was never called"
        assert all(c is False for c in calls), (
            f"Expected use_cache=False on all calls, got {calls}"
        )

    def test_use_cache_defaults_to_true(self, tmp_path: Path) -> None:
        calls: list[bool] = []

        def _capturing_load(
            ticker: str,
            start: str,
            end_date: str | None = None,
            use_cache: bool = True,
        ) -> pd.DataFrame:
            calls.append(use_cache)
            return _mock_load_data(ticker, start, end_date=end_date, use_cache=use_cache)

        with patch("backtesting_engine.multi_asset.load_data", side_effect=_capturing_load):
            run_multi_asset(
                tickers=["SPY"],
                start="2010-01-01",
                end=None,
                execution=ExecutionConfig(),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=tmp_path,
            )

        assert all(c is True for c in calls), (
            f"Expected use_cache=True by default, got {calls}"
        )
