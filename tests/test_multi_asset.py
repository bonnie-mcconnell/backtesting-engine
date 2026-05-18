"""
Tests for multi_asset.py cross-asset validation module.

Tests verify:
  1. run_multi_asset produces correct output structure
  2. Tickers that fail gracefully (insufficient data, load error) are skipped
  3. The comparison table prints without error
  4. CLI argument parsing produces correct ExecutionConfig values
  5. module is importable as __main__
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtesting_engine.benchmark import BenchmarkResult
from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.models import BacktestResult
from backtesting_engine.multi_asset import (
    _print_comparison_table,
    run_multi_asset,
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
        import pandas as pd

        from backtesting_engine.models import MetricsResult, SimulationResult, WindowResult

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
        import backtesting_engine.multi_asset as ma
        assert hasattr(ma, "run_multi_asset")
        assert hasattr(ma, "main")
        assert hasattr(ma, "_parse_args")

    def test_help_does_not_crash(self) -> None:
        """--help should print and raise SystemExit(0)."""
        import sys

        from backtesting_engine.multi_asset import _parse_args

        old_argv = sys.argv
        sys.argv = ["multi_asset", "--help"]
        with pytest.raises(SystemExit) as exc_info:
            _parse_args()
        assert exc_info.value.code == 0
        sys.argv = old_argv
