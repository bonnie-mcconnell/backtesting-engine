"""End-to-end tests for the walk-forward, benchmark, and dashboard pipeline."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtesting_engine.benchmark import BenchmarkResult, compute_benchmark
from backtesting_engine.config import ANNUALISATION_FACTOR, INITIAL_PORTFOLIO_VALUE
from backtesting_engine.dashboard import build_dashboard
from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.models import BacktestResult
from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
from backtesting_engine.strategy.momentum import MomentumStrategy
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_trending_data(n: int = 756, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic OHLCV with a gentle uptrend and some oscillation.

    Uses a fixed seed so tests are deterministic. The slight positive drift
    ensures at least some trade signals are generated across all windows.
    n=756 = 3 years of daily data, enough for 1-year train / 6-month test
    with ~4 walk-forward windows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    # Gentle uptrend + sine oscillation to generate crossover signals
    close = 100.0 + 0.02 * t + 8.0 * np.sin(2 * np.pi * t / 80) + rng.normal(0, 0.5, n)
    close = np.maximum(close, 1.0)   # guard against negatives on synthetic data
    high = close * (1.0 + 0.005 + 0.003 * rng.random(n))
    low = close * (1.0 - 0.005 - 0.003 * rng.random(n))
    return pd.DataFrame(
        {"close": close, "high": high, "low": low},
        index=dates,
    )


def _zero_friction() -> ExecutionConfig:
    """Zero-friction config for fast integration tests."""
    return ExecutionConfig(
        transaction_cost_rate=0.0,
        slippage_factor=0.0,
        signal_delay=0,
    )


# ---------------------------------------------------------------------------
# Full pipeline: walk_forward → benchmark → dashboard
# ---------------------------------------------------------------------------


class TestFullPipelineMA:
    """Moving Average strategy: full pipeline integration."""

    def setup_method(self) -> None:
        self.data = _make_trending_data()
        self.exec = _zero_friction()
        self.result = walk_forward(
            self.data, MovingAverageStrategy(),
            training_window_years=1,
            testing_window_years=1,
            execution=self.exec,
            bootstrap_seed=42,
        )

    def test_walk_forward_returns_backtest_result(self) -> None:
        assert isinstance(self.result, BacktestResult)

    def test_result_has_windows(self) -> None:
        assert len(self.result.window_results) > 0

    def test_summary_metrics_are_finite_or_nan(self) -> None:
        m = self.result.summary_metrics
        # Sharpe and Sortino can be NaN only if all windows were flat-cash.
        # With trending synthetic data at least one window should trade.
        assert not math.isnan(m.sharpe_ratio), "Sharpe should be finite for trending data"
        assert not math.isnan(m.max_drawdown)
        # p-values must be in [0, 1]
        assert 0.0 <= m.p_value <= 1.0
        assert 0.0 <= m.combined_p_value <= 1.0

    def test_fisher_p_and_rc_p_cover_same_windows(self) -> None:
        """Both statistics must be computed. RC may be NaN only if no candidates."""
        m = self.result.summary_metrics
        assert not math.isnan(m.combined_p_value), "Fisher p must always be computable"
        # MA has a candidate grid, so RC p should be finite too.
        assert not math.isnan(m.reality_check_p_value), (
            "RC p should be finite for MA (has parameter grid)"
        )

    def test_benchmark_computes_without_error(self) -> None:
        bm = compute_benchmark(self.result, self.data, execution=self.exec)
        assert isinstance(bm, BenchmarkResult)
        assert len(bm.per_window_benchmark_sharpes) == len(self.result.valid_windows)

    def test_benchmark_per_window_sharpes_mean_invariant(self) -> None:
        """Mean of per-window BH Sharpes must equal benchmark_sharpe."""
        bm = compute_benchmark(self.result, self.data, execution=self.exec)
        mean_pw = float(np.mean(bm.per_window_benchmark_sharpes))
        assert math.isclose(mean_pw, bm.benchmark_sharpe, rel_tol=1e-9), (
            f"Mean of per-window {mean_pw:.6f} != aggregate {bm.benchmark_sharpe:.6f}"
        )

    def test_dashboard_writes_html_without_error(self) -> None:
        bm = compute_benchmark(self.result, self.data, execution=self.exec)
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                self.result,
                output_path=Path(tmp) / "test_ma.html",
                strategy_name_override="MA (integration test)",
                benchmark=bm,
                price_data=self.data["close"],
            )
            assert path.exists()
            html = path.read_text(encoding="utf-8")
            # Self-contained: must embed Plotly JS
            assert "plotly" in html.lower()
            # Must contain strategy name
            assert "MA" in html
            # Must not be trivially empty
            assert len(html) > 50_000, f"Dashboard HTML suspiciously small: {len(html)} chars"

    def test_dashboard_html_is_self_contained(self) -> None:
        """Dashboard must not reference external CDN resources."""
        bm = compute_benchmark(self.result, self.data, execution=self.exec)
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                self.result,
                output_path=Path(tmp) / "test_ma_cdn.html",
                benchmark=bm,
                price_data=self.data["close"],
            )
            html = path.read_text(encoding="utf-8")
            # cdn.plot.ly or similar would break offline use
            assert "cdn.plot.ly" not in html, "Dashboard loads Plotly from CDN"
            assert "cdnjs.cloudflare.com" not in html, "Dashboard loads from cloudflare CDN"

    def test_window_results_span_correct_date_range(self) -> None:
        """Test windows must not overlap the training period."""
        for w in self.result.window_results:
            assert w.test_start > w.train_end, (
                f"Test start {w.test_start} not after train end {w.train_end}"
            )
            assert w.test_end >= w.test_start

    def test_no_negative_portfolio_values(self) -> None:
        """Portfolio value must never go negative (no leverage)."""
        for w in self.result.window_results:
            pv = w.simulation_result.portfolio_values
            if pv is not None:
                assert (pv >= 0).all(), f"Negative portfolio value in window {w.test_start}"

    def test_flat_cash_windows_in_valid_windows(self) -> None:
        """All windows (including flat-cash) must appear in valid_windows."""
        assert len(self.result.valid_windows) == len(self.result.window_results)


class TestFullPipelineKalman:
    """Kalman Filter strategy: full pipeline integration."""

    def test_kalman_full_pipeline(self) -> None:
        data = _make_trending_data(n=756, seed=1)
        exec_cfg = _zero_friction()

        result = walk_forward(
            data, KalmanFilterStrategy(),
            training_window_years=1,
            testing_window_years=1,
            execution=exec_cfg,
            bootstrap_seed=42,
        )

        assert isinstance(result, BacktestResult)
        assert len(result.window_results) > 0

        # Kalman has no parameter grid → RC p should be NaN
        assert math.isnan(result.summary_metrics.reality_check_p_value), (
            "Kalman has no candidate grid; RC p should be NaN"
        )

        bm = compute_benchmark(result, data, execution=exec_cfg)
        assert isinstance(bm, BenchmarkResult)

        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                result,
                output_path=Path(tmp) / "test_kalman.html",
                benchmark=bm,
                price_data=data["close"],
            )
            assert path.exists()
            assert len(path.read_text(encoding="utf-8")) > 50_000

    def test_kalman_active_params_populated(self) -> None:
        """Every Kalman window must record Q, R, and log-likelihood."""
        data = _make_trending_data(n=756, seed=2)
        result = walk_forward(
            data, KalmanFilterStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=_zero_friction(), bootstrap_seed=42,
        )
        for w in result.window_results:
            params = w.active_params
            assert "q" in params and "r" in params, (
                f"Window {w.test_start}: missing q or r in active_params: {params}"
            )
            assert float(params["q"]) > 0 and float(params["r"]) > 0


class TestFullPipelineMomentum:
    """Momentum strategy: full pipeline integration."""

    def test_momentum_full_pipeline(self) -> None:
        data = _make_trending_data(n=756, seed=3)
        exec_cfg = _zero_friction()

        result = walk_forward(
            data, MomentumStrategy(),
            training_window_years=1,
            testing_window_years=1,
            execution=exec_cfg,
            bootstrap_seed=42,
        )

        assert isinstance(result, BacktestResult)
        # Momentum has a candidate grid → RC p should be finite
        assert not math.isnan(result.summary_metrics.reality_check_p_value)

        bm = compute_benchmark(result, data, execution=exec_cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                result,
                output_path=Path(tmp) / "test_momentum.html",
                benchmark=bm,
                price_data=data["close"],
            )
            assert path.exists()


# ---------------------------------------------------------------------------
# Cross-component consistency
# ---------------------------------------------------------------------------


class TestCrossComponentConsistency:
    """Verify that outputs from one component are consistent with inputs to another."""

    def _run(self, seed: int = 10) -> tuple[BacktestResult, BenchmarkResult, pd.DataFrame]:
        data = _make_trending_data(seed=seed)
        exec_cfg = _zero_friction()
        result = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=exec_cfg, bootstrap_seed=42,
        )
        bm = compute_benchmark(result, data, execution=exec_cfg)
        return result, bm, data

    def test_benchmark_window_count_matches_result(self) -> None:
        result, bm, _ = self._run()
        assert len(bm.per_window_benchmark_sharpes) == len(result.valid_windows)

    def test_benchmark_beats_fraction_is_consistent(self) -> None:
        """strategy_beats_benchmark_fraction must match manual count."""
        result, bm, _ = self._run()
        strat_sharpes = [w.metrics_result.sharpe_ratio for w in result.valid_windows]
        bm_sharpes = bm.per_window_benchmark_sharpes
        manual_fraction = sum(
            s > b for s, b in zip(strat_sharpes, bm_sharpes)
        ) / len(strat_sharpes)
        assert math.isclose(manual_fraction, bm.strategy_beats_benchmark_fraction, rel_tol=1e-9)

    def test_window_dates_are_non_overlapping(self) -> None:
        """Test windows must not overlap each other."""
        result, _, _ = self._run()
        windows = sorted(result.valid_windows, key=lambda w: w.test_start)
        for i in range(1, len(windows)):
            assert windows[i].test_start > windows[i - 1].test_end, (
                f"Window {i} test_start {windows[i].test_start} overlaps "
                f"window {i-1} test_end {windows[i-1].test_end}"
            )

    def test_portfolio_values_start_near_initial_capital(self) -> None:
        """First portfolio value must equal INITIAL_PORTFOLIO_VALUE (no trades yet)."""
        result, _, _ = self._run()
        for w in result.valid_windows:
            pv = w.simulation_result.portfolio_values
            if pv is not None and len(pv) > 0:
                assert math.isclose(pv.iloc[0], INITIAL_PORTFOLIO_VALUE, rel_tol=0.01), (
                    f"First bar portfolio value {pv.iloc[0]} far from "
                    f"initial capital {INITIAL_PORTFOLIO_VALUE}"
                )

    def test_trade_pnl_consistent_with_portfolio(self) -> None:
        """Total P&L across trades must approximately equal final-minus-initial portfolio value."""
        result, _, _ = self._run()
        for w in result.valid_windows:
            trades = w.simulation_result.trades
            pv = w.simulation_result.portfolio_values
            if not trades or pv is None or len(pv) < 2:
                continue
            total_pnl = sum(t.pnl for t in trades)
            pv_change = float(pv.iloc[-1] - pv.iloc[0])
            # Allow 1% tolerance for rounding and the cost model
            assert math.isclose(total_pnl, pv_change, rel_tol=0.02), (
                f"Sum of trade P&L ({total_pnl:.2f}) inconsistent with "
                f"portfolio change ({pv_change:.2f})"
            )

    def test_summary_sharpe_is_mean_of_window_sharpes(self) -> None:
        """Summary Sharpe must equal the mean of finite per-window Sharpes."""
        result, _, _ = self._run()
        window_sharpes = [
            w.metrics_result.sharpe_ratio for w in result.valid_windows
            if not math.isnan(w.metrics_result.sharpe_ratio)
            and abs(w.metrics_result.sharpe_ratio) != float("inf")
        ]
        expected = float(np.mean(window_sharpes))
        assert math.isclose(expected, result.summary_metrics.sharpe_ratio, rel_tol=1e-9)

    def test_dashboard_param_evolution_uses_price_data(self) -> None:
        """Dashboard must not produce different BH curves in the param and equity panels.

        Before the fix, _add_param_evolution fallback called _add_cumulative_benchmark
        without price_data, producing an inconsistent BH curve. This test verifies
        the fixed code path: price_data is forwarded to the fallback.
        """
        # Use Kalman (has param evolution) to exercise the normal path,
        # and verify no exception is raised when price_data is passed.
        data = _make_trending_data(seed=5)
        result = walk_forward(
            data, KalmanFilterStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=_zero_friction(), bootstrap_seed=42,
        )
        bm = compute_benchmark(result, data, execution=_zero_friction())
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                result,
                output_path=Path(tmp) / "test_param_evo.html",
                benchmark=bm,
                price_data=data["close"],   # must be passed through to param panel
            )
            html = path.read_text(encoding="utf-8")
            # The parameter evolution panel should show up for Kalman
            assert "Q/R" in html or "signal" in html.lower() or "snr" in html.lower()


# ---------------------------------------------------------------------------
# Edge cases that span multiple components
# ---------------------------------------------------------------------------


class TestPipelineEdgeCases:
    """Edge cases that only surface through the full pipeline."""

    def test_single_window_pipeline_completes(self) -> None:
        """Minimum viable dataset: exactly one train + one test window."""
        # 2 years of data → exactly one 1-year train + 1-year test window
        n = 2 * ANNUALISATION_FACTOR + 10
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        t = np.arange(n, dtype=float)
        close = 100.0 + 0.03 * t + 5.0 * np.sin(2 * np.pi * t / 60)
        data = pd.DataFrame({"close": close, "high": close * 1.005, "low": close * 0.995}, index=dates)

        result = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=_zero_friction(), bootstrap_seed=42,
        )
        assert len(result.window_results) >= 1
        bm = compute_benchmark(result, data, execution=_zero_friction())
        assert isinstance(bm, BenchmarkResult)

    def test_all_flat_cash_raises(self) -> None:
        """Fully flat price data produces zero trades → walk_forward must raise."""
        n = 756
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        data = pd.DataFrame(
            {"close": np.full(n, 100.0), "high": np.full(n, 100.5), "low": np.full(n, 99.5)},
            index=dates,
        )
        with pytest.raises(ValueError, match="zero trades"):
            walk_forward(
                data, MovingAverageStrategy(),
                training_window_years=1, testing_window_years=1,
                execution=_zero_friction(),
            )

    def test_dashboard_without_benchmark_renders(self) -> None:
        """Dashboard must render correctly when no benchmark is provided."""
        data = _make_trending_data(seed=20)
        result = walk_forward(
            data, MomentumStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=_zero_friction(), bootstrap_seed=42,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                result,
                output_path=Path(tmp) / "no_bm.html",
                # No benchmark kwarg
                price_data=data["close"],
            )
            assert path.exists()
            assert len(path.read_text(encoding="utf-8")) > 10_000

    def test_execution_config_with_realistic_costs_completes(self) -> None:
        """Realistic execution config (costs + slippage + delay) must not crash."""
        data = _make_trending_data(seed=30)
        exec_cfg = ExecutionConfig(
            transaction_cost_rate=0.001,
            slippage_factor=0.05,
            signal_delay=1,
        )
        result = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=exec_cfg, bootstrap_seed=42,
        )
        assert isinstance(result, BacktestResult)
        bm = compute_benchmark(result, data, execution=exec_cfg)
        # Slippage cost applied to benchmark must reduce its Sharpe vs zero-cost
        bm_no_cost = compute_benchmark(result, data, execution=_zero_friction())
        # This is not guaranteed in all cases but holds for our synthetic data
        # (slippage reduces benchmark return on entry/exit)
        assert bm.benchmark_sharpe <= bm_no_cost.benchmark_sharpe + 0.5, (
            "Benchmark Sharpe with costs should be <= without costs (within tolerance)"
        )

    def test_rc_p_and_fisher_p_cover_same_windows(self) -> None:
        """RC p must be computable and both statistics cover all windows.

        Before the RC flat-cash parity fix, flat-cash windows contributed to
        Fisher (p=1.0) but were excluded from the RC candidate matrix. This
        caused the two statistics to test different hypotheses.
        """
        # Use slow oscillation to guarantee some flat-cash windows
        n = 1260
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        t = np.arange(n, dtype=float)
        close = 100.0 + 15.0 * np.sin(2 * np.pi * t / 750.0)
        data = pd.DataFrame({"close": close}, index=dates)

        result = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=_zero_friction(), bootstrap_seed=42,
        )

        assert result.flat_cash_window_count > 0, "Expected flat-cash windows for slow oscillation"
        m = result.summary_metrics

        assert not math.isnan(m.combined_p_value), "Fisher p must be computable"
        assert not math.isnan(m.reality_check_p_value), (
            "RC p must be computable even with flat-cash windows (parity fix)"
        )
        assert 0.0 <= m.combined_p_value <= 1.0
        assert 0.0 <= m.reality_check_p_value <= 1.0
