"""End-to-end tests for the walk-forward, benchmark, and dashboard pipeline.

Performance design
------------------
Integration tests are expensive by nature: they run real walk_forward calls
with full strategy grid searches. The original version used setup_method (which
re-runs walk_forward for every test method in a class) and a _run() helper
called independently six times in TestCrossComponentConsistency.

This version uses class-scoped fixtures so each expensive walk_forward call
runs exactly once per class, not once per test method.

  TestFullPipelineMA:            1 MA walk_forward (was 9 × setup_method)
  TestFullPipelineKalman:        1 Kalman walk_forward (was per-test)
  TestFullPipelineMomentum:      1 Momentum walk_forward (was per-test)
  TestCrossComponentConsistency: 1 MA walk_forward (was 6 × _run())
  TestPipelineEdgeCases:         4 separate calls (each tests distinct behaviour)

Total: 8 walk_forward calls vs ~26 previously.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtesting_engine.benchmark import BenchmarkResult, compute_benchmark
from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    INITIAL_PORTFOLIO_VALUE,
)
from backtesting_engine.dashboard import build_dashboard
from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.models import BacktestResult
from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
from backtesting_engine.strategy.momentum import MomentumStrategy
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_trending_data(n: int = 756, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic OHLCV with a gentle uptrend and oscillation.

    Uses a fixed seed so tests are deterministic. The slight positive drift
    + sine oscillation ensures at least some MA crossover signals are
    generated across all windows. n=756 = ~3 years, enough for 1+1yr windows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    close = 100.0 + 0.02 * t + 8.0 * np.sin(2 * np.pi * t / 80) + rng.normal(0, 0.5, n)
    close = np.maximum(close, 1.0)
    high = close * (1.0 + 0.005 + 0.003 * rng.random(n))
    low = close * (1.0 - 0.005 - 0.003 * rng.random(n))
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=dates)


def _zero_friction() -> ExecutionConfig:
    return ExecutionConfig(
        transaction_cost_rate=0.0,
        slippage_factor=0.0,
        signal_delay=0,
    )


# ---------------------------------------------------------------------------
# Class-scoped fixtures for shared expensive results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def ma_pipeline():
    """Full MA pipeline result shared across all TestFullPipelineMA tests.

    scope="class": walk_forward runs once for the class, not once per test method.
    """
    data = _make_trending_data()
    exec_cfg = _zero_friction()
    result = walk_forward(
        data, MovingAverageStrategy(),
        training_window_years=1,
        testing_window_years=1,
        execution=exec_cfg,
        bootstrap_seed=42,
    )
    benchmark = compute_benchmark(result, data, execution=exec_cfg)
    return {"result": result, "data": data, "exec": exec_cfg, "benchmark": benchmark}


@pytest.fixture(scope="class")
def kalman_pipeline():
    """Full Kalman pipeline result shared across TestFullPipelineKalman tests."""
    data = _make_trending_data(seed=1)
    exec_cfg = _zero_friction()
    result = walk_forward(
        data, KalmanFilterStrategy(),
        training_window_years=1,
        testing_window_years=1,
        execution=exec_cfg,
        bootstrap_seed=42,
    )
    benchmark = compute_benchmark(result, data, execution=exec_cfg)
    return {"result": result, "data": data, "exec": exec_cfg, "benchmark": benchmark}


@pytest.fixture(scope="class")
def momentum_pipeline():
    """Full Momentum pipeline result shared across TestFullPipelineMomentum tests."""
    data = _make_trending_data(seed=3)
    exec_cfg = _zero_friction()
    result = walk_forward(
        data, MomentumStrategy(),
        training_window_years=1,
        testing_window_years=1,
        execution=exec_cfg,
        bootstrap_seed=42,
    )
    benchmark = compute_benchmark(result, data, execution=exec_cfg)
    return {"result": result, "data": data, "exec": exec_cfg, "benchmark": benchmark}


@pytest.fixture(scope="class")
def cross_component_pipeline():
    """Shared MA pipeline for TestCrossComponentConsistency.

    scope="class": walk_forward runs once for the class, not once per test method.
    """
    data = _make_trending_data(seed=10)
    exec_cfg = _zero_friction()
    result = walk_forward(
        data, MovingAverageStrategy(),
        training_window_years=1,
        testing_window_years=1,
        execution=exec_cfg,
        bootstrap_seed=42,
    )
    bm = compute_benchmark(result, data, execution=exec_cfg)
    return {"result": result, "data": data, "exec": exec_cfg, "benchmark": bm}


# ---------------------------------------------------------------------------
# Full pipeline: MA strategy
# ---------------------------------------------------------------------------

class TestFullPipelineMA:
    """Moving Average strategy: full pipeline integration.

    All tests share a single walk_forward result via the ma_pipeline fixture.
    Inspect result properties, not the walk_forward call itself - so
    sharing is correct and doesn't weaken the assertions.
    """

    def test_walk_forward_returns_backtest_result(self, ma_pipeline: dict) -> None:
        assert isinstance(ma_pipeline["result"], BacktestResult)

    def test_result_has_windows(self, ma_pipeline: dict) -> None:
        assert len(ma_pipeline["result"].window_results) > 0

    def test_summary_metrics_are_finite_or_nan(self, ma_pipeline: dict) -> None:
        m = ma_pipeline["result"].summary_metrics
        assert not math.isnan(m.sharpe_ratio), "Sharpe should be finite for trending data"
        assert not math.isnan(m.max_drawdown)
        assert 0.0 <= m.p_value <= 1.0
        assert 0.0 <= m.combined_p_value <= 1.0

    def test_fisher_p_and_rc_p_cover_same_windows(self, ma_pipeline: dict) -> None:
        m = ma_pipeline["result"].summary_metrics
        assert not math.isnan(m.combined_p_value), "Fisher p must always be computable"
        assert not math.isnan(m.reality_check_p_value), (
            "RC p should be finite for MA (has parameter grid)"
        )

    def test_benchmark_computes_without_error(self, ma_pipeline: dict) -> None:
        bm = ma_pipeline["benchmark"]
        assert isinstance(bm, BenchmarkResult)
        assert len(bm.per_window_benchmark_sharpes) == len(
            ma_pipeline["result"].valid_windows
        )

    def test_benchmark_per_window_sharpes_mean_invariant(
        self, ma_pipeline: dict
    ) -> None:
        bm = ma_pipeline["benchmark"]
        mean_pw = float(np.mean(bm.per_window_benchmark_sharpes))
        assert math.isclose(mean_pw, bm.benchmark_sharpe, rel_tol=1e-9)

    def test_dashboard_writes_html_without_error(self, ma_pipeline: dict) -> None:
        p = ma_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                p["result"],
                output_path=Path(tmp) / "test_ma.html",
                strategy_name_override="MA (integration test)",
                benchmark=p["benchmark"],
                price_data=p["data"]["close"],
            )
            assert path.exists()
            html = path.read_text(encoding="utf-8")
            assert "plotly" in html.lower()
            assert "MA" in html
            assert len(html) > 50_000, f"Dashboard HTML suspiciously small: {len(html)} chars"

    def test_dashboard_html_is_self_contained(self, ma_pipeline: dict) -> None:
        """Dashboard must not load scripts from external CDN.

        The embedded Plotly JS bundle contains the string 'cdn.plot.ly' as a
        URL used internally for topojson data - this is fine and expected.
        What we are testing is that Plotly itself is embedded (not loaded via
        a <script src="..."> tag pointing to a CDN), so the dashboard works
        offline.

        The correct check: no <script src="...cdn..."> tag exists in the HTML.
        Searching for bare 'cdn.plot.ly' would produce a false failure because
        that string appears inside the embedded JS bundle.
        """
        p = ma_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                p["result"],
                output_path=Path(tmp) / "test_ma_cdn.html",
                benchmark=p["benchmark"],
                price_data=p["data"]["close"],
            )
            html = path.read_text(encoding="utf-8")
            # Must not load Plotly or any other resource from external CDN.
            # These patterns would indicate a <script src="..."> dependency.
            assert 'src="https://cdn.plot.ly' not in html, (
                "Dashboard loads Plotly from cdn.plot.ly via <script src>"
            )
            assert 'src="https://cdnjs.cloudflare.com' not in html, (
                "Dashboard loads a resource from cloudflare CDN"
            )
            # Must include the Plotly bundle inline.
            assert "var Plotly" in html or "window.Plotly" in html or "plotly.js" in html.lower(), (
                "Dashboard does not appear to embed Plotly JS inline"
            )

    def test_window_results_span_correct_date_range(
        self, ma_pipeline: dict
    ) -> None:
        for w in ma_pipeline["result"].window_results:
            assert w.test_start > w.train_end, (
                f"Test start {w.test_start} not after train end {w.train_end}"
            )
            assert w.test_end >= w.test_start

    def test_no_negative_portfolio_values(self, ma_pipeline: dict) -> None:
        """Portfolio value must never go negative (no leverage)."""
        for w in ma_pipeline["result"].window_results:
            pv = w.simulation_result.portfolio_values
            if pv is not None:
                assert (pv >= 0).all(), (
                    f"Negative portfolio value in window {w.test_start}"
                )

    def test_flat_cash_windows_in_valid_windows(self, ma_pipeline: dict) -> None:
        """All windows including flat-cash must appear in valid_windows."""
        r = ma_pipeline["result"]
        assert len(r.valid_windows) == len(r.window_results)


# ---------------------------------------------------------------------------
# Full pipeline: Kalman filter strategy
# ---------------------------------------------------------------------------

class TestFullPipelineKalman:
    """Kalman Filter strategy: full pipeline integration."""

    def test_returns_backtest_result(self, kalman_pipeline: dict) -> None:
        assert isinstance(kalman_pipeline["result"], BacktestResult)

    def test_has_windows(self, kalman_pipeline: dict) -> None:
        assert len(kalman_pipeline["result"].window_results) > 0

    def test_rc_p_is_nan_no_candidate_grid(self, kalman_pipeline: dict) -> None:
        """Kalman has no parameter grid → RC p should be NaN, not raise."""
        assert math.isnan(
            kalman_pipeline["result"].summary_metrics.reality_check_p_value
        ), "Kalman has no candidate grid; RC p should be NaN"

    def test_active_params_populated(self, kalman_pipeline: dict) -> None:
        """Every Kalman window must record Q, R (MLE-fitted noise variances)."""
        for w in kalman_pipeline["result"].window_results:
            params = w.active_params
            assert "q" in params and "r" in params, (
                f"Window {w.test_start}: missing q or r in active_params: {params}"
            )
            assert float(params["q"]) > 0 and float(params["r"]) > 0

    def test_dashboard_writes_html(self, kalman_pipeline: dict) -> None:
        p = kalman_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                p["result"],
                output_path=Path(tmp) / "test_kalman.html",
                benchmark=p["benchmark"],
                price_data=p["data"]["close"],
            )
            assert path.exists()
            assert len(path.read_text(encoding="utf-8")) > 50_000

    def test_dashboard_param_evolution_uses_price_data(
        self, kalman_pipeline: dict
    ) -> None:
        """Dashboard parameter evolution panel must render without error.

        Kalman's param_evolution_spec() returns Q/R SNR entries, so the
        parameter evolution panel should contain recognisable Kalman labels.
        Guards against the dashboard silently falling back to an equity-only
        panel when param_evolution_spec is populated.
        """
        p = kalman_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                p["result"],
                output_path=Path(tmp) / "test_param_evo.html",
                benchmark=p["benchmark"],
                price_data=p["data"]["close"],
            )
            html = path.read_text(encoding="utf-8")
            assert "Q/R" in html or "signal" in html.lower() or "snr" in html.lower()


# ---------------------------------------------------------------------------
# Full pipeline: Momentum strategy
# ---------------------------------------------------------------------------

class TestFullPipelineMomentum:
    """Momentum strategy: full pipeline integration."""

    def test_returns_backtest_result(self, momentum_pipeline: dict) -> None:
        assert isinstance(momentum_pipeline["result"], BacktestResult)

    def test_rc_p_is_finite(self, momentum_pipeline: dict) -> None:
        """Momentum has a lookback grid → RC p must be finite."""
        assert not math.isnan(
            momentum_pipeline["result"].summary_metrics.reality_check_p_value
        )

    def test_dashboard_writes_html(self, momentum_pipeline: dict) -> None:
        p = momentum_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            path = build_dashboard(
                p["result"],
                output_path=Path(tmp) / "test_momentum.html",
                benchmark=p["benchmark"],
                price_data=p["data"]["close"],
            )
            assert path.exists()


# ---------------------------------------------------------------------------
# Cross-component consistency
# ---------------------------------------------------------------------------

class TestCrossComponentConsistency:
    """Verify outputs from one component are consistent with inputs to another.

    All tests share a single walk_forward result via cross_component_pipeline.
    """

    def test_benchmark_window_count_matches_result(
        self, cross_component_pipeline: dict
    ) -> None:
        r, bm = cross_component_pipeline["result"], cross_component_pipeline["benchmark"]
        assert len(bm.per_window_benchmark_sharpes) == len(r.valid_windows)

    def test_benchmark_beats_fraction_is_consistent(
        self, cross_component_pipeline: dict
    ) -> None:
        """strategy_beats_benchmark_fraction must match a manual per-window count."""
        r, bm = cross_component_pipeline["result"], cross_component_pipeline["benchmark"]
        strat_sharpes = [w.metrics_result.sharpe_ratio for w in r.valid_windows]
        bm_sharpes = bm.per_window_benchmark_sharpes
        manual = sum(s > b for s, b in zip(strat_sharpes, bm_sharpes)) / len(strat_sharpes)
        assert math.isclose(manual, bm.strategy_beats_benchmark_fraction, rel_tol=1e-9)

    def test_window_dates_are_non_overlapping(
        self, cross_component_pipeline: dict
    ) -> None:
        r = cross_component_pipeline["result"]
        windows = sorted(r.valid_windows, key=lambda w: w.test_start)
        for i in range(1, len(windows)):
            assert windows[i].test_start > windows[i - 1].test_end, (
                f"Window {i} test_start {windows[i].test_start} overlaps "
                f"window {i-1} test_end {windows[i-1].test_end}"
            )

    def test_portfolio_values_start_near_initial_capital(
        self, cross_component_pipeline: dict
    ) -> None:
        """First portfolio value must equal INITIAL_PORTFOLIO_VALUE (no trades yet)."""
        for w in cross_component_pipeline["result"].valid_windows:
            pv = w.simulation_result.portfolio_values
            if pv is not None and len(pv) > 0:
                assert math.isclose(pv.iloc[0], INITIAL_PORTFOLIO_VALUE, rel_tol=0.01), (
                    f"First bar portfolio {pv.iloc[0]} far from "
                    f"initial capital {INITIAL_PORTFOLIO_VALUE}"
                )

    def test_trade_pnl_consistent_with_portfolio(
        self, cross_component_pipeline: dict
    ) -> None:
        """Total P&L across trades must approximately equal final−initial portfolio value."""
        for w in cross_component_pipeline["result"].valid_windows:
            trades = w.simulation_result.trades
            pv = w.simulation_result.portfolio_values
            if not trades or pv is None or len(pv) < 2:
                continue
            total_pnl = sum(t.pnl for t in trades)
            pv_change = float(pv.iloc[-1] - pv.iloc[0])
            assert math.isclose(total_pnl, pv_change, rel_tol=0.02), (
                f"Sum of trade P&L ({total_pnl:.2f}) inconsistent with "
                f"portfolio change ({pv_change:.2f})"
            )

    def test_summary_sharpe_is_mean_of_window_sharpes(
        self, cross_component_pipeline: dict
    ) -> None:
        r = cross_component_pipeline["result"]
        window_sharpes = [
            w.metrics_result.sharpe_ratio for w in r.valid_windows
            if not math.isnan(w.metrics_result.sharpe_ratio)
            and abs(w.metrics_result.sharpe_ratio) != float("inf")
        ]
        expected = float(np.mean(window_sharpes))
        assert math.isclose(expected, r.summary_metrics.sharpe_ratio, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Edge cases spanning multiple components
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    """Edge cases that only surface through the full pipeline.

    These tests each exercise distinct behaviour and cannot share a result -
    each one uses a different data shape, strategy config, or execution config.
    """

    def test_single_window_pipeline_completes(self) -> None:
        """Minimum viable dataset: exactly one train + one test window."""
        n = 2 * ANNUALISATION_FACTOR + 10
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        t = np.arange(n, dtype=float)
        close = 100.0 + 0.03 * t + 5.0 * np.sin(2 * np.pi * t / 60)
        data = pd.DataFrame({
            "close": close,
            "high": close * 1.005,
            "low": close * 0.995,
        }, index=dates)
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
        data = pd.DataFrame({
            "close": np.full(n, 100.0),
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
        }, index=dates)
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
        bm_no_cost = compute_benchmark(result, data, execution=_zero_friction())
        assert bm.benchmark_sharpe <= bm_no_cost.benchmark_sharpe + 0.5

    def test_rc_p_and_fisher_p_cover_same_windows(self) -> None:
        """RC p must be computable even when some windows are flat-cash.

        Flat-cash windows contribute p=1.0 to Fisher but must also contribute
        zero-return arrays to the RC candidate matrix. Without this parity,
        the two statistics test different window sets and are non-comparable.
        """
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
        assert result.flat_cash_window_count > 0, (
            "Expected flat-cash windows for slow oscillation data"
        )
        m = result.summary_metrics
        assert not math.isnan(m.combined_p_value)
        assert not math.isnan(m.reality_check_p_value), (
            "RC p must be computable even with flat-cash windows (parity fix)"
        )
        assert 0.0 <= m.combined_p_value <= 1.0
        assert 0.0 <= m.reality_check_p_value <= 1.0
