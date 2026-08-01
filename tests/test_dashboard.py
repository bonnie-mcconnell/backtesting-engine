"""
Tests for dashboard.py.

Verifies that per-window bar colouring uses per-window benchmark Sharpe
rather than the aggregate mean - the key correctness invariant for the
"strategy vs benchmark" bar chart panel.
"""

import pandas as pd
import plotly.subplots as sp

from backtesting_engine.benchmark import BenchmarkResult
from backtesting_engine.dashboard import _NEGATIVE, _POSITIVE, _add_window_sharpes
from backtesting_engine.models import MetricsResult, SimulationResult, WindowResult


def _make_window(start: pd.Timestamp, end: pd.Timestamp, sharpe: float) -> WindowResult:
    sim = SimulationResult(trades=[], portfolio_values=None, message="")
    m = MetricsResult(
        sharpe_ratio=sharpe, sortino_ratio=0.5, max_drawdown=-0.1,
        calmar_ratio=1.0, omega_ratio=1.1, p_value=0.3,
    )
    return WindowResult(
        train_start=start, train_end=start, test_start=start, test_end=end,
        simulation_result=sim, metrics_result=m,
    )


class TestDashboardBarColoring:
    """
    Per-window Sharpe bars must be coloured against the per-window benchmark
    Sharpe, not the aggregate mean. Using the mean produces incorrect colours
    in windows where the benchmark itself had an unusually good or bad year.
    """

    def test_per_window_coloring_not_aggregate(self) -> None:
        """Build a real WindowResult/BenchmarkResult pair where per-window and
        aggregate benchmark Sharpe disagree on which colour a bar should get,
        call the actual dashboard function, and inspect the rendered colours -
        rather than reimplementing the coloring rule separately, which would
        prove the rule is sensible but not that the dashboard applies it.
        """
        dates = pd.date_range("2020-01-01", periods=3, freq="YS")
        windows = [
            _make_window(dates[0], dates[1], sharpe=0.4),
            _make_window(dates[1], dates[2], sharpe=0.4),
        ]
        bm = BenchmarkResult(
            benchmark_sharpe=0.5,  # aggregate mean
            benchmark_sortino=0.5,
            benchmark_max_drawdown=-0.1,
            information_ratio=0.0,
            sharpe_diff_t_stat=0.0,
            sharpe_diff_p_value=0.5,
            strategy_beats_benchmark_fraction=0.5,
            per_window_benchmark_sharpes=[0.1, 0.9],  # window 0 weak, window 1 strong
        )
        # Strategy Sharpe is 0.4 in both windows.
        # Per-window (correct): window 0 → 0.4 > 0.1 → green; window 1 → 0.4 < 0.9 → red
        # Aggregate (wrong):    both → 0.4 < 0.5 → red

        fig = sp.make_subplots(rows=1, cols=1)
        _add_window_sharpes(fig, windows, row=1, col=1, benchmark=bm)
        colors = fig.data[0].marker.color

        assert colors[0] == _POSITIVE, "Window 0 should be green (beats per-window BM)"
        assert colors[1] == _NEGATIVE, "Window 1 should be red (loses to per-window BM)"
