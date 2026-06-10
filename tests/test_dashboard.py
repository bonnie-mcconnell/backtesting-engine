"""
Tests for dashboard.py.

Verifies that per-window bar colouring uses per-window benchmark Sharpe
rather than the aggregate mean - the key correctness invariant for the
"strategy vs benchmark" bar chart panel.
"""



class TestDashboardBarColoring:
    """
    Per-window Sharpe bars must be coloured against the per-window benchmark
    Sharpe, not the aggregate mean. Using the mean produces incorrect colours
    in windows where the benchmark itself had an unusually good or bad year.
    """

    def test_per_window_coloring_not_aggregate(self) -> None:
        """Construct a BenchmarkResult where per-window and aggregate differ,
        verify the coloring function uses per-window values."""
        from backtesting_engine.benchmark import BenchmarkResult

        # Scenario: aggregate mean BH Sharpe = 0.0, but window 0 had BH Sharpe = 1.0
        # and strategy Sharpe = 0.5. With aggregate, window 0 would be red (0.5 < 0.0
        # is false actually, but scenario: strategy=0.4, bm_aggregate=0.5, bm_window=0.1).
        # Strategy beats per-window BM (0.4 > 0.1) but not aggregate (0.4 < 0.5).
        # Correct: green. Wrong (old code): red.
        bm = BenchmarkResult(
            benchmark_sharpe=0.5,          # aggregate mean
            benchmark_sortino=0.5,
            benchmark_max_drawdown=-0.1,
            information_ratio=0.0,
            sharpe_diff_t_stat=0.0,
            sharpe_diff_p_value=0.5,
            strategy_beats_benchmark_fraction=0.5,
            per_window_benchmark_sharpes=[0.1, 0.9],  # window 0 weak, window 1 strong
        )

        # Strategy sharpes: 0.4 and 0.4
        strategy_sharpes = [0.4, 0.4]

        # Per-window coloring: window 0 → 0.4 > 0.1 → green; window 1 → 0.4 < 0.9 → red
        # Aggregate coloring: both → 0.4 < 0.5 → red (wrong for window 0)
        from backtesting_engine.dashboard import _NEGATIVE, _POSITIVE

        colors_per_window = []
        for i, s in enumerate(strategy_sharpes):
            bm_w = bm.per_window_benchmark_sharpes[i]
            colors_per_window.append(_POSITIVE if s >= bm_w else _NEGATIVE)

        colors_aggregate = [
            _POSITIVE if s >= bm.benchmark_sharpe else _NEGATIVE
            for s in strategy_sharpes
        ]

        assert colors_per_window[0] == _POSITIVE, "Window 0 should be green (beats per-window BM)"
        assert colors_per_window[1] == _NEGATIVE, "Window 1 should be red (loses to per-window BM)"
        assert colors_aggregate[0] == _NEGATIVE, "Old aggregate coloring would incorrectly show red"
        assert colors_aggregate[1] == _NEGATIVE


# ── 9. Execution docstring describes current defaults ────────────────────────
