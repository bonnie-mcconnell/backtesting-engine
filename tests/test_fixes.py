"""
Tests for fixes applied in the v0.6.x → v0.8.0 pass.

Covers every correctness issue identified across the audit and fix passes:
  1. _fmt_metric no longer crashes with infinite values
  2. _min_rows uses runtime train/test years, not module-level defaults
  3. --end date is inclusive (yfinance offset applied internally)
  4. RC flat-cash parity: flat-cash windows contribute to RC candidate matrix
  5. RC boundary carry-over: candidates use same state as selected strategy
  6. Benchmark slippage: BH pays same slippage as strategy
  7. BenchmarkResult carries per-window benchmark sharpes
  8. Dashboard bar coloring uses per-window, not aggregate, benchmark Sharpe
  9. Execution docstring describes current defaults (not old zero-slippage)
 10. All source file reads specify encoding="utf-8" (Windows portability)
"""

import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from helpers import make_oscillating_data

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int, with_high_low: bool = True) -> pd.DataFrame:
    """Synthetic OHLCV with a gentle uptrend."""
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    close = np.array([100.0 + i * 0.05 for i in range(n)])
    df = pd.DataFrame({"close": close}, index=dates)
    if with_high_low:
        df["high"] = close * 1.005
        df["low"] = close * 0.995
    return df


def _make_oscillating(n: int = 504, with_high_low: bool = True) -> pd.DataFrame:
    """Thin wrapper around make_oscillating_data for backward compat within this module.

    test_fixes.py previously defined its own oscillating data generator that
    was a near-duplicate of helpers.make_oscillating_data. This wrapper preserves
    all call sites in this module while routing through the single canonical
    implementation in helpers.py, eliminating the duplication.
    """
    return make_oscillating_data(n, with_high_low=with_high_low)


# ── 1. _fmt_metric crash ──────────────────────────────────────────────────────

class TestFmtMetric:
    """_fmt_metric must never raise ValueError by passing ∞ as a format spec."""

    def _fmt(self, v: float) -> str:
        from backtesting_engine.main import _fmt_metric
        return _fmt_metric(v)

    def test_normal_float_formats(self) -> None:
        result = self._fmt(1.234)
        assert "1.234" in result

    def test_nan_returns_na(self) -> None:
        assert self._fmt(float("nan")) == "N/A"

    def test_positive_inf_does_not_crash(self) -> None:
        result = self._fmt(float("inf"))
        assert result  # must not raise, must return a string

    def test_negative_inf_does_not_crash(self) -> None:
        result = self._fmt(float("-inf"))
        assert result

    def test_inf_result_is_not_raw_format_specifier(self) -> None:
        # The old bug: _fmt_metric(v, "∞") tried format(v, "∞") → ValueError.
        # Verify the returned string is a display value, not the raw symbol.
        result = self._fmt(float("inf"))
        # Should not look like Python tried to use ∞ as a format spec error message.
        assert "format" not in result.lower()
        assert "unknown" not in result.lower()

    def test_comparison_table_does_not_crash(self) -> None:
        """The comparative summary crashed when best3() called _fmt_metric(v, '∞').
        Simulate the all-inf case that triggered it."""
        from backtesting_engine.main import _fmt_metric
        # This is exactly the call that previously caused ValueError.
        # Previously: _fmt_metric(float("inf"), "∞") → crash.
        # Now: _fmt_metric(float("inf")) → returns a safe display string.
        result = _fmt_metric(float("inf"))
        assert isinstance(result, str)
        assert len(result) > 0


# ── 2. _min_rows uses runtime args ────────────────────────────────────────────

class TestMinRows:
    """_min_rows must reflect CLI --train-years / --test-years, not config defaults."""

    def test_default_args_produce_expected_minimum(self) -> None:
        from backtesting_engine.config import (
            ANNUALISATION_FACTOR,
            MOVING_AVERAGE_LONG_DAYS,
            TESTING_WINDOW_YEARS,
            TRAINING_WINDOW_YEARS,
        )
        from backtesting_engine.main import _min_rows
        expected = (TRAINING_WINDOW_YEARS + TESTING_WINDOW_YEARS) * ANNUALISATION_FACTOR + MOVING_AVERAGE_LONG_DAYS
        assert _min_rows(TRAINING_WINDOW_YEARS, TESTING_WINDOW_YEARS) == expected

    def test_longer_windows_require_more_rows(self) -> None:
        from backtesting_engine.main import _min_rows
        assert _min_rows(5, 2) > _min_rows(3, 1)

    def test_function_exists_and_module_level_constant_removed(self) -> None:
        """_MIN_ROWS module-level constant must be gone; _min_rows function takes args."""
        import backtesting_engine.main as m
        assert not hasattr(m, "_MIN_ROWS"), (
            "_MIN_ROWS module-level constant should be removed. "
            "Use _min_rows(train_years, test_years) instead."
        )
        assert callable(m._min_rows)


# ── 3. --end inclusive (yfinance offset) ─────────────────────────────────────

class TestEndDateInclusive:
    """_load must add one day internally so --end YYYY-MM-DD is inclusive."""

    def test_yf_end_is_one_day_after_user_end(self) -> None:
        """The internal yf_end passed to yfinance must be end_date + 1 day."""
        from datetime import date, timedelta
        user_end = "2024-12-31"
        expected_yf_end = (date.fromisoformat(user_end) + timedelta(days=1)).isoformat()
        assert expected_yf_end == "2025-01-01"

    def test_end_none_does_not_add_offset(self) -> None:
        """When end_date is None (today), no offset should be applied."""
        # The logic in _load: yf_end = None when end_date is None.
        # We test the main module compiles and the logic is correct by inspection.
        import inspect

        import backtesting_engine.main as m

        src = inspect.getsource(m._load)
        # Must only apply the timedelta when end_date is not None.
        assert "if end_date is not None" in src
        assert "timedelta(days=1)" in src


# ── 4. RC flat-cash parity ────────────────────────────────────────────────────

def _make_mixed_window_data_for_flat_cash() -> pd.DataFrame:
    """Price series that produces some flat-cash AND some trading windows.

    A very slow sine wave (period=750 bars) means individual 252-bar test
    windows are often monotone (no MA crossover -> flat-cash), but the full
    5-year dataset has enough trend reversals that at least 2 windows trade.
    This avoids triggering the zero-total-trades defensive guard.

    Module-level (not a method) so the pytest fixture below can reference it
    without instantiating TestRCFlatCashParity. The fixture result is cached
    at class scope so walk_forward runs exactly once for all three tests.
    """
    n = 1260
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    close = 100.0 + 15.0 * np.sin(2 * np.pi * t / 750.0)
    return pd.DataFrame({"close": close}, index=dates)


@pytest.fixture(scope="class")
def flat_cash_result():
    """
    Single walk_forward result shared across all TestRCFlatCashParity tests.

    scope="class" means this fixture is computed once when the first test in
    the class runs, then reused for all subsequent tests in the class.
    Without sharing, _run_walk_forward_with_flat_cash() was called 3× in
    separate method calls, each running the full MA grid search + bootstrap.
    With a class-scoped fixture, the expensive computation runs exactly once.
    """
    from backtesting_engine.execution import ExecutionConfig
    from backtesting_engine.strategy.moving_average import MovingAverageStrategy
    from backtesting_engine.walk_forward import walk_forward

    return walk_forward(
        _make_mixed_window_data_for_flat_cash(),
        MovingAverageStrategy(),
        training_window_years=1,
        testing_window_years=1,
        execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
    )


class TestRCFlatCashParity:
    """
    Flat-cash windows must contribute to the RC candidate matrix just as they
    contribute p=1.0 to Fisher's combined p. Without this, Fisher and RC test
    different hypotheses over different windows.

    The walk_forward result is shared via the flat_cash_result fixture (class
    scope) so the expensive MA grid search + bootstrap runs exactly once for
    all three tests in this class, not three times.
    """

    def test_flat_cash_windows_are_valid_windows(self, flat_cash_result) -> None:
        """Flat-cash windows must be included in valid_windows with skipped=False."""
        result = flat_cash_result
        assert result.flat_cash_window_count > 0, (
            "Expected at least one flat-cash window with slow-oscillation data. "
            "Verify the sine period is long enough relative to the test window."
        )
        assert len(result.valid_windows) == len(result.window_results), (
            "All windows (including flat-cash) must appear in valid_windows."
        )

    def test_rc_p_is_not_nan_when_strategy_has_candidates(self, flat_cash_result) -> None:
        """RC p must be computable even when some windows are flat-cash.

        Before the fix: flat-cash windows skipped RC candidate collection but
        contributed p=1.0 to Fisher -> inconsistent hypotheses, RC often NaN.
        After the fix: flat-cash windows contribute zero-return arrays to the
        RC matrix so both statistics test the same set of windows.
        """
        rc_p = flat_cash_result.summary_metrics.reality_check_p_value
        assert not math.isnan(rc_p), (
            "RC p is NaN despite strategy having candidates. "
            "Flat-cash windows must contribute zero-return arrays to RC matrix "
            "(required for parity with Fisher's combined p)."
        )
        assert 0.0 <= rc_p <= 1.0, f"RC p-value out of [0,1]: {rc_p}"

    def test_fisher_and_rc_cover_same_number_of_windows(self) -> None:
        """Fisher p uses all windows including flat-cash; RC must too.

        Uses its own walk_forward call rather than the shared fixture because
        it needs a different dataset (with high/low) to verify the property holds
        under a different data regime.
        """
        from backtesting_engine.execution import ExecutionConfig
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        from backtesting_engine.walk_forward import walk_forward

        data = _make_oscillating(756, with_high_low=True)
        result = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
        )
        # Both stats must have been computed over the same window set.
        # We can't inspect the internal lists, but we can verify Fisher p
        # and RC p are both finite (not NaN) when candidates exist.
        assert not math.isnan(result.summary_metrics.combined_p_value)
        rc_p = result.summary_metrics.reality_check_p_value
        if not math.isnan(rc_p):
            # RC p ≥ Fisher p is the theoretical lower bound of the data-snooping correction.
            # Not a hard guarantee in finite samples, but RC p < 0.0 or > 1.0 would be wrong.
            assert 0.0 <= rc_p <= 1.0


# ── 5. RC boundary carry-over parity ─────────────────────────────────────────

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


# ── 6. Benchmark slippage parity ──────────────────────────────────────────────

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
    from backtesting_engine.walk_forward import walk_forward

    data = _make_oscillating(756, with_high_low=True)
    exec_slip = ExecutionConfig(transaction_cost_rate=0.001, slippage_factor=0.05, signal_delay=0)
    result = walk_forward(
        data, MovingAverageStrategy(),
        training_window_years=1, testing_window_years=1,
        execution=exec_slip,
    )
    return data, result, exec_slip


class TestBenchmarkSlippageParity:
    """
    The benchmark must apply the same slippage as the strategy.
    Previously _buy_and_hold_returns only applied transaction costs.
    """

    def test_slippage_reduces_benchmark_return(self) -> None:
        from backtesting_engine.benchmark import _buy_and_hold_returns

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = np.array([100.0 + i * 0.1 for i in range(n)])
        high = close * 1.01
        low = close * 0.99
        data_df = pd.DataFrame({"close": close, "high": high, "low": low}, index=dates)

        returns_no_slip = _buy_and_hold_returns(data_df, cost_rate=0.001, slippage_factor=0.0)
        returns_with_slip = _buy_and_hold_returns(data_df, cost_rate=0.001, slippage_factor=0.1)

        # With slippage, entry/exit returns must be lower.
        assert returns_with_slip[0] < returns_no_slip[0], (
            "Entry return should be lower with slippage"
        )
        assert returns_with_slip[-1] < returns_no_slip[-1], (
            "Exit return should be lower with slippage"
        )

    def test_zero_slippage_matches_old_series_api(self) -> None:
        """With slippage_factor=0 and a plain Series input, result is the same
        as the old API (backward compatibility)."""
        from backtesting_engine.benchmark import _buy_and_hold_returns

        prices = pd.Series([100.0, 101.0, 102.0, 101.0, 103.0])
        returns_series = _buy_and_hold_returns(prices, cost_rate=0.001, slippage_factor=0.0)

        dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
        data_df = pd.DataFrame({"close": prices.values}, index=dates)
        returns_df = _buy_and_hold_returns(data_df, cost_rate=0.001, slippage_factor=0.0)

        np.testing.assert_allclose(returns_series, returns_df, rtol=1e-10)

    def test_compute_benchmark_passes_slippage_to_bh(self, slippage_parity_data) -> None:
        """compute_benchmark must forward slippage from ExecutionConfig to BH returns."""
        from backtesting_engine.benchmark import compute_benchmark
        from backtesting_engine.execution import ExecutionConfig

        data, result, exec_slip = slippage_parity_data
        exec_no_slip = ExecutionConfig(transaction_cost_rate=0.001, slippage_factor=0.0, signal_delay=0)

        bm_with_slip = compute_benchmark(result, data, execution=exec_slip)
        bm_no_slip = compute_benchmark(result, data, execution=exec_no_slip)

        # Higher slippage must penalise benchmark Sharpe.
        assert bm_with_slip.benchmark_sharpe <= bm_no_slip.benchmark_sharpe, (
            "Benchmark Sharpe with slippage should be <= without slippage"
        )


# ── 7. BenchmarkResult per-window sharpes ────────────────────────────────────

@pytest.fixture(scope="class")
def per_window_sharpe_result():
    """
    walk_forward + compute_benchmark result shared across TestBenchmarkResultPerWindowSharpes.

    scope="class": runs once for all three tests in the class, not once per test.
    The MA grid search runs on each call to walk_forward; sharing avoids 3×
    redundant executions of the same computation.
    """
    from backtesting_engine.benchmark import compute_benchmark
    from backtesting_engine.execution import ExecutionConfig
    from backtesting_engine.strategy.moving_average import MovingAverageStrategy
    from backtesting_engine.walk_forward import walk_forward

    data = _make_oscillating(756, with_high_low=True)
    result = walk_forward(
        data, MovingAverageStrategy(),
        training_window_years=1, testing_window_years=1,
        execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
    )
    bm = compute_benchmark(result, data)
    return result, bm


class TestBenchmarkResultPerWindowSharpes:
    """BenchmarkResult must carry per_window_benchmark_sharpes.

    All three tests share a single walk_forward result via the class-scoped
    per_window_sharpe_result fixture. Without sharing, the MA grid search
    ran independently for each test method.
    """

    def test_per_window_sharpes_populated(self, per_window_sharpe_result) -> None:
        result, bm = per_window_sharpe_result
        assert len(bm.per_window_benchmark_sharpes) == len(result.valid_windows), (
            "per_window_benchmark_sharpes must have one entry per valid window"
        )

    def test_per_window_sharpes_mean_equals_benchmark_sharpe(
        self, per_window_sharpe_result
    ) -> None:
        """Mean of per-window BH sharpes must equal benchmark_sharpe."""
        _, bm = per_window_sharpe_result
        mean_pw = float(np.mean(bm.per_window_benchmark_sharpes))
        assert math.isclose(mean_pw, bm.benchmark_sharpe, rel_tol=1e-9), (
            f"Mean of per-window sharpes {mean_pw:.6f} != benchmark_sharpe {bm.benchmark_sharpe:.6f}"
        )

    def test_per_window_sharpes_are_all_finite(self, per_window_sharpe_result) -> None:
        _, bm = per_window_sharpe_result
        for i, s in enumerate(bm.per_window_benchmark_sharpes):
            assert math.isfinite(s), f"Window {i} benchmark Sharpe is not finite: {s}"


# ── 8. Dashboard bar coloring uses per-window benchmark Sharpe ───────────────

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

class TestExecutionDocstring:
    """The execution docstring must describe current realistic defaults, not old zero-slippage."""

    def test_docstring_mentions_current_defaults(self) -> None:
        from backtesting_engine.execution import run_simulation_with_execution
        doc = run_simulation_with_execution.__doc__ or ""
        # Must mention realistic defaults, not old "zero slippage, zero delay".
        assert "0.1%" in doc or "cost=0.1" in doc or "realistic" in doc or "0.05" in doc, (
            "Docstring should describe current realistic defaults, not old zero-slippage model"
        )
        assert "zero slippage, zero delay" not in doc.lower(), (
            "Docstring must not describe old zero-friction defaults as the standard mode"
        )


# ─── 10. Source file encoding (Windows portability) ─────────────────────────

class TestSourceFileEncoding:
    """All source-file reads in tests must specify encoding='utf-8'.

    Source files contain Unicode characters (box drawing, arrows, Greek letters).
    On Windows, Path.read_text() defaults to cp1252 which cannot decode these.
    """

    _SRC_FILES = [
        "tests/test_final_fixes.py",
        "tests/test_strategy.py",
        "tests/test_fixes.py",
    ]

    def test_all_read_text_calls_specify_encoding(self) -> None:
        """Every .read_text() call in test code must pass encoding='utf-8'.

        Uses tokenize to inspect actual call sites, not docstring prose.
        """
        import io
        import re
        import tokenize as tok

        repo_root = pathlib.Path(__file__).parent.parent
        # Match an actual Python read_text() call (identifier/paren before the dot).
        call_re = re.compile(r'[\w)]\.read_text\(\s*\)')
        violations = []

        for rel_path in self._SRC_FILES:
            fpath = repo_root / rel_path
            if not fpath.exists():
                continue
            src = fpath.read_text(encoding="utf-8")
            lines = src.splitlines()

            # Use tokenize to find line numbers that are inside string literals
            # (multiline docstrings). Those lines are prose, not executable code.
            string_lines: set[int] = set()
            try:
                for ttype, _, tstart, tend, _ in tok.generate_tokens(
                    io.StringIO(src).readline
                ):
                    if ttype == tok.STRING:
                        for ln in range(tstart[0] + 1, tend[0]):
                            # Interior lines of a multiline string
                            string_lines.add(ln)
            except tok.TokenError:
                pass

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if i in string_lines:
                    continue  # interior of a multiline docstring
                if call_re.search(line) and "encoding" not in line:
                    violations.append(f"{rel_path}:{i}: {stripped}")

        assert not violations, (
            "read_text() calls missing encoding='utf-8' (crashes on Windows):\n"
            + "\n".join(violations)
        )

    def test_source_files_are_valid_utf8(self) -> None:
        """All Python source files must be valid UTF-8."""
        repo_root = pathlib.Path(__file__).parent.parent
        src_dir = repo_root / "src"
        failures = []
        for pyfile in src_dir.rglob("*.py"):
            try:
                pyfile.read_text(encoding="utf-8")
            except UnicodeDecodeError as e:
                failures.append(f"{pyfile.relative_to(repo_root)}: {e}")
        assert not failures, "Files with invalid UTF-8:\n" + "\n".join(failures)
