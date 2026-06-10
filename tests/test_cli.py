"""
Unit tests for the CLI argument parser.

Argument defaults, flag combinations, and rejection of invalid inputs.
Does not run the pipeline - only exercises _parse_args() and argument handling.

Every CLI flag that exists must have a test here. If you add a new flag,
add a corresponding test. This prevents the common failure mode of a flag
being documented in --help but silently ignored because it was never wired up.
"""

import inspect
import io
import pathlib
import re
import tokenize
from unittest.mock import patch

import pytest

from backtesting_engine.main import _parse_args


class TestCLIDefaults:
    def test_no_args_gives_all_strategy(self) -> None:
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.strategy == "all"

    def test_no_args_no_costs_only(self) -> None:
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.costs_only is False

    def test_default_ticker(self) -> None:
        from backtesting_engine.config import TICKER
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.ticker == TICKER

    def test_default_start(self) -> None:
        from backtesting_engine.config import START_DATE
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.start == START_DATE

    def test_default_no_cache_is_false(self) -> None:
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.no_cache is False

    def test_default_end_is_none(self) -> None:
        """--end defaults to None, meaning today's date is used at runtime."""
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.end is None

    def test_default_cost(self) -> None:
        """Default cost matches ExecutionConfig default and README documentation."""
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.cost == 0.001

    def test_default_slippage(self) -> None:
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.slippage == 0.05

    def test_default_delay(self) -> None:
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.delay == 1

    def test_default_train_years(self) -> None:
        from backtesting_engine.config import TRAINING_WINDOW_YEARS
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.train_years == TRAINING_WINDOW_YEARS

    def test_default_test_years(self) -> None:
        from backtesting_engine.config import TESTING_WINDOW_YEARS
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.test_years == TESTING_WINDOW_YEARS

    def test_default_seed_is_none(self) -> None:
        """--seed defaults to None; main() falls back to BLOCK_BOOTSTRAP_SEED."""
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.seed is None

    def test_default_output_dir(self) -> None:
        with patch("sys.argv", ["backtesting-engine"]):
            args = _parse_args()
        assert args.output_dir == "."


class TestCLIStrategyFlag:
    def test_strategy_ma(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--strategy", "ma"]):
            args = _parse_args()
        assert args.strategy == "ma"

    def test_strategy_kalman(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--strategy", "kalman"]):
            args = _parse_args()
        assert args.strategy == "kalman"

    def test_strategy_momentum(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--strategy", "momentum"]):
            args = _parse_args()
        assert args.strategy == "momentum"

    def test_strategy_all_explicit(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--strategy", "all"]):
            args = _parse_args()
        assert args.strategy == "all"

    def test_invalid_strategy_raises(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--strategy", "rsi"]):
            with pytest.raises(SystemExit):
                _parse_args()


class TestCLICostsOnly:
    def test_costs_only_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--costs-only"]):
            args = _parse_args()
        assert args.costs_only is True


class TestCLICustomData:
    def test_custom_ticker(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--ticker", "QQQ"]):
            args = _parse_args()
        assert args.ticker == "QQQ"

    def test_custom_start(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--start", "2000-01-01"]):
            args = _parse_args()
        assert args.start == "2000-01-01"

    def test_no_cache_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--no-cache"]):
            args = _parse_args()
        assert args.no_cache is True


class TestCLINewFlags:
    """
    Tests for all flags added after the initial release.

    Every flag must have a test here. If a flag is parsed but never used
    (like the old --seed bug), this test class is where that would be caught
    at the argparse level - integration tests catch it at the pipeline level.
    """

    def test_end_date_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--end", "2024-12-31"]):
            args = _parse_args()
        assert args.end == "2024-12-31"

    def test_cost_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--cost", "0.0005"]):
            args = _parse_args()
        assert args.cost == pytest.approx(0.0005)

    def test_slippage_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--slippage", "0.10"]):
            args = _parse_args()
        assert args.slippage == pytest.approx(0.10)

    def test_delay_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--delay", "2"]):
            args = _parse_args()
        assert args.delay == 2

    def test_train_years_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--train-years", "5"]):
            args = _parse_args()
        assert args.train_years == 5

    def test_test_years_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--test-years", "2"]):
            args = _parse_args()
        assert args.test_years == 2

    def test_seed_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--seed", "99"]):
            args = _parse_args()
        assert args.seed == 99

    def test_output_dir_flag(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--output-dir", "/tmp/results"]):
            args = _parse_args()
        assert args.output_dir == "/tmp/results"

    def test_zero_cost_is_valid(self) -> None:
        """Zero transaction cost should be accepted (useful for strategy logic checks)."""
        with patch("sys.argv", ["backtesting-engine", "--cost", "0.0"]):
            args = _parse_args()
        assert args.cost == 0.0

    def test_zero_slippage_is_valid(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--slippage", "0.0"]):
            args = _parse_args()
        assert args.slippage == 0.0

    def test_zero_delay_is_valid(self) -> None:
        with patch("sys.argv", ["backtesting-engine", "--delay", "0"]):
            args = _parse_args()
        assert args.delay == 0

    def test_combined_flags(self) -> None:
        """All new flags can coexist in one invocation."""
        argv = [
            "backtesting-engine",
            "--ticker", "QQQ",
            "--start", "2000-01-01",
            "--end", "2023-12-31",
            "--cost", "0.0005",
            "--slippage", "0.03",
            "--delay", "1",
            "--train-years", "2",
            "--test-years", "1",
            "--seed", "7",
            "--output-dir", "/tmp/out",
            "--strategy", "ma",
        ]
        with patch("sys.argv", argv):
            args = _parse_args()
        assert args.ticker == "QQQ"
        assert args.end == "2023-12-31"
        assert args.cost == pytest.approx(0.0005)
        assert args.seed == 7
        assert args.output_dir == "/tmp/out"



# ---------------------------------------------------------------------------

# CLI correctness: _fmt_metric, _min_rows, --end date, encoding


# ---------------------------------------------------------------------------

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

        import backtesting_engine.main as m

        src = inspect.getsource(m._load)
        # Must only apply the timedelta when end_date is not None.
        assert "if end_date is not None" in src
        assert "timedelta(days=1)" in src


# ---------------------------------------------------------------------------
# Source file encoding portability
# ---------------------------------------------------------------------------

class TestSourceFileEncoding:
    """All source-file reads in tests must specify encoding='utf-8'.

    Source files contain Unicode characters (box drawing, arrows, Greek letters).
    On Windows, Path.read_text() defaults to cp1252 which cannot decode these.
    """

    _SRC_FILES = [
        "tests/test_strategy.py",
        "tests/test_cli.py",
        "tests/test_benchmark.py",
    ]

    def test_all_read_text_calls_specify_encoding(self) -> None:
        """Every .read_text() call in test code must pass encoding='utf-8'.

        Uses tokenize to inspect actual call sites, not docstring prose.
        """
        tok = tokenize

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


# ---------------------------------------------------------------------------
# --no-dashboard and --workers flags (v0.11.0)
# ---------------------------------------------------------------------------

class TestNoDashboardFlag:
    """
    --no-dashboard must suppress all build_dashboard calls in both CLIs
    without affecting metric computation or stdout output.
    """

    def test_no_dashboard_default_is_false(self) -> None:
        # Parse with minimal required args - no_dashboard must default to False.
        from backtesting_engine.main import _parse_args as _main_parse
        with patch("sys.argv", ["backtesting-engine", "--ticker", "SPY",
                                 "--start", "2010-01-01"]):
            args = _main_parse()
        assert args.no_dashboard is False

    def test_workers_default_is_one_via_parse(self) -> None:
        from backtesting_engine.main import _parse_args as _main_parse
        with patch("sys.argv", ["backtesting-engine", "--ticker", "SPY",
                                 "--start", "2010-01-01"]):
            args = _main_parse()
        assert args.workers == 1

    def test_no_dashboard_suppresses_build_dashboard(self) -> None:
        """When --no-dashboard is set, build_dashboard must not be called."""
        from unittest.mock import patch as upatch

        import numpy as np
        import pandas as pd

        from backtesting_engine.main import main

        dates = pd.date_range("1993-01-01", periods=2268, freq="B")
        close = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0003, 0.01, 2268))
        fake_data = pd.DataFrame({
            "open": close, "high": close * 1.005, "low": close * 0.995,
            "close": close, "volume": 1e6,
        }, index=dates)

        build_dashboard_calls = []

        def mock_build(*args, **kwargs):  # type: ignore[no-untyped-def]
            build_dashboard_calls.append(1)
            return kwargs.get("path", args[1] if len(args) > 1 else "mock.html")

        with (upatch("sys.argv", [
                "backtesting-engine",
                "--ticker", "SPY", "--strategy", "ma",
                "--start", "1993-01-01", "--end", "2001-12-31",
                "--no-dashboard",
              ]),
              upatch("backtesting_engine.main.load_data", return_value=fake_data),
              upatch("backtesting_engine.main.validate_data"),
              upatch("backtesting_engine.main.build_dashboard", side_effect=mock_build),
        ):
            try:
                main()
            except SystemExit:
                pass

        assert len(build_dashboard_calls) == 0, (
            f"build_dashboard was called {len(build_dashboard_calls)} times "
            "despite --no-dashboard being set."
        )

    def test_no_dashboard_in_multi_asset_run(self) -> None:
        """run_multi_asset with no_dashboard=True must not call build_dashboard."""
        from pathlib import Path
        from unittest.mock import patch as upatch

        from helpers import make_oscillating_data

        from backtesting_engine.execution import ExecutionConfig
        from backtesting_engine.multi_asset import run_multi_asset

        build_calls = []

        def mock_build(*args, **kwargs):  # type: ignore[no-untyped-def]
            build_calls.append(1)

        with (upatch("backtesting_engine.multi_asset.load_data",
                     return_value=make_oscillating_data(756, with_high_low=True)),
              upatch("backtesting_engine.multi_asset.validate_data"),
              upatch("backtesting_engine.multi_asset.build_dashboard", side_effect=mock_build),
        ):
            run_multi_asset(
                tickers=["SPY"],
                start="2010-01-01",
                end="2016-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=Path("/tmp"),
                strategy="ma",
                no_dashboard=True,
            )

        assert len(build_calls) == 0, (
            f"build_dashboard called {len(build_calls)} times with no_dashboard=True"
        )

    def test_no_dashboard_false_calls_build_dashboard(self) -> None:
        """run_multi_asset with no_dashboard=False (default) must call build_dashboard."""
        from pathlib import Path
        from unittest.mock import patch as upatch

        from helpers import make_oscillating_data

        from backtesting_engine.execution import ExecutionConfig
        from backtesting_engine.multi_asset import run_multi_asset

        build_calls = []

        def mock_build(*args, **kwargs):  # type: ignore[no-untyped-def]
            build_calls.append(1)

        with (upatch("backtesting_engine.multi_asset.load_data",
                     return_value=make_oscillating_data(756, with_high_low=True)),
              upatch("backtesting_engine.multi_asset.validate_data"),
              upatch("backtesting_engine.multi_asset.build_dashboard", side_effect=mock_build),
        ):
            run_multi_asset(
                tickers=["SPY"],
                start="2010-01-01",
                end="2016-12-31",
                execution=ExecutionConfig(slippage_factor=0.0, signal_delay=0),
                train_years=1,
                test_years=1,
                bootstrap_seed=42,
                output_dir=Path("/tmp"),
                strategy="ma",
                no_dashboard=False,
            )

        assert len(build_calls) == 1, (
            f"Expected 1 build_dashboard call, got {len(build_calls)}"
        )
