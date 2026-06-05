"""
Unit tests for the CLI argument parser.

Argument defaults, flag combinations, and rejection of invalid inputs.
Does not run the pipeline - only exercises _parse_args() and argument handling.

Every CLI flag that exists must have a test here. If you add a new flag,
add a corresponding test. This prevents the common failure mode of a flag
being documented in --help but silently ignored because it was never wired up.
"""

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

