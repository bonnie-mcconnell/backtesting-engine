"""
Unit tests for the CLI argument parser.

Tests cover argument defaults, flag combinations, and rejection of invalid inputs.
These tests do not run the full backtesting pipeline - they only verify that
argparse is configured correctly and that _parse_args() returns the expected
Namespace for each combination of flags.
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
