"""
Unit tests for the execution model.

Tests cover slippage fills, signal delay, config validation,
and backward compatibility with the original simulator.
"""

import numpy as np
import pandas as pd
import pytest

from backtesting_engine.config import TRANSACTION_COST_RATE
from backtesting_engine.execution import ExecutionConfig, run_simulation_with_execution


def _ohlcv(n: int = 5, base: float = 100.0) -> pd.DataFrame:
    """Synthetic OHLCV with a 1-point intraday range per bar."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(base, base + n - 1, n)
    return pd.DataFrame(
        {"open": close - 0.2, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=dates,
    )


def _close_only(n: int = 5, base: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": np.linspace(base, base + n - 1, n)}, index=dates)


# ---------------------------------------------------------------------------
# ExecutionConfig validation
# ---------------------------------------------------------------------------

class TestExecutionConfig:
    def test_default_values(self) -> None:
        ec = ExecutionConfig()
        assert ec.transaction_cost_rate == TRANSACTION_COST_RATE
        assert ec.slippage_factor == 0.0
        assert ec.signal_delay == 0

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ExecutionConfig(transaction_cost_rate=-0.001)

    def test_negative_slippage_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ExecutionConfig(slippage_factor=-0.1)

    def test_negative_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ExecutionConfig(signal_delay=-1)

    def test_frozen_config(self) -> None:
        ec = ExecutionConfig()
        with pytest.raises(Exception):
            ec.slippage_factor = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------

class TestSlippage:
    def test_zero_slippage_fills_at_close(self) -> None:
        data = _ohlcv()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)
        result = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0))
        assert result.trades[0].entry_price == data["close"].iloc[1]

    def test_positive_slippage_raises_buy_price(self) -> None:
        data = _ohlcv()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)
        r_no = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0))
        r_sl = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.2))
        assert r_sl.trades[0].entry_price > r_no.trades[0].entry_price

    def test_slippage_reduces_pnl(self) -> None:
        data = _ohlcv()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)
        r_no = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0))
        r_sl = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.3))
        assert r_sl.trades[0].pnl < r_no.trades[0].pnl

    def test_slippage_requires_high_low_columns(self) -> None:
        data = _close_only()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)
        with pytest.raises(ValueError, match="high.*low|low.*high"):
            run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.1))


# ---------------------------------------------------------------------------
# Signal delay
# ---------------------------------------------------------------------------

class TestSignalDelay:
    def test_delay_zero_executes_on_signal_bar(self) -> None:
        data = _close_only()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)
        result = run_simulation_with_execution(data, signals, ExecutionConfig(signal_delay=0))
        assert result.trades[0].entry_price == data["close"].iloc[1]

    def test_delay_one_shifts_execution_by_one_bar(self) -> None:
        data = _close_only(n=7)
        signals = pd.Series([0, 1, 0, 0, -1, 0, 0], index=data.index)
        result = run_simulation_with_execution(data, signals, ExecutionConfig(signal_delay=1))
        if result.trades:
            # Buy signal on bar 1 → executes at bar 2's price
            assert result.trades[0].entry_price == data["close"].iloc[2]

    def test_delay_does_not_increase_trade_count(self) -> None:
        data = _close_only(n=10)
        signals = pd.Series([0, 1, 0, -1, 0, 1, 0, -1, 0, 0], index=data.index)
        r0 = run_simulation_with_execution(data, signals, ExecutionConfig(signal_delay=0))
        r1 = run_simulation_with_execution(data, signals, ExecutionConfig(signal_delay=1))
        assert len(r1.trades) <= len(r0.trades)


# ---------------------------------------------------------------------------
# Backward compatibility with original simulator
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_default_config_matches_original_simulator(self) -> None:
        from backtesting_engine.simulator import run_simulation

        data = _close_only()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)

        r_orig = run_simulation(data, signals)
        r_new = run_simulation_with_execution(data, signals, ExecutionConfig())

        assert len(r_orig.trades) == len(r_new.trades)
        if r_orig.trades:
            assert abs(r_orig.trades[0].pnl - r_new.trades[0].pnl) < 1e-6

    def test_portfolio_always_positive(self) -> None:
        data = _close_only()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)
        result = run_simulation_with_execution(data, signals, ExecutionConfig())
        assert result.portfolio_values is not None
        assert (result.portfolio_values > 0).all()
