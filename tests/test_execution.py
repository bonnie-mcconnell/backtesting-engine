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
        # Defaults now match CLI/README: conservative realistic execution
        assert ec.slippage_factor == 0.05
        assert ec.signal_delay == 1

    def test_zero_friction_explicit(self) -> None:
        """Zero-friction config is available; it must be explicit, not default."""
        ec = ExecutionConfig(slippage_factor=0.0, signal_delay=0)
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
        result = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0, signal_delay=0))
        assert result.trades[0].entry_price == data["close"].iloc[1]

    def test_positive_slippage_raises_buy_price(self) -> None:
        data = _ohlcv()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)
        r_no = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0, signal_delay=0))
        r_sl = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.2, signal_delay=0))
        assert r_sl.trades[0].entry_price > r_no.trades[0].entry_price

    def test_slippage_reduces_pnl(self) -> None:
        data = _ohlcv()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)
        r_no = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0, signal_delay=0))
        r_sl = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.3, signal_delay=0))
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
        result = run_simulation_with_execution(data, signals, ExecutionConfig(signal_delay=0, slippage_factor=0.0))
        assert result.trades[0].entry_price == data["close"].iloc[1]

    def test_delay_one_shifts_execution_by_one_bar(self) -> None:
        data = _close_only(n=7)
        signals = pd.Series([0, 1, 0, 0, -1, 0, 0], index=data.index)
        result = run_simulation_with_execution(data, signals, ExecutionConfig(signal_delay=1, slippage_factor=0.0))
        if result.trades:
            # Buy signal on bar 1 → executes at bar 2's price
            assert result.trades[0].entry_price == data["close"].iloc[2]

    def test_delay_does_not_increase_trade_count(self) -> None:
        data = _close_only(n=10)
        signals = pd.Series([0, 1, 0, -1, 0, 1, 0, -1, 0, 0], index=data.index)
        r0 = run_simulation_with_execution(data, signals, ExecutionConfig(signal_delay=0, slippage_factor=0.0))
        r1 = run_simulation_with_execution(data, signals, ExecutionConfig(signal_delay=1, slippage_factor=0.0))
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
        r_new = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0, signal_delay=0))

        assert len(r_orig.trades) == len(r_new.trades)
        if r_orig.trades:
            assert abs(r_orig.trades[0].pnl - r_new.trades[0].pnl) < 1e-6

    def test_portfolio_always_positive(self) -> None:
        data = _close_only()
        signals = pd.Series([0, 1, 0, -1, 0], index=data.index)
        result = run_simulation_with_execution(data, signals, ExecutionConfig(slippage_factor=0.0, signal_delay=0))
        assert result.portfolio_values is not None
        assert (result.portfolio_values > 0).all()


# ── Execution docstring currency ──────────────────────────────────────────────

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


# ── Cost sweep signal delay threading ─────────────────────────────────────────

class TestCostSweepSignalDelay:
    """
    cost_sensitivity_sweep must honour the signal_delay parameter.

    The sweep runs each (cost, slippage) cell in a worker process via
    _sweep_worker. If signal_delay is not threaded through to the worker,
    sweep results will silently use a different delay than the caller
    specified - meaning a sweep at delay=0 would actually use delay=1
    (or vice versa), making the heatmap non-comparable to a direct
    walk_forward() call with the same parameters.
    """

    def test_delay_zero_sharpe_differs_from_delay_one(self) -> None:
        """Sharpe must differ between delay=0 and delay=1 - fill timing changes outcomes."""
        import math

        from helpers import make_oscillating_data

        from backtesting_engine.execution import ExecutionConfig
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        from backtesting_engine.walk_forward import walk_forward

        data = make_oscillating_data(756, with_high_low=True)

        r0 = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=ExecutionConfig(signal_delay=0), bootstrap_seed=42,
        )
        r1 = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=ExecutionConfig(signal_delay=1), bootstrap_seed=42,
        )
        s0 = r0.summary_metrics.sharpe_ratio
        s1 = r1.summary_metrics.sharpe_ratio
        assert not math.isnan(s0) and not math.isnan(s1)
        assert s0 != s1, (
            "delay=0 and delay=1 produced identical Sharpe ratios - "
            "signal_delay is not affecting simulation results."
        )

    def test_sweep_signal_delay_matches_direct_walk_forward(self) -> None:
        """
        Fisher p from cost_sensitivity_sweep(signal_delay=0) must match a direct
        walk_forward(signal_delay=0) call on the same data and seed.

        If signal_delay is not reaching _sweep_worker, sweep results will use
        a different delay than specified - making the heatmap non-comparable
        to direct walk_forward calls with the same parameters.
        """
        import math

        from helpers import make_oscillating_data

        from backtesting_engine.execution import ExecutionConfig, cost_sensitivity_sweep
        from backtesting_engine.strategy.moving_average import MovingAverageStrategy
        from backtesting_engine.walk_forward import walk_forward

        data = make_oscillating_data(756, with_high_low=True)

        direct = walk_forward(
            data, MovingAverageStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=ExecutionConfig(
                transaction_cost_rate=0.001,
                slippage_factor=0.05,
                signal_delay=0,
            ),
            bootstrap_seed=42,
        )
        direct_p = direct.summary_metrics.combined_p_value

        sweep = cost_sensitivity_sweep(
            data, MovingAverageStrategy(),
            cost_rates=[0.001],
            slippage_factors=[0.05],
            training_window_years=1,
            testing_window_years=1,
            signal_delay=0,
            bootstrap_seed=42,
        )
        sweep_p = sweep[(0.001, 0.05)]

        assert not math.isnan(sweep_p), "sweep returned NaN p-value"
        assert abs(sweep_p - direct_p) < 0.001, (
            f"sweep p={sweep_p:.4f} does not match direct walk_forward p={direct_p:.4f}. "
            "signal_delay is probably not reaching _sweep_worker correctly."
        )
