"""
Trade execution simulator (baseline, no slippage or signal delay).

Processes a price DataFrame and a signal Series to produce a SimulationResult:
the list of executed trades and the daily portfolio value series.

This is the original, cost-only simulator. For walk-forward runs, prefer
run_simulation_with_execution() from execution.py, which adds slippage,
signal delay, and a configurable ExecutionConfig. This module is retained
because test_simulator.py exercises it directly and its explicit bar-by-bar
loop makes the execution model easy to read and reason about line by line.

Execution model:
  - Signals are processed bar-by-bar in chronological order.
  - A buy signal when flat opens a long position at that bar's closing price.
  - A sell signal when long closes the position at that bar's closing price.
  - Only one position can be held at a time (no pyramiding, no short selling).
  - Transaction costs are applied symmetrically on entry and exit.
  - Any position still open at the end of the window is closed at the final bar.

The loop is intentionally explicit rather than vectorised. This makes the
execution logic transparent and line-by-line testable, at the cost of speed.
For daily data over a single asset across decades, the performance is fine.
"""

from dataclasses import dataclass

import pandas as pd

from backtesting_engine.config import (
    INITIAL_PORTFOLIO_VALUE,
    POSITION_SIZE_FRACTION,
    TRANSACTION_COST_RATE,
)
from backtesting_engine.models import SimulationResult, Trade

# Valid signal values. Anything outside this set is a caller contract violation.
_VALID_SIGNALS = frozenset({-1, 0, 1})


@dataclass
class _OpenPosition:
    """
    State for a single open long position.

    Grouping entry_price, entry_date, and shares into one object means the
    type system enforces that you cannot hold one field without the others.
    This eliminates the assert-based None-narrowing pattern that breaks
    silently when Python is run with the -O (optimise) flag.
    """
    entry_price: float
    entry_date: pd.Timestamp
    shares: float


def run_simulation(data: pd.DataFrame, signals: pd.Series) -> SimulationResult:
    """
    Simulate trade execution from price data and strategy signals.

    Args:
        data: Historical market data with DatetimeIndex and 'close' column.
              Must be aligned with signals (same index).
        signals: Integer signal series with values in {-1, 0, 1}.
                 1 = buy, -1 = sell, 0 = hold.

    Returns:
        SimulationResult with executed trades, daily portfolio value series,
        and a status message if no trades were executed.

    Raises:
        ValueError: If data and signals have different lengths.
        ValueError: If signals contains values outside {-1, 0, 1}.
    """
    if len(data) != len(signals):
        raise ValueError(
            f"Data length {len(data)} does not match signals length {len(signals)}."
        )

    unique_signals = set(signals.unique())
    invalid = unique_signals - _VALID_SIGNALS
    if invalid:
        raise ValueError(
            f"signals contains invalid values {invalid}. "
            f"Only {{-1, 0, 1}} are permitted."
        )

    close_prices = data["close"].to_numpy(dtype=float)

    cash: float = INITIAL_PORTFOLIO_VALUE
    position: _OpenPosition | None = None

    trades: list[Trade] = []
    portfolio_values: list[float] = []

    for idx, (date, signal) in enumerate(signals.items()):
        current_price = close_prices[idx]
        date = pd.Timestamp(str(date))

        if position is None and signal == 1:
            # Open long position.
            position_value = cash * POSITION_SIZE_FRACTION
            buy_cost = position_value * TRANSACTION_COST_RATE
            shares = position_value / current_price
            cash -= position_value + buy_cost
            position = _OpenPosition(
                entry_price=current_price,
                entry_date=date,
                shares=shares,
            )

        elif position is not None and signal == -1:
            # Close long position.
            sell_proceeds = position.shares * current_price
            sell_cost = sell_proceeds * TRANSACTION_COST_RATE
            buy_cost = position.shares * position.entry_price * TRANSACTION_COST_RATE
            pnl = (sell_proceeds - sell_cost) - (
                position.shares * position.entry_price + buy_cost
            )
            trades.append(Trade(
                entry_date=position.entry_date,
                exit_date=date,
                entry_price=position.entry_price,
                exit_price=current_price,
                shares=position.shares,
                transaction_costs=buy_cost + sell_cost,
                pnl=pnl,
            ))
            cash += sell_proceeds - sell_cost
            position = None

        # Record portfolio value at end of this bar.
        shares_held = position.shares if position is not None else 0.0
        portfolio_values.append(cash + shares_held * current_price)

    # Force-close any position still open at end of window.
    if position is not None:
        final_price = close_prices[-1]
        final_date = pd.Timestamp(str(signals.index[-1]))

        sell_proceeds = position.shares * final_price
        sell_cost = sell_proceeds * TRANSACTION_COST_RATE
        buy_cost = position.shares * position.entry_price * TRANSACTION_COST_RATE
        pnl = (sell_proceeds - sell_cost) - (
            position.shares * position.entry_price + buy_cost
        )
        trades.append(Trade(
            entry_date=position.entry_date,
            exit_date=final_date,
            entry_price=position.entry_price,
            exit_price=final_price,
            shares=position.shares,
            transaction_costs=buy_cost + sell_cost,
            pnl=pnl,
        ))
        # Update the final portfolio value to reflect the closing costs.
        cash += sell_proceeds - sell_cost
        portfolio_values[-1] = cash

    portfolio_series = pd.Series(portfolio_values, index=signals.index, dtype=float)

    if not trades:
        return SimulationResult(
            trades=[],
            portfolio_values=portfolio_series,
            message="No trades executed.",
        )

    return SimulationResult(trades=trades, portfolio_values=portfolio_series, message="")
