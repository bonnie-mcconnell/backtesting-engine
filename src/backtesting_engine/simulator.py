"""
Trade execution simulator - baseline, cost-only, no slippage or signal delay.

This module provides `run_simulation()`: a readable, bar-by-bar implementation
of the core execution loop used for unit testing and as a transparent reference.

Relationship to `execution.py`
-------------------------------
`run_simulation_with_execution()` in execution.py is the production path used
by all walk-forward runs. It extends this baseline with:
  - Configurable slippage (fill at close ± fraction × daily range)
  - Configurable signal delay (shift signals forward N bars)
  - `ExecutionConfig` dataclass for per-run parameter control

`run_simulation()` here is the zero-slippage, zero-delay baseline. It always
uses the global `TRANSACTION_COST_RATE` and fills at the signal bar's close
price. This is the "best case" execution model.

When to use which:
  - `run_simulation()`: unit tests that verify execution logic on synthetic
    close-only data, and the backward-compatibility test in test_execution.py.
  - `run_simulation_with_execution()`: everything else, including all walk-forward
    runs. Always pass an explicit `ExecutionConfig` to document your assumptions.

Execution model (applies to both):
  - Signals processed bar-by-bar in chronological order.
  - Buy signal when flat → open long at that bar's closing price.
  - Sell signal when long → close position at that bar's closing price.
  - Only one position at a time (no pyramiding, no short selling).
  - Transaction costs applied symmetrically on entry and exit.
  - Position still open at end of window → force-closed at final bar's close.
  - Position sizing: `position_value = cash × fraction / (1 + cost_rate)`
    so that entry spend + fee exactly equals available cash (no leverage).

The loop is intentionally explicit rather than vectorised. Transparency over
speed. For daily data on a single asset over 30 years the runtime is fine.
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
            # Size so cost-inclusive spend fits within available cash:
            #   position_value = cash * POSITION_SIZE_FRACTION / (1 + cost_rate)
            available = cash * POSITION_SIZE_FRACTION
            position_value = available / (1.0 + TRANSACTION_COST_RATE)
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
