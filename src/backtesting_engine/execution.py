"""
Realistic execution model with slippage, signal delay, and cost sensitivity.

The gap between backtested and live performance is primarily an execution
problem. Fill-at-close with a fixed percentage fee is the best-case scenario.
Real execution has three additional frictions:

1. Slippage
   You rarely fill at exactly the close. Market orders fill somewhere in the
   bid-ask spread; the larger your order relative to average volume, the
   more the price moves against you before it fills. We model slippage as a
   fraction of the day's high-low range:

       fill_price_buy  = close + slippage_factor × (high - low)
       fill_price_sell = close - slippage_factor × (high - low)

   This is a standard approximation used in academic backtesting literature
   (e.g. Lesmond, Ogden & Trzcinka, 1999). slippage_factor = 0 recovers
   the fill-at-close model. A value of 0.1 means you fill at a price 10%
   of the daily range away from close, which is conservative for liquid
   ETFs like SPY.

2. Signal delay
   Real strategies cannot act on a signal at the same bar that generated it.
   The close price is not known until after market close; by then it is too
   late to act on that bar's close. With delay=1, the signal fired on day t
   results in a fill at day t+1's open (approximated here as t+1's close).
   This is a common and important realism check - many strategies that look
   good at close prices become marginal or negative with a one-day delay.

3. Cost sensitivity analysis
   We sweep over (transaction_cost_rate, slippage_factor) grids and compute
   the full walk-forward Fisher p-value at each point. The result is a 2D
   heatmap showing at what cost level the strategy loses statistical
   significance. The breakeven cost is the most important single number for
   deciding whether a backtested strategy is worth pursuing live.

ExecutionConfig
---------------
A dataclass holding all execution parameters. Passed to run_simulation()
instead of using global config constants, allowing cost sensitivity sweeps
without touching config.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting_engine.config import (
    INITIAL_PORTFOLIO_VALUE,
    POSITION_SIZE_FRACTION,
    TRANSACTION_COST_RATE,
)
from backtesting_engine.models import SimulationResult, Trade

_VALID_SIGNALS = frozenset({-1, 0, 1})


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Parameters controlling trade execution realism.

    Attributes
    ----------
    transaction_cost_rate : float
        Proportional fee per side (e.g. 0.001 = 0.1%). Applied to both
        entry and exit. Default matches the engine-wide config.
    slippage_factor : float
        Fraction of the daily high-low range added to buy fills and
        subtracted from sell fills. 0.0 = fill at close (no slippage).
        0.1 = 10% of the daily range. Default 0.0 for backward compatibility.
    signal_delay : int
        Number of bars to delay signal execution. 0 = fill at signal bar's
        close. 1 = fill at next bar's close (standard one-day delay).
        Default 0 for backward compatibility.
    """
    transaction_cost_rate: float = TRANSACTION_COST_RATE
    slippage_factor: float = 0.0
    signal_delay: int = 0

    def __post_init__(self) -> None:
        if self.transaction_cost_rate < 0:
            raise ValueError("transaction_cost_rate must be non-negative.")
        if self.slippage_factor < 0:
            raise ValueError("slippage_factor must be non-negative.")
        if self.signal_delay < 0:
            raise ValueError("signal_delay must be non-negative.")


def run_simulation_with_execution(
    data: pd.DataFrame,
    signals: pd.Series,
    execution: ExecutionConfig | None = None,
) -> SimulationResult:
    """
    Simulate trade execution with configurable execution realism.

    This is a drop-in replacement for run_simulation() that accepts an
    ExecutionConfig. With default ExecutionConfig, behaviour is identical
    to run_simulation().

    OHLCV data is required when slippage_factor > 0 (needs 'high' and 'low'
    columns). If only 'close' is present and slippage_factor > 0, a
    ValueError is raised.

    Args:
        data: OHLCV DataFrame with DatetimeIndex. Must have 'close'.
              If slippage_factor > 0, must also have 'high' and 'low'.
        signals: Integer signal series with values in {-1, 0, 1}.
        execution: ExecutionConfig instance. Defaults to zero slippage,
                   zero delay, standard transaction costs.

    Returns:
        SimulationResult with executed trades and portfolio value series.

    Raises:
        ValueError: On mismatched lengths, invalid signals, or missing
                    columns required by the execution config.
    """
    if execution is None:
        execution = ExecutionConfig()

    if len(data) != len(signals):
        raise ValueError(
            f"Data length {len(data)} does not match signals length {len(signals)}."
        )

    unique_signals = set(signals.unique())
    invalid = unique_signals - _VALID_SIGNALS
    if invalid:
        raise ValueError(
            f"signals contains invalid values {invalid}. Only {{-1, 0, 1}} are permitted."
        )

    if execution.slippage_factor > 0:
        missing = [c for c in ("high", "low") if c not in data.columns]
        if missing:
            raise ValueError(
                f"Slippage model requires 'high' and 'low' columns, "
                f"but {missing} are missing from data."
            )

    close = data["close"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float) if "high" in data.columns else close
    low = data["low"].to_numpy(dtype=float) if "low" in data.columns else close

    # Apply signal delay: shift signals forward by signal_delay bars.
    # Signals that fall off the end are dropped (never executed).
    if execution.signal_delay > 0:
        delay = execution.signal_delay
        delayed_values = np.zeros(len(signals), dtype=int)
        delayed_values[delay:] = signals.to_numpy()[:-delay]
        signals = pd.Series(delayed_values, index=signals.index)

    cash: float = INITIAL_PORTFOLIO_VALUE
    shares_held: float = 0.0
    entry_price: float | None = None
    entry_date: pd.Timestamp | None = None

    trades: list[Trade] = []
    portfolio_values: list[float] = []

    slippage = execution.slippage_factor
    cost_rate = execution.transaction_cost_rate

    for idx, (date, signal) in enumerate(signals.items()):
        date = pd.Timestamp(str(date))
        daily_range = high[idx] - low[idx]

        # Fill prices with slippage: buys fill above close, sells below.
        buy_fill = close[idx] + slippage * daily_range
        sell_fill = close[idx] - slippage * daily_range

        if shares_held == 0.0 and signal == 1:
            position_value = cash * POSITION_SIZE_FRACTION
            buy_cost = position_value * cost_rate
            shares_held = position_value / buy_fill
            cash -= position_value + buy_cost
            entry_price = buy_fill
            entry_date = date

        elif shares_held > 0.0 and signal == -1:
            assert entry_price is not None and entry_date is not None
            sell_proceeds = shares_held * sell_fill
            sell_cost = sell_proceeds * cost_rate
            buy_cost = shares_held * entry_price * cost_rate
            pnl = (sell_proceeds - sell_cost) - (shares_held * entry_price + buy_cost)

            trades.append(Trade(
                entry_date=entry_date,
                exit_date=date,
                entry_price=entry_price,
                exit_price=sell_fill,
                shares=shares_held,
                transaction_costs=buy_cost + sell_cost,
                pnl=pnl,
            ))
            cash += sell_proceeds - sell_cost
            shares_held = 0.0
            entry_price = None
            entry_date = None

        portfolio_values.append(cash + shares_held * close[idx])

    # Force-close any open position at end of window.
    if shares_held > 0.0:
        assert entry_price is not None and entry_date is not None
        sell_fill_final = close[-1] - slippage * (high[-1] - low[-1])
        sell_proceeds = shares_held * sell_fill_final
        sell_cost = sell_proceeds * cost_rate
        buy_cost = shares_held * entry_price * cost_rate
        pnl = (sell_proceeds - sell_cost) - (shares_held * entry_price + buy_cost)

        trades.append(Trade(
            entry_date=entry_date,
            exit_date=pd.Timestamp(str(signals.index[-1])),
            entry_price=entry_price,
            exit_price=sell_fill_final,
            shares=shares_held,
            transaction_costs=buy_cost + sell_cost,
            pnl=pnl,
        ))
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


def cost_sensitivity_sweep(
    data: pd.DataFrame,
    strategy: object,
    cost_rates: list[float],
    slippage_factors: list[float],
    training_window_years: int = 3,
    testing_window_years: int = 1,
) -> dict[tuple[float, float], float]:
    """
    Sweep over (cost_rate, slippage_factor) pairs and return Fisher p-values.

    Runs a full walk-forward analysis at each combination. The result maps
    each (cost_rate, slippage) pair to the Fisher combined p-value, which
    can be visualised as a heatmap to identify the breakeven cost level -
    the point at which the strategy loses statistical significance.

    This is computationally intensive: O(len(cost_rates) × len(slippage_factors)
    × n_windows × n_bootstrap) operations. For a typical SPY run with a 5×5
    grid, this takes ~15 minutes.

    Args:
        data: Full historical OHLCV DataFrame. Must include 'close'.
              'high' and 'low' are required for slippage_factor > 0.
        strategy: Any BaseStrategy implementation.
        cost_rates: List of transaction cost rates to test (e.g. [0.0005, 0.001]).
        slippage_factors: List of slippage fractions to test (e.g. [0.0, 0.05]).
        training_window_years: Years per training window.
        testing_window_years: Years per test window.

    Returns:
        Dict mapping (cost_rate, slippage_factor) → Fisher p-value.
    """
    from backtesting_engine.walk_forward import walk_forward  # avoid circular import

    results: dict[tuple[float, float], float] = {}
    len(cost_rates) * len(slippage_factors)
    done = 0

    for cost in cost_rates:
        for slip in slippage_factors:
            exec_config = ExecutionConfig(
                transaction_cost_rate=cost,
                slippage_factor=slip,
                signal_delay=1,  # always use 1-day delay for realism
            )
            try:
                result = walk_forward(
                    data,
                    strategy,  # type: ignore[arg-type]
                    training_window_years=training_window_years,
                    testing_window_years=testing_window_years,
                    execution=exec_config,
                )
                results[(cost, slip)] = result.summary_metrics.combined_p_value
            except ValueError:
                results[(cost, slip)] = float("nan")
            done += 1

    return results