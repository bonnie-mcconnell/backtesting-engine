"""
Realistic execution model with slippage, signal delay, and cost sensitivity.

The gap between backtested and live performance is mostly an execution problem.
Fill-at-close with a fixed fee is the best-case scenario. Real execution adds:

1. Slippage - market orders rarely fill at exactly the close. Modelled as a
   fraction of the day's high-low range:
       fill_buy  = close + slippage_factor × (high - low)
       fill_sell = close - slippage_factor × (high - low)
   slippage_factor=0 recovers fill-at-close. 0.05 (the default) is conservative
   for liquid ETFs like SPY. Reference: Lesmond, Ogden & Trzcinka (1999).

2. Signal delay - the close isn't known until after market close; you can't act
   on it the same bar. With delay=1 the signal fires on day t and fills at
   day t+1's close (approximating a t+1 open fill). Many strategies that look
   good on close prices go marginal with a one-day delay.

3. Cost sensitivity sweep - runs a full walk-forward at each (cost, slippage)
   grid point and returns a 2D heatmap of Fisher p-values. The breakeven cost
   level is the most important single number for deciding whether to pursue
   a strategy live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backtesting_engine.config import (
    BLOCK_BOOTSTRAP_SEED,
    INITIAL_PORTFOLIO_VALUE,
    POSITION_SIZE_FRACTION,
    TRANSACTION_COST_RATE,
)
from backtesting_engine.models import SimulationResult, Trade

if TYPE_CHECKING:
    # TYPE_CHECKING is False at runtime, so this import doesn't create a cycle.
    # execution.py is imported by strategy/__init__.py; importing strategy.base
    # unconditionally would create a circular dependency.
    from backtesting_engine.strategy.base import BaseStrategy

_VALID_SIGNALS = frozenset({-1, 0, 1})


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Parameters controlling trade execution realism.

    Defaults: cost=0.1%/side, slippage=5% of daily range, delay=1 bar.
    These are intentionally conservative so out-of-the-box results reflect
    realistic execution, not fill-at-close.

    For zero-friction comparison (e.g. to verify strategy logic in isolation):
        ExecutionConfig(transaction_cost_rate=0, slippage_factor=0, signal_delay=0)
    """
    transaction_cost_rate: float = TRANSACTION_COST_RATE   # 0.001 = 0.1% per side
    slippage_factor: float = 0.05    # fraction of daily high-low range; 0 = fill at close
    signal_delay: int = 1            # bars; 1 prevents lookahead bias

    def __post_init__(self) -> None:
        if self.transaction_cost_rate < 0:
            raise ValueError("transaction_cost_rate must be non-negative.")
        if self.slippage_factor < 0:
            raise ValueError("slippage_factor must be non-negative.")
        if self.signal_delay < 0:
            raise ValueError("signal_delay must be non-negative.")


@dataclass
class _OpenPosition:
    """State for a single open long position.

    A dataclass rather than loose variables so the type system enforces that
    entry_price and entry_date are always set together.
    """
    entry_price: float
    entry_date: pd.Timestamp
    shares: float


def run_simulation_with_execution(
    data: pd.DataFrame,
    signals: pd.Series,
    execution: ExecutionConfig | None = None,
) -> SimulationResult:
    """
    Simulate trade execution with configurable execution realism.

    Args:
        data: OHLCV DataFrame with DatetimeIndex. Must have 'close'.
              'high' and 'low' required when slippage_factor > 0.
        signals: Integer signal series with values in {-1, 0, 1}.
        execution: ExecutionConfig instance. Defaults to cost=0.1%,
                   slippage=5% of daily range, delay=1 bar.

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
    low  = data["low"].to_numpy(dtype=float)  if "low"  in data.columns else close

    if execution.signal_delay > 0:
        delay = execution.signal_delay
        delayed_values = np.zeros(len(signals), dtype=int)
        delayed_values[delay:] = signals.to_numpy()[:-delay]
        signals = pd.Series(delayed_values, index=signals.index)

    cash: float = INITIAL_PORTFOLIO_VALUE
    position: _OpenPosition | None = None

    trades: list[Trade] = []
    portfolio_values: list[float] = []

    slippage  = execution.slippage_factor
    cost_rate = execution.transaction_cost_rate

    for idx, (date, signal) in enumerate(signals.items()):
        date = pd.Timestamp(str(date))
        daily_range = high[idx] - low[idx]

        buy_fill  = close[idx] + slippage * daily_range
        sell_fill = close[idx] - slippage * daily_range

        if position is None and signal == 1:
            # Cost-inclusive sizing: position_value × (1 + cost_rate) = cash × fraction
            # so that position_value + buy_cost fits exactly within available cash.
            # The naive formula (cash * fraction, then subtract cost) creates a small
            # negative cash balance after every trade.
            available = cash * POSITION_SIZE_FRACTION
            position_value = available / (1.0 + cost_rate)
            buy_cost = position_value * cost_rate
            shares = position_value / buy_fill
            cash -= position_value + buy_cost
            position = _OpenPosition(
                entry_price=buy_fill,
                entry_date=date,
                shares=shares,
            )

        elif position is not None and signal == -1:
            sell_proceeds = position.shares * sell_fill
            sell_cost = sell_proceeds * cost_rate
            buy_cost  = position.shares * position.entry_price * cost_rate
            pnl = (
                (sell_proceeds - sell_cost)
                - (position.shares * position.entry_price + buy_cost)
            )
            trades.append(Trade(
                entry_date=position.entry_date,
                exit_date=date,
                entry_price=position.entry_price,
                exit_price=sell_fill,
                shares=position.shares,
                transaction_costs=buy_cost + sell_cost,
                pnl=pnl,
            ))
            cash += sell_proceeds - sell_cost
            position = None

        shares_held = position.shares if position is not None else 0.0
        portfolio_values.append(cash + shares_held * close[idx])

    # force-close any open position at window end
    if position is not None:
        sell_fill_final = close[-1] - slippage * (high[-1] - low[-1])
        sell_proceeds = position.shares * sell_fill_final
        sell_cost = sell_proceeds * cost_rate
        buy_cost  = position.shares * position.entry_price * cost_rate
        pnl = (
            (sell_proceeds - sell_cost)
            - (position.shares * position.entry_price + buy_cost)
        )
        trades.append(Trade(
            entry_date=position.entry_date,
            exit_date=pd.Timestamp(str(signals.index[-1])),
            entry_price=position.entry_price,
            exit_price=sell_fill_final,
            shares=position.shares,
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


def _sweep_worker(
    args: tuple[
        tuple[float, float],
        pd.DataFrame,
        str,
        int,
        int,
        int,
        int,
    ],
) -> tuple[tuple[float, float], float]:
    """
    Module-level worker for ProcessPoolExecutor.

    Has to be at module scope to be picklable on Windows (spawn-based).
    Strategy reconstructed by class name to avoid pickling scipy optimizer
    state from KalmanFilterStrategy.
    """
    from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
    from backtesting_engine.strategy.momentum import MomentumStrategy
    from backtesting_engine.strategy.moving_average import MovingAverageStrategy
    from backtesting_engine.walk_forward import walk_forward

    (cost, slip), data, strategy_name, train_yrs, test_yrs, seed, signal_delay = args

    _strategy_map = {
        "MovingAverageStrategy": MovingAverageStrategy,
        "KalmanFilterStrategy":  KalmanFilterStrategy,
        "MomentumStrategy":      MomentumStrategy,
    }
    strategy_cls = _strategy_map.get(strategy_name)
    if strategy_cls is None:
        return (cost, slip), float("nan")

    exec_config = ExecutionConfig(
        transaction_cost_rate=cost,
        slippage_factor=slip,
        signal_delay=signal_delay,
    )
    try:
        result = walk_forward(
            data,
            strategy_cls(),
            training_window_years=train_yrs,
            testing_window_years=test_yrs,
            execution=exec_config,
            bootstrap_seed=seed,
        )
        return (cost, slip), result.summary_metrics.combined_p_value
    except ValueError:
        return (cost, slip), float("nan")


def cost_sensitivity_sweep(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    cost_rates: list[float],
    slippage_factors: list[float],
    training_window_years: int = 3,
    testing_window_years: int = 1,
    n_workers: int = 1,
    bootstrap_seed: int = BLOCK_BOOTSTRAP_SEED,
    signal_delay: int = 1,
) -> dict[tuple[float, float], float]:
    """
    Sweep over (cost_rate, slippage_factor) pairs and return Fisher p-values.

    Each combination is fully independent so the sweep parallelises trivially.
    Set n_workers > 1 for multiple CPU cores; -1 uses all available.
    On an 8-core machine a 5×5 grid that takes ~12 minutes serially finishes
    in ~2 minutes.

    Returns:
        Dict mapping (cost_rate, slippage_factor) → Fisher p-value.
        NaN if walk_forward raised for that combination.
    """
    # deferred imports - these are only needed when a sweep is actually run,
    # and keeping them here makes it obvious this is the only parallel entry point
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if n_workers == -1:
        n_workers = os.cpu_count() or 1

    strategy_name = strategy.__class__.__name__
    execution_signal_delay = signal_delay
    pairs = [(c, s) for c in cost_rates for s in slippage_factors]
    worker_args = [
        ((c, s), data, strategy_name, training_window_years, testing_window_years,
         bootstrap_seed, execution_signal_delay)
        for c, s in pairs
    ]

    results: dict[tuple[float, float], float] = {}

    if n_workers == 1:
        for wargs in worker_args:
            key, p_val = _sweep_worker(wargs)
            results[key] = p_val
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_sweep_worker, wargs): wargs for wargs in worker_args}
            for future in as_completed(futures):
                key, p_val = future.result()
                results[key] = p_val

    return results
