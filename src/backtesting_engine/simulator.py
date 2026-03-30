"""
Simulator module for backtesting engine.
The run_simulation function processes the trading signals, executes trades according to the defined strategy, 
and calculates the resulting profit or loss for each trade. 
It also tracks the portfolio value over time, accounting for transaction costs and position sizing. 
The results are returned in a structured SimulationResult object for analysis.
"""
import pandas as pd

from backtesting_engine.config import INITIAL_PORTFOLIO_VALUE, POSITION_SIZE_FRACTION, TRANSACTION_COST_RATE
from backtesting_engine.models import Trade, SimulationResult 


def run_simulation(data: pd.DataFrame, signals: pd.Series) -> SimulationResult:
    """
    Simulates trade execution from price data and strategy signals.
    Calculates net profit or loss for each trade and aggregates results.

    Args:
        data (pd.DataFrame): The historical market data.
        signals (pd.Series): A series of trading signals.

    Returns:
        SimulationResult containing executed trades, daily portfolio values, and a status message.
    """
    cash = INITIAL_PORTFOLIO_VALUE

    shares_held = 0
    portfolio_values = []

    entry_price: float | None = None
    entry_date: pd.Timestamp | None = None
    trades = []

    if len(data) != len(signals):
            raise ValueError(f"Data length {len(data)} does not match signals length {len(signals)}.")

    close_prices = data['close'].to_numpy()

    for idx, (date, signal) in enumerate(signals.items()):
        date = pd.Timestamp(str(date))
        current_price = float(close_prices[idx])

        if shares_held == 0 and signal == 1:  # Buy signal
            position_value = cash * POSITION_SIZE_FRACTION
            buy_cost = position_value * TRANSACTION_COST_RATE
            shares_held = position_value / current_price
            cash -= (position_value + buy_cost)
            entry_price = current_price
            entry_date = date

        elif shares_held > 0 and signal == -1:  # Sell signal
            if entry_price is None or entry_date is None:
                continue
            sell_proceeds = shares_held * current_price
            sell_cost = sell_proceeds * TRANSACTION_COST_RATE
            buy_cost = shares_held * entry_price * TRANSACTION_COST_RATE
            pnl = (sell_proceeds - sell_cost) - (shares_held * entry_price + buy_cost)
            trade = Trade(
                entry_date=entry_date,
                exit_date=date,
                entry_price=entry_price,
                exit_price=current_price,
                shares=shares_held,
                transaction_costs=buy_cost + sell_cost,
                pnl=pnl
            )
            trades.append(trade)
            cash += (sell_proceeds - sell_cost)

            shares_held = 0
            entry_price = None
            entry_date = None
        
        portfolio_values.append(cash + (shares_held * current_price))
    
    portfolio_series = pd.Series(portfolio_values, index=signals.index)

    if not trades:
        return SimulationResult(trades=[], portfolio_values=portfolio_series, message="No trades executed.")

    return SimulationResult(trades=trades, portfolio_values=portfolio_series, message="")