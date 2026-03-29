"""
Moving average crossover strategy implementation.
Generates buy/sell signals based on the golden cross (50/200-day MA crossover).
"""
import pandas as pd

from backtesting_engine.strategy.base import Strategy
from backtesting_engine.config import MOVING_AVERAGE_LONG_DAYS, MOVING_AVERAGE_SHORT_DAYS


class MovingAverageStrategy(Strategy):
    """
    Implements a simple moving average crossover strategy.
    Generates buy signals when the short-term moving average crosses above the long-term moving average,
    and sell signals when it crosses below.
    Crossover signal fires after the price move has already begun,
    as moving averages are lagging indicators (computed from past prices).
    This is a known trade-off for this strategy.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on moving average crossover strategy.
        
        Buy signal (1) when 50 day MA crosses above 200 day MA.
        Sell signal (-1) when 50 day MA crosses below 200 day MA.
        Hold signal (0) otherwise.
        
        Args:
            data: DataFrame with 'close' prices and DatetimeIndex.
        
        Returns:
            Series of signals aligned to data's index: 1 (buy), -1 (sell), 0 (hold).
        """
        short_ma = data['close'].rolling(MOVING_AVERAGE_SHORT_DAYS).mean() 
        long_ma = data['close'].rolling(MOVING_AVERAGE_LONG_DAYS).mean() 
        
        position = (short_ma > long_ma)      # True where short is above long
        
        # 1 = crossed up, -1 = crossed down, 0 = no change
        signals = position.astype(int).diff().fillna(0) 
        
        return signals.astype(int)