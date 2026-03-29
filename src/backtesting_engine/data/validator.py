"""
Data validation functions for backtesting input data.
Doesn't transform data, only checks for required structure and content.
"""
import pandas as pd

from backtesting_engine.config import MOVING_AVERAGE_LONG_DAYS


def validate_data(data: pd.DataFrame) -> None:
    """
    Validate the input data for backtesting.
    
    Checks:
    - Index is a DatetimeIndex with no duplicates and is monotonically increasing.
    - DataFrame is not empty.
    - Contains 'close' column.
    - 'close' column has no missing values, is not empty, and contains no negative values.
    - DataFrame has sufficient rows for the moving average calculation.
    
    Args:
        data: DataFrame to validate.
    Raises:
        ValueError: If any validation check fails.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")
    if data.index.has_duplicates:
        raise ValueError("Data index contains duplicate timestamps.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("Data index must be monotonically increasing (no out-of-order dates).")
    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column.")
    if len(data) < MOVING_AVERAGE_LONG_DAYS:
        raise ValueError(f"Data contains {len(data)} rows, minimum required is {MOVING_AVERAGE_LONG_DAYS}.")
    if data['close'].isnull().any():
        raise ValueError("'close' column contains missing values.")
    if (data['close'] <= 0).any():
        raise ValueError("'close' column contains zero or negative values.")