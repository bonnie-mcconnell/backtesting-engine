"""
Structural validation for backtesting input data.

Validates that a DataFrame meets the structural and content requirements
before it enters the pipeline. Does not transform data - only checks and raises.

The minimum row check is parameterised rather than importing a strategy-specific
constant. The caller (typically the orchestrator or main.py) is responsible for
passing the appropriate minimum for the strategy in use.
"""

import pandas as pd


def validate_data(data: pd.DataFrame, min_rows: int = 1) -> None:
    """
    Validate input data for backtesting.

    Checks:
      - Index is a DatetimeIndex (required for time-series alignment).
      - Index has no duplicate timestamps (duplicate dates cause silent index bugs).
      - Index is monotonically increasing (non-chronological data breaks rolling windows).
      - 'close' column is present.
      - 'close' has no missing values (NaN propagates silently through metrics).
      - All 'close' values are strictly positive (non-positive prices are data errors).
      - DataFrame has at least min_rows rows (caller sets this based on strategy needs).

    Args:
        data: DataFrame to validate.
        min_rows: Minimum number of rows required. Defaults to 1 (no minimum).
                  Pass the strategy's lookback period to enforce sufficiency.

    Raises:
        ValueError: On any failed check, with a message describing the problem.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            f"Data index must be a DatetimeIndex, got {type(data.index).__name__}."
        )
    if data.index.has_duplicates:
        n_dupes = data.index.duplicated().sum()
        raise ValueError(
            f"Data index contains {n_dupes} duplicate timestamp(s). "
            "Each date must appear exactly once."
        )
    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Data index is not monotonically increasing. "
            "Dates must be in chronological order."
        )
    if "close" not in data.columns:
        raise ValueError(
            f"Data must contain a 'close' column. Found columns: {list(data.columns)}."
        )
    if data.empty:
        raise ValueError("Data is empty.")

    n_null = int(data["close"].isnull().sum())
    if n_null > 0:
        raise ValueError(
            f"'close' column contains {n_null} missing value(s). "
            "All rows must have a valid closing price."
        )
    n_nonpositive = int((data["close"] <= 0).sum())
    if n_nonpositive > 0:
        raise ValueError(
            f"'close' column contains {n_nonpositive} zero or negative value(s). "
            "All closing prices must be strictly positive."
        )
    if len(data) < min_rows:
        raise ValueError(
            f"Data has {len(data)} rows but at least {min_rows} are required. "
            "Provide a longer price history."
        )
