"""
Data ingestion: download OHLCV data with adjusted closing prices via yfinance.

Returns a DataFrame with a DatetimeIndex and columns:
  close  - split- and dividend-adjusted closing price (required by all strategies)
  high   - unadjusted daily high  (required by execution model for slippage)
  low    - unadjusted daily low   (required by execution model for slippage)
  volume - daily volume           (available for future volume-based strategies)

Why adjusted close but unadjusted high/low?
Adjusted closing prices are necessary for accurate return computation - without
adjustment, dividends and splits create artificial price gaps. Intraday high/low
are used only for slippage estimation (fill = close ± factor × range), where
absolute levels matter less than the range itself.
"""

import pandas as pd
import yfinance as yf


def load_data(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker.

    Args:
        ticker: Exchange ticker symbol (e.g. 'SPY', 'AAPL').
        start_date: Start date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with DatetimeIndex and columns:
            close  - adjusted closing price (float64)
            high   - unadjusted daily high  (float64)
            low    - unadjusted daily low   (float64)
            volume - daily share volume     (float64, if available)

    Raises:
        ValueError: If yfinance returns no data for the given ticker/range.
    """
    end_date = pd.Timestamp.now()
    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' "
            f"between {start_date} and {end_date.date()}. "
            "Check that the ticker is valid and the date range has trading activity."
        )

    # yfinance returns MultiIndex columns (field, ticker) with auto_adjust=False.
    raw.columns = raw.columns.get_level_values(0)

    available = set(raw.columns)
    required = {"Adj Close", "High", "Low"}
    missing = required - available
    if missing:
        raise ValueError(
            f"yfinance response missing expected columns: {missing}. "
            f"Available: {sorted(available)}."
        )

    data = raw[["Adj Close", "High", "Low"]].rename(columns={
        "Adj Close": "close",
        "High": "high",
        "Low": "low",
    })

    if "Volume" in available:
        data["volume"] = raw["Volume"]

    data.index = pd.DatetimeIndex(data.index)
    return data