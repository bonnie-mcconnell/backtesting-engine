"""
Data ingestion module responsible for downloading historical stock data using yfinance. 
This module fetches data for a given ticker and date range, returning it as a pandas DataFrame. 
"""
import yfinance as yf
import pandas as pd

def download_data(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Download historical stock data using yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g 'SPY').
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date computed inside function as current date.
    
    Returns:
        DataFrame containing DatetimeIndex and 'close' price column containing adjusted closing prices.

    Raises:
        ValueError: If no data is found for the given ticker and date range.
    """
    end_date = pd.Timestamp.now()
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data is None or data.empty:
        raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}.")
    data.columns = data.columns.get_level_values(0)  # flatten MultiIndex
    data = data[['Adj Close']].rename(columns={'Adj Close': 'close'})
    return data