"""
Unit tests for data ingestion module.
The network call to yfinance is mocked to keep tests fast and deterministic.
"""
import pandas as pd
import pytest
from unittest.mock import patch

from backtesting_engine.data.ingestion import load_data


def test_load_data_returns_close_column() -> None:
    # Create a minimal fake DataFrame that yfinance would return
    import numpy as np
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    fake_data = pd.DataFrame({"Adj Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)
    fake_data.columns = pd.MultiIndex.from_tuples([("Adj Close", "SPY")])

    with patch("yfinance.download", return_value=fake_data):
        result = load_data("SPY", "2020-01-01")

    assert "close" in result.columns
    assert isinstance(result.index, pd.DatetimeIndex)


def test_load_data_raises_on_empty_response() -> None:
    with patch("yfinance.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError):
            load_data("INVALID", "2020-01-01")
