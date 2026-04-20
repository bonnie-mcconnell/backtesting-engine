"""
Unit tests for the data ingestion module.

The network call to yfinance is mocked in all tests to keep the suite
fast, deterministic, and free of external dependencies.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from backtesting_engine.data.ingestion import load_data


def _fake_download(ticker: str = "SPY") -> pd.DataFrame:
    """
    Minimal fake yfinance response with MultiIndex columns.

    Mirrors the structure yfinance returns with auto_adjust=False:
    columns are (field, ticker) tuples.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            ("Adj Close", ticker): [100.0, 101.0, 102.0, 103.0, 104.0],
            ("High", ticker): [101.0, 102.0, 103.0, 104.0, 105.0],
            ("Low", ticker): [99.0, 100.0, 101.0, 102.0, 103.0],
            ("Volume", ticker): [1e6, 1e6, 1e6, 1e6, 1e6],
        },
        index=dates,
    )


def test_load_data_returns_close_column() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01")
    assert "close" in result.columns


def test_load_data_returns_high_and_low_columns() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01")
    assert "high" in result.columns
    assert "low" in result.columns


def test_load_data_returns_datetime_index() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01")
    assert isinstance(result.index, pd.DatetimeIndex)


def test_load_data_close_values_match_adj_close() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01")
    assert list(result["close"]) == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_load_data_high_values_correct() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01")
    assert list(result["high"]) == [101.0, 102.0, 103.0, 104.0, 105.0]


def test_load_data_low_values_correct() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01")
    assert list(result["low"]) == [99.0, 100.0, 101.0, 102.0, 103.0]


def test_load_data_raises_on_empty_response() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="No data returned"):
            load_data("INVALID", "2020-01-01")


def test_load_data_output_passes_validation() -> None:
    from backtesting_engine.data.validator import validate_data

    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01")
    validate_data(result, min_rows=1)
