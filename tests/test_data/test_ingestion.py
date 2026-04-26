"""
Unit tests for the data ingestion module.

The network call to yfinance is mocked in all tests to keep the suite
fast, deterministic, and free of external dependencies.

Mock construction note
----------------------
yfinance returns a DataFrame with a two-level (field, ticker) MultiIndex
when auto_adjust=False. We build these with pd.MultiIndex.from_tuples and
pass the data as a row-oriented list so that column ordering is explicit
and unambiguous. Using dict-of-tuples syntax to construct a MultiIndex
DataFrame is fragile: Python dicts are ordered from 3.7+, but the intent
is clearest when the MultiIndex is built explicitly.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from backtesting_engine.data.ingestion import load_data


def _multiindex_df(
    ticker: str,
    dates: pd.DatetimeIndex,
    rows: list[list[float]],
) -> pd.DataFrame:
    """
    Build a yfinance-style DataFrame with a (field, ticker) MultiIndex.

    Columns are always in the order: Adj Close, High, Low, Volume.
    Each row in `rows` must have four values in that same order.
    """
    columns = pd.MultiIndex.from_tuples([
        ("Adj Close", ticker),
        ("High",      ticker),
        ("Low",       ticker),
        ("Volume",    ticker),
    ])
    return pd.DataFrame(rows, index=dates, columns=columns)


def _fake_download(ticker: str = "SPY") -> pd.DataFrame:
    """
    Minimal fake yfinance response. All close values are within [low, high]
    so _reconcile_adjusted_close makes no changes.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    #        Adj Close   High    Low   Volume
    rows = [
        [100.0,   101.0,  99.0,  1e6],
        [101.0,   102.0, 100.0,  1e6],
        [102.0,   103.0, 101.0,  1e6],
        [103.0,   104.0, 102.0,  1e6],
        [104.0,   105.0, 103.0,  1e6],
    ]
    return _multiindex_df(ticker, dates, rows)


def test_load_data_returns_close_column() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    assert "close" in result.columns


def test_load_data_returns_high_and_low_columns() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    assert "high" in result.columns
    assert "low" in result.columns


def test_load_data_returns_datetime_index() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_load_data_close_values_match_adj_close() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    assert list(result["close"]) == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_load_data_high_values_correct() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    assert list(result["high"]) == [101.0, 102.0, 103.0, 104.0, 105.0]


def test_load_data_low_values_correct() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    assert list(result["low"]) == [99.0, 100.0, 101.0, 102.0, 103.0]


def test_load_data_raises_on_empty_response() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="No data returned"):
            load_data("INVALID", "2020-01-01", use_cache=False)


def test_load_data_output_passes_validation() -> None:
    from backtesting_engine.data.validator import validate_data

    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    validate_data(result, min_rows=1)


def _fake_download_with_exdiv(ticker: str = "SPY") -> pd.DataFrame:
    """
    Fake yfinance response where bar 2's adjusted close (100.3) falls below
    the unadjusted low (100.5) - simulating a small ex-dividend adjustment.

    _reconcile_adjusted_close must clip close to low on that bar (≤ 0.5%
    discrepancy), and leave all other bars unchanged.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    #        Adj Close    High    Low   Volume
    rows = [
        [100.0,    101.0,   99.0,  1e6],
        [101.0,    102.0,  100.0,  1e6],
        [100.3,    103.0,  100.5,  1e6],  # adj close < low: ex-div clip
        [103.0,    104.0,  102.0,  1e6],
        [104.0,    105.0,  103.0,  1e6],
    ]
    return _multiindex_df(ticker, dates, rows)


def test_load_data_clips_close_below_low_on_exdiv() -> None:
    # Adjusted close of 100.3 below low of 100.5 on bar 2 - should be clipped.
    with patch("backtesting_engine.data.ingestion.yf.download",
               return_value=_fake_download_with_exdiv()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    # Clipped close must equal low (100.5), not original adjusted close (100.3).
    assert result["close"].iloc[2] == pytest.approx(100.5)


def test_load_data_clipped_close_within_band() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download",
               return_value=_fake_download_with_exdiv()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    # After reconciliation, every close must sit within [low, high].
    assert (result["close"] >= result["low"]).all()
    assert (result["close"] <= result["high"]).all()


def _fake_download_bad_data(ticker: str = "SPY") -> pd.DataFrame:
    """
    Fake response where bar 1's adjusted close (80.0) is 20% below the low
    (100.0) - far above the 0.5% ex-dividend tolerance. Should raise.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    #        Adj Close   High    Low   Volume
    rows = [
        [100.0,   101.0,  99.0,  1e6],
        [ 80.0,   102.0, 100.0,  1e6],  # 20% below low - data error, not ex-div
        [100.0,   101.0,  99.0,  1e6],
    ]
    return _multiindex_df(ticker, dates, rows)


def test_load_data_raises_on_large_close_band_violation() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download",
               return_value=_fake_download_bad_data()):
        with pytest.raises(ValueError, match="ex-dividend tolerance"):
            load_data("SPY", "2020-01-01", use_cache=False)
