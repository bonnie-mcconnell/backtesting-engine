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

    Columns are in the order: Adj Close, Close, High, Low, Volume.
    Each row in `rows` must have five values in that same order.

    'Close' (unadjusted) is required by ingestion.py to compute the
    dividend/split adjustment factor: adjustment = Adj Close / Close.
    Without it, adjusted high/low cannot be computed correctly.
    """
    columns = pd.MultiIndex.from_tuples([
        ("Adj Close", ticker),
        ("Close",     ticker),
        ("High",      ticker),
        ("Low",       ticker),
        ("Volume",    ticker),
    ])
    return pd.DataFrame(rows, index=dates, columns=columns)


def _fake_download(ticker: str = "SPY") -> pd.DataFrame:
    """
    Minimal fake yfinance response. All close values are within [low, high]
    so _reconcile_adjusted_close makes no changes.

    Close == Adj Close in all rows (no dividends/splits), so adjustment = 1.0
    and adjusted high/low equal unadjusted high/low.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    #        Adj Close  Close   High    Low   Volume
    rows = [
        [100.0,  100.0,  101.0,  99.0,  1e6],
        [101.0,  101.0,  102.0, 100.0,  1e6],
        [102.0,  102.0,  103.0, 101.0,  1e6],
        [103.0,  103.0,  104.0, 102.0,  1e6],
        [104.0,  104.0,  105.0, 103.0,  1e6],
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


def test_load_data_retries_empty_response_then_succeeds() -> None:
    download = [pd.DataFrame(), _fake_download()]
    with (
        patch("backtesting_engine.data.ingestion.yf.download", side_effect=download) as mock_download,
        patch("time.sleep") as mock_sleep,
    ):
        result = load_data("SPY", "2020-01-01", use_cache=False)

    assert len(result) == 5
    assert mock_download.call_count == 2
    mock_sleep.assert_called_once_with(1)


def test_load_data_retries_exception_then_succeeds() -> None:
    download = [ConnectionError("temporary yfinance failure"), _fake_download()]
    with (
        patch("backtesting_engine.data.ingestion.yf.download", side_effect=download) as mock_download,
        patch("time.sleep") as mock_sleep,
    ):
        result = load_data("SPY", "2020-01-01", use_cache=False)

    assert len(result) == 5
    assert mock_download.call_count == 2
    mock_sleep.assert_called_once_with(1)


def test_load_data_reports_last_retry_exception() -> None:
    with (
        patch(
            "backtesting_engine.data.ingestion.yf.download",
            side_effect=ConnectionError("temporary yfinance failure"),
        ) as mock_download,
        patch("time.sleep") as mock_sleep,
    ):
        with pytest.raises(ValueError, match="after 3 attempts"):
            load_data("SPY", "2020-01-01", use_cache=False)

    assert mock_download.call_count == 3
    assert mock_sleep.call_count == 2


def test_load_data_output_passes_validation() -> None:
    from backtesting_engine.data.validator import validate_data

    with patch("backtesting_engine.data.ingestion.yf.download", return_value=_fake_download()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    validate_data(result, min_rows=1)


def _fake_download_with_exdiv(ticker: str = "SPY") -> pd.DataFrame:
    """
    Fake yfinance response simulating an ex-dividend date where the adjusted
    close sits below the adjusted low - requiring a clip.

    On ex-dividend dates, yfinance reduces Adj Close by the dividend amount.
    The adjusted high/low are also scaled by (Adj Close / Close). When the
    dividend is large relative to the intraday range, the scaled close can
    sit slightly outside the scaled [low, high] band.

    Here: Close=101.0, Adj Close=100.3 (adjustment ≈ 0.9931).
    Adjusted low = 100.8 * 0.9931 ≈ 100.23, but adj_close = 100.3 > 100.23,
    so no clip... We use a different scenario: Adj Close < raw Low, which
    happens when Close (unadjusted) falls below the raw Low due to yfinance
    data revisions. We use raw Low > Adj Close directly:
    adj_close=100.3, raw_low=100.8, adj_factor=100.3/101.0≈0.9931,
    adj_low=100.8*0.9931≈100.1. adj_close(100.3) > adj_low(100.1): no clip.

    Correct design: make adj_close sit below adj_low by having a very small
    (Adj Close / Close) ratio combined with a high raw Low:
    Adj Close=99.0, Close=101.0 (factor=0.9802), Low=101.5 → adj_low=99.50.
    adj_close(99.0) < adj_low(99.50): clip fires, sets close=99.50. ✓
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    #        Adj Close  Close    High    Low     Volume
    rows = [
        [100.0,  100.0,  101.0,   99.0,   1e6],
        [101.0,  101.0,  102.0,  100.0,   1e6],
        [ 99.0,  101.0,  103.0,  101.5,   1e6],  # adj_close=99 < adj_low≈99.5: clip
        [103.0,  103.0,  104.0,  102.0,   1e6],
        [104.0,  104.0,  105.0,  103.0,   1e6],
    ]
    return _multiindex_df(ticker, dates, rows)


def _expected_exdiv_bar2_close() -> float:
    """Expected close after clip on bar 2 of _fake_download_with_exdiv."""
    # adjustment = 99.0 / 101.0
    adj = 99.0 / 101.0
    # adj_low = 101.5 * adj
    return 101.5 * adj


def test_load_data_clips_close_below_low_on_exdiv() -> None:
    """Adjusted close below adjusted low must be clipped up to adjusted low."""
    with patch("backtesting_engine.data.ingestion.yf.download",
               return_value=_fake_download_with_exdiv()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    expected = _expected_exdiv_bar2_close()
    assert result["close"].iloc[2] == pytest.approx(expected, rel=1e-6)


def test_load_data_clipped_close_within_band() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download",
               return_value=_fake_download_with_exdiv()):
        result = load_data("SPY", "2020-01-01", use_cache=False)
    # After reconciliation, every close must sit within [low, high].
    assert (result["close"] >= result["low"]).all()
    assert (result["close"] <= result["high"]).all()


def _fake_download_bad_data(ticker: str = "SPY") -> pd.DataFrame:
    """
    Fake response where bar 1's adjusted close sits ~20% below the adjusted low -
    far above the 0.5% ex-dividend tolerance. Should raise ValueError.

    Adj Close=80.0, Close=100.0 → adjustment=0.80.
    Raw Low=100.0 → adj_low=80.0. adj_close=80.0 == adj_low → no violation?

    We need adj_close < adj_low. Use: Adj Close=79.0, Close=100.0, raw Low=100.0.
    adj_low = 100.0 * 0.79 = 79.0 = adj_close → still no gap.

    Correct approach: Adj Close=78.0, Close=100.0, raw Low=100.0.
    adj_low = 100.0 * 0.78 = 78.0. adj_close=78.0 == adj_low → still 0 gap.

    The clip only fires when adj_close < adj_low, i.e., Adj Close < raw_Low * (Adj Close/Close).
    That simplifies to: Close < raw_Low. So we need unadjusted close below raw low.
    Adj Close=78.0, Close=80.0 (factor=0.975), raw Low=100.0 → adj_low=97.5.
    adj_close=78.0 < adj_low=97.5. Gap = 97.5 - 78.0 = 19.5, fraction = 19.5/78.0 = 25%. ✓
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    #        Adj Close  Close   High    Low   Volume
    rows = [
        [100.0,  100.0,  101.0,  99.0,  1e6],
        [ 78.0,   80.0,  102.0, 100.0,  1e6],  # 25% below adj_low → raises
        [100.0,  100.0,  101.0,  99.0,  1e6],
    ]
    return _multiindex_df(ticker, dates, rows)


def test_load_data_raises_on_large_close_band_violation() -> None:
    with patch("backtesting_engine.data.ingestion.yf.download",
               return_value=_fake_download_bad_data()):
        with pytest.raises(ValueError, match="ex-dividend tolerance"):
            load_data("SPY", "2020-01-01", use_cache=False)


def _fake_download_no_close_column(ticker: str = "SPY") -> pd.DataFrame:
    """Fake yfinance response missing the 'Close' (unadjusted) column.

    Previously this silently produced unadjusted high/low values; now it
    must raise ValueError because adjustment factor cannot be computed.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    columns = pd.MultiIndex.from_tuples([
        ("Adj Close", ticker),
        ("High",      ticker),
        ("Low",       ticker),
        ("Volume",    ticker),
    ])
    rows = [
        [100.0, 101.0, 99.0, 1e6],
        [101.0, 102.0, 100.0, 1e6],
        [102.0, 103.0, 101.0, 1e6],
    ]
    return pd.DataFrame(rows, index=dates, columns=columns)


def test_load_data_raises_when_close_column_missing() -> None:
    """'Close' is required to compute the adjustment factor.

    Without it, ingestion.py cannot build adjusted high/low, so it must
    raise ValueError rather than silently return unadjusted values.
    """
    with patch("backtesting_engine.data.ingestion.yf.download",
               return_value=_fake_download_no_close_column()):
        with pytest.raises(ValueError, match="Close"):
            load_data("SPY", "2020-01-01", use_cache=False)


def test_adjusted_high_low_use_adjustment_factor() -> None:
    """Adjusted high/low must equal raw_high/low * (Adj Close / Close).

    With adjustment < 1.0 (dividend paid), adj_high < raw_high.
    """
    # Adj Close = 90, Close = 100 → adjustment = 0.90
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    columns = pd.MultiIndex.from_tuples([
        ("Adj Close", "SPY"), ("Close", "SPY"),
        ("High", "SPY"), ("Low", "SPY"), ("Volume", "SPY"),
    ])
    rows = [
        [90.0, 100.0, 110.0, 88.0, 1e6],
        [91.0, 100.0, 111.0, 89.0, 1e6],
    ]
    df = pd.DataFrame(rows, index=dates, columns=columns)
    with patch("backtesting_engine.data.ingestion.yf.download", return_value=df):
        result = load_data("SPY", "2020-01-01", use_cache=False)

    adj = 90.0 / 100.0
    assert result["high"].iloc[0] == pytest.approx(110.0 * adj)
    assert result["low"].iloc[0] == pytest.approx(88.0 * adj)
