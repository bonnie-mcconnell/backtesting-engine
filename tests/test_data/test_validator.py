"""
Unit tests for data structural validation.

validate_data raises ValueError on any structural problem and passes
silently on well-formed data.
"""

import numpy as np
import pandas as pd
import pytest

from backtesting_engine.data.validator import validate_data


def _valid_data(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": np.linspace(100.0, 110.0, n)}, index=dates)


class TestValidateData:
    def test_valid_data_does_not_raise(self) -> None:
        validate_data(_valid_data())

    def test_non_datetime_index_raises(self) -> None:
        data = _valid_data()
        data.index = range(len(data))  # type: ignore[assignment]
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_data(data)

    def test_duplicate_index_raises(self) -> None:
        dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03"])
        data = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=dates)
        with pytest.raises(ValueError, match="duplicate"):
            validate_data(data)

    def test_non_monotonic_index_raises(self) -> None:
        dates = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-01"])
        data = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=dates)
        with pytest.raises(ValueError, match="monotonically"):
            validate_data(data)

    def test_missing_close_column_raises(self) -> None:
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        data = pd.DataFrame({"open": [100.0] * 5}, index=dates)
        with pytest.raises(ValueError, match="'close'"):
            validate_data(data)

    def test_null_close_values_raises(self) -> None:
        data = _valid_data()
        data.iloc[2, 0] = float("nan")
        with pytest.raises(ValueError, match="missing"):
            validate_data(data)

    def test_negative_close_value_raises(self) -> None:
        data = _valid_data()
        data.iloc[2, 0] = -1.0
        with pytest.raises(ValueError, match="negative"):
            validate_data(data)

    def test_zero_close_value_raises(self) -> None:
        data = _valid_data()
        data.iloc[2, 0] = 0.0
        with pytest.raises(ValueError, match="negative"):
            validate_data(data)

    def test_min_rows_enforced(self) -> None:
        data = _valid_data(n=5)
        with pytest.raises(ValueError, match="5 rows"):
            validate_data(data, min_rows=10)

    def test_min_rows_default_accepts_any_nonempty(self) -> None:
        validate_data(_valid_data(n=1))

    def test_empty_dataframe_raises(self) -> None:
        dates = pd.date_range("2020-01-01", periods=0, freq="B")
        data = pd.DataFrame({"close": []}, index=dates)
        with pytest.raises(ValueError):
            validate_data(data)
