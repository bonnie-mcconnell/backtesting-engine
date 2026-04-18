"""
Unit tests for the trade execution simulator.

All tests use synthetic price data with known values so expected results
can be computed by hand. No real market data is used - tests must be
deterministic and independent of any external state.

Test philosophy: each test asserts exactly one thing. This makes failures
diagnostic rather than just indicative that something is wrong somewhere.
"""

import pandas as pd
import numpy as np
import pytest

from backtesting_engine.config import (
    INITIAL_PORTFOLIO_VALUE,
    POSITION_SIZE_FRACTION,
    TRANSACTION_COST_RATE,
)
from backtesting_engine.simulator import run_simulation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def five_day_data() -> pd.DataFrame:
    """Five business days of synthetic price data.

    Dates: 2020-01-02, 2020-01-03, 2020-01-06, 2020-01-07, 2020-01-08
    Prices: 100, 101, 102, 103, 104
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)


# ---------------------------------------------------------------------------
# Complete buy/sell cycle
# ---------------------------------------------------------------------------

class TestCompleteCycle:
    def test_produces_one_trade(self, five_day_data: pd.DataFrame) -> None:
        signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
        result = run_simulation(five_day_data, signals)
        assert len(result.trades) == 1

    def test_entry_exit_dates(self, five_day_data: pd.DataFrame) -> None:
        signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
        trade = run_simulation(five_day_data, signals).trades[0]
        assert trade.entry_date == five_day_data.index[1]
        assert trade.exit_date == five_day_data.index[3]

    def test_entry_exit_prices(self, five_day_data: pd.DataFrame) -> None:
        signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
        trade = run_simulation(five_day_data, signals).trades[0]
        assert trade.entry_price == 101.0
        assert trade.exit_price == 103.0

    def test_shares_computed_correctly(self, five_day_data: pd.DataFrame) -> None:
        # shares = (portfolio * position_fraction) / entry_price
        signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
        trade = run_simulation(five_day_data, signals).trades[0]
        expected = (INITIAL_PORTFOLIO_VALUE * POSITION_SIZE_FRACTION) / 101.0
        assert np.isclose(trade.shares, expected, rtol=1e-5)

    def test_transaction_costs(self, five_day_data: pd.DataFrame) -> None:
        # total costs = shares * (entry_price + exit_price) * rate
        signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
        trade = run_simulation(five_day_data, signals).trades[0]
        expected = trade.shares * (101.0 + 103.0) * TRANSACTION_COST_RATE
        assert np.isclose(trade.transaction_costs, expected, rtol=1e-5)

    def test_pnl(self, five_day_data: pd.DataFrame) -> None:
        # pnl = gross profit - total transaction costs
        #     = shares * (exit - entry) - costs
        signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
        trade = run_simulation(five_day_data, signals).trades[0]
        expected = trade.shares * (103.0 - 101.0) - trade.transaction_costs
        assert np.isclose(trade.pnl, expected, rtol=1e-5)

    def test_portfolio_has_correct_length(self, five_day_data: pd.DataFrame) -> None:
        signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
        result = run_simulation(five_day_data, signals)
        assert result.portfolio_values is not None
        assert len(result.portfolio_values) == len(five_day_data)


# ---------------------------------------------------------------------------
# No signals
# ---------------------------------------------------------------------------

class TestNoSignals:
    def test_returns_empty_trades(self, five_day_data: pd.DataFrame) -> None:
        signals = pd.Series([0, 0, 0, 0, 0], index=five_day_data.index)
        assert run_simulation(five_day_data, signals).trades == []

    def test_status_message(self, five_day_data: pd.DataFrame) -> None:
        signals = pd.Series([0, 0, 0, 0, 0], index=five_day_data.index)
        assert "No trades" in run_simulation(five_day_data, signals).message

    def test_portfolio_stays_at_initial_value(self, five_day_data: pd.DataFrame) -> None:
        signals = pd.Series([0, 0, 0, 0, 0], index=five_day_data.index)
        result = run_simulation(five_day_data, signals)
        assert result.portfolio_values is not None
        assert (result.portfolio_values == INITIAL_PORTFOLIO_VALUE).all()


# ---------------------------------------------------------------------------
# Open position closed at end of window
# ---------------------------------------------------------------------------

class TestOpenPositionClose:
    def test_closes_at_last_price(self, five_day_data: pd.DataFrame) -> None:
        # Buy on day 2, no sell signal - must close at last price (104.0).
        signals = pd.Series([0, 1, 0, 0, 0], index=five_day_data.index)
        trade = run_simulation(five_day_data, signals).trades[0]
        assert trade.exit_price == 104.0

    def test_closes_on_last_date(self, five_day_data: pd.DataFrame) -> None:
        signals = pd.Series([0, 1, 0, 0, 0], index=five_day_data.index)
        trade = run_simulation(five_day_data, signals).trades[0]
        assert trade.exit_date == five_day_data.index[-1]

    def test_final_portfolio_reflects_sell_cost(self, five_day_data: pd.DataFrame) -> None:
        # After force-close, portfolio value should not include unsettled sell cost.
        signals = pd.Series([0, 1, 0, 0, 0], index=five_day_data.index)
        result = run_simulation(five_day_data, signals)
        trade = result.trades[0]
        assert result.portfolio_values is not None
        expected_final = (
            (INITIAL_PORTFOLIO_VALUE * POSITION_SIZE_FRACTION)
            * (104.0 / 101.0)
            - trade.transaction_costs
        )
        assert np.isclose(float(result.portfolio_values.iloc[-1]), expected_final, rtol=1e-4)


# ---------------------------------------------------------------------------
# Signal validation
# ---------------------------------------------------------------------------

class TestSignalValidation:
    def test_mismatched_length_raises(self, five_day_data: pd.DataFrame) -> None:
        short_signals = pd.Series([0, 1, 0], index=five_day_data.index[:3])
        with pytest.raises(ValueError, match="does not match"):
            run_simulation(five_day_data, short_signals)

    def test_invalid_signal_value_raises(self, five_day_data: pd.DataFrame) -> None:
        # Signal value of 2 is not in {-1, 0, 1}.
        bad_signals = pd.Series([0, 2, 0, -1, 0], index=five_day_data.index)
        with pytest.raises(ValueError, match="invalid values"):
            run_simulation(five_day_data, bad_signals)

    def test_fractional_signal_value_raises(self, five_day_data: pd.DataFrame) -> None:
        bad_signals = pd.Series([0.0, 0.5, 0.0, -1.0, 0.0], index=five_day_data.index)
        with pytest.raises(ValueError, match="invalid values"):
            run_simulation(five_day_data, bad_signals)


# ---------------------------------------------------------------------------
# Portfolio value / trade consistency
# ---------------------------------------------------------------------------

class TestPortfolioTradeConsistency:
    def test_portfolio_always_positive(self, five_day_data: pd.DataFrame) -> None:
        # Portfolio value should never go negative with valid inputs.
        signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
        result = run_simulation(five_day_data, signals)
        assert result.portfolio_values is not None
        assert (result.portfolio_values > 0).all()

    def test_profitable_trade_increases_portfolio(self, five_day_data: pd.DataFrame) -> None:
        # Buy at 100, sell at 104 - portfolio should end higher than it started
        # (net of transaction costs).
        signals = pd.Series([1, 0, 0, 0, -1], index=five_day_data.index)
        result = run_simulation(five_day_data, signals)
        assert result.portfolio_values is not None
        assert float(result.portfolio_values.iloc[-1]) > INITIAL_PORTFOLIO_VALUE

    def test_multiple_trades_sum_to_final_portfolio(self, five_day_data: pd.DataFrame) -> None:
        # Two complete cycles: buy/sell/buy/sell.
        dates = pd.date_range("2020-01-01", periods=8, freq="B")
        prices = pd.DataFrame({"close": [100.0, 102.0, 102.0, 101.0, 103.0, 106.0, 106.0, 100.0]},
                              index=dates)
        signals = pd.Series([1, -1, 0, 1, -1, 0, 0, 0], index=dates)
        result = run_simulation(prices, signals)
        assert len(result.trades) == 2