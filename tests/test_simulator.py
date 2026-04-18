"""
Unit tests for the trade execution simulator.

Tests use controlled synthetic data with known prices and signals so that
all expected values can be computed by hand and hard-coded. No real market
data is used - tests must be deterministic and independent of external state.
"""
import pandas as pd
import numpy as np
import pytest

from backtesting_engine.config import (
    TRANSACTION_COST_RATE,
    INITIAL_PORTFOLIO_VALUE,
    POSITION_SIZE_FRACTION,
)
from backtesting_engine.simulator import run_simulation


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def five_day_data() -> pd.DataFrame:
    """Five business days of synthetic price data starting 2020-01-01.

    Dates: 2020-01-01, 2020-01-02, 2020-01-05, 2020-01-06, 2020-01-07
    Prices: 100, 101, 102, 103, 104
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)


# ---------------------------------------------------------------------------
# Complete buy/sell cycle
# ---------------------------------------------------------------------------

def test_complete_cycle_produces_one_trade(five_day_data: pd.DataFrame) -> None:
    # Buy on day 2 (price 101), sell on day 4 (price 103)
    signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    assert len(result.trades) == 1


def test_complete_cycle_entry_exit_dates(five_day_data: pd.DataFrame) -> None:
    signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    trade = result.trades[0]
    assert trade.entry_date == pd.Timestamp("2020-01-02")
    assert trade.exit_date == pd.Timestamp("2020-01-06")


def test_complete_cycle_entry_exit_prices(five_day_data: pd.DataFrame) -> None:
    signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    trade = result.trades[0]
    assert trade.entry_price == 101.0
    assert trade.exit_price == 103.0


def test_complete_cycle_shares(five_day_data: pd.DataFrame) -> None:
    # shares = (portfolio * position_fraction) / entry_price
    # = (100_000 * 1.0) / 101.0 = 990.099...
    signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    trade = result.trades[0]
    expected_shares = (INITIAL_PORTFOLIO_VALUE * POSITION_SIZE_FRACTION) / 101.0
    assert np.isclose(trade.shares, expected_shares, rtol=1e-5)


def test_complete_cycle_transaction_costs(five_day_data: pd.DataFrame) -> None:
    # buy cost = shares * entry_price * rate
    # sell cost = shares * exit_price * rate
    # total = shares * (entry + exit) * rate
    signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    trade = result.trades[0]
    expected_costs = trade.shares * (101.0 + 103.0) * TRANSACTION_COST_RATE
    assert np.isclose(trade.transaction_costs, expected_costs, rtol=1e-5)


def test_complete_cycle_pnl(five_day_data: pd.DataFrame) -> None:
    # pnl = (sell_proceeds - sell_cost) - (buy_cost_basis + buy_transaction_cost)
    # = (shares*103 - shares*103*rate) - (shares*101 + shares*101*rate)
    # = shares * (103 - 103*rate - 101 - 101*rate)
    # = shares * (2 - 204*rate)
    signals = pd.Series([0, 1, 0, -1, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    trade = result.trades[0]
    expected_pnl = trade.shares * (103.0 - 101.0) - trade.transaction_costs
    assert np.isclose(trade.pnl, expected_pnl, rtol=1e-5)


# ---------------------------------------------------------------------------
# No signals
# ---------------------------------------------------------------------------

def test_no_signals_returns_empty_trades(five_day_data: pd.DataFrame) -> None:
    signals = pd.Series([0, 0, 0, 0, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    assert result.trades == []


def test_no_signals_message(five_day_data: pd.DataFrame) -> None:
    signals = pd.Series([0, 0, 0, 0, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    assert result.message == "No trades executed."


def test_no_signals_portfolio_stays_flat(five_day_data: pd.DataFrame) -> None:
    # with no trades, portfolio value should remain at initial value every day
    signals = pd.Series([0, 0, 0, 0, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    assert result.portfolio_values is not None
    assert (result.portfolio_values == INITIAL_PORTFOLIO_VALUE).all()


# ---------------------------------------------------------------------------
# Buy with no sell - position closed at end of window
# ---------------------------------------------------------------------------

def test_buy_no_sell_closes_at_window_end(five_day_data: pd.DataFrame) -> None:
    # Buy on day 2, no sell signal - should close at last price (104.0) on last date
    signals = pd.Series([0, 1, 0, 0, 0], index=five_day_data.index)
    result = run_simulation(five_day_data, signals)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == 101.0
    assert trade.exit_price == 104.0
    assert trade.exit_date == pd.Timestamp("2020-01-07")


# ---------------------------------------------------------------------------
# Data/signal mismatch guard
# ---------------------------------------------------------------------------

def test_mismatched_data_signals_raises(five_day_data: pd.DataFrame) -> None:
    # signals with different length should raise ValueError
    short_signals = pd.Series([0, 1, 0], index=five_day_data.index[:3])
    with pytest.raises(ValueError):
        run_simulation(five_day_data, short_signals)