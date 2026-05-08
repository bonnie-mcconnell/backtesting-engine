"""
Tests for trade-level diagnostic metrics added in the fix pass.

These cover: exposure fraction, win rate, avg win/loss ratio, avg holding period.
Each test uses synthetic data where the expected value can be derived by hand,
so the tests are independent of the strategy and exercise the metric code directly.
"""

import math

import numpy as np
import pandas as pd

from backtesting_engine.metrics import _trade_diagnostics, calculate_metrics
from backtesting_engine.models import Trade

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _ts(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _trade(entry_date: str, exit_date: str, pnl: float, costs: float = 1.0) -> Trade:
    entry = pd.Timestamp(entry_date)
    exit_ = pd.Timestamp(exit_date)
    return Trade(
        entry_date=entry,
        exit_date=exit_,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        shares=1.0,
        transaction_costs=costs,
        pnl=pnl,
    )


def _flat_pv(n: int = 20, initial: float = 10_000.0) -> pd.Series:
    """Portfolio that never changes - strategy was in cash the whole window."""
    return pd.Series([initial] * n, index=_ts(n), dtype=float)


def _trending_pv(n: int = 20, initial: float = 10_000.0) -> pd.Series:
    """Portfolio that changes every bar - strategy was fully invested."""
    vals = np.linspace(initial, initial * 1.05, n)
    return pd.Series(vals, index=_ts(n), dtype=float)


# ---------------------------------------------------------------------------
# Exposure fraction
# ---------------------------------------------------------------------------

class TestExposureFraction:
    def test_no_trades_gives_zero_exposure(self) -> None:
        """No trades → strategy held cash entire window → exposure = 0."""
        pv = _trending_pv()
        exposure, *_ = _trade_diagnostics(pv, trades=[])
        assert exposure == 0.0

    def test_none_trades_gives_nan(self) -> None:
        """trades=None means diagnostics not available → NaN, not 0."""
        pv = _trending_pv()
        exposure, *_ = _trade_diagnostics(pv, trades=None)
        assert math.isnan(exposure)

    def test_single_trade_exposure_fraction(self) -> None:
        """A trade spanning 5 of 20 bars → exposure = 5/20 = 0.25."""
        dates = _ts(20)
        pv = pd.Series([10_000.0] * 20, index=dates, dtype=float)
        # Trade open from day 2 to day 6 inclusive = 5 bars
        trade = _trade(str(dates[2].date()), str(dates[6].date()), pnl=50.0)
        exposure, *_ = _trade_diagnostics(pv, trades=[trade])
        assert math.isclose(exposure, 5 / 20, rel_tol=1e-6)

    def test_exposure_bounded_zero_one(self) -> None:
        pv = _trending_pv()
        trade = _trade("2020-01-02", "2020-01-15", pnl=10.0)
        exposure, *_ = _trade_diagnostics(pv, trades=[trade])
        if not math.isnan(exposure):
            assert 0.0 <= exposure <= 1.0


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_winners(self) -> None:
        trades = [_trade("2020-01-02", "2020-01-10", pnl=50.0) for _ in range(5)]
        _, _, win_rate, *_ = _trade_diagnostics(_trending_pv(), trades=trades)
        assert math.isclose(win_rate, 1.0), f"Expected 1.0, got {win_rate}"

    def test_all_losers(self) -> None:
        trades = [_trade("2020-01-02", "2020-01-10", pnl=-50.0) for _ in range(5)]
        _, _, win_rate, *_ = _trade_diagnostics(_trending_pv(), trades=trades)
        assert math.isclose(win_rate, 0.0), f"Expected 0.0, got {win_rate}"

    def test_mixed_win_rate(self) -> None:
        trades = [
            _trade("2020-01-02", "2020-01-10", pnl=100.0),
            _trade("2020-01-12", "2020-01-20", pnl=-50.0),
            _trade("2020-01-22", "2020-01-30", pnl=75.0),
            _trade("2020-02-03", "2020-02-10", pnl=-25.0),
        ]
        _, _, win_rate, *_ = _trade_diagnostics(_trending_pv(40), trades=trades)
        assert math.isclose(win_rate, 0.5), f"Expected 0.5, got {win_rate}"

    def test_no_trades_gives_nan(self) -> None:
        _, _, win_rate, *_ = _trade_diagnostics(_flat_pv(), trades=[])
        assert math.isnan(win_rate)


# ---------------------------------------------------------------------------
# Avg win/loss ratio
# ---------------------------------------------------------------------------

class TestAvgWinLossRatio:
    def test_equal_wins_and_losses(self) -> None:
        trades = [
            _trade("2020-01-02", "2020-01-10", pnl=100.0),
            _trade("2020-01-12", "2020-01-20", pnl=-100.0),
        ]
        _, _, _, avg_wl, _ = _trade_diagnostics(_trending_pv(30), trades=trades)
        assert math.isclose(avg_wl, 1.0, rel_tol=1e-6), f"Expected 1.0, got {avg_wl}"

    def test_wins_twice_losses(self) -> None:
        trades = [
            _trade("2020-01-02", "2020-01-10", pnl=200.0),
            _trade("2020-01-12", "2020-01-20", pnl=-100.0),
        ]
        _, _, _, avg_wl, _ = _trade_diagnostics(_trending_pv(30), trades=trades)
        assert math.isclose(avg_wl, 2.0, rel_tol=1e-6), f"Expected 2.0, got {avg_wl}"

    def test_no_losses_gives_inf(self) -> None:
        trades = [_trade("2020-01-02", "2020-01-10", pnl=100.0)]
        _, _, _, avg_wl, _ = _trade_diagnostics(_trending_pv(), trades=trades)
        assert math.isinf(avg_wl) and avg_wl > 0

    def test_no_wins_gives_zero(self) -> None:
        trades = [_trade("2020-01-02", "2020-01-10", pnl=-100.0)]
        _, _, _, avg_wl, _ = _trade_diagnostics(_trending_pv(), trades=trades)
        assert avg_wl == 0.0


# ---------------------------------------------------------------------------
# Average holding period
# ---------------------------------------------------------------------------

class TestAvgHoldingDays:
    def test_single_trade_five_days(self) -> None:
        trades = [_trade("2020-01-02", "2020-01-07", pnl=10.0)]
        _, _, _, _, avg_hold = _trade_diagnostics(_trending_pv(), trades=trades)
        assert math.isclose(avg_hold, 5.0, rel_tol=1e-6), f"Expected 5.0, got {avg_hold}"

    def test_mean_of_two_different_holds(self) -> None:
        trades = [
            _trade("2020-01-02", "2020-01-07", pnl=10.0),   # 5 days
            _trade("2020-01-10", "2020-01-20", pnl=10.0),   # 10 days
        ]
        _, _, _, _, avg_hold = _trade_diagnostics(_trending_pv(30), trades=trades)
        assert math.isclose(avg_hold, 7.5, rel_tol=1e-6), f"Expected 7.5, got {avg_hold}"

    def test_no_trades_gives_nan(self) -> None:
        _, _, _, _, avg_hold = _trade_diagnostics(_flat_pv(), trades=[])
        assert math.isnan(avg_hold)


# ---------------------------------------------------------------------------
# Integration: diagnostics populated in MetricsResult via calculate_metrics
# ---------------------------------------------------------------------------

class TestDiagnosticsIntegration:
    def test_calculate_metrics_populates_trade_count(self) -> None:
        pv = _trending_pv(30)
        trades = [
            _trade("2020-01-02", "2020-01-10", pnl=50.0),
            _trade("2020-01-12", "2020-01-20", pnl=-20.0),
        ]
        result = calculate_metrics(pv, trades=trades)
        assert result.trade_count == 2

    def test_calculate_metrics_no_trades_gives_zero_count(self) -> None:
        pv = _trending_pv()
        result = calculate_metrics(pv, trades=None)
        assert result.trade_count == 0

    def test_calculate_metrics_win_rate_in_result(self) -> None:
        pv = _trending_pv(30)
        trades = [
            _trade("2020-01-02", "2020-01-10", pnl=50.0),
            _trade("2020-01-12", "2020-01-20", pnl=-20.0),
        ]
        result = calculate_metrics(pv, trades=trades)
        assert math.isclose(result.win_rate, 0.5)

    def test_calculate_metrics_no_trades_produces_nan_diagnostics(self) -> None:
        pv = _trending_pv()
        result = calculate_metrics(pv, trades=None)
        # trades=None → diagnostics unavailable, not zero
        assert math.isnan(result.win_rate)
        assert math.isnan(result.avg_holding_days)
        assert math.isnan(result.avg_win_loss_ratio)
        assert math.isnan(result.exposure_fraction)

    def test_calculate_metrics_empty_trades_gives_zero_exposure(self) -> None:
        pv = _trending_pv()
        result = calculate_metrics(pv, trades=[])
        # trades=[] means strategy ran but made no trades → exposure = 0
        assert result.exposure_fraction == 0.0
        assert result.trade_count == 0
