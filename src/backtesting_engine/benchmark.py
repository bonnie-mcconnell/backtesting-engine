"""
Buy-and-hold benchmark comparison.

The standard critique of any active strategy is: does it beat just holding
the index? Beating buy-and-hold on raw Sharpe over one historical period is
not the right bar - buy-and-hold has survivorship bias, the comparison
ignores the path dependency of active strategies, and a single Sharpe comparison
has no notion of statistical significance.

This module computes:

  1. Buy-and-hold metrics on the same test windows used by walk_forward(),
     so the comparison is apples-to-apples (same data slices, same costs).

  2. Information ratio: (strategy Sharpe - benchmark Sharpe) / tracking error.
     This is the correct risk-adjusted excess return over benchmark. A positive
     IR means the strategy added alpha per unit of active risk, not just per
     unit of total volatility.

  3. A paired t-test on per-window Sharpe differences. Because we have ~26
     independent windows, we can test whether the strategy consistently beats
     buy-and-hold across windows, not just on average. This is more meaningful
     than a single Sharpe comparison.

  4. Relative drawdown: strategy max drawdown vs benchmark max drawdown. A
     strategy with worse drawdown than buy-and-hold is hard to justify even
     if its Sharpe is similar.

These metrics are appended to the BacktestResult summary as a BenchmarkResult
and surfaced in both the console output and the dashboard.

Reference
---------
Grinold, R. & Kahn, R. (2000). Active Portfolio Management (2nd ed.).
  Chapter 2 defines the information ratio and its relationship to alpha.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    TRANSACTION_COST_RATE,
)
from backtesting_engine.metrics import _max_drawdown, _sharpe, _sortino
from backtesting_engine.models import BacktestResult


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Buy-and-hold benchmark metrics computed over the same walk-forward windows.

    Attributes
    ----------
    benchmark_sharpe : float
        Mean per-window Sharpe for a buy-and-hold position, same data slices
        as the strategy's walk-forward windows.
    benchmark_sortino : float
        Mean per-window Sortino for buy-and-hold.
    benchmark_max_drawdown : float
        Worst per-window drawdown for buy-and-hold. Not averaged - the worst
        single-window drawdown is the relevant risk figure.
    information_ratio : float
        Annualised mean active return divided by annualised active return volatility.

        Active return at bar t = strategy_return[t] - benchmark_return[t].
        IR = mean(active_returns) / std(active_returns) * sqrt(252).

        This is the Grinold & Kahn (2000) definition. IR > 0 means the strategy
        generated positive risk-adjusted excess return over buy-and-hold.

        Note: active returns are only defined on bars where BOTH the strategy
        portfolio and the benchmark have a return. Bars where the strategy holds
        no position contribute a zero active return (not a gap).
    sharpe_diff_t_stat : float
        Paired t-statistic for per-window Sharpe differences (strategy - benchmark).
        Tests whether strategy Sharpe consistently differs from benchmark Sharpe
        across walk-forward windows.
    sharpe_diff_p_value : float
        Two-sided p-value for the Sharpe difference t-test.
    strategy_beats_benchmark_fraction : float
        Fraction of walk-forward windows in which strategy Sharpe > benchmark Sharpe.
    """
    benchmark_sharpe: float
    benchmark_sortino: float
    benchmark_max_drawdown: float
    information_ratio: float
    sharpe_diff_t_stat: float
    sharpe_diff_p_value: float
    strategy_beats_benchmark_fraction: float


def compute_benchmark(
    result: BacktestResult,
    data: pd.DataFrame,
) -> BenchmarkResult:
    """
    Compute buy-and-hold benchmark metrics over the same windows as a BacktestResult.

    The benchmark is a fully-invested long position opened at the start of each
    test window and closed at the end, with the same transaction cost applied on
    entry and exit.

    The information ratio is computed from per-bar active returns - the difference
    between the strategy's daily return and the buy-and-hold daily return on each
    bar. This is the Grinold & Kahn (2000) definition: IR = mean(active) / std(active)
    annualised by sqrt(252). Using per-window Sharpe differences instead would give
    a dimensionally inconsistent "Sharpe ratio of Sharpe ratios", not a true IR.

    Args:
        result: BacktestResult from walk_forward(). Provides window boundaries,
                portfolio value series, and per-window Sharpe ratios.
        data: The full price DataFrame used for the original walk_forward() call.

    Returns:
        BenchmarkResult with per-window and aggregate comparison metrics.

    Raises:
        ValueError: If result has no valid (non-skipped) windows.
    """
    valid = result.valid_windows
    if not valid:
        raise ValueError("BacktestResult has no valid windows to benchmark against.")

    benchmark_sharpes: list[float] = []
    benchmark_sortinos: list[float] = []
    benchmark_drawdowns: list[float] = []
    strategy_sharpes: list[float] = [w.metrics_result.sharpe_ratio for w in valid]
    all_active_returns: list[np.ndarray] = []

    for window in valid:
        window_data = data.loc[window.test_start:window.test_end, "close"]
        bh_returns = _buy_and_hold_returns(window_data)
        benchmark_sharpes.append(_sharpe(bh_returns))
        benchmark_sortinos.append(_sortino(bh_returns))
        benchmark_drawdowns.append(_max_drawdown(bh_returns))

        # Per-bar active returns: strategy_return[t] - bh_return[t].
        # Strategy portfolio values give daily returns; align to bh_returns index.
        pv = window.simulation_result.portfolio_values
        if pv is not None and len(pv) > 1:
            strat_returns = pv.pct_change().dropna().to_numpy()
            # bh_returns has length len(window_data)-1; strategy may differ
            # if force-close adjusted the last bar. Use the shorter length.
            n = min(len(strat_returns), len(bh_returns))
            active = strat_returns[:n] - bh_returns[:n]
            all_active_returns.append(active)

    bm_sharpe_arr = np.array(benchmark_sharpes)
    strat_sharpe_arr = np.array(strategy_sharpes)
    sharpe_diffs = strat_sharpe_arr - bm_sharpe_arr

    # Information ratio: annualised mean active return / annualised active volatility.
    # Computed from per-bar active returns concatenated across all windows.
    if all_active_returns:
        active_concat = np.concatenate(all_active_returns)
        active_std = float(np.std(active_concat, ddof=1))
        if active_std < 1e-10:
            ir = 0.0
        else:
            ir = float(np.mean(active_concat) / active_std * np.sqrt(ANNUALISATION_FACTOR))
    else:
        ir = float("nan")

    # Paired t-test on per-window Sharpe differences.
    # Tests whether the strategy consistently differs from buy-and-hold
    # in risk-adjusted terms, not just on raw returns.
    if len(sharpe_diffs) >= 2:
        t_stat, p_val = stats.ttest_1samp(sharpe_diffs, popmean=0.0)
    else:
        t_stat, p_val = float("nan"), float("nan")

    beats_fraction = float(np.mean(strat_sharpe_arr > bm_sharpe_arr))

    return BenchmarkResult(
        benchmark_sharpe=float(np.mean(bm_sharpe_arr)),
        benchmark_sortino=float(np.mean(benchmark_sortinos)),
        benchmark_max_drawdown=float(min(benchmark_drawdowns)),
        information_ratio=ir,
        sharpe_diff_t_stat=float(t_stat),
        sharpe_diff_p_value=float(p_val),
        strategy_beats_benchmark_fraction=beats_fraction,
    )


def _buy_and_hold_returns(close: pd.Series) -> np.ndarray:
    """
    Daily returns for a buy-and-hold position over a price series.

    Applies a round-trip transaction cost at entry and exit. The entry cost
    reduces the effective first-day return; the exit cost reduces the last.
    All intermediate returns are unadjusted.

    Args:
        close: Closing price series for one test window.

    Returns:
        Array of daily returns, same length as len(close) - 1.
    """
    prices = close.to_numpy(dtype=float)
    if len(prices) < 2:
        return np.array([], dtype=float)

    returns: np.ndarray = np.diff(prices) / prices[:-1]

    # Entry cost on day 0: deduct from first day's return.
    # Exit cost on last day: deduct from last day's return.
    # This mirrors how run_simulation applies costs.
    returns[0] -= TRANSACTION_COST_RATE
    returns[-1] -= TRANSACTION_COST_RATE

    return returns
