"""
Buy-and-hold benchmark comparison.

The standard critique of any active strategy is: does it beat just holding
the index? Beating buy-and-hold on raw Sharpe over one historical period is
not the right bar - buy-and-hold has survivorship bias, the comparison
ignores the path dependency of active strategies, and a single Sharpe comparison
has no notion of statistical significance.

This module computes:

  1. Buy-and-hold metrics on the same test windows used by walk_forward(),
     so the comparison uses identical data slices and identical costs.

  2. Information ratio: annualised mean active return divided by active return
     volatility. IR uses active risk (tracking error) as the denominator rather
     than total volatility, which is the right comparison when strategies differ
     in turnover and time in market. A positive IR means the strategy added
     alpha per unit of active risk.

  3. A paired t-test on per-window Sharpe differences. Because we have ~26
     independent windows, we can test whether the strategy consistently beats
     buy-and-hold across windows, not just on average. This is more meaningful
     than a single Sharpe comparison.

  4. Relative drawdown: strategy max drawdown vs benchmark max drawdown. A
     strategy with worse drawdown than buy-and-hold is hard to justify even
     if its Sharpe is similar.

These metrics are returned as a separate `BenchmarkResult` from `compute_benchmark()`
and surfaced in both the console output and the dashboard. They are deliberately
separate from `BacktestResult` rather than embedded in it: `BacktestResult`
contains everything that walk-forward produces without any benchmark assumption;
`BenchmarkResult` adds a specific normative comparison (vs buy-and-hold) that
the caller explicitly requests.

Grinold, R. & Kahn, R. (2000). Active Portfolio Management (2nd ed.).
Chapter 2 defines the information ratio and its relationship to alpha.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    TRANSACTION_COST_RATE,
)
from backtesting_engine.metrics import _max_drawdown, _sharpe, _sortino
from backtesting_engine.models import BacktestResult

if TYPE_CHECKING:
    from backtesting_engine.execution import ExecutionConfig


@dataclass(frozen=True)
class BenchmarkResult:
    """Buy-and-hold benchmark metrics computed over the same walk-forward windows."""
    benchmark_sharpe: float          # mean per-window Sharpe for buy-and-hold
    benchmark_sortino: float         # mean per-window Sortino for buy-and-hold
    benchmark_max_drawdown: float    # worst per-window drawdown (not averaged)
    information_ratio: float         # annualised active return / active return vol; IR > 0 = positive alpha
    sharpe_diff_t_stat: float        # paired t-stat for per-window Sharpe differences
    sharpe_diff_p_value: float       # two-sided p-value for the Sharpe difference t-test
    strategy_beats_benchmark_fraction: float  # fraction of windows where strategy Sharpe > B&H
    per_window_benchmark_sharpes: list[float] = dataclasses.field(default_factory=list)
    # per-window B&H Sharpe in order - used by dashboard to colour bars against the
    # correct per-window benchmark rather than the aggregate mean


def compute_benchmark(
    result: BacktestResult,
    data: pd.DataFrame,
    execution: "ExecutionConfig | None" = None,
) -> BenchmarkResult:
    """
    Compute buy-and-hold benchmark metrics over the same windows as a BacktestResult.

    The benchmark is a fully-invested long position opened at the start of each
    test window and closed at the end, with the same transaction cost AND slippage
    applied on entry and exit. Both frictions are included so the comparison is
    directly comparable: the strategy pays cost + slippage on every fill;
    the benchmark must pay the same on its single round-trip per window.

    Args:
        result: BacktestResult from walk_forward().
        data: The full price DataFrame used for the original walk_forward() call.
              Must include 'high' and 'low' columns when slippage_factor > 0.
        execution: ExecutionConfig used for the strategy run. When provided,
                   the benchmark applies the same transaction_cost_rate and
                   slippage_factor. Defaults to the global TRANSACTION_COST_RATE
                   with zero slippage for backward compatibility.

    Returns:
        BenchmarkResult with per-window and aggregate comparison metrics.

    Raises:
        ValueError: If result has no valid windows.
    """
    cost_rate = execution.transaction_cost_rate if execution is not None else TRANSACTION_COST_RATE
    slippage = execution.slippage_factor if execution is not None else 0.0

    valid = result.valid_windows
    if not valid:
        raise ValueError("BacktestResult has no valid windows to benchmark against.")

    benchmark_sharpes: list[float] = []
    benchmark_sortinos: list[float] = []
    benchmark_drawdowns: list[float] = []
    strategy_sharpes: list[float] = [w.metrics_result.sharpe_ratio for w in valid]
    all_active_returns: list[np.ndarray] = []

    for window in valid:
        window_data = data.loc[window.test_start:window.test_end]
        bh_returns = _buy_and_hold_returns(window_data, cost_rate=cost_rate, slippage_factor=slippage)
        benchmark_sharpes.append(_sharpe(bh_returns))
        benchmark_sortinos.append(_sortino(bh_returns))
        benchmark_drawdowns.append(_max_drawdown(bh_returns))

        pv = window.simulation_result.portfolio_values
        if pv is not None and len(pv) > 1:
            strat_returns = pv.pct_change().dropna().to_numpy()
            n = min(len(strat_returns), len(bh_returns))
            active = strat_returns[:n] - bh_returns[:n]
            all_active_returns.append(active)

    bm_sharpe_arr = np.array(benchmark_sharpes)
    strat_sharpe_arr = np.array(strategy_sharpes)
    sharpe_diffs = strat_sharpe_arr - bm_sharpe_arr

    if all_active_returns:
        active_concat = np.concatenate(all_active_returns)
        active_std = float(np.std(active_concat, ddof=1))
        if active_std < 1e-10:
            ir = 0.0
        else:
            ir = float(np.mean(active_concat) / active_std * np.sqrt(ANNUALISATION_FACTOR))
    else:
        ir = float("nan")

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
        per_window_benchmark_sharpes=benchmark_sharpes,
    )


def _buy_and_hold_returns(
    window_data: "pd.DataFrame | pd.Series",
    cost_rate: float = TRANSACTION_COST_RATE,
    slippage_factor: float = 0.0,
) -> np.ndarray:
    """
    Daily returns for a buy-and-hold position over one test window.

    Applies cost_rate and slippage_factor on entry (day 0) and exit (last day)
    to match the execution model used by the strategy. Slippage is modelled as
    a fraction of the daily high-low range, identical to run_simulation_with_execution.

    Args:
        window_data: Either a pd.DataFrame with a 'close' column (and optionally
                     'high'/'low' for slippage), or a plain pd.Series of closing
                     prices (backward-compatible; slippage is zero when H/L absent).
        cost_rate: Transaction cost rate per side.
        slippage_factor: Fraction of daily H-L range added to entry fill and
                         subtracted from exit fill. Requires 'high'/'low' columns
                         when window_data is a DataFrame; ignored otherwise.

    Returns:
        Array of daily returns, length len(window_data) - 1.
    """
    # Accept both a plain close Series (old API) and an OHLCV DataFrame (new API).
    if isinstance(window_data, pd.Series):
        close = window_data.to_numpy(dtype=float)
        high = close
        low = close
    else:
        close = window_data["close"].to_numpy(dtype=float)
        high = window_data["high"].to_numpy(dtype=float) if "high" in window_data.columns else close
        low = window_data["low"].to_numpy(dtype=float) if "low" in window_data.columns else close

    if len(close) < 2:
        return np.array([], dtype=float)

    returns: np.ndarray = np.diff(close) / close[:-1]

    # Entry cost + slippage drag on day 0.
    entry_range = high[0] - low[0]
    entry_slip = slippage_factor * entry_range / max(close[0], 1e-10)
    returns[0] -= cost_rate + entry_slip

    # Exit cost + slippage drag on last day.
    exit_range = high[-1] - low[-1]
    exit_slip = slippage_factor * exit_range / max(close[-1], 1e-10)
    returns[-1] -= cost_rate + exit_slip

    return returns
