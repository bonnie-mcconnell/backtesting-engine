"""
Walk-forward validation orchestrator.

For each rolling window the orchestrator:
  1. Slices training data and calls strategy.fit(train_data).
  2. Generates out-of-sample signals with warmup context.
  3. Simulates trades via the configured execution model.
  4. Computes per-window metrics and block-bootstrap p-value.
  5. Collects all candidate TEST-period returns for White's Reality Check.
  6. Stores calibrated parameters on WindowResult for drift visualisation.

Statistical aggregation
-----------------------
Fisher's combined p-value (-2 Σ ln(pᵢ) ~ χ²(2k)) is the primary
significance metric across windows.

White's Reality Check is the secondary metric for strategies with parameter
search. It uses TEST-period returns of every candidate, not training returns.
This is the critical distinction: storing training returns would be circular
(the winner was selected by training Sharpe, so of course it looks best
in-sample). The Reality Check must use out-of-sample returns for all
candidates to test whether ANY candidate beat the benchmark by luck.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    TESTING_WINDOW_YEARS,
    TRAINING_WINDOW_YEARS,
)
from backtesting_engine.execution import ExecutionConfig, run_simulation_with_execution
from backtesting_engine.metrics import _calmar, calculate_metrics
from backtesting_engine.models import BacktestResult, MetricsResult, WindowResult
from backtesting_engine.reality_check import build_candidate_return_matrix, white_reality_check
from backtesting_engine.strategy.base import BaseStrategy


def walk_forward(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    training_window_years: int = TRAINING_WINDOW_YEARS,
    testing_window_years: int = TESTING_WINDOW_YEARS,
    execution: ExecutionConfig | None = None,
) -> BacktestResult:
    """
    Perform walk-forward validation with execution config, Reality Check,
    and parameter evolution tracking.

    Args:
        data: Full historical OHLCV DataFrame (DatetimeIndex, 'close' required;
              'high' and 'low' required when execution.slippage_factor > 0).
        strategy: Trading strategy implementing BaseStrategy.
        training_window_years: Years per training window.
        testing_window_years: Years per test window.
        execution: ExecutionConfig controlling costs, slippage, and delay.
                   Defaults to standard config (no slippage, no delay).

    Returns:
        BacktestResult with per-window results, Fisher combined p-value,
        White's Reality Check p-value, parameter evolution, and skipped count.

    Raises:
        ValueError: If no valid windows could be evaluated, or if
                    training_window_years or testing_window_years are non-positive.
    """
    if training_window_years <= 0 or testing_window_years <= 0:
        raise ValueError(
            f"training_window_years and testing_window_years must be positive integers, "
            f"got training={training_window_years}, testing={testing_window_years}."
        )

    if execution is None:
        execution = ExecutionConfig()

    train_days = training_window_years * ANNUALISATION_FACTOR
    test_days = testing_window_years * ANNUALISATION_FACTOR

    window_start = 0
    window_results: list[WindowResult] = []
    skipped = 0

    # Collect (short, long) → returns dicts per window, for Reality Check.
    # Key insight: these are TEST-period returns for every candidate, collected
    # AFTER fit() so parameters are calibrated but test data is unseen.
    window_candidate_returns: list[dict[Any, pd.Series]] = []

    while window_start + train_days + test_days <= len(data):
        train_end = window_start + train_days
        test_end  = window_start + train_days + test_days

        train_data = data.iloc[window_start:train_end]
        test_data  = data.iloc[train_end:test_end]

        # ── Step 1: Calibrate parameters in-sample ──────────────────────
        strategy.fit(train_data)

        # ── Step 2: Extract active params for WindowResult ──────────────
        active_params = _extract_active_params(strategy)

        # ── Step 3: Generate signals with warmup context ─────────────────
        context_data = _get_context(strategy, train_data)
        signals = strategy.generate_signals_with_context(context_data, test_data)

        # ── Step 4: Simulate with execution model ────────────────────────
        sim = run_simulation_with_execution(test_data, signals, execution)

        if not sim.trades:
            window_results.append(WindowResult(
                train_start=data.index[window_start],
                train_end=data.index[train_end - 1],
                test_start=data.index[train_end],
                test_end=data.index[test_end - 1],
                simulation_result=sim,
                metrics_result=_empty_metrics(),
                skipped=True,
                active_params=active_params,
            ))
            skipped += 1
            window_start += test_days
            continue

        # ── Step 5: Compute out-of-sample metrics ─────────────────────────
        if sim.portfolio_values is None:
            # run_simulation_with_execution always sets portfolio_values, even
            # when no trades execute. This guard exists because asserts are
            # stripped by Python's -O flag; a None here would cause a confusing
            # AttributeError on pct_change() several frames away.
            raise ValueError(
                "Simulation returned None portfolio_values. "
                "This indicates a bug in run_simulation_with_execution()."
            )
        metrics = calculate_metrics(sim.portfolio_values)

        window_results.append(WindowResult(
            train_start=data.index[window_start],
            train_end=data.index[train_end - 1],
            test_start=data.index[train_end],
            test_end=data.index[test_end - 1],
            simulation_result=sim,
            metrics_result=metrics,
            skipped=False,
            active_params=active_params,
        ))

        # ── Step 6: Collect TEST-period candidate returns ─────────────────
        # Call candidate_test_returns() AFTER fit() so the parameter search
        # has run, but pass test_data so returns are out-of-sample.
        candidate_returns = strategy.candidate_test_returns(test_data, context_data)
        if candidate_returns:
            window_candidate_returns.append(candidate_returns)

        window_start += test_days

    if not window_results:
        raise ValueError(
            "No windows could be evaluated. "
            f"Need at least {training_window_years + testing_window_years} "
            "years of data."
        )

    valid = [w for w in window_results if not w.skipped]
    if not valid:
        raise ValueError(
            "Every window was skipped - no trades generated. "
            "Check strategy parameters and data range."
        )

    summary = _build_summary_metrics(valid, window_candidate_returns)

    return BacktestResult(
        strategy_name=strategy.__class__.__name__,
        window_results=window_results,
        summary_metrics=summary,
        skipped_window_count=skipped,
    )


# ---------------------------------------------------------------------------
# Strategy dispatch helpers
# ---------------------------------------------------------------------------

def _get_context(strategy: BaseStrategy, train_data: pd.DataFrame) -> pd.DataFrame:
    """
    Return the warmup context slice from the tail of training data.

    Uses strategy.context_window_size() so the orchestrator needs no knowledge
    of strategy internals. Every BaseStrategy subclass declares how many bars
    of warmup it requires; the orchestrator simply honours that declaration.

    This replaces the previous isinstance dispatch, which required editing the
    orchestrator every time a new strategy was added (open/closed violation).
    """
    n = strategy.context_window_size()
    if n <= 0:
        return train_data.iloc[0:0]   # empty slice, correct dtype/columns preserved
    return train_data.iloc[-n:]


def _extract_active_params(strategy: BaseStrategy) -> dict[str, object]:
    """
    Extract calibrated parameters from the strategy for WindowResult storage.

    Delegates to strategy.active_params() - the standard BaseStrategy interface.
    All strategies that calibrate parameters during fit() override active_params()
    to return their current calibrated values. The orchestrator does not need to
    know which strategy it is talking to.
    """
    return strategy.active_params()


# ---------------------------------------------------------------------------
# Summary metrics aggregation
# ---------------------------------------------------------------------------

def _build_summary_metrics(
    valid_windows: list[WindowResult],
    window_candidate_returns: list[dict[Any, pd.Series]],
) -> MetricsResult:
    """
    Aggregate per-window metrics with Fisher combined p and Reality Check p.

    Aggregation choices:
    - Sharpe, Sortino, Omega, p_value: mean across windows. This is the standard
      walk-forward aggregation - each window is an independent evaluation and
      the mean represents typical out-of-sample performance.
    - max_drawdown: worst (minimum) across windows, not mean. Users care about
      the worst-case drawdown they would have experienced, not the average.
      A strategy with -30% dd in one window and -5% in four others has a
      reported max_dd of -30%, not -10%.
    - calmar_ratio: computed from the stitched portfolio returns across all
      windows, not averaged per-window. Max drawdown is path-dependent and
      can span window boundaries; per-window averaging misses cross-window
      drawdowns and overstates the ratio by up to 3x.
    - inf values are excluded from means (treated as undefined for that window).

    Reality Check uses test-period returns of every candidate from every
    window. build_candidate_return_matrix() intersects the candidate keys
    across windows (only candidates evaluated in ALL windows are included)
    and concatenates their return series into a (T_total × k) matrix.
    """
    def mean_metric(attr: str) -> float:
        vals = [getattr(w.metrics_result, attr) for w in valid_windows]
        finite_vals = [v for v in vals if not (math.isnan(v) or abs(v) == float("inf"))]
        if not finite_vals:
            return float("inf")
        return float(np.mean(finite_vals))

    # Worst-case max drawdown across all windows (not mean).
    worst_max_dd = float(min(
        w.metrics_result.max_drawdown for w in valid_windows
    ))

    # Calmar from stitched returns - necessary because max drawdown can span windows.
    stitched_calmar = _calmar_from_stitched(valid_windows)

    fisher_p = _fisher_combined_p([w.metrics_result.p_value for w in valid_windows])

    rc_p = float("nan")
    if window_candidate_returns:
        try:
            # Convert pd.Series values to np.ndarray for the matrix builder.
            arr_dicts: list[dict[Any, np.ndarray]] = [
                {k: v.to_numpy() for k, v in w.items()}
                for w in window_candidate_returns
            ]
            candidate_matrix = build_candidate_return_matrix(arr_dicts)
            rc_p = white_reality_check(candidate_matrix)
        except ValueError:
            # build_candidate_return_matrix raises ValueError when no
            # candidate parameter pair was evaluated in every window -
            # this happens when some windows are too short for certain
            # (short, long) combinations. NaN is the correct output:
            # the Reality Check is not computable, not zero or one.
            rc_p = float("nan")

    return MetricsResult(
        sharpe_ratio=mean_metric("sharpe_ratio"),
        sortino_ratio=mean_metric("sortino_ratio"),
        max_drawdown=worst_max_dd,
        calmar_ratio=stitched_calmar,
        omega_ratio=mean_metric("omega_ratio"),
        p_value=mean_metric("p_value"),
        combined_p_value=fisher_p,
        reality_check_p_value=rc_p,
    )


def _calmar_from_stitched(valid_windows: list[WindowResult]) -> float:
    """
    Compute Calmar ratio from the stitched portfolio across all walk-forward windows.

    Calmar = annualised geometric return / abs(max drawdown) computed on the
    concatenated daily returns from every valid test window in sequence.

    This correctly captures cross-window drawdowns that per-window averaging
    misses. If strategy loses 10% at the end of window 1 and another 10% at
    the start of window 2, the stitched max_dd reflects the -19% compound
    drawdown; per-window averaging would show only -10%.
    """
    all_returns: list[np.ndarray] = []
    for w in valid_windows:
        pv = w.simulation_result.portfolio_values
        if pv is not None and len(pv) > 1:
            rets = pv.pct_change().dropna().to_numpy(dtype=float)
            if len(rets) > 0:
                all_returns.append(rets)

    if not all_returns:
        return float("nan")

    stitched = np.concatenate(all_returns)
    return _calmar(stitched)


def _fisher_combined_p(p_values: list[float]) -> float:
    """
    Fisher's combined probability test.

    Under the joint null, -2 Σ ln(pᵢ) ~ χ²(2k), where k is the number of
    windows. More powerful than averaging p-values because it weights each
    window by -ln(pᵢ) - windows with small p-values contribute more to the
    test statistic.

    Important caveat: Fisher's method is sensitive to individual strong
    signals. A single window with p=0.001 contributes as much to the χ²
    statistic as ~14 windows with p=0.5. This is a feature when one window
    represents a genuine effect, but it means the combined p-value can fall
    below 0.05 even when only one of twenty-nine windows showed anything
    noteworthy. Interpret in conjunction with the per-window p-values
    displayed in the dashboard.

    The windows are not strictly independent (rolling data overlap in adjacent
    windows) so the χ²(2k) approximation is slightly anti-conservative.
    Fisher's p is treated as a heuristic ordering criterion, not a formal test.
    """
    clipped = np.clip(p_values, 1e-300, 1.0 - 1e-10)
    chi2_stat = -2.0 * np.sum(np.log(clipped))
    return float(stats.chi2.sf(chi2_stat, 2 * len(p_values)))


def _empty_metrics() -> MetricsResult:
    nan = float("nan")
    return MetricsResult(
        sharpe_ratio=nan, sortino_ratio=nan, max_drawdown=nan,
        calmar_ratio=nan, omega_ratio=nan, p_value=nan,
        combined_p_value=nan, reality_check_p_value=nan,
    )
