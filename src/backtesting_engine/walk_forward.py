"""
Walk-forward validation orchestrator.

Slices rolling train/test windows, fits the strategy on each training period,
runs execution-realistic simulation on each test period, and aggregates results
with Fisher combined p and White's Reality Check.

The RC candidate matrix uses TEST-period returns for every grid candidate,
not training returns. Storing training returns would be circular - the winner
was selected by training Sharpe, so of course it looks best in-sample.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    BLOCK_BOOTSTRAP_SEED,
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
    bootstrap_seed: int = BLOCK_BOOTSTRAP_SEED,
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
                   Defaults to ExecutionConfig() - 0.1% cost, 5% slippage,
                   1-day signal delay. Pass ExecutionConfig(transaction_cost_rate=0,
                   slippage_factor=0, signal_delay=0) for zero-friction testing.
        bootstrap_seed: Random seed for block bootstrap and Reality Check.

    Returns:
        BacktestResult with per-window results, Fisher combined p-value,
        White's Reality Check p-value, parameter evolution, and flat-cash count.

    Raises:
        ValueError: If no valid windows could be evaluated, or if window
                    year parameters are non-positive.
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
    flat_cash_windows = 0
    window_candidate_returns: list[dict[Any, pd.Series]] = []

    while window_start + train_days + test_days <= len(data):
        train_end = window_start + train_days
        test_end  = window_start + train_days + test_days

        train_data = data.iloc[window_start:train_end]
        test_data  = data.iloc[train_end:test_end]

        strategy.fit(train_data)

        active_params    = strategy.active_params()
        formatted_params = strategy.format_params()
        param_evo_spec   = strategy.param_evolution_spec()

        context_data = _get_context(strategy, train_data)
        signals = strategy.generate_signals_with_context(context_data, test_data)

        sim = run_simulation_with_execution(test_data, signals, execution)

        if not sim.trades:
            # strategy held cash this window - valid result, not an error
            flat_cash_metrics = _flat_cash_metrics()
            window_results.append(WindowResult(
                train_start=data.index[window_start],
                train_end=data.index[train_end - 1],
                test_start=data.index[train_end],
                test_end=data.index[test_end - 1],
                simulation_result=sim,
                metrics_result=flat_cash_metrics,
                skipped=False,
                active_params=active_params,
                formatted_params=formatted_params,
                param_evolution_spec=param_evo_spec,
            ))
            flat_cash_windows += 1

            # RC parity: flat-cash windows must contribute to the RC candidate matrix
            # with zero returns, otherwise Fisher and RC are testing different windows.
            flat_candidates = strategy.candidate_test_returns(test_data, context_data)
            if flat_candidates:
                zero_candidates = {
                    k: pd.Series(np.zeros(len(v)), index=v.index, dtype=float)
                    for k, v in flat_candidates.items()
                }
                window_candidate_returns.append(zero_candidates)

            window_start += test_days
            continue

        if sim.portfolio_values is None:
            # should never happen - run_simulation_with_execution always sets this
            raise ValueError(
                "Simulation returned None portfolio_values. "
                "This indicates a bug in run_simulation_with_execution()."
            )
        metrics = calculate_metrics(sim.portfolio_values, trades=sim.trades, seed=bootstrap_seed)

        window_results.append(WindowResult(
            train_start=data.index[window_start],
            train_end=data.index[train_end - 1],
            test_start=data.index[train_end],
            test_end=data.index[test_end - 1],
            simulation_result=sim,
            metrics_result=metrics,
            skipped=False,
            active_params=active_params,
            formatted_params=formatted_params,
            param_evolution_spec=param_evo_spec,
        ))

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

    all_windows = window_results

    total_trades = sum(len(w.simulation_result.trades) for w in all_windows)
    if total_trades == 0:
        raise ValueError(
            f"Strategy '{strategy.__class__.__name__}' generated zero trades across "
            f"all {len(all_windows)} walk-forward windows. "
            "Check that strategy parameters produce signals on this data range."
        )

    summary = _build_summary_metrics(all_windows, window_candidate_returns, bootstrap_seed=bootstrap_seed)

    return BacktestResult(
        strategy_name=strategy.__class__.__name__,
        window_results=window_results,
        summary_metrics=summary,
        flat_cash_window_count=flat_cash_windows,
    )


# ---------------------------------------------------------------------------
# Strategy dispatch helpers
# ---------------------------------------------------------------------------

def _get_context(strategy: BaseStrategy, train_data: pd.DataFrame) -> pd.DataFrame:
    """Tail of training data used as warmup context before each test window."""
    n = strategy.context_window_size()
    if n <= 0:
        return train_data.iloc[0:0]
    return train_data.iloc[-n:]


# ---------------------------------------------------------------------------
# Summary metrics aggregation
# ---------------------------------------------------------------------------

def _build_summary_metrics(
    valid_windows: list[WindowResult],
    window_candidate_returns: list[dict[Any, pd.Series]],
    bootstrap_seed: int = BLOCK_BOOTSTRAP_SEED,
) -> MetricsResult:
    """
    Aggregate per-window metrics with Fisher combined p and Reality Check p.

    A few aggregation choices worth noting:
    - max_drawdown: worst (minimum) across windows, not mean. The worst-case
      drawdown is what a live trader experienced; the average obscures that.
    - calmar_ratio: computed from the stitched return series across all windows.
      Per-window averaging misses cross-window drawdowns - a 10% loss at the end
      of window N followed by 10% at the start of window N+1 is a -19% compound
      drawdown, not -10%.
    - inf values are excluded from means (treat as undefined for that window).
    """
    def mean_metric(attr: str) -> float:
        vals = [getattr(w.metrics_result, attr) for w in valid_windows]
        finite_vals = [v for v in vals if not (math.isnan(v) or abs(v) == float("inf"))]
        if not finite_vals:
            return float("inf")
        return float(np.mean(finite_vals))

    worst_max_dd = float(min(
        w.metrics_result.max_drawdown for w in valid_windows
    ))

    stitched_calmar = _calmar_from_stitched(valid_windows)

    fisher_p = _fisher_combined_p([w.metrics_result.p_value for w in valid_windows])

    rc_p = float("nan")
    if window_candidate_returns:
        try:
            arr_dicts: list[dict[Any, np.ndarray]] = [
                {k: v.to_numpy() for k, v in w.items()}
                for w in window_candidate_returns
            ]
            candidate_matrix = build_candidate_return_matrix(arr_dicts)
            rc_p = white_reality_check(candidate_matrix, seed=bootstrap_seed)
        except ValueError:
            # no candidate pair was evaluated in every window - RC not computable
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
        exposure_fraction=mean_metric("exposure_fraction"),
        trade_count=sum(w.metrics_result.trade_count for w in valid_windows),
        win_rate=mean_metric("win_rate"),
        avg_win_loss_ratio=mean_metric("avg_win_loss_ratio"),
        avg_holding_days=mean_metric("avg_holding_days"),
    )


def _calmar_from_stitched(valid_windows: list[WindowResult]) -> float:
    """
    Compute Calmar from the stitched portfolio across all walk-forward windows.

    Necessary because max drawdown can span window boundaries. Per-window
    averaging misses that.
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
    Fisher's combined probability test: -2 Σ ln(pᵢ) ~ χ²(2k).

    Windows are not strictly independent (rolling data overlap) so this is
    approximate. Treat the result as directional, not a precise threshold.
    """
    clipped = np.clip(p_values, 1e-300, 1.0 - 1e-10)
    chi2_stat = -2.0 * np.sum(np.log(clipped))
    return float(stats.chi2.sf(chi2_stat, 2 * len(p_values)))


def _flat_cash_metrics() -> MetricsResult:
    """
    Metrics for a window where the strategy held cash (no trades executed).

    sharpe/sortino = 0.0 (not NaN or inf) so flat-cash windows are included
    in summary means rather than silently dropped. omega = 1.0 for the same
    reason. p_value = 1.0, maximally consistent with the null.
    """
    return MetricsResult(
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=float("nan"),
        omega_ratio=1.0,
        p_value=1.0,
        combined_p_value=float("nan"),
        reality_check_p_value=float("nan"),
        exposure_fraction=0.0,
        trade_count=0,
        win_rate=float("nan"),
        avg_win_loss_ratio=float("nan"),
        avg_holding_days=float("nan"),
    )
