"""
Walk-forward validation orchestrator.

Walk-forward validation addresses the core problem with single-split backtesting:
one out-of-sample period might be lucky or unlucky, and you cannot tell which.
Rolling windows produce multiple independent out-of-sample evaluations, and
consistency across windows is evidence of robustness.

For each window:
  1. Slice training data [window_start, train_end).
  2. Call strategy.fit(train_data) - parameters are calibrated in-sample.
  3. Slice test data [train_end, test_end).
  4. Call strategy.generate_signals_with_context(context, test_data) - warmup
     context eliminates NaN MAs at the start of the test period.
  5. Run the simulator on test data and test signals.
  6. Compute performance metrics.
  7. Advance window_start by one test period and repeat.

The training window slides forward with the test window, so the strategy is
always calibrated on the three years immediately preceding each test period.
This means each window captures the market regime most recent to the test period,
rather than a fixed early-history regime.

Statistical significance is assessed via Fisher's combined p-value method across
all walk-forward windows. Fisher's method (-2 * sum(log(p_i))) follows a
chi-squared distribution with 2k degrees of freedom under the joint null, and is
more powerful than simply averaging individual p-values (which is not a valid
statistical operation).
"""

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    TESTING_WINDOW_YEARS,
    TRAINING_WINDOW_YEARS,
)
from backtesting_engine.metrics import calculate_metrics
from backtesting_engine.models import BacktestResult, MetricsResult, WindowResult
from backtesting_engine.simulator import run_simulation
from backtesting_engine.strategy.base import BaseStrategy
from backtesting_engine.strategy.moving_average import MovingAverageStrategy


def walk_forward(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    training_window_years: int = TRAINING_WINDOW_YEARS,
    testing_window_years: int = TESTING_WINDOW_YEARS,
) -> BacktestResult:
    """
    Perform walk-forward validation on the given data using the given strategy.

    Args:
        data: Full historical price DataFrame (DatetimeIndex, 'close' column).
        strategy: Trading strategy implementing BaseStrategy (fit + generate_signals).
        training_window_years: Number of years per training window.
        testing_window_years: Number of years per test window.

    Returns:
        BacktestResult containing all window results, summary metrics,
        and a count of windows that were skipped due to no trades.

    Raises:
        ValueError: If no valid windows could be evaluated (data too short,
                    or no window produced any trades).
    """
    train_days = training_window_years * ANNUALISATION_FACTOR
    test_days = testing_window_years * ANNUALISATION_FACTOR

    window_start = 0
    window_results: list[WindowResult] = []
    skipped = 0

    while window_start + train_days + test_days <= len(data):
        train_end = window_start + train_days
        test_end = window_start + train_days + test_days

        train_data = data.iloc[window_start:train_end]
        test_data = data.iloc[train_end:test_end]

        # Step 1: Calibrate strategy parameters in-sample.
        strategy.fit(train_data)

        # Step 2: Generate signals with warmup context to eliminate NaN MAs.
        # Pass the last long_window rows of training data as context so the
        # strategy's moving averages are fully warmed up on day 1 of the test.
        if isinstance(strategy, MovingAverageStrategy):
            context_rows = strategy.long_window_
            context_data = train_data.iloc[-context_rows:]
            signals = strategy.generate_signals_with_context(context_data, test_data)
        else:
            signals = strategy.generate_signals(test_data)

        # Step 3: Simulate trades on the test period.
        simulation_result = run_simulation(test_data, signals)

        if not simulation_result.trades:
            # Record the skipped window so the caller has full visibility.
            window_results.append(WindowResult(
                train_start=data.index[window_start],
                train_end=data.index[train_end - 1],
                test_start=data.index[train_end],
                test_end=data.index[test_end - 1],
                simulation_result=simulation_result,
                metrics_result=_empty_metrics(),
                skipped=True,
            ))
            skipped += 1
            window_start += test_days
            continue

        # Step 4: Compute metrics on the out-of-sample test period.
        assert simulation_result.portfolio_values is not None
        metrics = calculate_metrics(simulation_result.portfolio_values)

        window_results.append(WindowResult(
            train_start=data.index[window_start],
            train_end=data.index[train_end - 1],
            test_start=data.index[train_end],
            test_end=data.index[test_end - 1],
            simulation_result=simulation_result,
            metrics_result=metrics,
            skipped=False,
        ))

        window_start += test_days

    if not window_results:
        raise ValueError(
            "No windows could be evaluated. "
            "Check that the data is long enough for the configured window sizes "
            f"(need at least {training_window_years + testing_window_years} years of data)."
        )

    valid = [w for w in window_results if not w.skipped]
    if not valid:
        raise ValueError(
            "Every window was skipped because no trades were generated. "
            "The strategy may not produce signals for the configured data range."
        )

    summary = _build_summary_metrics(valid)

    return BacktestResult(
        strategy_name=strategy.__class__.__name__,
        window_results=window_results,
        summary_metrics=summary,
        skipped_window_count=skipped,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_summary_metrics(valid_windows: list[WindowResult]) -> MetricsResult:
    """
    Aggregate per-window metrics into a single summary.

    Simple metrics (Sharpe, Sortino, etc.) are mean-aggregated across windows.
    The combined p-value uses Fisher's method, which is the statistically
    correct approach for combining independent p-values.

    Fisher's method: -2 * sum(log(p_i)) ~ chi-squared(2k) under joint null.
    This is more sensitive than averaging because it is dominated by
    windows with very small p-values (strong evidence), rather than giving
    equal weight to windows with p near 0.5.

    Args:
        valid_windows: Non-skipped WindowResult objects.

    Returns:
        MetricsResult with aggregated values and Fisher combined_p_value.
    """
    def mean_metric(attr: str) -> float:
        return float(np.mean([getattr(w.metrics_result, attr) for w in valid_windows]))

    combined_p = _fisher_combined_p([w.metrics_result.p_value for w in valid_windows])

    return MetricsResult(
        sharpe_ratio=mean_metric("sharpe_ratio"),
        sortino_ratio=mean_metric("sortino_ratio"),
        max_drawdown=mean_metric("max_drawdown"),
        calmar_ratio=mean_metric("calmar_ratio"),
        omega_ratio=mean_metric("omega_ratio"),
        p_value=mean_metric("p_value"),       # per-window mean, for reference
        combined_p_value=combined_p,          # Fisher combined - use this for inference
    )


def _fisher_combined_p(p_values: list[float]) -> float:
    """
    Fisher's combined probability test across multiple independent p-values.

    Under the joint null hypothesis that no window shows real strategy edge,
    the test statistic X = -2 * sum(ln(p_i)) follows chi-squared(2k) where
    k is the number of p-values. The combined p-value is P(chi2 >= X).

    p-values of exactly 0 are clipped to a small positive value to avoid
    log(0). p-values of exactly 1 are clipped below 1 for the same reason.

    Args:
        p_values: List of per-window p-values in (0, 1].

    Returns:
        Combined p-value in [0, 1]. Small values mean the joint evidence
        across windows is unlikely under the null.
    """
    clipped = np.clip(p_values, 1e-300, 1.0 - 1e-10)
    chi2_stat = -2.0 * np.sum(np.log(clipped))
    df = 2 * len(p_values)
    # Survival function: P(chi2 >= observed)
    return float(stats.chi2.sf(chi2_stat, df))


def _empty_metrics() -> MetricsResult:
    """Placeholder MetricsResult for skipped (no-trade) windows."""
    return MetricsResult(
        sharpe_ratio=float("nan"),
        sortino_ratio=float("nan"),
        max_drawdown=float("nan"),
        calmar_ratio=float("nan"),
        omega_ratio=float("nan"),
        p_value=float("nan"),
        combined_p_value=float("nan"),
    )