"""
Walk-forward orchestrator.
Slices data into training/testing windows, runs the strategy on test data, simulates trades, and calculates performance metrics for each window.
"""
import pandas as pd
import numpy as np

from backtesting_engine.models import WindowResult, BacktestResult, MetricsResult
from backtesting_engine.metrics import calculate_metrics
from backtesting_engine.strategy.base import BaseStrategy
from backtesting_engine.simulator import run_simulation
from backtesting_engine.config import TRAINING_WINDOW_YEARS, TESTING_WINDOW_YEARS, ANNUALISATION_FACTOR

def walk_forward(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    training_window_years: int = TRAINING_WINDOW_YEARS,
    testing_window_years: int = TESTING_WINDOW_YEARS,
) -> BacktestResult:
    """
    Performs walk-forward analysis on the given data using the specified strategy.

    Args:
        data (pd.DataFrame): The historical price data for backtesting.
        strategy (BaseStrategy): The trading strategy to be evaluated.
        training_window_years (int): The number of years to use for the training window.
        testing_window_years (int): The number of years to use for the testing window.

    Returns:
        BacktestResult: The results of the walk-forward analysis, including performance metrics and trade details.
    """

    train_days = training_window_years * ANNUALISATION_FACTOR
    test_days = testing_window_years * ANNUALISATION_FACTOR

    window_start = 0
    window_results: list[WindowResult] = []
    
    while window_start + train_days + test_days <= len(data):
        train_start = window_start
        train_end = window_start + train_days
        test_start = window_start + train_days
        test_end = window_start + train_days + test_days

        # train_data available for strategies requiring parameter calibration
        test_data = data.iloc[test_start:test_end]

        # run simulator for strategy on test data
        signals = strategy.generate_signals(test_data)
        
        simulation_result = run_simulation(test_data, signals)
        
        if not simulation_result.trades or simulation_result.portfolio_values is None:
            window_start += test_days
            continue

        # run metrics on simulation result
        metrics = calculate_metrics(simulation_result.portfolio_values)
        
        window_results.append(WindowResult(
            train_start=data.index[train_start],
            train_end=data.index[train_end - 1],
            test_start=data.index[test_start],
            test_end=data.index[test_end - 1],
            simulation_result=simulation_result,
            metrics_result=metrics,
        ))

        window_start += test_days

    if not window_results:
        raise ValueError("No valid windows were processed. Check if the data length is sufficient for the specified training and testing window sizes.")
    
    mean_sharpe = _aggregate_metric(window_results, "sharpe_ratio")
    mean_sortino = _aggregate_metric(window_results, "sortino_ratio")
    mean_max_drawdown = _aggregate_metric(window_results, "max_drawdown")
    mean_calmar = _aggregate_metric(window_results, "calmar_ratio")
    mean_omega = _aggregate_metric(window_results, "omega_ratio")
    mean_p_value = _aggregate_metric(window_results, "p_value")

    summary_metrics = MetricsResult(
        sharpe_ratio=mean_sharpe,
        sortino_ratio=mean_sortino,
        max_drawdown=mean_max_drawdown,
        calmar_ratio=mean_calmar,
        omega_ratio=mean_omega,
        p_value=mean_p_value,
    )

    return BacktestResult(
        strategy_name=strategy.__class__.__name__,
        window_results=window_results,
        summary_metrics=summary_metrics
    )
    

def _aggregate_metric(window_results: list[WindowResult], metric: str) -> float:
    """
    Compute the mean of a named metric across all walk-forward windows.

    Args:
        window_results: List of completed window results.
        metric: Field name on MetricsResult to aggregate (e.g. 'sharpe_ratio').

    Returns:
        Mean value of the metric across all windows.
    """
    values = [getattr(w.metrics_result, metric) for w in window_results]
    return float(np.mean(values))
