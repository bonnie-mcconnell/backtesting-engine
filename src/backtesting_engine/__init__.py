"""
backtesting-engine - Walk-forward validated backtesting with statistical rigour.

Public API
----------
Run both strategies with a single call:

    from backtesting_engine import walk_forward, MovingAverageStrategy, KalmanFilterStrategy
    from backtesting_engine import ExecutionConfig, build_dashboard, load_data, validate_data

    data = load_data("SPY", "1993-01-01")
    validate_data(data, min_rows=1260)

    result = walk_forward(
        data,
        MovingAverageStrategy(),
        execution=ExecutionConfig(slippage_factor=0.05, signal_delay=1),
    )
    build_dashboard(result, output_path=Path("dashboard.html"))

For cost sensitivity analysis:

    from backtesting_engine import cost_sensitivity_sweep
    sweep = cost_sensitivity_sweep(data, MovingAverageStrategy(),
                                   cost_rates=[0.001, 0.002],
                                   slippage_factors=[0.0, 0.05])
"""

from backtesting_engine.benchmark import BenchmarkResult, compute_benchmark
from backtesting_engine.dashboard import build_dashboard
from backtesting_engine.data.ingestion import load_data
from backtesting_engine.data.validator import validate_data
from backtesting_engine.execution import (
    ExecutionConfig,
    cost_sensitivity_sweep,
    run_simulation_with_execution,
)
from backtesting_engine.metrics import calculate_metrics
from backtesting_engine.models import (
    BacktestResult,
    MetricsResult,
    SimulationResult,
    Trade,
    WindowResult,
)
from backtesting_engine.reality_check import build_candidate_return_matrix, white_reality_check
from backtesting_engine.strategy.base import BaseStrategy, returns_from_signals
from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
from backtesting_engine.strategy.momentum import MomentumStrategy
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward

__all__ = [
    # Core pipeline
    "walk_forward",
    "calculate_metrics",
    # Strategies
    "BaseStrategy",
    "returns_from_signals",
    "MovingAverageStrategy",
    "MomentumStrategy",
    "KalmanFilterStrategy",
    # Execution
    "ExecutionConfig",
    "run_simulation_with_execution",
    "cost_sensitivity_sweep",
    # Data
    "load_data",
    "validate_data",
    # Results
    "BacktestResult",
    "MetricsResult",
    "SimulationResult",
    "Trade",
    "WindowResult",
    # Visualisation
    "build_dashboard",
    # Benchmark
    "compute_benchmark",
    "BenchmarkResult",
    # Statistical testing
    "white_reality_check",
    "build_candidate_return_matrix",
]
