"""
backtesting-engine - Walk-forward validated backtesting with statistical rigour.

Public API
----------
The three things most users need:

    from backtesting_engine import walk_forward, MovingAverageStrategy, KalmanFilterStrategy

    result = walk_forward(data, MovingAverageStrategy())
    result = walk_forward(data, KalmanFilterStrategy())

For execution realism (slippage, signal delay):

    from backtesting_engine import walk_forward, ExecutionConfig
    result = walk_forward(data, strategy, execution=ExecutionConfig(slippage_factor=0.05))

For the interactive dashboard:

    from backtesting_engine import build_dashboard
    build_dashboard(result, output_path=Path("dashboard.html"))

For data loading:

    from backtesting_engine import load_data, validate_data
"""

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
from backtesting_engine.strategy.base import BaseStrategy
from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward

__all__ = [
    # Core pipeline
    "walk_forward",
    "calculate_metrics",
    # Strategies
    "BaseStrategy",
    "MovingAverageStrategy",
    "KalmanFilterStrategy",
    # Execution
    "ExecutionConfig",
    "cost_sensitivity_sweep",
    "run_simulation_with_execution",
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
]