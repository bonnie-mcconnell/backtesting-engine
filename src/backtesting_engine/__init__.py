"""
backtesting-engine - Walk-forward validated backtesting with statistical rigour.

    from backtesting_engine import load_data, validate_data, walk_forward, MovingAverageStrategy
    from backtesting_engine.config import ANNUALISATION_FACTOR

    data = load_data("SPY", "1993-01-01", end_date="2024-12-31")
    validate_data(data, min_rows=4 * ANNUALISATION_FACTOR)  # (3+1)*252 for one window
    result = walk_forward(data, MovingAverageStrategy())

ExecutionConfig defaults (0.1% cost, 5% slippage, 1-day delay) match the CLI.
See the README's "Library usage" section for cost sensitivity sweeps, zero-friction
runs, reproducible seeded runs, and cross-asset validation - this docstring stays
short on purpose so it doesn't drift out of sync with the README's longer version.
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
from backtesting_engine.multi_asset import run_multi_asset
from backtesting_engine.reality_check import build_candidate_return_matrix, white_reality_check
from backtesting_engine.strategy.base import BaseStrategy, returns_from_signals
from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
from backtesting_engine.strategy.momentum import MomentumStrategy
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.summary import write_summary_csv, write_summary_json
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
    # Summary output
    "write_summary_json",
    "write_summary_csv",
    # Cross-asset validation
    "run_multi_asset",
]
