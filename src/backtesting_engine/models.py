"""
Data models defining the contracts between pipeline components.
All models use dataclasses to eliminate boilerplate while maintaining clear field definitions.
"""
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    transaction_costs: float
    pnl: float     # net profit or loss for single trade


@dataclass
class SimulationResult:
    trades: list[Trade]
    portfolio_values: pd.Series | None = None
    message: str = ""

@dataclass(frozen=True)
class MetricsResult:
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    omega_ratio: float
    p_value: float


@dataclass
class WindowResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    simulation_result: SimulationResult
    metrics_result: MetricsResult


@dataclass
class BacktestResult:
    strategy_name: str
    window_results: list[WindowResult]
    summary_metrics: MetricsResult

