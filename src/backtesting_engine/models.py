"""
Typed data contracts between pipeline components.

All models use dataclasses. Immutable results (Trade, MetricsResult,
WindowResult, BacktestResult) are frozen so downstream code cannot
accidentally mutate them after construction.
"""
from dataclasses import dataclass, field
import pandas as pd


@dataclass(frozen=True)
class Trade:
    """A single completed round-trip trade (entry + exit)."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    transaction_costs: float   # total round-trip cost in dollars
    pnl: float                 # net profit/loss after all transaction costs


@dataclass
class SimulationResult:
    """Output of a single simulation run over one data window."""
    trades: list[Trade]
    portfolio_values: pd.Series | None = None
    message: str = ""


@dataclass(frozen=True)
class MetricsResult:
    """
    Performance metrics computed from a portfolio value series.

    All ratio metrics are annualised. p_value is from the block-bootstrap
    test of the null hypothesis that the observed Sharpe arose by chance.
    combined_p_value is Fisher's combined statistic across walk-forward windows
    (only populated on BacktestResult.summary_metrics, NaN otherwise).
    """
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float        # expressed as a fraction, e.g. -0.15 means -15%
    calmar_ratio: float
    omega_ratio: float
    p_value: float             # per-window bootstrap p-value
    combined_p_value: float = float("nan")
    # Fisher's method applied across all windows - see walk_forward._fisher_combined_p


@dataclass(frozen=True)
class WindowResult:
    """Results for one walk-forward window (train slice + test slice)."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    simulation_result: SimulationResult
    metrics_result: MetricsResult
    skipped: bool = False      # True if this window was dropped (no trades generated)


@dataclass
class BacktestResult:
    """
    Aggregated output of a full walk-forward run.

    window_results contains every window attempted, including skipped ones
    (WindowResult.skipped == True), so the caller has full visibility.
    summary_metrics is computed only over non-skipped windows.
    """
    strategy_name: str
    window_results: list[WindowResult]
    summary_metrics: MetricsResult
    skipped_window_count: int = 0

    @property
    def valid_windows(self) -> list[WindowResult]:
        """Windows that produced at least one trade."""
        return [w for w in self.window_results if not w.skipped]