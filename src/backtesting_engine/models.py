"""
Typed data contracts between pipeline components.

All result types are frozen dataclasses - immutable after construction,
hashable for use in sets and dicts, and safe against downstream mutation.

Design note on WindowResult.active_params
------------------------------------------
Storing the active parameters (e.g. calibrated MA windows or Kalman Q/R)
on each WindowResult makes the parameter evolution visible without requiring
callers to instrument the strategy. This is essential for two things:
  1. The dashboard parameter evolution panel (shows how parameters drift
     across training windows, revealing regime-dependent adaptation)
  2. Reproducibility - given a BacktestResult you can re-run any individual
     window exactly without re-running the full walk-forward.
"""

from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class Trade:
    """A single completed round-trip (entry + exit)."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    transaction_costs: float   # total round-trip cost in dollars
    pnl: float                 # net P&L after all transaction costs


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

    All ratio metrics are annualised (252 trading days).

    Attributes
    ----------
    p_value : float
        Per-window block-bootstrap Sharpe p-value.
    combined_p_value : float
        Fisher's combined p-value across all walk-forward windows.
        NaN on individual windows; populated on BacktestResult.summary_metrics.
    reality_check_p_value : float
        White's Reality Check p-value, corrected for data-snooping across the
        parameter search grid. NaN for strategies without parameter search
        (KalmanFilterStrategy) or when no candidate returns were collected.
    """
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float          # fraction, e.g. -0.15 = -15%
    calmar_ratio: float
    omega_ratio: float
    p_value: float
    combined_p_value: float = float("nan")
    reality_check_p_value: float = float("nan")


@dataclass(frozen=True)
class WindowResult:
    """Results for one walk-forward window.

    Attributes
    ----------
    active_params : dict[str, Any]
        The calibrated parameters used for this window's test period.
        For MovingAverageStrategy: {'short_window': int, 'long_window': int}.
        For KalmanFilterStrategy: {'q': float, 'r': float, 'log_likelihood': float}.
        Empty dict for parameter-free strategies.
        Stored here (rather than on the strategy object) so the full history
        of parameter evolution across windows is preserved in BacktestResult.
    """
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    simulation_result: SimulationResult
    metrics_result: MetricsResult
    skipped: bool = False
    active_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """
    Aggregated output of a full walk-forward run.

    window_results contains every window attempted, including skipped ones.
    summary_metrics is computed only over non-skipped windows.
    """
    strategy_name: str
    window_results: list[WindowResult]
    summary_metrics: MetricsResult
    skipped_window_count: int = 0

    @property
    def valid_windows(self) -> list[WindowResult]:
        """Non-skipped windows that produced at least one trade."""
        return [w for w in self.window_results if not w.skipped]

    @property
    def param_evolution(self) -> list[dict[str, Any]]:
        """
        List of active_params dicts from each valid window, in time order.

        Useful for plotting how calibrated parameters evolved across the
        walk-forward horizon. Returns an empty list if no params were stored.
        """
        return [w.active_params for w in self.valid_windows if w.active_params]