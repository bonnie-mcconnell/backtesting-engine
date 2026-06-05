"""
Typed data contracts between pipeline components.

Trade, MetricsResult, and WindowResult are frozen=True (immutable after creation).
SimulationResult and BacktestResult are not frozen - they hold mutable pd.Series
and lists that can't be hashed. Treat both as read-only after construction.
"""

import warnings
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Trade:
    """A single completed round-trip (entry + exit)."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    transaction_costs: float  # total round-trip cost in dollars
    pnl: float                # net P&L after all transaction costs


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

    All ratio metrics are annualised (252 trading days). p_value is the
    per-window block-bootstrap Sharpe p-value. combined_p_value and
    reality_check_p_value are populated on BacktestResult.summary_metrics
    after the full walk-forward completes; they're NaN on individual windows.

    Trade diagnostics (exposure_fraction through avg_holding_days) are NaN
    when trades were not provided to calculate_metrics().
    """
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float           # fraction, e.g. -0.15 = -15%
    calmar_ratio: float
    omega_ratio: float
    p_value: float
    combined_p_value: float = float("nan")
    reality_check_p_value: float = float("nan")
    # trade diagnostics - only populated when trades are passed to calculate_metrics()
    exposure_fraction: float = float("nan")
    trade_count: int = 0
    win_rate: float = float("nan")
    avg_win_loss_ratio: float = float("nan")
    avg_holding_days: float = float("nan")


@dataclass(frozen=True)
class WindowResult:
    """Results for one walk-forward window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    simulation_result: SimulationResult
    metrics_result: MetricsResult
    skipped: bool = False
    # always False in current code; kept for API compat (see __post_init__)
    active_params: dict[str, object] = field(default_factory=dict)
    # calibrated params for this window, e.g. {'short_window': 50, 'long_window': 200}
    formatted_params: str = ""
    param_evolution_spec: list[tuple[str, str]] = field(default_factory=list)
    # list of (display_label, active_params_key) pairs for the dashboard param panel

    def __post_init__(self) -> None:
        if self.skipped:
            warnings.warn(
                "WindowResult.skipped=True is deprecated. No-trade windows are now "
                "valid flat-cash windows (Sharpe=0, p=1.0). Remove skipped=True.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class BacktestResult:
    """
    Aggregated output of a full walk-forward run.

    window_results contains every window attempted, including flat-cash windows
    where the strategy held cash and made no trades. summary_metrics is computed
    over all windows; flat-cash windows contribute Sharpe=0 and p=1.0.
    """
    strategy_name: str
    window_results: list[WindowResult]
    summary_metrics: MetricsResult
    flat_cash_window_count: int = 0
    # windows where the strategy held cash the entire period - valid results, not errors

    @property
    def valid_windows(self) -> list[WindowResult]:
        """All walk-forward windows, including flat-cash (no-trade) windows.

        To inspect only windows with actual trades:
            traded = [w for w in result.valid_windows if w.simulation_result.trades]
        """
        return [w for w in self.window_results if not w.skipped]

    @property
    def param_evolution(self) -> list[dict[str, object]]:
        """Per-window active_params dicts in time order, for plotting parameter drift."""
        return [w.active_params for w in self.valid_windows if w.active_params]
