"""
Typed data contracts between pipeline components.

Trade, MetricsResult, and WindowResult are frozen=True (immutable, hashable).
SimulationResult is not frozen because portfolio_values is a mutable pd.Series.
BacktestResult is not frozen because window_results is a mutable list and the
nested pd.Series objects are not hashable. Treat it as read-only after construction.

active_params is stored on each WindowResult rather than on the strategy object
so the full history of parameter evolution is preserved in BacktestResult. This
makes the parameter evolution dashboard panel work without re-running the strategy,
and lets you reproduce any individual window without re-running the full walk-forward.
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

    All ratio metrics are annualised (252 trading days). p_value is the
    per-window block-bootstrap Sharpe p-value. combined_p_value and
    reality_check_p_value are NaN on individual windows and populated on
    BacktestResult.summary_metrics after the full walk-forward completes.
    Trade diagnostics (exposure_fraction through avg_holding_days) are NaN
    when trades were not provided to calculate_metrics().
    """
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float          # fraction, e.g. -0.15 = -15%
    calmar_ratio: float
    omega_ratio: float
    p_value: float
    combined_p_value: float = float("nan")
    reality_check_p_value: float = float("nan")
    # Trade diagnostics (populated by calculate_metrics when trades are available)
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
    # Always False in current code. Retained for API compatibility: callers who
    # pass skipped=True will receive a DeprecationWarning. Previously used to mark
    # no-trade windows; those are now valid flat-cash windows (Sharpe=0, p=1.0).
    active_params: dict[str, object] = field(default_factory=dict)
    # Calibrated parameters for this window's test period, e.g.
    # {'short_window': 50, 'long_window': 200}. Stored on WindowResult rather
    # than the strategy so the full evolution history is preserved in
    # BacktestResult and any window can be reproduced independently.
    formatted_params: str = ""
    # param_evolution_spec: list of (display_label, active_params_key) pairs
    # returned by strategy.param_evolution_spec() at window creation time.
    # The dashboard uses this to render the parameter evolution panel without
    # any isinstance checks or strategy-specific key names.
    param_evolution_spec: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.skipped:
            warnings.warn(
                "WindowResult.skipped=True is deprecated and will be removed in a "
                "future version. No-trade windows are now valid flat-cash windows "
                "(Sharpe=0, p=1.0) and are never marked as skipped. "
                "Remove the skipped=True argument from your code.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class BacktestResult:
    """
    Aggregated output of a full walk-forward run.

    window_results contains every window attempted, including flat-cash windows
    (windows where the strategy held cash and made no trades).
    summary_metrics is computed over all windows, with flat-cash windows
    contributing Sharpe=0 and p=1.0.

    Not frozen: window_results is a mutable list, and pd.Series in nested
    SimulationResult objects are not hashable, so frozen=True would require
    a custom __hash__. The trade-off is intentional - treat this as read-only
    after construction.
    """
    strategy_name: str
    window_results: list[WindowResult]
    summary_metrics: MetricsResult
    # flat_cash_window_count: windows where the strategy held cash (no trades
    # executed). Included in summary metrics with Sharpe=0 and p_value=1.0.
    # A cash-holding window is a valid out-of-sample result, not an error.
    flat_cash_window_count: int = 0

    @property
    def valid_windows(self) -> list[WindowResult]:
        """All walk-forward windows, including flat-cash (no-trade) windows.

        Every window has skipped=False since flat-cash windows are valid results,
        not skipped ones.  A window where the strategy held cash contributes
        Sharpe=0 and p=1.0 to the aggregate summary.

        To inspect only windows with actual trades, filter explicitly:
            traded = [w for w in result.valid_windows if w.simulation_result.trades]
        """
        return [w for w in self.window_results if not w.skipped]

    @property
    def param_evolution(self) -> list[dict[str, object]]:
        """
        List of active_params dicts from each valid window, in time order.

        Useful for plotting how calibrated parameters evolved across the
        walk-forward horizon. Returns an empty list if no params were stored.
        """
        return [w.active_params for w in self.valid_windows if w.active_params]
