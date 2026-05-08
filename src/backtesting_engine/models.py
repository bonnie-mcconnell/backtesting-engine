"""
Typed data contracts between pipeline components.

Immutability notes:
  - Trade, MetricsResult, WindowResult: frozen=True - immutable, hashable.
  - SimulationResult: not frozen - portfolio_values is a mutable pd.Series.
  - BacktestResult: not frozen - window_results is a mutable list, and nested
    pd.Series objects are not hashable, so frozen=True would require a custom
    __hash__. Treat BacktestResult as read-only after construction.

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

    Trade diagnostics
    -----------------
    exposure_fraction : float
        Fraction of bars where the strategy held a position (0–1). A strategy
        that is in-market 30% of the time has exposure_fraction = 0.30.
        Lower exposure = more time in cash = less market beta, but also less
        opportunity to earn returns. NaN if no portfolio values available.
    trade_count : int
        Total number of completed round-trip trades in this window.
    win_rate : float
        Fraction of trades with positive net P&L (0–1). NaN if no trades.
    avg_win_loss_ratio : float
        Mean winning P&L / mean losing P&L (absolute). A ratio > 1 means
        winners are larger than losers on average. NaN if no wins or no losses.
    avg_holding_days : float
        Mean number of calendar days between trade entry and exit. NaN if no trades.
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
    """Results for one walk-forward window.

    Attributes
    ----------
    active_params : dict[str, object]
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
    # Always False in current code. Retained for API compatibility - callers who
    # pass skipped=True explicitly will receive a DeprecationWarning. Previously
    # used to mark no-trade windows; those are now valid flat-cash windows
    # (Sharpe=0, p=1.0). Will be removed in a future major version.
    active_params: dict[str, object] = field(default_factory=dict)
    # formatted_params: human-readable parameter string from strategy.format_params().
    # Stored at window creation time so the orchestrator and dashboard never
    # need to know which strategy class was used.  Empty string = no parameters
    # (e.g. a parameter-free strategy).
    # Note: Python dataclasses do not support PEP 257 field docstrings - a
    # triple-quoted string after a field is a free-floating string literal with
    # no effect on help(), IDE tooltips, or any tooling. Use comments instead.
    formatted_params: str = ""
    # param_evolution_spec: list of (display_label, active_params_key) pairs
    # returned by strategy.param_evolution_spec() at window creation time.
    # The dashboard uses this to render the parameter evolution panel without
    # any isinstance checks or strategy-specific key names.
    param_evolution_spec: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.skipped:
            import warnings
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
    # flat_cash_window_count: number of walk-forward windows where the strategy
    # held cash (no trades executed).  These windows ARE included in summary
    # metrics with Sharpe=0 and p_value=1.0 - a cash-holding window is a valid
    # out-of-sample result and must be counted in performance attribution.
    # Note: triple-quoted strings after dataclass fields are free-floating string
    # literals with no effect on tooling; use inline comments (this) instead.
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
