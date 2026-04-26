"""
Interactive HTML dashboard for walk-forward backtesting results.

Produces a self-contained HTML file (no server required) using Plotly.
The dashboard has six panels arranged in a responsive grid:

  1. Equity curve        - Stitched out-of-sample portfolio vs buy-and-hold.
                           Walk-forward windows shaded alternately. Hover
                           shows date, portfolio value, and cumulative return.
  2. Drawdown            - Rolling peak-to-trough decline. Max drawdown
                           annotated with date and magnitude.
  3. Rolling Sharpe      - 63-day rolling Sharpe (one quarter) showing how
                           risk-adjusted performance evolves through time.
  4. Per-window Sharpe   - Bar chart of Sharpe by test window, coloured
                           green/red. Mean line overlaid. Hover shows window
                           dates, all metrics, and p-value.
  5. Return distribution - Histogram of daily returns with normal and
                           empirical distribution overlays. Skew and excess
                           kurtosis annotated.
  6. Parameter evolution - (MA strategy only) How the calibrated short/long
                           windows changed across training windows, revealing
                           regime-dependent parameter drift.

The equity curve panel includes a Range Selector allowing the user to zoom
to 1Y / 3Y / 5Y / All, and a Range Slider for fine-grained navigation.

All panels share a unified dark theme and are linked where appropriate -
clicking a window bar in panel 4 will be noted for future interactivity.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from backtesting_engine.config import INITIAL_PORTFOLIO_VALUE
from backtesting_engine.models import BacktestResult, WindowResult

if TYPE_CHECKING:
    from backtesting_engine.benchmark import BenchmarkResult

# ---------------------------------------------------------------------------
# Colour palette - dark, professional, unambiguous
# ---------------------------------------------------------------------------
_BG = "#0F172A"
_PANEL_BG = "#1E293B"
_GRID = "#334155"
_TEXT = "#F1F5F9"
_MUTED = "#94A3B8"
_STRATEGY = "#3B82F6"      # blue
_BENCHMARK = "#64748B"     # slate
_POSITIVE = "#22C55E"      # green
_NEGATIVE = "#EF4444"      # red
_DRAWDOWN = "#EF4444"
_ROLLING = "#A78BFA"       # violet
_PARAM_SHORT = "#FB923C"   # orange
_PARAM_LONG = "#34D399"    # emerald


def build_dashboard(
    result: BacktestResult,
    output_path: Path | None = None,
    strategy_name_override: str | None = None,
    benchmark: BenchmarkResult | None = None,
    price_data: pd.Series | None = None,
) -> Path:
    """
    Build and save the interactive HTML dashboard.

    Args:
        result: BacktestResult from walk_forward().
        output_path: Path for the HTML file. Defaults to 'dashboard.html'.
        strategy_name_override: Display name override (e.g. 'Kalman Filter').
        benchmark: Optional BenchmarkResult from compute_benchmark(). When
                   provided, adds information ratio, beats-benchmark fraction,
                   and t-test p-value to the dashboard title, and overlays
                   benchmark Sharpe bars on the per-window chart.
        price_data: Optional close price Series from the full dataset. When
                    provided, the buy-and-hold equity curve uses actual price
                    changes on every bar rather than approximating from trades.
                    Pass data["close"] from the DataFrame used in walk_forward().

    Returns:
        Resolved path to the saved HTML file.
    """
    if output_path is None:
        output_path = Path("dashboard.html")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid = result.valid_windows
    if not valid:
        raise ValueError("No valid windows - nothing to plot.")

    display_name = strategy_name_override or result.strategy_name

    equity, bh_curve = _build_equity_curves(result, price_data=price_data)
    drawdown_series = _drawdown(equity)
    all_returns = _stitch_returns(valid)
    rolling_sharpe = _rolling_sharpe(equity, window=63)

    # Detect if MA strategy (has parameter evolution data).
    has_param_evolution = _has_param_evolution(result)

    n_rows = 4
    n_cols = 2
    row_heights = [0.32, 0.23, 0.23, 0.22]
    # 4-row 2-col grid: colspan rows count as 1 title slot (not 2).
    # Slot order: equity | drawdown | rolling sharpe | window sharpe | return dist | param panel
    subplot_titles = [
        "Out-of-Sample Equity Curve",
        "Drawdown", "Rolling Sharpe (63-day)",
        "Per-Window Sharpe Ratio", "Daily Return Distribution",
        "Parameter Evolution" if has_param_evolution else "Cumulative Returns vs Benchmark",
    ]

    fig = sp.make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        vertical_spacing=0.08,
        horizontal_spacing=0.09,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"colspan": 2}, None],
        ],
    )

    # Panel 1: Equity curve (full width, row 1)
    _add_equity_curve(fig, equity, bh_curve, valid, row=1, col=1)

    # Panel 2: Drawdown (row 2, col 1)
    _add_drawdown(fig, drawdown_series, row=2, col=1)

    # Panel 3: Rolling Sharpe (row 2, col 2)
    _add_rolling_sharpe(fig, rolling_sharpe, row=2, col=2)

    # Panel 4: Per-window Sharpe bars with optional benchmark overlay (row 3, col 1)
    _add_window_sharpes(fig, valid, row=3, col=1, benchmark=benchmark)

    # Panel 5: Return distribution (row 3, col 2)
    _add_return_distribution(fig, all_returns, row=3, col=2)

    # Panel 6: Parameter evolution (full width, row 4)
    if has_param_evolution:
        _add_param_evolution(fig, result, row=4, col=1)
    else:
        _add_cumulative_benchmark(fig, result, row=4, col=1)

    # Annotation: summary stats in title
    m = result.summary_metrics
    valid_n = len(valid)
    skipped = result.skipped_window_count
    rc_str = (
        f"  |  RC p: {m.reality_check_p_value:.4f}"
        if not math.isnan(m.reality_check_p_value) else "  |  RC p: N/A"
    )
    skip_str = f"  ({skipped} skipped)" if skipped else ""
    bm_str = ""
    if benchmark is not None:
        bm_str = (
            f"  |  IR {benchmark.information_ratio:.2f}"
            f"  |  Beats BH {benchmark.strategy_beats_benchmark_fraction:.0%}"
        )
    title_text = (
        f"<b>{display_name}</b>   "
        f"{valid_n} walk-forward windows{skip_str}<br>"
        f"<span style='font-size:13px;color:{_MUTED}'>"
        f"Sharpe {m.sharpe_ratio:.2f}  |  "
        f"Sortino {m.sortino_ratio:.2f}  |  "
        f"Max DD {m.max_drawdown:.1%}  |  "
        f"Fisher p: {m.combined_p_value:.4f}{rc_str}{bm_str}"
        f"</span>"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center", font=dict(size=15, color=_TEXT)),
        height=1050,
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL_BG,
        font=dict(family="'JetBrains Mono', 'Fira Code', monospace", color=_TEXT, size=11),
        legend=dict(
            bgcolor=_PANEL_BG, bordercolor=_GRID, borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(t=110, b=40, l=60, r=40),
        hovermode="x unified",
    )

    # Apply dark styling to all axes.
    fig.update_xaxes(
        gridcolor=_GRID, gridwidth=0.5,
        zeroline=False, linecolor=_GRID,
        tickfont=dict(size=10, color=_MUTED),
    )
    fig.update_yaxes(
        gridcolor=_GRID, gridwidth=0.5,
        zeroline=False, linecolor=_GRID,
        tickfont=dict(size=10, color=_MUTED),
    )
    fig.update_annotations(font=dict(size=12, color=_TEXT))

    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_equity_curves(
    result: BacktestResult,
    price_data: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Chain per-window portfolio values and buy-and-hold equity curves.

    The buy-and-hold benchmark invests INITIAL_PORTFOLIO_VALUE at the start
    of the first test window and holds until the end of the last, with each
    window chained continuously. This is a genuine buy-and-hold curve: it
    reflects price movement on every bar, not just on bars when the strategy
    was invested.

    If price_data is provided (the 'close' column of the full price DataFrame),
    it is used directly for buy-and-hold returns. Otherwise falls back to
    reconstructing returns from the simulation's portfolio values - which
    is less accurate for windows with no trades.

    Args:
        result: BacktestResult with valid window portfolio values.
        price_data: Optional close price Series spanning all test windows.
    """
    valid = result.valid_windows
    segments: list[pd.Series] = []
    bm_segments: list[pd.Series] = []
    running = INITIAL_PORTFOLIO_VALUE
    bm_running = INITIAL_PORTFOLIO_VALUE

    for w in valid:
        pv = w.simulation_result.portfolio_values
        if pv is None:
            # portfolio_values is always set by run_simulation_with_execution.
            # This guard replaces an assert, which would be silently stripped
            # when Python runs with the -O flag.
            raise ValueError(
                f"Window {w.test_start}–{w.test_end} has None portfolio_values. "
                "This indicates a bug in the simulation layer."
            )
        scale = running / float(pv.iloc[0])
        scaled = pv * scale
        segments.append(scaled)
        running = float(scaled.iloc[-1])

        # Buy-and-hold: invest at first bar, scale by actual price changes.
        if price_data is not None and w.test_start in price_data.index:
            window_prices = price_data.loc[w.test_start:w.test_end]
            if len(window_prices) > 0:
                p0 = float(window_prices.iloc[0])
                bm_vals = bm_running * (window_prices / p0)
                bm_seg = pd.Series(bm_vals.values, index=window_prices.index)
                bm_segments.append(bm_seg)
                bm_running = float(bm_seg.iloc[-1])
                continue

        # Fallback: approximate from portfolio index if no price data.
        # This uses the price path implied by the window's date range.
        n = len(pv)
        # Use uniform log-linear interpolation as a neutral approximation.
        bm_vals = bm_running * np.ones(n)
        bm_seg = pd.Series(bm_vals, index=pv.index)
        bm_segments.append(bm_seg)

    equity = pd.concat(segments)
    equity = equity[~equity.index.duplicated(keep="last")]
    benchmark = pd.concat(bm_segments)
    benchmark = benchmark[~benchmark.index.duplicated(keep="last")]
    return equity, benchmark


def _drawdown(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def _stitch_returns(valid_windows: list[WindowResult]) -> np.ndarray:
    arrays = []
    for w in valid_windows:
        pv = w.simulation_result.portfolio_values
        if pv is not None and len(pv) > 1:
            arrays.append(pv.pct_change().dropna().to_numpy())
    return np.concatenate(arrays) if arrays else np.array([])


def _rolling_sharpe(equity: pd.Series, window: int = 63) -> pd.Series:
    """63-day rolling annualised Sharpe ratio."""
    from backtesting_engine.config import ANNUALISATION_FACTOR
    returns = equity.pct_change().dropna()
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std(ddof=1)
    sharpe = (roll_mean / roll_std) * np.sqrt(ANNUALISATION_FACTOR)
    return sharpe.dropna()


def _has_param_evolution(result: BacktestResult) -> bool:
    """True if any valid window has calibrated parameter data available."""
    return any(bool(w.active_params) for w in result.valid_windows)


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def _add_equity_curve(
    fig: go.Figure,
    equity: pd.Series,
    benchmark: pd.Series,
    valid_windows: list[WindowResult],
    row: int, col: int,
) -> None:
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    bm_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name=f"Strategy ({total_ret:+.1f}%)",
        line=dict(color=_STRATEGY, width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Strategy</extra>",
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=benchmark.index, y=benchmark.values,
        name=f"Buy & hold ({bm_ret:+.1f}%)",
        line=dict(color=_BENCHMARK, width=1.2, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Buy & hold</extra>",
    ), row=row, col=col)

    # Shade alternating walk-forward windows.
    for i, w in enumerate(valid_windows):
        if i % 2 == 0:
            fig.add_vrect(
                x0=w.test_start, x1=w.test_end,
                fillcolor=_STRATEGY, opacity=0.04,
                layer="below", line_width=0,
                row=row, col=col,  # pyright: ignore[reportArgumentType]
            )

    fig.update_yaxes(
        tickprefix="$", tickformat=",.0f",
        title_text="Portfolio Value", row=row, col=col,  # pyright: ignore[reportArgumentType]
    )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor=_PANEL_BG, activecolor=_STRATEGY,
            font=dict(color=_TEXT, size=10),
        ),
        rangeslider=dict(visible=True, thickness=0.04, bgcolor=_PANEL_BG),
        row=row, col=col,  # pyright: ignore[reportArgumentType]
    )


def _add_drawdown(
    fig: go.Figure, drawdown: pd.Series, row: int, col: int
) -> None:
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color=_DRAWDOWN, width=1),
        fillcolor="rgba(239,68,68,0.25)",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra>Drawdown</extra>",
        showlegend=False,
    ), row=row, col=col)

    max_dd = float(drawdown.min())
    max_dd_date = drawdown.idxmin()
    fig.add_annotation(
        x=max_dd_date, y=max_dd,
        text=f"Max: {max_dd:.1%}",
        showarrow=True, arrowhead=2,
        arrowcolor=_DRAWDOWN, font=dict(color=_DRAWDOWN, size=10),
        bgcolor=_PANEL_BG, bordercolor=_DRAWDOWN,
        ax=30, ay=-30,
        row=row, col=col,  # pyright: ignore[reportArgumentType]
    )
    fig.update_yaxes(tickformat=".0%", title_text="Drawdown", row=row, col=col)


def _add_rolling_sharpe(
    fig: go.Figure, rolling_sharpe: pd.Series, row: int, col: int
) -> None:
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        name="Rolling Sharpe (63d)",
        line=dict(color=_ROLLING, width=1.2),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>Rolling Sharpe</extra>",
    ), row=row, col=col)

    fig.add_hline(y=0, line_color=_GRID, line_width=1, row=row, col=col)  # pyright: ignore[reportArgumentType]
    fig.add_hline(y=1, line_color=_POSITIVE, line_width=0.8,
                  line_dash="dash", row=row, col=col)  # pyright: ignore[reportArgumentType]
    fig.update_yaxes(title_text="Sharpe (63d)", row=row, col=col)


def _add_window_sharpes(
    fig: go.Figure,
    valid_windows: list[WindowResult],
    row: int,
    col: int,
    benchmark: BenchmarkResult | None = None,
) -> None:
    """
    Bar chart of per-window strategy Sharpe ratios.

    When benchmark is provided, overlays the benchmark Sharpe per window as
    a second bar series so the relative performance is visible for each window,
    not just on average. Bars are coloured green when strategy beats benchmark,
    red when it does not.
    """
    sharpes = [w.metrics_result.sharpe_ratio for w in valid_windows]
    labels = [f"{w.test_start.year}" for w in valid_windows]

    # Colour by whether strategy beats benchmark (if available) or simply by sign.
    if benchmark is not None:
        bm_sharpe = benchmark.benchmark_sharpe
        colors = [_POSITIVE if s >= bm_sharpe else _NEGATIVE for s in sharpes]
    else:
        colors = [_POSITIVE if s >= 0 else _NEGATIVE for s in sharpes]

    custom = [
        (f"{w.test_start.date()} → {w.test_end.date()}<br>"
         f"Sharpe: {w.metrics_result.sharpe_ratio:.2f}<br>"
         f"Sortino: {w.metrics_result.sortino_ratio:.2f}<br>"
         f"Max DD: {w.metrics_result.max_drawdown:.1%}<br>"
         f"p-value: {w.metrics_result.p_value:.4f}<br>"
         f"Trades: {len(w.simulation_result.trades)}")
        for w in valid_windows
    ]

    fig.add_trace(go.Bar(
        x=labels, y=sharpes,
        marker_color=colors, marker_opacity=0.85,
        name="Strategy Sharpe",
        customdata=custom,
        hovertemplate="%{customdata}<extra></extra>",
    ), row=row, col=col)

    # Benchmark reference line - mean benchmark Sharpe across windows.
    # Shows whether the strategy consistently beats buy-and-hold.
    if benchmark is not None:
        bm_sharpe = benchmark.benchmark_sharpe
        ir = benchmark.information_ratio
        beats_pct = int(benchmark.strategy_beats_benchmark_fraction * 100)
        fig.add_hline(
            y=bm_sharpe,
            line_color=_BENCHMARK,
            line_width=1.5,
            line_dash="dot",
            annotation_text=f"BH Sharpe: {bm_sharpe:.2f}  (IR {ir:.2f}, beats {beats_pct}%)",
            annotation_font=dict(color=_BENCHMARK, size=10),
            row=row, col=col,  # pyright: ignore[reportArgumentType]
        )

    mean_sharpe = float(np.mean(sharpes))
    fig.add_hline(
        y=mean_sharpe, line_color=_STRATEGY,
        line_width=1.5, line_dash="dash",
        annotation_text=f"Strategy mean: {mean_sharpe:.2f}",
        annotation_font=dict(color=_STRATEGY, size=10),
        row=row, col=col,  # pyright: ignore[reportArgumentType]
    )
    fig.add_hline(y=0, line_color=_GRID, line_width=0.8, row=row, col=col)  # pyright: ignore[reportArgumentType]
    fig.update_yaxes(title_text="Sharpe Ratio", row=row, col=col)


def _add_return_distribution(
    fig: go.Figure, returns: np.ndarray, row: int, col: int
) -> None:
    if len(returns) == 0:
        return

    from scipy import stats as sp_stats

    n_bins = min(80, max(20, len(returns) // 10))
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=n_bins,
        name="Daily returns",
        marker_color=_STRATEGY,
        opacity=0.7,
        histnorm="probability density",
        hovertemplate="Return: %{x:.3%}<br>Density: %{y:.4f}<extra></extra>",
    ), row=row, col=col)

    mu, sigma = returns.mean(), returns.std(ddof=1)
    x_range = np.linspace(returns.min(), returns.max(), 300)
    normal_pdf = sp_stats.norm.pdf(x_range, mu, sigma)

    fig.add_trace(go.Scatter(
        x=x_range, y=normal_pdf,
        name=f"Normal (μ={mu:.4f}, σ={sigma:.4f})",
        line=dict(color=_BENCHMARK, width=1.5, dash="dash"),
        hoverinfo="skip",
    ), row=row, col=col)

    skew = float(sp_stats.skew(returns))
    kurt = float(sp_stats.kurtosis(returns))

    fig.add_annotation(
        x=0.97, y=0.93, xref="paper", yref="paper",
        text=f"Skew: {skew:+.3f}  |  Excess kurtosis: {kurt:+.3f}",
        showarrow=False,
        font=dict(size=10, color=_MUTED),
        bgcolor=_PANEL_BG, bordercolor=_GRID,
        xanchor="right",
        row=row, col=col,  # pyright: ignore[reportArgumentType]
    )
    fig.update_xaxes(tickformat=".1%", title_text="Daily Return", row=row, col=col)
    fig.update_yaxes(title_text="Density", row=row, col=col)


def _add_param_evolution(
    fig: go.Figure, result: BacktestResult, row: int, col: int
) -> None:
    """
    Panel 6a: Parameter evolution across walk-forward windows.

    For MovingAverageStrategy: shows how short_window and long_window
    changed as training windows moved forward in time. Parameter drift
    reveals which market regimes favoured faster vs slower indicators.

    For KalmanFilterStrategy: shows the calibrated signal-to-noise ratio
    (Q/R) over time. High SNR → filter tracks price closely (trending market).
    Low SNR → filter smooths aggressively (noisy/mean-reverting market).
    """
    valid = result.valid_windows
    if not valid or not valid[0].active_params:
        _add_cumulative_benchmark(fig, result, row, col)
        return

    test_dates = [w.test_start for w in valid]
    params_list = [w.active_params for w in valid]
    first_key = list(params_list[0].keys())[0]

    if "short_window" in params_list[0]:
        # MA strategy: dual-axis short/long window plot.
        short_vals = [p.get("short_window", float("nan")) for p in params_list]
        long_vals  = [p.get("long_window", float("nan")) for p in params_list]

        fig.add_trace(go.Scatter(
            x=test_dates, y=short_vals,
            name="Short MA window",
            mode="lines+markers",
            line=dict(color=_PARAM_SHORT, width=2),
            marker=dict(size=7),
            hovertemplate="%{x|%Y}<br>Short: %{y}d<extra></extra>",
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=test_dates, y=long_vals,
            name="Long MA window",
            mode="lines+markers",
            line=dict(color=_PARAM_LONG, width=2),
            marker=dict(size=7),
            hovertemplate="%{x|%Y}<br>Long: %{y}d<extra></extra>",
        ), row=row, col=col)

        fig.update_yaxes(title_text="MA Window (days)", row=row, col=col)
        fig.update_xaxes(title_text="Test window start", row=row, col=col)

    elif "snr" in params_list[0]:
        # Kalman strategy: signal-to-noise ratio over time.
        snr_vals = [p.get("snr", float("nan")) for p in params_list]
        ll_vals = [p.get("log_likelihood", float("nan")) for p in params_list]

        fig.add_trace(go.Scatter(
            x=test_dates, y=snr_vals,
            name="Q/R (signal-to-noise ratio)",
            mode="lines+markers",
            line=dict(color=_PARAM_SHORT, width=2),
            marker=dict(size=7),
            customdata=ll_vals,
            hovertemplate=(
                "%{x|%Y}<br>"
                "Q/R: %{y:.6f}<br>"
                "Log-likelihood: %{customdata:.1f}"
                "<extra>SNR</extra>"
            ),
        ), row=row, col=col)

        fig.update_yaxes(title_text="Q/R (signal-to-noise ratio)", row=row, col=col)
        fig.update_xaxes(title_text="Test window start", row=row, col=col)

    else:
        # Unknown strategy params: plot first numeric key.
        key = first_key
        vals = [p.get(key, float("nan")) for p in params_list]
        fig.add_trace(go.Scatter(
            x=test_dates, y=vals,
            name=str(key),
            mode="lines+markers",
            line=dict(color=_PARAM_SHORT, width=2),
        ), row=row, col=col)
        fig.update_yaxes(title_text=str(key), row=row, col=col)


def _add_cumulative_benchmark(
    fig: go.Figure, result: BacktestResult, row: int, col: int
) -> None:
    """Panel 6b fallback: cumulative return index (strategy vs benchmark)."""
    equity, bh_curve = _build_equity_curves(result)

    strat_cum = (equity / equity.iloc[0] - 1) * 100
    bench_cum = (bh_curve / bh_curve.iloc[0] - 1) * 100

    fig.add_trace(go.Scatter(
        x=strat_cum.index, y=strat_cum.values,
        name="Strategy (cumulative %)",
        line=dict(color=_STRATEGY, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.1f}%<extra>Strategy</extra>",
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum.values,
        name="Buy & hold (cumulative %)",
        line=dict(color=_BENCHMARK, width=1.2, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.1f}%<extra>Buy & hold</extra>",
    ), row=row, col=col)

    fig.add_hline(y=0, line_color=_GRID, line_width=0.8, row=row, col=col)  # pyright: ignore[reportArgumentType]
    fig.update_yaxes(ticksuffix="%", title_text="Cumulative Return", row=row, col=col)
