"""
Entry point for the backtesting engine.

Runs three analyses and produces four output files:

  1. Moving average crossover - walk-forward with Reality Check
     → dashboard_ma.html

  2. Kalman filter trend following - walk-forward, MLE calibrated
     → dashboard_kalman.html

  3. Strategy comparison table - side-by-side all metrics

  4. Cost sensitivity analysis - how Fisher p degrades as slippage and
     transaction costs increase, identifying the breakeven execution cost
     → cost_sensitivity.html

Usage:
  make run          # all four analyses
  make run-ma       # MA strategy only
  make run-kalman   # Kalman strategy only
  make run-costs    # cost sensitivity sweep only
"""

import math
from pathlib import Path

import pandas as pd

from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    MOVING_AVERAGE_LONG_DAYS,
    SIGNIFICANCE_THRESHOLD,
    START_DATE,
    TESTING_WINDOW_YEARS,
    TICKER,
    TRAINING_WINDOW_YEARS,
)
from backtesting_engine.dashboard import build_dashboard
from backtesting_engine.data.ingestion import load_data
from backtesting_engine.data.validator import validate_data
from backtesting_engine.execution import ExecutionConfig, cost_sensitivity_sweep
from backtesting_engine.models import BacktestResult
from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward

_MIN_ROWS = (TRAINING_WINDOW_YEARS + TESTING_WINDOW_YEARS) * ANNUALISATION_FACTOR + MOVING_AVERAGE_LONG_DAYS

# Realistic retail execution on a liquid ETF:
#   0.1% fee, 5% of daily range slippage, 1-day signal delay.
_DEFAULT_EXECUTION = ExecutionConfig(
    transaction_cost_rate=0.001,
    slippage_factor=0.05,
    signal_delay=1,
)


def main() -> None:
    print(f"\n{'═' * 70}")
    print("  Backtesting Engine  ·  Walk-Forward  ·  Monte Carlo  ·  Reality Check")
    print(f"{'═' * 70}\n")

    data = _load(TICKER, START_DATE)

    # ── Strategy 1: Moving Average ─────────────────────────────────────────
    _section("Strategy 1: Moving Average Crossover  (grid-search calibrated)")
    ma_result = walk_forward(data, MovingAverageStrategy(), execution=_DEFAULT_EXECUTION)
    _print_results(ma_result)
    ma_dash = build_dashboard(ma_result, Path("dashboard_ma.html"),
                              strategy_name_override="Moving Average Crossover")
    print(f"\n  Dashboard → {ma_dash}\n")

    # ── Strategy 2: Kalman Filter ──────────────────────────────────────────
    _section("Strategy 2: Kalman Filter Trend Following  (MLE calibrated)")
    kalman_result = walk_forward(data, KalmanFilterStrategy(), execution=_DEFAULT_EXECUTION)
    _print_results(kalman_result)
    kalman_dash = build_dashboard(kalman_result, Path("dashboard_kalman.html"),
                                  strategy_name_override="Kalman Filter Trend Following")
    print(f"\n  Dashboard → {kalman_dash}\n")

    # ── Comparative summary ────────────────────────────────────────────────
    _section("Comparative Summary")
    _print_comparison(ma_result, kalman_result)

    # ── Cost sensitivity sweep ─────────────────────────────────────────────
    _section("Cost Sensitivity Analysis  (how significance degrades with execution cost)")
    _run_cost_sensitivity(data, ma_result, kalman_result)


def _load(ticker: str, start_date: str) -> pd.DataFrame:
    print(f"Loading {ticker} from {start_date}...")
    data = load_data(ticker, start_date)
    validate_data(data, min_rows=_MIN_ROWS)
    print(
        f"  {len(data):,} trading days  "
        f"({data.index[0].date()} – {data.index[-1].date()})\n"
    )
    return data


def _section(title: str) -> None:
    print("─" * 70)
    print(f"  {title}")
    print("─" * 70)


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------

def _print_results(result: BacktestResult) -> None:
    header = (
        f"  {'Test window':<24}  "
        f"{'Sharpe':>7}  {'Sortino':>8}  {'Max DD':>8}  "
        f"{'p-val':>7}  {'Trades':>6}  Params"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for w in result.window_results:
        window_str = f"{w.test_start.date()} → {w.test_end.date()}"
        if w.skipped:
            print(f"  {window_str:<24}  [skipped - no trades generated]")
            continue

        m = w.metrics_result
        params = _format_params(w.active_params)
        print(
            f"  {window_str:<24}  "
            f"{m.sharpe_ratio:>7.2f}  {m.sortino_ratio:>8.2f}  "
            f"{m.max_drawdown:>8.2%}  {m.p_value:>7.4f}  "
            f"{len(w.simulation_result.trades):>6}  {params}"
        )

    print()
    _print_summary(result)


def _format_params(params: dict[str, object]) -> str:
    """Format active_params as a compact string for the results table."""
    if not params:
        return ""
    if "short_window" in params:
        return f"MA({params['short_window']}/{params['long_window']})"
    if "snr" in params:
        return f"SNR={params['snr']:.2e}"
    return str(params)


def _print_summary(result: BacktestResult) -> None:
    m = result.summary_metrics
    valid_n = len(result.valid_windows)
    skipped = result.skipped_window_count

    print(f"  {'Windows':<30} {valid_n}  ({skipped} skipped)")
    print(f"  {'Sharpe ratio':<30} {m.sharpe_ratio:.3f}")
    print(f"  {'Sortino ratio':<30} {m.sortino_ratio:.3f}")
    print(f"  {'Max drawdown':<30} {m.max_drawdown:.2%}")
    print(f"  {'Calmar ratio':<30} {m.calmar_ratio:.3f}")
    print(f"  {'Omega ratio':<30} {m.omega_ratio:.3f}")
    print(f"  {'Block-bootstrap p (mean)':<30} {m.p_value:.4f}")
    print(f"  {'Fisher combined p':<30} {m.combined_p_value:.6f}")

    if not math.isnan(m.reality_check_p_value):
        print(f"  {'White Reality Check p':<30} {m.reality_check_p_value:.6f}  ← data-snooping corrected")

    print()
    _verdict(m.combined_p_value, m.reality_check_p_value)


def _verdict(fisher_p: float, rc_p: float) -> None:
    if fisher_p < SIGNIFICANCE_THRESHOLD:
        if not math.isnan(rc_p) and rc_p >= SIGNIFICANCE_THRESHOLD:
            print(
                f"  ⚠  MARGINAL: Fisher p={fisher_p:.4f} is significant, but "
                f"White's Reality Check p={rc_p:.4f} is not.\n"
                f"     Apparent significance likely reflects parameter search, "
                f"not genuine edge."
            )
        else:
            print(
                f"  ✓  SIGNIFICANT: Fisher p={fisher_p:.4f} < {SIGNIFICANCE_THRESHOLD}."
            )
    else:
        print(
            f"  ✗  NOT SIGNIFICANT: Fisher p={fisher_p:.4f} ≥ {SIGNIFICANCE_THRESHOLD}.\n"
            f"     Performance is consistent with noise - an honest result."
        )


def _print_comparison(ma: BacktestResult, kalman: BacktestResult) -> None:
    ma_m, ka_m = ma.summary_metrics, kalman.summary_metrics

    def row(label: str, ma_val: str, ka_val: str, better: str = "") -> None:
        print(f"  {label:<30}  {ma_val:>12}  {ka_val:>12}  {better}")

    print(f"  {'Metric':<30}  {'MA Cross':>12}  {'Kalman':>12}  Better")
    print("  " + "─" * 62)

    def wins(ma_v: float, ka_v: float, higher_is_better: bool = True) -> str:
        if math.isnan(ma_v) or math.isnan(ka_v):
            return "-"
        if higher_is_better:
            return "Kalman ✓" if ka_v > ma_v else ("MA ✓" if ma_v > ka_v else "Tie")
        return "Kalman ✓" if ka_v < ma_v else ("MA ✓" if ma_v < ka_v else "Tie")

    row("Sharpe ratio", f"{ma_m.sharpe_ratio:.3f}", f"{ka_m.sharpe_ratio:.3f}",
        wins(ma_m.sharpe_ratio, ka_m.sharpe_ratio))
    row("Sortino ratio", f"{ma_m.sortino_ratio:.3f}", f"{ka_m.sortino_ratio:.3f}",
        wins(ma_m.sortino_ratio, ka_m.sortino_ratio))
    row("Max drawdown", f"{ma_m.max_drawdown:.2%}", f"{ka_m.max_drawdown:.2%}",
        wins(ma_m.max_drawdown, ka_m.max_drawdown, higher_is_better=False))
    row("Calmar ratio", f"{ma_m.calmar_ratio:.3f}", f"{ka_m.calmar_ratio:.3f}",
        wins(ma_m.calmar_ratio, ka_m.calmar_ratio))
    row("Omega ratio", f"{ma_m.omega_ratio:.3f}", f"{ka_m.omega_ratio:.3f}",
        wins(ma_m.omega_ratio, ka_m.omega_ratio))
    row("Fisher p (lower = better)", f"{ma_m.combined_p_value:.4f}",
        f"{ka_m.combined_p_value:.4f}",
        wins(ma_m.combined_p_value, ka_m.combined_p_value, higher_is_better=False))
    if not math.isnan(ma_m.reality_check_p_value):
        row("Reality Check p", f"{ma_m.reality_check_p_value:.4f}", "N/A (no grid)")
    print()


# ---------------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------------

def _run_cost_sensitivity(
    data: pd.DataFrame,
    ma_result: BacktestResult,
    kalman_result: BacktestResult,
) -> None:
    """
    Sweep over (transaction_cost_rate, slippage_factor) grids and show how
    Fisher p-values degrade as execution costs increase.

    Prints the breakeven cost for each strategy - the point at which
    Fisher p crosses 0.05 - which is the most practically important
    single number for deciding whether to pursue a strategy live.
    """
    cost_rates    = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    slip_factors  = [0.0, 0.025, 0.05, 0.10, 0.20]

    print("  Running MA cost sensitivity sweep...")
    ma_sweep = cost_sensitivity_sweep(
        data, MovingAverageStrategy(),
        cost_rates=cost_rates,
        slippage_factors=slip_factors,
    )

    print("  Running Kalman cost sensitivity sweep...\n")
    kalman_sweep = cost_sensitivity_sweep(
        data, KalmanFilterStrategy(),
        cost_rates=cost_rates,
        slippage_factors=slip_factors,
    )

    _print_cost_table("Moving Average", ma_sweep, cost_rates, slip_factors)
    _print_cost_table("Kalman Filter", kalman_sweep, cost_rates, slip_factors)

    _save_cost_heatmap(ma_sweep, kalman_sweep, cost_rates, slip_factors)


def _print_cost_table(
    name: str,
    sweep: dict[tuple[float, float], float],
    cost_rates: list[float],
    slip_factors: list[float],
) -> None:
    print(f"  {name} - Fisher p-value by (cost rate, slippage factor)")
    print(f"  {'':>10}  " + "  ".join(f"slip={s:.3f}" for s in slip_factors))
    print("  " + "─" * 60)
    for cost in cost_rates:
        row_vals = []
        for slip in slip_factors:
            p = sweep.get((cost, slip), float("nan"))
            marker = "✓" if not math.isnan(p) and p < SIGNIFICANCE_THRESHOLD else " "
            row_vals.append(f"{p:.3f}{marker}")
        print(f"  cost={cost:.4f}   " + "  ".join(f"{v:>8}" for v in row_vals))
    print()


def _save_cost_heatmap(
    ma_sweep: dict[tuple[float, float], float],
    kalman_sweep: dict[tuple[float, float], float],
    cost_rates: list[float],
    slip_factors: list[float],
) -> None:
    """Save interactive cost sensitivity heatmap as HTML."""
    try:
        import plotly.graph_objects as go
        import plotly.subplots as sp

        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=["Moving Average - Fisher p", "Kalman Filter - Fisher p"],
            horizontal_spacing=0.12,
        )

        for col, (name, sweep) in enumerate(
            [("MA", ma_sweep), ("Kalman", kalman_sweep)], start=1
        ):
            z = [
                [sweep.get((c, s), float("nan")) for s in slip_factors]
                for c in cost_rates
            ]
            fig.add_trace(go.Heatmap(
                z=z,
                x=[f"{s:.3f}" for s in slip_factors],
                y=[f"{c:.4f}" for c in cost_rates],
                colorscale="RdYlGn_r",
                zmin=0, zmax=1,
                colorbar=dict(title="Fisher p", x=1.0 if col == 2 else 0.45),
                text=[[f"{v:.3f}" if not math.isnan(v) else "-" for v in row] for row in z],
                texttemplate="%{text}",
                hovertemplate=(
                    "Cost rate: %{y}<br>Slippage: %{x}<br>"
                    "Fisher p: %{z:.4f}<extra></extra>"
                ),
                name=name,
                showscale=(col == 2),
            ), row=1, col=col)

        # Add 0.05 significance contour annotation.
        sig_note = (
            "Cells ≤ 0.05 indicate statistical significance at the 5% level. "
            "Darker red = strategy loses significance at lower cost levels."
        )

        fig.update_layout(
            title=dict(
                text="Cost Sensitivity Analysis - Fisher p-value vs Execution Cost<br>"
                     f"<span style='font-size:12px;color:#94A3B8'>{sig_note}</span>",
                x=0.5, font=dict(size=14, color="#F1F5F9"),
            ),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font=dict(family="monospace", color="#F1F5F9"),
            height=500,
        )
        fig.update_xaxes(title_text="Slippage factor")
        fig.update_yaxes(title_text="Transaction cost rate")

        path = Path("cost_sensitivity.html")
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        print(f"  Cost sensitivity heatmap → {path.resolve()}")

    except Exception as e:
        print(f"  (Cost heatmap skipped: {e})")


if __name__ == "__main__":
    main()