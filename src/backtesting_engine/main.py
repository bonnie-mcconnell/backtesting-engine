"""
Entry point for the backtesting engine.

Runs walk-forward analysis on one or more strategies and produces HTML dashboards.

Usage:
  make run          # all strategies + cost sensitivity
  make run-ma       # moving average only
  make run-kalman   # Kalman filter only
  make run-momentum # momentum only
  make run-costs    # cost sensitivity sweep only

Or directly:
  backtesting-engine                  # all strategies
  backtesting-engine --strategy ma    # moving average only
  backtesting-engine --strategy kalman
  backtesting-engine --strategy momentum
  backtesting-engine --costs-only

Output files:
  dashboard_ma.html       Moving average walk-forward dashboard
  dashboard_kalman.html   Kalman filter walk-forward dashboard
  dashboard_momentum.html Momentum walk-forward dashboard
  cost_sensitivity.html   Cost sensitivity heatmap
"""

import argparse
import math
from pathlib import Path

import pandas as pd

from backtesting_engine.benchmark import BenchmarkResult, compute_benchmark
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
from backtesting_engine.strategy.momentum import MomentumStrategy
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtesting engine with statistical significance testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  backtesting-engine                    Run all strategies
  backtesting-engine --strategy ma      Moving average only
  backtesting-engine --strategy kalman  Kalman filter only
  backtesting-engine --strategy momentum  Momentum only
  backtesting-engine --costs-only       Cost sensitivity sweep only
  backtesting-engine --ticker QQQ --start 2000-01-01
        """,
    )
    parser.add_argument(
        "--strategy",
        choices=["ma", "kalman", "momentum", "all"],
        default="all",
        help="Strategy to run (default: all)",
    )
    parser.add_argument(
        "--costs-only",
        action="store_true",
        help="Run cost sensitivity sweep only (skips strategy evaluation)",
    )
    parser.add_argument(
        "--ticker",
        default=TICKER,
        help=f"Ticker symbol to backtest (default: {TICKER})",
    )
    parser.add_argument(
        "--start",
        default=START_DATE,
        metavar="YYYY-MM-DD",
        help=f"Start date for historical data (default: {START_DATE})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh data download, ignoring local cache",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"\n{'═' * 70}")
    print("  Backtesting Engine  ·  Walk-Forward  ·  Monte Carlo  ·  Reality Check")
    print(f"{'═' * 70}\n")

    data = _load(args.ticker, args.start, use_cache=not args.no_cache)

    run_ma      = args.strategy in ("ma",      "all") and not args.costs_only
    run_kalman  = args.strategy in ("kalman",  "all") and not args.costs_only
    run_momentum = args.strategy in ("momentum", "all") and not args.costs_only

    ma_result      = None
    kalman_result  = None
    momentum_result = None

    # ── Strategy 1: Moving Average ─────────────────────────────────────────
    if run_ma:
        _section("Strategy 1: Moving Average Crossover  (grid-search calibrated)")
        ma_result = walk_forward(data, MovingAverageStrategy(), execution=_DEFAULT_EXECUTION)
        _print_results(ma_result)
        ma_benchmark = compute_benchmark(ma_result, data)
        _print_benchmark(ma_benchmark)
        ma_dash = build_dashboard(
            ma_result,
            Path("dashboard_ma.html"),
            strategy_name_override="Moving Average Crossover",
            benchmark=ma_benchmark,
            price_data=data["close"],
        )
        print(f"\n  Dashboard → {ma_dash}\n")

    # ── Strategy 2: Kalman Filter ──────────────────────────────────────────
    if run_kalman:
        _section("Strategy 2: Kalman Filter Trend Following  (MLE calibrated)")
        kalman_result = walk_forward(data, KalmanFilterStrategy(), execution=_DEFAULT_EXECUTION)
        _print_results(kalman_result)
        kalman_benchmark = compute_benchmark(kalman_result, data)
        _print_benchmark(kalman_benchmark)
        kalman_dash = build_dashboard(
            kalman_result,
            Path("dashboard_kalman.html"),
            strategy_name_override="Kalman Filter Trend Following",
            benchmark=kalman_benchmark,
            price_data=data["close"],
        )
        print(f"\n  Dashboard → {kalman_dash}\n")

    # ── Strategy 3: Momentum ──────────────────────────────────────────────
    if run_momentum:
        _section("Strategy 3: Time-Series Momentum  (lookback grid-search calibrated)")
        momentum_result = walk_forward(data, MomentumStrategy(), execution=_DEFAULT_EXECUTION)
        _print_results(momentum_result)
        momentum_benchmark = compute_benchmark(momentum_result, data)
        _print_benchmark(momentum_benchmark)
        momentum_dash = build_dashboard(
            momentum_result,
            Path("dashboard_momentum.html"),
            strategy_name_override="Time-Series Momentum",
            benchmark=momentum_benchmark,
            price_data=data["close"],
        )
        print(f"\n  Dashboard → {momentum_dash}\n")

    # ── Comparative summary ────────────────────────────────────────────────
    ready = [r for r in [ma_result, kalman_result, momentum_result] if r is not None]
    if len(ready) >= 2 and ma_result is not None and kalman_result is not None and momentum_result is not None:
        _section("Comparative Summary")
        _print_comparison(ma_result, kalman_result, momentum_result)

    # ── Cost sensitivity sweep ─────────────────────────────────────────────
    if args.costs_only or args.strategy == "all":
        _section("Cost Sensitivity Analysis  (how significance degrades with execution cost)")
        _run_cost_sensitivity(data, ma_result, kalman_result, momentum_result)


def _load(ticker: str, start_date: str, use_cache: bool = True) -> pd.DataFrame:
    print(f"Loading {ticker} from {start_date}...")
    data = load_data(ticker, start_date, use_cache=use_cache)
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
    def _fmt(v: float, fmt: str = ".3f") -> str:
        if math.isnan(v):
            return "N/A"
        if abs(v) == float("inf"):
            return "∞ (no downside)" if fmt == ".3f" else "∞"
        return format(v, fmt)

    print(f"  {'Sharpe ratio':<30} {_fmt(m.sharpe_ratio)}")
    print(f"  {'Sortino ratio':<30} {_fmt(m.sortino_ratio)}")
    print(f"  {'Max drawdown':<30} {m.max_drawdown:.2%}")
    print(f"  {'Calmar ratio':<30} {_fmt(m.calmar_ratio)}")
    print(f"  {'Omega ratio':<30} {_fmt(m.omega_ratio)}")
    print(f"  {'Block-bootstrap p (mean)':<30} {m.p_value:.4f}")
    print(f"  {'Fisher combined p':<30} {m.combined_p_value:.6f}  (approx: windows not fully independent)")

    if not math.isnan(m.reality_check_p_value):
        print(f"  {'White Reality Check p':<30} {m.reality_check_p_value:.6f}  ← data-snooping corrected")
    else:
        print(f"  {'White Reality Check p':<30} N/A (no parameter grid)")

    print()
    _verdict(m.combined_p_value, m.reality_check_p_value)


def _verdict(fisher_p: float, rc_p: float) -> None:
    if fisher_p < SIGNIFICANCE_THRESHOLD:
        if not math.isnan(rc_p) and rc_p >= SIGNIFICANCE_THRESHOLD:
            print(
                f"  ⚠  MARGINAL: Fisher p={fisher_p:.4f} significant, but "
                f"White's Reality Check p={rc_p:.4f} is not.\n"
                f"     Significance likely reflects parameter search, not genuine edge.\n"
                f"     Reality Check benchmark: zero return (cash), not buy-and-hold."
            )
        else:
            print(
                f"  ✓  SIGNIFICANT: Fisher p={fisher_p:.4f} < {SIGNIFICANCE_THRESHOLD}.\n"
                f"     Note: Fisher assumes window independence (approximate).\n"
                f"     Check information ratio - significance vs cash ≠ significance vs BH."
            )
    else:
        print(
            f"  ✗  NOT SIGNIFICANT: Fisher p={fisher_p:.4f} ≥ {SIGNIFICANCE_THRESHOLD}.\n"
            f"     Performance consistent with noise across {SIGNIFICANCE_THRESHOLD:.0%} threshold."
        )


def _print_benchmark(bm: BenchmarkResult) -> None:
    print(f"  {'Benchmark (buy-and-hold)'}")
    print(f"  {'  Sharpe':<30} {bm.benchmark_sharpe:.3f}")
    print(f"  {'  Max drawdown':<30} {bm.benchmark_max_drawdown:.2%}")
    print(f"  {'  Information ratio':<30} {bm.information_ratio:.3f}")
    beats = f"{bm.strategy_beats_benchmark_fraction:.0%} of windows"
    print(f"  {'  Strategy beats BH in':<30} {beats}")
    if not math.isnan(bm.sharpe_diff_p_value):
        print(f"  {'  Sharpe diff t-test p':<30} {bm.sharpe_diff_p_value:.4f}")
    print()


def _print_comparison(
    ma: BacktestResult,
    kalman: BacktestResult,
    momentum: BacktestResult,
) -> None:
    ma_m = ma.summary_metrics
    ka_m = kalman.summary_metrics
    mo_m = momentum.summary_metrics

    def row(label: str, ma_v: str, ka_v: str, mo_v: str) -> None:
        print(f"  {label:<28}  {ma_v:>10}  {ka_v:>10}  {mo_v:>10}")

    def best3(a: float, b: float, c: float, higher_is_better: bool = True) -> tuple[str, str, str]:
        """Return (ma_str, kalman_str, momentum_str) with ✓ on the best."""
        def _fmt_val(v: float) -> str:
            if math.isnan(v):
                return "N/A"
            if abs(v) == float("inf"):
                return "∞"
            return f"{v:.3f}"

        # Skip inf and nan when finding the best finite value
        finite = [(i, v) for i, v in enumerate([a, b, c])
                  if not math.isnan(v) and abs(v) != float("inf")]
        if not finite:
            return _fmt_val(a), _fmt_val(b), _fmt_val(c)

        best_val = max(v for _, v in finite) if higher_is_better else min(v for _, v in finite)
        best_idxs = {i for i, v in finite if v == best_val}

        results = []
        for i, v in enumerate([a, b, c]):
            s = _fmt_val(v)
            if i in best_idxs and abs(v) != float("inf"):
                s += " ✓"
            results.append(s)
        return results[0], results[1], results[2]

    def best3_p(a: float, b: float, c: float) -> tuple[str, str, str]:
        """Lower is better for p-values."""
        if any(math.isnan(x) for x in [a, b, c]):
            return f"{a:.4f}", f"{b:.4f}", f"{c:.4f}"
        best_val = min(a, b, c)
        def fmt(v: float) -> str:
            s = f"{v:.4f}"
            return s + " ✓" if v == best_val else s
        return fmt(a), fmt(b), fmt(c)

    print(f"  {'Metric':<28}  {'MA Cross':>10}  {'Kalman':>10}  {'Momentum':>10}")
    print("  " + "─" * 66)

    ma_s, ka_s, mo_s = best3(ma_m.sharpe_ratio, ka_m.sharpe_ratio, mo_m.sharpe_ratio)
    row("Sharpe ratio", ma_s, ka_s, mo_s)

    ma_s, ka_s, mo_s = best3(ma_m.sortino_ratio, ka_m.sortino_ratio, mo_m.sortino_ratio)
    row("Sortino ratio", ma_s, ka_s, mo_s)

    ma_s, ka_s, mo_s = best3(ma_m.max_drawdown, ka_m.max_drawdown, mo_m.max_drawdown,
                               higher_is_better=False)
    row("Max drawdown", f"{ma_m.max_drawdown:.2%}", f"{ka_m.max_drawdown:.2%}", f"{mo_m.max_drawdown:.2%}")

    ma_s, ka_s, mo_s = best3(ma_m.calmar_ratio, ka_m.calmar_ratio, mo_m.calmar_ratio)
    row("Calmar ratio", ma_s, ka_s, mo_s)

    ma_s, ka_s, mo_s = best3(ma_m.omega_ratio, ka_m.omega_ratio, mo_m.omega_ratio)
    row("Omega ratio", ma_s, ka_s, mo_s)

    ma_s, ka_s, mo_s = best3_p(ma_m.combined_p_value, ka_m.combined_p_value, mo_m.combined_p_value)
    row("Fisher p (lower = better)", ma_s, ka_s, mo_s)

    rc_ma = f"{ma_m.reality_check_p_value:.4f}" if not math.isnan(ma_m.reality_check_p_value) else "N/A"
    rc_mo = f"{mo_m.reality_check_p_value:.4f}" if not math.isnan(mo_m.reality_check_p_value) else "N/A"
    row("Reality Check p", rc_ma, "N/A", rc_mo)
    print()


# ---------------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------------

def _run_cost_sensitivity(
    data: pd.DataFrame,
    ma_result: BacktestResult | None,
    kalman_result: BacktestResult | None,
    momentum_result: BacktestResult | None,
) -> None:
    """
    Sweep over (transaction_cost_rate, slippage_factor) grids and show how
    Fisher p-values degrade as execution costs increase.

    The breakeven cost - where Fisher p crosses 0.05 - is the most practically
    important single number for deciding whether to pursue a strategy live.

    Set n_workers > 1 in cost_sensitivity_sweep() to parallelise across CPU cores.
    On an 8-core machine this reduces a ~12-minute sequential sweep to ~2 minutes.
    """
    cost_rates   = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    slip_factors = [0.0, 0.025, 0.05, 0.10, 0.20]

    sweeps: dict[str, dict[tuple[float, float], float]] = {}

    print("  Running MA cost sensitivity sweep...")
    sweeps["MA Crossover"] = cost_sensitivity_sweep(
        data, MovingAverageStrategy(),
        cost_rates=cost_rates,
        slippage_factors=slip_factors,
    )

    print("  Running Kalman cost sensitivity sweep...")
    sweeps["Kalman Filter"] = cost_sensitivity_sweep(
        data, KalmanFilterStrategy(),
        cost_rates=cost_rates,
        slippage_factors=slip_factors,
    )

    print("  Running Momentum cost sensitivity sweep...\n")
    sweeps["Momentum"] = cost_sensitivity_sweep(
        data, MomentumStrategy(),
        cost_rates=cost_rates,
        slippage_factors=slip_factors,
    )

    for name, sweep in sweeps.items():
        _print_cost_table(name, sweep, cost_rates, slip_factors)

    _save_cost_heatmap(sweeps, cost_rates, slip_factors)


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
    sweeps: dict[str, dict[tuple[float, float], float]],
    cost_rates: list[float],
    slip_factors: list[float],
) -> None:
    """Save interactive cost sensitivity heatmap as HTML."""
    try:
        import plotly.graph_objects as go
        import plotly.subplots as sp

        n_cols = len(sweeps)

        fig = sp.make_subplots(
            rows=1, cols=n_cols,
            subplot_titles=[f"{name} - Fisher p" for name in sweeps],
            horizontal_spacing=0.08,
        )

        for col, (name, sweep) in enumerate(sweeps.items(), start=1):
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
                colorbar=dict(title="Fisher p"),
                text=[[f"{v:.3f}" if not math.isnan(v) else "-" for v in row] for row in z],
                texttemplate="%{text}",
                hovertemplate=(
                    "Cost rate: %{y}<br>Slippage: %{x}<br>"
                    "Fisher p: %{z:.4f}<extra></extra>"
                ),
                name=name,
                showscale=(col == n_cols),
            ), row=1, col=col)

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
            height=500 if n_cols <= 2 else 450,
        )
        fig.update_xaxes(title_text="Slippage factor")
        fig.update_yaxes(title_text="Transaction cost rate")

        path = Path("cost_sensitivity.html")
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        print(f"  Cost sensitivity heatmap → {path.resolve()}")

    except ImportError:
        # plotly is listed as an optional dependency. The cost sweep still runs
        # and prints its table; only the HTML heatmap is skipped.
        print("  (Cost heatmap skipped: plotly not installed. Run `pip install plotly`.)")
    except Exception as e:
        # Unexpected plotly rendering error - surface it rather than swallowing.
        print(f"  (Cost heatmap failed unexpectedly: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
