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
    BLOCK_BOOTSTRAP_SEED,
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
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "End date for historical data (default: today). "
            "Set this to a fixed date to produce reproducible results that do not "
            "change as new data arrives.  Example: --end 2024-12-31"
        ),
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.001,
        metavar="RATE",
        help="Transaction cost rate per side, e.g. 0.001 = 0.1%% (default: 0.001)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.05,
        metavar="FACTOR",
        help=(
            "Slippage as fraction of daily high-low range (default: 0.05). "
            "0 = fill at close, 0.1 = 10%% of daily range"
        ),
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=1,
        metavar="BARS",
        help="Signal execution delay in bars (default: 1 = next-day fill)",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=TRAINING_WINDOW_YEARS,
        metavar="N",
        help=f"Training window length in years (default: {TRAINING_WINDOW_YEARS})",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        default=TESTING_WINDOW_YEARS,
        metavar="N",
        help=f"Test window length in years (default: {TESTING_WINDOW_YEARS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Bootstrap random seed override (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory for output dashboards (default: current directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"\n{'═' * 70}")
    print("  Backtesting Engine  ·  Walk-Forward  ·  Monte Carlo  ·  Reality Check")
    print(f"{'═' * 70}\n")

    execution = ExecutionConfig(
        transaction_cost_rate=args.cost,
        slippage_factor=args.slippage,
        signal_delay=args.delay,
    )

    # Bootstrap seed: use CLI override if provided, else fall back to config default.
    # Passing an explicit seed makes every bootstrap reproducible.
    bootstrap_seed = args.seed if args.seed is not None else BLOCK_BOOTSTRAP_SEED

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load(
        args.ticker,
        args.start,
        end_date=args.end,
        use_cache=not args.no_cache,
    )

    run_ma      = args.strategy in ("ma",      "all") and not args.costs_only
    run_kalman  = args.strategy in ("kalman",  "all") and not args.costs_only
    run_momentum = args.strategy in ("momentum", "all") and not args.costs_only

    ma_result      = None
    kalman_result  = None
    momentum_result = None

    train_yrs = args.train_years
    test_yrs  = args.test_years

    # ── Strategy 1: Moving Average ─────────────────────────────────────────
    if run_ma:
        _section("Strategy 1: Moving Average Crossover  (grid-search calibrated)")
        ma_result = walk_forward(
            data, MovingAverageStrategy(), execution=execution,
            training_window_years=train_yrs, testing_window_years=test_yrs,
            bootstrap_seed=bootstrap_seed,
        )
        _print_results(ma_result)
        ma_benchmark = compute_benchmark(ma_result, data, execution=execution)
        _print_benchmark(ma_benchmark)
        ma_dash = build_dashboard(
            ma_result,
            output_dir / "dashboard_ma.html",
            strategy_name_override="Moving Average Crossover",
            benchmark=ma_benchmark,
            price_data=data["close"],
        )
        print(f"\n  Dashboard → {ma_dash}\n")

    # ── Strategy 2: Kalman Filter ──────────────────────────────────────────
    if run_kalman:
        _section("Strategy 2: Kalman Filter Trend Following  (MLE calibrated)")
        kalman_result = walk_forward(
            data, KalmanFilterStrategy(), execution=execution,
            training_window_years=train_yrs, testing_window_years=test_yrs,
            bootstrap_seed=bootstrap_seed,
        )
        _print_results(kalman_result)
        kalman_benchmark = compute_benchmark(kalman_result, data, execution=execution)
        _print_benchmark(kalman_benchmark)
        kalman_dash = build_dashboard(
            kalman_result,
            output_dir / "dashboard_kalman.html",
            strategy_name_override="Kalman Filter Trend Following",
            benchmark=kalman_benchmark,
            price_data=data["close"],
        )
        print(f"\n  Dashboard → {kalman_dash}\n")

    # ── Strategy 3: Momentum ──────────────────────────────────────────────
    if run_momentum:
        _section("Strategy 3: Time-Series Momentum  (lookback grid-search calibrated)")
        momentum_result = walk_forward(
            data, MomentumStrategy(), execution=execution,
            training_window_years=train_yrs, testing_window_years=test_yrs,
            bootstrap_seed=bootstrap_seed,
        )
        _print_results(momentum_result)
        momentum_benchmark = compute_benchmark(momentum_result, data, execution=execution)
        _print_benchmark(momentum_benchmark)
        momentum_dash = build_dashboard(
            momentum_result,
            output_dir / "dashboard_momentum.html",
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
        _run_cost_sensitivity(
            data, ma_result, kalman_result, momentum_result,
            output_dir=output_dir,
            train_yrs=train_yrs, test_yrs=test_yrs,
            bootstrap_seed=bootstrap_seed,
        )


def _load(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    end_str = end_date or "today"
    print(f"Loading {ticker} from {start_date} to {end_str}...")
    data = load_data(ticker, start_date, end_date=end_date, use_cache=use_cache)
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


def _fmt_metric(v: float, fmt: str = ".3f") -> str:
    """Format a float metric for console output, handling NaN and inf gracefully."""
    if math.isnan(v):
        return "N/A"
    if abs(v) == float("inf"):
        return "∞" if fmt != ".3f" else "∞ (no downside)"
    return format(v, fmt)


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
        m = w.metrics_result
        trade_n = len(w.simulation_result.trades)
        # Use formatted_params from WindowResult - set by strategy.format_params()
        # at window creation time so main.py has no strategy-specific knowledge.
        params = w.formatted_params if trade_n > 0 else "[flat-cash]"
        print(
            f"  {window_str:<24}  "
            f"{m.sharpe_ratio:>7.2f}  {m.sortino_ratio:>8.2f}  "
            f"{m.max_drawdown:>8.2%}  {m.p_value:>7.4f}  "
            f"{trade_n:>6}  {params}"
        )

    print()
    _print_summary(result)


def _print_summary(result: BacktestResult) -> None:
    m = result.summary_metrics
    valid_n = len(result.valid_windows)
    flat_cash = result.flat_cash_window_count

    print(f"  {'Windows':<30} {valid_n}  ({flat_cash} flat-cash)")

    print(f"  {'Sharpe ratio':<30} {_fmt_metric(m.sharpe_ratio)}")
    print(f"  {'Sortino ratio':<30} {_fmt_metric(m.sortino_ratio)}")
    print(f"  {'Max drawdown':<30} {m.max_drawdown:.2%}")
    print(f"  {'Calmar ratio':<30} {_fmt_metric(m.calmar_ratio)}")
    print(f"  {'Omega ratio':<30} {_fmt_metric(m.omega_ratio)}")
    print(f"  {'Block-bootstrap p (mean)':<30} {m.p_value:.4f}")
    print(f"  {'Fisher combined p':<30} {m.combined_p_value:.6f}  (approx: windows not fully independent)")

    if not math.isnan(m.reality_check_p_value):
        print(f"  {'White Reality Check p':<30} {m.reality_check_p_value:.6f}  ← data-snooping corrected")
    else:
        print(f"  {'White Reality Check p':<30} N/A (no parameter grid)")

    # Trade diagnostics
    print()
    print(f"  {'── Trade diagnostics ──'}")
    total_trades = sum(len(w.simulation_result.trades) for w in result.valid_windows)
    print(f"  {'Total trades':<30} {total_trades}")
    if not math.isnan(m.exposure_fraction):
        print(f"  {'Avg exposure':<30} {m.exposure_fraction:.1%}  (fraction of bars in-market)")
    if not math.isnan(m.win_rate):
        print(f"  {'Win rate':<30} {m.win_rate:.1%}")
    if not math.isnan(m.avg_win_loss_ratio):
        wl_str = f"{m.avg_win_loss_ratio:.2f}×" if not math.isinf(m.avg_win_loss_ratio) else "∞ (no losses)"
        print(f"  {'Avg win / avg loss':<30} {wl_str}")
    if not math.isnan(m.avg_holding_days):
        print(f"  {'Avg holding period':<30} {m.avg_holding_days:.1f} days")

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
        # Skip inf and nan when finding the best finite value
        finite = [(i, v) for i, v in enumerate([a, b, c])
                  if not math.isnan(v) and abs(v) != float("inf")]
        if not finite:
            return _fmt_metric(a, "∞"), _fmt_metric(b, "∞"), _fmt_metric(c, "∞")

        best_val = max(v for _, v in finite) if higher_is_better else min(v for _, v in finite)
        best_idxs = {i for i, v in finite if v == best_val}

        results = []
        for i, v in enumerate([a, b, c]):
            s = _fmt_metric(v, "∞")
            if i in best_idxs and abs(v) != float("inf"):
                s += " ✓"
            results.append(s)
        return results[0], results[1], results[2]

    def best3_p(a: float, b: float, c: float) -> tuple[str, str, str]:
        """Lower is better for p-values."""
        if any(math.isnan(x) for x in [a, b, c]):
            return f"{a:.4f}", f"{b:.4f}", f"{c:.4f}"
        best_val = min(a, b, c)
        return (
            f"{a:.4f}" + (" ✓" if a == best_val else ""),
            f"{b:.4f}" + (" ✓" if b == best_val else ""),
            f"{c:.4f}" + (" ✓" if c == best_val else ""),
        )

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
    output_dir: Path = Path("."),
    train_yrs: int = TRAINING_WINDOW_YEARS,
    test_yrs: int = TESTING_WINDOW_YEARS,
    bootstrap_seed: int = BLOCK_BOOTSTRAP_SEED,
) -> None:
    """
    Sweep over (transaction_cost_rate, slippage_factor) grids and show how
    Fisher p-values degrade as execution costs increase.

    The breakeven cost - where Fisher p crosses 0.05 - is the most practically
    important single number for deciding whether to pursue a strategy live.

    bootstrap_seed is forwarded to every walk_forward call in the sweep so that
    results are reproducible when --seed is passed on the CLI.
    """
    cost_rates   = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    slip_factors = [0.0, 0.025, 0.05, 0.10, 0.20]

    sweeps: dict[str, dict[tuple[float, float], float]] = {}

    print("  Running MA cost sensitivity sweep...")
    sweeps["MA Crossover"] = cost_sensitivity_sweep(
        data, MovingAverageStrategy(),
        cost_rates=cost_rates,
        slippage_factors=slip_factors,
        training_window_years=train_yrs,
        testing_window_years=test_yrs,
        bootstrap_seed=bootstrap_seed,
    )

    print("  Running Kalman cost sensitivity sweep...")
    sweeps["Kalman Filter"] = cost_sensitivity_sweep(
        data, KalmanFilterStrategy(),
        cost_rates=cost_rates,
        slippage_factors=slip_factors,
        training_window_years=train_yrs,
        testing_window_years=test_yrs,
        bootstrap_seed=bootstrap_seed,
    )

    print("  Running Momentum cost sensitivity sweep...\n")
    sweeps["Momentum"] = cost_sensitivity_sweep(
        data, MomentumStrategy(),
        cost_rates=cost_rates,
        slippage_factors=slip_factors,
        training_window_years=train_yrs,
        testing_window_years=test_yrs,
        bootstrap_seed=bootstrap_seed,
    )

    for name, sweep in sweeps.items():
        _print_cost_table(name, sweep, cost_rates, slip_factors)

    _save_cost_heatmap(sweeps, cost_rates, slip_factors, output_dir=output_dir)


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
    output_dir: Path = Path("."),
) -> None:
    """Save interactive cost sensitivity heatmap as HTML.

    Uses include_plotlyjs=True (embeds ~3 MB of JS) rather than 'cdn' so the
    file is genuinely self-contained and works offline. The README previously
    claimed 'self-contained HTML, no server required' while depending on a CDN;
    embedding makes that claim accurate.
    """
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

        path = output_dir / "cost_sensitivity.html"
        # include_plotlyjs=True embeds the full Plotly JS bundle (~3 MB) so the
        # dashboard is genuinely self-contained and works offline.
        fig.write_html(str(path), include_plotlyjs=True, full_html=True)
        print(f"  Cost sensitivity heatmap → {path.resolve()}")

    except ImportError:
        # plotly is listed as a core dependency - if missing, re-install.
        print("  (Cost heatmap skipped: plotly not installed. Run `pip install plotly`.)")
    except Exception as e:
        # Unexpected plotly rendering error - surface it rather than swallowing.
        print(f"  (Cost heatmap failed unexpectedly: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
