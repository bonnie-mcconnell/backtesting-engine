"""
Cross-asset walk-forward validation.

Runs the same strategy independently on multiple tickers and produces a
comparison table showing whether the null result on SPY holds across asset
classes. A strategy that works on SPY but not on QQQ, TLT, or GLD is
probably fitting to a US equity bull-market regime. Consistently large
p-values across all four assets is a stronger null result.

Tickers: SPY (US large-cap equity), QQQ (US tech equity), TLT (long-duration
treasuries), GLD (gold). Start date 2005-01-01 is the earliest date where all
four have sufficient history for 3-year training windows.

Reference: Asness, C., Moskowitz, T. & Pedersen, L.H. (2013). Value and Momentum
Everywhere. Journal of Finance, 68(3), 929-985.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path

from backtesting_engine.benchmark import BenchmarkResult, compute_benchmark
from backtesting_engine.config import (
    BLOCK_BOOTSTRAP_SEED,
    TESTING_WINDOW_YEARS,
    TRAINING_WINDOW_YEARS,
)
from backtesting_engine.dashboard import build_dashboard
from backtesting_engine.data.ingestion import load_data
from backtesting_engine.data.validator import validate_data
from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.models import BacktestResult, MetricsResult
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward

# Minimum rows needed for at least one 3+1yr walk-forward window.
# MA long window (200 days) added to ensure the strategy can fit on train data.
_MIN_ROWS = (TRAINING_WINDOW_YEARS + TESTING_WINDOW_YEARS) * 252 + 200

_DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]


def run_multi_asset(
    tickers: list[str],
    start: str,
    end: str | None,
    execution: ExecutionConfig,
    train_years: int,
    test_years: int,
    bootstrap_seed: int,
    output_dir: Path,
) -> dict[str, tuple[BacktestResult, BenchmarkResult]]:
    """
    Run MA crossover walk-forward on each ticker independently.

    Args:
        tickers: List of ticker symbols (e.g. ["SPY", "QQQ", "TLT", "GLD"]).
        start: Start date string (ISO format).
        end: End date string (ISO format) or None for today.
        execution: ExecutionConfig applied to all tickers.
        train_years: Training window in years.
        test_years: Test window in years.
        bootstrap_seed: Random seed for bootstrap reproducibility.
        output_dir: Directory to write individual dashboards.

    Returns:
        Dict mapping ticker → (BacktestResult, BenchmarkResult).
        Tickers that fail (insufficient data, download error) are omitted with
        a warning printed to stdout.
    """
    results: dict[str, tuple[BacktestResult, BenchmarkResult]] = {}

    for ticker in tickers:
        print(f"\n  ── {ticker} {'─' * (40 - len(ticker))}")
        try:
            # yfinance end is exclusive; shift by one calendar day so the user's
            # --end date is included in the dataset (same logic as main.py).
            yf_end: str | None = None
            if end is not None:
                yf_end = (date.fromisoformat(end) + timedelta(days=1)).isoformat()

            data = load_data(ticker, start, end_date=yf_end, use_cache=True)
            validate_data(data, min_rows=_MIN_ROWS)
        except Exception as exc:
            print(f"  ⚠  {ticker}: skipped ({type(exc).__name__}: {exc})")
            continue

        print(
            f"  {len(data):,} days  "
            f"({data.index[0].date()} – {data.index[-1].date()})"
        )

        try:
            result = walk_forward(
                data,
                MovingAverageStrategy(),
                training_window_years=train_years,
                testing_window_years=test_years,
                execution=execution,
                bootstrap_seed=bootstrap_seed,
            )
        except ValueError as exc:
            print(f"  ⚠  {ticker}: walk_forward failed ({exc})")
            continue

        benchmark = compute_benchmark(result, data, execution=execution)

        # Write per-ticker dashboard.
        dash_path = output_dir / f"dashboard_ma_{ticker.lower()}.html"
        try:
            build_dashboard(
                result,
                dash_path,
                strategy_name_override=f"MA Crossover: {ticker}",
                benchmark=benchmark,
                price_data=data["close"],
            )
            print(f"  Dashboard → {dash_path}")
        except Exception as exc:
            # Dashboard failure should not abort the analysis.
            print(f"  ⚠  Dashboard failed for {ticker}: {exc}")

        _print_ticker_summary(ticker, result, benchmark)
        results[ticker] = (result, benchmark)

    return results


def _print_ticker_summary(
    ticker: str,
    result: BacktestResult,
    benchmark: BenchmarkResult,
) -> None:
    m: MetricsResult = result.summary_metrics
    n_windows = len(result.valid_windows)
    flat = result.flat_cash_window_count

    print(
        f"  Windows: {n_windows}  ({flat} flat-cash)  "
        f"Sharpe: {m.sharpe_ratio:.3f}  "
        f"Fisher p: {m.combined_p_value:.4f}  "
        f"vs BH: {benchmark.strategy_beats_benchmark_fraction:.0%}  "
        f"IR: {benchmark.information_ratio:.3f}"
    )
    rc = m.reality_check_p_value
    rc_str = f"{rc:.4f}" if not math.isnan(rc) else "N/A"
    print(f"  RC p: {rc_str}  Max DD: {m.max_drawdown:.1%}  BH DD: {benchmark.benchmark_max_drawdown:.1%}")


def _print_comparison_table(
    results: dict[str, tuple[BacktestResult, BenchmarkResult]],
) -> None:
    if not results:
        print("\n  No results to compare.")
        return

    print("\n" + "═" * 70)
    print("  Cross-Asset Comparison: MA Crossover")
    print("═" * 70)
    print(
        f"  {'Ticker':<8}  {'Sharpe':>7}  {'Fisher p':>9}  "
        f"{'RC p':>7}  {'vs BH':>7}  {'IR':>7}  {'Max DD':>8}"
    )
    print("  " + "─" * 66)

    for ticker, (result, benchmark) in results.items():
        m = result.summary_metrics
        rc = m.reality_check_p_value
        rc_str = f"{rc:.4f}" if not math.isnan(rc) else "   N/A"
        sig = "✓" if m.combined_p_value < 0.05 else " "
        print(
            f"  {ticker:<8}  "
            f"{m.sharpe_ratio:>7.3f}  "
            f"{m.combined_p_value:>8.4f}{sig}  "
            f"{rc_str:>7}  "
            f"{benchmark.strategy_beats_benchmark_fraction:>7.0%}  "
            f"{benchmark.information_ratio:>7.3f}  "
            f"{m.max_drawdown:>8.1%}"
        )

    print()

    # Verdict: count how many tickers are significant.
    n_sig = sum(
        1 for r, _ in results.values()
        if r.summary_metrics.combined_p_value < 0.05
    )
    n_total = len(results)

    print(
        f"  {n_sig}/{n_total} tickers significant at p<0.05. "
    )
    if n_sig == 0:
        print(
            "  Consistent null result across asset classes: "
            "strengthens the SPY finding.\n"
            "  The MA crossover has no detectable edge on these assets under "
            "realistic execution costs."
        )
    elif n_sig == n_total:
        print(
            "  Significant on all tickers: "
            "warrants further investigation with RC and benchmark comparison.\n"
            "  Check whether IR > 0 across all tickers before concluding genuine edge."
        )
    else:
        print(
            "  Mixed result: significant on some tickers but not others.\n"
            "  This is consistent with regime-specific edge rather than a genuine effect.\n"
            "  Inspect per-window results on significant tickers before concluding."
        )
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-asset walk-forward validation. "
            "Tests MA crossover across multiple tickers and produces a comparison table."
        ),
        epilog="Example: python -m backtesting_engine.multi_asset --tickers SPY QQQ TLT GLD",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=_DEFAULT_TICKERS,
        metavar="TICKER",
        help=f"Ticker symbols to test (default: {' '.join(_DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--start",
        default="2005-01-01",
        metavar="YYYY-MM-DD",
        help="Start date (default: 2005-01-01, earliest date with data for all default tickers)",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date, inclusive (default: today). Use --end 2024-12-31 for reproducibility.",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.001,
        metavar="RATE",
        help="Transaction cost rate per side (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.05,
        metavar="FACTOR",
        help="Slippage as fraction of daily high-low range (default: 0.05)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=1,
        metavar="BARS",
        help="Signal execution delay in bars (default: 1)",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=TRAINING_WINDOW_YEARS,
        metavar="N",
        help=f"Training window in years (default: {TRAINING_WINDOW_YEARS})",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        default=TESTING_WINDOW_YEARS,
        metavar="N",
        help=f"Test window in years (default: {TESTING_WINDOW_YEARS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BLOCK_BOOTSTRAP_SEED,
        metavar="N",
        help=f"Bootstrap random seed (default: {BLOCK_BOOTSTRAP_SEED})",
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
    print("  Cross-Asset Validation  ·  MA Crossover  ·  Walk-Forward")
    print(f"{'═' * 70}")
    print(f"  Tickers:     {' '.join(args.tickers)}")
    print(f"  Period:      {args.start} → {args.end or 'today'}")
    print(f"  Execution:   cost={args.cost:.4f}  slippage={args.slippage:.3f}  delay={args.delay}")
    print(f"  Windows:     {args.train_years}yr train / {args.test_years}yr test")
    print(f"  Seed:        {args.seed}\n")

    execution = ExecutionConfig(
        transaction_cost_rate=args.cost,
        slippage_factor=args.slippage,
        signal_delay=args.delay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = run_multi_asset(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        execution=execution,
        train_years=args.train_years,
        test_years=args.test_years,
        bootstrap_seed=args.seed,
        output_dir=output_dir,
    )

    _print_comparison_table(all_results)

    if not all_results:
        print("  No tickers completed successfully.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
