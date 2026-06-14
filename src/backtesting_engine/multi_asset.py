"""
Cross-asset walk-forward validation.

Runs a strategy independently on multiple tickers and produces a comparison
table showing whether the null result on SPY holds across asset classes.
A strategy that works on SPY but not on QQQ, TLT, or GLD is probably fitting
to a US equity bull-market regime. Consistently large p-values across all four
assets and all three strategies is a materially stronger null result.

Three strategies are supported: MA crossover (--strategy ma), Kalman filter
trend-following (--strategy kalman), and time-series momentum
(--strategy momentum). --strategy all runs all three in sequence.

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
from backtesting_engine.strategy.base import BaseStrategy
from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
from backtesting_engine.strategy.momentum import MomentumStrategy
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import walk_forward

# Minimum rows needed for at least one 3+1yr walk-forward window.
# MA long window (200 days) added to ensure the strategy can fit on train data.
_MIN_ROWS = (TRAINING_WINDOW_YEARS + TESTING_WINDOW_YEARS) * 252 + 200

_DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]

# Maps CLI strategy name → (factory, short label for filenames and table headers).
# Factory is a zero-argument callable so each ticker gets a fresh strategy instance.
_STRATEGY_MAP: dict[str, tuple[type[BaseStrategy], str]] = {
    "ma":       (MovingAverageStrategy,  "MA Crossover"),
    "kalman":   (KalmanFilterStrategy,   "Kalman Filter"),
    "momentum": (MomentumStrategy,       "Momentum"),
}


def run_multi_asset(
    tickers: list[str],
    start: str,
    end: str | None,
    execution: ExecutionConfig,
    train_years: int,
    test_years: int,
    bootstrap_seed: int,
    output_dir: Path,
    strategy: str = "ma",
    no_dashboard: bool = False,
) -> dict[str, tuple[BacktestResult, BenchmarkResult]]:
    """
    Run walk-forward validation on each ticker independently.

    When strategy is "all", each of MA crossover, Kalman filter, and momentum
    is run on every ticker. The return dict is keyed by "{ticker}:{short_name}"
    (e.g. "SPY:ma", "SPY:kalman", "SPY:momentum") so all results are accessible
    from a single dict. When strategy is a single name, the dict is keyed by
    ticker only.

    Args:
        tickers: List of ticker symbols (e.g. ["SPY", "QQQ", "TLT", "GLD"]).
        start: Start date string (ISO format).
        end: End date string (ISO format) or None for today.
        execution: ExecutionConfig applied to all tickers.
        train_years: Training window in years.
        test_years: Test window in years.
        bootstrap_seed: Random seed for bootstrap reproducibility.
        output_dir: Directory to write individual dashboards.
        strategy: One of "ma", "kalman", "momentum", "all" (default: "ma").

    Returns:
        Dict mapping key → (BacktestResult, BenchmarkResult). Key is ticker
        when a single strategy is requested, "{ticker}:{short}" when "all".
        Tickers that fail (insufficient data, download error) are omitted with
        a warning printed to stdout.

    Raises:
        ValueError: If strategy is not one of the recognised names.
    """
    if strategy not in (*_STRATEGY_MAP, "all"):
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Valid options: {sorted(_STRATEGY_MAP)} + ['all']."
        )

    strategies_to_run: list[tuple[str, str]] = (
        [(k, v[1]) for k, v in _STRATEGY_MAP.items()]
        if strategy == "all"
        else [(strategy, _STRATEGY_MAP[strategy][1])]
    )

    results: dict[str, tuple[BacktestResult, BenchmarkResult]] = {}
    multi_strategy = strategy == "all"

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

        for strat_key, strat_label in strategies_to_run:
            strat_cls = _STRATEGY_MAP[strat_key][0]
            result_key = f"{ticker}:{strat_key}" if multi_strategy else ticker

            if multi_strategy:
                print(f"  ··· {strat_label}")

            try:
                result = walk_forward(
                    data,
                    strat_cls(),
                    training_window_years=train_years,
                    testing_window_years=test_years,
                    execution=execution,
                    bootstrap_seed=bootstrap_seed,
                )
            except ValueError as exc:
                print(f"  ⚠  {ticker} ({strat_label}): walk_forward failed ({exc})")
                continue

            benchmark = compute_benchmark(result, data, execution=execution)

            if not no_dashboard:
                dash_path = output_dir / f"dashboard_{strat_key}_{ticker.lower()}.html"
                try:
                    build_dashboard(
                        result,
                        dash_path,
                        strategy_name_override=f"{strat_label}: {ticker}",
                        benchmark=benchmark,
                        price_data=data["close"],
                    )
                    print(f"  Dashboard → {dash_path}")
                except Exception as exc:
                    # Dashboard failure should not abort the analysis.
                    print(f"  ⚠  Dashboard failed for {ticker} ({strat_label}): {exc}")

            _print_ticker_summary(ticker, result, benchmark)
            results[result_key] = (result, benchmark)

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
    rc_bh = m.reality_check_bh_p_value
    rc_bh_str = f"{rc_bh:.4f}" if not math.isnan(rc_bh) else "N/A"
    print(
        f"  RC p (cash): {rc_str}  RC p (B&H): {rc_bh_str}"
        f"  Max DD: {m.max_drawdown:.1%}  BH DD: {benchmark.benchmark_max_drawdown:.1%}"
    )


def _print_comparison_table(
    results: dict[str, tuple[BacktestResult, BenchmarkResult]],
    strategy_label: str = "MA Crossover",
) -> None:
    if not results:
        print("\n  No results to compare.")
        return

    print("\n" + "═" * 80)
    print(f"  Cross-Asset Comparison: {strategy_label}")
    print("═" * 80)
    print(
        f"  {'Ticker':<8}  {'Sharpe':>7}  {'Fisher p':>9}  "
        f"{'RC(cash)':>9}  {'RC(B&H)':>9}  {'vs BH':>7}  {'IR':>7}  {'Max DD':>8}"
    )
    print("  " + "─" * 76)

    for ticker, (result, benchmark) in results.items():
        m = result.summary_metrics
        rc = m.reality_check_p_value
        rc_str = f"{rc:.4f}" if not math.isnan(rc) else "   N/A"
        rc_bh = m.reality_check_bh_p_value
        rc_bh_str = f"{rc_bh:.4f}" if not math.isnan(rc_bh) else "   N/A"
        sig = "✓" if m.combined_p_value < 0.05 else " "
        print(
            f"  {ticker:<8}  "
            f"{m.sharpe_ratio:>7.3f}  "
            f"{m.combined_p_value:>8.4f}{sig}  "
            f"{rc_str:>9}  "
            f"{rc_bh_str:>9}  "
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
            f"  {strategy_label} has no detectable edge on these assets under "
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
            "Tests a strategy across multiple tickers and produces a comparison table. "
            "Use --strategy all to run MA crossover, Kalman filter, and momentum in sequence."
        ),
        epilog=(
            "Examples:\n"
            "  backtesting-multi --tickers SPY QQQ TLT GLD\n"
            "  backtesting-multi --strategy momentum --tickers SPY QQQ TLT GLD\n"
            "  backtesting-multi --strategy all --tickers SPY QQQ"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=_DEFAULT_TICKERS,
        metavar="TICKER",
        help=f"Ticker symbols to test (default: {' '.join(_DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--strategy",
        default="ma",
        choices=["ma", "kalman", "momentum", "all"],
        help=(
            "Strategy to run on each ticker. "
            "'all' runs MA crossover, Kalman filter, and momentum in sequence "
            "and produces separate dashboards for each. "
            "Note: Kalman runs take 4-6 minutes per ticker. "
            "(default: ma)"
        ),
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
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        default=False,
        help="Skip per-ticker dashboard generation. Results are still printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    strategy_label = (
        "All Strategies"
        if args.strategy == "all"
        else _STRATEGY_MAP[args.strategy][1]
    )

    print(f"\n{'═' * 70}")
    print(f"  Cross-Asset Validation  ·  {strategy_label}  ·  Walk-Forward")
    print(f"{'═' * 70}")
    print(f"  Tickers:     {' '.join(args.tickers)}")
    print(f"  Strategy:    {strategy_label}")
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
        strategy=args.strategy,
        no_dashboard=args.no_dashboard,
    )

    if args.strategy == "all":
        # Print one table per strategy so the reader can compare strategies
        # within an asset class as well as across asset classes.
        for strat_key, (_, strat_label) in _STRATEGY_MAP.items():
            strat_results = {
                k.split(":")[0]: v
                for k, v in all_results.items()
                if k.endswith(f":{strat_key}")
            }
            _print_comparison_table(strat_results, strategy_label=strat_label)
    else:
        _print_comparison_table(all_results, strategy_label=strategy_label)

    if not all_results:
        print("  No tickers completed successfully.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
