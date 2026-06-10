"""
Summary output: JSON and CSV serialisation for backtest results.

Writes one record per strategy run. The output is intended for:
  - Comparing runs across parameter settings or time periods
  - CI checks that fail if significance drops below a threshold
  - Downstream reporting without re-running the full backtest

JSON schema (one object):
  {
    "runs": [
      {
        "strategy":     str,
        "ticker":       str,
        "date_range":   {"start": str, "end": str},
        "execution":    {...},          # ExecutionConfig fields
        "metrics":      {...},          # MetricsResult fields
        "benchmark":    {...} | null    # BenchmarkResult fields, or null
      },
      ...
    ]
  }

CSV: one row per strategy, columns are the flattened JSON fields.
Nested objects are prefixed: execution.transaction_cost_rate,
benchmark.information_ratio, etc.

NaN is written as the empty string in CSV and as null in JSON so that
downstream tooling (pandas read_csv, jq, Excel) handles missing values
correctly. Inf is written as "Infinity" / "-Infinity" rather than null
so the distinction between "no data" and "unbounded" is preserved.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from backtesting_engine.benchmark import BenchmarkResult
from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.models import BacktestResult

# A single run entry: strategy name, result, optional benchmark, run metadata.
RunEntry = tuple[str, BacktestResult, BenchmarkResult | None]


def write_summary_json(
    runs: list[RunEntry],
    path: Path,
    *,
    ticker: str = "",
    date_range: tuple[str, str] | None = None,
    execution: ExecutionConfig | None = None,
) -> None:
    """
    Write a JSON summary of one or more strategy runs.

    Args:
        runs: List of (strategy_name, BacktestResult, BenchmarkResult | None).
        path: Output path. Parent directory must exist.
        ticker: Ticker symbol, written into each run record for traceability.
        date_range: (start_date, end_date) ISO strings. Written into each run record.
        execution: ExecutionConfig used for the run. Written into each run record.

    Raises:
        OSError: If the output path is not writable.
    """
    payload = {
        "runs": [
            _build_run_record(name, result, benchmark, ticker, date_range, execution)
            for name, result, benchmark in runs
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, cls=_SummaryEncoder), encoding="utf-8")


def write_summary_csv(
    runs: list[RunEntry],
    path: Path,
    *,
    ticker: str = "",
    date_range: tuple[str, str] | None = None,
    execution: ExecutionConfig | None = None,
) -> None:
    """
    Write a CSV summary of one or more strategy runs.

    Nested dict fields are flattened with dot-separated keys
    (e.g. ``execution.transaction_cost_rate``). NaN is written as an empty
    string; Inf as "Infinity" / "-Infinity". One header row, one data row
    per strategy.

    Args:
        runs: List of (strategy_name, BacktestResult, BenchmarkResult | None).
        path: Output path. Parent directory must exist.
        ticker: Ticker symbol written into each row.
        date_range: (start_date, end_date) ISO strings.
        execution: ExecutionConfig used for the run.

    Raises:
        OSError: If the output path is not writable.
    """
    records = [
        _build_run_record(name, result, benchmark, ticker, date_range, execution)
        for name, result, benchmark in runs
    ]
    flat_records = [_flatten(r) for r in records]

    # Use the key order from the first record so all rows have the same columns.
    fieldnames = list(flat_records[0].keys()) if flat_records else []

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in flat_records:
            writer.writerow({k: _csv_value(v) for k, v in row.items()})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_run_record(
    strategy_name: str,
    result: BacktestResult,
    benchmark: BenchmarkResult | None,
    ticker: str,
    date_range: tuple[str, str] | None,
    execution: ExecutionConfig | None,
) -> dict[str, Any]:
    """Build a single run record dict."""
    # date_range from CLI args (strings) or derived from the data
    if date_range is not None:
        dr: dict[str, str] = {"start": date_range[0], "end": date_range[1]}
    else:
        windows = result.valid_windows
        if windows:
            dr = {
                "start": str(windows[0].train_start.date()),
                "end": str(windows[-1].test_end.date()),
            }
        else:
            dr = {}

    record: dict[str, Any] = {
        "strategy": strategy_name,
        "ticker": ticker,
        "date_range": dr,
        "execution": _serialise_execution(execution),
        "metrics": _serialise_metrics(result),
        "benchmark": _serialise_benchmark(benchmark),
    }
    return record


def _serialise_execution(execution: ExecutionConfig | None) -> dict[str, Any]:
    if execution is None:
        return {}
    return {
        "transaction_cost_rate": execution.transaction_cost_rate,
        "slippage_factor": execution.slippage_factor,
        "signal_delay": execution.signal_delay,
    }


def _serialise_metrics(result: BacktestResult) -> dict[str, Any]:
    m = result.summary_metrics
    return {
        "sharpe_ratio": m.sharpe_ratio,
        "sortino_ratio": m.sortino_ratio,
        "max_drawdown": m.max_drawdown,
        "calmar_ratio": m.calmar_ratio,
        "omega_ratio": m.omega_ratio,
        "p_value": m.p_value,
        "combined_p_value": m.combined_p_value,
        "reality_check_p_value": m.reality_check_p_value,
        "reality_check_bh_p_value": m.reality_check_bh_p_value,
        "exposure_fraction": m.exposure_fraction,
        "trade_count": m.trade_count,
        "win_rate": m.win_rate,
        "avg_win_loss_ratio": m.avg_win_loss_ratio,
        "avg_holding_days": m.avg_holding_days,
        "window_count": len(result.valid_windows),
        "flat_cash_window_count": result.flat_cash_window_count,
    }


def _serialise_benchmark(benchmark: BenchmarkResult | None) -> dict[str, Any] | None:
    if benchmark is None:
        return None
    return {
        "benchmark_sharpe": benchmark.benchmark_sharpe,
        "benchmark_sortino": benchmark.benchmark_sortino,
        "benchmark_max_drawdown": benchmark.benchmark_max_drawdown,
        "information_ratio": benchmark.information_ratio,
        "sharpe_diff_t_stat": benchmark.sharpe_diff_t_stat,
        "sharpe_diff_p_value": benchmark.sharpe_diff_p_value,
        "strategy_beats_benchmark_fraction": benchmark.strategy_beats_benchmark_fraction,
    }


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict with dot-separated keys."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            result.update(_flatten(v, key))
        else:
            result[key] = v
    return result


class _SummaryEncoder(json.JSONEncoder):
    """
    JSON encoder that handles float edge cases correctly.

    Python's default json module serialises float("inf") as the bare word
    Infinity, which is invalid JSON. NaN becomes NaN, also invalid. Both
    are intercepted here and converted to valid JSON values.

    NaN → null (null is the JSON convention for "no data")
    Inf → "Infinity" / "-Infinity" (string, not null, to preserve the
          distinction between "unbounded" and "not computed")
    """

    def iterencode(self, obj: Any, _one_shot: bool = False) -> Any:
        # Walk the object tree before encoding to catch inf/nan floats
        # that would otherwise be written as invalid JSON literals.
        return super().iterencode(_sanitise(obj), _one_shot)


def _sanitise(obj: Any) -> Any:
    """Recursively replace inf/nan floats with JSON-safe values."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    return obj


def _csv_value(v: Any) -> str:
    """Convert a value to its CSV string representation."""
    if isinstance(v, float):
        if math.isnan(v):
            return ""          # empty cell = missing, consistent with pandas read_csv default
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return str(v)
    if v is None:
        return ""
    return str(v)
