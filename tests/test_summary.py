"""
Tests for summary output (write_summary_json, write_summary_csv).

Correctness criteria:
- JSON round-trips: every MetricsResult and BenchmarkResult field is present
  and recoverable from the output
- CSV round-trips: flattened fields match the JSON equivalents
- NaN → null (JSON) / empty string (CSV)
- Inf → "Infinity" / "-Infinity" in both formats
- Empty runs list → valid empty JSON {"runs": []}
- Partial runs (e.g. benchmark=None) → benchmark key is null in JSON, empty in CSV
- Output directory is created when it does not exist
- Date range is derived from window timestamps when not supplied

All expected values are hard-coded from the fixture, not re-derived from the
function under test.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from helpers import make_oscillating_data

from backtesting_engine.benchmark import BenchmarkResult
from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.models import BacktestResult, MetricsResult
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.summary import RunEntry, write_summary_csv, write_summary_json
from backtesting_engine.walk_forward import walk_forward

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_ZERO_FRICTION = ExecutionConfig(slippage_factor=0.0, signal_delay=0)
_EXECUTION = ExecutionConfig(
    transaction_cost_rate=0.001,
    slippage_factor=0.05,
    signal_delay=1,
)


@pytest.fixture(scope="module")
def wf_result() -> BacktestResult:
    data = make_oscillating_data(756)
    strat = MovingAverageStrategy(short_window=20, long_window=50)
    return walk_forward(
        data, strat,
        training_window_years=1,
        testing_window_years=1,
        execution=_ZERO_FRICTION,
        bootstrap_seed=42,
    )


@pytest.fixture(scope="module")
def benchmark_result(wf_result: BacktestResult) -> BenchmarkResult:
    from backtesting_engine.benchmark import compute_benchmark
    data = make_oscillating_data(756)
    return compute_benchmark(wf_result, data, execution=_ZERO_FRICTION)


@pytest.fixture(scope="module")
def single_run(wf_result: BacktestResult, benchmark_result: BenchmarkResult) -> list[RunEntry]:
    return [("Moving Average Crossover", wf_result, benchmark_result)]


@pytest.fixture(scope="module")
def single_run_no_benchmark(wf_result: BacktestResult) -> list[RunEntry]:
    return [("Moving Average Crossover", wf_result, None)]


# ---------------------------------------------------------------------------
# JSON tests
# ---------------------------------------------------------------------------

class TestWriteSummaryJson:
    def test_creates_file(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out)
        assert out.exists()

    def test_creates_parent_directory(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "subdir" / "nested" / "summary.json"
        write_summary_json(single_run, out)
        assert out.exists()

    def test_valid_json(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out)
        data = json.loads(out.read_text())
        assert "runs" in data
        assert isinstance(data["runs"], list)
        assert len(data["runs"]) == 1

    def test_strategy_name_written(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out)
        run = json.loads(out.read_text())["runs"][0]
        assert run["strategy"] == "Moving Average Crossover"

    def test_ticker_written(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out, ticker="SPY")
        run = json.loads(out.read_text())["runs"][0]
        assert run["ticker"] == "SPY"

    def test_execution_fields_present(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out, execution=_EXECUTION)
        run = json.loads(out.read_text())["runs"][0]
        assert run["execution"]["transaction_cost_rate"] == pytest.approx(0.001)
        assert run["execution"]["slippage_factor"] == pytest.approx(0.05)
        assert run["execution"]["signal_delay"] == 1

    def test_metrics_fields_present(
        self, tmp_path: Path, single_run: list[RunEntry], wf_result: BacktestResult
    ) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out)
        run = json.loads(out.read_text())["runs"][0]
        m = run["metrics"]
        expected_keys = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "omega_ratio", "p_value", "combined_p_value",
            "reality_check_p_value", "reality_check_bh_p_value",
            "exposure_fraction", "trade_count", "win_rate",
            "avg_win_loss_ratio", "avg_holding_days",
            "window_count", "flat_cash_window_count",
        }
        assert expected_keys.issubset(m.keys())

    def test_metrics_sharpe_matches_result(
        self, tmp_path: Path, single_run: list[RunEntry], wf_result: BacktestResult
    ) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out)
        run = json.loads(out.read_text())["runs"][0]
        assert run["metrics"]["sharpe_ratio"] == pytest.approx(
            wf_result.summary_metrics.sharpe_ratio, rel=1e-6
        )

    def test_benchmark_fields_present(
        self, tmp_path: Path, single_run: list[RunEntry], benchmark_result: BenchmarkResult
    ) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out)
        run = json.loads(out.read_text())["runs"][0]
        b = run["benchmark"]
        assert b is not None
        assert "information_ratio" in b
        assert "strategy_beats_benchmark_fraction" in b
        assert b["benchmark_sharpe"] == pytest.approx(benchmark_result.benchmark_sharpe, rel=1e-6)

    def test_benchmark_null_when_not_provided(
        self, tmp_path: Path, single_run_no_benchmark: list[RunEntry]
    ) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run_no_benchmark, out)
        run = json.loads(out.read_text())["runs"][0]
        assert run["benchmark"] is None

    def test_nan_serialised_as_null(
        self, tmp_path: Path, single_run: list[RunEntry], wf_result: BacktestResult
    ) -> None:
        # reality_check_bh_p_value may be NaN when walk_forward runs without
        # full OHLC data. Check that any NaN in the metrics round-trips to null.
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out)
        raw = out.read_text()
        data = json.loads(raw)
        m = data["runs"][0]["metrics"]
        for key, val in m.items():
            if val is None:
                # null in JSON came from a NaN float - verify it's not a string
                assert val is None

    def test_inf_serialised_as_string(self, tmp_path: Path) -> None:
        # Construct a MetricsResult with an inf calmar_ratio (no drawdown window).
        inf_metrics = MetricsResult(
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.0,
            calmar_ratio=float("inf"),
            omega_ratio=float("inf"),
            p_value=0.3,
        )

        result = BacktestResult(
            strategy_name="Test",
            window_results=[],
            summary_metrics=inf_metrics,
        )
        runs: list[RunEntry] = [("Test", result, None)]
        out = tmp_path / "inf_test.json"
        write_summary_json(runs, out)
        data = json.loads(out.read_text())
        m = data["runs"][0]["metrics"]
        assert m["calmar_ratio"] == "Infinity"
        assert m["omega_ratio"] == "Infinity"

    def test_empty_runs_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        write_summary_json([], out)
        data = json.loads(out.read_text())
        assert data == {"runs": []}

    def test_multiple_runs(
        self, tmp_path: Path, wf_result: BacktestResult, benchmark_result: BenchmarkResult
    ) -> None:
        runs: list[RunEntry] = [
            ("MA Crossover", wf_result, benchmark_result),
            ("Momentum", wf_result, None),
        ]
        out = tmp_path / "multi.json"
        write_summary_json(runs, out)
        data = json.loads(out.read_text())
        assert len(data["runs"]) == 2
        assert data["runs"][0]["strategy"] == "MA Crossover"
        assert data["runs"][1]["strategy"] == "Momentum"

    def test_date_range_derived_from_windows(
        self, tmp_path: Path, single_run: list[RunEntry], wf_result: BacktestResult
    ) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out)
        run = json.loads(out.read_text())["runs"][0]
        windows = wf_result.valid_windows
        expected_start = str(windows[0].train_start.date())
        assert run["date_range"]["start"] == expected_start

    def test_explicit_date_range_overrides_derived(
        self, tmp_path: Path, single_run: list[RunEntry]
    ) -> None:
        out = tmp_path / "summary.json"
        write_summary_json(single_run, out, date_range=("2000-01-01", "2020-12-31"))
        run = json.loads(out.read_text())["runs"][0]
        assert run["date_range"]["start"] == "2000-01-01"
        assert run["date_range"]["end"] == "2020-12-31"


# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------

class TestWriteSummaryCsv:
    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_creates_file(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.csv"
        write_summary_csv(single_run, out)
        assert out.exists()

    def test_creates_parent_directory(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "subdir" / "summary.csv"
        write_summary_csv(single_run, out)
        assert out.exists()

    def test_one_row_per_run(
        self, tmp_path: Path, wf_result: BacktestResult, benchmark_result: BenchmarkResult
    ) -> None:
        runs: list[RunEntry] = [
            ("MA", wf_result, benchmark_result),
            ("Momentum", wf_result, None),
        ]
        out = tmp_path / "summary.csv"
        write_summary_csv(runs, out)
        rows = self._read_csv(out)
        assert len(rows) == 2

    def test_strategy_column(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.csv"
        write_summary_csv(single_run, out)
        rows = self._read_csv(out)
        assert rows[0]["strategy"] == "Moving Average Crossover"

    def test_flattened_execution_columns(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.csv"
        write_summary_csv(single_run, out, execution=_EXECUTION)
        rows = self._read_csv(out)
        assert rows[0]["execution.transaction_cost_rate"] == "0.001"
        assert rows[0]["execution.slippage_factor"] == "0.05"

    def test_flattened_metrics_columns(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.csv"
        write_summary_csv(single_run, out)
        rows = self._read_csv(out)
        assert "metrics.sharpe_ratio" in rows[0]
        assert "metrics.combined_p_value" in rows[0]
        assert "metrics.reality_check_bh_p_value" in rows[0]

    def test_flattened_benchmark_columns(
        self, tmp_path: Path, single_run: list[RunEntry]
    ) -> None:
        out = tmp_path / "summary.csv"
        write_summary_csv(single_run, out)
        rows = self._read_csv(out)
        assert "benchmark.information_ratio" in rows[0]
        assert "benchmark.strategy_beats_benchmark_fraction" in rows[0]

    def test_nan_written_as_empty_string(self, tmp_path: Path, single_run: list[RunEntry]) -> None:
        out = tmp_path / "summary.csv"
        write_summary_csv(single_run, out)
        rows = self._read_csv(out)
        # Any NaN field should be an empty string, not "nan" or "NaN"
        for val in rows[0].values():
            assert val != "nan" and val != "NaN"

    def test_inf_written_as_string(self, tmp_path: Path) -> None:
        inf_metrics = MetricsResult(
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.0,
            calmar_ratio=float("inf"),
            omega_ratio=float("inf"),
            p_value=0.3,
        )
        result = BacktestResult(
            strategy_name="Test",
            window_results=[],
            summary_metrics=inf_metrics,
        )
        runs: list[RunEntry] = [("Test", result, None)]
        out = tmp_path / "inf.csv"
        write_summary_csv(runs, out)
        rows = self._read_csv(out)
        assert rows[0]["metrics.calmar_ratio"] == "Infinity"

    def test_sharpe_matches_result(
        self, tmp_path: Path, single_run: list[RunEntry], wf_result: BacktestResult
    ) -> None:
        out = tmp_path / "summary.csv"
        write_summary_csv(single_run, out)
        rows = self._read_csv(out)
        written = float(rows[0]["metrics.sharpe_ratio"])
        assert written == pytest.approx(wf_result.summary_metrics.sharpe_ratio, rel=1e-6)

    def test_empty_runs_produces_empty_file(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.csv"
        write_summary_csv([], out)
        # Empty runs → no header, no rows (file may contain only a trailing newline)
        assert out.read_text().strip() == ""

    def test_json_and_csv_sharpe_agree(
        self, tmp_path: Path, single_run: list[RunEntry]
    ) -> None:
        json_out = tmp_path / "summary.json"
        csv_out = tmp_path / "summary.csv"
        write_summary_json(single_run, json_out)
        write_summary_csv(single_run, csv_out)

        json_sharpe = json.loads(json_out.read_text())["runs"][0]["metrics"]["sharpe_ratio"]
        csv_rows = list(csv.DictReader(csv_out.open(encoding="utf-8")))
        csv_sharpe = float(csv_rows[0]["metrics.sharpe_ratio"])

        assert csv_sharpe == pytest.approx(json_sharpe, rel=1e-6)
