"""
Unit tests for the walk-forward orchestrator.

Performance design
------------------
Most tests here inspect *properties* of a BacktestResult rather than verifying
the walk_forward() *call itself*. For those tests, we use module-scoped cached
fixtures (wf_result_504, wf_result_756) from conftest.py that run walk_forward
exactly once per module and share the result across all tests that request them.

Tests that verify walk_forward's *behaviour when called* (input validation,
seed reproducibility, strategy-specific output) keep their own calls.

This reduces the number of walk_forward() calls in this module from 28 to 5:
  - wf_result_504 (module fixture): 1 call
  - wf_result_756 (module fixture): 1 call
  - test_insufficient_data_raises: 1 call (expected to raise)
  - test_bootstrap_seed_produces_deterministic_results: 2 calls (seed test)
  - test_rc_p_nan_for_kalman_strategy: 1 call (Kalman-specific)
  - TestWalkForwardInputValidation: 5 calls (bad-input; raise early so fast)

All walk_forward calls use execution=_ZERO_FRICTION (slippage=0, signal_delay=0)
because the synthetic data only has a 'close' column. Tests that specifically
exercise the execution model are in test_execution.py which uses OHLCV fixtures.

Shared fixtures (wf_result_504, wf_result_756, oscillating_504, strategy) are
defined in conftest.py and auto-injected by pytest - no import required.
"""

import math

import numpy as np
import pandas as pd
import pytest
from helpers import make_oscillating_data

from backtesting_engine.execution import ExecutionConfig
from backtesting_engine.models import BacktestResult
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.walk_forward import (
    _fisher_combined_p,
    _flat_cash_metrics,
    walk_forward,
)

# Zero-friction config for unit tests using close-only synthetic data.
_ZERO_FRICTION = ExecutionConfig(slippage_factor=0.0, signal_delay=0)


# ── Window structure ──────────────────────────────────────────────────────────

class TestWindowCount:
    def test_one_window_from_504_days(self, wf_result_504: BacktestResult) -> None:
        assert len(wf_result_504.window_results) == 1

    def test_two_windows_from_756_days(self, wf_result_756: BacktestResult) -> None:
        assert len(wf_result_756.valid_windows) == 2

    def test_insufficient_data_raises(self, strategy: MovingAverageStrategy) -> None:
        dates = pd.date_range("2010-01-01", periods=100, freq="B")
        data = pd.DataFrame({"close": np.linspace(100.0, 110.0, 100)}, index=dates)
        with pytest.raises(ValueError):
            walk_forward(
                data, strategy,
                training_window_years=1, testing_window_years=1,
                execution=_ZERO_FRICTION,
            )


# ── Look-ahead bias ───────────────────────────────────────────────────────────

class TestNoLookaheadBias:
    def test_test_start_always_after_train_end(
        self, wf_result_504: BacktestResult
    ) -> None:
        """Test period must begin strictly after the training period ends.

        A test_start == train_end would allow the model to train on the bar
        it is then asked to predict - a direct form of look-ahead bias.
        """
        for w in wf_result_504.valid_windows:
            assert w.test_start > w.train_end, (
                f"Window has test_start={w.test_start} <= train_end={w.train_end}"
            )

    def test_no_date_overlap_between_train_and_test(
        self, wf_result_756: BacktestResult
    ) -> None:
        for w in wf_result_756.valid_windows:
            assert w.test_start > w.train_end


# ── Window advancement ────────────────────────────────────────────────────────

class TestWindowAdvancement:
    def test_each_window_advances_by_test_days(
        self, wf_result_756: BacktestResult
    ) -> None:
        """Consecutive windows must be contiguous: no gap and no overlap.

        Window k+1 test_start must be exactly one business day after window k
        test_end. A gap means data is silently discarded; an overlap means the
        same bar contributes to two test periods.
        """
        valid = wf_result_756.valid_windows
        assert len(valid) >= 2
        assert valid[1].test_start == valid[0].test_end + pd.offsets.BDay(1), (
            f"Window 1 test_start={valid[1].test_start} should be "
            f"1 BDay after window 0 test_end={valid[0].test_end}"
        )


# ── BacktestResult properties ─────────────────────────────────────────────────

class TestBacktestResult:
    def test_strategy_name_is_class_name(
        self, wf_result_504: BacktestResult
    ) -> None:
        assert wf_result_504.strategy_name == "MovingAverageStrategy"

    def test_summary_metrics_is_not_none(
        self, wf_result_504: BacktestResult
    ) -> None:
        assert wf_result_504.summary_metrics is not None

    def test_summary_sharpe_is_float(
        self, wf_result_504: BacktestResult
    ) -> None:
        assert isinstance(wf_result_504.summary_metrics.sharpe_ratio, float)

    def test_flat_cash_window_count_is_non_negative(
        self, wf_result_504: BacktestResult
    ) -> None:
        assert wf_result_504.flat_cash_window_count >= 0

    def test_valid_windows_have_skipped_false(
        self, wf_result_504: BacktestResult
    ) -> None:
        for w in wf_result_504.valid_windows:
            assert not w.skipped

    def test_combined_p_value_in_unit_interval(
        self, wf_result_504: BacktestResult
    ) -> None:
        p = wf_result_504.summary_metrics.combined_p_value
        assert 0.0 <= p <= 1.0, f"combined_p_value={p} not in [0, 1]"


# ── Fisher combined p ─────────────────────────────────────────────────────────

class TestFisherCombinedP:
    """Pure function tests - no walk_forward needed."""

    def test_all_significant_produces_small_combined(self) -> None:
        combined = _fisher_combined_p([0.01, 0.02, 0.01, 0.03])
        assert combined < 0.001

    def test_all_large_produces_large_combined(self) -> None:
        combined = _fisher_combined_p([0.5, 0.6, 0.7, 0.5])
        assert combined > 0.05

    def test_combined_in_unit_interval(self) -> None:
        combined = _fisher_combined_p([0.1, 0.3, 0.05, 0.8])
        assert 0.0 <= combined <= 1.0

    def test_single_p_value(self) -> None:
        combined = _fisher_combined_p([0.05])
        assert 0.0 <= combined <= 1.0

    def test_more_windows_amplifies_weak_signal(self) -> None:
        """Fisher's method gains power with more independent tests at the same level."""
        p4 = _fisher_combined_p([0.1] * 4)
        p16 = _fisher_combined_p([0.1] * 16)
        assert p16 < p4


# ── Flat-cash window semantics ────────────────────────────────────────────────

class TestFlatCashMetrics:
    """
    A no-trade window holds cash - it is a valid result, not a skipped window.

    The specific numeric values matter for correctness of summary aggregation:
      - Sharpe=0.0 (not nan): nan propagates incorrectly through np.nanmean
      - Sortino=0.0 (not inf): inf is silently excluded from nanmean,
        overstating aggregate Sortino
      - Omega=1.0 (not inf): same silent exclusion problem
      - p_value=1.0: no evidence of skill in a flat-cash window
      - max_drawdown=0.0: cash has no drawdown by definition

    If any value is wrong, summary metrics silently misrepresent performance.
    """

    def test_sharpe_is_zero_not_nan(self) -> None:
        assert _flat_cash_metrics().sharpe_ratio == 0.0

    def test_drawdown_is_zero(self) -> None:
        assert _flat_cash_metrics().max_drawdown == 0.0

    def test_p_value_is_one(self) -> None:
        assert _flat_cash_metrics().p_value == 1.0

    def test_exposure_is_zero(self) -> None:
        assert _flat_cash_metrics().exposure_fraction == 0.0

    def test_trade_count_is_zero(self) -> None:
        assert _flat_cash_metrics().trade_count == 0

    def test_sortino_is_zero_not_inf(self) -> None:
        m = _flat_cash_metrics()
        assert m.sortino_ratio == 0.0
        assert math.isfinite(m.sortino_ratio)

    def test_omega_is_one_not_inf(self) -> None:
        m = _flat_cash_metrics()
        assert m.omega_ratio == 1.0
        assert math.isfinite(m.omega_ratio)


# ── Active parameter storage ──────────────────────────────────────────────────

class TestActiveParamsStorage:
    def test_ma_windows_store_short_and_long(
        self, wf_result_504: BacktestResult
    ) -> None:
        for w in wf_result_504.valid_windows:
            assert "short_window" in w.active_params
            assert "long_window" in w.active_params

    def test_params_are_valid_positive_integers(
        self, wf_result_504: BacktestResult
    ) -> None:
        for w in wf_result_504.valid_windows:
            s = w.active_params["short_window"]
            lng = w.active_params["long_window"]
            assert isinstance(s, int) and isinstance(lng, int)
            assert s > 0 and lng > 0
            assert s < lng

    def test_param_evolution_length_matches_valid_windows(
        self, wf_result_504: BacktestResult
    ) -> None:
        assert len(wf_result_504.param_evolution) == len(wf_result_504.valid_windows)

    def test_param_evolution_contains_short_window(
        self, wf_result_504: BacktestResult
    ) -> None:
        for p in wf_result_504.param_evolution:
            assert "short_window" in p


# ── Reality Check integration ─────────────────────────────────────────────────

class TestRealityCheckWithTestReturns:
    def test_rc_p_value_is_finite_for_ma_strategy(
        self, wf_result_504: BacktestResult
    ) -> None:
        assert not math.isnan(wf_result_504.summary_metrics.reality_check_p_value), (
            "MA strategy has a candidate grid; RC p should be finite"
        )

    def test_rc_p_value_in_unit_interval(
        self, wf_result_504: BacktestResult
    ) -> None:
        p = wf_result_504.summary_metrics.reality_check_p_value
        assert 0.0 <= p <= 1.0

    def test_rc_p_nan_for_kalman_strategy(
        self, oscillating_504: pd.DataFrame
    ) -> None:
        """Kalman has no candidate grid → RC is undefined → must be NaN, not crash."""
        from backtesting_engine.strategy.kalman_filter import KalmanFilterStrategy
        result = walk_forward(
            oscillating_504, KalmanFilterStrategy(),
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION,
        )
        assert math.isnan(result.summary_metrics.reality_check_p_value)

    def test_bootstrap_seed_produces_deterministic_results(
        self, oscillating_504: pd.DataFrame, strategy: MovingAverageStrategy
    ) -> None:
        """Same seed must produce bit-identical p-values every run."""
        r1 = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION, bootstrap_seed=42,
        )
        r2 = walk_forward(
            oscillating_504, strategy,
            training_window_years=1, testing_window_years=1,
            execution=_ZERO_FRICTION, bootstrap_seed=42,
        )
        assert r1.summary_metrics.combined_p_value == r2.summary_metrics.combined_p_value
        assert (
            r1.summary_metrics.reality_check_p_value
            == r2.summary_metrics.reality_check_p_value
        )


# ── Summary metric aggregation ────────────────────────────────────────────────

class TestSummaryMetricAggregation:
    """
    Summary metrics aggregate per-window results in deliberate ways:
      - Sharpe: mean of per-window Sharpes (not stitched)
      - Calmar: computed on the full stitched return series (not per-window mean)
      - Max drawdown: worst per-window value (not mean)

    Each of these is a deliberate design choice with a specific rationale.
    Getting them wrong produces plausible-looking but incorrect numbers.
    """

    def test_max_drawdown_is_worst_not_mean(
        self, wf_result_756: BacktestResult
    ) -> None:
        valid = wf_result_756.valid_windows
        assert len(valid) >= 2, "Need at least 2 windows"
        per_window_dds = [w.metrics_result.max_drawdown for w in valid]
        worst = min(per_window_dds)
        mean = float(sum(per_window_dds) / len(per_window_dds))
        assert abs(wf_result_756.summary_metrics.max_drawdown - worst) < 1e-8, (
            f"summary max_dd={wf_result_756.summary_metrics.max_drawdown:.6f} "
            f"should equal worst={worst:.6f}, not mean={mean:.6f}"
        )

    def test_calmar_is_stitched_not_per_window_mean(
        self, wf_result_756: BacktestResult
    ) -> None:
        valid = wf_result_756.valid_windows
        assert valid
        from backtesting_engine.metrics import _calmar
        all_rets = []
        for w in valid:
            pv = w.simulation_result.portfolio_values
            if pv is not None and len(pv) > 1:
                all_rets.append(pv.pct_change().dropna().to_numpy())
        assert all_rets
        expected = _calmar(np.concatenate(all_rets))
        assert abs(wf_result_756.summary_metrics.calmar_ratio - expected) < 1e-6

    def test_sharpe_is_mean_of_per_window_not_stitched(
        self, wf_result_756: BacktestResult
    ) -> None:
        valid = wf_result_756.valid_windows
        assert valid
        per_sharpes = [w.metrics_result.sharpe_ratio for w in valid]
        expected = float(np.mean(per_sharpes))
        assert abs(wf_result_756.summary_metrics.sharpe_ratio - expected) < 1e-8


# ── Input validation ──────────────────────────────────────────────────────────

class TestWalkForwardInputValidation:
    """Bad-input validation. Each test raises early so the suite stays fast."""

    def test_zero_training_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=0,
                         testing_window_years=1, execution=_ZERO_FRICTION)

    def test_zero_testing_years_raises(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=1,
                         testing_window_years=0, execution=_ZERO_FRICTION)

    def test_negative_training_years_raises(
        self, strategy: MovingAverageStrategy
    ) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=-1,
                         testing_window_years=1, execution=_ZERO_FRICTION)

    def test_negative_testing_years_raises(
        self, strategy: MovingAverageStrategy
    ) -> None:
        data = make_oscillating_data(756)
        with pytest.raises(ValueError, match="positive"):
            walk_forward(data, strategy, training_window_years=1,
                         testing_window_years=-1, execution=_ZERO_FRICTION)

    def test_valid_years_do_not_raise(self, strategy: MovingAverageStrategy) -> None:
        data = make_oscillating_data(756)
        try:
            walk_forward(data, strategy, training_window_years=1,
                         testing_window_years=1, execution=_ZERO_FRICTION)
        except ValueError as e:
            pytest.fail(f"Valid years raised ValueError: {e}")
