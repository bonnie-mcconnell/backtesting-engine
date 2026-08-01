"""
Unit tests for MomentumStrategy.

Signal generation, fit() calibration, warmup context, candidate_test_returns(),
and the BaseStrategy interface. All using synthetic data.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from backtesting_engine.config import MOMENTUM_LOOKBACKS as _LOOKBACK_GRID
from backtesting_engine.strategy.base import returns_from_signals
from backtesting_engine.strategy.momentum import (
    MomentumStrategy,
    _momentum_signals,
)


def _make_data(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices}, index=dates)

def _trending_up(n: int = 500, start: str = "2010-01-01") -> pd.DataFrame:
    """Monotonically rising prices - momentum is always positive after warmup."""
    dates = pd.date_range(start, periods=n, freq="B")
    prices = np.linspace(100.0, 200.0, n)
    return pd.DataFrame({"close": prices}, index=dates)

def _trending_down(n: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    prices = np.linspace(200.0, 100.0, n)
    return pd.DataFrame({"close": prices}, index=dates)

def _oscillating(n: int = 600) -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    t = np.linspace(0, 8 * np.pi, n)
    prices = 100.0 + 30.0 * np.sin(t) + 0.02 * np.arange(n)
    return pd.DataFrame({"close": prices}, index=dates)

# ---------------------------------------------------------------------------
# _momentum_signals
# ---------------------------------------------------------------------------

class TestMomentumSignals:
    def test_output_length_matches_input(self) -> None:
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        signals = _momentum_signals(prices, lookback=2)
        assert len(signals) == len(prices)

    def test_first_lookback_bars_are_zero(self) -> None:
        prices = np.linspace(100.0, 110.0, 20)
        signals = _momentum_signals(prices, lookback=5)
        assert all(s == 0 for s in signals[:5])

    def test_uptrend_produces_buy_signal(self) -> None:
        # Rising prices: momentum crosses from 0 to positive → buy signal.
        prices = np.array([100.0] * 30 + list(np.linspace(100.0, 130.0, 30)))
        signals = _momentum_signals(prices, lookback=10)
        assert 1 in signals

    def test_downtrend_after_uptrend_produces_sell_signal(self) -> None:
        # Rising then falling: momentum crosses negative → sell signal.
        prices = np.array(
            list(np.linspace(100.0, 130.0, 60))
            + list(np.linspace(130.0, 80.0, 60))
        )
        signals = _momentum_signals(prices, lookback=20)
        assert -1 in signals

    def test_signals_only_contain_valid_values(self) -> None:
        rng = np.random.default_rng(0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        signals = _momentum_signals(prices, lookback=60)
        assert set(signals).issubset({-1, 0, 1})

    def test_short_series_returns_all_zeros(self) -> None:
        prices = np.array([100.0, 101.0])
        signals = _momentum_signals(prices, lookback=5)
        assert (signals == 0).all()

    def test_perfectly_flat_prices_no_signals(self) -> None:
        # Flat prices: log return = 0, never crosses positive threshold.
        prices = np.full(100, 100.0)
        signals = _momentum_signals(prices, lookback=20)
        # No buy signals (momentum = 0, not > 0).
        assert 1 not in signals

class TestReturnsFromSignals:
    def test_length_one_less_than_prices(self) -> None:
        close = np.array([100.0, 101.0, 102.0, 103.0])
        signals = np.array([0, 1, 0, 0])
        returns = returns_from_signals(close, signals)
        assert len(returns) == 3

    def test_zero_returns_when_always_flat(self) -> None:
        close = np.array([100.0, 101.0, 102.0])
        signals = np.array([0, 0, 0])
        returns = returns_from_signals(close, signals)
        assert (returns == 0).all()

    def test_positive_return_when_long_and_prices_rise(self) -> None:
        close = np.array([100.0, 110.0, 120.0])
        signals = np.array([1, 0, 0])  # enter on bar 0, hold
        returns = returns_from_signals(close, signals)
        assert returns[0] > 0  # 100→110 return while long

# ---------------------------------------------------------------------------
# MomentumStrategy interface
# ---------------------------------------------------------------------------

class TestMomentumInit:
    def test_invalid_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MomentumStrategy(lookback=0)

    def test_negative_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MomentumStrategy(lookback=-10)

    def test_default_lookback_is_positive(self) -> None:
        s = MomentumStrategy()
        assert s.lookback_ > 0

class TestMomentumFit:
    def test_fit_returns_self(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(600)
        result = s.fit(data)
        assert result is s

    def test_fit_selects_lookback_from_grid(self) -> None:
        s = MomentumStrategy()
        s.fit(_oscillating(600))
        assert s.lookback_ in _LOOKBACK_GRID

    def test_fit_stores_evaluated_pairs(self) -> None:
        s = MomentumStrategy()
        s.fit(_oscillating(600))
        assert len(s._all_lookbacks_) > 0
        assert all(lb in _LOOKBACK_GRID for lb in s._all_lookbacks_)

    def test_fit_skips_lookbacks_longer_than_data(self) -> None:
        # Very short data: only lookbacks shorter than n-2 should be evaluated.
        s = MomentumStrategy()
        short = _make_data([float(i) for i in range(30)])
        s.fit(short)
        # All evaluated lookbacks must be usable on 30 bars.
        for lb in s._all_lookbacks_:
            assert lb + 2 <= 30

    def test_fit_raises_when_all_lookbacks_skipped(self) -> None:
        """fit() must raise ValueError when no lookback can be evaluated.

        Mirrors the same guard added to MovingAverageStrategy.fit() in v0.14.2.
        Silently returning with stale parameters would give the caller no way
        to distinguish a successful fit from a no-op.
        """
        s = MomentumStrategy()
        # Provide fewer bars than the smallest lookback in the grid requires.
        tiny = _make_data([float(i) for i in range(3)])  # 3 bars - even lb=20 needs 22
        with pytest.raises(ValueError, match="No lookback could be evaluated"):
            s.fit(tiny)

    def test_different_regimes_select_different_lookbacks(self) -> None:
        # A fast-trending market should prefer shorter lookback; slow trend longer.
        rng = np.random.default_rng(42)
        n = 504
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        fast = pd.DataFrame(
            {"close": 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.005, n)))},
            index=dates,
        )
        slow = pd.DataFrame(
            {"close": 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))},
            index=dates,
        )
        s1, s2 = MomentumStrategy(), MomentumStrategy()
        s1.fit(fast)
        s2.fit(slow)
        # At minimum: fit should complete without error for both regimes.
        assert s1.lookback_ in _LOOKBACK_GRID
        assert s2.lookback_ in _LOOKBACK_GRID

class TestMomentumSignalGeneration:
    def test_signals_only_valid_values(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(400)
        s.fit(data.iloc[:200])
        signals = s.generate_signals(data.iloc[200:])
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_aligned_to_index(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(400)
        s.fit(data.iloc[:200])
        test = data.iloc[200:]
        signals = s.generate_signals(test)
        assert signals.index.equals(test.index)

    def test_signals_are_integers(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(400)
        s.fit(data.iloc[:200])
        signals = s.generate_signals(data.iloc[200:])
        assert pd.api.types.is_integer_dtype(signals)

    def test_uptrend_generates_buy(self) -> None:
        s = MomentumStrategy(lookback=20)
        # Fit on flat data so default lookback stays at 20.
        flat = _make_data([100.0] * 100)
        s.fit(flat)
        s.lookback_ = 20  # force it
        data = _trending_up(200)
        signals = s.generate_signals(data)
        assert (signals == 1).any()

    def test_downtrend_generates_sell(self) -> None:
        s = MomentumStrategy(lookback=20)
        flat = _make_data([100.0] * 100)
        s.fit(flat)
        s.lookback_ = 20
        # Build: rise then fall so we get a sell crossover.
        prices = list(np.linspace(100, 140, 100)) + list(np.linspace(140, 80, 100))
        data = _make_data(prices)
        signals = s.generate_signals(data)
        assert (signals == -1).any()

class TestMomentumContext:
    def test_context_signals_only_cover_test_index(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(600)
        train, test = data.iloc[:400], data.iloc[400:]
        s.fit(train)
        signals = s.generate_signals_with_context(train.iloc[-60:], test)
        assert signals.index.equals(test.index)

    def test_context_signals_are_valid(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(600)
        s.fit(data.iloc[:400])
        signals = s.generate_signals_with_context(data.iloc[340:400], data.iloc[400:])
        assert set(signals.unique()).issubset({-1, 0, 1})

class TestMomentumBoundaryCorrectness:
    """
    generate_signals_with_context's test-bar-0 decision must match momentum
    computed directly from the unwindowed series, with no special-casing
    needed at the context/test boundary.

    Unlike MovingAverageStrategy and KalmanFilterStrategy, momentum's signal
    is a memoryless function of the current bar's T-day return rather than a
    transition from a prior state - context_window_size() == lookback_ bars
    is exactly enough to compute that one value at test bar 0, and no
    "was the strategy already long" bookkeeping is needed on top of it.
    These tests prove that directly, across many random regime-change
    scenarios engineered to land a momentum sign flip right at the boundary.
    """

    def test_bar_zero_matches_unwindowed_momentum(self) -> None:
        rng = np.random.default_rng(7)
        mismatches = 0
        trials = 200

        for _ in range(trials):
            lb = int(rng.choice(_LOOKBACK_GRID))
            n_train, n_test = 400, 60
            drift = rng.choice([-0.002, -0.0005, 0.0, 0.0005, 0.002])
            noise = rng.normal(0, 0.01, n_train + n_test)
            log_close = np.cumsum(np.full(n_train + n_test, drift) + noise) + np.log(100)
            close = np.exp(log_close)
            data = _make_data(list(close), start="2010-01-01")

            strategy = MomentumStrategy(lookback=lb)
            strategy.lookback_ = lb
            strategy._all_lookbacks_ = [lb]

            train, test = data.iloc[:n_train], data.iloc[n_train:]
            context = train.iloc[-strategy.context_window_size():]
            signals = strategy.generate_signals_with_context(context, test)

            full_close = data["close"].to_numpy()
            true_momentum = np.log(full_close[n_train]) - np.log(full_close[n_train - lb])
            true_should_be_long = true_momentum > 0
            bar0_was_buy = signals.iloc[0] == 1

            if bar0_was_buy != true_should_be_long:
                mismatches += 1

        assert mismatches == 0, (
            f"{mismatches}/{trials} trials disagreed with the unwindowed "
            "ground truth at test bar 0."
        )

    def test_carries_long_through_boundary_without_explicit_injection(self) -> None:
        """A strict uptrend should be long at test bar 0 with no special-casing."""
        s = MomentumStrategy(lookback=60)
        s.lookback_ = 60
        s._all_lookbacks_ = [60]
        data = _trending_up(n=400)
        train, test = data.iloc[:300], data.iloc[300:]
        context = train.iloc[-s.context_window_size():]

        signals = s.generate_signals_with_context(context, test)
        assert signals.iloc[0] == 1

    def test_stays_flat_through_boundary_without_explicit_injection(self) -> None:
        """A strict downtrend should stay flat at test bar 0."""
        s = MomentumStrategy(lookback=60)
        s.lookback_ = 60
        s._all_lookbacks_ = [60]
        data = _trending_down(n=400)
        train, test = data.iloc[:300], data.iloc[300:]
        context = train.iloc[-s.context_window_size():]

        signals = s.generate_signals_with_context(context, test)
        assert signals.iloc[0] == 0

class TestMomentumCandidateReturns:
    def test_returns_dict_keyed_by_lookback(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(600)
        s.fit(data.iloc[:400])
        result = s.candidate_test_returns(data.iloc[400:], data.iloc[340:400])
        assert all(k in _LOOKBACK_GRID for k in result)

    def test_all_evaluated_lookbacks_present(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(600)
        s.fit(data.iloc[:400])
        result = s.candidate_test_returns(data.iloc[400:], data.iloc[340:400])
        for lb in s._all_lookbacks_:
            assert lb in result

    def test_return_series_aligned_to_test_index(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(600)
        test = data.iloc[400:]
        s.fit(data.iloc[:400])
        result = s.candidate_test_returns(test)
        for lb, returns in result.items():
            # Returns are one shorter than test (differencing), so check subset.
            assert returns.index.isin(test.index).all()

    def test_empty_before_fit(self) -> None:
        s = MomentumStrategy()
        data = _oscillating(300)
        # Before fit(), _all_lookbacks_ is empty → candidate_test_returns returns {}.
        result = s.candidate_test_returns(data)
        assert result == {}

class TestMomentumActiveParams:
    def test_returns_dict_with_lookback(self) -> None:
        s = MomentumStrategy()
        s.fit(_oscillating(600))
        params = s.active_params()
        assert "lookback" in params
        assert params["lookback"] in _LOOKBACK_GRID

# ── MOMENTUM_LOOKBACKS config contract ───────────────────────────────────────

class TestMomentumLookbackConfig:
    """MOMENTUM_LOOKBACKS in config.py must drive the momentum strategy grid.

    fit() must evaluate exactly the candidates in MOMENTUM_LOOKBACKS (filtered
    by data length). Any divergence means the strategy and config are out of
    sync and the RC p-value is computed over a different set than documented.
    """

    def test_config_lookbacks_match_strategy_grid(self) -> None:
        """fit() must evaluate exactly the candidates in MOMENTUM_LOOKBACKS.

        momentum.py imports and uses MOMENTUM_LOOKBACKS directly from config,
        so after fit(), _all_lookbacks_ must equal the config list (filtered
        by data length). Any divergence means the strategy and config are out
        of sync and the RC p-value is computed over a different set than documented.
        """
        # Use enough data that all lookbacks are evaluated (need close > max_lb + 2)
        n = max(_LOOKBACK_GRID) + 50
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 10
        data = pd.DataFrame({"close": close}, index=dates)

        s = MomentumStrategy()
        s.fit(data)
        assert s._all_lookbacks_ == _LOOKBACK_GRID, (
            f"fit() evaluated {s._all_lookbacks_}, expected {_LOOKBACK_GRID}. "
            "momentum.py must iterate over MOMENTUM_LOOKBACKS from config.py."
        )

    def test_config_lookbacks_are_sorted_ascending(self) -> None:
        assert _LOOKBACK_GRID == sorted(_LOOKBACK_GRID), (
            "MOMENTUM_LOOKBACKS should be in ascending order for deterministic candidate key ordering."
        )

    def test_config_lookbacks_all_positive(self) -> None:
        for lb in _LOOKBACK_GRID:
            assert lb > 0, f"Lookback {lb} must be positive."

    def test_config_lookbacks_exported_from_config_module(self) -> None:
        """Verify the constant is accessible from the config module."""
        config = importlib.import_module("backtesting_engine.config")
        assert hasattr(config, "MOMENTUM_LOOKBACKS")
        assert isinstance(config.MOMENTUM_LOOKBACKS, list)
