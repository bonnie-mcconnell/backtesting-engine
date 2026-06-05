"""
Performance metrics for backtested strategies.

All ratio metrics are annualised using ANNUALISATION_FACTOR (252 trading days).
The risk-free rate is subtracted before computing excess-return ratios; it
defaults to zero in config but can be changed without touching this module.

Public interface:
    calculate_metrics(portfolio_values) -> MetricsResult

Private helpers (_sharpe, _sortino, etc.) are importable for unit tests
and for use by the strategy grid-search in moving_average.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    BLOCK_BOOTSTRAP_SEED,
    N_PERMUTATIONS,
    RISK_FREE_RATE,
)
from backtesting_engine.models import MetricsResult, Trade


def calculate_metrics(
    portfolio_values: pd.Series,
    trades: Sequence[Trade] | None = None,
    seed: int = BLOCK_BOOTSTRAP_SEED,
) -> MetricsResult:
    """
    Compute all performance metrics from a daily portfolio value series.

    Args:
        portfolio_values: Daily mark-to-market portfolio values. NaN rows
                          are dropped before any calculation.
        trades: Optional Trade list from the same window. When provided,
                trade-level diagnostics (exposure, win rate, etc.) are included.

    Returns:
        MetricsResult with Sharpe, Sortino, max drawdown, Calmar, Omega,
        a block-bootstrap p-value, and trade diagnostics if trades provided.

    Raises:
        ValueError: If the returns series is empty after dropping NaNs.
    """
    clean = portfolio_values.dropna()
    returns = clean.pct_change().dropna()
    returns_array = returns.to_numpy(dtype=float)

    if len(returns_array) == 0:
        raise ValueError(
            "No returns to compute metrics. "
            "Check that portfolio_values has at least two non-NaN entries."
        )

    exposure, trade_count, win_rate, avg_wl, avg_hold = _trade_diagnostics(
        portfolio_values, trades
    )

    return MetricsResult(
        sharpe_ratio=_sharpe(returns_array),
        sortino_ratio=_sortino(returns_array),
        max_drawdown=_max_drawdown(returns_array),
        calmar_ratio=_calmar(returns_array),
        omega_ratio=_omega(returns_array),
        p_value=_monte_carlo_p_value(returns_array, seed=seed),
        exposure_fraction=exposure,
        trade_count=trade_count,
        win_rate=win_rate,
        avg_win_loss_ratio=avg_wl,
        avg_holding_days=avg_hold,
    )


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def _sharpe(returns_array: np.ndarray) -> float:
    """Annualised Sharpe ratio. Returns 0.0 for a flat return series."""
    excess = returns_array - RISK_FREE_RATE
    std = excess.std(ddof=1)
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(ANNUALISATION_FACTOR))


def _sortino(returns_array: np.ndarray) -> float:
    """
    Annualised Sortino ratio using downside deviation, not std of downside returns.

    Downside deviation per Sortino & van der Meer (1991):
        downside_dev = sqrt( mean( min(r_t - T, 0)^2 ) )

    This differs from std(downside_returns): std measures dispersion among
    negative returns, while downside deviation measures magnitude relative to zero.
    For a strategy with small consistent losses, std → 0 (inflated Sortino) while
    downside deviation correctly stays large.

    Returns float('inf') if there are no below-threshold returns.
    """
    excess = returns_array - RISK_FREE_RATE
    negative_excess = np.minimum(excess, 0.0)

    if np.all(negative_excess == 0.0):
        return float("inf")

    downside_dev = float(np.sqrt(np.mean(negative_excess ** 2)))
    if downside_dev < 1e-10:
        return float("inf")

    return float(excess.mean() / downside_dev * np.sqrt(ANNUALISATION_FACTOR))


def _max_drawdown(returns_array: np.ndarray) -> float:
    """
    Maximum drawdown as a fraction (e.g. -0.20 = -20%).

    Computed on cumulative product of (1+r_t). The initial 1.0 is prepended
    so a loss on bar 0 is correctly counted - without it the rolling max starts
    at the post-loss value and the first bar's drawdown disappears.
    """
    cumulative = np.concatenate([[1.0], np.cumprod(1.0 + returns_array)])
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown[1:].min())


def _calmar(returns_array: np.ndarray) -> float:
    """
    Annualised Calmar ratio using geometric (not arithmetic) compounding.

    prod(1+r)^(252/n) - 1 is the exact annualised return. The arithmetic
    version (1 + mean_daily)^252 overstates it by several percentage points
    at typical daily volatility due to Jensen's inequality.

    Returns float('inf') if max drawdown is zero.
    """
    n = len(returns_array)
    if n == 0:
        return 0.0
    cumulative = float(np.prod(1.0 + np.clip(returns_array, -0.9999, None)))
    annualised_return = cumulative ** (ANNUALISATION_FACTOR / n) - 1.0
    max_dd = abs(_max_drawdown(returns_array))
    if max_dd < 1e-10:
        return float("inf")
    return float(annualised_return / max_dd)


def _omega(returns_array: np.ndarray, threshold: float = RISK_FREE_RATE) -> float:
    """
    Omega ratio: probability-weighted gains above threshold / losses below.

    Unlike Sharpe, uses the full return distribution rather than just mean
    and variance - captures skewness and kurtosis naturally.
    Returns float('inf') if there are no returns below the threshold.
    """
    gains = (returns_array[returns_array > threshold] - threshold).sum()
    losses = (threshold - returns_array[returns_array < threshold]).sum()
    if losses < 1e-10:
        return float("inf")
    return float(gains / losses)


def _monte_carlo_p_value(
    returns_array: np.ndarray,
    seed: int = BLOCK_BOOTSTRAP_SEED,
) -> float:
    """
    Block-bootstrap p-value for the observed Sharpe ratio.

    Tests H₀ that the observed Sharpe arose by chance given the return
    structure of this window.

    Why block bootstrap: simple shuffling leaves mean and std unchanged
    (Sharpe is order-invariant), so every permutation produces the identical
    Sharpe. Block resampling preserves local autocorrelation while randomising
    the global sequence. See Politis & Romano (1994).

    Why circular blocks: blocks near the array end would be shorter than
    block_size under standard bootstrap, underrepresenting tail behaviour.
    Circular bootstrap wraps around the end so all positions are equally
    represented.

    Returns are centred (mean subtracted) before resampling. Without this,
    the bootstrap inherits the observed mean and p ≈ 0.5 for any
    positive-drift strategy regardless of signal quality.
    """
    observed_sharpe = _sharpe(returns_array)
    rng = np.random.default_rng(seed=seed)
    n = len(returns_array)
    block_size = max(1, int(np.sqrt(n)))

    centered = returns_array - returns_array.mean()
    circular = np.tile(centered, 2)

    bootstrapped_sharpes = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        starts = rng.integers(0, n, size=(n // block_size) + 1)
        indices = np.concatenate([
            np.arange(start, start + block_size) for start in starts
        ])
        shuffled = circular[indices[:n]]
        bootstrapped_sharpes[i] = _sharpe(shuffled)

    return float(np.mean(bootstrapped_sharpes >= observed_sharpe))


def _trade_diagnostics(
    portfolio_values: pd.Series,
    trades: Sequence[Trade] | None,
) -> tuple[float, int, float, float, float]:
    """
    Compute trade-level diagnostics: exposure, count, win rate, avg W/L, avg hold.

    Returns:
        Tuple of (exposure_fraction, trade_count, win_rate, avg_win_loss_ratio,
        avg_holding_days). NaN for any metric that can't be computed.
    """
    nan = float("nan")
    n_bars = len(portfolio_values)

    if trades is None:
        exposure = nan
    elif len(trades) > 0:
        # exact exposure from trade dates - the heuristic of checking
        # abs(portfolio_change) > threshold misclassifies quiet in-market days
        pv_index = portfolio_values.index
        in_market: set[object] = set()
        for t in trades:
            mask = (pv_index >= t.entry_date) & (pv_index <= t.exit_date)
            in_market.update(pv_index[mask].tolist())
        exposure = len(in_market) / n_bars if n_bars > 0 else nan
    else:
        exposure = 0.0

    if not trades:
        return exposure, 0, nan, nan, nan

    trade_count = len(trades)
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / trade_count) if trade_count > 0 else nan

    if len(wins) > 0 and len(losses) > 0:
        avg_wl = float(wins.mean() / abs(losses.mean()))
    elif len(wins) > 0:
        avg_wl = float("inf")
    else:
        avg_wl = 0.0

    hold_days = [(t.exit_date - t.entry_date).days for t in trades]
    avg_hold = float(np.mean(hold_days)) if hold_days else nan

    return exposure, trade_count, win_rate, avg_wl, avg_hold
