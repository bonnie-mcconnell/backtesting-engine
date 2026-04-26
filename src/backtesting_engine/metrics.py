"""
Performance metrics for backtested strategies.

All ratio metrics are annualised using ANNUALISATION_FACTOR (252 trading days).
The risk-free rate is subtracted before computing excess-return ratios; it
defaults to zero in config but can be changed without touching this module.

Public interface:
    calculate_metrics(portfolio_values) -> MetricsResult

All private helpers (_sharpe, _sortino, etc.) are also importable for unit
testing and for use by the strategy grid-search in moving_average.py.
"""

import numpy as np
import pandas as pd

from backtesting_engine.config import (
    ANNUALISATION_FACTOR,
    BLOCK_BOOTSTRAP_SEED,
    N_PERMUTATIONS,
    RISK_FREE_RATE,
)
from backtesting_engine.models import MetricsResult


def calculate_metrics(portfolio_values: pd.Series) -> MetricsResult:
    """
    Compute all performance metrics from a daily portfolio value series.

    Args:
        portfolio_values: Daily mark-to-market portfolio values. NaN rows
                          are dropped before any calculation.

    Returns:
        MetricsResult containing Sharpe, Sortino, max drawdown, Calmar,
        Omega, and a block-bootstrap p-value.

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

    return MetricsResult(
        sharpe_ratio=_sharpe(returns_array),
        sortino_ratio=_sortino(returns_array),
        max_drawdown=_max_drawdown(returns_array),
        calmar_ratio=_calmar(returns_array),
        omega_ratio=_omega(returns_array),
        p_value=_monte_carlo_p_value(returns_array),
    )


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def _sharpe(returns_array: np.ndarray) -> float:
    """
    Annualised Sharpe ratio: mean excess return divided by return volatility.

    Excess return = raw return minus daily risk-free rate (RISK_FREE_RATE).
    Returns 0.0 if standard deviation is negligibly small (flat return series).

    Args:
        returns_array: Daily returns as a NumPy array.

    Returns:
        Annualised Sharpe ratio.
    """
    excess = returns_array - RISK_FREE_RATE
    std = excess.std(ddof=1)
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(ANNUALISATION_FACTOR))


def _sortino(returns_array: np.ndarray) -> float:
    """
    Annualised Sortino ratio: mean excess return divided by downside deviation.

    Downside deviation is the square root of the mean squared negative excess
    return - the RMS of returns that fall below the threshold. This is the
    standard definition from Sortino & van der Meer (1991):

        downside_deviation = sqrt( mean( min(r_t - T, 0)^2 ) )

    where T is the threshold (here: RISK_FREE_RATE).

    This differs from using std(downside_returns): std measures dispersion
    *among* negative returns (around their mean), while downside deviation
    measures the magnitude of negative returns *relative to zero*. For a
    strategy with small, consistent losses, std approaches zero (giving an
    inflated Sortino) while downside deviation correctly stays large.

    Returns float('inf') if there are no below-threshold returns.

    Args:
        returns_array: Daily returns as a NumPy array.

    Returns:
        Annualised Sortino ratio.
    """
    excess = returns_array - RISK_FREE_RATE

    # All returns below the threshold - clip to their negative part.
    negative_excess = np.minimum(excess, 0.0)

    if np.all(negative_excess == 0.0):
        # No below-threshold returns - downside deviation is zero, Sortino = inf.
        return float("inf")

    # Downside deviation: RMS of negative excess returns.
    # Uses mean (not ddof=1) because we want E[min(r-T,0)^2], not a sample correction.
    downside_dev = float(np.sqrt(np.mean(negative_excess ** 2)))

    if downside_dev < 1e-10:
        return float("inf")

    return float(excess.mean() / downside_dev * np.sqrt(ANNUALISATION_FACTOR))


def _max_drawdown(returns_array: np.ndarray) -> float:
    """
    Maximum drawdown: the largest peak-to-trough decline in cumulative returns.

    Computed on the cumulative product of (1 + r_t), so it correctly handles
    compounding. The result is expressed as a fraction (e.g. -0.20 = -20%).

    The initial value 1.0 is prepended before computing the running maximum.
    Without it, a loss on the very first bar would appear as no drawdown because
    the rolling max would start at the post-loss value rather than the pre-loss
    value. Prepending 1.0 anchors the peak at the start of the period.

    Args:
        returns_array: Daily returns as a NumPy array.

    Returns:
        Maximum drawdown as a non-positive fraction.
    """
    # Prepend 1.0 to represent the initial (pre-return) portfolio value.
    cumulative = np.concatenate([[1.0], np.cumprod(1.0 + returns_array)])
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    # Slice off the prepended 1.0 row (always 0.0) before taking min.
    return float(drawdown[1:].min())


def _calmar(returns_array: np.ndarray) -> float:
    """
    Annualised Calmar ratio: annualised return divided by absolute max drawdown.

    Annualised return uses the geometric (compound) formula:
        annualised = prod(1 + r_t)^(252/n) - 1

    This is the exact annualised return from the actual return path.
    The alternative formula (1 + mean_daily)^252 compounds the arithmetic
    mean, which overstates annualised return by up to several percentage
    points when daily volatility is high (Jensen's inequality).

    Returns float('inf') if max drawdown is zero (no peak-to-trough decline).

    Args:
        returns_array: Daily returns as a NumPy array.

    Returns:
        Annualised Calmar ratio.
    """
    n = len(returns_array)
    if n == 0:
        return 0.0
    # Geometric annualised return: compound all daily returns, then annualise.
    # Clip returns to prevent log(0) on -100% days.
    cumulative = float(np.prod(1.0 + np.clip(returns_array, -0.9999, None)))
    annualised_return = cumulative ** (ANNUALISATION_FACTOR / n) - 1.0
    max_dd = abs(_max_drawdown(returns_array))
    if max_dd < 1e-10:
        return float("inf")
    return float(annualised_return / max_dd)


def _omega(returns_array: np.ndarray, threshold: float = RISK_FREE_RATE) -> float:
    """
    Omega ratio: probability-weighted gains above threshold divided by losses below.

    Unlike Sharpe, Omega uses the full return distribution rather than just
    mean and variance, so it captures skewness and kurtosis naturally.
    Threshold defaults to the risk-free rate for consistency with other metrics.

    Returns float('inf') if there are no returns below the threshold.

    Args:
        returns_array: Daily returns as a NumPy array.
        threshold: Minimum acceptable return (default: RISK_FREE_RATE).

    Returns:
        Omega ratio (always positive; >1 means more gain than loss).
    """
    gains = (returns_array[returns_array > threshold] - threshold).sum()
    losses = (threshold - returns_array[returns_array < threshold]).sum()
    if losses < 1e-10:
        return float("inf")
    return float(gains / losses)


def _monte_carlo_p_value(returns_array: np.ndarray) -> float:
    """
    Block-bootstrap p-value for the observed Sharpe ratio.

    Tests the null hypothesis that the observed Sharpe arose by chance given
    the return structure of this data. Builds a null distribution by resampling
    consecutive blocks of returns N_PERMUTATIONS times and computing the Sharpe
    of each resampled series.

    Why block bootstrap, not simple shuffling:
        The Sharpe ratio is order-invariant - shuffling individual returns leaves
        mean and std unchanged, so every permutation produces the identical Sharpe.
        Block bootstrapping preserves local autocorrelation structure (momentum,
        mean-reversion) while randomising the global sequence, producing a genuine
        null distribution. See Politis & Romano (1994).

    Why circular blocks:
        Standard block bootstrap draws blocks starting at random positions. Blocks
        near the end of the series are shorter (clipped at array boundary), which
        systematically underrepresents tail behaviour. Circular bootstrap wraps
        around the end of the array, ensuring every block has the full block_size
        and all positions are equally represented.

    Block size:
        int(sqrt(n)) follows Politis & Romano (1994) - large enough to preserve
        autocorrelation, small enough to provide meaningful randomisation.

    The reported p-value is the fraction of bootstrapped Sharpes that meet or
    exceed the observed Sharpe. A low p-value means the strategy's performance
    is unlikely to have arisen by chance.

    Args:
        returns_array: Daily returns as a NumPy array.

    Returns:
        p-value in [0, 1].
    """
    observed_sharpe = _sharpe(returns_array)
    rng = np.random.default_rng(seed=BLOCK_BOOTSTRAP_SEED)
    n = len(returns_array)
    block_size = max(1, int(np.sqrt(n)))

    # Tile array twice so circular indexing never goes out of bounds.
    circular = np.tile(returns_array, 2)

    bootstrapped_sharpes = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        # Sample random block start positions; modulo n keeps them in [0, n).
        starts = rng.integers(0, n, size=(n // block_size) + 1)
        indices = np.concatenate([
            np.arange(start, start + block_size) for start in starts
        ])
        shuffled = circular[indices[:n]]
        bootstrapped_sharpes[i] = _sharpe(shuffled)

    return float(np.mean(bootstrapped_sharpes >= observed_sharpe))
