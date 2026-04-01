"""
Calculates performance metrics for the backtest.
Metrics include Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, Omega ratio, and p-value from a Monte Carlo permutation test.
"""
import pandas as pd
import numpy as np

from backtesting_engine.models import MetricsResult
from backtesting_engine.config import ANNUALISATION_FACTOR, N_PERMUTATIONS


def calculate_metrics(portfolio_values: pd.Series) -> MetricsResult:
    """
    Calculates performance metrics for the backtest, including Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, Omega ratio, and p-value from a Monte Carlo permutation test.

    Args:
        portfolio_values (pd.Series): The daily portfolio values over the backtest period.

    Returns:
        MetricsResult containing all calculated performance metrics.
    """
    portfolio_values = portfolio_values.dropna()
    returns = portfolio_values.pct_change().dropna()
    returns_array = returns.to_numpy()
    if len(returns_array) == 0:
        raise ValueError("No returns to calculate metrics. Check if portfolio values are correctly computed.")

    return MetricsResult(
        sharpe_ratio=_sharpe(returns_array), 
        sortino_ratio=_sortino(returns_array),
        max_drawdown=_max_drawdown(returns_array),
        calmar_ratio=_calmar(returns_array),
        omega_ratio=_omega(returns_array),
        p_value=_monte_carlo_p_value(returns_array),
    )
    

def _monte_carlo_p_value(returns_array: np.ndarray) -> float:
    """
    Estimate the p-value of the observed Sharpe ratio using block bootstrapping.

    Simple return shuffling is invalid here because the Sharpe ratio is order-invariant —
    shuffling an array leaves its mean and std unchanged, producing identical Sharpe ratios
    across all permutations. Block bootstrapping instead samples consecutive blocks of
    returns, preserving local autocorrelation structure while randomising the global
    sequence. This produces a genuine null distribution of Sharpe ratios.

    Block size of sqrt(n) follows Politis & Romano (1994) — large enough to preserve
    autocorrelation, small enough to provide meaningful randomisation.

    The p-value is the fraction of bootstrapped Sharpe ratios that meet or exceed the
    observed Sharpe. A low p-value indicates the strategy's performance is unlikely to
    have arisen by chance given the return structure of the data.

    Args:
        returns_array: Daily returns as a NumPy array.

    Returns:
        p-value between 0 and 1.
    """
    observed_sharpe = _sharpe(returns_array)
    rng = np.random.default_rng(seed=42)
    n = len(returns_array)
    block_size = int(np.sqrt(n))  # standard block size choice
    random_sharpes = []
    
    for _ in range(N_PERMUTATIONS):
        # build a shuffled array by randomly sampling blocks
        indices = []
        while len(indices) < n:
            start = rng.integers(0, n)
            block = list(range(start, min(start + block_size, n)))
            indices.extend(block)
        shuffled = returns_array[indices[:n]]
        random_sharpes.append(_sharpe(shuffled))
    
    return float(np.mean(np.array(random_sharpes) >= observed_sharpe))


def _sharpe(returns_array: np.ndarray) -> float:
    """
    Annualised Sharpe ratio: mean return divided by return volatility.

    Returns 0.0 if standard deviation is near zero (flat returns series).

    Args:
        returns_array: Daily returns as a NumPy array.

    Returns:
        Annualised Sharpe ratio.
    """
    std = returns_array.std(ddof=1)
    if std < 1e-10:  # guard against near-zero std from floating point noise
        return 0.0
    return float(returns_array.mean() / std * np.sqrt(ANNUALISATION_FACTOR))


def _sortino(returns_array: np.ndarray) -> float:
    """
    Annualised Sortino ratio: mean return divided by downside volatility.

    Returns float('inf') if downside volatility is near zero (no downside risk).

    Args:
        returns_array: Daily returns as a NumPy array.

    Returns:
        Annualised Sortino ratio.
    """
    downside_returns = returns_array[returns_array < 0]
    if len(downside_returns) == 0:
        return float('inf')
    std = downside_returns.std(ddof=1)
    if std < 1e-10:  # guard against near-zero std from floating point noise
        return float('inf')
    return float(returns_array.mean() / std * np.sqrt(ANNUALISATION_FACTOR))

def _max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum drawdown: the largest peak-to-trough decline in cumulative returns.

    Args:
        returns: Daily returns as a NumPy array.

    Returns:
        Maximum drawdown as a percentage.
    """
    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def _calmar(returns_array: np.ndarray) -> float:
    """
    Annualised Calmar ratio: mean return divided by maximum drawdown.

    Returns float('inf') if maximum drawdown is zero (no risk).

    Args:
        returns_array: Daily returns as a NumPy array.

    Returns:
        Annualised Calmar ratio.

    """
    annualised = (1 + float(np.mean(returns_array))) ** ANNUALISATION_FACTOR - 1
    max_dd = abs(_max_drawdown(returns_array))
    if max_dd < 1e-10:
        return float('inf')
    return annualised / max_dd


def _omega(returns_array: np.ndarray) -> float:
    """
    Omega ratio: sum of gains above threshold divided by sum of losses below threshold.
    
    Returns float('inf') if there are no losses below the threshold.
    
    Args:
        returns_array: Daily returns as a NumPy array.
    
    Returns:
        Omega ratio.
    """
    threshold = 0.0
    gains = returns_array[returns_array > threshold] - threshold
    losses = threshold - returns_array[returns_array < threshold]
    return gains.sum() / losses.sum() if losses.sum() > 0 else float('inf')