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
    

# Permutes returns to build null distribution of Sharpe ratios under the hypothesis of no edge.
# p-value is the fraction of random strategies that match or exceed the observed Sharpe.
def _monte_carlo_p_value(returns_array: np.ndarray) -> float:
    observed_sharpe = _sharpe(returns_array)
    random_sharpes = []
    rng = np.random.default_rng(seed=42)  # reproducible results
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(returns_array)
        random_sharpes.append(_sharpe(shuffled))
    return float(np.mean(np.array(random_sharpes) >= observed_sharpe))


def _sharpe(returns_array: np.ndarray) -> float: 
    return returns_array.mean() / returns_array.std(ddof=1) * (np.sqrt(ANNUALISATION_FACTOR))


def _sortino(returns_array: np.ndarray) -> float:
    downside_returns = returns_array[returns_array < 0]
    if len(downside_returns) == 0:
        return float('inf')  # No downside volatility, perfect score
    return returns_array.mean() / downside_returns.std(ddof=1) * (np.sqrt(ANNUALISATION_FACTOR))


def _max_drawdown(returns: np.ndarray) -> float:
    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def _calmar(returns_array: np.ndarray) -> float:
    annualised = (1 + float(np.mean(returns_array))) ** ANNUALISATION_FACTOR - 1
    max_dd = abs(_max_drawdown(returns_array))
    if max_dd == 0.0:
        return float('inf')
    return annualised / max_dd


def _omega(returns_array: np.ndarray) -> float:
    threshold = 0.0
    gains = returns_array[returns_array > threshold] - threshold
    losses = threshold - returns_array[returns_array < threshold]
    return gains.sum() / losses.sum() if losses.sum() > 0 else float('inf')