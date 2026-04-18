"""
Unit tests for metrics module.
All expected values are computed by hand and hard-coded - never derived from the
implementation itself, which would make the test circular and unable to catch bugs.
"""
import numpy as np

from backtesting_engine.metrics import _sharpe, _sortino, _max_drawdown, _calmar, _omega


# --- Sharpe ratio ---

def test_sharpe_zero_mean_returns_zero() -> None:
    # mean([0.1, -0.1]) = 0.0, so Sharpe = 0.0 regardless of std
    returns = np.array([0.1, -0.1])
    assert _sharpe(returns) == 0.0


def test_sharpe_zero_std_returns_zero() -> None:
    # constant returns have near-zero std due to floating point - guard returns 0.0
    returns = np.array([0.05, 0.05, 0.05])
    assert _sharpe(returns) == 0.0


def test_sharpe_known_value() -> None:
    # returns = [0.02, 0.00, 0.02, 0.00]
    # mean = 0.01
    # std(ddof=1) = sqrt(sum((x - 0.01)^2) / 3) = sqrt(0.0004 / 3 * 4 / 4)
    # computed precisely: std = 0.011547005383792515
    # expected = 0.01 / 0.011547005383792515 * sqrt(252) = 13.744...
    returns = np.array([0.02, 0.00, 0.02, 0.00])
    expected = 0.01 / 0.011547005383792515 * np.sqrt(252)
    assert np.isclose(_sharpe(returns), expected, rtol=1e-5)


# --- Sortino ratio ---

def test_sortino_no_downside_returns_inf() -> None:
    # no negative returns - downside std is undefined, return inf
    returns = np.array([0.01, 0.02, 0.03])
    assert _sortino(returns) == float('inf')


def test_sortino_known_value() -> None:
    # returns = [0.04, -0.02]
    # mean = 0.01
    # downside returns = [-0.02], std(ddof=1) of single value = nan? No - ddof=1 with n=1 gives nan
    # use two downside values: returns = [0.05, -0.01, -0.03]
    # mean = 0.00333...
    # downside = [-0.01, -0.03], std(ddof=1) = 0.014142
    # sortino = 0.00333 / 0.014142 * sqrt(252) = 3.742...
    returns = np.array([0.05, -0.01, -0.03])
    mean = returns.mean()                              # 0.003333...
    downside_std = np.std([-0.01, -0.03], ddof=1)     # 0.014142...
    expected = mean / downside_std * np.sqrt(252)
    assert np.isclose(_sortino(returns), expected, rtol=1e-5)


# --- Maximum drawdown ---

def test_max_drawdown_known_value() -> None:
    # returns = [0.1, -0.2, 0.1]
    # cumulative = [1.1, 0.88, 0.968]
    # rolling_max = [1.1, 1.1, 1.1]
    # drawdown = [0, (0.88-1.1)/1.1, (0.968-1.1)/1.1] = [0, -0.2, -0.12]
    # max_drawdown = -0.2
    returns = np.array([0.1, -0.2, 0.1])
    assert np.isclose(_max_drawdown(returns), -0.2, rtol=1e-5)


def test_max_drawdown_no_drawdown() -> None:
    # monotonically increasing - never below peak, drawdown = 0
    returns = np.array([0.01, 0.02, 0.03])
    assert _max_drawdown(returns) == 0.0


def test_max_drawdown_always_non_positive() -> None:
    # max drawdown is always <= 0 by definition
    returns = np.array([0.01, 0.02, -0.05, 0.01])
    assert _max_drawdown(returns) <= 0.0


# --- Calmar ratio ---

def test_calmar_no_drawdown_returns_inf() -> None:
    # no drawdown - Calmar undefined, return inf
    returns = np.array([0.01, 0.02, 0.03])
    assert _calmar(returns) == float('inf')


def test_calmar_known_value() -> None:
    # returns = [0.1, -0.2, 0.1]
    # max_drawdown = -0.2 (from test above)
    # mean = 0.0, annualised = (1 + 0.0)^252 - 1 = 0.0
    # calmar = 0.0 / 0.2 = 0.0
    returns = np.array([0.1, -0.2, 0.1])
    assert np.isclose(_calmar(returns), 0.0, atol=1e-5)


# --- Omega ratio ---

def test_omega_no_losses_returns_inf() -> None:
    # no losses - Omega undefined, return inf
    returns = np.array([0.01, 0.02, 0.03])
    assert _omega(returns) == float('inf')


def test_omega_known_value() -> None:
    # returns = [0.03, 0.01, -0.01, -0.01]
    # gains above 0 = [0.03, 0.01], sum = 0.04
    # losses below 0 = [0.01, 0.01], sum = 0.02
    # omega = 0.04 / 0.02 = 2.0
    returns = np.array([0.03, 0.01, -0.01, -0.01])
    assert np.isclose(_omega(returns), 2.0, rtol=1e-5)