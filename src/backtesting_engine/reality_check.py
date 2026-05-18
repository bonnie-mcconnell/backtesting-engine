"""
White's Reality Check for data snooping bias correction.

Reference: White, H. (2000). "A Reality Check for Data Snooping."
           Econometrica, 68(5), 1097–1126.

When you search over a universe of strategies and report the best performer,
the naive p-value on that result is misleading. Even if no strategy has genuine
edge, the best of k strategies will appear significant by chance. White's RC
corrects for this. The null hypothesis is:

    H₀: No strategy in the universe has performance superior to a benchmark.

White (2000) uses zero return (cash) as the benchmark. A small RC p-value means
at least one strategy beats cash, not necessarily buy-and-hold. For trend-following
strategies frequently in cash, this is the natural null. To test against buy-and-hold,
subtract B&H returns from each candidate's series before calling this function.
See benchmark.py for the buy-and-hold comparison.

The test statistic is the maximum mean excess return across all k strategies.
The p-value is the fraction of bootstrap replications in which the centred
maximum exceeds the observed maximum.

We use the stationary bootstrap (Politis & Romano, 1994): block lengths drawn
geometrically with mean b = sqrt(T), same seed as the rest of the engine.
During walk-forward we store TEST-period returns for every grid candidate, not
just the winner. After walk-forward completes, RC runs over the full candidate
universe across all windows to give the data-snooping corrected p-value.

  rc_p_value < 0.05: statistically significant after data-snooping correction.
  rc_p_value >= 0.05: performance is consistent with having searched many candidates.

The corrected p-value is always >= the naive p-value. If the naive test rejects
but RC does not, the apparent significance came from searching, not from edge.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backtesting_engine.config import BLOCK_BOOTSTRAP_SEED, N_PERMUTATIONS


def white_reality_check(
    candidate_returns: np.ndarray,
    n_bootstrap: int | None = None,
    seed: int = BLOCK_BOOTSTRAP_SEED,
) -> float:
    """
    White's Reality Check p-value for a universe of candidate strategies.

    Args:
        candidate_returns: Array of shape (T, k) where T is the number of
            time periods (days) and k is the number of candidate strategies.
            Each column is the daily return series of one candidate.
            The first column is assumed to be the best (selected) strategy.
        n_bootstrap: Number of stationary bootstrap replications. Defaults to
            N_PERMUTATIONS (read at call time, not at import time). Pass an
            explicit value to override. The test suite patches the module-level
            N_PERMUTATIONS to 200 via conftest.py; using None as the default
            ensures that patch takes effect.
        seed: Random seed for reproducibility.

    Returns:
        Reality Check p-value in [0, 1]. Small values indicate the best
        strategy's performance is unlikely under the joint null.

    Raises:
        ValueError: If candidate_returns has fewer than 2 dimensions or
                    fewer than 1 strategy.
    """
    # Read N_PERMUTATIONS at call time so the test suite can patch it via
    # conftest.py. If n_bootstrap=N_PERMUTATIONS were a default argument,
    # Python would capture the value at function-definition time (import),
    # before conftest.py's session fixture runs.
    n_iters = n_bootstrap if n_bootstrap is not None else N_PERMUTATIONS
    if candidate_returns.ndim != 2:
        raise ValueError(
            f"candidate_returns must be 2D (T × k), got shape {candidate_returns.shape}."
        )
    n_periods, k = candidate_returns.shape
    if k < 1:
        raise ValueError("Need at least one candidate strategy.")
    if n_periods < 2:
        raise ValueError("Need at least 2 time periods to bootstrap.")

    # Mean excess returns for each candidate over the observed period.
    # Benchmark = cash (zero return), so excess return is just the raw return.
    mean_returns = candidate_returns.mean(axis=0)   # shape (k,)

    # Observed test statistic: maximum mean return across all candidates.
    # This is what White (2000) calls V̄_k (equation 3.2).
    observed_max = float(mean_returns.max())

    # Stationary bootstrap: variable block lengths drawn from Geometric(1/b).
    b = max(1, int(np.sqrt(n_periods)))
    rng = np.random.default_rng(seed)
    p = 1.0 / b   # geometric parameter → mean block length = b

    count_exceeds = 0
    for _ in range(n_iters):
        # Draw one stationary bootstrap resample of the return matrix.
        resampled = _stationary_bootstrap_resample(candidate_returns, p, n_periods, rng)

        # The Reality Check uses the CENTRED bootstrap statistic (White eq. 3.3):
        #
        #   V̄*_k = max_j [ mean(boot_j) - mean(original_j) ]
        #
        # Subtracting mean_returns(j) centres each column's bootstrap distribution
        # at zero under H₀. The p-value is the fraction of bootstrap iterations
        # where V̄*_k >= observed_max (the original un-centred max mean return).
        #
        # Why this gives correct coverage under H₀:
        # When all strategies have zero true expected return, mean_returns ≈ 0
        # and observed_max ≈ 0. The centred bootstrap statistic V̄*_k is the
        # maximum of k approximately-zero-mean distributions, and p = P(V̄*_k >= 0)
        # converges to about 0.5. When the best strategy has genuinely positive
        # expected return, observed_max is large and V̄*_k rarely exceeds it,
        # so p is small - evidence of genuine edge.
        #
        # Note: comparing against observed_max (not 0) is correct. The centring
        # shifts the bootstrap distribution, not the threshold.
        boot_stat = float((resampled.mean(axis=0) - mean_returns).max())
        if boot_stat >= observed_max:
            count_exceeds += 1

    return float(count_exceeds / n_iters)


def _stationary_bootstrap_resample(
    data: np.ndarray,
    p: float,
    n_periods: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one stationary bootstrap resample of length T from data.

    Block lengths are drawn iid from Geometric(p), start positions are
    uniform over [0, T). The array is treated as circular so that blocks
    wrap around the boundary without truncation.

    Args:
        data: Array of shape (T, k).
        p: Geometric distribution parameter (1/mean_block_length).
        T: Target resample length.
        rng: NumPy random generator.

    Returns:
        Resampled array of shape (T, k).
    """
    # Tile data twice so circular indexing never goes out of bounds.
    circular = np.vstack([data, data])

    indices = np.empty(n_periods, dtype=np.intp)
    pos = 0
    while pos < n_periods:
        start = int(rng.integers(0, n_periods))
        # Block length from Geometric(p): number of trials until first success.
        block_len = int(rng.geometric(p))
        end = min(pos + block_len, n_periods)
        n = end - pos
        # Circular indexing: start + i wraps safely because we tiled twice.
        indices[pos:end] = np.arange(start, start + n) % n_periods
        pos = end

    return circular[indices, :]


def build_candidate_return_matrix(
    window_candidate_returns: list[dict[Any, np.ndarray]],
) -> np.ndarray:
    """
    Assemble a (T_total, k) return matrix from per-window candidate returns.

    Each walk-forward window produces a dictionary mapping a parameter key
    to daily return arrays for the test period of that window:
      - MovingAverageStrategy: key is (short_window, long_window) tuple
      - MomentumStrategy:      key is lookback int
      - Any new strategy:      key is whatever active_params() uses

    This function stitches return arrays together across windows for each
    parameter key. Only keys present in ALL windows are included (intersection)
    so the matrix is rectangular. Keys not evaluated in some window (e.g.
    because the window was too short for that parameter combination) are dropped.

    Args:
        window_candidate_returns: List of dicts, one per walk-forward window.
            Each dict maps a parameter key → daily_returns np.ndarray.
            All dicts in the list must use the same key type (e.g. all tuples
            or all ints - never mix key types across windows in one call).

    Returns:
        2D array of shape (T_total, k) where T_total is the total number
        of test-period days across all windows and k is the number of
        parameter keys present in every window.

    Raises:
        ValueError: If no parameter keys are common across all windows,
                    or if the input list is empty.
    """
    if not window_candidate_returns:
        raise ValueError("No window candidate returns provided.")

    # Find the intersection of parameter pairs across all windows.
    common_keys = set(window_candidate_returns[0].keys())
    for w in window_candidate_returns[1:]:
        common_keys &= set(w.keys())

    if not common_keys:
        raise ValueError(
            "No parameter pair was evaluated in every window. "
            "This typically means window data is too short for some candidates."
        )

    # Sort keys for a deterministic column order.
    sorted_keys = sorted(common_keys)

    # Concatenate return arrays across windows for each candidate.
    columns = []
    for key in sorted_keys:
        col = np.concatenate([w[key] for w in window_candidate_returns])
        columns.append(col)

    return np.column_stack(columns)
