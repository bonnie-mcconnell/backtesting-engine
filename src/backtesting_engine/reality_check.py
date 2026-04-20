"""
White's Reality Check for data snooping bias correction.

Reference: White, H. (2000). "A Reality Check for Data Snooping."
           Econometrica, 68(5), 1097–1126.

The problem
-----------
When you search over a universe of strategies (or parameter combinations)
and report the best performer, the naive p-value from a significance test
on that best performer is misleading. Even if no strategy has genuine edge,
the best of k strategies will appear to have a low p-value by chance - the
classical multiple comparisons problem.

White's Reality Check corrects for this. The null hypothesis is:

    H₀: No strategy in the universe has performance superior to a benchmark
        (here: zero excess return, i.e. the benchmark is cash).

The test statistic is the maximum mean excess return across all k strategies
in the universe. The p-value is the fraction of bootstrap replications in
which the maximum bootstrapped statistic exceeds the observed maximum.

Implementation
--------------
We use the stationary bootstrap (Politis & Romano, 1994) rather than the
fixed block-length bootstrap. The stationary bootstrap draws block lengths
geometrically at random with mean block length b, which produces a
stationary null distribution. This avoids the bias that arises from fixed
block lengths when the series has heterogeneous autocorrelation.

The mean block length b = sqrt(T) is White's recommendation. We use the
same seed as the rest of the engine for reproducibility.

How we integrate it
-------------------
During walk-forward training, we store the returns of EVERY candidate
strategy evaluated during grid search, not just the winner. After
walk-forward completes, we run the Reality Check over the full universe
of candidate returns across all test windows.

This gives the data-snooping corrected p-value: the probability that the
best observed strategy performance arose by chance when we searched over
the full parameter grid.

Interpretation
--------------
  rc_p_value < 0.05 → statistically significant after data-snooping correction.
  rc_p_value ≥ 0.05 → performance is consistent with data snooping; no
                       evidence of genuine edge in the search universe.

Note: the corrected p-value is always ≥ the naive p-value. If the naive
test rejects but the Reality Check does not, you have a data snooping
problem - the apparent significance came from searching, not from edge.
"""

import numpy as np

from backtesting_engine.config import BLOCK_BOOTSTRAP_SEED, N_PERMUTATIONS


def white_reality_check(
    candidate_returns: np.ndarray,
    n_bootstrap: int = N_PERMUTATIONS,
    seed: int = BLOCK_BOOTSTRAP_SEED,
) -> float:
    """
    White's Reality Check p-value for a universe of candidate strategies.

    Args:
        candidate_returns: Array of shape (T, k) where T is the number of
            time periods (days) and k is the number of candidate strategies.
            Each column is the daily return series of one candidate.
            The first column is assumed to be the best (selected) strategy.
        n_bootstrap: Number of stationary bootstrap replications.
        seed: Random seed for reproducibility.

    Returns:
        Reality Check p-value in [0, 1]. Small values indicate the best
        strategy's performance is unlikely under the joint null.

    Raises:
        ValueError: If candidate_returns has fewer than 2 dimensions or
                    fewer than 1 strategy.
    """
    if candidate_returns.ndim != 2:
        raise ValueError(
            f"candidate_returns must be 2D (T × k), got shape {candidate_returns.shape}."
        )
    n_periods, k = candidate_returns.shape
    if k < 1:
        raise ValueError("Need at least one candidate strategy.")
    if n_periods < 2:
        raise ValueError("Need at least 2 time periods to bootstrap.")

    # Mean excess returns for each candidate.
    # We use excess return above zero (benchmark = cash).
    mean_returns = candidate_returns.mean(axis=0)   # shape (k,)

    # Observed test statistic: maximum mean return across all candidates.
    observed_max = float(mean_returns.max())

    # Stationary bootstrap: variable block lengths drawn from Geometric(1/b).
    b = max(1, int(np.sqrt(n_periods)))
    rng = np.random.default_rng(seed)
    p = 1.0 / b   # geometric parameter

    count_exceeds = 0
    for _ in range(n_bootstrap):
        # Build a bootstrap resample of length T using stationary bootstrap.
        resampled = _stationary_bootstrap_resample(candidate_returns, p, n_periods, rng)

        # Bootstrap statistic: max mean return across candidates in this resample.
        boot_means = resampled.mean(axis=0)

        # Reality Check uses the CENTRED bootstrap statistic:
        # boot_stat = max_j [mean(boot_j) - mean(original_j)]
        # Centering removes the mean of the original series so that under H₀
        # the bootstrap distribution is centred at zero.
        boot_stat = float((boot_means - mean_returns).max())
        if boot_stat >= observed_max - mean_returns.max():
            count_exceeds += 1

    # Note: under H₀, observed_max - mean_returns.max() = 0, so we are
    # computing P(max centred bootstrap statistic >= 0), which should be ~0.5
    # when H₀ holds. A small p-value means the observed maximum is in the
    # right tail of the null distribution - evidence of genuine edge.
    return float(count_exceeds / n_bootstrap)


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
    data.shape[1]
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
    window_candidate_returns: list[dict[tuple[int, int], np.ndarray]],
) -> np.ndarray:
    """
    Assemble a (T_total, k) return matrix from per-window candidate returns.

    Each walk-forward window produces a dictionary mapping (short, long)
    parameter pairs to daily return arrays for the test period of that
    window. This function stitches those arrays together across windows
    for each candidate.

    Only candidates present in ALL windows are included (intersection).
    Candidates that were not evaluated in some window (because the window
    was too short for that parameter pair) are dropped.

    Args:
        window_candidate_returns: List of dicts, one per walk-forward window.
            Each dict maps (short, long) → daily_returns np.ndarray.

    Returns:
        2D array of shape (T_total, k) where T_total is the total number
        of test-period days across all windows and k is the number of
        candidates present in every window.

    Raises:
        ValueError: If no candidates are common across all windows.
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
