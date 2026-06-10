"""
White's Reality Check for data snooping correction.

When a parameter grid is searched and the best performer reported, the naive
p-value is misleading. RC bootstraps the full candidate return matrix
simultaneously, preserving cross-candidate correlation, and returns the
fraction of bootstrap replications where the centred maximum exceeds the
observed maximum.

H₀: no strategy in the universe beats the zero-return (cash) benchmark.
A low RC p-value means the best candidate survives multiple-comparison
correction. If Fisher p < 0.05 but RC p >= 0.05, the apparent significance
came from the parameter search, not from genuine edge.

The stationary bootstrap (Politis & Romano, 1994) is used: block lengths
drawn geometrically with mean sqrt(T). Walk-forward stores test-period
returns for every grid candidate so RC runs over the full candidate universe
after all windows complete.

White, H. (2000). A Reality Check for Data Snooping. Econometrica, 68(5),
1097–1126.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backtesting_engine.config import BLOCK_BOOTSTRAP_SEED, N_PERMUTATIONS


def white_reality_check(
    candidate_returns: np.ndarray,
    n_bootstrap: int | None = None,
    seed: int = BLOCK_BOOTSTRAP_SEED,
    benchmark_returns: np.ndarray | None = None,
) -> float:
    """
    White's Reality Check p-value for a universe of candidate strategies.

    When benchmark_returns is provided, the test is run on active returns
    (candidate minus benchmark) rather than raw returns. This shifts the null
    from "no strategy beats cash" to "no strategy beats the benchmark", which
    is the correct question for an active equity strategy.

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
        benchmark_returns: Optional array of shape (T,) or (T, 1) containing
            benchmark (e.g. buy-and-hold) daily returns. When provided, the RC
            is computed on active returns (candidate_returns - benchmark_returns)
            column-wise. Must have the same length T as candidate_returns.

    Returns:
        Reality Check p-value in [0, 1]. Small values indicate the best
        strategy's performance is unlikely under the joint null.

    Raises:
        ValueError: If candidate_returns has fewer than 2 dimensions or
                    fewer than 1 strategy. If benchmark_returns is provided
                    but has a different length than candidate_returns.
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

    # When a benchmark is provided, test on active returns (strategy minus benchmark).
    # The null shifts from "no strategy beats cash" to "no strategy beats the benchmark".
    # All subsequent bootstrap operations work on test_returns identically - the
    # benchmark subtraction is just a change of basis for the return matrix.
    if benchmark_returns is not None:
        bm = np.asarray(benchmark_returns, dtype=float).ravel()
        if len(bm) != n_periods:
            raise ValueError(
                f"benchmark_returns length {len(bm)} must match candidate_returns "
                f"length {n_periods}."
            )
        # Broadcast subtract: each candidate column minus the benchmark series.
        test_returns = candidate_returns - bm[:, np.newaxis]
    else:
        test_returns = candidate_returns

    # Mean excess returns for each candidate over the observed period.
    # Benchmark = cash (zero return) or user-supplied benchmark, already
    # subtracted above. Excess return is thus raw return when no benchmark
    # is given, or active return when one is.
    mean_returns = test_returns.mean(axis=0)   # shape (k,)

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
        resampled = _stationary_bootstrap_resample(test_returns, p, n_periods, rng)

        # Centred bootstrap statistic (White eq. 3.3): subtract each column's
        # observed mean so the null distribution is anchored at zero.
        # p = fraction of iterations where the centred max >= observed max.
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
    indices = np.empty(n_periods, dtype=np.intp)
    pos = 0
    while pos < n_periods:
        start = int(rng.integers(0, n_periods))
        # Block length from Geometric(p): number of trials until first success.
        block_len = int(rng.geometric(p))
        end = min(pos + block_len, n_periods)
        n = end - pos
        # Circular wrapping so blocks that start near the end wrap to the beginning.
        indices[pos:end] = np.arange(start, start + n) % n_periods
        pos = end

    return data[indices, :]


def build_candidate_return_matrix(
    window_candidate_returns: list[dict[Any, np.ndarray]],
) -> np.ndarray:
    """
    Assemble a (T_total, k) return matrix from per-window candidate returns.

    Stitches return arrays across walk-forward windows for each candidate key.
    Only keys present in ALL windows are included so the matrix is rectangular.

    Args:
        window_candidate_returns: List of dicts, one per walk-forward window.
            Each dict maps a parameter key to a daily returns np.ndarray.

    Returns:
        2D array of shape (T_total, k).

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
