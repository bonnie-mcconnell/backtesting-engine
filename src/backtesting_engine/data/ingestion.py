"""
Data ingestion: download OHLCV data with adjusted closing prices via yfinance.

Returns a DataFrame with a DatetimeIndex and columns:
  close  - split- and dividend-adjusted closing price (required by all strategies)
  high   - unadjusted daily high  (required by execution model for slippage)
  low    - unadjusted daily low   (required by execution model for slippage)
  volume - daily volume           (available for future volume-based strategies)

Why adjusted close but unadjusted high/low?
Adjusted closing prices are necessary for accurate return computation — without
adjustment, dividends and splits create artificial price gaps that look like
overnight losses. Intraday high/low are used only for slippage estimation
(fill = close ± factor × range), where what matters is the intraday range
width, not its absolute level.

The one case where this mixing causes a problem is ex-dividend dates: the
adjusted close is reduced by the dividend amount, which can place it slightly
outside the unadjusted [low, high] band. load_data() clips the adjusted close
to [low, high] on such dates so the slippage model never produces a negative
intraday range. The clip is applied only when the discrepancy is below a
threshold (0.5% of close) — larger discrepancies indicate a data error and
raise rather than silently adjust.

Caching
-------
Downloaded data is cached as Parquet in ~/.cache/backtesting-engine/. A cached
file is used if it exists and was written within the last 24 hours. Set
use_cache=False to force a fresh download. The cache is keyed by (ticker,
start_date) so different tickers and start dates are cached independently.
"""

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

# Cache files older than this are considered stale and re-downloaded.
_CACHE_MAX_AGE_HOURS = 24
_CACHE_DIR = Path.home() / ".cache" / "backtesting-engine"


def load_data(
    ticker: str,
    start_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker, with local caching.

    On first call, downloads from yfinance and writes a Parquet file to
    ~/.cache/backtesting-engine/. On subsequent calls within 24 hours,
    reads from cache instead of hitting the network.

    Args:
        ticker: Exchange ticker symbol (e.g. 'SPY', 'AAPL').
        start_date: Start date in 'YYYY-MM-DD' format.
        use_cache: If True (default), serve from cache when available and fresh.
                   Set to False to force a fresh network download.

    Returns:
        DataFrame with DatetimeIndex and columns:
            close  - adjusted closing price (float64)
            high   - unadjusted daily high  (float64)
            low    - unadjusted daily low   (float64)
            volume - daily share volume     (float64, if available)

    Raises:
        ValueError: If yfinance returns no data for the given ticker/range,
                    or if the data fails the adjusted/unadjusted band check.
    """
    if use_cache:
        cached = _load_from_cache(ticker, start_date)
        if cached is not None:
            return cached

    data = _download_and_clean(ticker, start_date)

    if use_cache:
        _save_to_cache(data, ticker, start_date)

    return data


def _cache_path(ticker: str, start_date: str) -> Path:
    """Return the Parquet cache path for a (ticker, start_date) pair."""
    key = f"{ticker.upper()}_{start_date}"
    safe = hashlib.md5(key.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{ticker.upper()}_{safe}.parquet"


def _load_from_cache(ticker: str, start_date: str) -> pd.DataFrame | None:
    """
    Return cached data if it exists and is fresh, otherwise None.

    Freshness check: file modification time within _CACHE_MAX_AGE_HOURS.
    If the file exists but is stale, it is left in place (will be overwritten
    on next successful download).
    """
    import time

    path = _cache_path(ticker, start_date)
    if not path.exists():
        return None

    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > _CACHE_MAX_AGE_HOURS:
        return None

    try:
        return pd.read_parquet(path)
    except (OSError, Exception):
        # Corrupt or unreadable cache file — re-download. We catch broadly here
        # because pyarrow and fastparquet raise different exception types for
        # corrupt files. The failure is non-fatal: returning None triggers a
        # fresh download.
        return None


def _save_to_cache(data: pd.DataFrame, ticker: str, start_date: str) -> None:
    """Write data to Parquet cache, warning (but not raising) on write errors."""
    import warnings
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(ticker, start_date)
        data.to_parquet(path)
    except OSError as e:
        # Disk full, permissions, or path error. Non-fatal: caller has the data.
        warnings.warn(
            f"Cache write failed for {ticker}/{start_date} ({e}). "
            "Data will be re-downloaded on the next run.",
            stacklevel=2,
        )


def _download_and_clean(ticker: str, start_date: str) -> pd.DataFrame:
    """Download from yfinance and apply all cleaning steps."""
    end_date = pd.Timestamp.now()
    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' "
            f"between {start_date} and {end_date.date()}. "
            "Check that the ticker is valid and the date range has trading activity."
        )

    # yfinance returns MultiIndex columns (field, ticker) with auto_adjust=False.
    raw.columns = raw.columns.get_level_values(0)

    available = set(raw.columns)
    required = {"Adj Close", "High", "Low"}
    missing = required - available
    if missing:
        raise ValueError(
            f"yfinance response missing expected columns: {missing}. "
            f"Available: {sorted(available)}."
        )

    data = raw[["Adj Close", "High", "Low"]].rename(columns={
        "Adj Close": "close",
        "High": "high",
        "Low": "low",
    })

    if "Volume" in available:
        data["volume"] = raw["Volume"]

    data.index = pd.DatetimeIndex(data.index)

    # On ex-dividend dates the adjusted close is reduced by the dividend amount
    # and can sit slightly below the unadjusted low. The slippage model uses
    # close ± factor × (high - low), so close must lie within [low, high].
    # Clip small discrepancies (< 0.5% of price); raise on larger ones.
    _reconcile_adjusted_close(data)

    return data


# Discrepancy threshold above which we treat the mismatch as a data error
# rather than a benign ex-dividend rounding artefact.
_MAX_CLIP_FRACTION = 0.005  # 0.5% of close


def _reconcile_adjusted_close(data: pd.DataFrame) -> None:
    """
    Clip the adjusted close into [low, high] on ex-dividend dates.

    Modifies the DataFrame in place. Raises ValueError if any discrepancy
    exceeds _MAX_CLIP_FRACTION of the closing price, which indicates a real
    data problem rather than a routine dividend adjustment.
    """
    import numpy as np

    below_low = data["close"] < data["low"]
    above_high = data["close"] > data["high"]
    problem_rows = below_low | above_high

    if not problem_rows.any():
        return

    discrepancy = np.maximum(
        data.loc[problem_rows, "low"] - data.loc[problem_rows, "close"],
        data.loc[problem_rows, "close"] - data.loc[problem_rows, "high"],
    )
    max_frac = float((discrepancy / data.loc[problem_rows, "close"]).max())

    if max_frac > _MAX_CLIP_FRACTION:
        worst_date = discrepancy.idxmax()
        raise ValueError(
            f"Adjusted close sits {max_frac:.2%} outside the [low, high] band "
            f"on {worst_date.date()}. This exceeds the {_MAX_CLIP_FRACTION:.1%} "
            "ex-dividend tolerance and likely indicates a data error."
        )

    # Small discrepancy: clip silently. Only affects slippage fill prices.
    data["close"] = data["close"].clip(lower=data["low"], upper=data["high"])
