"""
Data ingestion: download OHLCV data with split- and dividend-adjusted prices via yfinance.

Returns a DataFrame with a DatetimeIndex and columns:
  close  - split- and dividend-adjusted closing price (required by all strategies)
  high   - split- and dividend-adjusted daily high  (required by execution model for slippage)
  low    - split- and dividend-adjusted daily low   (required by execution model for slippage)
  volume - daily volume                               (available for future volume-based strategies)

Why adjusted close AND adjusted high/low?
All three price columns are adjusted by the same split/dividend factor so that
the intraday range (high - low) remains consistent with the adjusted close.
If close is adjusted but high/low are not, the close can sit outside the [low, high]
band on ex-dividend dates, making fill = close ± slippage × (high - low) nonsensical.

The one edge case where adjustment still causes a minor issue is ex-dividend dates:
the adjusted close is reduced by the dividend amount, which can place it slightly
outside the adjusted [low, high] band due to rounding. load_data() clips the adjusted
close to [low, high] on such dates so the slippage model never produces a negative
intraday range. The clip is applied only when the discrepancy is below a threshold
(0.5% of close) - larger discrepancies indicate a data error and raise rather than
silently adjust.

Downloaded data is cached as Parquet in ~/.cache/backtesting-engine/. A cached
file is used if it exists and was written within the last 24 hours (or up to 1 year
for fixed end_date runs, since frozen data does not change). Set use_cache=False to
force a fresh download. The cache is keyed by (ticker, start_date, end_date) so
different tickers, start dates, and end dates are cached independently.
"""

import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Cache files older than this are considered stale and re-downloaded.
_CACHE_MAX_AGE_HOURS = 24
_CACHE_DIR = Path.home() / ".cache" / "backtesting-engine"


def load_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
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
        end_date: End date in 'YYYY-MM-DD' format, or None for today.
                  Setting a fixed end_date produces reproducible results
                  that do not change as new market data arrives.
        use_cache: If True (default), serve from cache when available and fresh.
                   Set to False to force a fresh network download.

    Returns:
        DataFrame with DatetimeIndex and columns:
            close  - adjusted closing price (float64)
            high   - adjusted daily high  (float64)
            low    - adjusted daily low   (float64)
            volume - daily share volume     (float64, if available)

    Raises:
        ValueError: If yfinance returns no data for the given ticker/range,
                    or if the data fails the adjusted/unadjusted band check.
    """
    if use_cache:
        cached = _load_from_cache(ticker, start_date, end_date)
        if cached is not None:
            return cached

    data = _download_and_clean(ticker, start_date, end_date)

    if use_cache:
        _save_to_cache(data, ticker, start_date, end_date)

    return data


def _cache_path(ticker: str, start_date: str, end_date: str | None = None) -> Path:
    """Return the Parquet cache path for a (ticker, start_date, end_date) triple."""
    end_str = end_date or "latest"
    key = f"{ticker.upper()}_{start_date}_{end_str}"
    safe = hashlib.md5(key.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{ticker.upper()}_{safe}.parquet"


def _load_from_cache(
    ticker: str, start_date: str, end_date: str | None = None
) -> pd.DataFrame | None:
    """
    Return cached data if it exists and is fresh, otherwise None.

    Freshness check: file modification time within _CACHE_MAX_AGE_HOURS.
    Fixed end_date runs use a much longer TTL (365 days) since the data
    is frozen by definition and should not be re-downloaded unnecessarily.
    If the file exists but is stale, it is left in place (will be overwritten
    on next successful download).
    """
    path = _cache_path(ticker, start_date, end_date)
    if not path.exists():
        return None

    # Fixed end_date → data is frozen; use 1-year TTL to avoid pointless re-downloads.
    max_age = (365 * 24) if end_date is not None else _CACHE_MAX_AGE_HOURS
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > max_age:
        return None

    try:
        return pd.read_parquet(path)
    except (OSError, Exception):
        # Corrupt or unreadable cache file - re-download. We catch broadly here
        # because pyarrow raises different exception types for corrupt files.
        # The failure is non-fatal: returning None triggers a fresh download.
        return None


def _save_to_cache(
    data: pd.DataFrame, ticker: str, start_date: str, end_date: str | None = None
) -> None:
    """Write data to Parquet cache, warning (but not raising) on write errors."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(ticker, start_date, end_date)
        data.to_parquet(path)
    except Exception as e:
        # Disk full, permissions, missing parquet engine, or path error.
        # Non-fatal: caller has the data.
        warnings.warn(
            f"Cache write failed for {ticker}/{start_date} ({e}). "
            "Data will be re-downloaded on the next run.",
            stacklevel=2,
        )


# Maximum number of download attempts before raising.
# Three attempts with exponential back-off (1s, 2s) handle transient network
# glitches without hanging a long run. More retries are rarely warranted for
# yfinance: if the server is down, waiting longer won't help.
_MAX_DOWNLOAD_RETRIES: int = 3


def _download_and_clean(
    ticker: str, start_date: str, end_date: str | None = None
) -> pd.DataFrame:
    """Download from yfinance (with retry) and apply all cleaning steps.

    Retries up to _MAX_DOWNLOAD_RETRIES times on empty responses or network
    errors, with exponential back-off (1 s, 2 s, ...). A transient yfinance
    timeout or rate-limit returns an empty DataFrame, which we treat as a
    retriable condition rather than an immediate hard failure.
    """
    end_ts = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.now()
    last_exc: Exception | None = None

    for attempt in range(_MAX_DOWNLOAD_RETRIES):
        if attempt > 0:
            wait = 2 ** (attempt - 1)   # 1 s, 2 s
            time.sleep(wait)

        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_ts,
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:
            # yfinance raises various exceptions (ConnectionError, JSONDecodeError,
            # etc.) on transient failures. Retry up to the limit, then re-raise
            # with a clear message so the caller knows which attempt failed.
            last_exc = exc
            continue

        if raw is not None and not raw.empty:
            break   # success
    else:
        # All attempts returned empty or raised.
        if last_exc is not None:
            raise ValueError(
                f"Failed to download data for '{ticker}' after {_MAX_DOWNLOAD_RETRIES} "
                f"attempts. Last error: {type(last_exc).__name__}: {last_exc}. "
                "Check your network connection and that the ticker is valid."
            ) from last_exc
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start_date} and "
            f"{end_ts.date()} after {_MAX_DOWNLOAD_RETRIES} attempts. "
            "Check that the ticker is valid and the date range has trading activity."
        )

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' "
            f"between {start_date} and {end_ts.date()}. "
            "Check that the ticker is valid and the date range has trading activity."
        )

    # yfinance returns MultiIndex columns (field, ticker) with auto_adjust=False.
    raw.columns = raw.columns.get_level_values(0)

    available = set(raw.columns)
    # "Close" (unadjusted) is required alongside "Adj Close" to compute the
    # adjustment factor. Without it we cannot build adjusted high/low, and
    # using unadjusted high/low with an adjusted close produces a close that
    # sits outside [low, high] on ex-dividend dates, breaking the slippage model.
    required = {"Adj Close", "Close", "High", "Low"}
    missing = required - available
    if missing:
        raise ValueError(
            f"yfinance response missing expected columns: {missing}. "
            f"Available: {sorted(available)}. "
            "All four columns are required: 'Adj Close' and 'Close' to compute "
            "the dividend/split adjustment factor, 'High' and 'Low' for slippage fills."
        )

    data = raw[["Adj Close", "High", "Low"]].rename(columns={
        "Adj Close": "close",
        "High": "high",
        "Low": "low",
    })

    # Compute adjustment factor from Adj Close / Close.
    # Both columns are now guaranteed present by the check above.
    adjustment = raw["Adj Close"] / raw["Close"]
    data["high"] = raw["High"] * adjustment
    data["low"] = raw["Low"] * adjustment

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
