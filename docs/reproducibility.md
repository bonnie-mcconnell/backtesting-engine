# Reproducibility

The results in the README were generated with a fixed end date so they do not change as new market data arrives. This document explains how to reproduce them exactly.

---

## Frozen result command

```bash
backtesting-engine \
  --ticker SPY \
  --start 1993-01-29 \
  --end 2024-12-31 \
  --cost 0.001 \
  --slippage 0.05 \
  --delay 1 \
  --train-years 3 \
  --test-years 1 \
  --seed 42 \
  --output-dir results/
```

This command is also in `Makefile` as `make run-frozen`.

---

## Why results can differ from a live run

Running `make run` without `--end` fetches data through today's date. If you run this after the README was written, you will get different results for two reasons:

1. **More data.** Additional out-of-sample windows will be included in the walk-forward run.
2. **yfinance backfill revisions.** Yahoo Finance occasionally revises historical adjusted prices when it detects data errors. These revisions are small but can shift per-window metrics by a few basis points.

Use `--end 2024-12-31` to reproduce the exact README results.

---

## Environment

The frozen results were generated with:

```
Python       3.11
pandas       2.2.x
numpy        1.26.x
scipy        1.12.x
yfinance     0.2.x
pyarrow      15.x
```

Run `pip show backtesting-engine pandas numpy scipy yfinance pyarrow` to check your installed versions.

---

## Cached data

On first run, `load_data()` downloads from Yahoo Finance and writes a Parquet file to `~/.cache/backtesting-engine/`. Subsequent runs with the same `--ticker`, `--start`, and `--end` read from cache.

For fixed `--end` runs, the cache TTL is 365 days (frozen data does not change). For live runs (no `--end`), the TTL is 24 hours.

To force a fresh download: `make run-frozen --no-cache` or add `--no-cache` to any command.

---

## Verifying the cache

```python
from pathlib import Path
cache_dir = Path.home() / ".cache" / "backtesting-engine"
for f in cache_dir.glob("*.parquet"):
    print(f.name, f.stat().st_size // 1024, "KB")
```
