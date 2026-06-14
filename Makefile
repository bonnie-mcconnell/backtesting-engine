.PHONY: run run-quick run-ma run-kalman run-momentum run-costs run-frozen run-custom run-multi run-multi-all test lint typecheck check install clean help

# ── Installation ──────────────────────────────────────────────────────────────

## Install all dependencies (dev + prod)
install:
	poetry install

# ── Running strategies ────────────────────────────────────────────────────────

## Quick first look: MA crossover on SPY, no cost sweep (~3 min)
run-quick:
	python -c "import os; os.makedirs('results', exist_ok=True)"
	poetry run backtesting-engine \
	  --ticker SPY --start 1993-01-29 --end 2024-12-31 \
	  --strategy ma --cost 0.001 --slippage 0.05 --delay 1 \
	  --train-years 3 --test-years 1 --output-dir results/

## Run all three strategies + cost sensitivity sweep (~10–15 min on Mac/Linux; longer on Windows, see docs/performance.md)
run:
	poetry run backtesting-engine

## Run moving average strategy only
run-ma:
	poetry run backtesting-engine --strategy ma

## Run Kalman filter strategy only
run-kalman:
	poetry run backtesting-engine --strategy kalman

## Run time-series momentum strategy only
run-momentum:
	poetry run backtesting-engine --strategy momentum

## Run cost sensitivity sweep only (skips strategy evaluation)
run-costs:
	poetry run backtesting-engine --costs-only

## Reproduce the exact README results (frozen end date, fixed seed, deterministic)
run-frozen:
	python -c "import os; os.makedirs('results', exist_ok=True)"
	poetry run backtesting-engine \
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

## Run cross-asset validation: MA strategy across SPY, QQQ, TLT, GLD (2005-2024)
## Tests whether null result on SPY holds across asset classes (~8 min)
run-multi:
	python -c "import os; os.makedirs('results', exist_ok=True)"
	poetry run backtesting-multi \
	  --strategy ma \
	  --tickers SPY QQQ TLT GLD \
	  --start 2005-01-01 \
	  --end 2024-12-31 \
	  --cost 0.001 \
	  --slippage 0.05 \
	  --delay 1 \
	  --train-years 3 \
	  --test-years 1 \
	  --seed 42 \
	  --output-dir results/

## Run cross-asset validation for all three strategies (~45-60 min due to Kalman)
## Produces the strongest null result: no strategy beats B&H across any asset class
run-multi-all:
	python -c "import os; os.makedirs('results', exist_ok=True)"
	poetry run backtesting-multi \
	  --strategy all \
	  --tickers SPY QQQ TLT GLD \
	  --start 2005-01-01 \
	  --end 2024-12-31 \
	  --cost 0.001 \
	  --slippage 0.05 \
	  --delay 1 \
	  --train-years 3 \
	  --test-years 1 \
	  --seed 42 \
	  --output-dir results/

## Run with a custom ticker and date range (all other flags use defaults)
## Example: make run-custom TICKER=QQQ START=2000-01-01 END=2023-12-31
## For full control, call the CLI directly: poetry run backtesting-engine --help
run-custom:
	poetry run backtesting-engine \
	  --ticker $(TICKER) \
	  --start $(START) \
	  $(if $(END),--end $(END),) \
	  $(if $(COST),--cost $(COST),)

# ── Quality checks ────────────────────────────────────────────────────────────

## Run the full test suite
test:
	poetry run pytest -v --tb=short

## Lint with ruff
lint:
	poetry run ruff check src/ tests/

## Type-check with mypy (strict mode)
typecheck:
	poetry run mypy src/

## Run lint + typecheck + tests (mirrors CI exactly)
check: lint typecheck test

# ── Maintenance ───────────────────────────────────────────────────────────────

## Remove generated dashboards, build artifacts, and cached market data
clean:
	rm -f dashboard_ma.html dashboard_kalman.html dashboard_momentum.html cost_sensitivity.html
	rm -rf results/
	rm -rf dist/
	rm -rf test-venv/
	rm -rf ~/.cache/backtesting-engine/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

## Print this help message
help:
	@echo ""
	@echo "  backtesting-engine - Make targets"
	@echo ""
	@grep -E '^##' Makefile | sed 's/## /  /'
	@echo ""
