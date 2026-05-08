.PHONY: run run-ma run-kalman run-momentum run-costs run-frozen run-custom test lint typecheck check install clean help

# ── Installation ──────────────────────────────────────────────────────────────

## Install all dependencies (dev + prod)
install:
	poetry install

# ── Running strategies ────────────────────────────────────────────────────────

## Run all three strategies + cost sensitivity sweep (~4–6 min on first run)
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
	mkdir -p results
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

## Run with a custom ticker, date range, and execution config
## Example: make run-custom TICKER=QQQ START=2000-01-01 END=2023-12-31
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
