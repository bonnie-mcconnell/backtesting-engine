.PHONY: run test lint typecheck check install clean

## Install all dependencies (dev + prod)
install:
	poetry install

## Run the backtesting engine
run:
	poetry run backtesting-engine

## Run the full test suite
test:
	poetry run pytest -v --tb=short

## Lint with ruff
lint:
	poetry run ruff check src/ tests/

## Type-check with mypy
typecheck:
	poetry run mypy src/

## Run lint + typecheck + tests (full CI locally)
check: lint typecheck test

## Remove generated artefacts
clean:
	rm -f backtest_results.png
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +