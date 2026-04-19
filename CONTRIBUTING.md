# Contributing

## Development setup

```bash
git clone https://github.com/bonnie-mcconnell/backtesting-engine.git
cd backtesting-engine
poetry install
make check   # lint + typecheck + tests
```

Python 3.11+ required. The project uses Poetry for dependency management.

---

## Adding a new strategy

Every strategy is a subclass of `BaseStrategy` implementing two required methods and two optional ones.

```python
from backtesting_engine.strategy.base import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):

    def fit(self, train_data: pd.DataFrame) -> "MyStrategy":
        """
        Calibrate any parameters using only train_data.
        Must not access test data. Must return self.
        """
        # e.g. fit a model, run a grid search, estimate parameters
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Return pd.Series of int signals aligned to data.index.
        Values must be exactly in {-1 (sell), 0 (hold), 1 (buy)}.
        """
        ...

    # Optional - override if your strategy has a lookback period
    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        """
        Generate signals for test_data using context_data as warmup history.
        Default delegates to generate_signals(test_data).
        """
        combined = pd.concat([context_data, test_data])
        return self.generate_signals(combined).loc[test_data.index]

    # Optional - override if your strategy has a parameter search grid
    def candidate_test_returns(
        self, test_data: pd.DataFrame, context_data=None
    ) -> dict:
        """
        Return test-period returns for every candidate in the search universe.
        Used by walk_forward to build the White's Reality Check candidate matrix.
        Return empty dict if no parameter search is performed.
        """
        return {}
```

The orchestrator (`walk_forward`) calls these in order for each rolling window:
1. `fit(train_data)` - parameters calibrated in-sample
2. `generate_signals_with_context(context, test_data)` - signals generated out-of-sample
3. `candidate_test_returns(test_data, context)` - all candidates evaluated on test data

Your strategy never touches test data during `fit()`. The orchestrator enforces this by construction - it passes `train_data` and `test_data` as separate arguments and never mixes them.

### Example: momentum strategy

```python
class MomentumStrategy(BaseStrategy):
    """Buy when 63-day return is positive, sell when negative."""

    def __init__(self, lookback: int = 63) -> None:
        self.lookback = lookback

    def fit(self, train_data: pd.DataFrame) -> "MomentumStrategy":
        return self   # no learned parameters

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        momentum = data["close"].pct_change(self.lookback)
        position = (momentum > 0).astype(int)
        return position.diff().fillna(0).astype(int)
```

Run it through the engine:

```python
from backtesting_engine import walk_forward, ExecutionConfig

result = walk_forward(
    data,
    MomentumStrategy(lookback=63),
    execution=ExecutionConfig(slippage_factor=0.05, signal_delay=1),
)
```

---

## Statistical design principles

**Strict data separation.** Every test period must be strictly out-of-sample relative to its training window. Never pass test data to `fit()`. Never use parameters calibrated on test data to generate signals.

**Report Fisher p, not per-window p.** Individual window p-values are noisy. The Fisher combined p-value (`-2 Σ ln(pᵢ) ~ χ²(2k)`) is the primary significance metric because it aggregates evidence across windows correctly. Per-window values are reported for transparency.

**White's Reality Check for parameter search.** If your strategy searches over a parameter grid during `fit()`, implement `candidate_test_returns()` to return test-period returns for every candidate. Without this, the Reality Check p-value is not computed and the significance result is vulnerable to data snooping.

**Execution costs matter more than you think.** Always run with `signal_delay=1`. A strategy that requires acting on a signal at the same bar that generated it is not implementable. Use `cost_sensitivity_sweep()` to find the breakeven cost level - the point at which your strategy loses statistical significance.

---

## Running tests

```bash
make test              # full suite
make check             # lint + typecheck + tests
poetry run pytest tests/test_kalman.py -v   # single module
```

Tests use synthetic data only - no network calls. The yfinance download is mocked in `test_data/test_ingestion.py`.

Every numeric test derives its expected value independently of the implementation (hand-calculated or from a separate reference). Never compute an expected value by calling the function under test - that makes the test circular.

---

## Code style

- Type hints on every function signature (enforced by mypy --strict)
- Docstrings explain *why* decisions were made, not *what* the code does
- No magic numbers - every constant lives in `config.py` with a name and justification
- `frozen=True` on result dataclasses; mutable only during construction
- Comments that cite specific references (Politis & Romano 1994, White 2000, etc.) are not decoration - they are attribution for non-obvious statistical choices

---

## Project structure

```
src/backtesting_engine/
├── strategy/base.py        ← Start here when adding a strategy
├── strategy/               ← One file per strategy
├── walk_forward.py         ← Touch only if changing orchestration logic
├── metrics.py              ← Touch only if adding a new metric
├── execution.py            ← Touch only if changing execution model
└── config.py               ← All constants live here
```

The dependency graph is strictly layered: `config` ← `models` ← `metrics/execution` ← `strategy` ← `walk_forward` ← `dashboard/main`. Nothing in a lower layer imports from a higher layer. Circular imports indicate a design problem.