# Contributing

## Setup

```bash
git clone https://github.com/bonnie-mcconnell/backtesting-engine.git
cd backtesting-engine
poetry install
make check   # lint + typecheck + tests - all three must pass before opening a PR
```

Python 3.11+. Poetry for dependency management.

---

## Adding a strategy

Subclass `BaseStrategy` in `src/backtesting_engine/strategy/`. Two methods are required, two optional.

```python
from backtesting_engine.strategy.base import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):

    def fit(self, train_data: pd.DataFrame) -> "MyStrategy":
        # Calibrate on training data only. Must not look at test data.
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Return pd.Series of int with values in {-1, 0, 1}.
        # 1 = buy, -1 = sell, 0 = hold.
        ...
```

**Optional - override if your strategy has a lookback window:**

```python
    def generate_signals_with_context(
        self, context_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.Series:
        combined = pd.concat([context_data, test_data])
        return self.generate_signals(combined).loc[test_data.index]
```

Without this, rolling windows at the start of each test period warm up on
test-period data - a mild look-ahead. The base class falls back to
`generate_signals(test_data)` if you don't override it.

**Optional - override if your strategy searches a parameter grid:**

```python
    def candidate_test_returns(
        self, test_data: pd.DataFrame, context_data=None
    ) -> dict:
        # Return {param_key: daily_return_series} for every candidate evaluated
        # during fit(). These must be test-period returns, not training returns.
        return {}
```

If you skip this, the Reality Check p-value is NaN - fine for parameter-free
strategies, a problem if you ran a grid search and are reporting the best result.

**Example: momentum strategy**

```python
class MomentumStrategy(BaseStrategy):

    def __init__(self, lookback: int = 63) -> None:
        self.lookback = lookback

    def fit(self, train_data: pd.DataFrame) -> "MomentumStrategy":
        return self  # no learned parameters

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        momentum = data["close"].pct_change(self.lookback)
        above_zero = (momentum > 0).astype(int)
        return above_zero.diff().fillna(0).astype(int)
```

---

## The data separation contract

`fit(train_data)` must not touch test data. The orchestrator passes train and test
slices as separate arguments and never mixes them. Don't store references that
allow indirect access to future data.

The `generate_signals_with_context` context window is the tail of *training* data,
used only to warm up rolling windows. Signal values during the context period are
discarded - only signals on `test_data.index` are returned.

---

## Statistical invariants

**Fisher p, not averaged p.** Use `summary_metrics.combined_p_value` as the
primary significance metric. Per-window `p_value` fields are diagnostic.

**Reality Check corrects for grid search.** If Fisher p < 0.05 but Reality Check
p >= 0.05, the apparent significance came from searching many parameter pairs,
not from genuine edge. Implement `candidate_test_returns()` if your strategy
uses a parameter grid.

**Signal delay is first-order.** Run with `signal_delay=1`. Many strategies that
look profitable at delay=0 become marginal at delay=1. Slippage is second-order
for daily strategies on liquid ETFs.

---

## Tests

```bash
make test                                    # full suite
poetry run pytest tests/test_kalman.py -v   # single module
poetry run pytest -k "test_centering" -v    # single test by name
```

Tests use synthetic data only - no network calls. yfinance is mocked in
`tests/test_data/test_ingestion.py`.

Hard-code expected values. Never compute expected output by calling the function
under test - that makes the test circular. If you cannot compute the expected
value by hand, write a property test instead (e.g. "result is always in [0, 1]").

---

## Code style

- Type hints on every signature. `mypy --strict` must pass.
- Docstrings explain *why* a decision was made, not *what* the code does.
- Constants in `config.py`, each with a one-line justification.
- `frozen=True` on result dataclasses unless there is a specific reason not to.
  Document the reason if you deviate.
- References in comments are attribution. If the code cites White (2000), that
  means the implementation follows that paper's specific construction and a reader
  should be able to verify it.

---

## Dependency graph

```
config ← models ← metrics / execution ← strategy ← walk_forward ← benchmark
                                                                  ← dashboard
                                                                  ← main
```

Nothing in a lower layer imports from a higher one at runtime. One intentional exception: `execution.py` imports `BaseStrategy` under `TYPE_CHECKING` with `from __future__ import annotations`. At runtime this import is never evaluated, so there is no circular dependency. At type-check time, mypy sees the full type. The reason this guard is needed is that `strategy/__init__.py` re-exports from `execution.py`, which would create a genuine cycle if `execution.py` imported from `strategy/` unconditionally. `TYPE_CHECKING` is the correct resolution here - not a workaround, but the standard Python mechanism for exactly this pattern.

---

## Scope

This is a backtesting and statistical evaluation tool, not a trading system.
No live data, no order management, no position sizing beyond full-portfolio
allocation. The execution model is a reasonable approximation for daily strategies
on liquid ETFs; it is not suitable for intraday strategies or thinly traded names.
