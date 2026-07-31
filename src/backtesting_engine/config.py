# All configurable constants for the backtesting engine.
# Change values here only - never use magic numbers elsewhere in the codebase.

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

INITIAL_PORTFOLIO_VALUE: float = 100_000.0
# retail-scale capital; large enough that 0.1% costs (~$100/trade) are non-trivial

POSITION_SIZE_FRACTION: float = 1.0
# full allocation per trade; ensures all returns are attributable to strategy

# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------

TRANSACTION_COST_RATE: float = 0.001
# 0.1% per side, reflecting retail brokerage (e.g. Interactive Brokers tiered)
# intentionally conservative - understates strategy performance relative to
# institutional costs

# ---------------------------------------------------------------------------
# Walk-forward windows
# ---------------------------------------------------------------------------

TRAINING_WINDOW_YEARS: int = 3
# 3:1 train/test ratio captures a full bull/bear cycle without regime overfitting

TESTING_WINDOW_YEARS: int = 1
# one-year OOS windows produce ~28 nominal windows on ~32 years of data (SPY 1993–2024;
# exact count depends on trading calendar: floor((n_bars - 3×252) / 252))

# ---------------------------------------------------------------------------
# Moving average crossover
# ---------------------------------------------------------------------------

MOVING_AVERAGE_SHORT_DAYS: int = 50
MOVING_AVERAGE_LONG_DAYS: int = 200
# industry-standard pair; 200-day MA is widely tracked institutionally

MA_SHORT_RANGE: tuple[int, int] = (20, 80)  # (min, max) days inclusive
MA_LONG_RANGE: tuple[int, int] = (100, 250)  # (min, max) days inclusive
MA_STEP: int = 10  # coarse grid by design - finer grids overfit

# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------

ANNUALISATION_FACTOR: int = 252
# NYSE/NASDAQ trading days per year

# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

RISK_FREE_RATE: float = 0.0
# daily risk-free rate for Sharpe/Sortino excess return calculation
# zero makes results comparable across time periods with different rate regimes
# for a 5% annual rate: set to 0.05 / 252 ≈ 0.000198

# ---------------------------------------------------------------------------
# Bootstrap significance test
# ---------------------------------------------------------------------------

N_PERMUTATIONS: int = 10_000
# below 1,000 the p-value estimate is noisy; above 100,000 gives diminishing returns
# 10,000 is standard in finance literature (White, 2000)
# note: test suite patches this to 200 via conftest.py to keep CI fast

SIGNIFICANCE_THRESHOLD: float = 0.05
# treat as a guideline, not a hard cutoff - p=0.049 and p=0.051 aren't meaningfully different

BLOCK_BOOTSTRAP_SEED: int = 7919
# 7919 is the 1000th prime. Verified stable: p-values for the SPY 1993-2024 run
# are within ±0.003 of seed=42 and seed=137. Override via --seed for sensitivity
# analysis; docs/methodology.md documents this check under "Verifying results
# are not seed-dependent".

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

TICKER: str = "SPY"
# S&P 500 ETF; 30+ years of history from inception (1993); no survivorship bias

START_DATE: str = "1993-01-01"
# CLI default; yfinance returns data from the first available trading day on or after this

# ---------------------------------------------------------------------------
# Momentum strategy
# ---------------------------------------------------------------------------

MOMENTUM_LOOKBACKS: list[int] = [20, 40, 60, 90, 120, 180, 250]
# 1 month through ~12 months, matching intermediate-horizon momentum documented
# in Moskowitz, Ooi & Pedersen (2012)
# coarser than the MA grid by design - momentum Sharpe is flatter across lookbacks
