# All configurable constants for the backtesting engine.
# Every constant lives here with a name and a justification.
# Change values here only - never use magic numbers elsewhere in the codebase.

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

INITIAL_PORTFOLIO_VALUE: float = 100_000.0
# Retail-scale capital. Large enough that 0.1% transaction costs (~$100/trade)
# are non-trivial, small enough to avoid market-impact distortions.

POSITION_SIZE_FRACTION: float = 1.0
# Full portfolio allocation per trade. Ensures all returns are attributable to
# strategy performance and eliminates cash-drag dilution from metrics.

# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------

TRANSACTION_COST_RATE: float = 0.001
# 0.1% per side, reflecting retail brokerage (e.g. Interactive Brokers tiered).
# Deliberately conservative - real institutional costs are lower, so this
# understates rather than overstates strategy performance.

# ---------------------------------------------------------------------------
# Walk-forward validation windows
# ---------------------------------------------------------------------------

TRAINING_WINDOW_YEARS: int = 3
# 3:1 train/test ratio captures a full bull/bear cycle without overfitting to
# a single market regime. Standard in academic strategy evaluation literature.

TESTING_WINDOW_YEARS: int = 1
# One-year out-of-sample windows produce ~26 independent evaluations on 30
# years of SPY data. Enough windows to distinguish consistency from luck.

# ---------------------------------------------------------------------------
# Moving average crossover strategy (golden cross / death cross)
# ---------------------------------------------------------------------------

MOVING_AVERAGE_SHORT_DAYS: int = 50
MOVING_AVERAGE_LONG_DAYS: int = 200
# Industry-standard pair for equity trend following. The 200-day MA is
# widely tracked by institutional participants, which gives the signal
# partial self-fulfilling properties on broad indices.

# Parameter search bounds used during walk-forward training.
# Grid is intentionally coarse to avoid overfitting on the training window.
MA_SHORT_RANGE: tuple[int, int] = (20, 80)    # (min, max) days inclusive
MA_LONG_RANGE: tuple[int, int] = (100, 250)   # (min, max) days inclusive
MA_STEP: int = 10                              # grid step for both axes

# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------

ANNUALISATION_FACTOR: int = 252
# NYSE/NASDAQ trading days per year. Used to scale daily Sharpe/Sortino/Calmar
# to annualised equivalents, enabling comparison with published benchmarks.

# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

RISK_FREE_RATE: float = 0.0
# Daily risk-free rate subtracted from returns before Sharpe/Sortino computation.
# Set to 0.0 as a conservative default - subtracting a positive rate can only
# reduce the apparent Sharpe, so zero is the most favourable assumption and
# produces an upper bound. Change to e.g. 0.05 / 252 for a 5% annual rate.

# ---------------------------------------------------------------------------
# Monte Carlo block bootstrap significance test
# ---------------------------------------------------------------------------

N_PERMUTATIONS: int = 10_000
# Below 1,000 the p-value estimate is noisy (+/-0.02 at p=0.05);
# above 100,000 gives diminishing accuracy returns. 10,000 is the
# standard choice in finance literature (e.g. White, 2000).

SIGNIFICANCE_THRESHOLD: float = 0.05
# Fisher's conventional threshold. Treat as a guideline, not a hard rule:
# p=0.049 and p=0.051 are not meaningfully different.

BLOCK_BOOTSTRAP_SEED: int = 42
# Fixed seed for reproducibility. Any seed produces valid statistics;
# this one ensures output is identical across runs for documentation purposes.

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

TICKER: str = "SPY"
# S&P 500 ETF. Chosen for liquidity, 30+ years of history from inception
# (1993), and absence of survivorship bias relative to individual equities.

START_DATE: str = "1993-01-01"
# SPY inception date, maximising available history for walk-forward windows.
# Earlier dates would require switching to a different index proxy.

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

FIGURE_DPI: int = 150
# High enough for crisp display on retina screens; low enough to keep
# file sizes manageable for README embedding.
