# All configurable constants for the backtesting engine.
# Change values here only, never use magic numbers elsewhere in the codebase.

# Portfolio settings
INITIAL_PORTFOLIO_VALUE: float = 100_000.0  # Retail-scale capital, large enough that 0.1% transaction costs (~$100/trade) are meaningful
POSITION_SIZE_FRACTION: float = 1.0  # Full portfolio allocation per trade: ensures all returns are attributable to strategy performance, eliminating cash drag from metrics.

# Transaction costs
TRANSACTION_COST_RATE: float = 0.001  # 0.1% per trade, reflects retail brokerage

# Walk-forward validation windows
TRAINING_WINDOW_YEARS: int = 3  # 3:1 train/test ratio - captures a full bull/bear cycle without overfitting to a single regime
TESTING_WINDOW_YEARS: int = 1 # 3:1 train/test ratio

# Moving average crossover strategy (golden cross/death cross)
# 50/200-day pair: industry-standard signal for equity trend following
MOVING_AVERAGE_SHORT_DAYS: int = 50
MOVING_AVERAGE_LONG_DAYS: int = 200

# Annualisation
ANNUALISATION_FACTOR: int = 252  # Trading days per year. used to annualise daily Sharpe ratio

# Monte Carlo permutation test
N_PERMUTATIONS: int = 10_000  # Below 1,000 the p-value estimate is noisy and above 100,000 gives diminishing returns
SIGNIFICANCE_THRESHOLD: float = 0.05  # Fisher's conventional threshold, treat as a guideline, not a hard rule

# Data
TICKER: str = "SPY"  # S&P 500 ETF, liquid, 30 years of data
START_DATE: str = "1993-01-01" # SPY inception date to maximise available history for walk-forward windows