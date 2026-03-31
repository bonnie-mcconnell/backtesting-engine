"""
Main entry point for the backtesting engine. 
Ingests and validates data, instantiates strategy, 
runs walk-foward orchestrator with data and strategy, 
prints results for each window, summary metrics and significance.
"""
from backtesting_engine.config import SIGNIFICANCE_THRESHOLD, TICKER, START_DATE
from backtesting_engine.strategy.moving_average import MovingAverageStrategy
from backtesting_engine.data.ingestion import load_data
from backtesting_engine.data.validator import validate_data
from backtesting_engine.walk_forward import walk_forward


def main() -> None:
    data = load_data(TICKER, START_DATE)
    validate_data(data)
    strategy = MovingAverageStrategy() 
    result = walk_forward(data, strategy)

    print("Walk-forward analysis completed. Results:")
    for window_result in result.window_results:
        print(f"Train: {window_result.train_start.date()} to {window_result.train_end.date()}, "
              f"Test: {window_result.test_start.date()} to {window_result.test_end.date()}, "
              f"Sharpe Ratio: {window_result.metrics_result.sharpe_ratio:.2f}, "
              f"Sortino Ratio: {window_result.metrics_result.sortino_ratio:.2f}, "
              f"Max Drawdown: {window_result.metrics_result.max_drawdown:.2%}, "
              f"P-value: {window_result.metrics_result.p_value:.4f}")
    
    print("\nSummary Metrics:")
    print(f"  Sharpe Ratio:  {result.summary_metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.summary_metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown:  {result.summary_metrics.max_drawdown:.2%}")
    print(f"  Calmar Ratio:  {result.summary_metrics.calmar_ratio:.2f}")
    print(f"  Omega Ratio:   {result.summary_metrics.omega_ratio:.2f}")
    print(f"  P-value:       {result.summary_metrics.p_value:.4f}")
    
    if result.summary_metrics.p_value < SIGNIFICANCE_THRESHOLD:
        print("Strategy is statistically significant (p < 0.05).")
    else:
        print("Strategy is not statistically significant (p >= 0.05).")


if __name__ == "__main__":
    main()