"""Human-readable command-line entry point for baseline MPT analysis."""

from src.config import EFFICIENT_FRONTIER_FIGURE_PATH
from src.pipeline import MPTAnalysisResult, PortfolioResult, run_mpt_analysis
from src.visualization import plot_efficient_frontier


def print_portfolio_summary(
    title: str,
    portfolio: PortfolioResult,
) -> None:
    """Print metrics and weights from one structured portfolio result."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Expected Return: {portfolio.expected_return:.4%}")
    print(f"Volatility: {portfolio.volatility:.4%}")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio:.4f}")
    print("\nWeights:")
    for ticker, weight in portfolio.weights.items():
        print(f"{ticker}: {weight:.4%}")


def main() -> None:
    """Print and plot results from the reusable analysis pipeline."""
    result: MPTAnalysisResult = run_mpt_analysis()

    print(f"Loaded dataset from:\n{result.dataset_path}")
    print(f"\nAssets: {len(result.asset_names)}")
    print(f"Observations: {result.observations}")
    print(f"Date range: {result.start_date} to {result.end_date}")

    print("\nExpected Annual Returns:")
    print(
        result.annual_expected_returns.to_string(
            float_format=lambda value: f"{value:.4%}"
        )
    )

    print_portfolio_summary(
        "Global Minimum Variance Portfolio",
        result.gmv_portfolio,
    )
    print_portfolio_summary(
        "Maximum Sharpe Ratio Portfolio",
        result.maximum_sharpe_portfolio,
    )

    print(
        "\nEfficient frontier generated successfully "
        f"({len(result.efficient_frontier)} points)."
    )

    figure_path = plot_efficient_frontier(
        frontier=result.efficient_frontier,
        gmv_return=result.gmv_portfolio.expected_return,
        gmv_volatility=result.gmv_portfolio.volatility,
        maximum_sharpe_return=result.maximum_sharpe_portfolio.expected_return,
        maximum_sharpe_volatility=result.maximum_sharpe_portfolio.volatility,
        output_path=EFFICIENT_FRONTIER_FIGURE_PATH,
    )
    print(f"\nFigure saved to:\n{figure_path}")


if __name__ == "__main__":
    main()

