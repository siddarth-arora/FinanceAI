"""Reusable programmatic interface for running baseline MPT analysis."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.config import (
    FRONTIER_POINTS,
    RAW_PRICES_PATH,
    RISK_FREE_RATE,
    TRADING_DAYS,
)
from src.data_loader import load_stock_prices
from src.optimizer import (
    EfficientFrontierPoint,
    generate_efficient_frontier,
    optimize_global_minimum_variance,
    optimize_maximum_sharpe_ratio,
)
from src.portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_variance,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
)
from src.returns import (
    calculate_covariance_matrices,
    calculate_daily_returns,
    calculate_expected_returns,
)


@dataclass(frozen=True)
class PortfolioResult:
    """JSON-friendly metrics and weights for one optimized portfolio."""

    expected_return: float
    variance: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return this portfolio using only JSON-serializable values."""
        return {
            "expected_return": self.expected_return,
            "variance": self.variance,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "weights": self.weights.copy(),
        }


@dataclass(frozen=True)
class MPTAnalysisResult:
    """Structured output returned by the complete baseline MPT pipeline."""

    dataset_path: Path
    asset_names: tuple[str, ...]
    observations: int
    start_date: date
    end_date: date
    trading_days: int
    risk_free_rate: float
    annual_expected_returns: pd.Series
    annual_covariance: pd.DataFrame
    gmv_portfolio: PortfolioResult
    maximum_sharpe_portfolio: PortfolioResult
    efficient_frontier: tuple[EfficientFrontierPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete result to a JSON-serializable dictionary."""
        expected_returns = {
            str(ticker): float(value)
            for ticker, value in self.annual_expected_returns.items()
        }
        covariance = {
            str(row_ticker): {
                str(column_ticker): float(
                    self.annual_covariance.loc[row_ticker, column_ticker]
                )
                for column_ticker in self.annual_covariance.columns
            }
            for row_ticker in self.annual_covariance.index
        }
        frontier = [
            {
                "target_return": point.target_return,
                "expected_return": point.expected_return,
                "variance": point.variance,
                "volatility": point.volatility,
                "weights": {
                    ticker: float(weight)
                    for ticker, weight in zip(
                        self.asset_names,
                        point.weights,
                        strict=True,
                    )
                },
            }
            for point in self.efficient_frontier
        ]

        return {
            "metadata": {
                "dataset_path": str(self.dataset_path),
                "assets": list(self.asset_names),
                "asset_count": len(self.asset_names),
                "observations": self.observations,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "trading_days": self.trading_days,
                "risk_free_rate": self.risk_free_rate,
                "frontier_point_count": len(self.efficient_frontier),
                "constraints": {
                    "fully_invested": True,
                    "long_only": True,
                    "leverage_allowed": False,
                },
            },
            "annual_expected_returns": expected_returns,
            "annual_covariance": covariance,
            "portfolios": {
                "global_minimum_variance": self.gmv_portfolio.to_dict(),
                "maximum_sharpe": self.maximum_sharpe_portfolio.to_dict(),
            },
            "efficient_frontier": frontier,
        }


def run_mpt_analysis() -> MPTAnalysisResult:
    """Run the local-data MPT workflow and return structured results."""
    prices = load_stock_prices()
    daily_returns = calculate_daily_returns(prices)
    _, annual_expected_returns = calculate_expected_returns(daily_returns)
    _, annual_covariance = calculate_covariance_matrices(daily_returns)

    gmv_optimization = optimize_global_minimum_variance(annual_covariance)
    maximum_sharpe_optimization = optimize_maximum_sharpe_ratio(
        annual_expected_returns,
        annual_covariance,
        RISK_FREE_RATE,
    )
    frontier = generate_efficient_frontier(
        annual_expected_returns,
        annual_covariance,
        FRONTIER_POINTS,
        gmv_weights=gmv_optimization.x,
    )

    asset_names = tuple(str(ticker) for ticker in prices.columns)
    return MPTAnalysisResult(
        dataset_path=RAW_PRICES_PATH,
        asset_names=asset_names,
        observations=len(prices),
        start_date=prices.index.min().date(),
        end_date=prices.index.max().date(),
        trading_days=TRADING_DAYS,
        risk_free_rate=RISK_FREE_RATE,
        annual_expected_returns=annual_expected_returns.copy(),
        annual_covariance=annual_covariance.copy(),
        gmv_portfolio=_evaluate_portfolio(
            gmv_optimization.x,
            asset_names,
            annual_expected_returns,
            annual_covariance,
        ),
        maximum_sharpe_portfolio=_evaluate_portfolio(
            maximum_sharpe_optimization.x,
            asset_names,
            annual_expected_returns,
            annual_covariance,
        ),
        efficient_frontier=tuple(frontier),
    )


def _evaluate_portfolio(
    weights: ArrayLike,
    asset_names: tuple[str, ...],
    expected_returns: ArrayLike,
    covariance_matrix: ArrayLike,
) -> PortfolioResult:
    """Evaluate optimizer weights and attach their asset names."""
    weight_vector = np.asarray(weights, dtype=float)
    return PortfolioResult(
        expected_return=calculate_portfolio_return(
            weight_vector,
            expected_returns,
        ),
        variance=calculate_portfolio_variance(
            weight_vector,
            covariance_matrix,
        ),
        volatility=calculate_portfolio_volatility(
            weight_vector,
            covariance_matrix,
        ),
        sharpe_ratio=calculate_sharpe_ratio(
            weight_vector,
            expected_returns,
            covariance_matrix,
            RISK_FREE_RATE,
        ),
        weights={
            ticker: float(weight)
            for ticker, weight in zip(asset_names, weight_vector, strict=True)
        },
    )
