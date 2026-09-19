"""Shared synthetic analysis payload for downstream consumers."""

import pytest

from src.optimizer import generate_efficient_frontier, optimize_maximum_sharpe_ratio
from src.portfolio import calculate_portfolio_volatility


@pytest.fixture
def payload(synthetic_market):
    returns, covariance = synthetic_market
    assets = ["A", "B", "C"]
    frontier = generate_efficient_frontier(returns, covariance, 15)
    maximum = optimize_maximum_sharpe_ratio(returns, covariance, 0.04)
    return {
        "metadata": {
            "assets": assets, "dataset_path": "/fixture/prices.parquet",
            "start_date": "2024-01-01", "end_date": "2024-12-31",
            "trading_days": 252, "risk_free_rate": 0.04,
            "constraints": {"fully_invested": True, "long_only": True,
                            "leverage_allowed": False},
        },
        "annual_expected_returns": dict(zip(assets, returns.tolist())),
        "annual_covariance": {
            ticker: dict(zip(assets, row.tolist()))
            for ticker, row in zip(assets, covariance)
        },
        "portfolios": {"maximum_sharpe": {
            "volatility": calculate_portfolio_volatility(maximum.x, covariance),
        }},
        "efficient_frontier": [
            {"expected_return": point.expected_return, "variance": point.variance,
             "volatility": point.volatility, "weights": dict(zip(assets, point.weights.tolist()))}
            for point in frontier
        ],
    }
