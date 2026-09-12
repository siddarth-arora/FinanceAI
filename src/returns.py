"""Calculate historical simple returns and MPT input statistics."""

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS
from src.data_loader import load_stock_prices


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns: R_t = (P_t / P_(t-1)) - 1."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("Prices must be provided as a Pandas DataFrame.")
    if len(prices) < 2:
        raise ValueError("At least two price observations are required.")
    if prices.isna().any().any():
        raise ValueError("Prices must not contain missing values.")

    previous_prices = prices.shift(1)
    daily_returns = (prices / previous_prices) - 1.0

    # The first row has no previous price, so its return is undefined.
    daily_returns = daily_returns.iloc[1:]

    if daily_returns.isna().any().any():
        raise ValueError("Calculated daily returns contain missing values.")
    if not np.isfinite(daily_returns.to_numpy()).all():
        raise ValueError("Calculated daily returns contain non-finite values.")

    return daily_returns


def calculate_expected_returns(
    daily_returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS,
) -> tuple[pd.Series, pd.Series]:
    """Calculate mean daily returns and arithmetic annual expected returns."""
    _validate_daily_returns(daily_returns)
    _validate_trading_days(trading_days)

    mean_daily_returns = daily_returns.mean()
    annual_expected_returns = mean_daily_returns * trading_days
    return mean_daily_returns, annual_expected_returns


def calculate_covariance_matrices(
    daily_returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate daily and annual sample covariance matrices."""
    _validate_daily_returns(daily_returns)
    _validate_trading_days(trading_days)

    # Pandas cov() uses the sample covariance denominator N - 1 by default.
    daily_covariance = daily_returns.cov()
    annual_covariance = daily_covariance * trading_days

    if not np.isfinite(annual_covariance.to_numpy()).all():
        raise ValueError("Calculated covariance matrix contains non-finite values.")

    return daily_covariance, annual_covariance


def _validate_daily_returns(daily_returns: pd.DataFrame) -> None:
    """Validate a daily-return table before estimating statistics."""
    if not isinstance(daily_returns, pd.DataFrame):
        raise TypeError("Daily returns must be provided as a Pandas DataFrame.")
    if daily_returns.empty:
        raise ValueError("Daily returns must not be empty.")
    if daily_returns.isna().any().any():
        raise ValueError("Daily returns must not contain missing values.")
    if not np.isfinite(daily_returns.to_numpy()).all():
        raise ValueError("Daily returns must not contain non-finite values.")


def _validate_trading_days(trading_days: int) -> None:
    """Require a positive integer annualization factor."""
    if not isinstance(trading_days, int) or isinstance(trading_days, bool):
        raise TypeError("Trading days must be a positive integer.")
    if trading_days <= 0:
        raise ValueError("Trading days must be a positive integer.")


def main() -> None:
    """Calculate and display the historical MPT inputs for verification."""
    prices = load_stock_prices()
    daily_returns = calculate_daily_returns(prices)
    mean_daily_returns, annual_expected_returns = calculate_expected_returns(
        daily_returns
    )
    daily_covariance, annual_covariance = calculate_covariance_matrices(
        daily_returns
    )

    print("First five daily simple returns:")
    print(daily_returns.head().to_string(float_format=lambda value: f"{value:.6f}"))

    print("\nDimensions:")
    print(f"Prices: {prices.shape}")
    print(f"Daily returns: {daily_returns.shape}")
    print(f"Mean daily returns: {mean_daily_returns.shape}")
    print(f"Annual expected returns: {annual_expected_returns.shape}")
    print(f"Daily covariance matrix: {daily_covariance.shape}")
    print(f"Annual covariance matrix: {annual_covariance.shape}")

    print("\nMean daily returns:")
    print(mean_daily_returns.to_string(float_format=lambda value: f"{value:.6f}"))

    print("\nAnnual expected returns (arithmetic mean x 252):")
    print(annual_expected_returns.to_string(float_format=lambda value: f"{value:.4%}"))

    print("\nTop-left 5 x 5 section of annual covariance matrix:")
    print(
        annual_covariance.iloc[:5, :5].to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )


if __name__ == "__main__":
    main()

