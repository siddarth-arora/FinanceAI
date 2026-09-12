"""Load and validate the frozen local stock-price dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_PRICES_PATH, TICKERS


def load_stock_prices(path: str | Path = RAW_PRICES_PATH) -> pd.DataFrame:
    """Load validated stock prices from a local Parquet file.

    The loader deliberately performs no downloading or data cleaning. If the
    frozen dataset is missing or inconsistent, it fails with a clear error.
    """
    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Stock-price dataset not found at {parquet_path}. "
            "Run `python -m src.download_data` explicitly to create it."
        )

    prices = pd.read_parquet(parquet_path, engine="pyarrow")
    _validate_loaded_prices(prices)
    return prices


def _validate_loaded_prices(prices: pd.DataFrame) -> None:
    """Check that loaded prices satisfy the frozen-dataset contract."""
    if prices.empty:
        raise ValueError("The stock-price dataset is empty.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("The dataset index must be a Pandas DatetimeIndex.")
    if prices.index.name != "Date":
        raise ValueError("The dataset index must be named 'Date'.")
    if prices.index.has_duplicates:
        duplicate_count = int(prices.index.duplicated().sum())
        raise ValueError(f"The dataset contains {duplicate_count} duplicate dates.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Dataset dates must be ordered from oldest to newest.")
    if prices.columns.has_duplicates:
        raise ValueError("The dataset contains duplicate asset columns.")

    actual_columns = tuple(prices.columns)
    if actual_columns != TICKERS:
        missing = sorted(set(TICKERS) - set(actual_columns))
        unexpected = sorted(set(actual_columns) - set(TICKERS))
        raise ValueError(
            "Dataset columns do not match configured tickers and order. "
            f"Missing: {missing or 'none'}; unexpected: {unexpected or 'none'}; "
            f"expected order: {list(TICKERS)}."
        )

    non_numeric_columns = [
        column
        for column in prices.columns
        if not pd.api.types.is_numeric_dtype(prices[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            f"Non-numeric price columns: {', '.join(non_numeric_columns)}"
        )
    if prices.isna().any().any():
        missing_count = int(prices.isna().sum().sum())
        raise ValueError(f"The dataset contains {missing_count} missing prices.")
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("The dataset contains infinite or non-finite prices.")
    if (prices <= 0).any().any():
        raise ValueError("All stock prices must be positive.")


def main() -> None:
    """Load the configured dataset and print a concise verification summary."""
    prices = load_stock_prices()

    print(f"Loaded dataset from: {RAW_PRICES_PATH}")
    print(f"Shape: {prices.shape}")
    print(f"Columns: {list(prices.columns)}")
    print(f"Index name: {prices.index.name}")
    print(f"Index dtype: {prices.index.dtype}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print("\nColumn data types:")
    print(prices.dtypes.to_string())
    print("\nMissing values by column:")
    print(prices.isna().sum().to_string())
    print("\nFirst five rows:")
    print(prices.head().round(2).to_string())


if __name__ == "__main__":
    main()

