"""Download, validate, and freeze adjusted historical stock prices."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    END_DATE,
    MAX_MISSING_FRACTION,
    MIN_OBSERVATIONS,
    RAW_PRICES_PATH,
    START_DATE,
    TICKERS,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the explicit acquisition step."""
    parser = argparse.ArgumentParser(
        description="Download and freeze adjusted closing prices from Yahoo Finance."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of an existing Parquet dataset.",
    )
    return parser.parse_args()


def ensure_output_is_safe(output_path: Path, overwrite: bool) -> None:
    """Refuse to replace the frozen dataset without explicit permission."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists at {output_path}. "
            "No download was attempted. Re-run with --overwrite only if you "
            "intend to replace the frozen dataset."
        )


def download_adjusted_close_prices() -> pd.DataFrame:
    """Download daily, adjustment-aware closing prices for configured tickers."""
    print("Downloading adjusted daily prices from Yahoo Finance...")
    print(f"Requested tickers: {', '.join(TICKERS)}")
    print(f"Requested interval: {START_DATE} (inclusive) to {END_DATE} (exclusive)")

    downloaded = yf.download(
        tickers=list(TICKERS),
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        actions=False,
        group_by="column",
        multi_level_index=True,
        progress=False,
        threads=True,
        timeout=30,
    )

    print(f"Raw Yahoo response shape: {downloaded.shape}")
    if downloaded.empty:
        raise ValueError("Yahoo Finance returned no data.")
    if not isinstance(downloaded.columns, pd.MultiIndex):
        raise ValueError("Expected Yahoo Finance to return MultiIndex columns.")
    if "Close" not in downloaded.columns.get_level_values(0):
        raise ValueError("Yahoo Finance response does not contain adjusted Close data.")

    return downloaded["Close"].copy()


def validate_and_clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate ticker coverage and create a common, complete price history."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price rows must use a DatetimeIndex.")

    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    # Parquet commonly stores these daily timestamps at millisecond precision.
    # Normalizing before saving makes strict round-trip verification consistent.
    prices.index = prices.index.as_unit("ms")

    if prices.index.has_duplicates:
        duplicate_count = int(prices.index.duplicated().sum())
        raise ValueError(f"Found {duplicate_count} duplicate date rows.")

    prices = prices.sort_index()

    downloaded_tickers = [
        ticker for ticker in prices.columns if prices[ticker].notna().any()
    ]
    missing_tickers = sorted(set(TICKERS) - set(downloaded_tickers))

    print(f"Adjusted-close shape before cleaning: {prices.shape}")
    print(f"Downloaded ticker coverage: {len(downloaded_tickers)}/{len(TICKERS)}")
    print(f"Tickers with data: {', '.join(downloaded_tickers)}")
    if missing_tickers:
        raise ValueError(f"Missing requested tickers: {', '.join(missing_tickers)}")

    prices = prices.loc[:, list(TICKERS)]

    non_numeric_columns = [
        column
        for column in prices.columns
        if not pd.api.types.is_numeric_dtype(prices[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            f"Non-numeric price columns: {', '.join(non_numeric_columns)}"
        )

    print(f"Downloaded date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print("\nMissing values before common-date cleaning:")
    missing_report = pd.DataFrame(
        {
            "missing_count": prices.isna().sum(),
            "missing_fraction": prices.isna().mean(),
        }
    )
    print(missing_report.to_string(float_format=lambda value: f"{value:.4%}"))

    serious_missing = missing_report[
        "missing_fraction"
    ] > MAX_MISSING_FRACTION
    if serious_missing.any():
        affected = missing_report.index[serious_missing].tolist()
        raise ValueError(
            "Missing-data fraction exceeds "
            f"{MAX_MISSING_FRACTION:.1%} for: {', '.join(affected)}"
        )

    first_and_last_dates = pd.DataFrame(
        {
            "first_valid_date": [prices[ticker].first_valid_index() for ticker in TICKERS],
            "last_valid_date": [prices[ticker].last_valid_index() for ticker in TICKERS],
        },
        index=TICKERS,
    )
    print("\nPer-asset available date ranges:")
    print(first_and_last_dates.to_string())

    original_rows = len(prices)
    prices = prices.dropna(axis="index", how="any")
    removed_rows = original_rows - len(prices)
    print(f"\nRows removed to align all assets: {removed_rows}")

    if len(prices) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Only {len(prices)} complete observations remain; "
            f"at least {MIN_OBSERVATIONS} are required."
        )
    if prices.isna().any().any():
        raise ValueError("Missing values remain after common-date cleaning.")
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Prices contain infinite or non-finite values.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must all be positive.")

    prices.index.name = "Date"
    print(f"Validated dataset shape: {prices.shape}")
    print(f"Validated date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Duplicate dates: {int(prices.index.duplicated().sum())}")
    print(f"Missing values after cleaning: {int(prices.isna().sum().sum())}")
    return prices


def save_and_verify_prices(prices: pd.DataFrame, output_path: Path) -> None:
    """Save prices as Parquet and verify that the file can be reopened unchanged."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output_path, engine="pyarrow")

    reopened = pd.read_parquet(output_path, engine="pyarrow")
    pd.testing.assert_frame_equal(prices, reopened, check_freq=False)

    print(f"\nDataset saved to: {output_path}")
    print(f"Reopened Parquet shape: {reopened.shape}")
    print("Parquet verification succeeded.")


def main() -> None:
    """Run the explicit data-acquisition pipeline."""
    args = parse_args()

    try:
        ensure_output_is_safe(RAW_PRICES_PATH, args.overwrite)
        downloaded_prices = download_adjusted_close_prices()
        validated_prices = validate_and_clean_prices(downloaded_prices)
        save_and_verify_prices(validated_prices, RAW_PRICES_PATH)
    except (FileExistsError, ValueError, OSError, AssertionError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
