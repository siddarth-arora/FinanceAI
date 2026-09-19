"""Exercise the actual local Parquet boundary, including invalid datasets."""

import pandas as pd
import pytest

from src.data_loader import load_stock_prices


def test_load_preserves_frozen_data(parquet_path, prices):
    before = parquet_path.read_bytes()
    pd.testing.assert_frame_equal(load_stock_prices(parquet_path), prices, check_freq=False)
    assert parquet_path.read_bytes() == before


def test_missing_file_fails_without_downloading(tmp_path):
    with pytest.raises(FileNotFoundError, match="explicitly"):
        load_stock_prices(tmp_path / "missing.parquet")


@pytest.mark.parametrize("invalid", ["ticker_order", "duplicate_dates", "zero", "negative"])
def test_invalid_frozen_dataset_is_rejected(tmp_path, prices, invalid):
    if invalid == "ticker_order":
        prices = prices[prices.columns[::-1]]
        message = "tickers and order"
    elif invalid == "duplicate_dates":
        prices.index = prices.index[:-1].append(prices.index[-2:-1])
        message = "duplicate dates"
    else:
        prices.iloc[0, 0] = 0 if invalid == "zero" else -1
        message = "positive"
    path = tmp_path / "invalid.parquet"
    prices.to_parquet(path)
    with pytest.raises(ValueError, match=message):
        load_stock_prices(path)
