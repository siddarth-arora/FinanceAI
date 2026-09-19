"""Deterministic fixtures independent of the private frozen dataset."""

import numpy as np
import pandas as pd
import pytest

from src.config import TICKERS


@pytest.fixture
def prices():
    days = np.arange(80)[:, None]
    assets = np.arange(len(TICKERS))[None, :]
    daily = 0.0002 + assets * 0.0001 + 0.008 * np.sin(days * (assets + 1) / 7)
    return pd.DataFrame(
        100 * np.cumprod(1 + daily, axis=0),
        columns=TICKERS,
        index=pd.date_range("2024-01-01", periods=80, name="Date"),
    )


@pytest.fixture
def parquet_path(tmp_path, prices):
    path = tmp_path / "prices.parquet"
    prices.to_parquet(path)
    return path


@pytest.fixture
def synthetic_market():
    return np.array([0.06, 0.12, 0.20]), np.array(
        [[0.01, 0.002, 0.001], [0.002, 0.04, 0.006], [0.001, 0.006, 0.09]]
    )
