"""Configurable parameters for the baseline MPT project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TICKERS = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "JNJ",
    "XOM",
    "PG",
)

# yfinance treats START_DATE as inclusive and END_DATE as exclusive.
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
FRONTIER_POINTS = 50

RAW_PRICES_PATH = PROJECT_ROOT / "data" / "raw" / "stock_prices.parquet"
EFFICIENT_FRONTIER_FIGURE_PATH = (
    PROJECT_ROOT / "outputs" / "figures" / "efficient_frontier.png"
)

# Acquisition-stage data-quality thresholds.
MIN_OBSERVATIONS = 750
MAX_MISSING_FRACTION = 0.05
