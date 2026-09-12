"""Visualization functions for baseline MPT results."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.config import EFFICIENT_FRONTIER_FIGURE_PATH
from src.optimizer import EfficientFrontierPoint


def plot_efficient_frontier(
    frontier: Sequence[EfficientFrontierPoint],
    gmv_return: float,
    gmv_volatility: float,
    maximum_sharpe_return: float,
    maximum_sharpe_volatility: float,
    output_path: str | Path = EFFICIENT_FRONTIER_FIGURE_PATH,
) -> Path:
    """Plot the efficient frontier and highlight two important portfolios."""
    if len(frontier) < 2:
        raise ValueError("At least two efficient-frontier points are required.")

    frontier_returns = np.array(
        [point.expected_return for point in frontier],
        dtype=float,
    )
    frontier_volatilities = np.array(
        [point.volatility for point in frontier],
        dtype=float,
    )
    highlighted_values = np.array(
        [
            gmv_return,
            gmv_volatility,
            maximum_sharpe_return,
            maximum_sharpe_volatility,
        ],
        dtype=float,
    )

    if not np.isfinite(frontier_returns).all():
        raise ValueError("Frontier expected returns must all be finite.")
    if not np.isfinite(frontier_volatilities).all():
        raise ValueError("Frontier volatilities must all be finite.")
    if not np.isfinite(highlighted_values).all():
        raise ValueError("Highlighted portfolio values must all be finite.")
    if np.any(frontier_volatilities < 0.0):
        raise ValueError("Frontier volatilities cannot be negative.")
    if gmv_volatility < 0.0 or maximum_sharpe_volatility < 0.0:
        raise ValueError("Highlighted portfolio volatilities cannot be negative.")

    figure, axes = plt.subplots(figsize=(10, 6))
    axes.plot(
        frontier_volatilities,
        frontier_returns,
        color="#2457A7",
        linewidth=2.5,
        label="Efficient frontier",
    )
    axes.scatter(
        gmv_volatility,
        gmv_return,
        color="#198754",
        edgecolor="white",
        linewidth=1.0,
        s=110,
        zorder=3,
        label="Global minimum variance",
    )
    axes.scatter(
        maximum_sharpe_volatility,
        maximum_sharpe_return,
        color="#D97706",
        edgecolor="white",
        linewidth=1.0,
        marker="*",
        s=220,
        zorder=3,
        label="Maximum Sharpe ratio",
    )

    axes.set_title("Markowitz Efficient Frontier")
    axes.set_xlabel("Annual volatility (risk)")
    axes.set_ylabel("Annual expected return")
    axes.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes.grid(alpha=0.25)
    axes.legend()

    figure.tight_layout()
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return figure_path

