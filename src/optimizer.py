"""Constrained numerical optimization for baseline Markowitz portfolios."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize

from src.portfolio import calculate_portfolio_variance, validate_weights


def optimize_global_minimum_variance(
    covariance_matrix: ArrayLike,
) -> OptimizeResult:
    """Find the fully invested, long-only portfolio with minimum variance."""
    covariance = np.asarray(covariance_matrix, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "Covariance matrix must be square; "
            f"received shape {covariance.shape}."
        )

    number_of_assets = covariance.shape[0]
    if number_of_assets == 0:
        raise ValueError("Covariance matrix must not be empty.")

    initial_weights = np.full(number_of_assets, 1.0 / number_of_assets)

    # This also validates that the covariance matrix is finite and symmetric.
    calculate_portfolio_variance(initial_weights, covariance)
    objective_scale = max(float(np.max(np.abs(covariance))), 1.0e-12)

    def variance_objective(weights: np.ndarray) -> float:
        """Scale w^T Sigma w for numerically consistent solver tolerances."""
        return float(weights.T @ covariance @ weights) / objective_scale

    full_investment_constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
    }
    long_only_bounds = [(0.0, 1.0)] * number_of_assets

    result = minimize(
        fun=variance_objective,
        x0=initial_weights,
        method="SLSQP",
        bounds=long_only_bounds,
        constraints=[full_investment_constraint],
        options={"ftol": 1e-12, "maxiter": 1_000, "disp": False},
    )

    if not result.success:
        raise RuntimeError(
            "Global minimum variance optimization failed: "
            f"{result.message}"
        )

    validate_weights(result.x, number_of_assets)
    # Expose the actual portfolio variance rather than the scaled solver value.
    result.fun = calculate_portfolio_variance(result.x, covariance)
    return result
