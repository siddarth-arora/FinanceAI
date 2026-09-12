"""Constrained numerical optimization for baseline Markowitz portfolios."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize

from src.portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_variance,
    calculate_sharpe_ratio,
    validate_weights,
)


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


def optimize_maximum_sharpe_ratio(
    expected_returns: ArrayLike,
    covariance_matrix: ArrayLike,
    risk_free_rate: float,
) -> OptimizeResult:
    """Find the fully invested, long-only portfolio with maximum Sharpe ratio."""
    return_vector = np.asarray(expected_returns, dtype=float)
    if return_vector.ndim != 1 or return_vector.size == 0:
        raise ValueError(
            "Expected returns must be a non-empty one-dimensional vector; "
            f"received shape {return_vector.shape}."
        )
    if not np.isfinite(return_vector).all():
        raise ValueError("Expected returns must contain only finite values.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")

    covariance = np.asarray(covariance_matrix, dtype=float)
    number_of_assets = len(return_vector)
    if covariance.shape != (number_of_assets, number_of_assets):
        raise ValueError(
            "Covariance matrix shape must match the expected-return vector; "
            f"received {covariance.shape} for {number_of_assets} assets."
        )

    initial_weights = np.full(number_of_assets, 1.0 / number_of_assets)

    # These calls validate all inputs using the portfolio evaluator's contract.
    calculate_portfolio_return(initial_weights, return_vector)
    calculate_portfolio_variance(initial_weights, covariance)

    def negative_sharpe_objective(weights: np.ndarray) -> float:
        """Return negative Sharpe because SciPy minimizes objectives."""
        portfolio_return = float(weights @ return_vector)
        portfolio_variance = float(weights.T @ covariance @ weights)

        if portfolio_variance <= 0.0:
            return 1.0e12

        portfolio_volatility = float(np.sqrt(portfolio_variance))
        return -((portfolio_return - risk_free_rate) / portfolio_volatility)

    full_investment_constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
    }
    long_only_bounds = [(0.0, 1.0)] * number_of_assets

    result = minimize(
        fun=negative_sharpe_objective,
        x0=initial_weights,
        method="SLSQP",
        bounds=long_only_bounds,
        constraints=[full_investment_constraint],
        options={"ftol": 1e-12, "maxiter": 1_000, "disp": False},
    )

    if not result.success:
        raise RuntimeError(
            "Maximum Sharpe ratio optimization failed: "
            f"{result.message}"
        )

    validate_weights(result.x, number_of_assets)
    calculate_sharpe_ratio(
        result.x,
        return_vector,
        covariance,
        risk_free_rate,
    )
    return result


def optimize_minimum_variance_for_target_return(
    expected_returns: ArrayLike,
    covariance_matrix: ArrayLike,
    target_return: float,
) -> OptimizeResult:
    """Find the long-only minimum-variance portfolio for a target return."""
    return_vector = np.asarray(expected_returns, dtype=float)
    if return_vector.ndim != 1 or return_vector.size == 0:
        raise ValueError(
            "Expected returns must be a non-empty one-dimensional vector; "
            f"received shape {return_vector.shape}."
        )
    if not np.isfinite(return_vector).all():
        raise ValueError("Expected returns must contain only finite values.")
    if not np.isfinite(target_return):
        raise ValueError("Target return must be finite.")

    covariance = np.asarray(covariance_matrix, dtype=float)
    number_of_assets = len(return_vector)
    if covariance.shape != (number_of_assets, number_of_assets):
        raise ValueError(
            "Covariance matrix shape must match the expected-return vector; "
            f"received {covariance.shape} for {number_of_assets} assets."
        )

    minimum_asset_return = float(return_vector.min())
    maximum_asset_return = float(return_vector.max())
    if target_return < minimum_asset_return or target_return > maximum_asset_return:
        raise ValueError(
            f"Target return {target_return:.6%} is infeasible for a long-only "
            f"portfolio; choose a value from {minimum_asset_return:.6%} to "
            f"{maximum_asset_return:.6%}."
        )

    initial_weights = _create_target_return_initial_weights(
        return_vector,
        target_return,
    )

    # These calls validate all inputs using the portfolio evaluator's contract.
    calculate_portfolio_return(initial_weights, return_vector)
    calculate_portfolio_variance(initial_weights, covariance)
    objective_scale = max(float(np.max(np.abs(covariance))), 1.0e-12)

    def variance_objective(weights: np.ndarray) -> float:
        """Scale w^T Sigma w for numerically consistent solver tolerances."""
        return float(weights.T @ covariance @ weights) / objective_scale

    full_investment_constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
    }
    target_return_constraint = {
        "type": "eq",
        "fun": lambda weights: float(weights @ return_vector - target_return),
    }
    long_only_bounds = [(0.0, 1.0)] * number_of_assets

    result = minimize(
        fun=variance_objective,
        x0=initial_weights,
        method="SLSQP",
        bounds=long_only_bounds,
        constraints=[full_investment_constraint, target_return_constraint],
        options={"ftol": 1e-12, "maxiter": 1_000, "disp": False},
    )

    if not result.success:
        raise RuntimeError(
            "Target-return minimum-variance optimization failed for "
            f"{target_return:.6%}: {result.message}"
        )

    validate_weights(result.x, number_of_assets)
    achieved_return = calculate_portfolio_return(result.x, return_vector)
    if not np.isclose(achieved_return, target_return, atol=1e-8, rtol=0.0):
        raise RuntimeError(
            f"Optimizer returned {achieved_return:.6%}, which does not satisfy "
            f"the target return {target_return:.6%}."
        )

    # Expose the actual portfolio variance rather than the scaled solver value.
    result.fun = calculate_portfolio_variance(result.x, covariance)
    return result


def _create_target_return_initial_weights(
    expected_returns: np.ndarray,
    target_return: float,
) -> np.ndarray:
    """Create feasible weights by mixing the lowest- and highest-return assets."""
    number_of_assets = len(expected_returns)
    lowest_index = int(np.argmin(expected_returns))
    highest_index = int(np.argmax(expected_returns))
    lowest_return = float(expected_returns[lowest_index])
    highest_return = float(expected_returns[highest_index])

    if np.isclose(lowest_return, highest_return):
        return np.full(number_of_assets, 1.0 / number_of_assets)

    high_weight = (target_return - lowest_return) / (
        highest_return - lowest_return
    )
    initial_weights = np.zeros(number_of_assets)
    initial_weights[lowest_index] = 1.0 - high_weight
    initial_weights[highest_index] = high_weight
    return initial_weights
