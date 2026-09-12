"""Pure mathematical functions for evaluating a portfolio."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


WEIGHT_TOLERANCE = 1e-8


def validate_weights(weights: ArrayLike, number_of_assets: int) -> NDArray[np.float64]:
    """Validate fully invested, long-only portfolio weights."""
    weight_vector = _as_vector(weights, "Weights")

    if len(weight_vector) != number_of_assets:
        raise ValueError(
            f"Expected {number_of_assets} weights, received {len(weight_vector)}."
        )
    if np.any(weight_vector < -WEIGHT_TOLERANCE):
        raise ValueError("Weights must be non-negative; short selling is not allowed.")
    if np.any(weight_vector > 1.0 + WEIGHT_TOLERANCE):
        raise ValueError("Individual weights cannot exceed 1; leverage is not allowed.")
    if not np.isclose(
        weight_vector.sum(), 1.0, atol=WEIGHT_TOLERANCE, rtol=0.0
    ):
        raise ValueError(
            f"Weights must sum to 1; received {weight_vector.sum():.12f}."
        )

    return weight_vector


def calculate_portfolio_return(
    weights: ArrayLike,
    expected_returns: ArrayLike,
) -> float:
    """Calculate portfolio expected return using R_p = w^T mu."""
    return_vector = _as_vector(expected_returns, "Expected returns")
    weight_vector = validate_weights(weights, len(return_vector))

    # (n,) @ (n,) -> scalar, which is the dot product w^T mu.
    return float(weight_vector @ return_vector)


def calculate_portfolio_variance(
    weights: ArrayLike,
    covariance_matrix: ArrayLike,
) -> float:
    """Calculate portfolio variance using sigma_p^2 = w^T Sigma w."""
    covariance = _as_covariance_matrix(covariance_matrix)
    weight_vector = validate_weights(weights, covariance.shape[0])

    # (n,) @ (n, n) @ (n,) -> scalar, representing w^T Sigma w.
    variance = float(weight_vector.T @ covariance @ weight_vector)

    if variance < -1e-12:
        raise ValueError("Portfolio variance cannot be negative.")
    return max(variance, 0.0)


def calculate_portfolio_volatility(
    weights: ArrayLike,
    covariance_matrix: ArrayLike,
) -> float:
    """Calculate portfolio volatility as the square root of variance."""
    variance = calculate_portfolio_variance(weights, covariance_matrix)
    return float(np.sqrt(variance))


def calculate_sharpe_ratio(
    weights: ArrayLike,
    expected_returns: ArrayLike,
    covariance_matrix: ArrayLike,
    risk_free_rate: float,
) -> float:
    """Calculate Sharpe = (portfolio return - risk-free rate) / volatility."""
    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")

    portfolio_return = calculate_portfolio_return(weights, expected_returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)

    if np.isclose(portfolio_volatility, 0.0):
        raise ValueError("Sharpe ratio is undefined for zero-volatility portfolios.")

    return float((portfolio_return - risk_free_rate) / portfolio_volatility)


def _as_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert an array-like input to a finite one-dimensional float vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received shape {vector.shape}.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _as_covariance_matrix(values: ArrayLike) -> NDArray[np.float64]:
    """Convert and validate a finite, symmetric, square covariance matrix."""
    covariance = np.asarray(values, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "Covariance matrix must be square; "
            f"received shape {covariance.shape}."
        )
    if covariance.shape[0] == 0:
        raise ValueError("Covariance matrix must not be empty.")
    if not np.isfinite(covariance).all():
        raise ValueError("Covariance matrix must contain only finite values.")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Covariance matrix must be symmetric.")
    return covariance

