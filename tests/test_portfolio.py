"""Unit tests for the fundamental portfolio mathematics."""

import numpy as np
import pytest

from src.portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_variance,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    validate_weights,
)


WEIGHTS = np.array([0.25, 0.75])
EXPECTED_RETURNS = np.array([0.08, 0.12])
COVARIANCE_MATRIX = np.array(
    [
        [0.04, 0.006],
        [0.006, 0.09],
    ]
)


def test_valid_weights_sum_to_one() -> None:
    validated = validate_weights(WEIGHTS, number_of_assets=2)

    assert validated.sum() == pytest.approx(1.0)


def test_weights_with_invalid_sum_are_rejected() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_weights([0.25, 0.50], number_of_assets=2)


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="short selling is not allowed"):
        validate_weights([-0.10, 1.10], number_of_assets=2)


def test_incorrect_number_of_weights_is_rejected() -> None:
    with pytest.raises(ValueError, match="Expected 2 weights"):
        validate_weights([1.0], number_of_assets=2)


def test_portfolio_return_matches_manual_calculation() -> None:
    manual_return = (0.25 * 0.08) + (0.75 * 0.12)

    calculated_return = calculate_portfolio_return(WEIGHTS, EXPECTED_RETURNS)

    assert manual_return == pytest.approx(0.11)
    assert calculated_return == pytest.approx(manual_return)


def test_portfolio_variance_matches_manual_calculation() -> None:
    manual_variance = (
        (0.25**2 * 0.04)
        + (2 * 0.25 * 0.75 * 0.006)
        + (0.75**2 * 0.09)
    )

    calculated_variance = calculate_portfolio_variance(
        WEIGHTS,
        COVARIANCE_MATRIX,
    )

    assert manual_variance == pytest.approx(0.055375)
    assert calculated_variance == pytest.approx(manual_variance)


def test_portfolio_volatility_is_square_root_of_variance() -> None:
    variance = calculate_portfolio_variance(WEIGHTS, COVARIANCE_MATRIX)
    volatility = calculate_portfolio_volatility(WEIGHTS, COVARIANCE_MATRIX)

    assert volatility == pytest.approx(np.sqrt(variance))


def test_sharpe_ratio_matches_manual_calculation() -> None:
    risk_free_rate = 0.03
    expected_return = 0.11
    volatility = np.sqrt(0.055375)
    manual_sharpe_ratio = (expected_return - risk_free_rate) / volatility

    calculated_sharpe_ratio = calculate_sharpe_ratio(
        WEIGHTS,
        EXPECTED_RETURNS,
        COVARIANCE_MATRIX,
        risk_free_rate,
    )

    assert calculated_sharpe_ratio == pytest.approx(manual_sharpe_ratio)


def test_non_symmetric_covariance_matrix_is_rejected() -> None:
    non_symmetric_covariance = np.array(
        [
            [0.04, 0.01],
            [0.02, 0.09],
        ]
    )

    with pytest.raises(ValueError, match="must be symmetric"):
        calculate_portfolio_variance(WEIGHTS, non_symmetric_covariance)

