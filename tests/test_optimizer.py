"""Check feasible solutions and efficient-frontier behavior on synthetic inputs."""

import numpy as np
import pytest

from src.optimizer import (
    generate_efficient_frontier,
    optimize_global_minimum_variance,
    optimize_maximum_sharpe_ratio,
    optimize_minimum_variance_for_target_return,
)
from src.portfolio import calculate_sharpe_ratio, validate_weights


def test_gmv_matches_closed_form_for_diagonal_covariance():
    covariance = np.diag([0.01, 0.04, 0.09])
    result = optimize_global_minimum_variance(covariance)
    expected = 1 / np.diag(covariance)
    expected /= expected.sum()
    validate_weights(result.x, 3)
    np.testing.assert_allclose(result.x, expected, atol=1e-6)


def test_frontier_targets_ordering_and_maximum_sharpe(synthetic_market):
    returns, covariance = synthetic_market
    frontier = generate_efficient_frontier(returns, covariance, 15)
    maximum = optimize_maximum_sharpe_ratio(returns, covariance, 0.04)
    maximum_sharpe = calculate_sharpe_ratio(maximum.x, returns, covariance, 0.04)
    assert np.all(np.diff([p.expected_return for p in frontier]) > 0)
    assert np.all(np.diff([p.volatility for p in frontier]) >= -1e-8)
    for point in frontier:
        validate_weights(point.weights, 3)
        assert point.expected_return == pytest.approx(point.target_return, abs=1e-8)
        assert calculate_sharpe_ratio(point.weights, returns, covariance, 0.04) <= maximum_sharpe + 1e-6


@pytest.mark.parametrize("target", [-0.01, 0.30])
def test_infeasible_target_raises(synthetic_market, target):
    with pytest.raises(ValueError, match="infeasible"):
        optimize_minimum_variance_for_target_return(*synthetic_market, target)
