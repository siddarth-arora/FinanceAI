"""Protect the historical arithmetic-return and sample-covariance contract."""

import numpy as np
import pandas as pd
import pytest

from src.returns import (
    calculate_covariance_matrices,
    calculate_daily_returns,
    calculate_expected_returns,
)


def test_simple_returns_and_arithmetic_annualization():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    daily = calculate_daily_returns(prices)
    np.testing.assert_allclose(daily["A"], [0.1, -0.1])
    mean, annual = calculate_expected_returns(daily)
    assert mean["A"] == pytest.approx(0.0)
    assert annual["A"] == pytest.approx(0.0)
    _, annual = calculate_expected_returns(pd.DataFrame({"A": [0.01, 0.03]}))
    assert annual["A"] == pytest.approx(0.02 * 252)


def test_covariance_uses_sample_denominator_and_252_days():
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.03, 0.01, 0.02]})
    daily, annual = calculate_covariance_matrices(returns)
    expected = np.array([[0.0001, -0.00005], [-0.00005, 0.0001]])
    np.testing.assert_allclose(daily, expected)
    np.testing.assert_allclose(annual, expected * 252)
    np.testing.assert_allclose(annual, annual.T)


@pytest.mark.parametrize("function", [calculate_daily_returns, calculate_expected_returns,
                                     calculate_covariance_matrices])
def test_missing_values_are_rejected(function):
    with pytest.raises(ValueError, match="missing"):
        function(pd.DataFrame({"A": [1.0, np.nan, 2.0]}))
