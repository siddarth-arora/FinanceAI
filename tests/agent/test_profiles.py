"""Profile selection never invents weights, loses provenance, or hides overlap."""

from copy import deepcopy
import json

import numpy as np
import pytest

from agent.profiles import select_profiles
from src.portfolio import calculate_sharpe_ratio


def test_profiles_are_ordered_unchanged_candidates_with_provenance(payload, synthetic_market):
    before = deepcopy(payload)
    profiles = select_profiles(payload)
    assert payload == before
    assert list(profiles) == ["low", "medium", "high"]
    for metric in ("expected_return", "volatility"):
        values = [getattr(profile, metric) for profile in profiles.values()]
        assert values == sorted(values)
    for profile in profiles.values():
        point = payload["efficient_frontier"][profile.selection["frontier_index"]]
        assert profile.weights == point["weights"]
        assert profile.expected_return == point["expected_return"]
        assert profile.variance == point["variance"]
        assert profile.volatility == point["volatility"]
        assert profile.metadata == payload["metadata"]
        assert profile.sharpe_ratio == pytest.approx(calculate_sharpe_ratio(
            list(profile.weights.values()), *synthetic_market, 0.04,
        ))
        assert "not forecasts or personalized investment advice" in profile.disclaimer
        assert json.loads(json.dumps(profile.to_dict(), allow_nan=False)) == profile.to_dict()
    assert profiles["low"].selection["frontier_index"] == 0
    assert profiles["high"].selection["frontier_index"] < len(payload["efficient_frontier"]) - 1


def test_selection_uses_documented_volatility_targets(payload):
    profiles = select_profiles(payload)
    points = payload["efficient_frontier"]
    maximum = payload["portfolios"]["maximum_sharpe"]["volatility"]
    expected_medium = min(range(len(points)), key=lambda i: abs(points[i]["volatility"] - maximum))
    midpoint = (maximum + points[-1]["volatility"]) / 2
    expected_high = min(range(expected_medium, len(points) - 1),
                        key=lambda i: abs(points[i]["volatility"] - midpoint))
    assert profiles["medium"].selection["frontier_index"] == expected_medium
    assert profiles["high"].selection["frontier_index"] == expected_high


def test_dictionary_order_does_not_change_sharpe(payload):
    expected = select_profiles(payload)
    payload["metadata"]["assets"].reverse()
    for point in payload["efficient_frontier"]:
        point["weights"] = dict(reversed(list(point["weights"].items())))
    actual = select_profiles(payload)
    for name in expected:
        assert actual[name].weights == expected[name].weights
        assert actual[name].sharpe_ratio == pytest.approx(expected[name].sharpe_ratio)


def test_sharpe_uses_payload_risk_free_rate(payload, synthetic_market):
    payload["metadata"]["risk_free_rate"] = 0.07
    profiles = select_profiles(payload)
    for profile in profiles.values():
        assert profile.sharpe_ratio == pytest.approx(calculate_sharpe_ratio(
            list(profile.weights.values()), *synthetic_market, 0.07,
        ))
        assert profile.metadata["risk_free_rate"] == 0.07


def test_equal_distance_ties_choose_earlier_point(payload):
    first = payload["efficient_frontier"][0]
    last = payload["efficient_frontier"][-1]
    payload["efficient_frontier"] = [first, deepcopy(first), last]
    payload["portfolios"]["maximum_sharpe"]["volatility"] = first["volatility"]
    profiles = select_profiles(payload)
    assert profiles["medium"].selection["frontier_index"] == 0
    assert profiles["high"].selection["frontier_index"] == 0


def test_missing_provenance_is_rejected(payload):
    del payload["metadata"]["start_date"]
    with pytest.raises(ValueError, match="Missing provenance field: start_date"):
        select_profiles(payload)


def test_profile_and_serialized_mutations_do_not_leak(payload):
    profiles = select_profiles(payload)
    encoded = profiles["low"].to_dict()
    encoded["metadata"]["assets"].clear()
    encoded["weights"].clear()
    assert profiles["low"].weights
    profiles["low"].metadata["constraints"].clear()
    profiles["low"].weights.clear()
    assert profiles["medium"].metadata["constraints"]
    assert payload["metadata"]["constraints"]
    assert payload["efficient_frontier"][0]["weights"]


def test_sparse_frontier_reuses_medium_instead_of_forcing_endpoint(payload):
    payload["efficient_frontier"] = [payload["efficient_frontier"][0],
                                     payload["efficient_frontier"][-1]]
    payload["portfolios"]["maximum_sharpe"]["volatility"] = payload["efficient_frontier"][0]["volatility"]
    profiles = select_profiles(payload)
    assert all(profile.selection["frontier_index"] == 0 for profile in profiles.values())
    assert "overlapping_profiles" in profiles["low"].warnings
    assert "no_distinct_higher_interior_selection" in profiles["high"].warnings


def test_endpoint_maximum_sharpe_is_flagged_and_not_reweighted(payload):
    endpoint = payload["efficient_frontier"][-1]
    payload["portfolios"]["maximum_sharpe"]["volatility"] = endpoint["volatility"]
    profiles = select_profiles(payload)
    for name in ("medium", "high"):
        profile = profiles[name]
        assert profile.weights == endpoint["weights"]
        assert "frontier_endpoint_selected" in profile.warnings
        assert "concentration_threshold_exceeded" in profile.warnings
        assert "few_material_holdings" in profile.warnings


@pytest.mark.parametrize("invalid", ["empty", "short", "nan", "weights", "ticker", "metric", "order"])
def test_invalid_frontier_is_rejected(payload, invalid):
    points = payload["efficient_frontier"]
    if invalid == "empty":
        points.clear()
    elif invalid == "short":
        del points[1:]
    elif invalid == "nan":
        points[0]["volatility"] = np.nan
    elif invalid == "weights":
        points[0]["weights"]["A"] = -0.5
    elif invalid == "ticker":
        points[0]["weights"]["UNKNOWN"] = points[0]["weights"].pop("A")
    elif invalid == "metric":
        points[0]["expected_return"] += 0.1
    else:
        points.reverse()
    with pytest.raises(ValueError):
        select_profiles(payload)


def test_zero_volatility_is_rejected(payload):
    for row in payload["annual_covariance"].values():
        for ticker in row:
            row[ticker] = 0.0
    for point in payload["efficient_frontier"]:
        point["variance"] = point["volatility"] = 0.0
    with pytest.raises(ValueError, match="positive portfolio volatility"):
        select_profiles(payload)
