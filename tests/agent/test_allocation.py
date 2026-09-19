"""Allocation precision, validation and preservation of the profile contract."""

from copy import deepcopy
from decimal import Decimal, localcontext
import json
import socket

import pytest

from agent.allocation import allocate_amount, validate_allocation_input
from agent.profiles import select_profiles


@pytest.mark.parametrize("name", ["low", "medium", "high"])
def test_allocation_preserves_profile_and_exact_products(payload, name):
    before = deepcopy(payload)
    result = allocate_amount("1234.56", "USD", payload, name)
    assert result.target_profile.to_dict() == select_profiles(payload)[name].to_dict()
    with localcontext() as context:
        context.prec = 410
        expected = {
            ticker: Decimal("1234.56") * Decimal(str(weight))
            for ticker, weight in result.target_profile.weights.items()
        }
        assert result.target_amounts == expected
        assert sum(expected.values()) + result.numerical_residual == result.amount
    assert payload == before
    serialized = result.to_dict()
    assert json.loads(json.dumps(serialized, allow_nan=False)) == serialized
    assert serialized["target_profile"]["metadata"] == payload["metadata"]
    assert serialized["target_profile"]["disclaimer"]


@pytest.mark.parametrize("amount", [True, False, None, [], "", "abc", "NaN",
                                   "Infinity", float("inf"), "-1", 0, "0.001",
                                   "1000000000000"])
def test_invalid_amount_is_rejected_before_reading_payload(amount):
    with pytest.raises(ValueError, match="Amount"):
        allocate_amount(amount, "USD", {})


@pytest.mark.parametrize("currency", [None, "", "US", "XYZ", "EUR", 123])
def test_unsupported_currency_is_rejected(currency):
    with pytest.raises(ValueError, match="Currency"):
        allocate_amount("10", currency, {})


@pytest.mark.parametrize("amount", [10, 10.0, "1e1", Decimal("10.000")])
def test_supported_amount_types_and_currency_normalization(amount):
    assert validate_allocation_input(amount, " inr ") == (Decimal("10.00"), "INR")


def test_unknown_profile_is_rejected(payload):
    with pytest.raises(ValueError, match="Profile"):
        allocate_amount("10", "USD", payload, "unknown")


def test_currency_is_a_label_without_conversion(payload):
    usd = allocate_amount("10", "USD", payload)
    inr = allocate_amount("10", "INR", payload)
    assert usd.target_amounts == inr.target_amounts
    assert usd.target_profile.to_dict() == inr.target_profile.to_dict()


def test_changed_weights_with_stale_metrics_are_rejected(payload):
    payload["efficient_frontier"][0]["weights"] = {"A": 0.2, "B": 0.3, "C": 0.5}
    with pytest.raises(ValueError, match="does not match its weights"):
        allocate_amount("10", "USD", payload)


def test_outputs_are_detached_from_payload_and_serialization(payload):
    result = allocate_amount("10", "USD", payload)
    encoded = result.to_dict()
    encoded["target_profile"]["weights"].clear()
    encoded["target_amounts"].clear()
    assert result.target_amounts
    assert result.target_profile.weights
    result.target_profile.metadata["constraints"].clear()
    assert payload["metadata"]["constraints"]


def test_allocation_is_offline_and_independent_of_decimal_context(payload, monkeypatch):
    def forbid_network(*args, **kwargs):
        pytest.fail("Allocation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    expected = allocate_amount("999999999999.99", "USD", payload).to_dict()
    with localcontext() as context:
        context.prec = 6
        actual = allocate_amount("999999999999.99", "USD", payload).to_dict()
    assert actual == expected
