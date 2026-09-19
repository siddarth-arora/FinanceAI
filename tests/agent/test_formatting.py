"""Check monetary reconciliation against hand-calculated rounding cases."""

from decimal import Decimal, localcontext

import pytest

from agent.allocation import allocate_amount
from agent.formatting import format_allocation_display


def test_largest_remainder_and_alphabetical_tie_break():
    targets = {"C": Decimal("0.333333"), "B": Decimal("0.333333"), "A": Decimal("0.333333")}
    display = format_allocation_display(Decimal("1.00"), targets)
    assert display["amounts"] == {"A": "0.34", "B": "0.33", "C": "0.33"}
    assert display["total"] == "1.00"
    reverse = format_allocation_display(Decimal("1.00"), dict(reversed(list(targets.items()))))
    assert reverse == display


def test_largest_remainder_prioritizes_fraction_over_ticker():
    display = format_allocation_display(
        Decimal("0.01"), {"A": Decimal("0.002"), "B": Decimal("0.008")},
    )
    assert display["amounts"] == {"A": "0.00", "B": "0.01"}


@pytest.mark.parametrize("amount", ["0.01", "1.00", "1234.56", "999999999999.99"])
@pytest.mark.parametrize("name", ["low", "medium", "high"])
def test_display_totals_reconcile_for_all_profiles_and_amount_sizes(payload, amount, name):
    result = allocate_amount(amount, "INR", payload, name)
    before = result.to_dict()
    display = before["display"]
    assert sum(Decimal(value) for value in display["amounts"].values()) == Decimal(amount)
    assert all(Decimal(value) >= 0 for value in display["amounts"].values())
    with localcontext() as context:
        context.prec = 410
        for ticker, target in result.target_amounts.items():
            assert target + Decimal(display["rounding_adjustments"][ticker]) == Decimal(display["amounts"][ticker])
    assert result.to_dict() == before


def test_hiding_dust_keeps_the_money_and_threshold_boundary():
    targets = {"A": Decimal("99.01"), "B": Decimal("0.49"), "C": Decimal("0.50")}
    display = format_allocation_display(Decimal("100.00"), targets, hide_dust=True)
    assert display["amounts"] == {"A": "99.01", "C": "0.50"}
    assert display["hidden_assets"] == ["B"]
    assert display["hidden_amount"] == "0.49"
    assert sum(Decimal(v) for v in display["amounts"].values()) + Decimal(display["hidden_amount"]) == Decimal("100")


def test_weight_sum_drift_and_tiny_negative_targets_are_display_only():
    targets = {"A": Decimal("100.00000001"), "B": Decimal("-0.000000001")}
    before = targets.copy()
    display = format_allocation_display(Decimal("100"), targets)
    assert display["amounts"] == {"A": "100.00", "B": "0.00"}
    assert targets == before
    assert display["rounding_adjustments"] == {"A": "-0.00000001", "B": "0.000000001"}


def test_display_is_independent_of_global_decimal_precision():
    targets = {"A": Decimal("500000000000.001"), "B": Decimal("499999999999.989")}
    expected = format_allocation_display(Decimal("999999999999.99"), targets)
    with localcontext() as context:
        context.prec = 6
        assert format_allocation_display(Decimal("999999999999.99"), targets) == expected
