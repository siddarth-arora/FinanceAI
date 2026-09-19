"""Presentation-only monetary rounding; target portfolio weights never change."""

from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any


def format_allocation_display(
    amount: Decimal,
    target_amounts: dict[str, Decimal],
    *,
    hide_dust: bool = False,
) -> dict[str, Any]:
    """Apportion whole cents/paise, retaining an audit of display adjustments.

    Inputs are validated allocation targets. Clamp tolerance-level negative
    targets and normalize numerical drift ONLY in the display proportions.
    Largest remainder assigns spare minor units; ticker order breaks ties.
    Exact rational arithmetic makes this independent of decimal context.
    """
    budget = Fraction(amount) * 100
    if budget.denominator != 1 or budget <= 0:
        raise ValueError("Display amount must be positive and in whole minor units.")
    positive = {ticker: max(Fraction(value), 0) for ticker, value in target_amounts.items()}
    total = sum(positive.values())
    if not total:
        raise ValueError("Display targets must have a positive total.")
    quotas = {ticker: budget * value / total for ticker, value in positive.items()}
    units = {ticker: quota.numerator // quota.denominator for ticker, quota in quotas.items()}
    remaining = int(budget) - sum(units.values())
    ranked = sorted(units, key=lambda ticker: (-(quotas[ticker] - units[ticker]), ticker))
    for ticker in ranked[:remaining]:
        units[ticker] += 1

    def money(minor_units: int) -> str:
        return f"{minor_units // 100}.{minor_units % 100:02d}"

    hidden = sorted(
        ticker for ticker, value in positive.items()
        if hide_dust and value / Fraction(amount) < Fraction(1, 200)
    )
    with localcontext() as context:
        context.prec = 400 + len(str(len(units)))
        adjustments = {
            ticker: format(Decimal(money(units[ticker])) - target, "f")
            for ticker, target in target_amounts.items()
        }
    return {
        "amounts": {ticker: money(value) for ticker, value in units.items() if ticker not in hidden},
        "hidden_assets": hidden,
        "hidden_amount": money(sum(units[ticker] for ticker in hidden)),
        "total": money(sum(units.values())),
        "dust_threshold": 0.005 if hide_dust else 0.0,
        "rounding_adjustments": adjustments,
        "method": "largest_remainder_with_alphabetical_ties",
        "notice": (
            "Display only: non-negative targets are proportionally reconciled "
            "to the input amount, then rounded to cents/paise. Adjustments do "
            "not change target weights or their metrics. Hidden holdings are "
            "included in hidden_amount; they are not cash."
        ),
    }
