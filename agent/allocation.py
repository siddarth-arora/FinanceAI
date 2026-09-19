"""Convert sampled profile weights to monetary targets without changing them."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from agent.profiles import ProfileResult, select_profiles


SUPPORTED_CURRENCIES = ("USD", "INR")
MINOR_UNIT = Decimal("0.01")


@dataclass(frozen=True)
class AllocationResult:
    """Exact decimal monetary targets and the unchanged source profile.

    Money serializes as decimal strings, while profile metrics and weights stay
    numeric decimals. numerical_residual records weight-sum floating-point drift;
    it is not leftover cash from a trade. No shares are bought by this operation.
    """

    amount: Decimal
    currency: str
    target_profile: ProfileResult
    target_amounts: dict[str, Decimal]
    numerical_residual: Decimal

    def to_dict(self) -> dict[str, Any]:
        """Return detached JSON data without rounding the monetary targets."""
        return {
            "mode": "fractional",
            "amount": format(self.amount, "f"),
            "currency": self.currency,
            "target_profile": self.target_profile.to_dict(),
            "target_amounts": {
                ticker: format(value, "f")
                for ticker, value in self.target_amounts.items()
            },
            "numerical_residual": format(self.numerical_residual, "f"),
            "notice": (
                "Target monetary allocations only; no trades, share counts, "
                "live prices, or currency conversion. Portfolio metrics describe "
                "the unchanged target weights."
            ),
        }


def validate_allocation_input(
    amount: str | int | float | Decimal, currency: str,
) -> tuple[Decimal, str]:
    """Validate a positive amount in whole minor units and a currency label."""
    if isinstance(amount, bool) or not isinstance(amount, (str, int, float, Decimal)):
        raise ValueError("Amount must be a finite positive decimal number.")
    try:
        value = Decimal(str(amount))
    except InvalidOperation as error:
        raise ValueError("Amount must be a finite positive decimal number.") from error
    if not value.is_finite() or value <= 0:
        raise ValueError("Amount must be a finite positive decimal number.")
    # Bound the supported input size, not the financial portfolio assumptions.
    # This keeps exact decimal calculations and downstream display manageable.
    if value > Decimal("999999999999.99"):
        raise ValueError("Amount must not exceed 999999999999.99.")
    with localcontext() as context:
        context.prec = max(28, len(value.as_tuple().digits) + 2)
        if value != value.quantize(MINOR_UNIT):
            raise ValueError("Amount must have at most two decimal places.")
        value = value.quantize(MINOR_UNIT)
    if not isinstance(currency, str) or currency.strip().upper() not in SUPPORTED_CURRENCIES:
        raise ValueError("Currency must be USD or INR; no conversion is performed.")
    return value, currency.strip().upper()


def allocate_amount(
    amount: str | int | float | Decimal,
    currency: str,
    payload: dict[str, Any],
    profile_name: str = "medium",
) -> AllocationResult:
    """Allocate to a profile selected and validated from the analysis payload.

    Accepting the analysis payload lets select_profiles verify the weights and
    their metrics rather than trusting an externally modified ProfileResult.
    Decimal products use the full round-trip string representation of each
    baseline float. Never normalize, round, or hide the target weights here.
    """
    value, currency = validate_allocation_input(amount, currency)
    if profile_name not in ("low", "medium", "high"):
        raise ValueError("Profile must be low, medium, or high.")
    profile = select_profiles(payload)[profile_name]
    # A binary64 weight's decimal string can reach 324 fractional places.
    # This covers exact products and sums, independent of the caller's context.
    with localcontext() as context:
        context.prec = 400 + len(str(len(profile.weights)))
        targets = {
            ticker: value * Decimal(str(weight))
            for ticker, weight in profile.weights.items()
        }
        residual = value - sum(targets.values(), Decimal(0))
    return AllocationResult(value, currency, profile, targets, residual)
