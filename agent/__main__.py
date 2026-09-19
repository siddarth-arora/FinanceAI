"""Inspect deterministic profiles and allocations using the frozen dataset."""

import argparse
import json

from agent.allocation import allocate_amount, validate_allocation_input
from agent.profiles import select_profiles
from src.pipeline import run_mpt_analysis


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amount", help="Positive monetary amount, at most two decimals")
    parser.add_argument("--currency", help="USD or INR label; no FX conversion (default: USD)")
    parser.add_argument(
        "--profile", choices=("low", "medium", "high"),
        help="Allocation profile (default: medium)",
    )
    parser.add_argument(
        "--hide-dust", action="store_true",
        help="Group display holdings below 0.5%%; retain exact targets",
    )
    args = parser.parse_args(argv)
    allocation_options = (
        args.currency is not None or args.profile is not None or args.hide_dust
    )
    if args.amount is None and allocation_options:
        parser.error("--currency, --profile and --hide-dust require --amount")
    currency = "USD" if args.currency is None else args.currency
    if args.amount is not None:
        try:
            validate_allocation_input(args.amount, currency)
        except ValueError as error:
            parser.error(str(error))

    payload = run_mpt_analysis().to_dict()
    if args.amount is None:
        profiles = select_profiles(payload)
        output = {name: profile.to_dict() for name, profile in profiles.items()}
    else:
        allocation = allocate_amount(
            args.amount, currency, payload, args.profile or "medium",
        )
        output = allocation.to_dict(hide_dust=args.hide_dust)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
