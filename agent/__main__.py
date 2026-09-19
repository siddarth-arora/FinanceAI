"""Inspect profiles, allocations and optional grounded explanations."""

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
    parser.add_argument("--explain", action="store_true", help="Include an offline explanation")
    parser.add_argument("--llm", action="store_true", help="With --explain, enable Groq API calls")
    parser.add_argument("--question", help="With --explain --llm, ask about existing historical results")
    args = parser.parse_args(argv)
    if (args.llm or args.question is not None) and not args.explain:
        parser.error("--llm and --question require --explain")
    if args.question is not None and not args.llm:
        parser.error("--question requires --llm; offline mode provides a general comparison")
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

    if args.explain:
        from agent.graph import run_agent, validate_question
        try:
            validate_question(args.question)
        except ValueError as error:
            parser.error(str(error))
        output = run_agent(amount=args.amount, currency=currency,
                           profile=args.profile or "medium", hide_dust=args.hide_dust,
                           question=args.question, use_llm=args.llm)
    elif args.amount is None:
        payload = run_mpt_analysis().to_dict()
        profiles = select_profiles(payload)
        output = {name: profile.to_dict() for name, profile in profiles.items()}
    else:
        payload = run_mpt_analysis().to_dict()
        allocation = allocate_amount(
            args.amount, currency, payload, args.profile or "medium",
        )
        output = allocation.to_dict(hide_dust=args.hide_dust)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
