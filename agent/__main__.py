"""Inspect deterministic profiles using the frozen local dataset."""

import json

from agent.profiles import select_profiles
from src.pipeline import run_mpt_analysis


def main() -> None:
    payload = run_mpt_analysis().to_dict()
    profiles = select_profiles(payload)
    print(json.dumps(
        {name: profile.to_dict() for name, profile in profiles.items()},
        indent=2,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
