"""Select relative risk profiles from an already-computed efficient frontier."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_variance,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
)


DISCLAIMER = (
    "Figures are historical in-sample estimates, not forecasts or personalized "
    "investment advice. Risk labels are relative to this sampled frontier; "
    "low risk does not mean a safe investment."
)
CONCENTRATION_THRESHOLD = 0.40
MATERIAL_WEIGHT_THRESHOLD = 0.01
MINIMUM_MATERIAL_HOLDINGS = 3


@dataclass(frozen=True)
class ProfileResult:
    """A sampled portfolio, its provenance, and an auditable selection rule."""

    name: str
    expected_return: float
    variance: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float]
    metadata: dict[str, Any]
    selection: dict[str, Any]
    diversification: dict[str, Any]
    warnings: list[str]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy, retaining full precision."""
        return asdict(self)


def select_profiles(payload: dict[str, Any]) -> dict[str, ProfileResult]:
    """Select unchanged frontier candidates without optimizing or using an LLM.

    Medium is nearest to Maximum Sharpe volatility; high is nearest the midpoint
    above Maximum Sharpe, restricted to interior candidates at or above medium.
    Ties choose the earlier index. When no interior candidate exists, high reuses
    medium and explicitly reports the overlap. Endpoints are never substituted
    just to make high distinct. Sharpe uses the baseline evaluator and risk-free
    rate. Invalid or inconsistent payload metrics are rejected, not repaired.
    """
    metadata = payload["metadata"]
    frontier = payload["efficient_frontier"]
    assets = metadata["assets"]
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("Metadata must identify a non-empty, unique asset universe.")
    for field in ("dataset_path", "start_date", "end_date", "trading_days", "constraints"):
        if field not in metadata:
            raise ValueError(f"Missing provenance field: {field}")
    risk_free_rate = metadata["risk_free_rate"]
    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")
    if len(frontier) < 2:
        raise ValueError("At least two sampled frontier points are required.")

    # Explicit ticker lookup prevents dictionary insertion order changing results.
    returns = [payload["annual_expected_returns"][ticker] for ticker in assets]
    covariance = [
        [payload["annual_covariance"][row][column] for column in assets]
        for row in assets
    ]
    for point in frontier:
        if set(point["weights"]) != set(assets):
            raise ValueError("Frontier weights must match the metadata asset universe.")
        weights = [point["weights"][ticker] for ticker in assets]
        calculated = {
            "expected_return": calculate_portfolio_return(weights, returns),
            "variance": calculate_portfolio_variance(weights, covariance),
            "volatility": calculate_portfolio_volatility(weights, covariance),
        }
        for metric, value in calculated.items():
            if not np.isfinite(point[metric]) or not np.isclose(
                point[metric], value, atol=1e-8, rtol=1e-8
            ):
                raise ValueError(f"Frontier {metric} does not match its weights.")
        if point["variance"] < 0 or point["volatility"] <= 0:
            raise ValueError("Profile Sharpe requires positive portfolio volatility.")

    for metric in ("expected_return", "volatility"):
        if np.any(np.diff([point[metric] for point in frontier]) < -1e-8):
            raise ValueError(f"Frontier {metric} must be non-decreasing.")

    maximum_sharpe_volatility = payload["portfolios"]["maximum_sharpe"]["volatility"]
    if not np.isfinite(maximum_sharpe_volatility) or maximum_sharpe_volatility <= 0:
        raise ValueError("Maximum Sharpe volatility must be finite and positive.")
    medium = min(
        range(len(frontier)),
        key=lambda index: abs(frontier[index]["volatility"] - maximum_sharpe_volatility),
    )
    high_target = (maximum_sharpe_volatility + frontier[-1]["volatility"]) / 2
    interior = range(medium, len(frontier) - 1)
    high = min(
        interior,
        key=lambda index: abs(frontier[index]["volatility"] - high_target),
        default=medium,
    )
    indices = {"low": 0, "medium": medium, "high": high}
    rules = {
        "low": "first_frontier_point",
        "medium": "nearest_maximum_sharpe_volatility",
        "high": "nearest_interior_volatility_midpoint_at_or_above_medium",
    }
    profiles = {}
    for name, index in indices.items():
        point = frontier[index]
        weights = [point["weights"][ticker] for ticker in assets]
        concentrated = [
            ticker for ticker in assets
            if point["weights"][ticker] > CONCENTRATION_THRESHOLD
        ]
        material_count = sum(weight > MATERIAL_WEIGHT_THRESHOLD for weight in weights)
        warnings = []
        if concentrated:
            warnings.append("concentration_threshold_exceeded")
        if material_count < MINIMUM_MATERIAL_HOLDINGS:
            warnings.append("few_material_holdings")
        overlaps = [other for other in indices if other != name and indices[other] == index]
        if overlaps:
            warnings.append("overlapping_profiles")
        if index == len(frontier) - 1:
            warnings.append("frontier_endpoint_selected")
        if name == "high" and high == medium:
            warnings.append("no_distinct_higher_interior_selection")
        profiles[name] = ProfileResult(
            name=name,
            expected_return=point["expected_return"],
            variance=point["variance"],
            volatility=point["volatility"],
            sharpe_ratio=calculate_sharpe_ratio(weights, returns, covariance, risk_free_rate),
            weights=deepcopy(point["weights"]),
            metadata=deepcopy(metadata),
            selection={
                "source": "efficient_frontier",
                "frontier_index": index,
                "rule": rules[name],
                "parameters": {
                    "maximum_sharpe_volatility": maximum_sharpe_volatility,
                    "high_target_volatility": high_target,
                    "tie_break": "earlier_frontier_index",
                },
                "overlaps_with": overlaps,
            },
            diversification={
                "concentration_threshold": CONCENTRATION_THRESHOLD,
                "concentrated_assets": concentrated,
                "material_weight_threshold": MATERIAL_WEIGHT_THRESHOLD,
                "minimum_material_holdings": MINIMUM_MATERIAL_HOLDINGS,
                "material_holdings": material_count,
            },
            warnings=warnings,
        )
    return profiles
