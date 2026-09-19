"""Verified facts and deterministic rendering for grounded explanations."""

from copy import deepcopy
from typing import Any

from agent.profiles import DISCLAIMER
from agent.tools import AnalysisTools


RULES = {
    'low': 'Low selects the first sampled frontier point, at global minimum variance.',
    'medium': 'Medium selects the sampled point nearest Maximum Sharpe volatility; it may differ from the exact optimum.',
    'high': 'High selects an interior point nearest the volatility midpoint above Maximum Sharpe, at or above medium; overlapping selections are possible.',
}
WARNINGS = {
    'concentration_threshold_exceeded': 'At least one holding exceeds the concentration threshold.',
    'few_material_holdings': 'Few holdings exceed the material-weight threshold.',
    'overlapping_profiles': 'This option shares a sampled point with another profile.',
    'frontier_endpoint_selected': 'This profile is at the frontier endpoint and may be concentrated in one asset.',
    'no_distinct_higher_interior_selection': 'High reuses medium because no distinct higher interior point was selected.',
}


def _metrics(label: str, point: dict) -> str:
    return (f"{label}: historical annual expected return {point['expected_return']:.4%}; "
            f"annual volatility {point['volatility']:.4%}; Sharpe ratio {point['sharpe_ratio']:.4f}.")


def _weights(label: str, weights: dict) -> str:
    return label + ' target weights: ' + ', '.join(f'{ticker} {weight:.4%}' for ticker, weight in weights.items()) + '.'


def build_facts(tools: AnalysisTools, allocation: dict | None = None) -> dict[str, str]:
    """Only Python-generated facts may enter the final explanation."""
    facts = {}
    for name in ('low', 'medium', 'high'):
        profile = tools.get_profile(name)
        facts[f'profile.{name}.summary'] = _metrics(name.title(), profile)
        facts[f'profile.{name}.selection'] = RULES[name]
        facts[f'profile.{name}.weights'] = _weights(name.title(), profile['weights'])
        if profile['warnings']:
            facts[f'profile.{name}.warnings'] = name.title() + ': ' + ' '.join(
                WARNINGS[code] for code in profile['warnings']
            )
    for ticker in tools.get_metadata()['assets']:
        stats = tools.get_asset_stats(ticker)
        facts[f'asset.{ticker}'] = (
            f"{ticker}: historical annual expected return {stats['annual_expected_return']:.4%}; "
            f"annual volatility {stats['annual_volatility']:.4%}."
        )
    for index in range(tools.frontier_count):
        point = tools.get_frontier_point(index)
        facts[f'frontier.{index}.summary'] = _metrics(f'Frontier point {index} (zero-based)', point)
        facts[f'frontier.{index}.weights'] = _weights(f'Frontier point {index}', point['weights'])
    if allocation is not None:
        display = allocation['display']
        facts['allocation'] = (
            f"{allocation['target_profile']['name'].title()} allocation for "
            f"{allocation['amount']} {allocation['currency']}: "
            + ', '.join(f'{ticker} {value}' for ticker, value in display['amounts'].items())
            + f"; grouped holdings {display['hidden_amount']}; total {display['total']}. "
            + allocation['notice']
        )
    return facts


def mandatory_facts(facts: dict[str, str]) -> list[str]:
    return [f'profile.{name}.summary' for name in ('low', 'medium', 'high')] + [
        key for key in facts if key.endswith('.warnings') or key == 'allocation'
    ]


def default_plan() -> dict[str, Any]:
    return {'status': 'supported', 'fact_ids': [f'profile.{name}.selection' for name in ('low', 'medium', 'high')]}


def render_explanation(plan: dict, facts: dict[str, str], metadata: dict) -> dict:
    """Render only known references, always including provenance and warnings."""
    chosen = plan['fact_ids']
    if (plan.get('status') not in ('supported', 'unsupported') or
            not isinstance(chosen, list) or any(not isinstance(key, str) or key not in facts for key in chosen)):
        raise ValueError('Invalid explanation references.')
    ids = list(dict.fromkeys(mandatory_facts(facts) + chosen))
    constraints = metadata['constraints']
    provenance = (
        f"Dataset: {metadata['start_date']} to {metadata['end_date']}; "
        f"{metadata['trading_days']} trading days per year; annual risk-free rate "
        f"{metadata['risk_free_rate']:.2%}. Constraints: fully invested="
        f"{constraints['fully_invested']}, long-only={constraints['long_only']}, "
        f"leverage allowed={constraints['leverage_allowed']}."
    )
    paragraphs = [provenance]
    if plan['status'] == 'unsupported':
        paragraphs.append('That question is outside the supported historical data. The available profile comparison follows.')
    paragraphs.extend(facts[key] for key in ids)
    paragraphs.append(DISCLAIMER)
    return {'text': '\n\n'.join(paragraphs), 'fact_ids': ids,
            'metadata': deepcopy(metadata), 'disclaimer': DISCLAIMER,
            'status': plan['status']}


def explain_offline(payload: dict, allocation: dict | None = None) -> dict:
    tools = AnalysisTools(payload)
    result = render_explanation(default_plan(), build_facts(tools, allocation), tools.get_metadata())
    return {**result, 'mode': 'offline', 'model': None, 'fallback_reason': None}
