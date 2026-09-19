"""Validate model-selected references and the complete rendered explanation."""

from typing import Any

from agent.explainer import render_explanation


MAX_SELECTED_FACTS = 12


def plan_schema(facts: dict[str, str]) -> dict[str, Any]:
    return {
        'type': 'object',
        'properties': {
            'status': {'type': 'string', 'enum': ['supported', 'unsupported']},
            'fact_ids': {'type': 'array', 'items': {'type': 'string', 'enum': list(facts)}},
        },
        'required': ['status', 'fact_ids'], 'additionalProperties': False,
    }


def validate_plan(plan: Any, facts: dict[str, str]) -> dict[str, Any]:
    """Reject prose, unknown IDs, duplicates, missing fields and oversized plans."""
    if not isinstance(plan, dict) or set(plan) != {'status', 'fact_ids'}:
        raise ValueError('invalid_plan_shape')
    if plan['status'] not in ('supported', 'unsupported'):
        raise ValueError('invalid_plan_status')
    ids = plan['fact_ids']
    if not isinstance(ids, list) or len(ids) > MAX_SELECTED_FACTS:
        raise ValueError('invalid_fact_count')
    if any(not isinstance(key, str) or key not in facts for key in ids):
        raise ValueError('unknown_fact')
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate_fact')
    if (plan['status'] == 'supported' and not ids) or (plan['status'] == 'unsupported' and ids):
        raise ValueError('inconsistent_support_status')
    return {'status': plan['status'], 'fact_ids': ids.copy()}


def validate_rendered(result: dict, plan: dict, facts: dict[str, str], metadata: dict) -> None:
    """Exact source rendering catches swapped numbers as well as invented ones.

    Arbitrary model prose is never accepted. Provenance, historical disclaimer,
    profile summaries, and applicable warnings are mandatory renderer content.
    This guarantees grounding in the catalogue, not relevance to every question.
    """
    validated = validate_plan(plan, facts)
    expected = render_explanation(validated, facts, metadata)
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError('rendered_content_mismatch')
