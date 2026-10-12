"""Validate model-selected references and the complete rendered explanation."""

import re
from typing import Any

from agent.explainer import render_explanation


MAX_SELECTED_FACTS = 12
# A hard bound enforced here, not in the schema: Groq cannot constrain length
# while decoding, so a schema maxLength made it reject the whole plan after the
# fact. Here an overlong answer costs only the prose. The prompt asks for less.
MAX_ANSWER_CHARS = 2500
# Any digit run, with optional decimals and separators. Deliberately greedy: a
# figure the model states must be checkable, whatever it is dressed in.
NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?')
# Failures that condemn only the prose, never the fact selection around it.
ANSWER_REJECTIONS = frozenset({'invalid_answer_type', 'invalid_answer_length',
                               'ungrounded_answer_number'})


def plan_schema(facts: dict[str, str], *, with_answer: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        'status': {'type': 'string', 'enum': ['supported', 'unsupported']},
        'fact_ids': {'type': 'array', 'items': {'type': 'string', 'enum': list(facts)}},
    }
    if with_answer:
        properties['answer'] = {'type': 'string'}
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def _numbers(text: str) -> set[str]:
    """Numeric tokens normalized so 1,234.50 and 1234.5 compare equal."""
    found = set()
    for raw in NUMBER.findall(text):
        cleaned = raw.replace(',', '')
        if '.' in cleaned:
            cleaned = cleaned.rstrip('0').rstrip('.')
        found.add(cleaned or '0')
    return found


def validate_answer(answer: Any, facts: dict[str, str], cited: list[str]) -> str:
    """Accept model prose only when every figure in it is one of the cited facts'.

    The model may explain, simplify and choose its own words; it may not
    introduce a number. Grounding is checked against the facts the plan actually
    cites, so prose cannot borrow a figure the reader is never shown. A failure
    raises, and the caller falls back to the facts alone rather than publishing
    an unverifiable claim.
    """
    if not isinstance(answer, str):
        raise ValueError('invalid_answer_type')
    text = answer.strip()
    if not text or len(text) > MAX_ANSWER_CHARS:
        raise ValueError('invalid_answer_length')
    allowed = _numbers(' '.join(facts[key] for key in cited))
    unsupported = sorted(_numbers(text) - allowed)
    if unsupported:
        raise ValueError('ungrounded_answer_number')
    return text


def validate_plan(plan: Any, facts: dict[str, str]) -> dict[str, Any]:
    """Reject unknown IDs, duplicates, missing fields, oversized plans and
    ungrounded prose. An `answer` is optional; when present it is number-checked."""
    if not isinstance(plan, dict) or not {'status', 'fact_ids'} <= set(plan) <= {'status', 'fact_ids', 'answer'}:
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
    validated = {'status': plan['status'], 'fact_ids': ids.copy()}
    if plan.get('answer') is not None and str(plan.get('answer')).strip():
        validated['answer'] = validate_answer(plan['answer'], facts, validated['fact_ids'])
    return validated


def validate_rendered(result: dict, plan: dict, facts: dict[str, str], metadata: dict,
                      question_mode: bool = False) -> None:
    """Exact source rendering catches swapped numbers as well as invented ones.

    Arbitrary model prose is never accepted. Provenance, historical disclaimer,
    profile summaries, and applicable warnings are mandatory renderer content.
    This guarantees grounding in the catalogue, not relevance to every question.
    """
    validated = validate_plan(plan, facts)
    expected = render_explanation(validated, facts, metadata, question_mode)
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError('rendered_content_mismatch')
