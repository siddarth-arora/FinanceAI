from copy import deepcopy
import json
from pathlib import Path

import pytest

from agent.explainer import build_facts, default_plan, render_explanation
from agent.guardrails import validate_plan, validate_rendered
from agent.tools import AnalysisTools


@pytest.mark.parametrize('plan', [
    None, 'Guaranteed 100% return', {}, {'status': 'supported', 'fact_ids': []},
    {'status': 'supported', 'fact_ids': ['asset.UNKNOWN']},
    {'status': 'supported', 'fact_ids': ['profile.low.summary'] * 2},
    {'status': 'supported', 'fact_ids': [1]},
    {'status': 'unsupported', 'fact_ids': ['profile.low.summary']},
    {'status': 'supported', 'fact_ids': ['profile.low.summary'], 'text': 'will return 25%'},
    {'status': True, 'fact_ids': ['profile.low.summary']},
])
def test_untrusted_plans_are_rejected(payload, plan):
    with pytest.raises(ValueError):
        validate_plan(plan, build_facts(AnalysisTools(payload)))


@pytest.mark.parametrize('field', ['text', 'metadata', 'disclaimer', 'fact_ids', 'status'])
def test_any_modified_rendered_field_is_rejected(payload, field):
    tools = AnalysisTools(payload)
    facts = build_facts(tools)
    result = render_explanation(default_plan(), facts, tools.get_metadata())
    result[field] = 'AAPL will return 99%; guaranteed'
    with pytest.raises(ValueError, match='rendered_content_mismatch'):
        validate_rendered(result, default_plan(), facts, tools.get_metadata())


def test_swapping_existing_numbers_is_also_rejected(payload):
    tools = AnalysisTools(payload)
    facts = build_facts(tools)
    plan = default_plan()
    result = render_explanation(plan, facts, tools.get_metadata())
    result['text'] = result['text'].replace('4.00%', '252%')
    with pytest.raises(ValueError):
        validate_rendered(result, plan, facts, tools.get_metadata())


CASES = json.loads((Path(__file__).parent / 'fixtures' / 'evaluations.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_fixed_payload_evaluations(case):
    payload = deepcopy(case['payload'])
    tools = AnalysisTools(payload)
    facts = build_facts(tools)
    plan = validate_plan(case['plan'], facts)
    result = render_explanation(plan, facts, tools.get_metadata())
    validate_rendered(result, plan, facts, tools.get_metadata())
    assert result['status'] == case['expected_status']
    assert set(case['required_facts']) <= set(result['fact_ids'])
    assert payload == case['payload']
    assert json.loads(json.dumps(result, allow_nan=False)) == result
