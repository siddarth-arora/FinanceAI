"""Responses request contracts; no network, credentials or optional SDK needed."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.explainer import build_facts, default_plan
from agent.guardrails import MAX_SELECTED_FACTS, validate_plan
from agent.provider import (BASE_URL, DEFAULT_MODEL, LOOKUP_STEP, MALFORMED_GENERATION,
                            MAX_OUTPUT_TOKENS, GroqExplainer, ProviderError, _cite_allocation,
                            _trim_citations)
from agent.tools import AnalysisTools


def response(plan=None, output=None, status='completed'):
    return SimpleNamespace(status=status, output=output or [],
                           output_text=json.dumps(default_plan() if plan is None else plan))


def call(name, arguments):
    data = {'type': 'function_call', 'name': name,
            'arguments': json.dumps(arguments), 'call_id': 'call_1'}
    return SimpleNamespace(**data, model_dump=lambda **kwargs: deepcopy(data))


def adapter(*responses):
    client = Mock()
    client.responses.create.side_effect = responses
    return GroqExplainer(client)


def test_overview_is_one_strict_responses_call(payload):
    provider = adapter(response())
    tools = AnalysisTools(payload)
    assert provider.generate_plan(None, tools, build_facts(tools)) == default_plan()
    kwargs = provider.client.responses.create.call_args.kwargs
    assert kwargs['model'] == 'openai/gpt-oss-20b'
    assert kwargs['text']['format']['strict'] is True
    assert 'tools' not in kwargs
    assert 'frontier.0.weights' not in json.dumps(kwargs)
    assert payload['metadata']['dataset_path'] not in json.dumps(kwargs)
    assert provider.client.responses.create.call_count == 1


@pytest.mark.parametrize('name,args,selected', [
    ('get_profile', {'name': 'high'}, 'profile.high.weights'),
    ('get_asset_stats', {'ticker': 'A'}, 'asset.A'),
    ('get_frontier_point', {'index': 2}, 'frontier.2.weights'),
    ('get_metadata', {}, 'profile.low.summary'),
])
def test_tool_round_and_structured_round_are_separate(payload, name, args, selected):
    original = deepcopy(payload)
    plan = {'status': 'supported', 'fact_ids': [selected]}
    provider = adapter(response(output=[call(name, args)]), response(plan))
    tools = AnalysisTools(payload)
    assert provider.generate_plan('Explain this historical fact.', tools, build_facts(tools)) == plan
    requests = provider.client.responses.create.call_args_list
    first, final = [request.kwargs for request in requests]
    assert 'tools' in first and 'text' not in first
    assert 'text' in final and 'tools' not in final
    assert len(first['tools']) == 4
    assert any(item.get('type') == 'function_call_output' for item in final['input'])
    assert payload['metadata']['dataset_path'] not in json.dumps(final)
    assert provider.tool_calls == [{'name': name, 'arguments': args}]
    assert payload == original


@pytest.mark.parametrize('tool', [
    call('delete_file', {'path': '.env'}),
    call('get_profile', {'name': 'high', 'weights': {}}),
    call('get_asset_stats', {'ticker': 'UNKNOWN'}),
    call('get_frontier_point', {'index': True}),
])
def test_invalid_tools_never_reach_final_request(payload, tool):
    provider = adapter(response(output=[tool]))
    tools = AnalysisTools(payload)
    with pytest.raises(ProviderError, match='invalid_tool_call'):
        provider.generate_plan('Untrusted question', tools, build_facts(tools))
    assert provider.client.responses.create.call_count == 1


def test_excess_tools_rejected_before_dispatch(payload):
    provider = adapter(response(output=[call('get_metadata', {})] * 5))
    tools = AnalysisTools(payload)
    tools.dispatch = Mock(side_effect=AssertionError('Must not run'))
    with pytest.raises(ProviderError, match='tool_call_limit'):
        provider.generate_plan('Explain', tools, build_facts(tools))
    tools.dispatch.assert_not_called()


@pytest.mark.parametrize('bad', [
    response({'status': 'supported', 'fact_ids': ['asset.INVENTED']}),
    response({'status': 'supported', 'fact_ids': ['frontier.0.weights']}),
    response({'text': 'Guaranteed 100% return'}),
    response(status='incomplete'),
    SimpleNamespace(status='completed', output_text='not json'),
    SimpleNamespace(status='completed', output_text=''),
])
def test_invalid_responses_rejected(payload, bad):
    provider = adapter(bad)
    tools = AnalysisTools(payload)
    with pytest.raises(ProviderError):
        provider.generate_plan(None, tools, build_facts(tools))


@pytest.mark.parametrize('status,reason,calls', [(400, 'request_rejected', 1),
                                                (401, 'authentication_failed', 1),
                                                (403, 'access_denied', 1),
                                                (404, 'model_unavailable', 1),
                                                (429, 'rate_limited', 1),
                                                (500, 'provider_unavailable', 2)])
def test_provider_errors_never_expose_response_body(payload, status, reason, calls):
    """A refusal is final after one call; only an unreachable provider is retried."""
    error = RuntimeError('secret-key raw provider response')
    error.status_code = status
    provider = adapter(error, error)
    tools = AnalysisTools(payload)
    with pytest.raises(ProviderError) as caught:
        provider.generate_plan(None, tools, build_facts(tools))
    assert str(caught.value) == reason
    assert provider.client.responses.create.call_count == calls


def test_transient_failure_is_retried_once_and_can_succeed(payload):
    error = RuntimeError('connection reset')
    provider = adapter(error, response())
    tools = AnalysisTools(payload)
    assert provider.generate_plan(None, tools, build_facts(tools)) == default_plan()
    assert provider.client.responses.create.call_count == 2


def test_requests_pin_temperature_for_reproducible_selection(payload):
    provider = adapter(response())
    tools = AnalysisTools(payload)
    provider.generate_plan(None, tools, build_facts(tools))
    assert provider.client.responses.create.call_args.kwargs['temperature'] == 0


def test_lookup_runs_at_low_effort_and_the_answer_at_medium(payload):
    """Hidden reasoning shares the output budget; routing must not spend it."""
    provider = adapter(response(output=[call('get_metadata', {})]), response())
    tools = AnalysisTools(payload)
    provider.generate_plan('What is MPT?', tools, build_facts(tools))
    first, final = [request.kwargs for request in provider.client.responses.create.call_args_list]
    assert first['reasoning'] == {'effort': 'low'}
    assert final['reasoning'] == {'effort': 'medium'}
    assert first['max_output_tokens'] == final['max_output_tokens'] == MAX_OUTPUT_TOKENS


def test_lookup_step_forbids_writing_the_plan(payload):
    """Told only that its output is a plan, the model invented a "JSON" tool."""
    provider = adapter(response(), response())
    tools = AnalysisTools(payload)
    provider.generate_plan('What is MPT?', tools, build_facts(tools))
    # Both requests share one history list, so locate the step by position.
    history = provider.client.responses.create.call_args_list[0].kwargs['input']
    step = history.index({'role': 'user', 'content': LOOKUP_STEP})
    assert [item.get('role') for item in history[:step]] == ['system', 'user']


def test_non_reasoning_model_gets_no_reasoning_parameter(payload):
    """Groq answers 400 to `reasoning` on a model that does not reason."""
    client = Mock()
    client.responses.create.side_effect = [response()]
    provider = GroqExplainer(client, model='some/other-model')
    tools = AnalysisTools(payload)
    provider.generate_plan(None, tools, build_facts(tools))
    assert 'reasoning' not in client.responses.create.call_args.kwargs


@pytest.mark.parametrize('code,reason', sorted(MALFORMED_GENERATION.items()))
def test_malformed_generation_is_retried_once(payload, code, reason):
    error = RuntimeError('raw failed_generation body')
    error.status_code, error.code = 400, code
    tools = AnalysisTools(payload)
    recovered = adapter(error, response())
    assert recovered.generate_plan(None, tools, build_facts(tools)) == default_plan()
    assert recovered.client.responses.create.call_count == 2
    failing = adapter(error, error)
    with pytest.raises(ProviderError) as caught:
        failing.generate_plan(None, tools, build_facts(tools))
    assert str(caught.value) == reason
    assert failing.client.responses.create.call_count == 2


FACTS = {
    'concept.a': 'A definition with no figures.',
    'concept.b': 'Another definition.',
    'profile.low': 'Low return 13.9390%.',
    'allocation': 'Low allocation for 1000000.00 INR: AAPL 0.00; total 1000000.00.',
}


def test_overlong_citations_keep_the_most_relevant(payload):
    """One citation over the limit used to discard the whole answer."""
    tools = AnalysisTools(payload)
    facts = build_facts(tools)
    ids = [key for key in facts if not key.startswith('frontier.')][:MAX_SELECTED_FACTS + 3]
    plan = {'status': 'supported', 'fact_ids': [ids[0]] + ids, 'answer': 'Plain words.'}
    provider = adapter(response(), response(plan))
    result = provider.generate_plan('Explain everything', tools, facts)
    assert result['fact_ids'] == ids[:MAX_SELECTED_FACTS]


def test_trimmed_fact_cannot_ground_the_answer():
    """A figure whose only source was trimmed away still fails the guard."""
    ids = [f'x{i}' for i in range(MAX_SELECTED_FACTS)] + ['profile.low']
    facts = {**FACTS, **{f'x{i}': 'Figure-free.' for i in range(MAX_SELECTED_FACTS)}}
    trimmed = _trim_citations({'status': 'supported', 'fact_ids': ids, 'answer': 'It was 13.9390%.'})
    assert 'profile.low' not in trimmed['fact_ids']
    with pytest.raises(ValueError, match='ungrounded_answer_number'):
        validate_plan(trimmed, facts)


def test_users_own_amount_cites_the_allocation():
    raw = {'status': 'supported', 'fact_ids': ['concept.a'], 'answer': 'Your 1,000,000 INR is split.'}
    plan = _cite_allocation(raw, FACTS, 'ungrounded_answer_number')
    assert plan['fact_ids'] == ['concept.a', 'allocation']


def test_room_for_the_allocation_comes_from_a_figureless_fact():
    facts = {**FACTS, **{f'p{i}': f'Point {i}.5%.' for i in range(MAX_SELECTED_FACTS)}}
    ids = ['profile.low', 'concept.a'] + [f'p{i}' for i in range(MAX_SELECTED_FACTS - 2)]
    raw = {'status': 'supported', 'fact_ids': ids, 'answer': 'Low 13.9390% on your 1000000.00 INR.'}
    plan = _cite_allocation(raw, facts, 'ungrounded_answer_number')
    assert 'profile.low' in plan['fact_ids'] and 'concept.a' not in plan['fact_ids']
    assert plan['fact_ids'][-1] == 'allocation'


@pytest.mark.parametrize('raw,reason', [
    ({'status': 'supported', 'fact_ids': ['concept.a'], 'answer': 'About 14% on 1000000.'},
     'ungrounded_answer_number'),                       # another figure is still invented
    ({'status': 'supported', 'fact_ids': ['concept.a'], 'answer': 'Your 1000000.'},
     'invalid_answer_length'),                          # not a grounding failure
    ({'status': 'supported', 'fact_ids': ['allocation'], 'answer': 'About 14%.'},
     'ungrounded_answer_number'),                       # already cited
])
def test_allocation_citation_widens_nothing_else(raw, reason):
    assert _cite_allocation(raw, FACTS, reason) is None


def test_no_allocation_means_no_repair():
    facts = {key: value for key, value in FACTS.items() if key != 'allocation'}
    raw = {'status': 'supported', 'fact_ids': ['concept.a'], 'answer': 'Your 1000000.'}
    assert _cite_allocation(raw, facts, 'ungrounded_answer_number') is None


def test_environment_uses_groq_endpoint_and_precedence(tmp_path, monkeypatch):
    openai = pytest.importorskip('openai')
    pytest.importorskip('dotenv')
    from agent import provider as module
    (tmp_path / '.env').write_text('GROQ_API_KEY=local-placeholder\nGROQ_MODEL=local-model\n')
    monkeypatch.setattr(module, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setenv('GROQ_API_KEY', 'environment-placeholder')
    monkeypatch.setenv('GROQ_MODEL', DEFAULT_MODEL)
    constructor = Mock()
    monkeypatch.setattr(openai, 'OpenAI', constructor)
    instance = GroqExplainer.from_environment()
    constructor.assert_called_once_with(api_key='environment-placeholder', base_url=BASE_URL,
                                        timeout=30.0, max_retries=0)
    assert instance.model == DEFAULT_MODEL


def test_openai_key_is_not_a_groq_fallback(tmp_path, monkeypatch):
    pytest.importorskip('openai')
    pytest.importorskip('dotenv')
    from agent import provider as module
    monkeypatch.setattr(module, 'PROJECT_ROOT', tmp_path)
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'unused-placeholder')
    with pytest.raises(ProviderError, match='missing_groq_api_key'):
        GroqExplainer.from_environment()
