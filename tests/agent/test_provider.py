"""Responses request contracts; no network, credentials or optional SDK needed."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.explainer import build_facts, default_plan
from agent.provider import BASE_URL, DEFAULT_MODEL, GroqExplainer, ProviderError
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


@pytest.mark.parametrize('status,reason', [(401, 'authentication_failed'), (403, 'access_denied'),
                                         (429, 'rate_limited'), (500, 'provider_unavailable')])
def test_provider_errors_never_expose_response_body(payload, status, reason):
    error = RuntimeError('secret-key raw provider response')
    error.status_code = status
    provider = adapter(error)
    tools = AnalysisTools(payload)
    with pytest.raises(ProviderError) as caught:
        provider.generate_plan(None, tools, build_facts(tools))
    assert str(caught.value) == reason
    assert provider.client.responses.create.call_count == 1


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
