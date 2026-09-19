"""Full workflow regression: immutable results, bounded failures and offline use."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from agent import graph
from agent.explainer import default_plan
from agent.profiles import DISCLAIMER, select_profiles
from agent.provider import DEFAULT_MODEL, ProviderError


def fake_provider(plan=None, error=None):
    return Mock(model=DEFAULT_MODEL, tool_calls=[],
                generate_plan=Mock(return_value=default_plan() if plan is None else plan, side_effect=error))


@pytest.mark.parametrize('sequential', [True, False])
def test_offline_workflow_matches_profiles_and_allocation(payload, monkeypatch, sequential):
    original = deepcopy(payload)
    if sequential:
        monkeypatch.setattr(graph, '_langgraph', lambda: None)
    else:
        pytest.importorskip('langgraph.graph')
    provider = fake_provider(error=AssertionError('Offline must never call provider'))
    result = graph.run_agent(payload, amount='1000.00', currency='INR', profile='high',
                             hide_dust=True, provider=provider)
    assert result['orchestration'] == ('sequential' if sequential else 'langgraph')
    assert result['profiles'] == {name: p.to_dict() for name, p in select_profiles(payload).items()}
    assert result['allocation']['display']['total'] == '1000.00'
    assert result['explanation']['mode'] == 'offline'
    assert result['explanation']['model'] is None
    assert result['explanation']['fallback_reason'] is None
    assert DISCLAIMER in result['explanation']['text']
    assert 'allocation' in result['explanation']['fact_ids']
    provider.generate_plan.assert_not_called()
    assert payload == original
    json.dumps(result, allow_nan=False)


def test_valid_model_plan_only_selects_facts(payload):
    provider = fake_provider({'status': 'supported', 'fact_ids': ['asset.A']})
    result = graph.run_agent(payload, use_llm=True, provider=provider, question='Describe A')
    explanation = result['explanation']
    assert explanation['mode'] == 'llm'
    assert explanation['model'] == DEFAULT_MODEL
    assert explanation['provider'] == 'groq'
    assert 'asset.A' in explanation['fact_ids']
    assert explanation['question_answered'] is True
    provider.generate_plan.assert_called_once()
    provider.close.assert_not_called()  # Injected clients belong to the caller.


@pytest.mark.parametrize('provider,reason', [
    (fake_provider(error=ProviderError('rate_limited')), 'rate_limited'),
    (fake_provider(error=RuntimeError('secret')), 'explanation_failed'),
    (fake_provider({'status': 'supported', 'fact_ids': ['unknown']}), 'explanation_failed'),
    (fake_provider({'status': 'supported', 'fact_ids': [], 'text': 'Buy UNKNOWN'}), 'explanation_failed'),
])
def test_model_failures_fall_back_without_changing_finances(payload, provider, reason):
    offline = graph.run_agent(payload)
    result = graph.run_agent(payload, use_llm=True, provider=provider, question='Explain')
    assert result['profiles'] == offline['profiles']
    assert result['explanation']['text'] == offline['explanation']['text']
    assert result['explanation']['mode'] == 'fallback'
    assert result['explanation']['fallback_reason'] == reason
    assert result['explanation']['question_answered'] is False
    assert 'secret' not in json.dumps(result)


def test_unsupported_question_preserves_disclaimer_and_provenance(payload):
    provider = fake_provider({'status': 'unsupported', 'fact_ids': []})
    result = graph.run_agent(payload, use_llm=True, provider=provider, question='Predict tomorrow')
    assert result['explanation']['status'] == 'unsupported'
    assert result['explanation']['question_answered'] is False
    assert result['explanation']['metadata'] == payload['metadata']
    assert DISCLAIMER in result['explanation']['text']


def test_missing_credentials_fall_back_and_owned_client_closes(payload, monkeypatch):
    monkeypatch.setattr(graph.GroqExplainer, 'from_environment',
                        Mock(side_effect=ProviderError('missing_groq_api_key')))
    result = graph.run_agent(payload, use_llm=True)
    assert result['explanation']['fallback_reason'] == 'missing_groq_api_key'
    provider = fake_provider()
    monkeypatch.setattr(graph.GroqExplainer, 'from_environment', lambda: provider)
    assert graph.run_agent(payload, use_llm=True)['explanation']['mode'] == 'llm'
    provider.close.assert_called_once()


def test_invalid_baseline_is_not_hidden_by_fallback(payload):
    payload['efficient_frontier'][0]['expected_return'] = 99.0
    provider = fake_provider()
    with pytest.raises(ValueError):
        graph.run_agent(payload, use_llm=True, provider=provider)
    provider.generate_plan.assert_not_called()


@pytest.mark.parametrize('kwargs', [{'question': ''}, {'question': 'x' * 2001},
                                   {'question': 'Explain without LLM opt-in'},
                                   {'amount': 'NaN'}, {'amount': '1', 'profile': 'unknown'}])
def test_bad_user_inputs_fail_before_loading(monkeypatch, kwargs):
    load = Mock(side_effect=AssertionError('Must not load'))
    monkeypatch.setattr(graph, 'load_analysis', load)
    with pytest.raises(ValueError):
        graph.run_agent(**kwargs)
    load.assert_not_called()


def test_no_optional_dependencies_still_produces_explanation(payload, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def baseline_import(name, *args, **kwargs):
        if name.split('.')[0] in {'langgraph', 'openai', 'dotenv'}:
            raise ImportError('Optional dependency intentionally unavailable')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', baseline_import)
    offline = graph.run_agent(payload)
    fallback = graph.run_agent(payload, use_llm=True)
    assert offline['orchestration'] == 'sequential'
    assert offline['explanation']['mode'] == 'offline'
    assert fallback['explanation']['fallback_reason'] == 'missing_agent_dependencies'
    assert offline['explanation']['text'] == fallback['explanation']['text']


def test_workflow_loads_through_cache_interface(payload, monkeypatch):
    load = Mock(return_value=payload)
    monkeypatch.setattr(graph, 'load_analysis', load)
    assert graph.run_agent()['explanation']['metadata'] == payload['metadata']
    load.assert_called_once_with()


def test_transport_cleanup_cannot_override_provider_fallback(payload, monkeypatch):
    provider = fake_provider(error=ProviderError('provider_unavailable'))
    provider.close.side_effect = RuntimeError('raw transport error')
    monkeypatch.setattr(graph.GroqExplainer, 'from_environment', lambda: provider)
    result = graph.run_agent(payload, use_llm=True)
    assert result['explanation']['fallback_reason'] == 'provider_unavailable'
    assert 'raw transport' not in json.dumps(result)
