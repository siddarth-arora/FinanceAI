"""Full workflow regression: immutable results, bounded failures and offline use."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from agent import graph
from agent.explainer import build_facts, default_plan
from agent.preferences import Preferences
from agent.profiles import DISCLAIMER, select_profiles
from agent.provider import DEFAULT_MODEL, ProviderError
from agent.tools import AnalysisTools


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
    explanation = result['explanation']
    assert result['profiles'] == offline['profiles']
    # Question mode narrows the mandatory block, so the text is not the overview's;
    # what must hold is that it stays Python-rendered, grounded and prose-free.
    assert explanation['answer'] is None
    assert explanation['mode'] == 'fallback'
    assert explanation['fallback_reason'] == reason
    assert explanation['question_answered'] is False
    assert DISCLAIMER in explanation['text']
    assert explanation['text'].startswith('Dataset: ')
    facts = build_facts(AnalysisTools(payload))
    rendered = explanation['text'].split('\n\n')[1:-1]
    assert all(paragraph in facts.values() for paragraph in rendered)
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


def test_context_states_user_selections_for_fact_choice(payload):
    """A question must be answerable about the user's own profile and money."""
    provider = fake_provider({'status': 'supported', 'fact_ids': ['allocation']})
    graph.run_agent(payload, use_llm=True, provider=provider, question='How does this affect me?',
                    amount='1000.00', currency='USD', profile='high',
                    preferences=Preferences(emphasis='lower_risk'))
    context = provider.generate_plan.call_args.args[3]
    assert 'Selected profile: high.' in context
    assert 'USD' in context and 'allocation' in context
    assert 'lower risk' in context
    # Routing context only: no figure the model could restate as a finding.
    assert '1000.00' not in context


def test_context_reports_absent_amount_and_emphasis(payload):
    provider = fake_provider()
    graph.run_agent(payload, use_llm=True, provider=provider, question='Explain')
    context = provider.generate_plan.call_args.args[3]
    assert 'No amount was supplied' in context
    assert 'emphasis' not in context


def test_concept_facts_answer_method_questions(payload):
    provider = fake_provider({'status': 'supported', 'fact_ids': ['concept.mpt', 'concept.in_sample']})
    result = graph.run_agent(payload, use_llm=True, provider=provider,
                             question='What is MPT and what does it mean for me?')
    explanation = result['explanation']
    assert explanation['status'] == 'supported'
    assert explanation['question_answered'] is True
    assert 'Markowitz' in explanation['text']
    assert 'in-sample hindsight' in explanation['text']


def test_mandatory_only_plan_does_not_claim_to_answer(payload):
    """Citing nothing beyond the always-present facts is not an answer.

    The question matches no concept trigger, so the floor cannot rescue it and
    the flag must report the plan as it stands.
    """
    provider = fake_provider({'status': 'supported', 'fact_ids': ['profile.low.summary']})
    result = graph.run_agent(payload, use_llm=True, provider=provider,
                             question='Who audits the custodian?')
    assert result['explanation']['mode'] == 'llm'
    assert result['explanation']['question_answered'] is False


def test_concept_floor_rescues_a_plan_that_answered_nothing(payload):
    """Model selection is not reproducible; a method question must still land."""
    provider = fake_provider({'status': 'supported', 'fact_ids': ['profile.low.summary']})
    result = graph.run_agent(payload, use_llm=True, provider=provider,
                             question='Explain the MPT theory and how it affects my investment')
    explanation = result['explanation']
    assert explanation['mode'] == 'llm'
    assert explanation['question_answered'] is True
    assert 'concept.mpt' in explanation['fact_ids']
    assert 'Markowitz' in explanation['text']


def test_concept_floor_leaves_a_real_selection_untouched(payload):
    provider = fake_provider({'status': 'supported', 'fact_ids': ['asset.A']})
    result = graph.run_agent(payload, use_llm=True, provider=provider, question='What is MPT?')
    assert [f for f in result['explanation']['fact_ids'] if f.startswith('concept.')] == []


def test_concept_floor_never_overrides_an_unsupported_refusal(payload):
    provider = fake_provider({'status': 'unsupported', 'fact_ids': []})
    result = graph.run_agent(payload, use_llm=True, provider=provider,
                             question='Will the MPT theory predict next year?')
    assert result['explanation']['status'] == 'unsupported'
    assert result['explanation']['question_answered'] is False
    assert [f for f in result['explanation']['fact_ids'] if f.startswith('concept.')] == []


def test_concept_floor_does_not_apply_to_the_overview(payload):
    provider = fake_provider()
    result = graph.run_agent(payload, use_llm=True, provider=provider)
    assert [f for f in result['explanation']['fact_ids'] if f.startswith('concept.')] == []


def test_fallback_still_defines_the_terms_without_claiming_an_answer(payload):
    """An unreachable provider must not discard Python's own definitions.

    AGENTS.md forbids presenting a fallback as answering a question, so the mode
    and the flag stay honest while the relevant concepts are still rendered.
    """
    provider = fake_provider(error=ProviderError('provider_unavailable'))
    result = graph.run_agent(payload, use_llm=True, provider=provider,
                             question='What is MPT and how does it affect my investment?')
    explanation = result['explanation']
    assert explanation['mode'] == 'fallback'
    assert explanation['fallback_reason'] == 'provider_unavailable'
    assert explanation['question_answered'] is False
    assert 'concept.mpt' in explanation['fact_ids']
    assert 'Markowitz' in explanation['text']


def test_fallback_overview_adds_no_concepts(payload):
    provider = fake_provider(error=ProviderError('rate_limited'))
    result = graph.run_agent(payload, use_llm=True, provider=provider)
    assert result['explanation']['mode'] == 'fallback'
    assert [f for f in result['explanation']['fact_ids'] if f.startswith('concept.')] == []
