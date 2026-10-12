"""Concept definitions and the deterministic keyword floor."""

import pytest

from agent.explainer import (CONCEPTS, MAX_FLOOR_CONCEPTS, build_facts, concept_matches,
                             request_context)
from agent.tools import AnalysisTools



@pytest.mark.parametrize('question,expected', [
    ('What is MPT?', 'concept.mpt'),
    ('How volatile is this?', 'concept.volatility'),
    ('Will this affect my investment going forward?', 'concept.in_sample'),
    ('What does the Sharpe ratio mean?', 'concept.sharpe'),
    ('How much money do I allocate?', 'concept.allocation'),
    ('Can I use leverage or short selling?', 'concept.constraints'),
])
def test_concept_keywords_match_their_topic(payload, question, expected):
    facts = build_facts(AnalysisTools(payload))
    assert expected in concept_matches(question, facts)


def test_concept_matches_are_bounded_deterministic_and_known(payload):
    facts = build_facts(AnalysisTools(payload))
    question = 'Explain MPT, the frontier, profiles, volatility, returns, Sharpe and diversification'
    first = concept_matches(question, facts)
    assert first == concept_matches(question, facts)
    assert len(first) <= MAX_FLOOR_CONCEPTS
    assert all(key in facts for key in first)


def test_unrelated_question_matches_no_concept(payload):
    facts = build_facts(AnalysisTools(payload))
    assert concept_matches('Who is the prime minister?', facts) == []


def test_concepts_are_definitions_not_measurements():
    """A concept must never carry a figure the renderer would present as a result."""
    import re
    for key, text in CONCEPTS.items():
        assert not re.search(r'\d+\.\d+%|\d{2,}', text), f'{key} contains a measurement'
        assert text.strip() == text and text.endswith('.')


def test_concepts_reach_the_catalogue(payload):
    facts = build_facts(AnalysisTools(payload))
    assert set(CONCEPTS) <= set(facts)


def test_request_context_omits_figures_and_states_selections(payload):
    allocation = {'currency': 'USD', 'amount': '1234.56'}
    context = request_context('high', allocation, None)
    assert 'Selected profile: high.' in context
    assert '1234.56' not in context
