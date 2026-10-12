"""Model prose is allowed, but never a figure the cited facts do not establish."""

import pytest

from agent.guardrails import (ANSWER_REJECTIONS, MAX_ANSWER_CHARS, plan_schema,
                              validate_answer, validate_plan)


FACTS = {
    'a': 'Low: historical annual expected return 13.9390%; annual volatility 12.4798%.',
    'b': 'NVDA target weight 65.6879%.',
    'c': 'Diversification reduces variance when holdings do not move together.',
    'd': 'Total 1,000,000.00 INR.',
}


@pytest.mark.parametrize('answer', [
    'Spreading money across things that do not move together lowers the bumpiness.',
    'The low option grew 13.9390% a year with 12.4798% ups and downs.',
    'Low returned 13.939% per year.',                      # trailing zero dropped
    'The total is 1000000.00 INR, the same figure written without separators.',
])
def test_grounded_prose_is_accepted(answer):
    assert validate_answer(answer, FACTS, ['a', 'b', 'c', 'd']) == answer.strip()


@pytest.mark.parametrize('answer', [
    'Low returned about 14% a year.',                      # rounded: not verbatim
    'Expect roughly 20% going forward.',                   # invented
    'Low and medium together average 24.2166%.',           # computed
    'That is 1250 rupees per month.',                      # unit conversion
])
def test_ungrounded_figures_are_rejected(answer):
    with pytest.raises(ValueError) as caught:
        validate_answer(answer, FACTS, ['a', 'b', 'c'])
    assert str(caught.value) == 'ungrounded_answer_number'


def test_prose_cannot_borrow_a_figure_the_reader_never_sees():
    """Grounding is checked against cited facts, not the whole catalogue."""
    with pytest.raises(ValueError):
        validate_answer('NVDA is 65.6879% of it.', FACTS, ['c'])
    assert validate_answer('NVDA is 65.6879% of it.', FACTS, ['b', 'c'])


@pytest.mark.parametrize('answer,reason', [
    ('', 'invalid_answer_length'),
    ('   ', 'invalid_answer_length'),
    ('x' * (MAX_ANSWER_CHARS + 1), 'invalid_answer_length'),
    (123, 'invalid_answer_type'),
    (None, 'invalid_answer_type'),
])
def test_malformed_answers_are_rejected(answer, reason):
    with pytest.raises(ValueError) as caught:
        validate_answer(answer, FACTS, ['a'])
    assert str(caught.value) == reason
    assert reason in ANSWER_REJECTIONS


def test_plan_without_an_answer_stays_valid():
    plan = validate_plan({'status': 'supported', 'fact_ids': ['a']}, FACTS)
    assert 'answer' not in plan


def test_blank_answer_is_dropped_not_fatal():
    plan = validate_plan({'status': 'supported', 'fact_ids': ['a'], 'answer': '  '}, FACTS)
    assert 'answer' not in plan


def test_plan_rejects_an_ungrounded_answer():
    with pytest.raises(ValueError) as caught:
        validate_plan({'status': 'supported', 'fact_ids': ['a'], 'answer': 'About 14%.'}, FACTS)
    assert str(caught.value) == 'ungrounded_answer_number'


def test_schema_requests_prose_only_for_a_question():
    assert 'answer' not in plan_schema(FACTS)['properties']
    with_answer = plan_schema(FACTS, with_answer=True)
    # Length is bounded by validate_answer, never the schema: a schema bound
    # failing at Groq discarded the whole plan, not just the prose.
    assert with_answer['properties']['answer'] == {'type': 'string'}
    assert set(with_answer['required']) == {'status', 'fact_ids', 'answer'}
    assert with_answer['additionalProperties'] is False


def test_unexpected_plan_keys_are_still_refused():
    with pytest.raises(ValueError) as caught:
        validate_plan({'status': 'supported', 'fact_ids': ['a'], 'text': 'Buy NVDA'}, FACTS)
    assert str(caught.value) == 'invalid_plan_shape'
