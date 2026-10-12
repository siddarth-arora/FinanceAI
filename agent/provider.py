"""Bounded Groq Responses requests using the optional OpenAI SDK."""

import json
import os
from pathlib import Path
from typing import Any

from agent.guardrails import (ANSWER_REJECTIONS, MAX_SELECTED_FACTS, _numbers, plan_schema,
                              validate_plan)
from agent.tools import AnalysisTools


BASE_URL = 'https://api.groq.com/openai/v1'
DEFAULT_MODEL = 'openai/gpt-oss-20b'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_TOOL_CALLS = 4
# gpt-oss is a reasoning model: hidden reasoning counts against this budget, and
# at 2048 it was exhausted before any output on about half of question calls.
MAX_OUTPUT_TOKENS = 4096
# Groq rejects `reasoning` (HTTP 400) for any model that does not reason.
REASONING_MODEL_PREFIX = 'openai/gpt-oss-'
# Groq's codes for a malformed generation: random, so worth one retry, unlike a
# 400 caused by the request itself. Mapped to our own codes; no body is kept.
MALFORMED_GENERATION = {'tool_use_failed': 'invalid_tool_call',
                        'json_validate_failed': 'invalid_model_plan'}
# Without this the model, told its final output is a plan, delivered the plan
# during lookup by calling a nonexistent "JSON" tool.
LOOKUP_STEP = ('Step 1 of 2: look-ups only. Call any of the four tools whose data the '
               'question needs, or reply with the single word READY if none is needed. '
               'Do not write the plan or the answer yet; the fact catalogue and the plan '
               'format come in the next step.')


class ProviderError(Exception):
    """A stable reason code, never a raw provider response or credential."""


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if key != 'dataset_path'}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _trim_citations(raw: Any) -> Any:
    """Keep the most relevant citations instead of discarding an over-long plan.

    A broad question ("explain the theory and what it means for me") led the
    model to cite 13-15 facts, and one over the limit cost the whole answer.
    IDs arrive ordered by relevance, so the first MAX_SELECTED_FACTS distinct
    ones are kept. Nothing is admitted that validation would refuse: every kept
    ID is still checked against the catalogue, and the prose is number-checked
    against only the kept facts, so a figure from a dropped fact fails.
    """
    if isinstance(raw, dict) and isinstance(raw.get('fact_ids'), list):
        ids = raw['fact_ids']
        if all(isinstance(key, str) for key in ids):
            raw = {**raw, 'fact_ids': list(dict.fromkeys(ids))[:MAX_SELECTED_FACTS]}
    return raw


def _cite_allocation(raw: Any, available: dict, reason: str) -> dict | None:
    """Cite the user's own allocation when that alone grounds the answer.

    Asked how something affects "my investment", the model names the user's
    amount but often omits the "allocation" fact that states it, and the prose
    was discarded for quoting the user's own input. Appending that one fact
    makes the figure's source visible, which is what the cited-facts rule
    protects. Nothing else is widened: if any figure is still unfound, the plan
    is refused here and the caller strips the prose as before.
    """
    if (reason != 'ungrounded_answer_number' or 'allocation' not in available
            or not isinstance(raw, dict) or 'allocation' in raw.get('fact_ids', [])):
        return None
    ids = list(raw['fact_ids'])
    if len(ids) >= MAX_SELECTED_FACTS:
        # Make room by dropping a fact with no figures (a concept definition):
        # it can never be the source of a number the answer states.
        figureless = [key for key in ids if key in available and not _numbers(available[key])]
        if not figureless:
            return None
        ids.remove(figureless[-1])
    ids.append('allocation')
    try:
        return validate_plan({**raw, 'fact_ids': ids}, available)
    except (ValueError, TypeError):
        return None


class GroqExplainer:
    def __init__(self, client: Any, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model
        self.tool_calls: list[dict] = []
        # Set when prose was discarded for stating an unverifiable figure.
        self.answer_rejected: str | None = None

    @classmethod
    def from_environment(cls) -> 'GroqExplainer':
        # Called only for explicit LLM use; no secrets enter graph state or logs.
        try:
            from dotenv import dotenv_values
            from openai import OpenAI
        except ImportError:
            raise ProviderError('missing_agent_dependencies') from None
        local = dotenv_values(PROJECT_ROOT / '.env', interpolate=False)
        key = os.environ.get('GROQ_API_KEY', local.get('GROQ_API_KEY'))
        model = os.environ.get('GROQ_MODEL', local.get('GROQ_MODEL')) or DEFAULT_MODEL
        if not key or not key.strip():
            raise ProviderError('missing_groq_api_key')
        return cls(OpenAI(api_key=key.strip(), base_url=BASE_URL,
                          timeout=30.0, max_retries=0), model=model)

    def _request(self, effort: str, **kwargs):
        """Pinned temperature keeps fact selection reproducible for the same input.

        `effort` bounds hidden reasoning. Tool retrieval only routes, so it runs
        at "low"; left at the default it drafted the whole answer while deciding
        which tools to call and ran out of budget before calling any.

        A transient reach failure or a malformed generation is retried once,
        because one blip otherwise costs the whole answer; a refusal (bad request,
        auth, access, unknown model, rate limit) is final and retrying it would
        only waste the wait.
        """
        if self.model.startswith(REASONING_MODEL_PREFIX):
            kwargs['reasoning'] = {'effort': effort}
        last = 'provider_unavailable'
        for attempt in range(2):
            try:
                response = self.client.responses.create(
                    model=self.model, max_output_tokens=MAX_OUTPUT_TOKENS, temperature=0, **kwargs,
                )
            except Exception as error:
                code = getattr(error, 'code', None)
                if code in MALFORMED_GENERATION:
                    last = MALFORMED_GENERATION[code]
                    continue
                status = getattr(error, 'status_code', None)
                # 400/404 mean the configured model cannot serve this request
                # (e.g. GROQ_MODEL names a model the Responses API lacks): final.
                last = {400: 'request_rejected', 401: 'authentication_failed',
                        403: 'access_denied', 404: 'model_unavailable',
                        429: 'rate_limited'}.get(status, 'provider_unavailable')
                if last != 'provider_unavailable':
                    raise ProviderError(last) from None
                continue
            if response.status != 'completed':
                raise ProviderError('incomplete_response')
            return response
        raise ProviderError(last)

    def generate_plan(self, question: str | None, tools: AnalysisTools, facts: dict,
                      context: str | None = None) -> dict:
        """At most two requests: optional tool retrieval, then strict references.

        Groq does not combine tool use with structured outputs. Each request
        therefore has exactly one role. Model prose arrives only in `answer`, and
        only after every figure in it is found in the facts it cites; the rendered
        text stays Python's. Only requested frontier facts enter the catalogue. `context`
        states the user's own selections so a question can be answered about
        their profile and allocation rather than in general.
        """
        self.tool_calls = []
        self.answer_rejected = None
        instructions = (Path(__file__).parent / 'prompts/system.md').read_text()
        prompt = question or 'Explain the three historical profiles.'
        if context:
            prompt = f'{prompt}\n\nThe user\'s current selections: {context}'
        history = [{'role': 'system', 'content': instructions},
                   {'role': 'user', 'content': prompt}]
        available = {key: value for key, value in facts.items() if not key.startswith('frontier.')}
        if question:
            history.append({'role': 'user', 'content': LOOKUP_STEP})
            response = self._request('low', input=history, tools=tools.definitions(),
                                     tool_choice='auto')
            calls = [item for item in response.output if item.type == 'function_call']
            if len(calls) > MAX_TOOL_CALLS:
                raise ProviderError('tool_call_limit')
            history.extend(item.model_dump(exclude_none=True) for item in response.output)
            for call in calls:
                try:
                    arguments = json.loads(call.arguments)
                    result = tools.dispatch(call.name, arguments)
                except (ValueError, TypeError):
                    raise ProviderError('invalid_tool_call') from None
                self.tool_calls.append({'name': call.name, 'arguments': arguments})
                if call.name == 'get_frontier_point':
                    prefix = f"frontier.{arguments['index']}."
                    available.update({key: value for key, value in facts.items() if key.startswith(prefix)})
                history.append({'type': 'function_call_output', 'call_id': call.call_id,
                                'output': json.dumps(_public(result), allow_nan=False)})
        instruction = 'Return only the final reference plan. Approved facts:\n'
        if question:
            instruction = (
                'Return the final reference plan. In "answer", reply to the question in your own '
                'words, at the register the question asks for, in at most 150 words, using only '
                'what the facts you cite establish. State no number, percentage or amount that does '
                'not appear verbatim in those facts, and include in fact_ids the fact behind every '
                'figure you state; prefer words such as "more", "less" or "all of it" over figures. '
                'An answer containing any other number is discarded. Cite only the facts the '
                'answer uses, most relevant first: usually two to six. Approved facts:\n')
        history.append({'role': 'user', 'content': instruction + json.dumps(available, allow_nan=False)})
        response = self._request('medium', input=history, text={'format': {
            'type': 'json_schema', 'name': 'explanation_plan', 'strict': True,
            'schema': plan_schema(available, with_answer=bool(question)),
        }})
        try:
            raw = _trim_citations(json.loads(response.output_text))
        except (ValueError, TypeError):
            raise ProviderError('invalid_model_plan') from None
        try:
            return validate_plan(raw, available)
        except ValueError as error:
            cited = _cite_allocation(raw, available, str(error))
            if cited is not None:
                return cited
            # Ungrounded prose costs only the prose: the cited facts were still
            # chosen from the catalogue, so the reader keeps a verified answer.
            if str(error) in ANSWER_REJECTIONS and isinstance(raw, dict):
                self.answer_rejected = str(error)
                try:
                    return validate_plan({key: value for key, value in raw.items()
                                          if key != 'answer'}, available)
                except (ValueError, TypeError):
                    pass
            raise ProviderError('invalid_model_plan') from None
        except TypeError:
            raise ProviderError('invalid_model_plan') from None

    def close(self) -> None:
        self.client.close()
