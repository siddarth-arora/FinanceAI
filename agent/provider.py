"""Bounded Groq Responses requests using the optional OpenAI SDK."""

import json
import os
from pathlib import Path
from typing import Any

from agent.guardrails import plan_schema, validate_plan
from agent.tools import AnalysisTools


BASE_URL = 'https://api.groq.com/openai/v1'
DEFAULT_MODEL = 'openai/gpt-oss-20b'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_TOOL_CALLS = 4


class ProviderError(Exception):
    """A stable reason code, never a raw provider response or credential."""


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if key != 'dataset_path'}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


class GroqExplainer:
    def __init__(self, client: Any, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model
        self.tool_calls: list[dict] = []

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

    def _request(self, **kwargs):
        try:
            response = self.client.responses.create(
                model=self.model, max_output_tokens=2048, **kwargs,
            )
        except Exception as error:
            status = getattr(error, 'status_code', None)
            reason = {401: 'authentication_failed', 403: 'access_denied',
                      429: 'rate_limited'}.get(status, 'provider_unavailable')
            raise ProviderError(reason) from None
        if response.status != 'completed':
            raise ProviderError('incomplete_response')
        return response

    def generate_plan(self, question: str | None, tools: AnalysisTools, facts: dict) -> dict:
        """At most two requests: optional tool retrieval, then strict references.

        Groq does not combine tool use with structured outputs. Each request
        therefore has exactly one role. No retries or arbitrary model text reach
        the final explanation. Only requested frontier facts enter the catalogue.
        """
        self.tool_calls = []
        instructions = (Path(__file__).parent / 'prompts/system.md').read_text()
        history = [{'role': 'system', 'content': instructions},
                   {'role': 'user', 'content': question or 'Explain the three historical profiles.'}]
        available = {key: value for key, value in facts.items() if not key.startswith('frontier.')}
        if question:
            response = self._request(input=history, tools=tools.definitions(), tool_choice='auto')
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
        history.append({'role': 'user', 'content': 'Return only the final reference plan. Approved facts:\n'
                        + json.dumps(available, allow_nan=False)})
        response = self._request(input=history, text={'format': {
            'type': 'json_schema', 'name': 'explanation_plan', 'strict': True,
            'schema': plan_schema(available),
        }})
        try:
            return validate_plan(json.loads(response.output_text), available)
        except (ValueError, TypeError):
            raise ProviderError('invalid_model_plan') from None

    def close(self) -> None:
        self.client.close()
