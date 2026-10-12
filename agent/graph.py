"""Optional LangGraph orchestration over the deterministic analysis contract."""

from copy import deepcopy
from typing import Any, TypedDict

from agent.allocation import allocate_amount, validate_allocation_input
from agent.cache import load_analysis
from agent.explainer import (build_facts, concept_matches, default_plan, mandatory_facts,
                             render_explanation, request_context)
from agent.guardrails import MAX_SELECTED_FACTS, validate_plan, validate_rendered
from agent.news import NewsProvider, sanitize_news
from agent.preferences import Preferences, select_buckets
from agent.provider import DEFAULT_MODEL, GroqExplainer, ProviderError
from agent.tools import AnalysisTools


class AgentState(TypedDict, total=False):
    payload: dict
    tools: AnalysisTools
    profiles: dict
    allocation: dict | None
    buckets: dict | None
    news: list
    news_status: str
    facts: dict
    plan: dict
    mode: str
    model: str | None
    answer_rejected: str | None
    fallback_reason: str | None
    tool_calls: list
    result: dict


def validate_question(question: str | None) -> None:
    if question is not None and (not isinstance(question, str) or
                                 not question.strip() or len(question) > 2000):
        raise ValueError('Question must contain 1 to 2000 characters.')


def _langgraph():
    try:
        from langgraph.graph import END, START, StateGraph
        return StateGraph, START, END
    except ImportError:
        return None


def run_agent(payload: dict | None = None, *, amount: Any = None, currency: str = 'USD',
              profile: str = 'medium', hide_dust: bool = False, question: str | None = None,
              use_llm: bool = False, provider: GroqExplainer | None = None,
              preferences: Preferences | None = None,
              news_provider: NewsProvider | None = None) -> dict:
    """Explain an isolated analysis, offline unless use_llm is explicit.

    LangGraph is optional. Without it the same nodes run sequentially. Baseline
    and allocation validation failures propagate; only explanation failures fall
    back. An injected provider is caller-owned and is never used in offline mode.
    Preferences add three needs-driven buckets chosen from the frontier. News is
    fetched only from an explicitly injected news_provider, sanitized, treated as
    untrusted context, and never affects calculations; its failure is non-fatal.
    """
    validate_question(question)
    if question is not None and not use_llm:
        raise ValueError('A question requires use_llm=True; offline mode provides a general comparison.')
    if amount is not None:
        validate_allocation_input(amount, currency)
        if profile not in ('low', 'medium', 'high'):
            raise ValueError('Unknown profile.')

    def load_node(state):
        return {'payload': deepcopy(payload) if payload is not None else load_analysis()}

    def profile_node(state):
        tools = AnalysisTools(state['payload'])
        return {'tools': tools, 'profiles': {name: tools.get_profile(name)
                                           for name in ('low', 'medium', 'high')}}

    def bucket_node(state):
        if preferences is None:
            return {'buckets': None}
        return {'buckets': select_buckets(state['tools'], preferences)}

    def news_node(state):
        if news_provider is None:
            return {'news': [], 'news_status': 'not_requested'}
        assets = state['tools'].get_metadata()['assets']
        try:
            return {'news': sanitize_news(list(news_provider.fetch(list(assets))), assets),
                    'news_status': 'ok'}
        except Exception:
            # Never copy provider errors; explanations proceed without news.
            return {'news': [], 'news_status': 'unavailable'}

    def allocation_node(state):
        allocation = None if amount is None else allocate_amount(
            amount, currency, state['payload'], profile).to_dict(hide_dust=hide_dust)
        return {'allocation': allocation}

    def _with_concept_floor(plan: dict, facts: dict, *, model_selected: bool) -> dict:
        """Add the definitions matching the question when nothing else addresses it.

        Selection by the model is not reproducible even at temperature 0, so an
        otherwise valid call can cite only the mandatory facts and address the
        question not at all; a fallback plan is the generic overview and is never
        question-aware. The keyword floor is deterministic and cites text Python
        wrote, so it cannot invent a figure or a claim. An "unsupported" verdict is
        left alone: the model is entitled to refuse.
        """
        if not question or plan['status'] != 'supported':
            return plan
        mandatory = set(mandatory_facts(facts))
        if model_selected and any(key not in mandatory for key in plan['fact_ids']):
            return plan
        added = [key for key in concept_matches(question, facts) if key not in plan['fact_ids']]
        if not added:
            return plan
        return validate_plan({'status': 'supported',
                              'fact_ids': (plan['fact_ids'] + added)[:MAX_SELECTED_FACTS]}, facts)

    def explain_node(state):
        facts = build_facts(state['tools'], state['allocation'], state['buckets'], state['news'])
        update = {'facts': facts, 'plan': default_plan(facts), 'mode': 'offline',
                  'model': None, 'fallback_reason': None, 'tool_calls': [],
                  'answer_rejected': None}
        if not use_llm:
            return update
        active = provider
        update.update(mode='fallback', model=DEFAULT_MODEL)
        try:
            if active is None:
                active = GroqExplainer.from_environment()
            update['model'] = active.model
            context = request_context(profile, state['allocation'], state['buckets'])
            update['plan'] = validate_plan(
                active.generate_plan(question, state['tools'], facts, context), facts)
            update['mode'] = 'llm'
        except ProviderError as error:
            # ProviderError messages are fixed codes defined by our adapter.
            update['fallback_reason'] = str(error)
        except Exception:
            # Never copy SDK exceptions, raw output, or credentials to JSON.
            update['fallback_reason'] = 'explanation_failed'
        finally:
            if active is not None:
                update['tool_calls'] = deepcopy(active.tool_calls)
                # An injected provider is caller-owned and need not define this.
                rejected = getattr(active, 'answer_rejected', None)
                update['answer_rejected'] = rejected if isinstance(rejected, str) else None
                if provider is None:
                    try:
                        active.close()
                    except Exception:
                        # Transport cleanup must not override a validated result
                        # or expose an SDK error after a successful fallback.
                        pass
        # Applied to a fallback plan too: an unreachable provider is no reason to
        # discard definitions Python already wrote for this question. The mode
        # stays "fallback" and question_answered stays false, so a fallback is
        # never presented as having answered.
        update['plan'] = _with_concept_floor(update['plan'], facts,
                                             model_selected=update['mode'] == 'llm')
        return update

    def guardrail_node(state):
        plan = validate_plan(state['plan'], state['facts'])
        metadata = state['tools'].get_metadata()
        question_mode = bool(question)
        explanation = render_explanation(plan, state['facts'], metadata, question_mode)
        validate_rendered(explanation, plan, state['facts'], metadata, question_mode)
        explanation.update({key: state[key] for key in
                            ('mode', 'model', 'fallback_reason', 'tool_calls', 'answer_rejected')})
        explanation['provider'] = 'groq' if use_llm else None
        # A plan that adds nothing beyond the always-present mandatory facts has
        # not addressed the question, however successful the provider call was.
        selected = [key for key in plan['fact_ids'] if key not in set(mandatory_facts(state['facts']))]
        explanation['question_answered'] = bool(
            question and state['mode'] == 'llm' and plan['status'] == 'supported' and selected)
        return {'result': {'profiles': state['profiles'], 'buckets': state['buckets'],
                           'allocation': state['allocation'],
                           'news': [item.to_dict() for item in state['news']],
                           'news_status': state['news_status'], 'explanation': explanation}}

    nodes = [('load_analysis', load_node), ('build_profiles', profile_node),
             ('select_buckets', bucket_node), ('fetch_news', news_node),
             ('allocate', allocation_node), ('explain', explain_node), ('guardrail', guardrail_node)]
    graph_api = _langgraph()
    if graph_api is None:
        state = {}
        for _, node in nodes:
            state.update(node(state))
    else:
        StateGraph, START, END = graph_api
        builder = StateGraph(AgentState)
        previous = START
        for name, node in nodes:
            builder.add_node(name, node)
            builder.add_edge(previous, name)
            previous = name
        builder.add_edge(previous, END)
        state = builder.compile().invoke({})
    result = state['result']
    result['orchestration'] = 'langgraph' if graph_api else 'sequential'
    return result
