"""Optional LangGraph orchestration over the deterministic analysis contract."""

from copy import deepcopy
from typing import Any, TypedDict

from agent.allocation import allocate_amount, validate_allocation_input
from agent.cache import load_analysis
from agent.explainer import build_facts, default_plan, render_explanation
from agent.guardrails import validate_plan, validate_rendered
from agent.provider import DEFAULT_MODEL, GroqExplainer, ProviderError
from agent.tools import AnalysisTools


class AgentState(TypedDict, total=False):
    payload: dict
    tools: AnalysisTools
    profiles: dict
    allocation: dict | None
    facts: dict
    plan: dict
    mode: str
    model: str | None
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
              use_llm: bool = False, provider: GroqExplainer | None = None) -> dict:
    """Explain an isolated analysis, offline unless use_llm is explicit.

    LangGraph is optional. Without it the same nodes run sequentially. Baseline
    and allocation validation failures propagate; only explanation failures fall
    back. An injected provider is caller-owned and is never used in offline mode.
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

    def allocation_node(state):
        allocation = None if amount is None else allocate_amount(
            amount, currency, state['payload'], profile).to_dict(hide_dust=hide_dust)
        return {'allocation': allocation}

    def explain_node(state):
        facts = build_facts(state['tools'], state['allocation'])
        update = {'facts': facts, 'plan': default_plan(), 'mode': 'offline',
                  'model': None, 'fallback_reason': None, 'tool_calls': []}
        if not use_llm:
            return update
        active = provider
        update.update(mode='fallback', model=DEFAULT_MODEL)
        try:
            if active is None:
                active = GroqExplainer.from_environment()
            update['model'] = active.model
            update['plan'] = validate_plan(active.generate_plan(question, state['tools'], facts), facts)
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
                if provider is None:
                    try:
                        active.close()
                    except Exception:
                        # Transport cleanup must not override a validated result
                        # or expose an SDK error after a successful fallback.
                        pass
        return update

    def guardrail_node(state):
        plan = validate_plan(state['plan'], state['facts'])
        explanation = render_explanation(plan, state['facts'], state['tools'].get_metadata())
        validate_rendered(explanation, plan, state['facts'], state['tools'].get_metadata())
        explanation.update({key: state[key] for key in ('mode', 'model', 'fallback_reason', 'tool_calls')})
        explanation['provider'] = 'groq' if use_llm else None
        explanation['question_answered'] = bool(question and state['mode'] == 'llm' and plan['status'] == 'supported')
        return {'result': {'profiles': state['profiles'], 'allocation': state['allocation'],
                           'explanation': explanation}}

    nodes = [('load_analysis', load_node), ('build_profiles', profile_node),
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
