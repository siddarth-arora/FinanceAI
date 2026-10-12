"""Verified facts and deterministic rendering for grounded explanations."""

from copy import deepcopy
from typing import Any

from agent.news import news_facts
from agent.profiles import DISCLAIMER
from agent.tools import AnalysisTools


RULES = {
    'low': 'Low selects the first sampled frontier point, at global minimum variance.',
    'medium': 'Medium selects the sampled point nearest Maximum Sharpe volatility; it may differ from the exact optimum.',
    'high': 'High selects an interior point nearest the volatility midpoint above Maximum Sharpe, at or above medium; overlapping selections are possible.',
}
WARNINGS = {
    'concentration_threshold_exceeded': 'At least one holding exceeds the concentration threshold.',
    'few_material_holdings': 'Few holdings exceed the material-weight threshold.',
    'overlapping_profiles': 'This option shares a sampled point with another profile.',
    'frontier_endpoint_selected': 'This profile is at the frontier endpoint and may be concentrated in one asset.',
    'no_distinct_higher_interior_selection': 'High reuses medium because no distinct higher interior point was selected.',
}
# Method statements, not measurements: definitions of what this analysis computes
# and what its outputs do or do not mean. Python owns every word; the model may
# only cite them, so a "what does this mean" question needs no model prose.
CONCEPTS = {
    'concept.mpt': (
        'Modern Portfolio Theory (Markowitz) treats a portfolio by its combined risk and return '
        'rather than by each holding alone. Because assets do not move identically, a mix can carry '
        'less volatility than the weighted average of its parts. This analysis applies that method to '
        'a frozen historical sample: it estimates each asset\'s historical mean return and the sample '
        'covariance between assets, then solves for the weight sets with the least variance at each '
        'achievable return level.'
    ),
    'concept.frontier': (
        'The efficient frontier is the set of portfolios with the lowest variance available at each '
        'level of historical expected return; a portfolio below it is dominated by one with the same '
        'risk and more return. This analysis samples the frontier at a fixed number of points, each '
        'solved independently under the stated constraints. A frontier point is one complete set of '
        'target weights together with its historical return, volatility and Sharpe ratio.'
    ),
    'concept.profiles': (
        'The low, medium and high profiles are not separate strategies: each is one sampled frontier '
        'point chosen by a fixed published rule, so they differ only in where they sit on the same '
        'frontier. Their labels are relative positions within this sample, not statements about '
        'suitability for any individual.'
    ),
    'concept.volatility': (
        'Volatility here is the annualized standard deviation of the portfolio\'s historical daily '
        'returns, measuring how widely those returns dispersed around their own mean. It describes '
        'variability in the sample; it is not a maximum loss, a probability, or a forecast of future '
        'variability.'
    ),
    'concept.expected_return': (
        'Historical expected return is the arithmetic mean of realized daily returns over the frozen '
        'sample, annualized. It is a summary of what this sample did, not a prediction, a target or a '
        'guaranteed rate. A different sample window would produce different estimates and therefore '
        'different optimized weights.'
    ),
    'concept.sharpe': (
        'The Sharpe ratio divides historical return above the configured risk-free rate by volatility, '
        'summarizing reward per unit of variability. A higher value means this sample compensated its '
        'variability better; it ranks portfolios within this analysis and is not a quality rating.'
    ),
    'concept.diversification': (
        'Diversification reduces portfolio variance only to the extent that holdings fail to move '
        'together, which the sample covariance quantifies. Holdings that rise and fall together add '
        'little risk reduction, so concentration warnings report when few holdings carry most of the '
        'weight.'
    ),
    'concept.in_sample': (
        'These weights were optimized on the same historical window they are reported over, so the '
        'optimizer already knew which assets performed well in it. Figures of this kind are in-sample '
        'hindsight and systematically flatter the method; they are not a backtest of a decision made '
        'in advance, and they do not estimate future results.'
    ),
    'concept.allocation': (
        'An allocation applies the selected profile\'s target weights to a stated amount, producing '
        'fractional monetary targets. It places no trades, computes no share counts, assumes no '
        'currency conversion or live prices, and does not account for costs, taxes or minimum '
        'investment sizes.'
    ),
    'concept.constraints': (
        'Every portfolio here is fully invested with long-only weights between 0 and 1 that sum to 1, '
        'with no short selling and no leverage. Constraints restrict the achievable set, so a request '
        'outside it is reported as infeasible rather than met by relaxing them.'
    ),
}


def _metrics(label: str, point: dict) -> str:
    return (f"{label}: historical annual expected return {point['expected_return']:.4%}; "
            f"annual volatility {point['volatility']:.4%}; Sharpe ratio {point['sharpe_ratio']:.4f}.")


def _weights(label: str, weights: dict) -> str:
    return label + ' target weights: ' + ', '.join(f'{ticker} {weight:.4%}' for ticker, weight in weights.items()) + '.'


def build_facts(tools: AnalysisTools, allocation: dict | None = None,
                buckets: dict | None = None, news: list | None = None) -> dict[str, str]:
    """Only Python-generated facts may enter the final explanation."""
    facts = dict(CONCEPTS)
    for name in ('low', 'medium', 'high'):
        profile = tools.get_profile(name)
        facts[f'profile.{name}.summary'] = _metrics(name.title(), profile)
        facts[f'profile.{name}.selection'] = RULES[name]
        facts[f'profile.{name}.weights'] = _weights(name.title(), profile['weights'])
        if profile['warnings']:
            facts[f'profile.{name}.warnings'] = name.title() + ': ' + ' '.join(
                WARNINGS[code] for code in profile['warnings']
            )
    for ticker in tools.get_metadata()['assets']:
        stats = tools.get_asset_stats(ticker)
        facts[f'asset.{ticker}'] = (
            f"{ticker}: historical annual expected return {stats['annual_expected_return']:.4%}; "
            f"annual volatility {stats['annual_volatility']:.4%}."
        )
    for index in range(tools.frontier_count):
        point = tools.get_frontier_point(index)
        facts[f'frontier.{index}.summary'] = _metrics(f'Frontier point {index} (zero-based)', point)
        facts[f'frontier.{index}.weights'] = _weights(f'Frontier point {index}', point['weights'])
    if allocation is not None:
        display = allocation['display']
        facts['allocation'] = (
            f"{allocation['target_profile']['name'].title()} allocation for "
            f"{allocation['amount']} {allocation['currency']}: "
            + ', '.join(f'{ticker} {value}' for ticker, value in display['amounts'].items())
            + f"; grouped holdings {display['hidden_amount']}; total {display['total']}. "
            + allocation['notice']
        )
    if buckets is not None:
        facts.update(bucket_facts(buckets))
    if news:
        facts.update(news_facts(news))
    return facts


BUCKET_ROLES = {
    'conservative': 'the lowest-volatility option inside your window',
    'balanced': 'the highest Sharpe ratio inside your window',
    'growth': 'the higher-return end of your window',
}
REASONS = {
    'return_floor_above_frontier': 'The requested return floor exceeds every sampled frontier point.',
    'risk_ceiling_below_frontier': 'The requested risk ceiling is below the lowest-volatility frontier point.',
    'risk_ceiling_and_return_floor_conflict': 'No sampled frontier point satisfies both the risk ceiling and the return floor.',
}


def _limit(value: float | None) -> str:
    return 'no limit' if value is None else f'{value:.2%}'


def _delta(a: dict, b: dict) -> str:
    return (f"{a['expected_return'] - b['expected_return']:+.2%} expected return and "
            f"{a['volatility'] - b['volatility']:+.2%} volatility")


def bucket_facts(result: dict) -> dict[str, str]:
    """Python-written rationale: window, role of each bucket and cross-bucket trade-offs."""
    prefs = result['preferences']
    facts = {'bucket.window': (
        f"Your stated needs: emphasis {prefs['emphasis'].replace('_', ' ')}; maximum annual volatility "
        f"{_limit(prefs['max_volatility'])}; minimum historical expected return "
        f"{_limit(prefs['min_expected_return'])}. The unchanged efficient frontier spans "
        f"{result['frontier_range']['min_volatility']:.2%} to {result['frontier_range']['max_volatility']:.2%} "
        f"volatility.")}
    if result['status'] != 'ok':
        closest = result['closest']
        facts['bucket.infeasible'] = (
            REASONS[result['reason']] + f" The closest sampled frontier point (index {closest['frontier_index']}) has "
            f"historical expected return {closest['expected_return']:.4%} and volatility {closest['volatility']:.4%}; "
            'no portfolio was invented to meet the request.')
        return facts
    buckets = result['buckets']
    for name, bucket in buckets.items():
        facts[f'bucket.{name}.summary'] = _metrics(name.title() + f' bucket (frontier point {bucket["frontier_index"]})', bucket)
        facts[f'bucket.{name}.weights'] = _weights(name.title() + ' bucket', bucket['weights'])
        if bucket['warnings']:
            facts[f'bucket.{name}.warnings'] = name.title() + ' bucket: ' + ' '.join(
                WARNINGS[code] for code in bucket['warnings'])
    for name, other in (('conservative', 'balanced'), ('growth', 'balanced')):
        a, b = buckets[name], buckets[other]
        facts[f'bucket.{name}.why'] = (
            f"{name.title()} is {BUCKET_ROLES[name]}. Versus Balanced it changes historical "
            f"{_delta(a, b)} (Sharpe {a['sharpe_ratio'] - b['sharpe_ratio']:+.4f}).")
    facts['bucket.balanced.why'] = f"Balanced has {BUCKET_ROLES['balanced']} (Sharpe {buckets['balanced']['sharpe_ratio']:.4f})."
    recommended = result['recommended']
    facts['bucket.recommended'] = (
        f"{recommended.title()} matches your stated emphasis ({prefs['emphasis'].replace('_', ' ')}). "
        'This is a match to the stated preference over historical data, not personalized advice.')
    return facts


# Word-stem triggers per concept. Deterministic and Python-owned: the floor below
# must never depend on the model, because the model's selection is not reproducible.
CONCEPT_KEYWORDS: dict[str, tuple[str, ...]] = {
    'concept.mpt': ('mpt', 'modern portfolio', 'markowitz', 'theory', 'how does this work',
                    'how does it work', 'how is this calculated', 'optimis', 'optimiz'),
    'concept.frontier': ('frontier', 'efficient', 'curve', 'graph', 'chart', 'point'),
    'concept.profiles': ('profile', 'conservative', 'balanced', 'growth', 'low risk',
                         'high risk', 'which should i', 'difference between'),
    'concept.volatility': ('volatil', 'risk', 'std', 'deviation', 'swing', 'fluctuat'),
    'concept.expected_return': ('expected return', 'historical return', 'return mean',
                                'average return', 'cagr', 'how much will i', 'how much can i'),
    'concept.sharpe': ('sharpe', 'risk-adjusted', 'risk adjusted', 'per unit of risk'),
    'concept.diversification': ('diversif', 'correlat', 'concentrat', 'spread', 'sector'),
    'concept.in_sample': ('future', 'forecast', 'predict', 'next year', 'going forward',
                          'affect my', 'effect my', 'impact my', 'should i invest',
                          'reliable', 'trust', 'hindsight', 'backtest'),
    'concept.allocation': ('allocat', 'amount', 'invest', 'money', 'buy', 'rupee', 'dollar',
                           'share', 'how much'),
    'concept.constraints': ('constraint', 'short', 'leverage', 'borrow', 'cash', 'fully invested'),
}
MAX_FLOOR_CONCEPTS = 4


def concept_matches(question: str, facts: dict[str, str],
                    limit: int = MAX_FLOOR_CONCEPTS) -> list[str]:
    """Concept IDs whose triggers appear in the question, in catalogue order.

    A deterministic floor, not a ranking: it guarantees that a question about
    method or meaning is met with the definitions Python already wrote, even when
    the model selects nothing useful.
    """
    lowered = question.casefold()
    matched = [key for key, triggers in CONCEPT_KEYWORDS.items()
               if key in facts and any(trigger in lowered for trigger in triggers)]
    return matched[:limit]


def request_context(profile: str, allocation: dict | None, buckets: dict | None) -> str:
    """State the user's own selections so fact choice can be specific to them.

    This is routing context for selection only. It carries no figures the model
    could restate, and the renderer never reads it.
    """
    lines = [f'Selected profile: {profile}.']
    if allocation is not None:
        lines.append(f"The user supplied an amount, so the fact 'allocation' describes their own "
                     f"{allocation['currency']} targets for the {profile} profile.")
    else:
        lines.append('No amount was supplied, so no allocation fact exists.')
    if buckets is not None:
        prefs = buckets['preferences']
        lines.append(f"Stated emphasis: {prefs['emphasis'].replace('_', ' ')}; "
                     f"needs-driven 'bucket.*' facts are available"
                     + (f", recommending {buckets['recommended']}." if buckets.get('recommended')
                        else ' and report an infeasible window.'))
    return ' '.join(lines)


def mandatory_facts(facts: dict[str, str], cited: list[str] | None = None) -> list[str]:
    """Facts the renderer includes whatever the plan selected.

    An overview must disclose everything, so `cited` is omitted and the full
    block applies. A specific question gets a narrowed block: forcing ten
    unrelated statistical paragraphs into a one-line answer buries it. Risk
    disclosure is never dropped, only focused: a warning survives whenever the
    answer discusses the option it belongs to, and an infeasible window is
    always reported.
    """
    full = [f'profile.{name}.summary' for name in ('low', 'medium', 'high')] + [
        key for key in facts
        if key.endswith('.warnings') or key in ('allocation', 'bucket.window', 'bucket.infeasible')
        or (key.startswith('bucket.') and key.endswith('.summary'))
    ]
    if cited is None:
        return full
    discussed = {key.rsplit('.', 1)[0] for key in cited}
    return [key for key in full
            if key == 'bucket.infeasible'
            or (key.endswith('.warnings') and key.rsplit('.', 1)[0] in discussed)]


def default_plan(facts: dict[str, str] | None = None) -> dict[str, Any]:
    ids = [f'profile.{name}.selection' for name in ('low', 'medium', 'high')]
    if facts:
        ids += [key for key in ('bucket.recommended', 'bucket.conservative.why',
                                'bucket.balanced.why', 'bucket.growth.why') if key in facts]
    return {'status': 'supported', 'fact_ids': ids}


def render_explanation(plan: dict, facts: dict[str, str], metadata: dict,
                       question_mode: bool = False) -> dict:
    """Render only known references, always including provenance and warnings.

    `answer` carries the model's own words when the plan supplied prose that
    passed number validation; `text` stays the verbatim grounded rendering, so
    every figure a reader sees is still Python's. In question mode the mandatory
    block is narrowed to what the answer discusses.
    """
    chosen = plan['fact_ids']
    if (plan.get('status') not in ('supported', 'unsupported') or
            not isinstance(chosen, list) or any(not isinstance(key, str) or key not in facts for key in chosen)):
        raise ValueError('Invalid explanation references.')
    ids = list(dict.fromkeys(mandatory_facts(facts, chosen if question_mode else None) + chosen))
    constraints = metadata['constraints']
    provenance = (
        f"Dataset: {metadata['start_date']} to {metadata['end_date']}; "
        f"{metadata['trading_days']} trading days per year; annual risk-free rate "
        f"{metadata['risk_free_rate']:.2%}. Constraints: fully invested="
        f"{constraints['fully_invested']}, long-only={constraints['long_only']}, "
        f"leverage allowed={constraints['leverage_allowed']}."
    )
    paragraphs = [provenance]
    if plan['status'] == 'unsupported':
        paragraphs.append('That question is outside the supported historical data. The available profile comparison follows.')
    paragraphs.extend(facts[key] for key in ids)
    paragraphs.append(DISCLAIMER)
    return {'text': '\n\n'.join(paragraphs), 'answer': plan.get('answer'), 'fact_ids': ids,
            'metadata': deepcopy(metadata), 'disclaimer': DISCLAIMER,
            'status': plan['status']}


def explain_offline(payload: dict, allocation: dict | None = None) -> dict:
    tools = AnalysisTools(payload)
    result = render_explanation(default_plan(), build_facts(tools, allocation), tools.get_metadata())
    return {**result, 'mode': 'offline', 'model': None, 'fallback_reason': None}
