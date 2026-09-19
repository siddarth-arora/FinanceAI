from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import cache
from agent.explainer import explain_offline
from agent.tools import AnalysisTools


def test_tools_are_detached_and_do_not_change_payload(payload):
    before = deepcopy(payload)
    tools = AnalysisTools(payload)
    tools.get_profile('low')['weights'].clear()
    tools.get_metadata()['assets'].clear()
    tools.get_frontier_point(0)['weights'].clear()
    payload['metadata']['assets'].clear()
    assert tools.get_metadata() == before['metadata']
    assert tools.get_profile('low')['weights']
    assert tools.get_frontier_point(0)['weights']
    assert tools.get_asset_stats('A')['annual_volatility'] == pytest.approx(0.1)


@pytest.mark.parametrize('name,args', [
    ('download', {}), ('get_profile', {'name': 'unknown'}),
    ('get_profile', {'name': 'low', 'weights': [1]}),
    ('get_asset_stats', {'ticker': 'UNKNOWN'}),
    ('get_frontier_point', {'index': True}), ('get_frontier_point', {'index': -1}),
    ('get_frontier_point', {'index': 999}), ('get_frontier_point', {'index': 0.5}),
    ('get_metadata', {'path': '.env'}),
])
def test_invalid_tools_fail(payload, name, args):
    with pytest.raises(ValueError):
        AnalysisTools(payload).dispatch(name, args)


def test_offline_explanation_has_all_profiles_provenance_and_disclaimer(payload):
    result = explain_offline(payload)
    assert result['mode'] == 'offline'
    assert result['metadata'] == payload['metadata']
    assert '2024-01-01' in result['text'] and '4.00%' in result['text']
    assert 'long-only=True' in result['text']
    assert result['text'].endswith(result['disclaimer'])
    for name in ('low', 'medium', 'high'):
        assert f'profile.{name}.summary' in result['fact_ids']


def test_cache_reuses_analysis_without_sharing_mutable_values(tmp_path, monkeypatch):
    path = tmp_path / 'prices.parquet'
    path.write_bytes(b'fixture')
    run = Mock(return_value=SimpleNamespace(to_dict=lambda: {'metadata': {'assets': ['A']}}))
    monkeypatch.setattr(cache, 'RAW_PRICES_PATH', path)
    monkeypatch.setattr(cache, 'run_mpt_analysis', run)
    cache.clear_analysis_cache()
    cache.load_analysis()['metadata']['assets'].clear()
    assert cache.load_analysis()['metadata']['assets'] == ['A']
    assert run.call_count == 1
    path.write_bytes(b'changed fixture')
    cache.load_analysis()
    assert run.call_count == 2
    cache.clear_analysis_cache()
