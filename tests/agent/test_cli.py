"""CLI contracts, including validation before the analysis is invoked."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import __main__ as cli
from agent.profiles import select_profiles


def test_no_arguments_preserve_existing_profile_output(payload, monkeypatch, capsys):
    run = Mock(return_value=SimpleNamespace(to_dict=lambda: payload))
    monkeypatch.setattr(cli, "run_mpt_analysis", run)
    cli.main([])
    assert json.loads(capsys.readouterr().out) == {
        name: profile.to_dict() for name, profile in select_profiles(payload).items()
    }
    run.assert_called_once_with()


@pytest.mark.parametrize("args, currency, profile", [
    (["--amount", "1000"], "USD", "medium"),
    (["--amount", "1000", "--currency", "INR", "--profile", "high", "--hide-dust"], "INR", "high"),
])
def test_allocation_arguments_return_structured_output(payload, monkeypatch, capsys, args, currency, profile):
    run = Mock(return_value=SimpleNamespace(to_dict=lambda: payload))
    monkeypatch.setattr(cli, "run_mpt_analysis", run)
    cli.main(args)
    output = json.loads(capsys.readouterr().out)
    assert output["amount"] == "1000.00"
    assert output["currency"] == currency
    assert output["target_profile"]["name"] == profile
    assert output["target_profile"]["metadata"] == payload["metadata"]
    assert output["display"]["total"] == "1000.00"
    run.assert_called_once_with()


@pytest.mark.parametrize("args", [
    ["--profile", "high"], ["--currency", "USD"], ["--hide-dust"],
    ["--amount", "NaN"], ["--amount", "0.001"],
    ["--amount", "10", "--currency", "EUR"],
    ["--amount", "10", "--currency", ""],
    ["--amount", "10", "--currency", " "],
    ["--amount", "10", "--profile", "unknown"],
    ["--llm"], ["--question", "Explain"], ["--explain", "--question", "Explain"],
    ["--explain", "--llm", "--question", ""],
    ["--explain", "--llm", "--question", "x" * 2001],
])
def test_invalid_arguments_fail_before_analysis(monkeypatch, capsys, args):
    run = Mock(side_effect=AssertionError("Analysis should not run"))
    monkeypatch.setattr(cli, "run_mpt_analysis", run)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    run.assert_not_called()


def test_explanation_cli_forwards_validated_options(monkeypatch, capsys):
    from agent import graph
    run = Mock(return_value={'explanation': {'mode': 'offline'}})
    monkeypatch.setattr(graph, 'run_agent', run)
    cli.main(['--explain', '--amount', '1000', '--currency', 'INR', '--hide-dust'])
    assert json.loads(capsys.readouterr().out)['explanation']['mode'] == 'offline'
    run.assert_called_once_with(amount='1000', currency='INR', profile='medium',
                                hide_dust=True, question=None, use_llm=False)


def test_llm_cli_requires_explicit_opt_in(monkeypatch, capsys):
    from agent import graph
    run = Mock(return_value={'explanation': {'mode': 'llm'}})
    monkeypatch.setattr(graph, 'run_agent', run)
    cli.main(['--explain', '--llm', '--question', 'Explain the high profile'])
    assert run.call_args.kwargs['use_llm'] is True
    assert run.call_args.kwargs['question'] == 'Explain the high profile'
