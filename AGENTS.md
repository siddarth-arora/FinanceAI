# Instructions for AI Coding Agents

This file is the repository-level working contract for AI coding agents. Read
`README.md` and the relevant source modules before proposing or making changes.

## Objective

Preserve this repository as a transparent, deterministic baseline
implementation of traditional Markowitz Modern Portfolio Theory. Future agentic
or service code may consume the baseline, but must not replace or silently
alter its numerical calculations.

## Non-negotiable boundaries

- Yahoo Finance access belongs only in `src/download_data.py`.
- `src.main`, `src.pipeline`, and all mathematical modules must run from the
  frozen local Parquet dataset without network access.
- Never make dataset download or overwrite implicit.
- Never parse CLI output or the PNG to obtain portfolio data.
- Consume structured results through `src.pipeline.run_mpt_analysis()`.
- Keep LLM, agent, RAG, prompt, tool-calling, and UI logic outside the baseline
  financial modules.
- Baseline modules must never import a future agentic/service package. The
  future layer may import `src.pipeline`; dependency direction is one-way.
- Do not use an LLM to calculate returns, covariance, portfolio metrics, or
  optimized weights. Use the deterministic functions in this repository.
- Do not add random-portfolio simulation to the baseline frontier.

## Financial contract

Unless a human explicitly approves and documents a change:

- Use simple percentage returns, not log returns.
- Estimate expected returns using historical arithmetic daily means.
- Estimate covariance using historical daily sample covariance.
- Annualize both using 252 trading days.
- Require weights to sum to 1.
- Require every weight to remain between 0 and 1.
- Do not allow short selling or leverage.
- Use the configured annual risk-free rate for Sharpe calculations.
- Generate the efficient upper branch with constrained target-return
  optimization, beginning at GMV.

Historical expected returns are inputs estimated from the frozen sample. Never
label them as predictions, guaranteed returns, or personalized recommendations.

## Module ownership

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Configurable parameters and paths only |
| `src/download_data.py` | Explicit Yahoo download, validation, and Parquet save |
| `src/data_loader.py` | Local Parquet loading and validation only |
| `src/returns.py` | Simple returns, expected returns, and covariance |
| `src/portfolio.py` | Pure portfolio return, variance, volatility, and Sharpe math |
| `src/optimizer.py` | Constrained GMV, Maximum Sharpe, target-return, and frontier optimization |
| `src/pipeline.py` | Reusable orchestration and structured result contract |
| `src/visualization.py` | Plotting only |
| `src/main.py` | Human-readable CLI and plot invocation |

Do not move formulas into `main.py`, plotting into optimizers, or acquisition
logic into the local pipeline.

## Integration instructions

Use this interface:

```python
from src.pipeline import run_mpt_analysis

result = run_mpt_analysis()
payload = result.to_dict()
```

The returned dictionary contains:

```text
metadata
annual_expected_returns
annual_covariance
portfolios.global_minimum_variance
portfolios.maximum_sharpe
efficient_frontier
```

Rules for consumers:

1. Treat all rates and weights as decimal numbers; `0.25` means 25%.
2. Use ticker-keyed weight mappings rather than relying on inferred ordering.
3. Preserve `metadata`, including dataset dates, trading days, risk-free rate,
   and constraints, when passing results downstream.
4. Use full-precision values for calculations. Round only in user-facing text.
5. Select candidates from `efficient_frontier`; do not invent or interpolate
   weights with an LLM.
6. Recalculate or validate any externally modified weights with
   `src.portfolio` before presenting them.
7. Clearly distinguish historical expected return from future realized return.
8. Do not mutate the frozen Parquet file during analysis.

If an API is later added, the endpoint should call `run_mpt_analysis()` and
serialize `result.to_dict()`. It should not duplicate the MPT workflow.

## Downstream implementation status and handoff

`agent/` currently contains deterministic profile selection and fractional
allocation with display reconciliation. Both run offline. It calls no
LLM and has no model provider, prompts, or LLM dependencies configured.
`agent/__main__.py` calls the supported pipeline and prints profile JSON, or
allocation JSON when `--amount` is supplied; `agent/profiles.py` exposes
`select_profiles(payload)` and `ProfileResult`.

Consumers should use `select_profiles(run_mpt_analysis().to_dict())` and
`ProfileResult.to_dict()` rather than parsing the profile CLI. Preserve the
profile's metadata, selection rule, diversification diagnostics, warnings, and
disclaimer when passing it downstream. Profiles may overlap, and medium is a
sampled approximation to Maximum Sharpe, not necessarily the exact optimum.

The fractional allocation interface is
`agent.allocation.allocate_amount(amount, currency, payload, profile_name)`.
It selects profiles through the validated payload, and serializes money as
decimal strings while leaving rates and weights numeric. Keep target weights
unchanged and distinguish calculated amounts from rounded display amounts. Currency input must not imply an FX
conversion or live-price lookup. `AllocationResult.to_dict()` provides precise
`target_amounts` and a separate `display` view that reconciles minor units with
largest-remainder rounding. Display normalization, rounding, or dust grouping
must never change the target weights or historical metrics. Display amounts are
not effective traded weights. Whole-share allocation remains an optional
later extension requiring effective-weight validation as described above.

Future LLM orchestration belongs in `agent/` or a separate service, with its
dependencies separate from the baseline requirements. Its role may include
explaining tradeoffs, filtering already-computed efficient candidates, and
formatting results. It must not silently change weights, constraints, expected
returns, covariance, or the configured risk-free rate. LangGraph is proposed in
the roadmap; the provider and model remain undecided.

See `ROADMAP.md` for completed and planned milestones. Continue the established
workflow of small validated commits on `feature/ai-agentic-features`; update the
README and roadmap when a milestone changes the supported behavior.

## Required checks before completing a change

Run from the repository root with the virtual environment active:

```bash
python -m pytest -v
python -m src.main
git diff --check
git status
```

For programmatic-contract changes, also verify JSON serialization:

```bash
python -c "import json; from src.pipeline import run_mpt_analysis; json.dumps(run_mpt_analysis().to_dict()); print('JSON contract valid')"
```

Before changing configuration or a financial formula:

1. State the current assumption.
2. Explain the proposed assumption and its financial effect.
3. Update tests and documentation in the same change.
4. Confirm whether the frozen dataset must be regenerated.

Do not claim completion if optimization failed, target returns were not met,
weights violate constraints, tests fail, or documentation no longer matches the
result schema.
