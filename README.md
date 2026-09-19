# Baseline Markowitz Modern Portfolio Theory

This repository implements a transparent baseline version of Markowitz Modern
Portfolio Theory (MPT) for a B.Tech major project. It downloads a fixed period
of historical adjusted stock prices once, freezes them as a local Parquet
dataset, estimates historical returns and covariance, and constructs long-only
efficient portfolios with constrained numerical optimization.

The numerical MPT layer is independent from the downstream `agent/` package
and future LLM or user-interface code.

**Current milestone:** baseline hardening (Phase 0), deterministic risk profiles
(Phase 1), fractional allocation (Phase 2), and grounded agent explanations
(Phase 3) are complete. Optional LangGraph orchestration uses Groq's
`openai/gpt-oss-20b` through the OpenAI SDK Responses API. The model selects
approved fact references; Python renders their financial statements. Baseline,
profile, allocation, and default explanation commands run offline. Phase 4
(the API service) is next.
See [ROADMAP.md](ROADMAP.md) for progress,
remaining decisions, and the sequence of small feature-branch commits.

## Current scope

Implemented:

- Explicit Yahoo Finance data acquisition
- Frozen, locally stored adjusted-price dataset
- Daily simple percentage returns
- Historical arithmetic expected returns
- Sample covariance matrix
- Global Minimum Variance (GMV) portfolio
- Maximum Sharpe Ratio portfolio
- Minimum-variance portfolio for a target return
- Efficient upper frontier using repeated constrained optimization
- Efficient-frontier visualization
- Reusable structured Python/JSON result contract
- Synthetic tests for portfolio math, returns, loading, optimization and JSON output
- Deterministic low / medium / high profile selection from the sampled frontier
- Concentration flags, selection provenance, and an offline profile JSON command
- Fractional allocation Python interface and CLI with exact monetary targets
- Reconciled monetary display, rounding audit, and optional dust grouping
- Read-only analysis tools, cached analysis, and LangGraph explanation workflow
- Opt-in Groq Responses calls, reference guardrails, and offline fallback
- Fixed evaluation fixtures and CI with and without optional dependencies

Not implemented:

- Whole-share allocation, share counts, or currency conversion
- RAG or unrestricted financial question answering
- API service or GUI
- Machine learning or return prediction
- Sentiment or news analysis
- Black-Litterman or CAPM return estimation
- Random/Monte Carlo portfolio simulation
- Short selling or leverage
- Advanced covariance estimators

## Architecture

```text
Yahoo Finance
      |
      | explicit: python -m src.download_data
      v
data/raw/stock_prices.parquet
      |
      | no Yahoo Finance calls below this point
      v
data_loader.py -> returns.py -> portfolio.py -> optimizer.py
                                      |
                                      v
                                  pipeline.py
                                  /         \
                                 v           v
                         main.py (CLI)   agent/profiles.py
                                 |
                                 v
                         visualization.py
```

`src.pipeline.run_mpt_analysis()` is the supported programmatic integration
boundary. `src.main` is the human-facing command-line entry point.

The pipeline solves GMV once and passes the same weights into frontier generation.
Standalone calls to `generate_efficient_frontier()` still solve GMV themselves;
its optional `gmv_weights` argument is reserved for the unmodified GMV result
computed with the same covariance matrix. Reuse changes no financial assumptions
or serialized results and does not require regenerating the frozen dataset.

## Project structure

```text
FinanceAI/
├── AGENTS.md
├── README.md
├── requirements.txt
├── ROADMAP.md
├── agent/
│   ├── __init__.py
│   ├── __main__.py
│   ├── profiles.py
│   ├── allocation.py
│   ├── formatting.py
│   ├── cache.py
│   ├── tools.py
│   ├── explainer.py
│   ├── guardrails.py
│   ├── provider.py
│   ├── graph.py
│   ├── requirements.txt
│   └── prompts/system.md
├── data/
│   ├── raw/
│   │   └── stock_prices.parquet
│   └── processed/
├── outputs/
│   └── figures/
│       └── efficient_frontier.png
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── download_data.py
│   ├── data_loader.py
│   ├── returns.py
│   ├── portfolio.py
│   ├── optimizer.py
│   ├── pipeline.py
│   ├── visualization.py
│   └── main.py
└── tests/
    ├── conftest.py
    ├── test_portfolio.py
    ├── test_returns.py
    ├── test_data_loader.py
    ├── test_optimizer.py
    ├── test_pipeline.py
    └── agent/
        ├── test_profiles.py
        ├── test_allocation.py
        ├── test_formatting.py
        ├── test_cli.py
        ├── test_tools.py
        ├── test_guardrails.py
        ├── test_provider.py
        ├── test_graph.py
        └── fixtures/evaluations.json
```

## Financial method

For adjusted price \(P_t\), the daily simple return is:

```text
R_t = (P_t / P_(t-1)) - 1
```

The historical mean daily return and sample covariance are annualized using
252 trading days:

```text
mu_daily  = mean(daily returns)
mu_annual = mu_daily * 252

Sigma_daily  = sample covariance(daily returns)
Sigma_annual = Sigma_daily * 252
```

For weights \(w\), annual expected returns \(\mu\), and annual covariance
\(\Sigma\):

```text
Portfolio expected return = w^T mu
Portfolio variance        = w^T Sigma w
Portfolio volatility      = sqrt(w^T Sigma w)
Sharpe ratio              = (portfolio return - risk-free rate) / volatility
```

For ten stocks, the dimensions are:

```text
w      -> (10,)
mu     -> (10,)
Sigma  -> (10, 10)
```

In NumPy, `weights @ expected_returns` implements \(w^T\mu\), while
`weights.T @ covariance_matrix @ weights` implements \(w^T\Sigma w\).

The efficient frontier is generated by repeatedly minimizing variance for a
specified target return. Only targets from the GMV return upward are retained;
the lower branch is inefficient because it has higher risk and lower expected
return than GMV.

## Baseline assumptions

| Setting | Baseline value | Meaning |
| --- | ---: | --- |
| Stock universe | AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, JNJ, XOM, PG | Configurable US large-stock baseline |
| Start date | 2021-01-01 inclusive | Fixed historical window |
| End date | 2026-01-01 exclusive | Last possible observation is 2025-12-31 |
| Price field | Auto-adjusted close | Accounts for splits and dividend adjustments |
| Return type | Simple percentage return | Log returns are not used |
| Expected return | Historical arithmetic mean | Not a forecast |
| Covariance | Historical sample covariance | Pandas uses the N-1 denominator |
| Trading days | 252 | Annualization factor |
| Risk-free rate | 0.04 annually | Experimental Sharpe-ratio configuration |
| Weight sum | 1 | All capital is allocated |
| Weight bounds | 0 to 1 | No short selling or leverage |
| Frontier points | 50 | Numerical resolution, not a financial assumption |

Annualizing the arithmetic mean by 252 is not the same as calculating a
compounded annual return. Multiplying covariance by 252 assumes daily return
behavior is sufficiently stable and ignores material cross-day covariance.

## Dependencies

| Dependency | Purpose |
| --- | --- |
| NumPy | Vectors, matrices, and portfolio algebra |
| Pandas | Historical time-series tables and return statistics |
| PyArrow | Pandas Parquet read/write engine |
| SciPy | SLSQP constrained optimization |
| Matplotlib | Efficient-frontier figure |
| yfinance | Explicit acquisition stage only |
| pytest | Unit tests |

Exact tested direct-dependency versions are recorded in `requirements.txt`.

## Development setup

The current baseline was tested with Python 3.13.2 on macOS.

```bash
git clone <repository-url>
cd FinanceAI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

In VS Code, run **Python: Select Interpreter** from the Command Palette and
choose `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on
Windows.

Verify imports:

```bash
python -c "import numpy, pandas, pyarrow, scipy, matplotlib, yfinance, pytest; print('Imports succeeded')"
```

## Local API-key storage

Store `GROQ_API_KEY=your_key_here` in `.env` at the repository root. An optional
`GROQ_MODEL` defaults to `openai/gpt-oss-20b`. Shell environment values take
precedence over `.env`. `OPENAI_API_KEY` is not used by the Groq adapter.
The ignored `.env` and `.env.*` files remain local; `.env.example` is a tracked
template without secrets. Only explicit LLM use loads these settings.

Install the optional client/framework dependencies separately:

```bash
python -m pip install -r agent/requirements.txt
```

`agent/provider.py` uses the OpenAI SDK with Groq's
`https://api.groq.com/openai/v1` endpoint and `client.responses.create()`.
Tool retrieval and strict structured output use separate requests, following
[Groq's Responses API](https://console.groq.com/docs/responses-api) and
[structured-output limitations](https://console.groq.com/docs/structured-outputs).
Each explanation uses at most two requests, with a 30-second timeout per
request, no SDK retries, and at most four read-only tool calls. Provider errors
are reduced to stable codes; API keys and raw provider error bodies are not
included in results. Local dataset paths are omitted from tool data sent to Groq.

## Create the frozen dataset

Data acquisition is always explicit:

```bash
python -m src.download_data
```

The script prints the response shape, ticker coverage, date ranges, missing
values, rows removed during common-date alignment, and final validation status
before saving `data/raw/stock_prices.parquet`. It then reopens the Parquet file
and compares it with the validated in-memory data.

If the file already exists, the command stops before contacting Yahoo Finance.
Replace it only when deliberately changing the frozen experiment:

```bash
python -m src.download_data --overwrite
```

Changing the ticker universe or dates in `src/config.py` requires regenerating
the dataset explicitly. The Parquet file is ignored by Git, so a fresh clone
must either run the acquisition command or receive the frozen file through an
approved project data-sharing mechanism.

## Inspect intermediate calculations

Verify the local dataset without contacting Yahoo Finance:

```bash
python -m src.data_loader
```

Inspect sample daily returns, annual expected returns, and covariance
dimensions:

```bash
python -m src.returns
```

These commands do not alter the frozen dataset.

## Run the complete baseline

```bash
python -m src.main
```

This command:

1. Loads `data/raw/stock_prices.parquet`.
2. Calculates daily simple returns and annual MPT inputs.
3. Finds GMV and Maximum Sharpe portfolios.
4. Generates the efficient upper frontier.
5. Prints portfolio metrics and weights.
6. Saves `outputs/figures/efficient_frontier.png`.

It never contacts Yahoo Finance. The generated PNG is ignored by Git.

For the current frozen dataset, the verified headline values are approximately:

| Portfolio | Expected return | Volatility | Sharpe ratio |
| --- | ---: | ---: | ---: |
| Global Minimum Variance | 13.9390% | 12.4798% | 0.7964 |
| Maximum Sharpe Ratio | 34.0434% | 19.8070% | 1.5168 |

These values are in-sample historical estimates, not predictions or investment
recommendations. They will change if the frozen dataset or configuration is
changed.

## Programmatic integration contract

Applications and future agentic components should import the reusable pipeline;
they should not invoke `main()` or parse terminal output.

```python
from src.pipeline import run_mpt_analysis

result = run_mpt_analysis()

print(result.gmv_portfolio.weights)
print(result.maximum_sharpe_portfolio.sharpe_ratio)
print(result.efficient_frontier[0].volatility)
```

`run_mpt_analysis()` returns an `MPTAnalysisResult` containing:

- Dataset path, asset names, observation count, and date range
- Trading-day and risk-free-rate assumptions
- Annual expected-return Series
- Annual covariance DataFrame
- Evaluated GMV portfolio
- Evaluated Maximum Sharpe portfolio
- All verified efficient-frontier candidates

For a language-independent service payload, call:

```python
import json

from src.pipeline import run_mpt_analysis

payload = run_mpt_analysis().to_dict()
json_text = json.dumps(payload)
```

The dictionary has these top-level keys:

```text
metadata
annual_expected_returns
annual_covariance
portfolios
efficient_frontier
```

Portfolio and frontier weights are mappings from ticker to weight. All rates
and weights are numeric decimals: `0.25` means 25%. Formatting them as percentage
strings should happen only in the consuming presentation layer.

The `metadata` section includes the assumptions and constraints needed for
provenance. An agent should present those assumptions alongside results and
must not describe historical expected returns as forecasts.

See `AGENTS.md` for a direct working contract intended for coding agents that
extend or consume this repository.

## Deterministic risk profiles (roadmap Phase 1)

The `agent/` package consumes the supported pipeline payload. This first step
requires no LLM, API key, network connection, or additional dependencies:

```bash
python -m agent
```

`agent/__main__.py` calls `run_mpt_analysis().to_dict()`, passes the result to
`agent.profiles.select_profiles()`, and prints JSON. With `--amount`, it calls
`agent.allocation.allocate_amount()` and prints the allocation instead.
`agent/profiles.py` applies
fixed selection rules and uses `src.portfolio` to validate metrics and compute
Sharpe. These original commands do not load prompts or call a model; the new
`--explain` route invokes the workflow described below.

It prints full-precision JSON for three relative in-sample risk profiles. To use
these in another Python component:

```python
from agent.profiles import select_profiles
from src.pipeline import run_mpt_analysis

profiles = select_profiles(run_mpt_analysis().to_dict())
medium = profiles["medium"].to_dict()
```

| Profile | Selection from `efficient_frontier` |
| --- | --- |
| Low | First point, at GMV |
| Medium | Point nearest Maximum Sharpe volatility |
| High | Interior point at or above medium nearest the volatility midpoint between Maximum Sharpe and the endpoint |

Medium approximates Maximum Sharpe because the exact optimizer solution may
lie between sampled points. Equal-distance ties choose the earlier index.
If high cannot select a distinct interior point, it reuses medium and flags the
overlap. If medium itself lands on the endpoint, high also reuses it and both
carry an endpoint warning. There is no guarantee of three distinct portfolios
or a diversified high profile.

Each `ProfileResult.to_dict()` includes `name`, `expected_return`, `variance`,
`volatility`, `sharpe_ratio`, ticker-keyed `weights`, complete `metadata`,
`selection`, `diversification`, `warnings`, and `disclaimer`. Selection includes
the original frontier index, rule, parameters, and overlaps. Metadata retains
the dataset window, trading days, risk-free rate, and constraints. Consumers
should display this provenance and disclaimer alongside the numbers.

All weights and existing metrics are copied unchanged. Sharpe is evaluated with
`src.portfolio.calculate_sharpe_ratio` using the payload's configured risk-free
rate. The selector checks that supplied metrics agree with their weights and
rejects invalid inputs. A weight above 40%, or fewer than three holdings above
1%, produces a diagnostic flag; neither changes the allocation or optimizer
constraints. Rates and weights remain decimal numbers, with rounding reserved
for presentation. The baseline JSON schema is unchanged.

These profiles describe historical estimates within the frozen universe. They
are not forecasts or personalized investment advice, and “low” does not mean
safe. Fractional allocation and grounded explanations are available below.
See `ROADMAP.md` for remaining phases.

## Fractional amount allocation (roadmap Phase 2)

Both the Python interface and CLI run offline. The original `python -m agent`
command still prints all three profiles. Supply an amount for an allocation:

```bash
python -m agent --amount 10000.00 --currency USD --profile medium
python -m agent --amount 10000.00 --currency INR --profile high --hide-dust
```

With `--amount`, currency defaults to USD and profile to medium. Currency,
profile and dust options require an amount. Invalid inputs fail before analysis.
The Python interface consumes a baseline analysis payload and selects a
validated profile, rejecting modified weights with stale metrics:

```python
from agent.allocation import allocate_amount
from src.pipeline import run_mpt_analysis

allocation = allocate_amount("10000.00", "USD", run_mpt_analysis().to_dict(), "medium")
payload = allocation.to_dict()
```

Inputs accept a decimal string, integer, float, or `Decimal`. Prefer strings or
`Decimal` for money. Amounts must be positive, in whole cents/paise, and no greater
than `999999999999.99` (an application input limit). USD and INR are supported
labels; no currency conversion or live-price request occurs.

`AllocationResult.to_dict()` contains `mode`, `amount`, `currency`,
`target_profile`, `target_amounts`, `numerical_residual`, `display`, and `notice`.
The complete source profile retains its metadata, warnings and disclaimer. Monetary values
serialize as decimal strings to preserve precision; weights and portfolio metrics
retain their existing numeric representation. Target amounts multiply the input
by each weight's full round-trip decimal representation, without normalizing
weights. `numerical_residual` records tiny weight-sum drift, not uninvested cash.
All target holdings, including dust, remain present. These are target amounts,
not trades or share counts. No financial assumption or frozen dataset changes.

The separate `display` view uses two decimal places and always reconciles to the
input amount. For presentation only, negative targets within the baseline's
numerical tolerance become zero and positive target amounts are proportionally
scaled to remove weight-sum drift. Each amount is floored to whole cents/paise,
then remaining units go to the largest fractional remainders. Alphabetical ticker
order breaks ties. `rounding_adjustments` records each difference from the exact
target, and `method` and `notice` explain the procedure. Target weights and their
historical metrics remain unchanged; display values are not an effective traded
portfolio.

All display holdings are visible by default. `--hide-dust` (or
`allocation.to_dict(hide_dust=True)`) groups target holdings below 0.5% into
`display.hidden_assets` and `display.hidden_amount`. Visible `display.amounts`
plus the hidden amount equal `display.total`, which equals the input amount.
Hidden holdings are allocated money, not leftover cash. Full-precision targets
always include every ticker. Whole-share mode and currency conversion remain
unimplemented.

## Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Tests cover portfolio mathematics, arithmetic returns and sample covariance,
local dataset validation, optimizer feasibility and frontier ordering, and the
JSON pipeline contract. Profile tests cover selection rules, ordering,
concentration and overlap warnings, ticker mapping, configured risk-free rates,
invalid inputs, provenance, and copy isolation. Synthetic inputs and temporary
Parquet fixtures keep the suite independent of live data and the git-ignored
frozen dataset. The pipeline test rejects network connections and verifies that analysis leaves
the input file unchanged.

The Phase 3 checkpoint has 184 passing tests with the optional dependencies
installed. Agent tests block network connections, use fake Responses clients,
and cover read-only tools, reference validation, fixed payload evaluations,
unchanged profiles, fallback behavior, and operation without optional dependencies.
CI runs a baseline-only environment and one with agent dependencies; tests of
installed SDK/framework behavior are skipped in the baseline-only job.
Before completing each change,
run the repository checks with the virtual environment active:

```bash
python -m pytest -v
python -m src.main
python -c "import json; from src.pipeline import run_mpt_analysis; json.dumps(run_mpt_analysis().to_dict(), allow_nan=False); print('JSON contract valid')"
python -m agent
python -m agent --amount 10000 --currency USD --profile medium
python -m agent --explain
git diff --check
git status
```

Tests use synthetic data; the CLI commands and the JSON contract check use
the frozen local Parquet file. They do not download or regenerate it. Continue
development on `feature/ai-agentic-features` with a focused, validated commit for
each step. Implementation boundaries are documented in [AGENTS.md](AGENTS.md).

## Reproducibility and limitations

- The ticker universe is selected in advance, so universe-selection and
  survivorship considerations remain.
- Historical mean returns are noisy and are not reliable forecasts by
  themselves.
- Sample covariance is sensitive to the selected period.
- Transaction costs, taxes, liquidity, position-size limits, and turnover are
  not modeled.
- The 4% risk-free rate is a fixed experiment setting, not a live market rate.
- Adjusted prices represent a total-return-oriented history; raw closes are not
  used.
- Rows are aligned to dates available for every asset.
- Optimizer output can contain extremely small numerical weights near zero;
  retain full precision for calculations and round only for display.

This baseline is intended to remain understandable and auditable. Any change
to a financial assumption should be made explicitly in configuration, tested,
and documented before downstream systems consume the new results.

## Grounded explanations (roadmap Phase 3)

```bash
# Offline comparison, with optional allocation
python -m agent --explain
python -m agent --explain --amount 10000 --currency USD --profile medium

# Explicit Groq calls; optional dependencies and GROQ_API_KEY required
python -m agent --explain --llm
python -m agent --explain --llm --question "What are the high profile's weights?"
python -m agent --explain --llm --question "Show frontier point 2 (zero-based)."
```

`--question` requires `--explain --llm` and accepts 1–2000 characters.
An explanation result contains `profiles`, optional `allocation` (otherwise
`null`), `explanation`, and `orchestration`. The explanation preserves full
metadata locally and includes `text`, `fact_ids`, `disclaimer`, `status`, `mode`,
`provider`, `model`, `fallback_reason`, `tool_calls`, and `question_answered`.
`mode` distinguishes `offline`, `llm`, and `fallback`; `status` is `supported` or
`unsupported`. `question_answered` records whether the model marked a supplied
question supported; it is not a guarantee of relevance or completeness.

The LangGraph nodes are `load_analysis → build_profiles → allocate → explain →
guardrail`. Without LangGraph, the same nodes run sequentially and
`orchestration` reports `sequential`. Without credentials or a client dependency,
or on provider/validation failure, explicit LLM requests return a general
deterministic comparison with `mode=fallback` and a reason code. Such a comparison
does not answer the follow-up question. Invalid baseline data or invalid monetary
inputs raise errors; they are never hidden by an explanation fallback.

```python
from agent.graph import run_agent

result = run_agent(amount="10000", currency="USD", profile="medium")  # offline
text = result["explanation"]["text"]
# Pass payload=run_mpt_analysis().to_dict() to consume an existing snapshot.
# Add use_llm=True and optionally question="..." to enable Groq explicitly.
```

Read-only tools are available through `agent.tools.AnalysisTools(payload)`:
`get_profile(name)`, `get_asset_stats(ticker)`, `get_frontier_point(index)`, and
`get_metadata()`. They return isolated copies and cannot edit portfolios or
request a download. `agent.explainer.explain_offline(payload)` produces a
comparison with all profiles, warnings, provenance and the historical disclaimer.
`agent.cache.load_analysis()` caches the baseline payload per process and frozen
file stat; clear the cache or restart after configuration changes.

The model may select up to twelve known fact IDs. Python always adds all three
profile metrics, applicable warnings, provenance, and the historical disclaimer.
It validates the plan and verifies exact rendered content, so a model cannot
introduce its own figures or swap otherwise-valid numbers between labels.
This is a constrained explanation system: it does not generate arbitrary prose,
calculate new comparisons, forecast returns, or make personalized recommendations.
Fact grounding does not guarantee that the model selects the most relevant facts.
Unsupported questions receive an explicit notice and the available comparison.

Only `--llm` sends the question and approved analysis facts (including allocation
amounts, if supplied) to Groq. Original local metadata remains in returned JSON;
tool results sent to Groq omit `dataset_path`. No browser, code execution, market
data, trade, or file-writing tool is exposed. Live smoke checks passed for both
a general comparison and a frontier-point tool call using `openai/gpt-oss-20b`.
