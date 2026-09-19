# FinanceAI — Implementation Status and Roadmap

> Scope: a deterministic Markowitz MPT baseline, extended by an agentic layer that
> reads the efficient frontier and presents three risk-profiled allocations, plus a
> GUI. This document follows the working contract in `AGENTS.md`; every planned item
> below is placed so that it does **not** break that contract.

**Current checkpoint:** Phase 0 required work, Phase 1, fractional Phase 2, and
Phase 3 are complete on `feature/ai-agentic-features` (184 passing tests with
optional dependencies installed). Groq Responses calls use `openai/gpt-oss-20b`
through the OpenAI SDK, with LangGraph orchestration. Model-selected fact IDs
are rendered and validated in Python. LLM calls require explicit opt-in;
baseline, profiles, allocations, and default explanations remain offline.

| Milestone | Status | Next action |
| --- | --- | --- |
| Phase 0: baseline hardening | Complete; optional display metadata deferred | Preserve the verified numerical baseline |
| Phase 1: deterministic profiles | Complete | Consume the existing structured profile contract |
| Phase 2: fractional allocation | Complete; optional whole shares deferred | Consume exact targets and reconciled display |
| Phase 3: LLM explanations | Complete | Preserve grounded facts and explicit offline fallback |
| Phases 4–6: API, GUI, research | Planned; not implemented | Build after the profile and allocation contracts stabilize |

---

## 1. Guiding rules (from `AGENTS.md`)

These constraints shape every design decision in Section 4.

| Rule | Consequence for the roadmap |
| --- | --- |
| Baseline stays deterministic; no LLM computes returns, covariance, metrics or weights | Deterministic code selects from `efficient_frontier`; future LLMs explain supplied results and never calculate financial figures |
| Dependency direction is one-way | `agent/`, `api/`, `gui/` may import `src.pipeline`; nothing in `src/` imports them |
| Consume results only via `run_mpt_analysis().to_dict()` | No parsing of CLI output or `efficient_frontier.png` |
| Yahoo Finance only in `src/download_data.py` | Future live-price features need a separate service; Yahoo access remains confined to `src/download_data.py` |
| Weights are decimals, full precision; round only for display | Formatting happens in `agent/formatting` and `gui/` only |
| Externally modified weights must be re-validated with `src.portfolio` | Any rounding to whole shares triggers re-evaluation before display |
| Historical expected return ≠ forecast; not personalized advice | Mandatory disclaimer + provenance block in every agent/GUI output |
| Financial-assumption changes need explicit approval, tests and docs | Items marked **[needs approval]** below |

---

## 2. What is implemented

### 2.1 Current repository structure

```text
FinanceAI/
├── AGENTS.md                  # Coding-agent contract
├── README.md                  # Method, setup, integration contract
├── requirements.txt           # Pinned baseline deps
├── data/raw/                  # Frozen stock_prices.parquet (git-ignored)
├── data/processed/            # Empty (reserved)
├── outputs/figures/           # efficient_frontier.png (git-ignored)
├── src/
│   ├── config.py              # Tickers, dates, 252 days, rf = 0.04, 50 frontier points
│   ├── download_data.py       # Explicit Yahoo download → validate → Parquet → round-trip check
│   ├── data_loader.py         # Offline Parquet load + schema validation
│   ├── returns.py             # Simple daily returns, annual mean, annual sample covariance
│   ├── portfolio.py           # w·μ, wᵀΣw, volatility, Sharpe, weight validation
│   ├── optimizer.py           # SLSQP: GMV, Max Sharpe, target-return, efficient frontier
│   ├── pipeline.py            # run_mpt_analysis() → MPTAnalysisResult / to_dict()
│   ├── visualization.py       # Frontier plot with GMV and Max Sharpe markers
│   └── main.py                # CLI summary + figure
├── agent/
│   ├── profiles.py            # Deterministic sampled-profile selection
│   ├── __main__.py            # Profiles, allocations, opt-in LLM explanations
│   ├── allocation.py          # Exact fractional monetary targets
│   ├── formatting.py          # Reconciled monetary display
│   ├── tools.py               # Read-only snapshot operations
│   ├── cache.py               # Isolated cached pipeline payload
│   ├── explainer.py           # Approved facts and deterministic rendering
│   ├── guardrails.py          # Plan and exact output validation
│   ├── provider.py            # Bounded Groq Responses requests
│   ├── graph.py               # LangGraph nodes and sequential fallback
│   ├── requirements.txt       # Optional SDK/framework dependencies
│   └── prompts/system.md
└── tests/
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

### 2.2 Feature status

| Area | Status | Notes |
| --- | --- | --- |
| Data acquisition (10 US large caps, 2021-01-01 → 2025-12-31) | ✅ Done | Explicit, refuses overwrite without `--overwrite`, missing-data and min-observation checks |
| Frozen offline dataset loading | ✅ Done | Validates index, ticker order, NaN, non-positive prices |
| Simple returns, μ, Σ (annualized ×252) | ✅ Done | Sample covariance (N−1) |
| Portfolio math + weight validation | ✅ Done | Long-only, sum-to-1, tolerance 1e-8 |
| Global Minimum Variance | ✅ Done | Scaled objective, result re-validated |
| Maximum Sharpe | ✅ Done | Negative-Sharpe SLSQP |
| Target-return minimum variance | ✅ Done | Feasibility check, warm start, achieved-return check |
| Efficient frontier (upper branch, 50 points) | ✅ Done | Warm-started from previous point; monotonic-volatility check |
| Structured JSON contract | ✅ Done | `metadata`, `annual_expected_returns`, `annual_covariance`, `portfolios`, `efficient_frontier` |
| Frontier plot | ✅ Done | Static PNG |
| CLI | ✅ Done | `python -m src.main` |
| Unit tests | ✅ Baseline covered | Synthetic tests for portfolio math, returns, loader, optimizer and JSON pipeline |
| Agentic layer | ✅ Phase 3 done | Groq, LangGraph, read-only tools, approved fact references and offline fallback |
| Risk profiles (low / medium / high) | ✅ Done | Sampled candidates, provenance, concentration and overlap flags |
| Amount-based allocation | ✅ Fractional mode done | Exact targets, reconciled display, USD/INR labels, CLI; whole shares deferred |
| API service | ❌ Not started | — |
| GUI | ❌ Not started | — |

### 2.3 Observations from reviewing the code

Original review findings and their current resolution:

1. **Frontier points have no Sharpe ratio.** `efficient_frontier` entries contain return, variance, volatility and weights, but not Sharpe. The agent needs it for comparisons. Resolved downstream: `agent.profiles` uses `src.portfolio.calculate_sharpe_ratio`; the baseline JSON schema is unchanged.
2. **The top of the frontier is a single stock.** The last frontier point targets `max(μ)`, which with long-only constraints is 100% in the highest-return asset (a synthetic run confirmed this). A "high risk" option taken naively from the end of the frontier would be an undiversified one-stock portfolio. See Phase 1 for interior selection and explicit endpoint/overlap warnings.
3. **Max Sharpe is not guaranteed to be one of the 50 frontier points.** It lies on the frontier curve but usually between sampled points. Resolved: medium selects the nearest frontier point by volatility, preserving the `AGENTS.md` frontier-only rule.
4. **Resolved: baseline test coverage.** Synthetic tests now cover returns, optimization, local loading and the pipeline JSON contract, without the real dataset.
5. **Resolved: duplicate GMV solve.** The pipeline passes its computed GMV weights into frontier generation. Standalone frontier calls still solve GMV themselves; regression tests verify identical values and one solve per pipeline run.
6. **`metadata.dataset_path` is an absolute local path.** Fine for CLI, but an API should not expose server filesystem paths to clients.
7. **No stock-level descriptive data** (names, sectors, per-asset volatility, price history) is in the payload, yet the GUI needs it to "show the user all the stocks". Per-asset volatility is derivable from the diagonal of `annual_covariance`; names and sectors need a static metadata file.

---

## 3. Target architecture

The baseline, deterministic profiles, fractional allocation, display formatting,
and CLI entry points exist today. Tools, LLM explanations, guardrails, API, and
GUI components below are planned.

```text
Yahoo Finance
      │  explicit only: python -m src.download_data
      ▼
data/raw/stock_prices.parquet
      │  (no network below this line inside src/)
      ▼
src/  (deterministic MPT baseline — unchanged ownership)
      │  run_mpt_analysis().to_dict()
      ▼
agent/  (separate downstream package)
  ├── profiles.py      deterministic low / medium / high selection
  ├── allocation.py    amount → per-stock allocation, re-validated
  ├── tools.py         read-only tools exposed to the LLM
  ├── explainer.py     LLM: explanation + formatting only
  └── guardrails.py    output checks (numbers match payload, disclaimer present)
      │
      ▼
api/  (FastAPI)  ──►  gui/  (web front end)
```

Key principle: **baseline portfolio calculations remain in `src/`.** Downstream
code may call those evaluators and deterministically convert weights into amounts.
The LLM receives already-computed facts and selects references; Python renders
the corresponding statements. It never calculates or modifies financial figures.

---

## 4. What must be implemented

### Phase 0 — Harden the baseline (inside `src/` and `tests/`)

**Status: required work complete.** Optional exports below are deferred.

- [x] `tests/test_returns.py` — simple-return formula, annualization, covariance symmetry, NaN rejection.
- [x] `tests/test_optimizer.py` — on a synthetic 3–5 asset case: GMV weights valid; Max Sharpe ≥ Sharpe of every frontier point (within tolerance); frontier returns increasing and volatilities non-decreasing; infeasible target raises.
- [x] `tests/test_pipeline.py` — `to_dict()` is JSON-serializable, has the documented keys, weights keyed by ticker and summing to 1. Use a small temporary Parquet fixture so tests do not depend on the git-ignored real dataset.
- [x] `tests/test_data_loader.py` — missing file, wrong ticker order, duplicate dates, non-positive prices.
- [x] Compute profile Sharpe downstream through `src.portfolio`; preserve the baseline frontier schema.
- [x] Reuse the pipeline GMV solution during frontier generation, with regression checks for unchanged results.
- [ ] Optional: expose per-asset annual volatility in `to_dict()` (derived from Σ diagonal, no new assumption).
- [ ] Optional: `data/reference/tickers.json` with company name and sector for display (static, no network).

**Exit criteria:** all `AGENTS.md` required checks pass, including the JSON contract check.

### Phase 1 — Deterministic risk profiles (`agent/profiles.py`)

**Status: complete.** This is plain Python with no LLM. It selects three
labelled options from the frontier; they are not guaranteed to be distinct.

| Profile | Implemented selection rule | Rationale |
| --- | --- | --- |
| Low risk / low return | `efficient_frontier[0]` | Lowest-volatility sampled efficient portfolio |
| Medium risk / medium return | Sampled frontier point nearest Maximum Sharpe volatility | Respects the frontier-only consumer contract; an approximation to Maximum Sharpe |
| High risk / high return | Interior point at or above medium nearest the volatility midpoint between Maximum Sharpe and the endpoint; reuse medium if no eligible interior point exists | Avoid forcing the endpoint solely to obtain a distinct option |

Ties choose the earlier frontier index. High may select medium even when other
interior points exist if medium is nearest the target. Overlaps are flagged.
If medium itself is the endpoint, high reuses it and both receive endpoint
warnings. Concentration flags never change weights; high may remain concentrated.

Tasks:

- [x] Implement `select_profiles(payload) -> dict[str, ProfileResult]` using only values already in the payload.
- [x] Recompute Sharpe for any frontier point with `src.portfolio.calculate_sharpe_ratio` (rule 6).
- [x] Flag any weight strictly above 40%, or fewer than three holdings strictly above 1%. Flag only; do not alter weights.
- [ ] **[needs approval]** If a hard concentration cap (e.g. max 30% per stock) is wanted, it must be a new constraint in `src/optimizer.py` with config, tests and README updates, because it changes the frontier itself. It must not be applied by editing weights in the agent.
- [x] Record the selection rule and its parameters in the profile output so the GUI can show *why* each option was chosen.
- [x] `tests/agent/test_profiles.py` — ordering guarantee: low volatility ≤ medium ≤ high, and low return ≤ medium ≤ high.

### Phase 2 — Amount-based allocation (`agent/allocation.py`)

**Status: fractional mode complete.** The Python interface and CLI produce
precise targets and a reconciled display view. These are target amounts, not
executable trades or share counts.
Keep whole-share mode outside this first allocation milestone.

- [x] Define the structured allocation result, preserving profile provenance, warnings and disclaimer.
- [x] Validate a finite positive investment amount and define supported currency labels. Currency labels alone do not perform FX conversion.
- [x] Define deterministic display rounding and residual handling so displayed amounts reconcile to the input without editing target weights.
- [x] Input: investment amount + currency + chosen profile.
- [x] **Fractional mode (default):** `amount × weight` per ticker. Weights unchanged, so no re-validation issue.
- [ ] **Whole-share mode (optional):** needs a current price per ticker. Current prices must come from a separate market-data service (e.g. `services/market_data/`), never from `src/`. Rounding down to whole shares leaves residual cash and changes effective weights, so:
  - [ ] recompute effective weights over the invested portion,
  - [ ] re-evaluate return / volatility / Sharpe with `src.portfolio`,
  - [ ] show both the target and the effective portfolio, plus leftover cash.
- [x] Optional `--hide-dust` groups weights below 0.5% in display only; retain every ticker in precise targets and report the hidden monetary total.
- [x] Fractional-mode tests: invalid amounts/currency, ticker mapping, unchanged weights, full-precision amounts, deterministic rounding, and displayed totals.
- [ ] Whole-share tests, when implemented: integer shares, effective-weight metrics and leftover cash.

### Phase 3 — Agentic layer (`agent/`)

**Status: complete.** At the user's request, use Groq's OpenAI-compatible
Responses endpoint with `openai/gpt-oss-20b`, configured with `GROQ_API_KEY`.
This supersedes the earlier OpenAI-hosted model recommendation. LangGraph
orchestrates the workflow; the LLM selects approved facts, which Python renders.
The client and framework dependencies are optional and isolated from `src/`.

The LLM's allowed role per `AGENTS.md`: explain trade-offs, filter already-computed candidates, format results.

- [x] Document Groq endpoint, model, credentials, bounded requests, and failure behavior.
- [x] Implement and test the read-only tools and deterministic explanation fallback independently of a hosted model.
- [x] **Framework:** optional LangGraph, with a small graph (same nodes run sequentially without it):
  1. `load_analysis` — call `run_mpt_analysis().to_dict()` (cached).
  2. `build_profiles` — Phase 1 function.
  3. `allocate` — Phase 2 function, if the user gave an amount.
  4. `explain` — choose an approved reference plan; fall back to a template on model/plan failure.
  5. `guardrail` — render verified facts and check exact content, provenance, and disclaimer.
- [x] **Read-only tools** for follow-up questions: `get_profile(name)`, `get_asset_stats(ticker)`, `get_frontier_point(index)`, `get_metadata()`. Tools may return existing weights; no tool creates or modifies weights.
- [x] **Prompt design:** historical estimates only; no invented/adjusted figures, new calculations, forecasts, or personalized recommendations. Python mandates provenance and disclaimer.
- [x] **Guardrails (`guardrails.py`):** enforce approved fact references instead of validating arbitrary prose:
  - [x] every percentage/amount is rendered directly from validated data (with display rounding),
  - [x] no model-invented tickers or weights,
  - [x] no model-written forecast language can enter final text,
  - [x] disclaimer, provenance, profile summaries, and applicable warnings are mandatory.
- [x] **Deterministic fallback:** works without the LLM, credentials, SDK, or graph dependency; emits explicit mode/reason codes. Invalid baseline data still fails.
- [x] Separate `agent/requirements.txt` so LLM dependencies never enter the baseline environment.
- [x] Evaluation set: three fixed payloads with expected checks, run in CI alongside fake-provider tests. CI tests both baseline-only and optional-agent installations.
- [x] CLI opt-in `--explain --llm`, optional `--question`, and live Groq verification (overview and frontier tool call).

Implementation limits: at most two requests per explanation, 30-second request
timeout, no SDK retries, and four tool calls in one round. Tool retrieval and
structured plans are separate requests because Groq does not combine them.
The reference catalogue guarantees factual grounding, not question relevance;
`question_answered` records the model's support decision, not an independent
semantic assessment. Offline/fallback output is a general comparison.

### Phase 4 — API service (`api/`)

- [ ] FastAPI app that calls `run_mpt_analysis()` once at startup (the dataset is frozen) and caches the payload.
- [ ] Endpoints:
  - `GET /analysis` — full `to_dict()` payload (with `dataset_path` removed or made relative),
  - `GET /profiles` — three deterministic profiles,
  - `POST /allocate` — `{amount, profile, mode}` → allocation,
  - `POST /explain` — agent explanation for the profiles / allocation,
  - `GET /health`.
- [ ] Pydantic schemas mirroring the JSON contract; contract test against `to_dict()`.
- [ ] Dockerfile and compose file (baseline + API; dataset mounted as a read-only volume).

### Phase 5 — GUI (`gui/`)

- [ ] **Stack:** React front end calling the API (fits the existing MERN experience); Streamlit is a faster alternative for an academic demo.
- [ ] **Screens:**
  - [ ] *Universe view* — table of the 10 stocks: name, sector, annual expected return, annual volatility, correlation heatmap from Σ.
  - [ ] *Interactive frontier* — rendered from `efficient_frontier` data (not the PNG); GMV, Max Sharpe and the three profiles highlighted; hover shows weights.
  - [ ] *Profile cards* — low / medium / high with return, volatility, Sharpe, weight donut chart and diversification flags.
  - [ ] *Allocation calculator* — amount input → per-stock amounts (and share counts if Phase 2 whole-share mode exists).
  - [ ] *Explanation panel* — agent text, with a persistent assumptions box (dataset dates, 252 days, rf, long-only, "historical, not a forecast").
- [ ] All percentage formatting happens here, never upstream.

### Phase 6 — Validation and research quality

These make the results defensible in a report or viva.

- [ ] **Out-of-sample backtest** (separate `research/` package): estimate μ, Σ on a training window, hold weights over a later test window, and compare realized return/volatility of the three profiles with equal-weight and GMV baselines. This is the most important evidence of whether the profiles behave as labelled.
- [ ] Sensitivity analysis: how weights change with a different date window or risk-free rate.
- [ ] **[needs approval]** Possible extensions, each as an explicit, documented, tested option rather than a silent replacement: Ledoit-Wolf covariance shrinkage, Black-Litterman returns, per-asset weight caps, transaction costs, periodic rebalancing, a larger or Indian (NSE) ticker universe.

---

## 5. Proposed final structure

```text
FinanceAI/
├── AGENTS.md
├── README.md
├── ROADMAP.md                     # this file
├── requirements.txt               # baseline only (unchanged)
├── data/
│   ├── raw/stock_prices.parquet
│   ├── processed/
│   └── reference/tickers.json     # NEW: names, sectors (static)
├── outputs/figures/
├── src/                           # baseline — ownership table unchanged
│   └── ...
├── agent/                         # Exists: profiles, allocation, explanations
│   ├── __init__.py
│   ├── requirements.txt           # Optional LLM/framework dependencies
│   ├── __main__.py                # Offline default, explicit --explain --llm
│   ├── profiles.py                # Exists
│   ├── allocation.py              # Exists: fractional targets
│   ├── formatting.py              # Exists: monetary display
│   ├── tools.py
│   ├── cache.py
│   ├── provider.py
│   ├── graph.py
│   ├── explainer.py
│   ├── guardrails.py
│   └── prompts/
│       └── system.md
├── services/
│   └── market_data/               # NEW, optional — live prices for whole-share mode
├── api/                           # NEW
│   ├── main.py
│   ├── schemas.py
│   └── Dockerfile
├── gui/                           # NEW — React or Streamlit
├── research/                      # NEW — backtests, sensitivity studies
└── tests/
    ├── test_portfolio.py
    ├── test_returns.py            # Exists
    ├── test_optimizer.py          # Exists
    ├── test_pipeline.py           # Exists
    ├── test_data_loader.py        # Exists
    ├── agent/
    │   ├── test_profiles.py
    │   ├── test_allocation.py
    │   ├── test_formatting.py
    │   ├── test_cli.py
    │   └── test_guardrails.py
    └── api/
        └── test_endpoints.py
```

---

## 6. Decisions and remaining choices

| Decision | Options | Affects |
| --- | --- | --- |
| High-risk selection rule | Selected: interior volatility midpoint, with explicit overlap fallback | Phase 1 |
| Concentration limits | Selected: flag only; a future hard cap needs approval | Phases 0–1 |
| Sharpe in frontier contract | Selected: compute downstream with baseline math | Phases 0–1 |
| Allocation mode | Implemented: fractional amounts; whole shares remain an optional later extension | Phase 2 |
| LLM framework | Implemented: optional LangGraph, same-node sequential fallback | Phase 3 |
| LLM provider and model | Implemented: Groq Responses, `openai/gpt-oss-20b`, OpenAI SDK | Phase 3 |
| GUI stack | React · Streamlit | Phase 5 |
| Ticker universe | Current 10 US stocks · larger / NSE universe (requires re-download) | All |

---

## 7. Definition of done (per change)

From `AGENTS.md`, run from the repository root:

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

Additionally for agent / API / GUI changes:

- profile ordering tests pass,
- guardrail and fixed evaluation tests pass,
- every user-facing output shows the dataset window, risk-free rate, constraints and the statement that figures are historical estimates rather than forecasts or personalized investment advice.

## 8. Incremental implementation decisions

The completed first milestone covers Phase 0 hardening and Phase 1
deterministic profiles. Work stays on `feature/ai-agentic-features`, with separate commits for roadmap
decisions, baseline tests, duplicate-GMV cleanup, and profile selection.

- All profile weights come unchanged from `efficient_frontier`, as required by
  `AGENTS.md`. Medium uses the nearest sampled point by volatility to Maximum
  Sharpe, breaking ties toward the earlier (lower-risk) point.
- High uses the interior sampled point at or above medium nearest the volatility
  midpoint between Maximum Sharpe and the endpoint. If no eligible interior point
  exists, reuse medium. Flag any overlap, including when medium is nearest the
  target; never manufacture an allocation.
- Profile names describe relative in-sample risk within this universe; low risk
  does not mean a safe investment.
- Sharpe is computed downstream with `src.portfolio.calculate_sharpe_ratio`; the
  baseline JSON schema stays unchanged.
- Concentration is flagged at weights above 40%, or fewer than three holdings
  above 1%. These are display diagnostics, not optimizer constraints.
- No financial assumptions change and the frozen dataset need not be regenerated.
- Fractional allocation and the grounded LLM workflow are complete. The API service is next.

### First milestone delivered

Phase 0 required work and Phase 1 are implemented. Optional asset metadata and
per-asset volatility exports remain deferred. `python -m agent` produces
profiles offline, including provenance and disclaimers. Tests cover normal and
sparse frontiers, endpoint concentration, ordering, ticker mapping, invalid
inputs and copy isolation. The full real-dataset baseline payload was compared
before and after GMV reuse and matched exactly. This checkpoint preceded the
Phase 2 allocation work documented below.

### Commit checkpoints and next sequence

The completed implementation was pushed as four focused commits:

| Commit | Change |
| --- | --- |
| `7e206f3` | Record incremental roadmap and selection decisions |
| `ca76f57` | Add offline baseline and optimizer contract tests |
| `877f4fb` | Reuse GMV without changing the serialized baseline results |
| `d60012b` | Add deterministic profiles, offline CLI, tests and documentation |

Continue on `feature/ai-agentic-features`. The following incremental steps are
now complete, each with relevant tests and updated documentation:

- [x] Define and implement fractional allocation inputs and the structured result.
- [x] Add display rounding and reconciliation while preserving full-precision targets.
- [x] Add read-only profile/analysis tools and deterministic explanation templates.
- [x] Add output guardrails and a fixed evaluation set.
- [x] Integrate the selected model and orchestration with tested offline fallback.

Phase 2 exit criteria met: no network dependency, unchanged target weights and
portfolio metrics, reconciled displayed amounts, retained provenance, and all
repository checks passing. Phase 3 model integration is also complete below.
Phase 2 requires no LLM or framework dependency.

### Phase 2 implementation checkpoint

`agent.allocation.allocate_amount(amount, currency, payload, profile_name)` now
returns precise monetary targets and the unchanged validated profile. USD/INR
are labels only. Inputs require whole cents/paise, positivity, and a maximum of
`999999999999.99`; this is an application input limit, not a portfolio constraint.
Money serializes as decimal strings. Numerical weight-sum drift is reported
explicitly; target weights are not normalized. Tests cover validation, exact
products, metadata preservation, stale metrics, isolation, and offline operation.
Display reconciliation and CLI allocation arguments are also implemented.
`python -m agent --amount 10000 --currency USD --profile medium` produces precise
targets plus the display view; the no-argument profile command is unchanged.
Display-only proportions remove numerical drift, then largest-remainder rounding
assigns cents/paise with alphabetical ties. Every adjustment is recorded.
`--hide-dust` groups holdings below 0.5% into a reported total without losing them.
Money remains decimal strings, including two-decimal display amounts. Tests cover
one-cent and large amounts, rounding ties, negative numerical dust, hidden totals,
all profiles, and CLI errors before analysis. All 106 tests and repository checks
pass; no baseline source, assumptions, dataset, or dependencies were changed.

### Phase 3 implementation decisions

Provider choice updated at the user's request: Groq's OpenAI-compatible Responses
API, `https://api.groq.com/openai/v1`, model `openai/gpt-oss-20b`, credential
`GROQ_API_KEY`. The earlier OpenAI-hosted model recommendation is superseded.

The first Phase 3 step added isolated read-only analysis tools, process-local
analysis caching, and deterministic explanations. Subsequent steps added reference
validation, the Groq client, and a LangGraph workflow. Model explanations
select and organize approved fact IDs; Python renders their associated
figures and sentences. This intentionally constrains wording so model-generated
numbers, tickers, predictions, or changed weights cannot enter final text.

The second checkpoint added model reference plans and exact rendered output checks,
including mandatory provenance and warnings. Three fixed synthetic payloads
exercise ordinary comparisons, concentration, and unsupported forecasts in CI.
All 137 tests passed at that checkpoint, before live integration.

The third checkpoint added the Groq Responses adapter, optional pinned
dependencies, prompt, and secret-free environment template. Tool retrieval and
strict JSON plans use separate bounded requests. Live checks passed with
`openai/gpt-oss-20b`, including `get_frontier_point(2)`.

The final Phase 3 checkpoint connects LangGraph, the adapter and CLI, retaining
the same-node sequential path without LangGraph. Tests block network access and
cover missing optional dependencies, provider failures, invalid plans, preserved
profiles, allocation provenance, unsupported questions, and explicit CLI opt-in.
Baseline source, financial assumptions, frozen data, and baseline dependencies
are unchanged. README and AGENTS now document the implemented interfaces.
The next milestone is Phase 4: the API service; no API or GUI is implemented yet.
