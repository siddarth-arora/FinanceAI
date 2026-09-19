# FinanceAI — Implementation Status and Roadmap

> Scope: a deterministic Markowitz MPT baseline, extended by an agentic layer that
> reads the efficient frontier and presents three risk-profiled allocations, plus a
> GUI. This document follows the working contract in `AGENTS.md`; every planned item
> below is placed so that it does **not** break that contract.

---

## 1. Guiding rules (from `AGENTS.md`)

These constraints shape every design decision in Section 4.

| Rule | Consequence for the roadmap |
| --- | --- |
| Baseline stays deterministic; no LLM computes returns, covariance, metrics or weights | The agent only **selects** from `efficient_frontier` and **explains**; it never produces numbers |
| Dependency direction is one-way | `agent/`, `api/`, `gui/` may import `src.pipeline`; nothing in `src/` imports them |
| Consume results only via `run_mpt_analysis().to_dict()` | No parsing of CLI output or `efficient_frontier.png` |
| Yahoo Finance only in `src/download_data.py` | Any live-price feature (e.g. share counts) lives in a separate service, never in `src/` |
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
└── tests/
    └── test_portfolio.py      # 9 unit tests for portfolio math
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
| Unit tests | 🟡 Partial | 9 tests, `portfolio.py` only (all passing) |
| Agentic layer | ❌ Not started | — |
| Risk profiles (low / medium / high) | ❌ Not started | — |
| Amount-based allocation | ❌ Not started | — |
| API service | ❌ Not started | — |
| GUI | ❌ Not started | — |

### 2.3 Observations from reviewing the code

These are gaps worth fixing before building on top of the baseline.

1. **Frontier points have no Sharpe ratio.** `efficient_frontier` entries contain return, variance, volatility and weights, but not Sharpe. The agent needs it for comparisons. Either compute it downstream with `src.portfolio.calculate_sharpe_ratio`, or add it to `EfficientFrontierPoint` / `to_dict()` (a contract change: update tests and README together).
2. **The top of the frontier is a single stock.** The last frontier point targets `max(μ)`, which with long-only constraints is 100% in the highest-return asset (a synthetic run confirmed this). A "high risk" option taken naively from the end of the frontier would be an undiversified one-stock portfolio. See Section 4.2.
3. **Max Sharpe is not guaranteed to be one of the 50 frontier points.** It lies on the frontier curve but usually between sampled points. Selection logic must decide whether to use `portfolios.maximum_sharpe` directly or the nearest frontier point.
4. **Test coverage is limited to `portfolio.py`.** No tests exist for `returns.py`, `optimizer.py`, `pipeline.py`, `data_loader.py` or the JSON contract.
5. **GMV is computed twice** per run (once in `run_mpt_analysis`, once inside `generate_efficient_frontier`). Harmless, but relevant once an API calls the pipeline often; caching the result is the simpler fix.
6. **`metadata.dataset_path` is an absolute local path.** Fine for CLI, but an API should not expose server filesystem paths to clients.
7. **No stock-level descriptive data** (names, sectors, per-asset volatility, price history) is in the payload, yet the GUI needs it to "show the user all the stocks". Per-asset volatility is derivable from the diagonal of `annual_covariance`; names and sectors need a static metadata file.

---

## 3. Target architecture

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
agent/  (new, separate package)
  ├── profiles.py      deterministic low / medium / high selection
  ├── allocation.py    amount → per-stock allocation, re-validated
  ├── tools.py         read-only tools exposed to the LLM
  ├── explainer.py     LLM: explanation + formatting only
  └── guardrails.py    output checks (numbers match payload, disclaimer present)
      │
      ▼
api/  (FastAPI)  ──►  gui/  (web front end)
```

Key principle: **numbers flow down from `src/` and are never regenerated above it.** The LLM receives already-computed profiles and writes prose around them.

---

## 4. What must be implemented

### Phase 0 — Harden the baseline (inside `src/` and `tests/`)

- [ ] `tests/test_returns.py` — simple-return formula, annualization, covariance symmetry, NaN rejection.
- [ ] `tests/test_optimizer.py` — on a synthetic 3–5 asset case: GMV weights valid; Max Sharpe ≥ Sharpe of every frontier point (within tolerance); frontier returns increasing and volatilities non-decreasing; infeasible target raises.
- [ ] `tests/test_pipeline.py` — `to_dict()` is JSON-serializable, has the documented keys, weights keyed by ticker and summing to 1. Use a small temporary Parquet fixture so tests do not depend on the git-ignored real dataset.
- [ ] `tests/test_data_loader.py` — missing file, wrong ticker order, duplicate dates, non-positive prices.
- [ ] Decide on Sharpe in frontier points (Observation 1). If added: update `optimizer.py`, `pipeline.py`, README schema, tests.
- [ ] Optional: expose per-asset annual volatility in `to_dict()` (derived from Σ diagonal, no new assumption).
- [ ] Optional: `data/reference/tickers.json` with company name and sector for display (static, no network).

**Exit criteria:** all `AGENTS.md` required checks pass, including the JSON contract check.

### Phase 1 — Deterministic risk profiles (`agent/profiles.py`)

This is plain Python with no LLM. It turns the frontier into three options.

| Profile | Proposed selection rule | Rationale |
| --- | --- | --- |
| Low risk / low return | `efficient_frontier[0]` | Lowest-volatility sampled efficient portfolio |
| Medium risk / medium return | Sampled frontier point nearest Maximum Sharpe volatility | Respects the frontier-only consumer contract; an approximation to Maximum Sharpe |
| High risk / high return | Frontier point at a fixed volatility position above Max Sharpe (e.g. the point closest to the midpoint between Max Sharpe volatility and the last frontier point's volatility) | Higher return while avoiding the single-stock endpoint |

Tasks:

- [ ] Implement `select_profiles(payload) -> dict[str, ProfileResult]` using only values already in the payload.
- [ ] Recompute Sharpe for any frontier point with `src.portfolio.calculate_sharpe_ratio` (rule 6).
- [ ] Add a diversification check: flag profiles where any weight exceeds a threshold (e.g. 40%) or fewer than N assets exceed a tiny weight (e.g. 1%). Flag only; do not alter weights.
- [ ] **[needs approval]** If a hard concentration cap (e.g. max 30% per stock) is wanted, it must be a new constraint in `src/optimizer.py` with config, tests and README updates, because it changes the frontier itself. It must not be applied by editing weights in the agent.
- [ ] Record the selection rule and its parameters in the profile output so the GUI can show *why* each option was chosen.
- [ ] `tests/agent/test_profiles.py` — ordering guarantee: low volatility ≤ medium ≤ high, and low return ≤ medium ≤ high.

### Phase 2 — Amount-based allocation (`agent/allocation.py`)

- [ ] Input: investment amount + currency + chosen profile.
- [ ] **Fractional mode (default):** `amount × weight` per ticker. Weights unchanged, so no re-validation issue.
- [ ] **Whole-share mode (optional):** needs a current price per ticker. Current prices must come from a separate market-data service (e.g. `services/market_data/`), never from `src/`. Rounding down to whole shares leaves residual cash and changes effective weights, so:
  - [ ] recompute effective weights over the invested portion,
  - [ ] re-evaluate return / volatility / Sharpe with `src.portfolio`,
  - [ ] show both the target and the effective portfolio, plus leftover cash.
- [ ] Hide dust weights (< 0.5% or similar) in display only; keep full precision in calculations.
- [ ] Unit tests for rounding, leftover cash, and that displayed allocations sum to the amount.

### Phase 3 — Agentic layer (`agent/`)

The LLM's allowed role per `AGENTS.md`: explain trade-offs, filter already-computed candidates, format results.

- [ ] **Framework:** LangGraph (consistent with the tooling already in use elsewhere), with a small graph:
  1. `load_analysis` — call `run_mpt_analysis().to_dict()` (cached).
  2. `build_profiles` — Phase 1 function.
  3. `allocate` — Phase 2 function, if the user gave an amount.
  4. `explain` — LLM writes a comparison of the three options from the structured data.
  5. `guardrail` — verify the output, else retry or fall back to a template.
- [ ] **Read-only tools** for follow-up questions: `get_profile(name)`, `get_asset_stats(ticker)`, `get_frontier_point(index)`, `get_metadata()`. No tool accepts or returns new weights.
- [ ] **Prompt design:** system prompt states that all numbers are historical in-sample estimates, forbids inventing or adjusting figures, and requires the dataset window and risk-free rate in the answer.
- [ ] **Guardrails (`guardrails.py`):**
  - [ ] every percentage or amount in the LLM text matches a payload value (after rounding),
  - [ ] no tickers outside `metadata.assets`,
  - [ ] no forecast language ("will return", "guaranteed"),
  - [ ] disclaimer and provenance present.
- [ ] **Deterministic fallback:** a templated explanation used when the LLM is unavailable or fails guardrails, so the product still works offline.
- [ ] Separate `agent/requirements.txt` so LLM dependencies never enter the baseline environment.
- [ ] Evaluation set: a handful of fixed payloads + expected checks, run in CI.

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
├── agent/                         # NEW — may import src.pipeline, never the reverse
│   ├── __init__.py
│   ├── requirements.txt
│   ├── profiles.py
│   ├── allocation.py
│   ├── tools.py
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
    ├── test_returns.py            # NEW
    ├── test_optimizer.py          # NEW
    ├── test_pipeline.py           # NEW
    ├── test_data_loader.py        # NEW
    ├── agent/
    │   ├── test_profiles.py
    │   ├── test_allocation.py
    │   └── test_guardrails.py
    └── api/
        └── test_endpoints.py
```

---

## 6. Open decisions

| Decision | Options | Affects |
| --- | --- | --- |
| High-risk selection rule | Volatility midpoint above Max Sharpe · fixed frontier index · target-return percentile | Phase 1 |
| Concentration limits | Flag only · new optimizer constraint **[needs approval]** | Phases 0–1 |
| Sharpe in frontier contract | Compute downstream · add to `to_dict()` | Phases 0–1 |
| Allocation mode | Fractional only · whole shares with live prices | Phase 2 |
| LLM provider / framework | LangGraph + hosted model · local model | Phase 3 |
| GUI stack | React · Streamlit | Phase 5 |
| Ticker universe | Current 10 US stocks · larger / NSE universe (requires re-download) | All |

---

## 7. Definition of done (per change)

From `AGENTS.md`, run from the repository root:

```bash
python -m pytest -v
python -m src.main
python -c "import json; from src.pipeline import run_mpt_analysis; json.dumps(run_mpt_analysis().to_dict()); print('JSON contract valid')"
git diff --check
git status
```

Additionally for agent / API / GUI changes:

- profile ordering tests pass,
- guardrail tests pass,
- every user-facing output shows the dataset window, risk-free rate, constraints and the statement that figures are historical estimates rather than forecasts or personalized investment advice.

## 8. Incremental implementation decisions

The first milestone covers Phase 0 hardening and Phase 1 deterministic profiles.
Work stays on `feature/ai-agentic-features`, with separate commits for roadmap
decisions, baseline tests, duplicate-GMV cleanup, and profile selection.

- All profile weights come unchanged from `efficient_frontier`, as required by
  `AGENTS.md`. Medium uses the nearest sampled point by volatility to Maximum
  Sharpe, breaking ties toward the earlier (lower-risk) point.
- High uses the interior sampled point at or above medium nearest the volatility
  midpoint between Maximum Sharpe and the endpoint. If no higher interior point
  exists, reuse medium and flag the overlap; never manufacture an allocation.
- Profile names describe relative in-sample risk within this universe; low risk
  does not mean a safe investment.
- Sharpe is computed downstream with `src.portfolio.calculate_sharpe_ratio`; the
  baseline JSON schema stays unchanged.
- Concentration is flagged at weights above 40%, or fewer than three holdings
  above 1%. These are display diagnostics, not optimizer constraints.
- No financial assumptions change and the frozen dataset need not be regenerated.
- Amount allocation and the LLM workflow follow as subsequent milestones.
