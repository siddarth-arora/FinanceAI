# FinanceAI — Full-Stack Dashboard Implementation Plan

> **Status:** M0–M13 implemented (backend, analytics, dashboard, tests). M14
> (dataset bootstrap) and M15 (Render/Vercel deployment) are next. See
> [Implementation status](#implementation-status).
> **Covers:** ROADMAP Phase 4 (FastAPI service) and Phase 5 (GUI), built as a
> FastAPI backend plus a Next.js frontend that reproduces the reference
> dashboard screenshot with real data.
> **Deployment:** the backend runs on Render (Docker) and downloads the frozen
> dataset at startup from a checksum-pinned release artifact. The frontend is
> on Vercel only. See §7.
> **Binding contract:** [AGENTS.md](AGENTS.md). Where this plan and AGENTS.md
> disagree, AGENTS.md wins. Read [README.md](README.md),
> [ROADMAP.md](ROADMAP.md) and [context.md](context.md) first.

---

## Implementation status

| Milestone | State | Evidence |
| --- | --- | --- |
| M0–M5 backend | Done | `api/`, `analytics/`, `data/reference/tickers.json`; 108 new Python tests (312 total) pass |
| M6–M12 dashboard | Done | `gui/`, all seven pages against the real API; `tsc`, ESLint, 17 Vitest tests, production build pass |
| M13 polish | Done | Responsive/mobile check, Playwright smoke (6 tests incl. mobile), CI jobs `api` and `gui`, docs |
| M14 dataset bootstrap | Not started | `api/bootstrap.py`, release asset, `dataset` CI job |
| M15 deployment | Not started | Dockerfile, `render.yaml`, Neon, Vercel project |

Deliberate deviations from the text below, made during implementation:

- **Schemas** live in four modules (`api/schemas/{common,profiles,requests,responses}.py`) instead of one per area.
- **No shadcn/ui, react-hook-form or zod.** A small set of accessible primitives
  (`gui/components/ui/primitives.tsx`) and native inputs covered every need with
  fewer dependencies.
- **Next.js 16 with Cache Components** (the scaffold default): client views are
  wrapped in `<Suspense>`, and dialogs close in effect cleanup because hidden
  routes stay mounted.
- **Charts follow the dataviz method**: categorical colors are validated slots,
  fixed per sector group, with "Other" in neutral gray. The donut is by group only;
  per-stock detail is in tables. Price history uses small multiples on a shared
  log scale instead of ten coloured lines.
- **Amount inputs** always hold the raw decimal and show a grouped preview below.
  Reformatting while typing broke select-and-replace.
- **Local development** used Python 3.11 with numpy 2.4.6 / scipy 1.17.1 (the
  pinned versions need 3.12+). `requirements.txt` is unchanged, and the baseline
  values match the README exactly.

## Table of contents

1. [Product summary](#1-product-summary)
2. [Non-negotiable rules for the API and UI](#2-non-negotiable-rules-for-the-api-and-ui)
3. [Screenshot → feature mapping](#3-screenshot--feature-mapping)
4. [Target architecture](#4-target-architecture)
5. [Backend (FastAPI)](#5-backend-fastapi)
6. [Frontend (Next.js)](#6-frontend-nextjs)
7. [Integration, tooling and deployment](#7-integration-tooling-and-deployment)
8. [Milestones and acceptance criteria](#8-milestones-and-acceptance-criteria)
9. [Documentation updates](#9-documentation-updates)
10. [Out of scope](#10-out-of-scope)
11. [Open decisions](#11-open-decisions)

---

## 1. Product summary

FinanceAI turns a **frozen 2021–2025 history of 10 US large-cap stocks**
(AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, JNJ, XOM, PG) into auditable
portfolio choices using Markowitz Modern Portfolio Theory. It **recommends and
explains portfolio mixes**; it does **not** hold the user's money, track real
holdings, place trades or forecast returns.

What already works (Python, offline, tested):

| Capability | Entry point |
| --- | --- |
| MPT analysis: returns, covariance, GMV, Max Sharpe, 50-point efficient frontier | `src.pipeline.run_mpt_analysis().to_dict()` |
| Cached analysis snapshot | `agent.cache.load_analysis()` |
| Low / medium / high risk profiles | `agent.profiles.select_profiles(payload)` |
| Needs-driven conservative / balanced / growth buckets | `agent.preferences.select_buckets(AnalysisTools(payload), Preferences(...))` |
| Amount → per-stock target amounts (exact + reconciled display) | `agent.allocation.allocate_amount(amount, currency, payload, profile)` |
| Grounded explanations, optional Groq LLM, offline fallback | `agent.graph.run_agent(payload, ...)` |
| Read-only per-asset and per-frontier-point stats | `agent.tools.AnalysisTools(payload)` |

What end users will get from the dashboard:

1. **Overview**: headline KPIs, the in-sample growth of their planned
   investment, and the asset-allocation donut.
2. **Portfolio**: holdings of the chosen profile, a side-by-side comparison of
   the profiles, and the stock universe with statistics, correlations and price
   history.
3. **Optimizer**: an interactive efficient frontier and a "tell us your limits"
   form that returns three needs-driven buckets.
4. **Allocation**: "I want to invest ₹X": per-stock amounts that reconcile
   exactly, a rounding audit, CSV export and save.
5. **Saved plans** (the screenshot's *Transactions* slot): a history of saved
   allocation plans, with view, compare, delete and export.
6. **AI Explainer** (the screenshot's *AI Advisor* slot): ask questions and get
   fact-grounded answers with mode/status badges.
7. **Settings**: currency, default amount and profile, AI toggle, read-only
   assumptions and API status.

---

## 2. Non-negotiable rules for the API and UI

These come from AGENTS.md, ROADMAP §1 and CLAUDE.md. Every reviewer should check them.

1. **Dependency direction is one-way.** `api/` and `analytics/` may import
   `src` and `agent`. `src/` and `agent/` must never import `api/`,
   `analytics/` or `gui/`.
2. **Do not duplicate the MPT workflow.** The API calls existing functions; it
   never re-implements optimization, profile selection or allocation.
3. **No financial maths in the frontend.** The UI formats numbers; it never
   sums money, recomputes weights, returns or Sharpe. Any new number is
   computed by a tested Python function and returned by the API.
4. **The API never accepts client-supplied weights.** Portfolios are always
   referenced by name (`low|medium|high`) or by frontier index, and resolved
   server-side from the cached payload.
5. **Rates and weights stay decimals end-to-end** (`0.25` = 25%). Percentage
   formatting happens only in `gui/lib/format.ts`.
6. **Money travels as decimal strings** (`"1000000.00"`) in both requests and
   responses. The frontend may convert to `Number` only to *display* or *plot*,
   never for arithmetic.
7. **Never expose `dataset_path`**, credentials, provider error bodies or stack
   traces. Strip `dataset_path` recursively from every response (see §5.5).
8. **Provenance on every screen.** Every page shows the dataset window, the 252
   trading-day annualization, the 4% risk-free rate, the long-only / fully
   invested / no-leverage constraints, and the disclaimer: *"Figures are
   historical in-sample estimates, not forecasts or personalized investment
   advice."*
9. **Currency is a label only.** USD/INR performs no FX conversion. Any `₹`
   figure on US stocks must carry the footnote *"Currency is a display label; no
   FX conversion is applied."*
10. **The LLM is opt-in** (`use_llm=true`). It selects fact IDs and, for a
    question only, may write `explanation.answer`, which is published only when
    every figure in it appears in the facts it cites (see AGENTS.md). No other
    route returns model-written prose.
11. **Never call `python -m src.download_data`** from the API, deployment
    scripts or CI. The dataset is frozen. Servers get it from the versioned
    release artifact, verified by SHA-256, through `api/bootstrap.py` (§7.2).
    That is a file download, not a Yahoo call.
12. **Additive changes only.** Existing CLI output and tests must stay
    unchanged. All existing checks in AGENTS.md must still pass.

---

## 3. Screenshot → feature mapping

The screenshot is a portfolio-*tracker* design. The table below shows how each
element is backed by **real data** so we keep the full visual design without
inventing numbers.

### 3.1 Shell

| Screenshot element | Implementation |
| --- | --- |
| Logo tile "F", "FinanceAI", "Portfolio intelligence" | Static brand block in the sidebar |
| Nav: Overview, Portfolio, Optimizer, Allocation, Transactions, AI Advisor, Settings | Same order and icons. *Transactions* → **Saved plans** and *AI Advisor* → **AI Explainer** (see §11 for why; labels are one constant to change) |
| Active nav item (dark navy pill) | `usePathname()` match, `bg-[--nav-active]` |
| Eyebrow "PORTFOLIO INTELLIGENCE", page title, subtitle | `<PageHeader eyebrow title subtitle />` per page |
| Date pill "🕒 Jan 2021 – Dec 2025" | `GET /api/metadata` → `start_date`/`end_date`, formatted `MMM yyyy`. **Read-only** (the dataset is frozen); the tooltip explains this |
| Avatar "FA" | Static initials (there is no auth in v1). It opens a menu with Settings and About |

### 3.2 Overview KPI cards

| Card | Screenshot value | Real source | Label / footnote |
| --- | --- | --- | --- |
| **Portfolio value** | ₹60,00,900 ↑ ₹1,85,000 all-time gain* | `GET /api/performance` → `summary.end_value` and `summary.gain`, for the user's *planned amount* in the selected profile over the selected range | Subtitle: "↑ ₹X gain over 2021–2025*". Footnote: "*In-sample simulation of your planned amount, buy-and-hold, using the same data the weights were chosen from (hindsight). Not a forecast." |
| **Historical return** | +18.4% | `GET /api/profiles` → `profiles[profile].expected_return` (annualized arithmetic mean) | "Historical annual estimate*". Secondary line: "CAGR x.x% (simulation)" from `summary.cagr` |
| **Assets in universe** | 10 | `metadata.asset_count` | "Predefined US equities" |
| **Risk profile** | Balanced | Settings store `profile` → label map `low→Conservative`, `medium→Balanced`, `high→Growth`, plus the Sharpe value | "Change in Optimizer" link |

### 3.3 Overview charts

| Screenshot element | Real source |
| --- | --- |
| "Portfolio performance": area/line chart 2021–2025, ₹ y-axis in lakhs, point markers | `GET /api/performance?profile&amount&currency&range&freq` → `series[]`. Optional dashed **equal-weight benchmark** line (toggle) |
| Range dropdown "5Y" | `range ∈ {1Y, 3Y, 5Y}` (5Y = full window). A sub-range restarts the simulation with the planned amount at the first trading day of that range |
| "Asset allocation": donut + legend (Technology 40%, Healthcare 27%, Consumer 19%, Other 14%) | `GET /api/profiles` → `breakdowns[profile].by_group`: the **top 3 groups + "Other"**, built from the new static `data/reference/tickers.json`. A toggle switches to "By stock" |
| Donut centre "Assets 100%" | Static "Allocated / 100%" (the portfolio is fully invested, a contract guarantee) |
| "View details" button | Navigates to `/portfolio?tab=holdings` |
| Blue info note under the donut | Replace the generic text with the profile's `warnings` (concentration / few holdings / overlap) rendered as human text, or "No concentration flags" |
| Footnote "*Illustrative UI values…" | Replaced by real footnotes (§3.2). **No illustrative values remain anywhere** |

---

## 4. Target architecture

```text
                       ┌───────────────────────────────────────┐
 Browser ──HTTP──►     │ gui/  Next.js (App Router, TypeScript) │
                       │  /api/* rewrite (dev) ─────────┐      │
                       └────────────────────────────────┼──────┘
                                                        ▼
                       ┌───────────────────────────────────────┐
                       │ api/  FastAPI  (uvicorn :8000)         │
                       │  lifespan: payload + prices cached     │
                       │  routers → agent/ & analytics/         │
                       │  SQLite (saved plans)                  │
                       └──────┬───────────────┬────────────────┘
                              │               │
              ┌───────────────▼──┐   ┌────────▼──────────────────┐
              │ agent/ (exists)  │   │ analytics/ (NEW, pure)    │
              │ profiles, buckets│   │ universe stats, sectors,  │
              │ allocation, LLM  │   │ in-sample performance     │
              └───────┬──────────┘   └────────┬──────────────────┘
                      │                       │
                      ▼                       ▼
              ┌──────────────────────────────────────────┐
              │ src/ (exists, unchanged) MPT baseline     │
              │ reads data/raw/stock_prices.parquet       │
              └──────────────────────────────────────────┘
```

New top-level folders inside `FinanceAI/`:

| Folder | Purpose | May import |
| --- | --- | --- |
| `analytics/` | New deterministic, pure computations that the dashboard needs but the payload lacks (correlation, sector weights, buy-and-hold simulation, normalized price history) | `src` (and `agent.profiles` constants only) |
| `api/` | HTTP layer: validation, serialization, sanitizing, persistence | `src`, `agent`, `analytics` |
| `gui/` | Next.js app | the HTTP API only |
| `data/reference/` | `tickers.json` (static names and sectors) | — |

`analytics/` is separate from `agent/` because it adds **new calculations**
(simulation, correlation) that are not explanation logic, and separate from
`research/` (ROADMAP Phase 6, out-of-sample backtests) because it is
in-sample, presentation-oriented and served live.

---

## 5. Backend (FastAPI)

### 5.1 Folder layout

```text
FinanceAI/
├── analytics/
│   ├── __init__.py
│   ├── reference.py        # load + validate data/reference/tickers.json
│   ├── universe.py         # per-asset stats, correlation matrix, normalized prices
│   ├── breakdown.py        # weights → sorted by-ticker / by-group breakdown
│   └── performance.py      # in-sample buy-and-hold simulation + summary metrics
├── api/
│   ├── __init__.py
│   ├── main.py             # FastAPI app, lifespan, middleware, router registration
│   ├── config.py           # pydantic-settings: CORS origins, DB url, rate limits
│   ├── deps.py             # get_payload(), get_prices(), get_reference(), get_session()
│   ├── sanitize.py         # public(): recursive dataset_path removal
│   ├── errors.py           # error envelope + exception handlers
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── common.py       # Money, Currency, ProfileName, Range, ErrorResponse
│   │   ├── meta.py
│   │   ├── universe.py
│   │   ├── frontier.py
│   │   ├── profiles.py
│   │   ├── performance.py
│   │   ├── allocation.py
│   │   ├── buckets.py
│   │   ├── explain.py
│   │   └── plans.py
│   ├── routers/
│   │   ├── meta.py         # /health, /metadata
│   │   ├── universe.py     # /universe, /universe/correlation, /universe/prices
│   │   ├── frontier.py     # /frontier
│   │   ├── profiles.py     # /profiles, /profiles/{name}
│   │   ├── performance.py  # /performance
│   │   ├── allocation.py   # /allocate, /allocate/export.csv
│   │   ├── buckets.py      # /buckets
│   │   ├── explain.py      # /explain
│   │   └── plans.py        # /plans CRUD
│   ├── db.py               # SQLModel engine, init_db(), Plan table
│   ├── ratelimit.py        # llm_limiter.check(request) for /explain with use_llm
│   ├── bootstrap.py        # startup: download + SHA-256 verify + validate the frozen Parquet (§7.2)
│   ├── requirements.txt    # API-only dependencies
│   └── Dockerfile          # Render image; build context = repo root (§7.3)
├── data/reference/tickers.json
├── render.yaml             # Render Blueprint (§7.3)
├── .dockerignore           # excludes data/raw/*.parquet, .env, gui/, tests/
└── tests/
    ├── analytics/
    │   ├── test_reference.py
    │   ├── test_universe.py
    │   ├── test_breakdown.py
    │   └── test_performance.py
    └── api/
        ├── conftest.py     # TestClient, synthetic payload/prices, socket block, temp DB
        ├── test_meta.py
        ├── test_universe.py
        ├── test_frontier.py
        ├── test_profiles.py
        ├── test_performance.py
        ├── test_allocation.py
        ├── test_buckets.py
        ├── test_explain.py
        ├── test_plans.py
        ├── test_bootstrap.py
        └── test_sanitize.py
```

### 5.2 Dependencies and configuration

`api/requirements.txt` (pin exact versions after installing, as the baseline does):

```text
fastapi
uvicorn[standard]
pydantic-settings
sqlmodel
psycopg[binary]  # Postgres driver for Neon in production (SQLite stays the local default)
limits           # in-memory rate limiting for /explain with use_llm
httpx            # TestClient
```

- Keep `requirements.txt` (baseline) unchanged. Install order:
  `pip install -r requirements.txt -r agent/requirements.txt -r api/requirements.txt`.
- The API works **without** `agent/requirements.txt`. `/explain` with
  `use_llm=true` then returns `mode: "fallback"`, as the agent already handles.

`api/config.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINANCEAI_", env_file=".env", extra="ignore")

    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cors_origin_regex: str | None = None             # Vercel preview deployments (§7.3)
    database_url: str = "sqlite:///data/app/financeai.db"   # Neon Postgres URL on Render
    explain_llm_rate_limit: str = "10/minute"
    enable_plans: bool = True
    # FINANCEAI_DATASET_URL / _SHA256 / _TOKEN are read by api/bootstrap.py, not by the app.


settings = ApiSettings()
```

- `.env` lives at **`FinanceAI/.env`** (that is where `agent/provider.py`
  reads `GROQ_API_KEY`). Add the `FINANCEAI_*` keys to `.env.example` without
  values.
- Add `data/app/` to `.gitignore` (SQLite file).

### 5.3 Application startup (`api/main.py`)

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.cache import load_analysis
from agent.profiles import select_profiles
from analytics.reference import load_reference
from api.config import settings
from api.db import init_db
from api.errors import register_error_handlers
from api.routers import allocation, buckets, explain, frontier, meta, performance, plans, profiles, universe
from src.data_loader import load_stock_prices


@asynccontextmanager
async def lifespan(app: FastAPI):
    payload = load_analysis()                 # frozen Parquet → MPT pipeline, once
    select_profiles(payload)                  # fail fast if the snapshot is invalid
    app.state.payload = payload
    app.state.prices = load_stock_prices()    # for analytics/performance & price history
    app.state.reference = load_reference(payload["metadata"]["assets"])  # must cover every ticker
    if settings.enable_plans:
        init_db()
    yield


app = FastAPI(title="FinanceAI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
)
register_error_handlers(app)
for module in (meta, universe, frontier, profiles, performance, allocation, buckets, explain, plans):
    app.include_router(module.router, prefix="/api")
```

`api/deps.py` gives each request an isolated copy, so no handler can mutate shared state:

```python
from copy import deepcopy
from fastapi import Request


def get_payload(request: Request) -> dict:
    return deepcopy(request.app.state.payload)


def get_prices(request: Request):
    return request.app.state.prices          # treat as read-only; analytics never mutates


def get_reference(request: Request) -> dict:
    return request.app.state.reference
```

### 5.4 Error handling (`api/errors.py`)

One envelope for every error:

```json
{ "error": { "code": "invalid_input", "message": "Amount must have at most two decimal places." } }
```

| Situation | HTTP | `code` |
| --- | --- | --- |
| Pydantic request validation | 422 | `validation_error` (include field locations, not internals) |
| `ValueError` raised by `agent.*` / `analytics.*` **inside a router `try`** | 422 | `invalid_input` (these messages are fixed strings and safe to show) |
| Unknown plan id | 404 | `not_found` |
| LLM rate limit hit | 429 | `rate_limited` |
| Anything else | 500 | `internal_error` with a generic message; log the traceback server-side only |

Do **not** register a global `ValueError` handler. Library `ValueError`s
(pandas, numpy) could leak internals. Catch `ValueError` explicitly around
agent and analytics calls instead.

LLM provider failures are **not** errors. `run_agent` returns HTTP 200 with
`explanation.mode = "fallback"` and a `fallback_reason` code.

### 5.5 Response sanitizing (`api/sanitize.py`)

`select_profiles` **requires** `metadata.dataset_path`
([agent/profiles.py:63](agent/profiles.py#L63)), so the payload passed to the
agent must keep it. Strip it from every *response* instead. It appears nested in
profiles, allocations, buckets and explanations.

```python
def public(value):
    """Recursively drop server-only keys before a value leaves the API."""
    if isinstance(value, dict):
        return {key: public(item) for key, item in value.items() if key != "dataset_path"}
    if isinstance(value, list):
        return [public(item) for item in value]
    return value
```

Every router returns `public(...)`. `tests/api/test_sanitize.py` calls
**every** endpoint and asserts that `"dataset_path"` and the absolute data
directory string do not appear anywhere in the response body.

### 5.6 New reference data: `data/reference/tickers.json`

Static and committed, with no network. It is the ROADMAP Phase 0 optional item.

```json
{
  "source": "Static GICS sector classification, manually curated for display only.",
  "assets": {
    "AAPL":  { "name": "Apple Inc.",                  "sector": "Information Technology", "group": "Technology" },
    "MSFT":  { "name": "Microsoft Corporation",       "sector": "Information Technology", "group": "Technology" },
    "NVDA":  { "name": "NVIDIA Corporation",          "sector": "Information Technology", "group": "Technology" },
    "GOOGL": { "name": "Alphabet Inc. (Class A)",     "sector": "Communication Services", "group": "Technology" },
    "META":  { "name": "Meta Platforms, Inc.",        "sector": "Communication Services", "group": "Technology" },
    "AMZN":  { "name": "Amazon.com, Inc.",            "sector": "Consumer Discretionary", "group": "Consumer" },
    "PG":    { "name": "The Procter & Gamble Company","sector": "Consumer Staples",       "group": "Consumer" },
    "JNJ":   { "name": "Johnson & Johnson",           "sector": "Health Care",            "group": "Healthcare" },
    "JPM":   { "name": "JPMorgan Chase & Co.",        "sector": "Financials",             "group": "Financials" },
    "XOM":   { "name": "Exxon Mobil Corporation",     "sector": "Energy",                 "group": "Energy" }
  }
}
```

- `sector` is the formal GICS sector. `group` is the coarser display bucket
  that the screenshot's donut uses (Technology / Healthcare / Consumer / Other).
  Document in the UI tooltip that GOOGL and META are grouped under Technology for display.
- `analytics/reference.py::load_reference(assets)` raises `ValueError` if the
  file's tickers are not **exactly** `metadata.assets`. The API then fails at
  startup rather than showing a partial donut.

### 5.7 New deterministic computations (`analytics/`)

All functions are pure, take explicit inputs, never touch the network or write
files, and have synthetic-data tests in `tests/analytics/`.

#### `analytics/universe.py`

```python
def asset_stats(payload: dict, reference: dict) -> list[dict]:
    """Per ticker: name, sector, group, annual_expected_return, annual_volatility.

    Volatility = sqrt(Σ_ii), i.e. AnalysisTools.get_asset_stats(ticker)['annual_volatility'];
    reuse AnalysisTools rather than re-deriving.
    """

def correlation_matrix(payload: dict) -> dict:
    """ρ_ij = Σ_ij / (σ_i σ_j) from annual_covariance; returns {'assets': [...], 'matrix': [[...]]}.
    Diagonal is exactly 1.0; matrix is symmetric; values clipped to [-1, 1] only for float noise ≤1e-12."""

def normalized_prices(prices: pd.DataFrame, range_key: str, freq: str) -> dict:
    """Each ticker rebased to 100 at the range start; sampled at the last trading day of each month/quarter."""
```

#### `analytics/breakdown.py`

```python
def breakdown(weights: dict[str, float], reference: dict) -> dict:
    """{'by_ticker': [{ticker, name, group, weight}] sorted desc,
        'by_group':  [{group, weight, tickers}] sorted desc}
    Sums of weights only; never renormalizes. Tiny negative optimizer noise (≥ -1e-8) is reported as is."""
```

The Overview donut uses the top 3 `by_group` entries plus an "Other" that the
**backend** computes (`top_groups(by_group, n=3)`), so the frontend does no
arithmetic.

#### `analytics/performance.py`: in-sample buy-and-hold simulation

This is the only genuinely new financial computation. It must be documented as
**hindsight**: the weights were optimized on the same 2021–2025 data.

Method:

- **Buy-and-hold, no rebalancing, no costs, no taxes, no FX.** The amount is
  split by the target weights at the first trading day of the range and each
  position then follows that stock's adjusted close.
- `value_t = amount × Σ_i w_i × P_i,t / P_i,t0`
- Range start: `5Y` = first observation. `3Y`/`1Y` = first trading day
  strictly after `last_date − N years`.
- Series sampling: the last trading day in each month (`freq="M"`, default) or
  quarter (`"Q"`), always including the first and last observations.

```python
RANGE_YEARS = {"1Y": 1, "3Y": 3, "5Y": None}


def range_start(index: pd.DatetimeIndex, range_key: str) -> pd.Timestamp:
    years = RANGE_YEARS[range_key]
    if years is None:
        return index[0]
    cutoff = index[-1] - pd.DateOffset(years=years)
    return index[index > cutoff][0]


def simulate_buy_and_hold(prices: pd.DataFrame, weights: dict[str, float],
                          amount: Decimal, range_key: str) -> pd.Series:
    window = prices.loc[range_start(prices.index, range_key):]
    vector = validate_weights([weights[t] for t in window.columns], len(window.columns))  # src.portfolio
    growth = (window / window.iloc[0]) @ vector
    return growth * float(amount)


def summarize(values: pd.Series, trading_days: int) -> dict:
    first, last = values.iloc[0], values.iloc[-1]
    years = (values.index[-1] - values.index[0]).days / 365.25
    daily = values.pct_change().dropna()
    return {
        "start_value": f"{first:.2f}",
        "end_value": f"{last:.2f}",
        "gain": f"{last - first:.2f}",
        "total_return": last / first - 1,
        "cagr": (last / first) ** (1 / years) - 1,
        "realized_volatility": float(daily.std() * trading_days ** 0.5),
        "max_drawdown": float((values / values.cummax() - 1).min()),
    }
```

- Validate `amount` with `agent.allocation.validate_allocation_input` so the
  rules match allocation exactly.
- `equal_weight` benchmark: the same function with `1/N` weights.
- Tests: on a synthetic 3-asset price frame, check a hand-computed end value,
  that the start value equals the amount, that a flat price gives zero gain, a
  known drawdown, range start selection, sampling includes the endpoints,
  weights are validated (rejects a sum ≠ 1), and no network.
- **Explain in the UI** why the CAGR differs from `expected_return`
  (arithmetic mean of daily returns × 252 vs compounded buy-and-hold).

### 5.8 Shared schema types (`api/schemas/common.py`)

```python
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

Money = Annotated[str, Field(pattern=r"^\d{1,12}(\.\d{1,2})?$", examples=["1000000.00"])]
Currency = Literal["USD", "INR"]
ProfileName = Literal["low", "medium", "high"]
RangeKey = Literal["1Y", "3Y", "5Y"]
Freq = Literal["M", "Q"]
Emphasis = Literal["lower_risk", "balanced", "higher_return"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
```

All request models extend `StrictModel` (unknown fields → 422). Response models
mirror the agent's `to_dict()` shapes field by field, so `/openapi.json` gives
the frontend precise TypeScript types. A contract test asserts that each
response model validates the real agent output for a synthetic payload, so
schema drift fails CI.

### 5.9 Endpoint specification

All routes are under `/api`. Read endpoints are `GET` and cacheable (add
`Cache-Control: public, max-age=3600`; the data is frozen).

#### `GET /api/health`
`{"status": "ok", "dataset_loaded": true, "llm_configured": bool}`.
`llm_configured` = optional deps importable **and** `GROQ_API_KEY` present. Never return the key.

#### `GET /api/metadata`
Provenance for the header pill, footers and Settings.

```json
{
  "assets": ["AAPL", "..."], "asset_count": 10, "observations": 1255,
  "start_date": "2021-01-04", "end_date": "2025-12-31",
  "trading_days": 252, "risk_free_rate": 0.04, "frontier_point_count": 50,
  "constraints": { "fully_invested": true, "long_only": true, "leverage_allowed": false },
  "disclaimer": "Figures are historical in-sample estimates, ...",
  "currency_notice": "Currency is a display label; no FX conversion is applied."
}
```
Source: `public(payload["metadata"])` + `agent.profiles.DISCLAIMER`.

#### `GET /api/universe`
`{"assets": [{ticker, name, sector, group, annual_expected_return, annual_volatility}], "metadata": {...}}`
Source: `analytics.universe.asset_stats`.

#### `GET /api/universe/correlation`
`{"assets": [...], "matrix": [[1.0, 0.62, ...], ...]}`, ordered like `metadata.assets`.

#### `GET /api/universe/prices?range=5Y&freq=M`
`{"range": "5Y", "start_date": ..., "end_date": ..., "series": [{"date": "2021-01-29", "AAPL": 98.7, ...}]}`,
rebased to 100. Feeds the universe price chart.

#### `GET /api/frontier`
Everything the interactive frontier chart needs, in one call.

```json
{
  "points": [ { "index": 0, "expected_return": 0.139, "volatility": 0.125, "sharpe_ratio": 0.796,
                "weights": { "AAPL": 0.05, "...": 0 } } ],
  "gmv":          { "expected_return": ..., "volatility": ..., "sharpe_ratio": ..., "weights": {...} },
  "max_sharpe":   { "...": "exact optimum; usually between sampled points" },
  "profiles":     { "low": 0, "medium": 17, "high": 31 },
  "assets":       [ { "ticker": "AAPL", "expected_return": ..., "volatility": ... } ],
  "risk_free_rate": 0.04
}
```
Source: `AnalysisTools(payload).get_frontier_point(i)` for each `i` (adds
Sharpe through `src.portfolio`), `payload["portfolios"]`, and
`profile["selection"]["frontier_index"]`. Precompute once at startup and cache
it on `app.state`.

#### `GET /api/profiles`
```json
{
  "profiles":   { "low": { ...ProfileResult.to_dict() }, "medium": {...}, "high": {...} },
  "breakdowns": { "low": { "by_ticker": [...], "by_group": [...], "top_groups": [...] }, "...": {} },
  "labels":     { "low": "Conservative", "medium": "Balanced", "high": "Growth" },
  "disclaimer": "..."
}
```
Profile dicts are passed through **unchanged** (minus `dataset_path`). The
breakdown is added alongside, never merged into the profile.

#### `GET /api/profiles/{name}`
A single profile plus its breakdown. `name ∉ {low, medium, high}` → 422.

#### `GET /api/performance`
Query: `profile=medium` **or** `frontier_index=12` (exactly one; both or
neither → 422), `amount` (Money, required), `currency`, `range=5Y`,
`freq=M`, `benchmark=equal_weight|none`.

```json
{
  "portfolio": { "kind": "profile", "profile": "medium", "frontier_index": 17 },
  "currency": "INR", "range": "5Y", "start_date": "2021-01-04", "end_date": "2025-12-31",
  "method": "in_sample_buy_and_hold",
  "series": [ { "date": "2021-01-04", "value": "1000000.00", "benchmark": "1000000.00" } ],
  "summary": { "start_value": "...", "end_value": "...", "gain": "...", "total_return": 0.0,
               "cagr": 0.0, "realized_volatility": 0.0, "max_drawdown": 0.0 },
  "benchmark_summary": { "...": "same shape or null" },
  "notice": "In-sample hindsight simulation: weights were selected using this same data. Buy-and-hold, no rebalancing, costs, taxes or FX. Not a forecast.",
  "currency_notice": "Currency is a display label; no FX conversion is applied."
}
```
Weights are resolved server-side from the profile or frontier point (rule 4).

#### `POST /api/allocate`
```json
{ "amount": "1000000.00", "currency": "INR", "profile": "medium", "hide_dust": false }
```
→ `public(allocate_amount(amount, currency, payload, profile).to_dict(hide_dust=hide_dust))`
plus `"breakdown"` from `analytics.breakdown` (weights from `target_profile`).
The response keeps `target_amounts` (exact) and `display` (reconciled) **both**.
The UI shows `display` and puts `target_amounts` and `rounding_adjustments`
in an "audit" disclosure.

#### `POST /api/allocate/export.csv`
Same body. It returns `text/csv` with
`ticker,name,group,weight,display_amount,exact_target_amount,currency`, plus a
trailer of provenance lines that start with `#`. Generate it server-side so the
browser never formats money for export.

#### `POST /api/buckets`
```json
{ "emphasis": "balanced", "max_volatility": 0.22, "min_expected_return": 0.15 }
```
→ `public(select_buckets(AnalysisTools(payload), Preferences(**body)))`. Pass
through `status`, `reason`, `closest`, `buckets`, `recommended`, `window` and
`frontier_range` unchanged. An infeasible window is **200** with
`status: "infeasible"`, not an error. Add `breakdowns` per bucket.

#### `POST /api/explain`
```json
{
  "question": "Why is the high profile concentrated?",
  "use_llm": true,
  "amount": "1000000.00", "currency": "INR", "profile": "medium", "hide_dust": false,
  "preferences": { "emphasis": "higher_return", "max_volatility": 0.25, "min_expected_return": null }
}
```

```python
@router.post("/explain", response_model=ExplainResponse)
def explain(body: ExplainRequest, request: Request, payload: dict = Depends(get_payload)):
    if body.use_llm:
        llm_limiter.check(request)          # raises RateLimited → 429; offline calls are never limited
    preferences = Preferences(**body.preferences.model_dump()) if body.preferences else None
    try:
        result = run_agent(payload, amount=body.amount, currency=body.currency,
                           profile=body.profile, hide_dust=body.hide_dust,
                           question=body.question, use_llm=body.use_llm,
                           preferences=preferences)
    except ValueError as error:
        raise InvalidInput(str(error)) from error
    return public(result)
```

- **Declare the handler with `def`, not `async def`.** `run_agent` blocks for up
  to 2 × 30 s of Groq calls; FastAPI runs a sync handler in its threadpool.
- Request validation: `question` is 1–2000 chars, and `question` without
  `use_llm=true` → 422. A model validator mirrors `run_agent`'s rule so the
  user gets a field-level error.
- Response passthrough: `profiles`, `buckets`, `allocation`, `news`,
  `news_status`, `orchestration`, and `explanation.{text, fact_ids, status,
  mode, provider, model, fallback_reason, tool_calls, question_answered,
  disclaimer, metadata}`.
- Rate-limit only `use_llm=true` requests, per client IP. The limit depends on
  the request *body*, so a route decorator can't express it. Use `limits`
  (slowapi's underlying library) with a `MovingWindowRateLimiter`
  wrapped in a small `llm_limiter.check(request)` helper in `api/ratelimit.py`.
  Offline explanations are unlimited.
- Never construct a news provider in the API. `news_provider` stays `None`
  until a live provider exists (ROADMAP: opt-in, outside `src/`).

#### Saved plans (`/api/plans`), behind `settings.enable_plans`

SQLite through SQLModel, single-user (no auth in v1; see §11).

```python
class Plan(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    name: str = Field(max_length=80)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    amount: str                     # decimal string
    currency: str
    profile: str
    hide_dust: bool
    dataset_start: str
    dataset_end: str
    allocation_json: str            # public(allocation.to_dict()) snapshot, re-serialized
    note: str | None = Field(default=None, max_length=500)
```

| Route | Behaviour |
| --- | --- |
| `GET /api/plans` | List (id, name, created_at, amount, currency, profile), newest first, paginated `?limit=&offset=` |
| `POST /api/plans` | Body `{name, note?, amount, currency, profile, hide_dust}`. **The server recomputes the allocation**; the client never sends amounts. Stores the snapshot |
| `GET /api/plans/{id}` | Full stored snapshot |
| `DELETE /api/plans/{id}` | 204 |
| `GET /api/plans/{id}/export.csv` | Same CSV format as the allocation export |

A snapshot keeps the dataset window, so an old plan stays reproducible and
honest if the frozen dataset is ever regenerated. The UI shows a "dataset
changed" badge when the snapshot window ≠ the current metadata.

### 5.10 Backend tests (`tests/api/`, `tests/analytics/`)

Follow the existing conventions: offline, synthetic, sockets blocked.

- `tests/api/conftest.py` builds a **synthetic payload** by running the real
  pipeline on a temporary synthetic Parquet (reuse the fixtures pattern from
  `tests/conftest.py`) and a synthetic price frame. It sets them on
  `app.state` through a test lifespan override and uses an in-memory SQLite
  database. It also reuses the socket blocker from `tests/agent/conftest.py`.
- Every endpoint: a happy path, every 422 branch, and the response model
  validates.
- `test_sanitize.py`: no `dataset_path` in any response.
- `test_explain.py`: offline mode, question-without-LLM 422, and an injected
  fake provider. Patch `GroqExplainer.from_environment` to return the fake
  that `tests/agent/test_graph.py` uses. Also check fallback on provider
  error (200, `mode=fallback`), the rate limit (429), and that no provider
  error text appears in the body.
- `test_allocation.py`: `display.total == amount`, exact targets unchanged
  versus calling `allocate_amount` directly, and CSV content.
- `test_performance.py`: `profile` XOR `frontier_index`, ranges, and a
  start value equal to the amount.
- `test_plans.py`: CRUD, the server recomputing amounts, and 404.
- `test_contract.py`: `GET /api/profiles` equals
  `public(select_profiles(payload)[n].to_dict())`, i.e. passed through unchanged.

### 5.11 Running the backend (Windows PowerShell, from `FinanceAI/`)

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r agent/requirements.txt -r api/requirements.txt
python -m api.bootstrap          # validates the local Parquet, or downloads it if FINANCEAI_DATASET_* are set
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://127.0.0.1:8000/docs   OpenAPI: http://127.0.0.1:8000/openapi.json
python -m pytest -v
```

---

## 6. Frontend (Next.js)

### 6.1 Stack

| Concern | Choice | Why |
| --- | --- | --- |
| Framework | Next.js (latest, App Router), TypeScript strict | Requested; React per ROADMAP Phase 5 |
| Styling | Tailwind CSS + shadcn/ui (Radix) | Matches the screenshot's card/pill look; accessible primitives |
| Icons | lucide-react | Matches the screenshot icons (Home, PieChart, Activity, LayoutList, RotateCw, Sparkles, Settings) |
| Charts | Recharts | Area, line, scatter and donut, which cover every chart here |
| Server state | TanStack Query | Caching, loading and error states, mutations |
| Client state | Zustand + `persist` | User settings (currency, amount, profile …) |
| Forms | react-hook-form + zod | Allocation, optimizer and plan forms |
| API types | `openapi-typescript` + `openapi-fetch` | Types generated from FastAPI; no hand-written DTOs |
| Font | `next/font` with **Plus Jakarta Sans** (or Inter) | Closest to the screenshot |
| Tests | Vitest + Testing Library; Playwright for e2e smoke | — |

### 6.2 Scaffold

```powershell
cd FinanceAI
npx create-next-app@latest gui --ts --tailwind --eslint --app --src-dir=false --import-alias "@/*"
cd gui
npx shadcn@latest init
npx shadcn@latest add button card select tabs table badge tooltip dialog sheet input label slider switch skeleton dropdown-menu separator toggle-group alert sonner
npm i @tanstack/react-query zustand recharts lucide-react openapi-fetch react-hook-form zod @hookform/resolvers date-fns
npm i -D openapi-typescript vitest @testing-library/react @testing-library/jest-dom jsdom @playwright/test
```

`package.json` scripts:

```json
{
  "gen:api": "openapi-typescript http://127.0.0.1:8000/openapi.json -o lib/api/schema.d.ts",
  "test": "vitest run",
  "e2e": "playwright test"
}
```

Commit `lib/api/schema.d.ts`. Regenerate it whenever an API schema changes.
CI checks that it is up to date (§7.3).

### 6.3 Folder layout

```text
gui/
├── app/
│   ├── layout.tsx                    # <html>, font, <Providers>
│   ├── providers.tsx                 # QueryClientProvider, TooltipProvider, Toaster
│   ├── globals.css                   # design tokens (§6.4)
│   └── (dashboard)/
│       ├── layout.tsx                # <Sidebar/> + <Topbar/> + {children} + <AssumptionsFooter/>
│       ├── page.tsx                  # Overview
│       ├── portfolio/page.tsx
│       ├── optimizer/page.tsx
│       ├── allocation/page.tsx
│       ├── plans/page.tsx            # "Transactions" slot → Saved plans
│       ├── plans/[id]/page.tsx
│       ├── explainer/page.tsx        # "AI Advisor" slot → AI Explainer
│       └── settings/page.tsx
├── components/
│   ├── shell/  sidebar.tsx, mobile-nav.tsx, topbar.tsx, page-header.tsx, date-range-pill.tsx, assumptions-footer.tsx, nav-items.ts
│   ├── kpi/    kpi-card.tsx, kpi-grid.tsx
│   ├── charts/ performance-chart.tsx, allocation-donut.tsx, frontier-chart.tsx, correlation-heatmap.tsx, price-history-chart.tsx, chart-tooltip.tsx
│   ├── portfolio/ holdings-table.tsx, profile-compare.tsx, universe-table.tsx, warnings-note.tsx
│   ├── optimizer/ preferences-form.tsx, bucket-card.tsx, infeasible-alert.tsx
│   ├── allocation/ allocation-form.tsx, allocation-table.tsx, rounding-audit.tsx, save-plan-dialog.tsx
│   ├── explainer/ question-box.tsx, answer-card.tsx, mode-badge.tsx, suggested-questions.tsx
│   └── common/ footnote.tsx, error-card.tsx, empty-state.tsx, api-status-banner.tsx
├── lib/
│   ├── api/ schema.d.ts (generated), client.ts, queries.ts, keys.ts
│   ├── format.ts                     # the ONLY formatting module
│   ├── labels.ts                     # profile labels, warning-code → text, group colours
│   └── store/settings.ts             # Zustand persisted settings
├── tests/ (vitest)  e2e/ (playwright)
├── next.config.ts
└── .env.local.example                # NEXT_PUBLIC_API_URL= (empty → dev rewrite to API_URL=http://127.0.0.1:8000)
```

### 6.4 Design system (from the screenshot)

Define the tokens in `globals.css` as CSS variables, wire them into Tailwind,
and include a dark variant (optional, in M13).

| Token | Light value | Used for |
| --- | --- | --- |
| `--bg` | `#F6F7F9` | Page background |
| `--surface` | `#FFFFFF` | Cards, sidebar |
| `--border` | `#E7E9EE` | Card borders (1px), dividers |
| `--text` | `#0F172A` | Titles, KPI values |
| `--text-muted` | `#64748B` | Subtitles, axis labels |
| `--nav-active` | `#0B1020` | Active nav pill, logo tile |
| `--primary` | `#3B6EF5` | Eyebrow text, chart line, links, avatar text |
| `--primary-soft` | `#EEF3FF` | Avatar bg, info notes, chart area gradient top |
| `--positive` | `#16A34A` | Positive returns ("+18.4%") |
| `--negative` | `#DC2626` | Negative returns, drawdown |
| `--chart-1..4` | `#3B6EF5`, `#10B981`, `#F5B83D`, `#8B5CF6` | Donut: Technology, Healthcare, Consumer, Other |

Layout metrics from the screenshot:

- The sidebar is 256px wide, white, with a right border. Nav items are 48px
  tall with a 12px radius; the active item is a navy pill with white text.
- Content padding is 32px. The KPI grid has 4 columns at ≥1280px, 2 at
  ≥768px and 1 below. Gap 20px.
- Cards: radius 16px, border 1px, padding 24px, no heavy shadows (at most `shadow-sm`).
- KPI value: 28–30px bold. Card label: 13–14px medium muted. Sub-line: 12px muted.
- Charts row: performance card ≈ 60% and allocation card ≈ 40% at ≥1280px;
  they stack below that.
- Eyebrow: 11–12px, bold, letter-spacing 0.12em, uppercase, `--primary`.

### 6.5 App shell

- **`nav-items.ts`** is the single source of nav order, labels, icons and routes:
  Overview `/`, Portfolio `/portfolio`, Optimizer `/optimizer`, Allocation
  `/allocation`, Saved plans `/plans` (icon `RotateCw` as in the screenshot),
  AI Explainer `/explainer` (icon `Sparkles`), Settings `/settings`.
- **Sidebar**: brand block plus nav. Below 1024px it becomes a `Sheet` opened
  from a hamburger in the Topbar.
- **Topbar / PageHeader**: eyebrow, title, subtitle on the left; on the right
  the `DateRangePill` (from `/api/metadata`) and the avatar menu.
- **AssumptionsFooter** (every page, required): one compact line, "Data
  2021-01-04 → 2025-12-31 · 252 trading days · risk-free 4.00% · long-only,
  fully invested, no leverage · Historical estimates, not forecasts or
  personalized advice", with a "Details" popover listing everything from
  `/api/metadata`.
- **ApiStatusBanner**: while `/api/health` is retrying (a Render free-tier
  cold start), show "Waking up the server…". If it still fails after ~90 s,
  show "Backend unavailable". In development the message adds a hint to start
  `uvicorn api.main:app`. See §7.4.

### 6.6 Data layer

`lib/api/client.ts`:

```ts
import createClient from "openapi-fetch";
import type { paths } from "./schema";

// Vercel: NEXT_PUBLIC_API_URL = Render URL (browser calls it directly, CORS-allowed).
// Local dev: unset → same-origin, next.config rewrites /api/* to uvicorn. See §7.4.
export const api = createClient<paths>({ baseUrl: process.env.NEXT_PUBLIC_API_URL ?? "" });

export class ApiError extends Error {
  constructor(public code: string, message: string, public status: number) { super(message); }
}

export function unwrap<T>(res: { data?: T; error?: any; response: Response }): T {
  if (res.error || res.data === undefined) {
    const e = res.error?.error ?? { code: "network_error", message: "Request failed" };
    throw new ApiError(e.code, e.message, res.response.status);
  }
  return res.data;
}
```

`lib/api/queries.ts` holds one hook per endpoint. Read data is frozen, so use
`staleTime: Infinity` for metadata, universe, frontier and profiles.

| Hook | Endpoint | Key |
| --- | --- | --- |
| `useHealth()` | `GET /api/health` | `["health"]`, refetch every 60 s |
| `useMetadata()` | `GET /api/metadata` | `["metadata"]` |
| `useUniverse()`, `useCorrelation()`, `usePriceHistory(range, freq)` | `/api/universe*` | `["universe", ...]` |
| `useFrontier()` | `/api/frontier` | `["frontier"]` |
| `useProfiles()` | `/api/profiles` | `["profiles"]` |
| `usePerformance(params)` | `/api/performance` | `["performance", params]`, `placeholderData: keepPreviousData` |
| `useAllocation(params)` | `POST /api/allocate` as a *query* (deterministic, keyed by inputs, debounced 300 ms) | `["allocation", params]` |
| `useBuckets(prefs)` | `POST /api/buckets` as a query, debounced | `["buckets", prefs]` |
| `useExplain()` | `POST /api/explain` | mutation |
| `usePlans()`, `usePlan(id)`, `useCreatePlan()`, `useDeletePlan()` | `/api/plans*` | `["plans"]`, invalidated by mutations |

`lib/store/settings.ts` (Zustand `persist`, localStorage wrapped in try/catch):

```ts
type Settings = {
  currency: "INR" | "USD";          // default "INR" to match the screenshot
  amount: string;                   // decimal string, default "1000000.00"
  profile: "low" | "medium" | "high"; // default "medium" → "Balanced"
  range: "1Y" | "3Y" | "5Y";        // default "5Y"
  hideDust: boolean;                // default false
  useLlm: boolean;                  // default false (opt-in)
  showBenchmark: boolean;           // default true
};
```

To avoid hydration mismatch, read the store only in client components and
render skeletons until `useSettings.persist.hasHydrated()`.

### 6.7 Formatting rules (`lib/format.ts`, the only place that formats)

```ts
const LOCALE = { INR: "en-IN", USD: "en-US" } as const;

export const money = (s: string, c: "INR" | "USD", digits = 0) =>
  new Intl.NumberFormat(LOCALE[c], { style: "currency", currency: c,
    minimumFractionDigits: digits, maximumFractionDigits: digits }).format(Number(s));
// money("6000900.00","INR") → "₹60,00,900"

export const moneyCompact = (n: number, c: "INR" | "USD") =>
  c === "INR"
    ? n >= 1e7 ? `₹${+(n / 1e7).toFixed(2)}Cr` : `₹${+(n / 1e5).toFixed(0)}L`   // axis ticks "₹70L"
    : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact" }).format(n);

export const pct = (x: number, { digits = 1, sign = false } = {}) =>
  `${sign && x > 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`;     // pct(0.184,{sign:true}) → "+18.4%"

export const ratio = (x: number) => x.toFixed(2);                  // Sharpe
export const monthYear = (iso: string) => format(parseISO(iso), "MMM yyyy");
```

- `Number(moneyString)` is allowed only inside the formatters and chart data
  mapping, never in sums.
- Unit-test every formatter (Indian grouping, negative numbers, zero,
  1 paisa, 999999999999.99).

### 6.8 Pages

Every page has: a `PageHeader`; loading skeletons that match the final layout;
an `ErrorCard` with Retry on failure; the `AssumptionsFooter`; and footnote
markers `*` where §3 requires them.

#### 6.8.1 Overview `/` (pixel-match the screenshot)

- **Header**: eyebrow "PORTFOLIO INTELLIGENCE", title "Portfolio Overview",
  subtitle "Track performance, diversification and allocation." Right side:
  DateRangePill and avatar.
- **KPI grid** (4 cards, each with a top-right icon as in the screenshot):
  1. *Portfolio value* (`$` or `₹` icon). Value: `money(perf.summary.end_value)`.
     Sub-line: "↑ {money(gain)} gain over {range}*" in green (red ↓ if negative).
     Below: a small "Planned: {money(amount)} · Edit" link that opens a popover
     with an amount input bound to `settings.amount`.
  2. *Historical return* (`ArrowUpRight`). Value:
     `pct(profile.expected_return, {sign:true})`, in green if positive. Sub-line:
     "Annual historical estimate* · CAGR {pct(cagr)}".
  3. *Assets in universe* (`Box`). Value: `metadata.asset_count`. Sub-line:
     "Predefined US equities". Click → `/portfolio?tab=universe`.
  4. *Risk profile* (`Gauge`). Value: label (Conservative/Balanced/Growth).
     Sub-line: "Sharpe {ratio} · Select a profile in Optimizer". Click → `/optimizer`.
- **Portfolio performance card**:
  - Title, plus the subtitle "In-sample value of your planned amount
    ({profile label})". A range `Select` (1Y/3Y/5Y) is bound to `settings.range`.
  - `PerformanceChart`: Recharts `AreaChart`. Blue line, blue→transparent
    gradient fill, dashed horizontal grid, y-axis `moneyCompact`, x-axis years.
    Dots on quarterly points (white fill, blue stroke) to match the screenshot.
    With `freq=M`, show only the hover dot.
  - The tooltip shows the date, value, the benchmark value if enabled, and the
    change from start.
  - A legend toggle for "Equal-weight benchmark" is bound to `settings.showBenchmark`.
  - Footnote: the `notice` from the API, plus the currency notice when INR.
- **Asset allocation card**:
  - Title "Asset allocation", subtitle "{profile label} profile, by sector
    group". "View details" → `/portfolio?tab=holdings`.
  - `AllocationDonut`: `PieChart` with `innerRadius≈70%`, `paddingAngle` 2,
    rounded corners, and the centre label "Allocated / 100%".
  - Legend: a coloured dot, the group name and a right-aligned bold `pct(weight, {digits:0})`,
    from `breakdowns[profile].top_groups`. A segmented control switches between
    "By group" and "By stock" (stock mode shows the top 4 + Other, again from the API).
  - `WarningsNote` in the blue info box: map warning codes through
    `labels.ts` (`concentration_threshold_exceeded` → "One holding exceeds
    40% of the portfolio", etc.), or "No concentration flags for this profile."

Data: `useMetadata`, `useProfiles`, and `usePerformance({profile, amount,
currency, range, freq:"Q", benchmark})`. All three requests run in parallel.

#### 6.8.2 Portfolio `/portfolio` (tabs: Holdings · Compare profiles · Stock universe)

- **Holdings** (for `settings.profile`; a profile switcher at the top):
  - Metric strip: expected return, volatility, Sharpe, material holdings,
    and the frontier index with its selection-rule text
    (`selection.rule`). "Why this portfolio?" opens a popover with the
    selection rule and parameters.
  - `HoldingsTable`: ticker, company, sector, weight (with a bar), and the
    amount for the planned amount from `useAllocation`. Each stock's historical
    return and volatility come from `/api/universe`. Rows with a weight
    < 0.5% are greyed, with a "dust" badge.
  - Group breakdown: a bar list.
  - `WarningsNote` and the profile `disclaimer`.
- **Compare profiles**: three columns (Conservative / Balanced / Growth), each
  with return, volatility, Sharpe, a mini donut, the top 3 holdings and
  warnings, plus overlap badges (`selection.overlaps_with`). A "Set as my
  profile" button updates `settings.profile`.
- **Stock universe**:
  - `UniverseTable`, sortable by return, volatility and name.
  - `CorrelationHeatmap`: a 10×10 CSS grid with a diverging scale
    (−1 red → 0 white → +1 blue), the value on hover, and an accessible
    `<table>` fallback.
  - `PriceHistoryChart`: multi-line chart rebased to 100, with a range select
    and toggles to show or hide each ticker.

#### 6.8.3 Optimizer `/optimizer`

- **Left: `PreferencesForm`**:
  - Emphasis `ToggleGroup`: Lower risk / Balanced / Higher return.
  - "Max volatility" slider with an enable switch. Its range is
    `[frontier_range.min_volatility, frontier_range.max_volatility]`.
  - "Min expected return" slider with an enable switch, ranged from the
    frontier min to max.
  - Values show as percentages but are sent as decimals.
  - A live preview of the request is debounced by 300 ms.
- **Right: `FrontierChart`** (Recharts `ComposedChart`):
  - The frontier as a line plus small dots; the x-axis is volatility and the
    y-axis is expected return, both shown as %.
  - Individual stocks as labelled scatter points.
  - GMV (diamond) and Max Sharpe (star) markers, with a tooltip noting
    "exact optimum, between sampled points".
  - L/M/H profile markers.
  - The **feasible window** shaded via `ReferenceArea`, from `window.first_index`
    to `last_index`.
  - Bucket markers (C/B/G), with the recommended one highlighted.
  - Hovering any point shows a tooltip with return, vol, Sharpe and its top 5
    weights. Clicking pins it in a side panel with full weights and a "View
    growth" button (→ `usePerformance({frontier_index})` in a dialog).
- **Below: three `BucketCard`s**, each with the name, a "Recommended" badge, metrics,
  a mini donut, warnings and overlaps. The bucket "why" text is optional:
  call `/api/explain` with `preferences` (offline) and show only the
  `bucket.*` facts. Never write explanatory prose in the frontend; use the
  fact text the API returns.
- **Infeasible state**: an `InfeasibleAlert` that maps `reason` to text
  (`return_floor_above_frontier` → "No portfolio in this universe reached that
  return historically. Highest available: {pct}"). It also shows the `closest`
  bucket card and a "Relax limits" button that resets the sliders to the
  frontier range.
- Footnote: "Buckets are existing efficient-frontier points; no new
  optimization is run."

#### 6.8.4 Allocation `/allocation`

- **`AllocationForm`** (react-hook-form + zod):
  - Amount: text input, validated with `/^\d{1,12}(\.\d{1,2})?$/`, showing
    Indian grouping while typing for INR.
  - Currency: segmented INR/USD.
  - Profile select.
  - A "Hide holdings below 0.5%" switch.
  - Inputs initialise from settings, and a "Save as default" button writes them back.
- **Results**:
  - Summary strip: total (always equals the amount), profile, holdings count
    and hidden-dust total.
  - `AllocationTable`: ticker, company, group, weight, amount, from
    `display.amounts`. If dust is hidden, add a final row "Small holdings (n)"
    with `display.hidden_amount`. The footer total is `display.total`.
  - A donut (by group) next to the table.
  - `RoundingAudit` (collapsible): the exact `target_amounts`,
    `rounding_adjustments`, `display.method` and `display.notice`.
  - The `notice` from the API, *verbatim*: "Target monetary allocations only;
    no trades, share counts, live prices, or currency conversion…".
- **Actions**: "Export CSV" (`POST /api/allocate/export.csv` → download),
  "Save plan" (`SavePlanDialog`: name and note → `POST /api/plans`, then a
  toast with a link), and "Explain this allocation" (→ `/explainer` with the
  allocation context prefilled via query params).

#### 6.8.5 Saved plans `/plans` and `/plans/[id]` (the screenshot's *Transactions* slot)

- **List**: a table with name, created date, amount, currency, profile and
  dataset window (a badge if it differs from the current one), plus row actions
  View, Export CSV and Delete (with a confirm dialog). The empty state says
  "No saved plans yet — create one in Allocation" with a CTA.
- **Detail**: the stored snapshot rendered with the same `AllocationTable` and
  donut, plus the note. A "Re-run with current data" action opens Allocation
  prefilled. There is also a compare mode that selects two plans to show side
  by side (amounts per ticker).
- Page subtitle: "Plans are saved target allocations, not executed trades."

#### 6.8.6 AI Explainer `/explainer`

- A **context bar** at the top shows what is sent: the profile, the amount and
  currency (if "Include my allocation" is on), and the preferences (if "Include
  my optimizer limits" is on). Each item is a removable chip.
- **AI mode switch** (bound to `settings.useLlm`), disabled with a tooltip when
  `/api/health.llm_configured` is false. Off means offline: questions are
  disabled and "Generate overview" returns the deterministic comparison.
- **`QuestionBox`**: a textarea, 1–2000 characters with a counter;
  submit is disabled when the AI is off.
- **`SuggestedQuestions`** chips: "Compare the three profiles", "Why is the
  high profile concentrated?", "What are the medium profile's weights?",
  "Show frontier point 10", "Which stock had the highest historical
  volatility?" The list also includes one *unsupported* example, "Will NVDA go
  up next year?", which demonstrates the refusal path.
- **`AnswerCard`** for each turn, newest first, kept only in page state with
  no memory between questions:
  - `ModeBadge`: `llm` (blue "AI-selected facts"), `offline` (grey
    "Deterministic"), `fallback` (amber "AI unavailable — {fallback_reason}").
  - Status: `supported` / `unsupported` (an amber note: "That question is outside
    the supported historical data").
  - `explanation.text` rendered as paragraphs, **verbatim**. Don't reword,
    summarise or convert markdown.
  - Disclosure "Sources": `fact_ids` and `tool_calls` (names and args only).
  - Footer: model name, orchestration, disclaimer.
- Long-request UX: a progress state ("Selecting facts…") and a cancel button
  (an AbortController aborts the fetch only; the server finishes regardless).
  On a 429, show "Too many AI requests — try again in a minute".

#### 6.8.7 Settings `/settings`

- **Preferences**: default currency, default amount, default profile, hide
  dust, show benchmark, and the performance chart sampling (monthly/quarterly).
- **AI**: an enable toggle and a status from `/api/health` ("Configured" /
  "Not configured: add GROQ_API_KEY to FinanceAI/.env and install
  agent/requirements.txt").
- **Data & assumptions** (read-only): the full `/api/metadata` as a definition
  list, the stock list with sectors, and the disclaimer.
- **About**: method summary (MPT, historical estimates), with links to the
  README sections.
- **Danger zone**: "Reset local settings" (clears the Zustand store) and
  "Delete all saved plans" (confirm dialog, then a `DELETE` of each plan, or
  add `DELETE /api/plans` if wanted).

### 6.9 Cross-cutting UI requirements

- **Responsive**: works from 360px up with no horizontal page scroll. Tables
  scroll inside their card. Charts use `ResponsiveContainer`.
- **Accessibility**: every chart has an `aria-label` and a "View as table"
  toggle, and the colour is never the only signal (signs and arrows go
  with green and red). Focus rings stay visible. Dialogs use Radix.
- **Number display**: right-align numeric columns and use `tabular-nums`.
- **Empty, loading and error states** for every data-driven component. Use
  skeletons, not spinners, for cards.
- **No illustrative or hard-coded financial numbers** anywhere in the
  components. Add a lint or grep check in CI for currency literals like `₹\d`
  in `components/`.

### 6.10 Frontend tests

- Vitest: `format.ts` (exhaustive); `labels.ts` warning mapping; `KpiCard`
  states; `AllocationTable` showing the hidden-dust row and total;
  `ModeBadge` variants; `PreferencesForm` sending decimals, not percentages.
- Mock the API with `msw` and fixtures captured from the real API
  (`gui/tests/fixtures/*.json`, generated once by a script calling the local API).
- Playwright smoke (`e2e/smoke.spec.ts`), against a running API with the real
  frozen dataset: Overview renders 4 KPIs and both charts; Allocation of
  `1000000.00 INR` shows a total of `₹10,00,000.00`; Optimizer infeasible
  limits show the alert; Explainer offline overview renders the disclaimer.

---

## 7. Integration, tooling and deployment

**Deployment decision:** the backend runs as a Docker web service on
**Render**, and the frontend is deployed on **Vercel** only. The frozen Parquet
dataset is **not** stored in Git or in the Docker image. The API server
**downloads it dynamically when it starts**, from a versioned,
checksum-pinned artifact.

### 7.1 Topology

```text
                 ┌──────────────────────────── Vercel ─────────────────────────────┐
 Browser ───────►│ Next.js (gui/)  static + client JS   env: NEXT_PUBLIC_API_URL   │
    │            └──────────────────────────────────────────────────────────────────┘
    │  fetch https://financeai-api.onrender.com/api/*   (CORS allow-list)
    ▼
 ┌──────────────────────────── Render (Docker web service) ─────────────────────────┐
 │ 1. python -m api.bootstrap ──HTTPS GET──► GitHub Release asset  dataset-v1        │
 │      verify SHA-256 → atomic write data/raw/stock_prices.parquet                   │
 │      → validate with src.data_loader.load_stock_prices()                           │
 │ 2. uvicorn api.main:app  (lifespan runs the MPT pipeline once on that file)        │
 │ env: FINANCEAI_DATASET_URL, FINANCEAI_DATASET_SHA256, GROQ_API_KEY, DB url, CORS   │
 └──────────────────────────────────────┬────────────────────────────────────────────┘
                                        ▼
                         Neon Postgres (saved plans; FINANCEAI_DATABASE_URL)
```

### 7.2 Dataset delivery (dynamic download at server startup)

#### Why a release artifact and not a Yahoo download on the server

"Dynamic download" here means **fetching our own frozen file**. It does
**not** mean running `python -m src.download_data` on Render:

- **Results would drift.** Yahoo revises adjusted closes after later
  dividends and splits, so a fresh download changes μ, Σ and every weight.
  The figures in the README, the screenshots in the report and saved plans
  would stop matching what the server shows.
- **AGENTS.md forbids it.** Dataset download must never be implicit, and
  "frozen" is the core design choice of the project.
- **It isn't reliable.** Yahoo often rate-limits or blocks datacenter IPs,
  so deploys would fail at random.
- **The release artifact satisfies AGENTS.md.** README calls for "an
  approved project data-sharing mechanism"; a versioned, checksum-pinned
  release artifact is exactly that.

Yahoo access stays confined to `src/download_data.py`, run manually on a
developer machine.

#### One-time: publish the frozen file (developer machine, from `FinanceAI/`)

```powershell
# 1. Confirm the local file is the verified one (headline values in README match)
python -m src.main
# 2. Record its checksum
(Get-FileHash data\raw\stock_prices.parquet -Algorithm SHA256).Hash.ToLower()
# 3. Publish it as an immutable, versioned release asset
gh release create dataset-v1 data/raw/stock_prices.parquet `
  --title "Frozen dataset v1 (2021-01-01 to 2025-12-31, 10 tickers)" `
  --notes "Adjusted closes from python -m src.download_data. SHA-256: <hash>. Do not replace this asset; publish dataset-v2 instead."
```

- Public repo: the URL is
  `https://github.com/<owner>/<repo>/releases/download/dataset-v1/stock_prices.parquet`
  and needs no token.
- Private repo: use the API asset URL
  `https://api.github.com/repos/<owner>/<repo>/releases/assets/<asset_id>` plus a
  fine-grained token with **read-only Contents** permission on this one repo,
  stored as `FINANCEAI_DATASET_TOKEN` on Render.
- **Never replace an existing asset.** A new dataset means a new tag
  (`dataset-v2`), a new SHA-256, and the AGENTS.md change procedure (state the
  assumption, get approval, update tests and docs). Then update the two Render
  env vars. Because the URL and hash are pinned, any deployed server can be
  traced to an exact dataset.

#### Startup bootstrap (`api/bootstrap.py`)

This lives in `api/`, not in `src/`, because it is deployment code. It never
contacts Yahoo and imports nothing from `yfinance`.

```python
"""Fetch the frozen dataset artifact before the API starts. Not a market-data download."""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

from src.config import RAW_PRICES_PATH
from src.data_loader import load_stock_prices

MAX_BYTES = 20 * 1024 * 1024        # the real file is ~105 KB; refuse anything absurd
TIMEOUT_SECONDS = 60


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(url: str) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={"User-Agent": "financeai-bootstrap"})
    token = os.environ.get("FINANCEAI_DATASET_TOKEN")
    if token:                                         # private GitHub release asset
        request.add_header("Accept", "application/octet-stream")
        # Unredirected: the token must not be forwarded to the storage host GitHub redirects to.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def ensure_dataset() -> str:
    url = os.environ.get("FINANCEAI_DATASET_URL")
    expected = os.environ.get("FINANCEAI_DATASET_SHA256", "").strip().lower()
    if RAW_PRICES_PATH.exists():
        if not expected or _sha256(RAW_PRICES_PATH) == expected:
            load_stock_prices()                       # local dev: existing file, still validated
            return "present"
    if not url or not expected:
        raise SystemExit("Dataset missing: set FINANCEAI_DATASET_URL and FINANCEAI_DATASET_SHA256.")
    if not url.startswith("https://"):
        raise SystemExit("FINANCEAI_DATASET_URL must use HTTPS.")
    RAW_PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = RAW_PRICES_PATH.with_suffix(".parquet.part")
    with urllib.request.urlopen(_request(url), timeout=TIMEOUT_SECONDS) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise SystemExit("Dataset artifact exceeds the size limit.")
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit("Dataset checksum mismatch; refusing to start.")
    partial.write_bytes(data)
    os.replace(partial, RAW_PRICES_PATH)              # atomic: never a half-written Parquet
    load_stock_prices()                               # schema, ticker order, NaN, positivity
    return "downloaded"


if __name__ == "__main__":
    print(f"dataset: {ensure_dataset()}", file=sys.stderr)
```

Behaviour and rules:

- **Fail closed.** Any download error, size overflow, checksum mismatch or
  validation error exits non-zero. The container then never starts serving and
  Render keeps the previous healthy deploy live. The API must **never** start
  on a missing or unverified dataset.
- **Re-download on every cold start.** Render's filesystem is ephemeral, so a
  restart or a free-tier wake-up downloads the file again. At ~105 KB this
  costs well under a second.
- **Local development is unchanged.** If the file exists and no hash is
  configured, the bootstrap only validates it. Developers keep using their
  local frozen copy.
- **The cache stays consistent.** `agent.cache.load_analysis()` keys on
  the file's path, mtime and size; the file is written before the pipeline first
  runs, so nothing stale is cached.
- **Logs never include the token or the full signed redirect URL.**
- Tests (`tests/api/test_bootstrap.py`, offline): monkeypatch
  `urllib.request.urlopen` to serve bytes from a synthetic Parquet fixture.
  Cover: a correct hash writes the file and validates; a wrong hash exits and
  leaves no file or `.part`; HTTP URLs are rejected; oversize is rejected; an
  existing file with a matching hash is not downloaded; the Authorization
  header is unredirected; no real socket is opened.

### 7.3 Backend on Render

#### `api/Dockerfile` (build context = repository root `FinanceAI/`)

```dockerfile
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY agent/requirements.txt ./agent/requirements.txt
COPY api/requirements.txt ./api/requirements.txt
RUN pip install -r requirements.txt -r agent/requirements.txt -r api/requirements.txt

COPY src ./src
COPY agent ./agent
COPY analytics ./analytics
COPY api ./api
COPY data/reference ./data/reference

RUN useradd --create-home app && mkdir -p data/raw && chown -R app /app
USER app

# 1) fetch + verify the frozen dataset, 2) serve. Render injects $PORT.
CMD ["sh", "-c", "python -m api.bootstrap && exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
```

`.dockerignore` (repo root):

```text
.git
.venv
**/__pycache__
.env
.env.*
data/raw/*.parquet
data/app/
outputs/
gui/
tests/
```

- `data/raw/*.parquet` is ignored, so the image provably carries no dataset
  and the bootstrap is the only source.
- `--proxy-headers` makes `request.client.host` the real visitor IP behind
  Render's proxy, which the `/explain` LLM rate limit needs.

#### `render.yaml` (Render Blueprint, committed at repo root)

```yaml
services:
  - type: web
    name: financeai-api
    runtime: docker
    plan: starter                 # free tier works but sleeps when idle (see below)
    region: singapore             # closest Render region to India
    dockerfilePath: ./api/Dockerfile
    dockerContext: .
    healthCheckPath: /api/health
    autoDeployTrigger: checksPass # deploy main only after GitHub Actions succeed
    envVars:
      - key: FINANCEAI_DATASET_URL
        sync: false               # set in dashboard
      - key: FINANCEAI_DATASET_SHA256
        sync: false
      - key: FINANCEAI_DATASET_TOKEN
        sync: false               # only for a private repo
      - key: FINANCEAI_DATABASE_URL
        sync: false               # Neon connection string
      - key: GROQ_API_KEY
        sync: false
      - key: FINANCEAI_CORS_ORIGINS
        value: '["https://financeai.vercel.app"]'
      - key: FINANCEAI_CORS_ORIGIN_REGEX
        value: '^https://financeai-[a-z0-9-]+-<vercel-team-slug>\.vercel\.app$'
```

Notes:

- **Render plan.** The free plan spins down after ~15 minutes idle; the next
  request then waits for container start, the dataset download and the MPT
  pipeline (tens of seconds). Use Starter for demo or viva days, or open the
  site a minute early. The GUI handles this with a "Waking up the server…"
  state (§7.4).
- **Health check.** `/api/health` must return 200 only after the lifespan has
  loaded the payload. A failed bootstrap therefore fails the health check and
  Render does not switch traffic.
- **Secrets.** `GROQ_API_KEY`, the dataset token and the DB URL live **only** in
  Render env vars, never in `render.yaml` values, the image or Vercel.

#### Database: Neon Postgres for saved plans

Render's disk is ephemeral, so the SQLite file would be lost on every deploy
or restart.

- Create a free Neon project and set
  `FINANCEAI_DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>/<db>?sslmode=require`
  on Render.
- Add `psycopg[binary]` to `api/requirements.txt`. SQLModel code is
  unchanged; local development keeps the SQLite default.
- v1 creates tables with `SQLModel.metadata.create_all` in `init_db()`. Add
  Alembic before changing the `Plan` table after launch.
- Alternative: Render Postgres. Its free instances expire, so prefer Neon for
  a long-lived demo.

#### Config additions (`api/config.py`)

```python
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cors_origin_regex: str | None = None          # Vercel preview deployments
    database_url: str = "sqlite:///data/app/financeai.db"
```

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

Pin the regex to your **team slug** (preview URLs look like
`financeai-<hash>-<team>.vercel.app`). A loose `financeai-.*\.vercel\.app`
would admit anyone's Vercel project with a similar name. CORS isn't
authentication, but there's no reason to be permissive.

### 7.4 Frontend on Vercel

Project settings (Vercel dashboard → Import Git repository):

| Setting | Value |
| --- | --- |
| Root Directory | `gui` |
| Framework preset | Next.js (auto-detected) |
| Node.js version | 22.x |
| Env `NEXT_PUBLIC_API_URL` (Production) | `https://financeai-api.onrender.com` |
| Env `NEXT_PUBLIC_API_URL` (Preview) | Same URL (or a separate `financeai-api-staging` Render service) |
| Ignored Build Step | `git diff --quiet HEAD^ HEAD -- .`, which skips frontend builds when only backend files changed |

- **The browser calls Render directly** (`NEXT_PUBLIC_API_URL`). There is
  no Vercel proxy, so a ~60 s `/api/explain` request never runs into a
  Vercel proxy or function timeout. Vercel serves only static assets and the
  Next.js app.
- **Nothing secret goes into Vercel.** Anything prefixed `NEXT_PUBLIC_` is
  shipped to every browser.
- `gui/lib/api/client.ts` changes from the earlier same-origin version:

```ts
// Production: absolute Render URL. Local dev: unset → same-origin + next.config rewrite.
export const api = createClient<paths>({ baseUrl: process.env.NEXT_PUBLIC_API_URL ?? "" });
```

- `gui/next.config.ts`: the rewrite exists **only for local development**:

```ts
import type { NextConfig } from "next";

const devApi = process.env.API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    if (process.env.NEXT_PUBLIC_API_URL) return [];          // deployed: browser calls Render directly
    return [{ source: "/api/:path*", destination: `${devApi}/api/:path*` }];
  },
  experimental: { proxyTimeout: 90_000 },                     // dev proxy only; explain can take ~60 s
};
export default nextConfig;
```

- **Cold-start UX** (because of the Render free tier):
  - Call `/api/health` as soon as the app loads, which also wakes the server.
  - `useHealth` retries with backoff for up to ~90 s, and during that time the
    `ApiStatusBanner` shows "Waking up the server…".
  - Data queries wait on `health.isSuccess` (TanStack Query `enabled`), so the
    page doesn't show a wall of errors.

### 7.5 Local development (two terminals)

```powershell
# Terminal 1 — API (from FinanceAI/). Uses your local frozen Parquet; bootstrap only validates it.
.venv\Scripts\Activate.ps1
python -m api.bootstrap
uvicorn api.main:app --reload --port 8000

# Terminal 2 — GUI (from FinanceAI/gui). Leave NEXT_PUBLIC_API_URL unset → dev rewrite is used.
copy .env.local.example .env.local
npm run dev        # http://localhost:3000
```

A teammate without the Parquet file sets `FINANCEAI_DATASET_URL` and
`FINANCEAI_DATASET_SHA256` in `FinanceAI/.env` (or the shell) and runs
`python -m api.bootstrap`. They get the exact same file as production and never
need Yahoo.

Optional production parity: `docker build -f api/Dockerfile -t financeai-api .`
then `docker run --env-file .env -p 8000:8000 financeai-api`. This
exercises the real download path.

### 7.6 CI/CD workflow

```text
feature branch ──push──► GitHub Actions (tests.yml) ──► PR
                                  │                      │
                                  │                      └─► Vercel preview deploy (gui/ changes only)
                                  ▼
                      merge to main ──► Actions green ──► Render auto-deploy (checksPass)
                                                     └──► Vercel production deploy
Render deploy: build image → container start → api.bootstrap (download + SHA-256 + validate)
            → uvicorn lifespan (pipeline) → /api/health 200 → traffic switched
```

`.github/workflows/tests.yml`: keep the existing `baseline` job exactly as it
is, and add these:

```yaml
  api:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.13' }
      - run: python -m pip install -r requirements.txt -r agent/requirements.txt -r api/requirements.txt
      - run: python -m pip check
      - run: python -m pytest tests/api tests/analytics -v      # synthetic fixtures only
      - name: OpenAPI schema drift
        run: |
          python -c "import json; from api.main import app; print(json.dumps(app.openapi()))" > openapi.json
          npx --yes openapi-typescript openapi.json -o gui/lib/api/schema.d.ts
          git diff --exit-code gui/lib/api/schema.d.ts

  dataset:                       # real frozen data, fetched exactly like Render does
    runs-on: ubuntu-latest
    env:
      FINANCEAI_DATASET_URL: ${{ vars.FINANCEAI_DATASET_URL }}
      FINANCEAI_DATASET_SHA256: ${{ vars.FINANCEAI_DATASET_SHA256 }}
      FINANCEAI_DATASET_TOKEN: ${{ secrets.FINANCEAI_DATASET_TOKEN }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.13' }
      - run: python -m pip install -r requirements.txt -r api/requirements.txt
      - run: python -m api.bootstrap
      - run: python -m src.main
      - run: python -c "import json; from src.pipeline import run_mpt_analysis; json.dumps(run_mpt_analysis().to_dict(), allow_nan=False); print('JSON contract valid')"

  gui:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: gui } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '22', cache: npm, cache-dependency-path: gui/package-lock.json }
      - run: npm ci
      - run: npm run lint
      - run: npx tsc --noEmit
      - run: npm test
      - run: npm run build
        env: { NEXT_PUBLIC_API_URL: https://example.invalid }
```

- `api` and `gui` jobs stay offline and use synthetic data. The `dataset` job
  is the only one that touches the real frozen file, and it uses the same
  bootstrap and checksum as Render. A broken release asset or wrong hash
  therefore fails CI before it can fail a deploy.
- Store `FINANCEAI_DATASET_URL` and `FINANCEAI_DATASET_SHA256` as GitHub
  Actions **variables** (they aren't secret), and the token (private repo
  only) as a **secret**.
- With `autoDeployTrigger: checksPass`, Render deploys `main` only after
  all of these jobs pass.

### 7.7 Release checklist (each production deploy)

1. All CI jobs are green on `main`.
2. The Render deploy log shows `dataset: downloaded` and the health check passed.
3. `GET https://financeai-api.onrender.com/api/metadata` shows the expected
   `start_date`, `end_date`, `observations` and `asset_count`.
4. `GET /api/frontier` shows a GMV `expected_return` ≈ 0.139390 and a
   `max_sharpe.sharpe_ratio` ≈ 1.5168, which are the README's verified values
   for `dataset-v1`.
5. The Vercel production URL loads Overview with real KPIs, and the browser
   console has no CORS errors.
6. `/api/health` reports `llm_configured: true` (if the AI is meant to be on),
   and an Explainer question returns `mode: "llm"`.
7. Creating a saved plan, redeploying the API, and checking the plan
   confirms that Neon persistence works.

---

## 8. Milestones and acceptance criteria

Work on a feature branch (e.g. `feature/api-gui`) with small, validated commits.
After every milestone, run the AGENTS.md required checks plus the new test suites.

| # | Milestone | Deliverables | Done when |
| --- | --- | --- | --- |
| M0 | Prep | `.env` at `FinanceAI/.env`; venv with all deps; branch; `.gitignore` adds `data/app/`, `data/raw/*.part`, `gui/node_modules`, `gui/.next`, `gui/.env.local` | All existing checks in AGENTS.md pass unchanged |
| M1 | API foundation | `api/main.py`, config, deps, sanitize, errors, `/health`, `/metadata`, test harness with synthetic payload and socket block | `uvicorn` starts; `/docs` loads; tests pass; no `dataset_path` in responses |
| M2 | Reference and analytics | `tickers.json`, `analytics/{reference,universe,breakdown,performance}.py` + tests | Hand-computed simulation matches; correlation diagonal = 1; reference must equal the universe |
| M3 | Read endpoints | `/universe`, `/universe/correlation`, `/universe/prices`, `/frontier`, `/profiles`, `/profiles/{name}`, `/performance` | Contract test: profiles pass through unchanged; performance start = amount |
| M4 | Action endpoints | `/allocate`, `/allocate/export.csv`, `/buckets`, `/explain` + rate limit | `display.total == amount`; infeasible buckets → 200; explain offline/fake-LLM/fallback tested; 429 tested |
| M5 | Saved plans | `api/db.py`, `/plans` CRUD + CSV | Server recomputes amounts; 404 path; works with `enable_plans=false` (routes return 404) |
| M6 | GUI foundation | Scaffold, tokens, shell (sidebar, topbar, footer, mobile nav), providers, generated types, `format.ts`, settings store, ApiStatusBanner | The shell matches the screenshot at 1440px; formatter tests pass; it builds |
| M7 | Overview | KPI grid, PerformanceChart, AllocationDonut, WarningsNote | Side-by-side with the screenshot: same layout, real values, footnotes present |
| M8 | Portfolio | Holdings, Compare, Universe tabs incl. heatmap and price history | Every number traces to an API field |
| M9 | Allocation | Form, table, donut, audit, CSV export, save | ₹10,00,000.00 reconciles exactly; hidden-dust row correct |
| M10 | Optimizer | Preferences form, FrontierChart with markers/window/buckets, bucket cards, infeasible state | Slider changes update buckets; infeasible handled; values sent as decimals |
| M11 | AI Explainer | Context bar, mode switch, question box, answer cards, suggested questions | Offline, LLM, fallback and unsupported paths all render correctly; text verbatim |
| M12 | Plans and Settings | List/detail/compare/delete; settings sections | Plans persist across API restarts; reset works |
| M13 | Polish | Responsive pass, a11y pass, optional dark mode, Playwright smoke, docs | Smoke passes; README/ROADMAP updated |
| M14 | Dataset delivery | `dataset-v1` release asset and recorded SHA-256; `api/bootstrap.py` + `test_bootstrap.py`; GitHub Actions variables | Fresh clone without Parquet: `python -m api.bootstrap` downloads, verifies and validates; wrong hash exits non-zero and leaves no file |
| M15 | Deployment | `api/Dockerfile`, `.dockerignore`, `render.yaml`, Neon DB, CORS regex, Vercel project (root `gui`), `NEXT_PUBLIC_API_URL`, CI jobs `api` / `dataset` / `gui` | Release checklist §7.7 passes end to end on the production URLs |

**Stretch goals** (each needs its own design note, tests and docs):

- **Allocate a bucket or any frontier point.** Add
  `agent.allocation.allocate_frontier_point(amount, currency, payload, index)`
  as an *additive* function that reuses the same validation and display code,
  then `POST /api/allocate` with `frontier_index`. The Optimizer cards then get
  an "Allocate" button.
- Live news provider (ROADMAP §8 "Next"): opt-in, outside `src/`, sanitized;
  shown as attributed context in the Explainer.
- Out-of-sample backtest page (ROADMAP Phase 6, `research/`): train window vs
  test window. This is the honest replacement for the in-sample performance chart.
- Auth (NextAuth/Auth.js + API JWT) if the app is ever deployed for multiple users.

---

## 9. Documentation updates

Update in the same commits as the behaviour change:

- **README.md**: new "API service" and "Dashboard" sections (setup, run
  commands, endpoint table, in-sample performance method and its hindsight
  caveat); add `api/`, `analytics/`, `gui/` and `data/reference/` to the
  project structure.
- **ROADMAP.md**: mark Phase 4/5 items as done; record the decisions from §11;
  move "React · Streamlit" to "Selected: Next.js (React)".
- **AGENTS.md**: add `analytics/` and `api/` rows to module ownership. Add the
  rule "API never accepts client weights; responses strip `dataset_path`".
  Extend the required checks with `python -m pytest tests/api tests/analytics -v`.
- **CLAUDE.md**: the commands for bootstrap, uvicorn and `npm run dev`, plus
  the `proxyTimeout` and `dataset_path` gotchas.
- **`.env.example`**: the `FINANCEAI_*` keys, including
  `FINANCEAI_DATASET_URL`, `FINANCEAI_DATASET_SHA256`,
  `FINANCEAI_DATASET_TOKEN`, `FINANCEAI_DATABASE_URL` and
  `FINANCEAI_CORS_ORIGIN_REGEX`, without values. Add `gui/.env.local.example`
  with `NEXT_PUBLIC_API_URL=` (empty for local dev).
- **README.md "Create the frozen dataset"**: add "Sharing the frozen dataset".
  Cover publishing a `dataset-vN` release asset, recording its SHA-256, the
  rule never to replace an asset, and how teammates and servers fetch it with
  `python -m api.bootstrap`.
- **README.md "Deployment"**: Render + Vercel + Neon setup, the env var
  table, and the release checklist from §7.7.

---

## 10. Out of scope

Do **not** build these, even though a tracker-style UI suggests them:

- Real holdings, brokerage linking, trade execution, order history or live P&L.
- Live prices, whole-share counts, FX conversion or INR-denominated returns.
- Forecasts, ML return prediction, Monte Carlo or "projected value" charts.
- Free-form LLM chat or any model-written text reaching the user.
- Changing any financial assumption (tickers, dates, risk-free rate,
  constraints, covariance estimator). These need explicit approval under AGENTS.md.
- Calling Yahoo Finance from the API or GUI.

---

## 11. Open decisions

Defaults are chosen so work can start. Change them by editing this table and
the relevant constants.

| Decision | Default in this plan | Alternative | Why the default |
| --- | --- | --- | --- |
| Nav label for the screenshot's "Transactions" | **Saved plans** | Keep "Transactions" | The product makes no transactions; a "Transactions" label would misrepresent saved targets as trades |
| Nav label for "AI Advisor" | **AI Explainer** | Keep "AI Advisor" | The disclaimer says "not personalized investment advice"; "Advisor" contradicts it |
| Portfolio value KPI | In-sample buy-and-hold value of the planned amount, with a hindsight footnote | Show only the planned amount | Keeps the screenshot's headline card with a real, reproducible number |
| Performance method | Buy-and-hold, no rebalancing | Daily or monthly rebalancing to target weights | Simplest to explain; matches "₹X invested then" |
| Sector grouping | GOOGL and META under Technology for display | Strict GICS sectors only | Matches the screenshot's 4-slice donut; strict GICS stays in `sector` |
| Default currency / amount | INR, ₹10,00,000 | USD, $10,000 | Matches the screenshot; a label only |
| Persistence | SQLite locally; **Neon Postgres** on Render; single user, no auth | Render Postgres; add auth | Render's disk is ephemeral; Neon's free tier doesn't expire |
| Hosting | **Render** (Docker) for the API, **Vercel** for the GUI | Railway, Cloud Run, HF Spaces | Decided |
| Dataset on servers | **Downloaded at startup** from the `dataset-vN` GitHub Release asset, verified by SHA-256 | Commit the Parquet; bake it into the image; re-download from Yahoo (rejected: drifts, implicit, unreliable) | Decided; keeps Git and the image data-free while staying frozen and reproducible |
| Render plan | Starter (always on) for demos | Free (sleeps after ~15 min idle) | The free tier works, but a cold start includes the download and pipeline run |
| Browser → API path | Direct calls to Render with a CORS allow-list | Vercel rewrite proxy | Avoids proxy timeouts on ~60 s explain calls |
| Charting library | Recharts | ECharts, visx | Covers all charts here with the least code |
