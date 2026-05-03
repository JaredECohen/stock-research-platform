# MarketMosaic — Your AI Investment Committee

> A multi-agent equity research and portfolio management platform for self-directed investors.
> Eight specialist agents (sector with integrated bull/bear, earnings, filings, valuation, comps
> with self-historical comparison, macro, risk, technical) collaborate under a Portfolio Manager
> orchestrator to produce structured stock memos with versioned lineage, persistent DCF models
> that roll forward each quarter, realized-outcome tracking against forward returns, and a
> long-term agent memory that learns across runs. Scenario-based model portfolios, discretionary
> research-notes injection, and full backtest support via `as_of=YYYY-MM-DD` round out the
> research-platform surface.

> **Disclaimer.** MarketMosaic is for investment research and education only. It does not provide
> personalized financial, investment, legal, or tax advice. Model portfolios and stock analyses are
> illustrative and scenario-based. Users should conduct their own research or consult a qualified
> advisor before making investment decisions.

---

## Table of Contents

1. [Product overview](#1-product-overview)
2. [Target user & problem](#2-target-user--problem)
3. [Business model](#3-business-model)
4. [Unit economics](#4-unit-economics)
5. [Technical architecture](#5-technical-architecture)
6. [Agent architecture](#6-agent-architecture)
7. [Data architecture](#7-data-architecture)
8. [API providers and keys](#8-api-providers-and-keys)
9. [How demo mode works](#9-how-demo-mode-works)
10. [How live mode works](#10-how-live-mode-works)
11. [Local setup](#11-local-setup)
12. [Deployment](#12-deployment)
13. [Class concepts → file references](#13-class-concepts--file-references)
14. [Repo layout](#14-repo-layout)
15. [Demo script](#15-demo-script)
16. [Limitations](#16-limitations)
17. [Future roadmap](#17-future-roadmap)
18. [Evaluation strategy](#18-evaluation-strategy)
19. [Multi-agent + memory-tier architecture](#19-multi-agent--memory-tier-architecture)
20. [Pending items / deferred work](#20-pending-items--deferred-work)
21. [Versioning, lineage & longitudinal features](#21-versioning-lineage--longitudinal-features)
22. [Operational tooling](#22-operational-tooling)
23. [Cost observability](#23-cost-observability)

---

## 1. Product overview

MarketMosaic is an AI-native investment research terminal. Eight specialized agents — orchestrated
by a Portfolio Manager — analyze stocks, rank ideas, build scenario-based model portfolios, and
synthesize macro views, all delivered through a clean finance-terminal-style UI.

The application **runs end-to-end with zero API keys** thanks to a programmatic demo dataset
covering ~28 large-cap names across 7 sectors. When real provider keys (FMP, Alpha Vantage, FRED,
Polygon, Tiingo, SEC EDGAR, OpenAI) are configured, the same code paths transparently swap to live
data and LLM-backed agents.

### Core workflows

| # | Workflow              | Page                | Backed by                                     |
|---|-----------------------|---------------------|-----------------------------------------------|
| 1 | Ask the PM (chat)     | `/chat`             | `Orchestrator` + agent graph                  |
| 2 | Single-stock memo     | `/research`         | `agents/graph.py::run_stock_memo`             |
| 3 | Agentic screener      | `/screener`         | `services/screener_service.py`                |
| 4 | Portfolio builder     | `/portfolio`        | `finance/portfolio_construction.py`           |
| 5 | Editable DCF          | `/dcf`              | `finance/dcf.py`                              |
| 6 | Comps analysis        | `/comps`            | `finance/comps.py`                            |
| 7 | Macro scenarios       | `/macro`            | `agents/macro_agent.py`                       |
| 8 | Mode / providers      | `/settings`         | `api/routes_health.py`                        |

---

## 2. Target user & problem

**Target user.** Self-directed investors with meaningful equity portfolios who actively research
single stocks but cannot afford institutional research tools (Bloomberg, FactSet, Visible Alpha)
that cost $20-30k+/year. Modern AI lets us ship the *workflow* of a research team — sector
analyst, earnings analyst, valuation analyst, risk committee — at a fraction of the cost.

**Problem.**

- Free tools (Yahoo, FinViz) are shallow and don't synthesize.
- Premium tools (BBG, FactSet) are out of reach.
- LLM chat alone hallucinates and lacks structured outputs / retrieval / explicit financial models.
- Robo-advisors solve allocation but not idea generation or research narrative.

MarketMosaic sits between robo-advisors and institutional terminals: explicit financial models +
multi-agent narrative + retrieval-grounded analysis with disciplined disclaimers.

---

## 3. Business model

| Tier      | Price       | Limits                                                           |
|-----------|-------------|------------------------------------------------------------------|
| Free      | $0          | 5 analyses/day, demo dataset, basic memos                        |
| Pro       | $29/month   | Unlimited analyses, live data, full DCF lab, screener export     |
| Premium   | $99/month   | Unlimited portfolios, advanced macro, transcripts, filings RAG   |
| Advisor   | $299/month  | Multi-account, white-label, audit trail, compliance disclaimers  |

Plus optional **API access** for fintechs / quant prosumers and **datafeed reseller** revenue.

---

## 4. Unit economics

**LLM/API cost per active retail user-month** depending on usage:

- Free tier: \$0–\$0.50 (cached + cheap-model only)
- Pro: \$1–\$3 (cheap model for extraction, strong model for PM/critic synthesis)
- Premium: \$3–\$8 (more memos, transcript RAG)

**Cost controls implemented in this codebase:**

- **Cached company data** (`services/data_service.py`) — provider results memoized.
- **Pre-computed screener scores** (`seed_demo_data.py::seed_screener_scores`) — never recompute on read.
- **Cheap model** (`OPENAI_CHEAP_MODEL`) for extraction/classification, **strong model** (`OPENAI_STRONG_MODEL`) only for PM synthesis + critic.
- **Demo / fallback mode** — every provider call cleanly degrades to local fixtures.
- **Document retrieval** (`services/retrieval_service.py`) instead of full-document prompting.
- **Pydantic structured outputs** (`schemas.py`) — agent calls return JSON, no streaming text wastage.

**Estimated gross margin:** 70–95% before fixed infrastructure / data licensing.

---

## 5. Technical architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Frontend (React)                           │
│  Vite · TypeScript · Tailwind · Recharts · Lucide                  │
│  Pages: Dashboard · Chat · Research · DCFLab · Comps · Screener ·   │
│         PortfolioBuilder · Macro · Settings                         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ /api/* + /health
┌───────────────────────────▼─────────────────────────────────────────┐
│                       FastAPI Backend                               │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Routes    │→ │  Services    │→ │   Providers  │→ │ Live APIs │ │
│  │  /api/*    │  │ data/finance │  │ FMP/AV/FRED…│  └───────────┘ │
│  └────┬───────┘  └──────────────┘  └──────┬───────┘                │
│       │                                    └─ fallback → DemoProvider│
│       ▼                                                             │
│  ┌──────────────────────────────────────┐                          │
│  │  Agents (Orchestrator + specialists) │  Pydantic structured     │
│  │  Sector · Earnings · Filing · Valua- │  outputs + tool use      │
│  │  tion · Comps · Macro · Risk · Critic│                          │
│  └──────────────────────────────────────┘                          │
│                                                                     │
│  Persistence: SQLAlchemy → SQLite (default) / Postgres             │
└─────────────────────────────────────────────────────────────────────┘
```

**Backend stack.** FastAPI · Pydantic v2 · SQLAlchemy 2 · SQLite (default) / Postgres ·
httpx · OpenAI SDK · uvicorn.

**Frontend stack.** React 18 · TypeScript · Vite · Tailwind · Recharts · React Router.

**Agent framework.** Hand-rolled LangGraph-style graph in [`backend/app/agents/graph.py`](backend/app/agents/graph.py)
with a planner → fan-out specialists → critic → PM synthesis topology. The graph emits
`AgentTrace` objects so the UI can show progress per agent. The architecture maps 1:1 to LangGraph
or the OpenAI Agents SDK — the dependencies are intentionally minimal so the demo container is
lean.

---

## 6. Agent architecture

```
                ┌─────────────────────────────┐
                │      PM Orchestrator        │
                │  (intent classify + route)  │
                └─────────────┬───────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
  ┌─────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐
  │ Sector  │  │ Earnings   │  │  Filing  │  │Valuation │  │ Comps  │
  │ + bull/ │  │ Call agent │  │  agent   │  │  + DCF   │  │ + own- │
  │ bear    │  │            │  │          │  │ (versn'd)│  │ history│
  └───┬─────┘  └─────┬──────┘  └────┬─────┘  └────┬─────┘  └────┬───┘
      │              │              │             │             │
      └──────────────┴──────────────┴─┬───────────┴─────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │ Macro · Risk · Tech-   │
                          │ nical (positioning)    │
                          └────────────┬───────────┘
                                       ▼
                          ┌────────────────────┐
                          │  Risk Committee /  │   News-impact agent
                          │  Critic (Anthropic │   (Anthropic Haiku
                          │  Opus 4.7)         │   4.5) for patches
                          └────────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │   PM Synthesis     │
                          │   (final memo)     │
                          └────────────────────┘
```

**Files.**

| Agent                       | File                                                                                                                              |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| PM Orchestrator             | [`backend/app/agents/orchestrator.py`](backend/app/agents/orchestrator.py)                                                        |
| Graph + memo synthesis      | [`backend/app/agents/graph.py`](backend/app/agents/graph.py)                                                                       |
| Sector specialist + bull/bear | [`backend/app/agents/sector_agents.py`](backend/app/agents/sector_agents.py) (configs: [`backend/app/data/sector_configs.json`](backend/app/data/sector_configs.json))  — Wave 3A integrated bull/bear with falsifiable tests, bear-first construction, sector lean as a prior |
| Earnings call analyst       | [`backend/app/agents/earnings_agent.py`](backend/app/agents/earnings_agent.py)                                                    |
| Filing analyst (10-K/Q/8-K) | [`backend/app/agents/filing_agent.py`](backend/app/agents/filing_agent.py) + structured-fact extractor [`agents/fact_extraction.py`](backend/app/agents/fact_extraction.py) |
| Valuation analyst (DCF)     | [`backend/app/agents/valuation_agent.py`](backend/app/agents/valuation_agent.py) + engine [`finance/dcf.py`](backend/app/finance/dcf.py) + persistent versioned model [`services/dcf_store.py`](backend/app/services/dcf_store.py) + LLM updater [`agents/dcf_updater.py`](backend/app/agents/dcf_updater.py) |
| Comps analyst               | [`backend/app/agents/comps_agent.py`](backend/app/agents/comps_agent.py) + engines [`finance/comps.py`](backend/app/finance/comps.py) + [`finance/comps_history.py`](backend/app/finance/comps_history.py) (Wave 3E self-historical lens) |
| Macro analyst               | [`backend/app/agents/macro_agent.py`](backend/app/agents/macro_agent.py)                                                          |
| Risk analyst                | [`backend/app/agents/risk_agent.py`](backend/app/agents/risk_agent.py)                                                            |
| Technical analyst           | [`backend/app/agents/technical_agent.py`](backend/app/agents/technical_agent.py) + math [`finance/technicals.py`](backend/app/finance/technicals.py) — positioning context only; does NOT influence rating |
| Risk committee / Critic     | [`backend/app/agents/critic_agent.py`](backend/app/agents/critic_agent.py) — Anthropic Opus 4.7 (cross-family) |
| News-impact agent           | [`backend/app/agents/news_impact_agent.py`](backend/app/agents/news_impact_agent.py) — Anthropic Haiku 4.5 (cross-family). Decides whether a fresh news alert is material to a memo's thesis; if yes, builds an `incremental_patch` snapshot |
| Reflection / memory writer  | [`backend/app/agents/reflection_agent.py`](backend/app/agents/reflection_agent.py) — fires on delta events (new earnings/filings/news), appends entries to long-term memory, distills cross-company patterns |
| Long-form drill-down        | [`backend/app/agents/long_form.py`](backend/app/agents/long_form.py) — Wave 3C per-tile markdown reports (always-on deterministic build + optional LLM enrichment) |
| Portfolio construction      | [`backend/app/agents/portfolio_agent.py`](backend/app/agents/portfolio_agent.py) + engine [`finance/portfolio_construction.py`](backend/app/finance/portfolio_construction.py) |
| Screener                    | [`backend/app/services/screener_service.py`](backend/app/services/screener_service.py) + [`finance/factor_scores.py`](backend/app/finance/factor_scores.py) |

Every agent has both an LLM-backed implementation (when `OPENAI_API_KEY` is set) and a
deterministic stub that returns the same Pydantic schema. This lets the entire product run
without any LLM credentials and keeps tests reproducible.

---

## 7. Data architecture

**Provider abstraction.** [`backend/app/providers/base.py`](backend/app/providers/base.py) declares
the `BaseProvider` protocol. Live providers and the demo provider all implement the same surface:

```
get_company_profile(ticker)     get_earnings_transcripts(ticker)
get_price_history(ticker, days) get_filings(ticker)
get_financial_statements(ticker)get_news(ticker)
get_ratios(ticker)              get_estimates(ticker)
get_key_metrics(ticker)         get_macro_series(series_id)
get_earnings(ticker)            list_tickers()
```

[`services/data_service.py`](backend/app/services/data_service.py) is the facade that tries
configured live providers (per capability) first, then falls back to the demo provider for any
endpoint that returns `None` or raises.

**Persistence.** SQLAlchemy ORM models in [`backend/app/models.py`](backend/app/models.py):

| Table | Purpose |
|---|---|
| `Company` | Master security universe with `universe_tier` (`auto_analysis` / `analyzed_on_demand` / `data_only`) |
| `StockMemo` | Legacy single-row-per-ticker memo (back-compat; readers should prefer `MemoSnapshot`) |
| `MemoSnapshot` | **Versioned** memos with `parent_version` lineage, `revision_log`, `as_of_date` for backtests, and trigger taxonomy (`first_run` / `full_reanalysis` / `incremental_patch` / `force_refresh` / `scheduled`) |
| `MemoOutcome` | Realized forward-return scoring per `(memo_snapshot_id, horizon_days)` — Wave 4A |
| `MemoRunCheckpoint` | `(run_id, step_name)` per-step cache so a retried memo skips already-completed work — Wave 6A |
| `DCFModel` | Versioned DCF assumptions + result + `assumption_changes` (LLM updater audit trail) — Wave 5A |
| `FinancialPeriod` | Long-format statement data; one row per `(ticker, period, statement, line_item)`, unique-keyed for idempotent backfill — Wave 2 |
| `FilingDoc` | SEC filings (raw text + parsed sections); `accession_number` is the unique key — Wave 2 |
| `EarningsTranscript` | Speaker-segmented transcripts; `(ticker, period)` unique — Wave 2 |
| `LLMCallLog` | Append-only audit log of every provider call (Wave 1A) — `run_id` / `agent_name` / `provider` / `model` / `tokens_in` / `tokens_out` / `duration_ms` / `success` / `error` |
| `ResearchSnapshot` | Lineage-aware cache (cold/warm/hot) with `parent_snapshot_ids` cascade |
| `CacheCostLog` | Per-snapshot cost ledger — token savings telemetry |
| `ScreenerScore` | Pre-computed factor scores per ticker / theme |
| `CachedDocument` | Chunked filings/transcripts/news (used by retrieval) |
| `PortfolioRun` | Saved portfolio constructions |

SQLite by default; switch to Postgres by setting `DATABASE_URL=postgresql+psycopg2://...`.

Long-term agent memory lives **outside** the DB as filesystem markdown:
`memory/companies/<TICKER>.md` and `memory/sectors/<sector_slug>.md`.
Atomic writes (temp file + rename); reflection appends only on delta
events; condense-on-cap pulls the oldest entries into a "historical
context" block via an LLM condenser (deterministic fallback).

---

## 8. API providers and keys

All keys are **optional**. Without keys, MarketMosaic runs against the demo dataset.

MarketMosaic supports both **OpenAI** and **Anthropic**. Set either key (or both) and choose
routing with `LLM_PROVIDER`:

- `auto` (default) — use Anthropic if its key is set, else OpenAI.
- `anthropic` — force Anthropic (requires `ANTHROPIC_API_KEY`).
- `openai` — force OpenAI (requires `OPENAI_API_KEY`).

| Variable                       | Purpose                                              | Free tier?  |
|--------------------------------|------------------------------------------------------|-------------|
| `LLM_PROVIDER`                 | `auto` / `openai` / `anthropic`                      | —           |
| `OPENAI_API_KEY`               | LLM-backed agents (PM, critic, sector, earnings...)  | Paid        |
| `OPENAI_STRONG_MODEL`          | Default `gpt-5.5` — used for synthesis + critic       | —           |
| `OPENAI_CHEAP_MODEL`           | Default `gpt-4.1-mini` — used for extraction          | —           |
| `ANTHROPIC_API_KEY`            | LLM-backed agents (PM, critic, sector, earnings...)  | Paid        |
| `ANTHROPIC_STRONG_MODEL`       | Default `claude-opus-4-7` — synthesis + critic       | —           |
| `ANTHROPIC_CHEAP_MODEL`        | Default `claude-haiku-4-5` — extraction              | —           |
| `FMP_API_KEY`                  | Profiles, fundamentals, ratios, prices               | ✓           |
| `ALPHA_VANTAGE_API_KEY`        | Earnings transcripts, news/sentiment                 | ✓ (limited) |
| `FRED_API_KEY`                 | Macro time series                                    | ✓           |
| `POLYGON_API_KEY`              | Prices, news (optional)                              | ✓ (limited) |
| `TIINGO_API_KEY`               | Prices, news (optional)                              | ✓           |
| `FINNHUB_API_KEY`              | Reserved (alt fundamentals)                          | ✓ (limited) |
| `INTRINIO_API_KEY`             | Reserved                                             | Paid        |
| `NASDAQ_DATA_LINK_API_KEY`     | Reserved                                             | Paid        |
| `SEC_USER_AGENT`               | Required by SEC EDGAR (no API key)                   | Free        |

Configuration is split across two files:

- **`config.env`** (in git) — committed defaults: per-agent model assignments, feature flags, runtime tuning. Edit via PR when you're committing a decision the whole team should run.
- **`.env`** (gitignored — copy from `example.env`) — secrets and per-deployment overrides: API keys, `DATABASE_URL`, `SEC_USER_AGENT`, etc.

Load order: `config.env` → `.env` → process env (each later source overrides). So a one-off `OPENAI_PM_MODEL=...` in `.env` overrides the committed default without changing `config.env`. The `Settings` page in the UI shows which providers are configured at runtime.

### Gemini access — direct API or Vertex AI

Two paths to Gemini, pick whichever your environment is set up for. Vertex wins when both are configured:

```bash
# Option A — direct API key (quick setup)
GEMINI_API_KEY=...                   # in .env

# Option B — Vertex AI via Google Cloud (auth via ADC)
LLM_PROVIDER=vertex                  # documentation/intent flag (in .env)
VERTEX_PROJECT_ID=your-project-id    # in .env
VERTEX_LOCATION=us-central1          # in .env (defaults to us-central1)
VERTEX_MODEL=gemini-3.1-pro          # in config.env (overrides per-agent envs)

# Then authenticate ADC (locally):
gcloud auth application-default login
# Or in production, set GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
```

---

## 9. How demo mode works

When `USE_DEMO_DATA=true` (default) or `ENABLE_LIVE_DATA=false`:

1. [`backend/app/data/demo_dataset.py`](backend/app/data/demo_dataset.py) builds a coherent
   multi-year dataset for ~28 large-cap names from a compact `COMPANY_PROFILES` table.
2. The dataset is exported to `backend/app/data/demo_*.json` for inspection.
3. [`DemoProvider`](backend/app/providers/demo_provider.py) wraps the dataset with the same
   `BaseProvider` interface live providers expose.
4. The agent graph runs against the demo provider. Without `OPENAI_API_KEY`, every agent produces
   a deterministic stub finding (still typed via Pydantic).
5. The frontend looks identical to live mode — there is no separate UI codepath.

Demo coverage includes profiles, 4 years of income / balance / cash flow statements, ratios,
earnings calendars + transcripts, 10-K / 10-Q / 8-K stubs, news, analyst estimates, and 13
macro series.

---

## 10. How live mode works

Set `ENABLE_LIVE_DATA=true` and configure provider keys. The data service tries providers in
priority order per capability; if a live call fails (network error, missing key, rate limit),
it transparently falls back to the demo provider for that single call.

When `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is present, agents call the active provider so
outputs are parsed directly into Pydantic schemas. OpenAI uses JSON-mode (`response_format`);
Anthropic is instructed to return strict JSON and the response is best-effort parsed. **Model
routing** in [`backend/app/agents/llm.py`](backend/app/agents/llm.py):

- `route="strong"` → `OPENAI_STRONG_MODEL` / `ANTHROPIC_STRONG_MODEL` (PM synthesis, critic)
- `route="cheap"` → `OPENAI_CHEAP_MODEL` / `ANTHROPIC_CHEAP_MODEL` (sector view, earnings
  extraction, classification)

Provider selection is resolved per call by `settings.active_llm_provider`, which honors
`LLM_PROVIDER` and the presence of each provider's API key.

---

## 11. Local setup

### Prerequisites

- Python 3.11+ (3.12 recommended; 3.13 works without `psycopg2`)
- Node 20+
- (Optional) Postgres 16

### Steps

```bash
# 1. Configure env (everything is optional — runs as-is)
# config.env (committed) holds model assignments + flags; copy example.env
# to .env (gitignored) for your secrets.
cp example.env .env

# 2. Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# In a second terminal: visit http://localhost:8000/docs for the OpenAPI UI.

# 3. Frontend
cd ../frontend
npm install
npm run dev
# Visit http://localhost:5173
```

### Backend tests

```bash
cd backend
python -m pytest -q
```

### End-to-end smoke test

```bash
cd backend
python -m scripts.smoke_test
```

This exercises every demo prompt against an in-process FastAPI test client.

---

## 12. Deployment

### Docker (single image, frontend + backend)

```bash
docker build -t marketmosaic .
docker run --rm -p 8000:8000 --env-file .env marketmosaic
# Visit http://localhost:8000
```

### docker-compose

```bash
docker compose up                 # backend only (sqlite)
docker compose --profile postgres up   # backend + postgres
```

### Cloud platforms

- **Render / Railway / Fly.io** — point at the root `Dockerfile`. Set env vars from `example.env`.
  Bind port `8000`. No persistent volume required for demo mode (SQLite is recreated on start
  from the demo dataset).
- **Vercel / Netlify (frontend only)** — deploy `frontend/` separately, set
  `VITE_BACKEND_URL` to the deployed FastAPI URL.

---

## 13. Class concepts → file references

| Concept                              | Where in this codebase                                                                |
|--------------------------------------|---------------------------------------------------------------------------------------|
| **Agentic planning / orchestration** | [`agents/orchestrator.py`](backend/app/agents/orchestrator.py), [`agents/graph.py`](backend/app/agents/graph.py) |
| **Multi-agent specialization**       | [`agents/sector_agents.py`](backend/app/agents/sector_agents.py), [`agents/earnings_agent.py`](backend/app/agents/earnings_agent.py), [`agents/filing_agent.py`](backend/app/agents/filing_agent.py), [`agents/valuation_agent.py`](backend/app/agents/valuation_agent.py), [`agents/comps_agent.py`](backend/app/agents/comps_agent.py), [`agents/macro_agent.py`](backend/app/agents/macro_agent.py), [`agents/risk_agent.py`](backend/app/agents/risk_agent.py) |
| **Tool use**                         | [`agents/tools.py`](backend/app/agents/tools.py) — agents call typed tools rather than touching providers directly |
| **Retrieval-augmented generation**   | [`services/retrieval_service.py`](backend/app/services/retrieval_service.py) — BM25-style chunked retrieval over filings/transcripts/news |
| **Structured outputs (Pydantic)**    | [`schemas.py`](backend/app/schemas.py) — every agent emits Pydantic-validated JSON     |
| **Critic / reflection loop**         | [`agents/critic_agent.py`](backend/app/agents/critic_agent.py) — runs after draft memo, surfaces challenges + suggested revisions |
| **Human-editable assumptions**       | [`pages/DCFLab.tsx`](frontend/src/pages/DCFLab.tsx) — editable WACC / margins / growth + sensitivity tables |
| **Model routing / cost control**     | [`agents/llm.py`](backend/app/agents/llm.py) — `route="strong"` for synthesis, `route="cheap"` for extraction |
| **Caching / demo fallback**          | [`providers/demo_provider.py`](backend/app/providers/demo_provider.py), [`services/data_service.py`](backend/app/services/data_service.py) — graceful per-capability fallback |
| **Disciplined disclaimers**          | [`agents/prompts.py`](backend/app/agents/prompts.py) — every PM/critic prompt frames output as research/education only |

---

## 14. Repo layout

```
.
├── README.md
├── BUSINESS_ONE_PAGER.md
├── config.env             # committed: model assignments, feature flags
├── example.env            # template for `.env` (secrets, gitignored)
├── Dockerfile
├── docker-compose.yml
├── backend/
│   ├── requirements.txt
│   ├── conftest.py
│   ├── scripts/
│   │   └── smoke_test.py
│   └── app/
│       ├── main.py            FastAPI app factory + startup seed
│       ├── config.py          pydantic-settings
│       ├── database.py        SQLAlchemy engine / session
│       ├── models.py          ORM models
│       ├── schemas.py         Pydantic schemas (API + agent IO)
│       ├── seed_demo_data.py  Idempotent seeder
│       ├── agents/
│       │   ├── orchestrator.py     PM intent + dispatcher
│       │   ├── graph.py            Memo graph: fan-out → critic → synthesis
│       │   ├── prompts.py          System + per-agent prompts
│       │   ├── llm.py              Model routing helper
│       │   ├── tools.py            Tools agents are allowed to call
│       │   ├── sector_agents.py    Sector specialist
│       │   ├── earnings_agent.py
│       │   ├── filing_agent.py
│       │   ├── valuation_agent.py
│       │   ├── comps_agent.py
│       │   ├── macro_agent.py      Scenario templates
│       │   ├── risk_agent.py
│       │   ├── critic_agent.py     Risk Committee
│       │   └── portfolio_agent.py
│       ├── api/
│       │   ├── routes_health.py    /health, /api/providers/status
│       │   ├── routes_stocks.py    /api/stocks*, /api/stocks/{t}/memo
│       │   ├── routes_screener.py  /api/screener
│       │   ├── routes_chat.py      /api/chat
│       │   ├── routes_dcf.py       /api/dcf/{t}*
│       │   ├── routes_comps.py     /api/comps/{t}
│       │   ├── routes_portfolio.py /api/portfolio/build
│       │   ├── routes_macro.py     /api/macro/{series,analyze}
│       │   └── routes_admin.py     /api/seed-demo-data
│       ├── services/
│       │   ├── data_service.py        Provider facade with fallback
│       │   ├── market_data_service.py
│       │   ├── fundamentals_service.py
│       │   ├── filings_service.py
│       │   ├── transcripts_service.py
│       │   ├── macro_service.py
│       │   ├── news_service.py
│       │   ├── valuation_service.py   Bridges fundamentals + DCF/comps
│       │   ├── portfolio_service.py
│       │   ├── screener_service.py    Theme-biased PM scores
│       │   └── retrieval_service.py   BM25 retrieval
│       ├── providers/
│       │   ├── base.py
│       │   ├── demo_provider.py
│       │   ├── fmp_provider.py
│       │   ├── alpha_vantage_provider.py
│       │   ├── sec_edgar_provider.py
│       │   ├── fred_provider.py
│       │   ├── polygon_provider.py
│       │   └── tiingo_provider.py
│       ├── finance/
│       │   ├── dcf.py                 Full engine: scenarios + sensitivity
│       │   ├── comps.py
│       │   ├── ratios.py
│       │   ├── risk.py
│       │   ├── factor_scores.py
│       │   └── portfolio_construction.py
│       ├── data/
│       │   ├── demo_dataset.py        Programmatic demo dataset
│       │   ├── sector_configs.json    Sector frameworks
│       │   ├── peer_groups.json       Curated comps peer sets
│       │   └── demo_*.json            Exported on first run
│       └── tests/
│           ├── test_dcf.py
│           ├── test_comps.py
│           ├── test_portfolio.py
│           └── test_routes.py
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── index.css
        ├── api/client.ts        Typed fetch client
        ├── types/index.ts       Mirrors Pydantic schemas
        ├── lib/format.ts
        ├── components/
        │   ├── Layout.tsx
        │   ├── AgentTrace.tsx
        │   ├── MemoCard.tsx
        │   └── Markdown.tsx
        └── pages/
            ├── Dashboard.tsx
            ├── Chat.tsx
            ├── Research.tsx
            ├── DCFLab.tsx
            ├── Comps.tsx
            ├── Screener.tsx
            ├── PortfolioBuilder.tsx
            ├── Macro.tsx
            └── Settings.tsx
```

---

## 15. Demo script

Each prompt below routes correctly without any API keys.

| #  | Prompt                                                                                                | Expected behavior                                                                                                                       |
|----|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Analyze NVDA as a long-term investment.                                                               | `single_stock_analysis` — full memo with sector / earnings / filing / valuation / comps / macro / risk findings + critic review.        |
| 2  | Compare MSFT and GOOGL from a portfolio manager's perspective.                                        | `stock_comparison` — two memos rendered side-by-side with PM synthesis paragraph.                                                      |
| 3  | Find 5 high-quality stocks that could benefit from falling rates.                                     | `thematic_screen` — falling-rates theme bias applied; top names ranked.                                                                |
| 4  | Build a 10-stock portfolio for a soft landing with falling rates and continued AI infrastructure spending. | `portfolio_construction` — diversified weights with sector cap, risk notes, watch items.                                          |
| 5  | What sectors benefit if inflation stays sticky?                                                       | `macro_question` — sticky-inflation scenario template + favored / pressured sectors.                                                   |
| 6  | Run a DCF for MSFT using base-case assumptions.                                                       | `dcf_analysis` — base / bull / bear scenarios with implied prices + upside.                                                            |
| 7  | Show me reasonable valuation growth stocks.                                                           | `thematic_screen` — valuation-tilted theme bias.                                                                                       |

Run all 7 with one command:

```bash
cd backend && python -m scripts.smoke_test
```

---

## 16. Limitations

- **Demo dataset is illustrative.** Numbers are shaped from rough public-company profiles but are
  not real-time. Live providers are required for production-grade research.
- **Earnings transcripts and 10-K/Q/8-K filings** in demo mode are stubs. SEC EDGAR + Alpha
  Vantage transcripts populate them in live mode.
- **Embeddings RAG** is gated behind `ENABLE_VECTOR_SEARCH`. Default retrieval is a BM25-style
  scorer that is fast and dependency-light.
- **Estimates** require a paid provider (FMP / Visible Alpha / Refinitiv). The schema is wired up
  but live data may show as unavailable.
- **No backtest engine** in v0 — the portfolio builder is forward-looking only.
- **No multi-asset coverage** — equities / ETFs only.

---

## 17. Future roadmap

- **Backtest engine** (per-portfolio historical replay, drawdown / Sharpe / regime).
- **Vector RAG** with pgvector + OpenAI embeddings for filings + transcripts.
- **Earnings preview** mode that diffs guidance to expectations and front-runs reaction.
- **News + filings monitor** that pings the user when a memo's thesis-breaker risk fires.
- **Custom universes** (small/mid cap, international, ETFs).
- **Multi-account / advisor** features (book-of-business view, audit trail, compliance pack).
- **Formal evaluation harness** — see §18 below.

---

## 18. Evaluation strategy

A multi-agent research product needs evaluation at three levels: **(a) financial outcomes** of
the investment recommendations, **(b) per-agent quality** of each specialist's structured output,
and **(c) end-to-end memo quality** as judged against expert benchmarks. Today MarketMosaic has
smoke tests and structured-output schemas; the items below describe how we'd build out a formal
eval harness.

### 18.1 Financial outcomes (the only metric that ultimately matters)

Build a **point-in-time backtest** harness that generates a memo or screener output as of date `T`
using *only data available at `T`*, then measures forward returns over windows (1m, 3m, 6m, 12m).

- **Per-rating IC.** For every memo, record the `rating_label` and `confidence_score`, then
  compute the **information coefficient** between conviction and forward returns across the
  universe. Target: positive Spearman correlation across periods, ideally >0.05 on rolling
  12-month windows.
- **Long–short paper portfolios.** Bin memos by rating (`Bullish` long, `Bearish` short),
  size positions equal-weight or by confidence, rebalance monthly, and report **CAGR, vol,
  Sharpe, max drawdown, hit rate, and alpha vs. SPY/QQQ**.
- **Theme screen efficacy.** For each theme (`falling_rates`, `ai_infrastructure`, …), build the
  top-decile theme portfolio at `T` and compare its forward return to a sector-neutral benchmark.
  This tells us whether the theme bias adds signal beyond the raw factor scores.
- **Portfolio-builder paper-trading.** Persist every `ModelPortfolio` and replay it forward.
  Compare against (i) equal-weight SPY, (ii) sector-matched ETF blend, (iii) Markowitz
  mean-variance benchmark. Report excess return, tracking error, drawdown.
- **Walk-forward validation.** Re-train any tunable thresholds (factor weights, sector tilts)
  on a rolling training window, evaluate out-of-sample, and report **stability of factor weights**
  to detect regime overfitting.
- **Survivorship-bias audit.** The demo universe is curated; live evals must include delisted
  names. Use a survivorship-free price universe (e.g., CRSP) before publishing performance.
- **Critical caveats.** Rolling Sharpe is noisy with <5y of data; report **bootstrapped confidence
  intervals**, not point estimates. Keep a **regime decomposition** (rates up / rates down /
  recession / expansion) so headline metrics aren't dominated by a single regime.

### 18.2 Per-agent evaluation

Each specialist agent emits a Pydantic schema, which makes per-agent eval tractable.

| Agent | Eval signal | Method |
|---|---|---|
| **Sector Analyst** | Does the sector framing match the actual sector framework? | Curated rubric per sector (drivers / KPIs / valuation lens) → LLM-judge scores 1–5 against rubric. |
| **Earnings Analyst** | Tone classification + key-takeaway recall | Hand-label N=200 transcripts (constructive / measured / cautious + 3 ground-truth takeaways each); measure F1 vs. labels. |
| **Filing Analyst** | Risk factor extraction recall | Compare extracted risks to the actual 10-K risk-factor headings (string-match + embedding similarity ≥0.8). |
| **Valuation Analyst** | DCF sanity checks | Property-based tests on `finance/dcf.py`: monotonicity (↑WACC → ↓price), bull > base > bear, terminal share of EV ≤80%. |
| **Comps Analyst** | Peer-set quality | Curated golden peer set per ticker; measure precision/recall of selected peers + accuracy of premium/discount sign. |
| **Macro Analyst** | Sector-impact directionality | Hand-label expected-direction-of-impact per (scenario × sector) cell; measure agreement %. |
| **Risk Analyst** | Risk recall + severity calibration | Compare to a curated risk taxonomy per ticker; reliability diagram for severity calibration. |
| **Risk Committee / Critic** | Did the critic catch the planted issue? | **Synthetic-fault injection**: add unsupported claims / one-sided framing / missing risks to a draft memo and check the critic flags them. Target: ≥80% catch rate. |
| **PM Orchestrator** | Intent classification accuracy | Held-out prompt set with labeled intents (extends `scripts/smoke_test.py`); report confusion matrix. |
| **Portfolio Construction** | Constraint satisfaction + diversification | Property tests: max position respected, sector cap respected, holdings ≥ requested floor, HHI under threshold. |

### 18.3 End-to-end memo quality (LLM-as-judge + human eval)

LLM-judge is fast and cheap but noisy; human review is the ground truth. Use both:

- **LLM-as-judge with rubric.** A separate strong model (different family from the one
  generating the memo, to mitigate same-family bias) scores each memo on a 5-point rubric:
  *factual grounding, balance (bull vs. bear weight), risk completeness, valuation discipline,
  compliance framing*. Sample 100 memos / week; alert on rubric-score regressions.
- **Pairwise preference (LMSYS-style).** Generate two memo variants (e.g., critic on vs. critic
  off; OpenAI vs. Anthropic; cheap-only vs. strong-synth), present both blinded to a panel of
  experienced investors, capture pairwise wins. Report **Bradley–Terry** ratings.
- **Hallucination tests.** Inject a planted fact into the retrieval corpus and measure whether
  the memo cites only sources that actually contain the claim.
- **Adversarial prompt suite.** Curated prompts that try to elicit advice-style language
  ("should I buy?", "tell me what to do"). Verify outputs stay framed as research/education and
  the disclaimer survives. Target: 100% pass rate.
- **Citation faithfulness.** For every claim with a source pointer, check that the cited chunk
  actually supports the claim (LLM-judge with the chunk text + claim, scored 0/1).
- **User-feedback loop.** Thumbs up/down on memos in production, with optional structured tags
  (wrong rating / risks missing / valuation off / unclear). Aggregate weekly and feed into
  prompt-tuning.

### 18.4 Cost & latency evaluation

- **Per-memo token budget**: track tokens by agent and route (cheap vs. strong) and alert on
  outliers (>2σ above the per-agent baseline).
- **End-to-end latency P50/P95** per intent type; target P95 <30s for `single_stock_analysis`.
- **Cache hit-rate** on `data_service` (provider results) and on retrieval chunks. Below 60%
  signals a caching bug.
- **Provider-routing cost A/B.** Run identical eval suites under `LLM_PROVIDER=openai` vs.
  `anthropic` and compare quality/cost frontier (rubric-score per dollar).

### 18.5 Regression gates in CI

- All property-based DCF / portfolio / comps tests must pass on every PR.
- Smoke-test prompts ([smoke_test.py](backend/scripts/smoke_test.py)) must route to the correct
  intent and return a populated payload.
- LLM-judge rubric scores on a frozen 50-memo regression set must stay within `±0.2` of the
  baseline; a drop triggers a manual review block.
- Hallucination + advice-language tests are **hard gates** (any failure blocks the PR).

### 18.6 Where this would live in the codebase

- `backend/app/eval/` — harness modules (`backtest.py`, `agent_eval.py`, `llm_judge.py`,
  `hallucination.py`, `regression_set.py`).
- `backend/app/eval/datasets/` — golden labels (peer sets, risk taxonomies, intent labels,
  scenario-sector directionality grids).
- `backend/scripts/run_eval.py` — CLI runner; emits JSON reports to `reports/<date>/`.
- CI workflow runs property tests + smoke tests on every PR; full LLM-judge + backtest sweep
  runs nightly and posts to a dashboard.

---

## 19. Multi-agent + memory-tier architecture

This section documents the hub-and-spokes multi-agent runtime, the persistent
research-snapshot cache, the cross-provider tool wrappers, and the always-on
monitoring loops layered on top of the legacy graph.

### 19.1 Topology

```
                       ┌─────────────────────────────┐
                       │         PM (hub)            │   GPT-5.5 Pro
                       │  intent → coordinate → memo │
                       └────────┬───────────┬────────┘
                                │           │
              ┌─────────────────┘           └─────────────────┐
              ▼                                                ▼
     ┌─────────────────┐                                ┌──────────────┐
     │ Sector spokes   │  Technology · Financials ·     │  Critic       │   Anthropic
     │ (GPT-5.4 each)  │  Consumer · Healthcare ·       │  (Opus 4.7)   │   cross-family
     └────┬────────────┘  Energy · Industrials ·        └──────────────┘
          │               Utilities                              ▲
          ▼                                                       │
     ┌─────────────────┐         ┌─────────────────────┐          │
     │ Tool sub-agents │ ◀────── │ Cache-backed tools  │ ─────────┘
     │ filing/earnings │         │ (cold/warm/hot)     │
     │ valuation/comps │         └─────────────────────┘
     │ risk            │
     └─────────────────┘

   ┌───────────────────────────────── monitoring (Phase 5) ────────────────────┐
   │ EDGAR poller (30m) · News loop (1h/ticker) · Social loop (daily) ·       │
   │ Macro loop (1h, broadcast on regime change)                              │
   └─────────────────────────────────────────────────────────────────────────┘
```

The legacy graph in [`agents/graph.py`](backend/app/agents/graph.py) remains
the default execution path. Setting `USE_AGENTS_SDK=true` routes
`single_stock_analysis` through the SDK runtime in
[`agents/sdk_runtime.py`](backend/app/agents/sdk_runtime.py) instead.

### 19.2 Memory tiers

Three persistent tiers keyed by `subject` + `kind` in
[`cache/snapshots.py`](backend/app/cache/snapshots.py):

| Tier | Kind(s) | TTL | Invalidated by |
|---|---|---|---|
| **COLD** (per ticker, quarterly) | `company_cold` | 90 days | New 10-K / 10-Q / 8-K via the EDGAR poller |
| **WARM** (per cohort + per ticker) | `sector_warm`, `company_warm:dcf`, `company_warm:comps` | 7 days | Lineage propagation: any cohort member's `company_cold` change marks the warm snapshot stale |
| **HOT** (per ticker, hours) | `news_hot`, `social_hot`, `macro_broadcast` | 2–24 h | Push from monitoring loops; news classified `material`/`breaking` invalidates the relevant `sector_warm` |

Each `cache_put` records `cost_tokens` to a `CacheCostLog` table so cache
savings are measurable. Smoke run on demo data: cold = 4980 tokens, warm = 0
tokens (every snapshot served from cache).

### 19.3 Cross-provider routing

[`agents/llm.py`](backend/app/agents/llm.py) now exposes three provider branches
(OpenAI, Anthropic, Gemini) with per-call `provider_override`. The critic
intentionally crosses families (Opus 4.7) regardless of `LLM_PROVIDER`. A
3-strike circuit breaker per provider short-circuits to typed empty/None
responses and logs a `provider_failure` row to `CacheCostLog`.

| Role | Default model |
|---|---|
| PM Orchestrator | `OPENAI_PM_MODEL=gpt-5.5-pro` |
| Sector + tool agents | `OPENAI_SECTOR_MODEL=gpt-5.4`, `OPENAI_TOOL_MODEL=gpt-5.4` |
| Critic | `ANTHROPIC_CRITIC_MODEL=claude-opus-4-7` |
| News + Social | `GEMINI_NEWS_MODEL=gemini-2.5-flash`, `GEMINI_SOCIAL_MODEL=gemini-2.5-flash` |
| Long-doc analysts | `GEMINI_LONGDOC_MODEL=gemini-3.1-pro` |

### 19.4 Always-on monitoring

Behind the `ENABLE_MONITORING` feature flag, four APScheduler-driven loops
keep the hot/cold tiers fresh:

- [`monitoring/edgar_poller.py`](backend/app/monitoring/edgar_poller.py) — every
  30 min; on a new accession, `cache.invalidate(ticker, kind="company_cold")`.
- [`monitoring/news_loop.py`](backend/app/monitoring/news_loop.py) — 1 h /
  ticker; pushes `NewsAlert` records into the hot cache; pings the relevant
  sector when severity is `material` or `breaking`.
- [`monitoring/social_loop.py`](backend/app/monitoring/social_loop.py) — daily.
- [`monitoring/macro_loop.py`](backend/app/monitoring/macro_loop.py) — hourly
  FRED snapshot; broadcasts a `MacroBroadcast` on regime change.

Status is exposed at `GET /api/admin/monitoring/status`.

### 19.5 Sector cross-talk

Sector agents subscribe to the latest `MacroBroadcast` and pending
`NewsAlerts` at the start of every run, and emit a `cross_sector_relevance`
list of tickers in *other* sectors that matter for the thesis. The PM
aggregates these into `final_verdict` and `scores`. Peer-sector handoffs are
exposed as a `query_peer_sector` function tool with a depth-2 cap to prevent
runaway recursion.

### 19.6 Feature flags

| Flag | Default | Effect |
|---|---|---|
| `USE_AGENTS_SDK` | `false` | Route `single_stock_analysis` through the SDK runtime |
| `ENABLE_MONITORING` | `false` | Spin up the APScheduler-backed monitoring loops |

Both default off so the existing test suite + smoke test remain deterministic.

---

## 20. Pending items / deferred work

All items previously listed here have shipped. The full delivery is
documented in [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) (Waves 1–8).

| Originally deferred | Status | Shipped in |
|---|---|---|
| OpenAI Agents SDK package upgrade | DONE | PR #6 |
| Live-mode token cost measurement | DONE — `LLMCallLog` + `cost_per_run` aggregation + USD estimates via `MODEL_PRICES_PER_MTOK` | Wave 1A + 8D |
| EDGAR / Gemini live integration tests | DONE — `pytest.mark.live` + `.github/workflows/nightly-live.yml` | Wave 4B + 8E |
| Frontend wiring of cross-sector / macro / news fields | DONE | PR #2 |
| Cohort-similarity invalidation tightening | DONE — KPI-only fingerprint | Wave 6B |
| News allow-list governance to JSON | DONE — `app/data/news_domains.json` | Wave 6C |
| SDK-runtime LLM wiring | DONE | PR #6 |
| Schema-version migration path | DONE — read-time upgrader registry | Wave 6D |

Out of scope for now: vector RAG with pgvector, full backtest engine
(replay portfolios, Sharpe / drawdown), multi-account / advisor
features, earnings preview mode, mobile UI.

---

## 21. Versioning, lineage & longitudinal features

The platform records what it thought, when, and why — not just the
latest answer. Three tables form the lineage backbone:

### 21.1 Memo lineage — `MemoSnapshot`

Every `run_stock_memo` call writes an immutable
[`MemoSnapshot`](backend/app/services/memo_store.py) row tagged with a
`trigger`:

| Trigger | Source | Critic runs? |
|---|---|---|
| `first_run` | First analysis of a ticker | Yes |
| `full_reanalysis` | EDGAR poller saw a new 10-K/Q/8-K → orchestrator re-fires | Yes |
| `incremental_patch` | News-impact agent verdict on a material/breaking alert | **No** (locked decision) |
| `force_refresh` | Explicit user-driven refresh | Yes |
| `scheduled` | Background cadence | Yes |

`parent_version` chains a patch off its predecessor so reviewers can
trace exactly what changed. `revision_log` carries the full audit trail
(fields patched, LLM rationales, critic_skipped flag, source alert)
on every patch. Backtest snapshots (`as_of_date` set) live in the same
table but are excluded from `latest_memo` by default so a backtest never
shadows a live recommendation.

### 21.2 DCF lineage — `DCFModel`

Each ticker's DCF is a versioned object that **rolls forward** every
quarter rather than rebuilding from scratch. The
[`agents/dcf_updater.py`](backend/app/agents/dcf_updater.py) flow:

1. Drop year-1 forecast (now an actual), shift the explicit forecast
   left, repeat the tail.
2. LLM reads prior assumptions vs actuals (revenue, op margin, capex %,
   FCF) from the Wave 2 history tables; proposes adjustments.
3. **Per-cycle delta capped at ±20%** of the prior assumption value
   (prevents a hallucinating LLM from moving WACC from 8.5% to 25%).
4. Each changed field requires a one-sentence rationale; fields without
   one are silently dropped (analyst discipline).
5. New version persisted as v(N+1) referencing v(N).

Trigger taxonomy: `initial`, `memo_rebuild`, `earnings_update`,
`force_refresh`. `assumption_changes` carries the per-version diff with
LLM rationale per change. Surfaced in the UI via
[`DCFVersionHistory`](frontend/src/components/DCFVersionHistory.tsx) on
the Research page.

### 21.3 Outcome tracking — `MemoOutcome`

[`services/outcome_service.py`](backend/app/services/outcome_service.py)
runs daily under `ENABLE_MONITORING`. For every `MemoSnapshot` whose
forward window has come of age, it computes:

- Forward return at 30 / 90 / 180 / 365 days
- SPY-relative alpha
- `thesis_held` — Bullish + positive return → True; Bearish + negative
  → True; opposite signs → False; Neutral → None (no directional bet)

Long horizons (90d / 365d) write a reflection entry into the company's
long-term memory file so the next sector pass can read its own track
record. Short horizons (30d / 180d) stay numeric only.

Track-record dashboard at `/track-record` exposes hit rate + avg alpha,
filterable by ticker / sector / horizon.

### 21.4 Long-term agent memory

Filesystem markdown notebooks at `memory/companies/<TICKER>.md` and
`memory/sectors/<sector_slug>.md`. Each holds a frontmatter block + a
condensed "historical context" + recent entries. Reflection writes
trigger only on **delta events** (new earnings, new filing, material
news) so the file grows with real signal, not every memo run.

Wave 3D added structured-fact extraction: when a filing/transcript
trigger fires, [`agents/fact_extraction.py`](backend/app/agents/fact_extraction.py)
pulls a small typed schema (segment performance, guidance changes,
capex commentary, M&A, leadership changes) from the source text — first
via deterministic regex (always runs, no LLM bill), then LLM-enriched
when keys are present. Persisted alongside the entry as a
fenced ` ```structured-facts ` JSON block; round-trips via the parser.

When entry count crosses `MEMORY_MAX_ENTRIES` (default 50), the oldest
`MEMORY_CONDENSE_BATCH` entries (default 10) fold into the historical-
context block via an LLM condenser (deterministic fallback).

Sector files also carry `cross_company_patterns`: transferable lessons
learned on one company that apply to peers. The next time the sector
agent runs on a peer, those patterns are surfaced in its prompt.

### 21.5 Backtest support — `as_of_date`

Every memo can be reproduced for any historical date via
`?as_of=YYYY-MM-DD`. Implementation:

- `as_of_context` ContextVar (Wave 1C) makes the cutoff visible to
  cache, providers, and memory without threading it through every signature.
- Cache keys auto-namespace `:asof:<YYYY-MM-DD>` so live and backtest
  payloads never collide.
- `data_service` clips every list-shaped historical payload (filings,
  transcripts, financial statements, prices, news) to the cutoff (Wave
  8B). Ratios are recomputed from clipped statements rather than
  serving today's snapshot.
- `MemoSnapshot.as_of_date` distinguishes backtest snapshots; default
  `latest_memo` lookup excludes them so a live recommendation isn't
  shadowed.
- Memory writes skip on backtest runs (a diagnostic shouldn't pollute
  the agent's notebook).

### 21.6 Research notes — discretionary context injection

User-curated investment notes live under `research_notes/` (books,
interviews, articles, frameworks, personal). Each declares routing
via YAML frontmatter:

```yaml
applies_to_agents: [sector, valuation, comps]
applies_to_sectors: ["*"]              # or named sectors
applies_to_sub_industries: []          # empty = no narrowing
applies_to_tickers: []                 # empty = no narrowing
weight: 0.8                            # 0-1 priority
expires: null                          # ISO date or null
status: active                         # active|archived|superseded
```

Two-tier injection (Wave 7A + 7B): summaries always inject (~30 tokens
each, capped at 6 per agent run); top-K full bodies inject via BM25
over the agent's working context, hard-capped at 4KB combined. Wave 7C
extends routing to all six LLM-consuming specialists (sector, valuation,
earnings, filing, comps, risk). The CLI indexer
([`scripts/index_research_notes.py`](backend/scripts/index_research_notes.py))
normalizes frontmatter, optionally regenerates summaries via LLM, and
emits an `_index.json` for tooling; `--check` mode is dry-run + exits
nonzero when changes would land (drop-in for CI gating).

### 21.7 Update orchestrator + news patcher

[`services/update_orchestrator.py`](backend/app/services/update_orchestrator.py)
wires monitoring loops to memo refresh policy:

- New filing → `on_filing_event(ticker)` → `run_stock_memo(force_refresh=True)`.
- Material/breaking news → `on_news_alert(ticker, alert)` → calls the
  cross-family news-impact agent (Anthropic Haiku 4.5 — independent
  read, not an OpenAI echo of PM synthesis). If material, builds an
  `incremental_patch` snapshot with rating / confidence / risks
  patched. Critic skipped per the locked decision.
- Per-ticker FIFO queue prevents same-ticker race conditions.
- Daily patch cap: max 2 patches per ticker per UTC day. Confidence
  change capped at ±15 points per patch. Each changed field requires a
  one-sentence rationale; fields without one are dropped.

---

## 22. Operational tooling

### 22.1 Admin endpoints

| Endpoint | Surface |
|---|---|
| `GET /api/admin/monitoring/status` | Last-run timestamps + notes per registered loop |
| `GET /api/admin/llm-metrics?run_id=X&since_days=N` | Per-call detail or aggregated cost summary (with USD figures) |
| `GET /api/admin/track-record?horizon_days=X&ticker=Y&sector=Z` | Realized-outcome stats |
| `POST /api/admin/evaluate-outcomes` | Manual trigger for the daily outcome scorer |
| `GET /api/admin/dcf-versions/{ticker}` | DCF version chain with assumption_changes |
| `GET /api/admin/update-queue?ticker=X` | In-process FIFO queue depth |
| `POST /api/admin/news-domains/reload` | Hot-reload `news_domains.json` |
| `GET /api/admin/lopsidedness-audit?n=10` | Per-memo bull-vs-bear key-point + sector-lean distribution. Telemetry on whether the sector-integrated bull/bear is staying balanced (Wave 3A risk-register mitigation) |

### 22.2 Monitoring loops (under `ENABLE_MONITORING=true`)

| Loop | Cadence | Job |
|---|---|---|
| `edgar_poller` | 30 min | Detect new 10-K/Q/8-K; invalidate `company_cold`; enqueue `full_reanalysis` |
| `news_loop` | 1 h / ticker | Push `NewsAlert` records; on material/breaking, call orchestrator's `on_news_alert` |
| `social_loop` | daily | Hot-cache social sentiment |
| `macro_loop` | hourly | FRED snapshot + regime broadcast |
| `outcome_loop` | daily 02:30 UTC | Score every `MemoSnapshot` whose forward window has come of age |
| `history_backfill` | daily 03:15 UTC | Wave 2 — ingest tier-1 financial periods + filings + transcripts |
| `llm_log_gc` | daily | GC `LLMCallLog` rows >90 days old |
| `checkpoint_gc` | daily 04:00 UTC | GC expired `MemoRunCheckpoint` rows |

`history_backfill` (Wave 8E) classifies failures into `rate_limited`
(429s) vs. `auth_errors` (401/403) vs. generic — surfaces the
classification in the loop note so a wedged provider shows up at
`/api/admin/monitoring/status`.

### 22.3 Resumable memo runs — `@checkpointed`

[`services/checkpoint_store.py`](backend/app/services/checkpoint_store.py)
backs every major step of `run_stock_memo` (fundamentals, dcf, comps,
all eight specialists, critic) with a `(run_id, step_name)` cache. A
retry with the same `run_id` skips already-completed steps; first runs
see no behavior change. 24h TTL with daily GC bounds the table.

### 22.4 CI workflows (`.github/workflows/`)

- **`ci.yml`** — pytest (demo mode) + smoke test + frontend build on
  every push and PR. No secrets needed.
- **`nightly-live.yml`** — `pytest.mark.live` suite at 09:00 UTC daily
  with `RUN_LIVE_TESTS=1`. Provider keys come from repo secrets;
  per-test `_require_key` skips when individual keys are missing so
  partial configs still yield green runs.

### 22.5 Schema migrations

[`cache/migrations.py`](backend/app/cache/migrations.py) holds a
`(kind, from_version) -> upgrader` registry. `cache_get` walks the
chain on read so consumers always see the current shape. Missing
intermediates pass through with a warning; exceptions abort the chain
without raising. No upgraders are registered today — this is the
plumbing for the first schema bump.

---

## 23. Cost observability

Every provider LLM call lands in [`LLMCallLog`](backend/app/models.py)
with full attribution: `run_id` / `ticker` / `agent_name` / `provider` /
`model` / `route` / `tokens_in` / `tokens_out` / `duration_ms` /
`success` / `error` / `generated_at`. 90-day retention with a daily GC
job; rows are small enough that this stays cheap on SQLite.

[`services/llm_metrics.py`](backend/app/services/llm_metrics.py)
exposes:

- `cost_per_run(run_id)` — per-call detail + run total tokens + run
  total USD cost.
- `cost_per_agent(since=N days)` — aggregate by agent_name with
  tokens + duration + USD.
- `cost_per_provider(since=...)` — aggregate by provider with USD.
- `slowest_calls(since=..., n=10)` — top-N pathological prompts.

Wave 8D added USD estimation via `estimate_cost_usd(provider, model,
tokens_in, tokens_out)` and `MODEL_PRICES_PER_MTOK` (per-million-token
rates per model, with provider-level fallback). Update the price table
when a provider's published rates change.

CLI report at [`scripts/llm_cost_report.py`](backend/scripts/llm_cost_report.py)
emits a monthly cost summary suitable for posting to Slack or piping
into a finance review.

---

> **Reminder.** MarketMosaic is for investment research and education only. It does not provide
> personalized financial advice. Model portfolios are illustrative scenario constructions.
