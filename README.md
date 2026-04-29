# MarketMosaic — Your AI Investment Committee

> A multi-agent equity research and portfolio management platform for self-directed investors.
> Specialist agents (sector, earnings, filings, valuation, comps, macro, risk) collaborate under a
> Portfolio Manager orchestrator to produce structured stock memos, ranked ideas, and scenario-based
> model portfolios.

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
  ┌────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐
  │ Sector │  │ Earnings   │  │  Filing  │  │Valuation │  │ Comps  │
  │ agent  │  │ Call agent │  │  agent   │  │  + DCF   │  │ agent  │
  └───┬────┘  └─────┬──────┘  └────┬─────┘  └────┬─────┘  └────┬───┘
      │             │              │             │             │
      └─────────────┴──────────────┴──┬──────────┴─────────────┘
                                      ▼
                            ┌──────────────────┐
                            │  Macro · Risk    │
                            │  Agents          │
                            └──────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │  Risk Committee /  │
                          │  Critic            │
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
| Sector specialist           | [`backend/app/agents/sector_agents.py`](backend/app/agents/sector_agents.py) (configs: [`backend/app/data/sector_configs.json`](backend/app/data/sector_configs.json)) |
| Earnings call analyst       | [`backend/app/agents/earnings_agent.py`](backend/app/agents/earnings_agent.py)                                                    |
| Filing analyst (10-K/Q/8-K) | [`backend/app/agents/filing_agent.py`](backend/app/agents/filing_agent.py)                                                        |
| Valuation analyst (DCF)     | [`backend/app/agents/valuation_agent.py`](backend/app/agents/valuation_agent.py) + engine [`finance/dcf.py`](backend/app/finance/dcf.py) |
| Comps analyst               | [`backend/app/agents/comps_agent.py`](backend/app/agents/comps_agent.py) + engine [`finance/comps.py`](backend/app/finance/comps.py) |
| Macro analyst               | [`backend/app/agents/macro_agent.py`](backend/app/agents/macro_agent.py)                                                          |
| Risk analyst                | [`backend/app/agents/risk_agent.py`](backend/app/agents/risk_agent.py)                                                            |
| Risk committee / Critic     | [`backend/app/agents/critic_agent.py`](backend/app/agents/critic_agent.py)                                                        |
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

- `Company` — master security universe
- `StockMemo` — generated memos (versioned, scored)
- `ScreenerScore` — pre-computed factor scores per ticker / theme
- `CachedDocument` — chunked filings, transcripts, news (used by retrieval)
- `PortfolioRun` — saved portfolio constructions

SQLite by default; switch to Postgres by setting `DATABASE_URL=postgresql+psycopg2://...`.

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

Set these in `.env` (copy from `example.env`). The `Settings` page shows which providers are
configured at runtime.

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
├── example.env
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

These items are tracked in the codebase but not yet completed. Each links to
the relevant code path so a future contributor can pick it up.

### 20.1 OpenAI Agents SDK package upgrade

The official `openai-agents` package depends on `openai>=2.26`,
`pydantic>=2.12`, and `starlette>=0.49`, all incompatible with the pinned
`fastapi==0.115.0`. The current implementation in
[`agents/sdk_runtime.py`](backend/app/agents/sdk_runtime.py) is a wire-
compatible in-process shim that mirrors `Agent`, `Runner`, `function_tool`,
and handoffs. **Action:** coordinate a FastAPI/pydantic/starlette bump and
replace the shim with the upstream package.

### 20.2 Live-mode token cost measurement

The cold-vs-warm cost ratio reported in §19.2 is from cache-write accounting
on demo data (LLMs disabled). Real-token measurement requires running the
smoke suite with `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` configured and
diffing actual usage. Hook into [`cache/snapshots.py::log_cost`](backend/app/cache/snapshots.py)
from the LLM helpers so live runs annotate `cost_tokens` with real values.

### 20.3 EDGAR / Gemini live integration tests

[`monitoring/edgar_poller.py`](backend/app/monitoring/edgar_poller.py) is a no-op
in demo mode (DemoProvider returns no filings). Add an opt-in
`RUN_LIVE_TESTS=1` test path that exercises:

- Real EDGAR submissions API → `company_cold` invalidation flow.
- Real Gemini news + social calls (with the search-grounding allow-list).
- The Anthropic critic round-trip when `ANTHROPIC_API_KEY` is set.

### 20.4 Frontend wiring of new memo fields

The sector finding now carries:

- `cross_sector_relevance: List[str]`
- `macro_alignment: str`
- `macro_broadcast: MacroBroadcast`
- `pending_news_alerts: List[NewsAlert]`

These ride inside `sector_agent_view.data` but aren't yet rendered by
[`frontend/src/components/MemoCard.tsx`](frontend/src/components/MemoCard.tsx)
or [`frontend/src/pages/Research.tsx`](frontend/src/pages/Research.tsx).
**Action:** add UI affordances for the cross-sector pull-through chips, the
macro regime banner, and a hot-news side panel.

### 20.5 Cohort-similarity invalidation tightening

[`services/sector_research_service.py::run_sector_research`](backend/app/services/sector_research_service.py)
attaches every cohort member's `company_cold` snapshot as a parent. Any
peer's 10-K refresh stales the warm snapshot — even when the changed peer's
ratios that matter for cohort math are unchanged. **Action:** hash the
specific KPI inputs (revenue / op income / capex / shares) into the warm
snapshot's `sources_used` so unchanged-in-relevant-ways refreshes don't
trigger a recompute.

### 20.6 News allow-list governance

[`agents/news_agent.py`](backend/app/agents/news_agent.py) ships a hard-coded
`_ALLOWED_DOMAINS` / `_BLOCKED_DOMAINS`. **Action:** move these to a JSON
config under `app/data/` so editorial governance doesn't require a code
change.

### 20.7 SDK-runtime LLM wiring

The shim's `Agent.handler` currently delegates to legacy implementations so
demo mode works without an LLM. **Action:** when the upstream SDK is in
place (§20.1), wire the PM/sector/tool handlers to actually issue model
calls + handoffs, and capture per-step cost into `CacheCostLog`.

### 20.8 Schema-version migration path

[`cache/snapshots.py`](backend/app/cache/snapshots.py) writes
`schema_version` into both the column and the payload. Forward-compat
readers exist in tests but there's no concrete upgrader. **Action:** add a
`backend/app/cache/migrations.py` that registers per-`(kind, from_version,
to_version)` transformer functions and runs on read when versions don't match.

---

> **Reminder.** MarketMosaic is for investment research and education only. It does not provide
> personalized financial advice. Model portfolios are illustrative scenario constructions.
