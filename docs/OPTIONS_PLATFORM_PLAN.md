# Options Research Platform — Stack & Architecture Plan

> **Scope**: Plan document for a separate repo. Captures the data, infrastructure, and feature stack for an options-focused research platform. Implementation will live in its own codebase.

## 1. Product Goals

A research-grade (not execution-grade) platform for options analysis. Core capabilities:

1. **Live options chains** — full chain views across all listed expiries and strikes for any US equity/ETF/index, refreshed in seconds.
2. **IV surfaces** — 3D vol surfaces (strike × expiry × IV), term structures, skew curves, surface evolution over time.
3. **Screening** — filter the universe by IV rank, IV percentile, Greeks (delta/gamma/theta/vega), open interest, volume, earnings proximity, unusual activity.
4. **Backtesting** — multi-leg strategy backtests with realistic fills, Greeks evolution, P&L attribution, vol regime tagging.
5. **LLM chat interface** — conversational layer that reads live options data and reasons about strategies (e.g., "find me a calendar spread on TSLA that benefits from elevated front-month IV", "analyze risk on this iron condor through earnings").

**Non-goals** (explicitly out of scope):
- Order routing / live execution
- Sub-millisecond latency
- Market-making or HFT use cases

---

## 2. Data Stack

### Primary providers

| Layer | Provider | Cost | Why |
|---|---|---|---|
| **Live OPRA chains, NBBO, trades** | **Polygon.io Options Advanced** | ~$199/mo | Full OPRA WebSocket + REST, sub-second updates, no brokerage relationship needed. Best DX. |
| **Pre-computed analytics & IV history** | **ORATS Live Data API** | $199/mo | 500+ proprietary indicators, IV rank/percentile, smoothed market values, 18yr backtestable history, IV surface params, earnings moves. Saves months of building. |
| **Underlying equities (prices, splits, dividends)** | Polygon (included w/ Options Advanced) or **Tiingo** | $0 incl. or $30/mo | Polygon options plan typically bundles equity quotes. |
| **Fundamentals & earnings dates** | **FMP Premium** | $59/mo | Earnings calendar drives a lot of options trades. |
| **Macro (rates, vol regimes)** | **FRED** | Free | Risk-free rate for pricing, regime context. |
| **News / catalysts** | **Benzinga Pro API** or scrape | $TBD | Optional v2 — flow/news-driven screens. |

**Total data spend (v1): ~$460/mo**. Reduce to ~$260/mo by skipping ORATS and computing analytics in-house (more eng time, less polish).

### Build vs. buy decision: ORATS

ORATS is the single biggest cost lever. Decide based on:
- **Buy ORATS** if IV rank, surface params, earnings moves, and 18yr indicator history are first-class features users see.
- **Skip ORATS** if you're comfortable building IV calc + surface fitting yourself and starting history from day-1 of your platform.

Recommend **buying ORATS for v1** to ship faster, then evaluate replacing pieces in v2 once you know what's actually used.

### Fallback / future providers
- **ThetaData** — cheaper alternative to Polygon for raw options (~$80/mo entry). Worth evaluating if Polygon costs become an issue at scale.
- **CBOE DataShop** — direct exchange data if going institutional.
- **Unusual Whales** — flow/sentiment overlay for v2.

---

## 3. Backend Architecture

### Language & framework
- **Python 3.12+** with **FastAPI** for API layer (matches your existing stock-research-platform stack — easier knowledge transfer).
- **Polars** for analytical DataFrames (faster than Pandas on options-sized data; chains can have 5000+ rows per symbol).
- **NumPy + SciPy** for IV calc, surface fitting.
- **`py_vollib`** or **`QuantLib`** for option pricing / Greeks if not relying on ORATS-computed values.

### Service layout

```
options-research-platform/
├── backend/
│   ├── app/
│   │   ├── api/                  # FastAPI routes
│   │   │   ├── chains.py         # GET /chain/{symbol}, /chain/{symbol}/snapshot
│   │   │   ├── surface.py        # GET /surface/{symbol} (3D IV grid)
│   │   │   ├── screener.py       # POST /screen (filter rules)
│   │   │   ├── backtest.py       # POST /backtest (strategy spec)
│   │   │   └── chat.py           # POST /chat (LLM stream)
│   │   ├── data/
│   │   │   ├── polygon_client.py # REST + WebSocket
│   │   │   ├── orats_client.py
│   │   │   ├── fmp_client.py
│   │   │   └── cache.py          # Redis-backed
│   │   ├── analytics/
│   │   │   ├── iv.py             # BS solve, vega-weighted fits
│   │   │   ├── surface.py        # SVI / SABR fitting
│   │   │   ├── greeks.py
│   │   │   └── screener_engine.py
│   │   ├── backtest/
│   │   │   ├── engine.py         # event-driven loop over historical snapshots
│   │   │   ├── strategies/       # iron condor, calendar, vertical, etc.
│   │   │   ├── fills.py          # mid + slippage models
│   │   │   └── metrics.py        # Sharpe, max DD, win rate, P&L attribution
│   │   ├── llm/
│   │   │   ├── tools.py          # tool definitions exposed to Claude/GPT
│   │   │   ├── context.py        # builds chain/surface context for prompts
│   │   │   └── audit.py          # trace logging (port from stock-research-platform)
│   │   ├── streaming/
│   │   │   └── ws_router.py      # fan-out Polygon WS to client WebSockets
│   │   └── models/               # SQLAlchemy / Pydantic schemas
│   └── tests/
├── frontend/
│   └── (see Section 6)
└── infra/
    ├── docker-compose.yml
    └── ...
```

### Why event-driven streaming + REST snapshots
- WebSocket from Polygon for live updates → fan out to client browsers via your own WS endpoint.
- REST snapshots for initial chain load and for screener/backtest queries.
- Don't build a Kafka pipeline yet — single-process fan-out is fine until many concurrent users.

---

## 4. Storage

| Data | Store | Notes |
|---|---|---|
| **Live chain snapshots** (last N minutes) | **Redis** | TTL-based, fast read for chain views. |
| **Historical chains (1-min bars)** | **Parquet on S3 (or local disk in dev)** | Partitioned by `symbol/date`. DuckDB queries directly. |
| **IV surface params over time** | **TimescaleDB** or Postgres + native partitioning | Cheap to store, fast time-range queries. |
| **User-saved screens, strategies, watchlists** | **Postgres** | Standard relational. |
| **Backtest results** | **Postgres** (metadata) + **Parquet** (per-trade ledger) | Metadata for browsing, ledger for deep dive. |
| **LLM call traces** | **Postgres** | Port the audit pattern from stock-research-platform Wave 1a. |

**Why DuckDB + Parquet over Timescale for historical chains**: chains are wide and snapshot-heavy. DuckDB on Parquet handles "give me ATM IV across all of Q1 for AAPL" in seconds without running a DB server. Use Timescale for narrower time-series like surface parameters where ongoing writes matter.

---

## 5. Core Feature Implementations

### 5.1 Live Chain View
- On symbol load: REST call to Polygon for snapshot → render full chain (calls/puts × strikes × expiries).
- Subscribe to Polygon WebSocket: `O:{symbol}*` for trades + quotes on every contract.
- Server fans out to client WS, throttled to ~2Hz per contract to avoid overwhelming the browser.
- Render Greeks (delta/gamma/theta/vega/rho) — from ORATS if available, else compute via `py_vollib` using current underlying mid + risk-free rate from FRED.

### 5.2 IV Surface
- Pull all OTM contracts across expiries.
- Filter: bid > 0, OI > threshold, |delta| < 0.5 for OTM definition.
- Fit per-expiry smile with **SVI** (Stochastic Volatility Inspired) or **SABR**. SVI is more standard for equity options.
- Render as a 3D mesh (Plotly or Three.js) with axes: log-moneyness × time-to-expiry × IV.
- Cache fits in Redis with 30s TTL during market hours.
- Historical surfaces: snapshot fits to Parquet every N minutes for replay.

### 5.3 Screener
- Define a **filter DSL** (JSON): `{"iv_rank": {">": 50}, "delta": {"between": [0.25, 0.4]}, "dte": {"between": [30, 60]}}`.
- Universe-wide scan: pre-compute aggregated metrics (IV rank, ATM IV, total OI/vol, earnings date) per symbol, refresh every 5 min.
- Contract-level scan: for symbols passing universe filter, scan their chain for matching contracts.
- Index aggregated metrics in Postgres for fast filtering; do contract-level scans on-demand.

### 5.4 Backtesting
- **Historical data source**: ORATS 1-min intraday since Aug 2020, or Polygon historical snapshots.
- **Engine**: event-driven loop over snapshots. At each step, evaluate strategy entry/exit rules, update positions, mark-to-market.
- **Strategies as code**: Python classes with `should_open(snapshot, state)`, `should_close(snapshot, state, position)`, `legs(snapshot)`.
- **Pre-built templates**: long call/put, vertical, iron condor, calendar, diagonal, butterfly, straddle, strangle, ratio.
- **Fill model**: mid - configurable slippage (bps of mid or fixed cents). Optional bid/ask fills for conservative.
- **Output**: per-trade ledger (Parquet), summary metrics (Sharpe, max DD, win rate, profit factor, Greeks-weighted exposure over time).
- **Compare runs** in UI: select multiple backtests, overlay equity curves.

### 5.5 LLM Chat Interface

**This is the hardest part to do well.** Done right, it's a research multiplier. Done wrong, it hallucinates Greeks.

**Architecture**:
- **Tool-using agent** (Claude with tool use, or OpenAI function calling).
- LLM doesn't see raw chains as text — it calls **tools** that return structured data:
  - `get_chain(symbol, expiry?)` → JSON chain
  - `get_iv_surface(symbol)` → fit params + key strikes
  - `get_iv_rank(symbol, lookback_days=252)` → number
  - `screen(filter)` → list of matches
  - `price_strategy(legs)` → cost, max profit, max loss, breakevens, Greeks
  - `backtest(strategy_spec, params)` → summary metrics
  - `get_earnings_date(symbol)` → date + expected move
  - `get_news(symbol, lookback_hours)` → headlines (v2)

**Why tool-based, not raw-data prompt**:
1. Chains are too big to fit in context efficiently.
2. The LLM is bad at arithmetic on options data — let the backend compute.
3. Audit trail: every tool call is logged, so you can verify *what the LLM actually read* before reasoning.

**Prompt strategy**:
- System prompt establishes: "You are an options research assistant. Always cite the data you use via tool calls. Never invent Greeks or prices. When recommending a strategy, call `price_strategy` to verify."
- Encourage chain-of-thought visible to user (toggleable).
- For "create a strategy that…" requests: agent plans → calls tools to gather data → proposes 2-3 candidates → calls `price_strategy` on each → presents with explicit risk metrics.

**Audit logging**: every prompt, every tool call, every response stored in Postgres. Reuse the pattern from `feat(wave-1a): LLM call trace logging` — already proven in stock-research-platform.

**Model choice**:
- **Claude Sonnet 4.6 or Opus 4.7** for chat (better tool use, better at financial reasoning, less prone to hallucinated numbers).
- **Haiku 4.5** for tool argument parsing or screener intent extraction (cheaper, faster).
- Use **prompt caching** on the system prompt + tool definitions (1hr TTL) — significant cost reduction since the chat opens with the same setup every turn.

---

## 6. Frontend

- **React + TypeScript + Vite** (matches stock-research-platform).
- **TailwindCSS** for styling.
- **TanStack Query** for REST data, **native WebSocket** for streaming.
- **Charting**:
  - **Lightweight Charts** for time-series (IV history, P&L curves).
  - **Plotly.js** for 3D IV surface (cheap to render, interactive).
  - **AG Grid** or **TanStack Table** for chain views (handle 5k+ rows with virtualization).
- **State**: Zustand for client state, TanStack Query for server cache.

**Key views**:
1. **Symbol page** — chain table, mini surface, IV history, key Greeks at ATM.
2. **Surface explorer** — full 3D, term structure curve, skew curves per expiry.
3. **Screener** — filter builder, results grid, save-as-watchlist.
4. **Strategy builder** — drag-drop legs, live P&L diagram, Greeks dashboard.
5. **Backtest** — strategy spec, run, compare runs, drill into per-trade ledger.
6. **Chat** — left rail conversation, right rail "context window" showing what tools were called and what data was returned (transparency).

---

## 7. Infrastructure & Ops

- **Local dev**: Docker Compose (Postgres, Redis, FastAPI, frontend).
- **Production**:
  - Single VPS to start (Hetzner / Fly.io / Railway). Don't over-engineer.
  - WebSocket fan-out is single-process Python initially. Move to dedicated Go service if it becomes a bottleneck.
  - S3 (or R2) for Parquet storage.
  - Managed Postgres (Neon / Supabase / RDS).
  - Managed Redis (Upstash / Redis Cloud).

- **Monitoring**:
  - Sentry for errors.
  - Prometheus + Grafana for latency / WS connection counts / Polygon API quota usage.
  - Daily Polygon quota alerts (you'll burn through tiers fast if a backtest goes wild).

- **Secrets**: `.env` for dev, AWS Secrets Manager / Doppler for prod.

- **CI**: GitHub Actions — pytest + ruff + mypy on backend, vitest + tsc on frontend.

---

## 8. Phased Roadmap

### Phase 1 — Foundations (4–6 weeks)
- [ ] Repo scaffold, FastAPI + React boilerplate.
- [ ] Polygon REST client (chains, snapshots).
- [ ] Postgres + Redis + S3 setup.
- [ ] Symbol page MVP: load chain, render table with Greeks (computed locally with `py_vollib`).
- [ ] Auth (single-user OK at first; Clerk/Supabase Auth for multi-user later).

### Phase 2 — Streaming & Surface (3–4 weeks)
- [ ] Polygon WebSocket integration + client fan-out.
- [ ] Live chain updates in UI.
- [ ] SVI fitter, surface page with 3D plot.
- [ ] Snapshot historical surfaces to Parquet hourly.

### Phase 3 — Screener (3–4 weeks)
- [ ] ORATS integration for IV rank, indicators (or build in-house).
- [ ] Universe metric pre-computation job (cron every 5 min during market hours).
- [ ] Filter DSL + UI builder.
- [ ] Save / share / watchlist.

### Phase 4 — Backtesting (4–6 weeks)
- [ ] Historical snapshot ingest from ORATS 1-min or Polygon flat-files.
- [ ] Event-driven engine.
- [ ] 5–10 pre-built strategies.
- [ ] Backtest comparison UI.

### Phase 5 — LLM Chat (3–4 weeks)
- [ ] Tool definitions + Anthropic SDK integration.
- [ ] Audit logging (port from stock-research-platform).
- [ ] Chat UI with tool-call transparency rail.
- [ ] Prompt caching, evaluation harness for common queries.

### Phase 6 — Polish & Public (ongoing)
- [ ] User accounts, billing (Stripe).
- [ ] Rate limiting per user tier.
- [ ] Onboarding, sample strategies, docs.

**Total to feature-complete v1: ~5–7 months solo, ~3–4 months with a small team.**

---

## 9. Cost Summary

### Monthly recurring (data + infra)

| Item | Cost |
|---|---|
| Polygon Options Advanced | $199 |
| ORATS Live Data API | $199 |
| FMP Premium | $59 |
| FRED, EDGAR | $0 |
| VPS (Hetzner CCX13 or Fly) | ~$30 |
| Managed Postgres (Neon Pro) | ~$20 |
| Managed Redis (Upstash) | ~$10 |
| S3 / R2 | ~$5 |
| Sentry, Grafana Cloud (free tiers) | $0 |
| **Subtotal** | **~$520/mo** |

### LLM costs (variable, depends on usage)
- Claude Sonnet 4.6 at ~$3/M input, $15/M output, with prompt caching on system + tools (~90% cache hit): roughly **$0.05–$0.20 per chat session** depending on length.
- Budget $50–$200/mo for early users, scales with adoption.

### One-time / optional
- TradingView lightweight charts: free.
- Plotly: free.
- AG Grid Community: free; Enterprise license $$$ if you want pinned/locked features.

---

## 10. Open Questions & Risks

**Open questions**:
1. **Is this a SaaS product or a personal tool?** Changes auth, billing, UX, ops complexity.
2. **Do you actually need ORATS, or build analytics in-house?** $199/mo × 12 = $2,400/yr — significant if user-base is small.
3. **What's the historical depth requirement for backtests?** ORATS goes back to 2007 EOD / 2020 1-min. If you need pre-2007, you're in flat-files-from-CBOE territory.
4. **LLM provider**: lock to Anthropic, or abstract to support OpenAI/Gemini? (Your current platform supports Vertex AI alt — same pattern would apply.)

**Risks**:
1. **Polygon options costs scale with usage.** Backtests that pull 5 years of 1-min chains for 100 symbols can blow through quotas. Build cost-aware caching from day one.
2. **OPRA redistribution rules.** If you let users export raw data, OPRA will care. Read their licensing carefully before launch.
3. **LLM hallucinations on Greeks/prices.** Mitigate with strict tool use + post-tool verification. Never let the LLM invent a number.
4. **Surface fitting is hard.** Bad fits look stupid. Have fallback to raw IV grid display when fits fail (fewer than N quotes per expiry, etc.).
5. **WebSocket connection limits.** Polygon has connection caps. If you want many concurrent users on live chains, you're aggregating one WS connection server-side and fanning out.

---

## 11. References

- Polygon Options API: https://polygon.io/options
- ORATS Data API: https://orats.com/data-api
- FMP Pricing: https://site.financialmodelingprep.com/pricing-plans
- SVI parameterization: Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces" (2014)
- `py_vollib`: https://github.com/vollib/py_vollib
- QuantLib: https://www.quantlib.org/
- OPRA fee/licensing: https://www.marketdata.app/education/options/opra-fees/
