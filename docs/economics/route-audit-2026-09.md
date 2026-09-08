# Route audit — 2026-09 (FEAT-002 Phase 1)

**Generated:** 2026-09-08 against `main` at `be6dfce` (post-PR #56).
**Scope:** every backend path in `app.openapi()` and every frontend route in
`frontend/src/App.tsx`, classified for the account-gating / freemium work
(FEAT-002). This document is the *inventory* the customer-policy layer
(`backend/app/auth/policy.py`, slice S1) is built from. Until that module
lands, `backend/app/tests/test_route_audit.py` pins the inventory: a route
added to the API without a row in the table below fails CI.

## How to regenerate

1. Backend paths (from `backend/`):

   ```
   ENABLE_LIVE_DATA=false USE_DEMO_DATA=true python -c "
   from app.main import app
   for p, ops in sorted(app.openapi()['paths'].items()):
       for m in ops:
           if m.upper() not in ('HEAD', 'OPTIONS'):
               print(m.upper(), p)"
   ```

2. Frontend routes: read the `<Route path=…>` elements in `frontend/src/App.tsx`.
3. Called-by: `grep -rn "/api/" frontend/src` for literal paths, then map each
   `api.<method>` in `frontend/src/api/client.ts` to its callers with
   `grep -rnE "api\.<method>\b|\.<method>\(" frontend/src/pages frontend/src/components`
   (several pages chain `api\n  .method(` across lines, so grep the bare
   method name too). `frontend/src/pages/TrackRecord.tsx` and
   `frontend/src/lib/logger.ts` use raw `fetch`, not the client.
4. External calls: read the handler and every service it calls. "LLM" means
   `app.agents.llm.chat_json` / `chat_text` is reachable inside the request;
   "providers" means `get_data_service()` / `provider_cache.cached_call` /
   `get_full_financials` is reachable (the provider cache makes most of these
   free on a warm cache but a cold ticker pays real calls); "DB-only" means
   nothing leaves the process except SQL.
5. Add or update the row, then run
   `python -m pytest -q app/tests/test_route_audit.py`.

## Legend

- **Auth today:** `public` (no guard), `admin-exempt` (matches
  `/api/admin` but listed in `admin_auth.EXEMPT_PREFIXES` because the browser
  calls it), `admin` (bearer `ADMIN_API_TOKEN`).
- **Proposed policy** (plan §3.3 as amended by the orchestrator decisions):
  - `public` — on the explicit allowlist, no token, IP-limited via slowapi.
  - `free` — any signed-in, verified user (Free Explorer or Pro).
  - `pro` — Pro plan (trial, subscription, or admin override) only.
  - `metered:<feature>` — charged against the monthly allowance of that
    feature (`memo_view` Free 3 distinct tickers / Pro unlimited;
    `research_run` Free 1 / Pro 20; `pm_chat` Free 10 / Pro 300). All meter
    periods are UTC calendar months.
  - `dcf` / `comps` — Pro, or Free for a ticker already counted in this
    month's `memo_view` events ("follows memo").
  - `admin` — bearer `ADMIN_API_TOKEN`; a customer JWT never satisfies it.
- **Rate scope** (DEVPLAN ceilings, keyed user-first then IP, stored in
  Postgres/SQLite `rate_limit_windows`):
  - `public_get` 60/min/IP (slowapi, unchanged mechanism, structured 429 body)
  - `data` 120/min/user + secondary IP ceiling
  - `series` 60/min/user
  - `llm_light` 10/min/user, at most 2 concurrent (`active_actions` lease)
  - `research` 3/hour/user, one active job per user+ticker, coalesced with
    any queued/running job for the ticker
  - `global:<name>` one DB window shared by every user
  - `admin` — admin token; existing slowapi per-route limits stay.
  Existing slowapi decorators (`chat` 30/min, `memo_read` 60/min,
  `memo_analyze` 5/min, `custom_screen` 30/min, `seed_universe` 1/min,
  `admin_backfill` 1/5min, default 60/min) remain as per-IP ceilings.

With `AUTH_ENABLED=false` (the shipped default) every row behaves exactly as
the **Auth today** column; the **Proposed policy** column only applies once the
flag is on.

## Backend routes

| Method | Path | Called by (`frontend/src`) | External calls reachable | Auth today | Proposed policy | Rate scope | Notes |
|---|---|---|---|---|---|---|---|
| GET | `/health` | none (client helper `health` unused; Render health check) | none (`get_data_service().mode()` reads config only) | public | public | public_get | unchanged |
| GET | `/api/providers/status` | `pages/Dashboard.tsx`, `pages/Settings.tsx`, `components/ProviderHealthBanner.tsx` | DB-only (`provider_cache.stale_stats`; `ds.status()` is config) | public | public | public_get | stays public per decision; feeds the provider-health banner on marketing pages |
| GET | `/api/stocks` | `pages/Research.tsx`, `pages/Comps.tsx`, `pages/DCFLab.tsx` (`listStocks`) | DB-only (`companies` table) | public | free | data | |
| GET | `/api/stocks/{ticker}` | none (client helper `getStock` has no page caller) | providers (`get_full_financials` → `company_cold` snapshot, cold ticker pays provider calls; `get_quote` live; `get_basic_stats`) | public | free | data | unknown ticker → 404 as today; no lazy-universe insert on this path |
| GET | `/api/stocks/{ticker}/prices` | none (client helper `getStockPrices` has no page caller) | providers (`get_price_history`, provider-cached) | public | free | data | |
| GET | `/api/stocks/{ticker}/memo` | `pages/Research.tsx` (`getStockMemo`; `components/AnalyzeStockGate.tsx` renders the 409 copy) | **LLM** (`run_stock_memo` runs inline when no snapshot or a stale one) + providers (`_ensure_lazy_universe` → profile lookup + 5-year `backfill_ticker`) | public | metered:memo_view | data (slowapi `memo_read` 60/min IP stays) | Under `AUTH_ENABLED`: never generates or backfills inline. No snapshot → 409 `no_memo` with `analyze_path`; stale snapshot served with `X-Memo-Stale: true` + reason; `ondemand=true` → routed through the analyze path (charged as `research_run`, 202); `as_of` → 404 `feature_disabled`. Free allowance counts **distinct tickers** per UTC month, not views. |
| GET | `/api/stocks/{ticker}/memory` | `components/MemoryTrail.tsx` (`stockMemory`) | DB/filesystem only (`memory/companies/<T>.md`) | public | pro | data | |
| GET | `/api/stocks/{ticker}/memos` | `components/MemoVersionTimeline.tsx` (`memoHistory`) | DB-only (`memo_snapshots`) | public | pro | data | |
| POST | `/api/stocks/{ticker}/analyze` | `pages/Research.tsx` (`analyzeStock`); client helper `analyzeStockSync` (`?sync=true`) has no page caller | providers in-request today (`_ensure_lazy_universe` profile + backfill); **LLM** inline when `sync=true`; async path enqueues a `RegenJob` (LLM + providers run in the **worker**) | public | metered:research_run | research (slowapi `memo_analyze` 5/min IP stays) | Under `AUTH_ENABLED`: `sync=true` → 403; charge reserved only when `enqueue` returns `created=True` (coalesced requests are free); lazy-universe resolution + backfill move into the worker's `execute_job` before `run_stock_memo`; worker commits the charge on success, releases on failure/orphan. |
| GET | `/api/stocks/{ticker}/analyze/status` | `pages/Research.tsx` (`analyzeStatus`) | DB-only (`regen_jobs`, `memo_snapshots`, checkpoints) | public | free | data | polling target; must stay cheap |
| GET | `/api/screener` | `pages/Dashboard.tsx`, `pages/Screener.tsx` (`screener`) | providers (`compute_universe_scores` → `get_full_financials` per universe ticker; warm `company_cold` cache normally, cold rows pay provider calls); no LLM | public | free | data | |
| POST | `/api/screener/run` | none | same as `GET /api/screener` | public | free | data | |
| POST | `/api/screener/custom` | `pages/Screener.tsx` (`customScreener`) | DB-only (`screener_metrics`, `screener_scores`) | public | free | data (slowapi `custom_screen` 30/min IP stays) | |
| POST | `/api/screener/nl` | none | **LLM** (`nl_screener._llm_translate`, one cheap-route call) + DB | public | pro + metered:pm_chat | llm_light (slowapi `custom_screen` 30/min IP stays) | counts as one Ask-the-PM turn per decision |
| POST | `/api/chat` | `pages/Chat.tsx` (`chat`) | **LLM** (`classify_intent`; SDK chat agent or memo-context answer; today `single_stock_analysis` / `stock_comparison` run `run_stock_memo` inline — up to 4 memos) + providers (`list_tickers`, chat-SDK tools fetch transcripts/filings) | public | metered:pm_chat | llm_light (slowapi `chat` 30/min IP stays) | Under `AUTH_ENABLED`: `allow_inline_memo=False` — answers from `memo_store.latest_memo`, returns `needs_analysis=[…]` when absent; never generates in-request. |
| GET | `/api/dcf/{ticker}/default-assumptions` | `pages/DCFLab.tsx` (`dcfDefaults`) | providers (`get_full_financials`, `get_estimates`) | public | dcf | data | |
| GET | `/api/dcf/{ticker}/consensus` | `pages/DCFLab.tsx` (`dcfConsensus`) | providers (`get_full_financials`, `get_estimates`) | public | dcf | data | |
| GET | `/api/dcf/{ticker}/saved` | `pages/DCFLab.tsx` (`dcfSaved`) | DB-only (`dcf_models`) | public | dcf | data | |
| POST | `/api/dcf/{ticker}` | `pages/DCFLab.tsx` (`runDCF`) | **LLM** (`build_full_dcf` → `scenario_assumptions.build_bull_bear`, one cheap-route call when a profile and key exist; deterministic fallback otherwise) + providers (`get_full_financials`, `get_estimates`) | public | dcf | series (60/min/user) | accepted as-is per plan §9 Q5; cost tracked via `LLMCallLog.feature` |
| GET | `/api/comps/{ticker}` | `pages/Comps.tsx` (`comps`) | **LLM** (`_llm_exposure_peers`, cheap route, cached 7d) + providers (`get_full_financials` for target + every peer) | public | comps | data | |
| POST | `/api/portfolio/build` | `pages/PortfolioBuilder.tsx` (`buildPortfolio`) | **LLM** (`portfolio_brief.extract_brief`, one call) + providers (`get_universe_dicts` → `get_full_financials`) | public | pro | llm_light (10/min/user, 2 concurrent) | |
| GET | `/api/macro/series` | `pages/Macro.tsx` (`macroSeries`) | providers (FRED via `data_service`, provider-cached) | public | pro | series | |
| POST | `/api/macro/analyze` | `pages/Macro.tsx` (`macroAnalyze`) | **LLM** (`run_macro_scenario`, one cheap-route call) + providers (`macro_snapshot` → FRED) | public | pro + metered:pm_chat | llm_light | counts as one Ask-the-PM turn per decision |
| GET | `/api/data-catalog/meta` | none | none (static `SERIES_REGISTRY`) | public | pro | data | unused by UI |
| GET | `/api/data-catalog/series` | none | none (static registry / `discover_by_query`) | public | pro | data | unused by UI |
| GET | `/api/data-catalog/series/{series_id}` | none | providers (`fetch_series` → `provider_cache.cached_call` → FRED/EIA/BLS/Census) | public | pro | series | `force_refresh` bypasses the cache — keep it admin-only or drop it under auth |
| GET | `/api/data-catalog/ticker/{ticker}/context` | none | providers (`get_company_profile` + `prepare_sector_context` → `fetch_snapshots`) | public | pro | series | unused by UI |
| GET | `/api/data-catalog/ticker/{ticker}/geography` | none | **LLM** only when `allow_llm=true` (`company_geography` cheap-route call); otherwise DB (filings) + seed file | public | pro | data | `allow_llm` forced `False` for non-admin callers |
| GET | `/api/data-catalog/ticker/{ticker}/overlay/{name}` | none | providers (`get_company_profile` + overlay functions → FRED/EIA) | public | pro | series | unused by UI |
| POST | `/api/admin/ui-log` | `lib/logger.ts` (raw `fetch`, 1.5s flush) | DB-only (`ui_logs`, append-only) | admin-exempt | public | public_get | exemption unchanged; body stays bounded |
| POST | `/api/admin/evaluate-outcomes` | `pages/TrackRecord.tsx` (raw `fetch`, bypasses `client.ts`) | providers (`get_price_series` for every due memo + benchmark, provider-cached) + DB | admin-exempt | pro | global:evaluate_outcomes (1 per 10 min, DB window) | exemption stays exactly as today (no `admin_auth` change). The raw `fetch` will not carry the bearer header — S5 must route it through `client.ts` or the button 401s under auth. |
| GET | `/api/admin/track-record` | `pages/TrackRecord.tsx` (`trackRecord`) | DB-only (`memo_outcomes`) | admin-exempt | pro | data | path unchanged |
| GET | `/api/admin/dcf-versions/{ticker}` | `components/DCFVersionHistory.tsx` (`dcfVersionHistory`) | DB-only (`dcf_models`) | admin-exempt | pro | data | |
| GET | `/api/admin/lopsidedness-audit` | none (client helper `lopsidednessAudit`, no page caller) | DB-only (`memo_snapshots`) | admin-exempt | pro | data | exemption stays exactly as today per decision; no UI caller — candidate for a later cleanup, out of scope here |
| GET | `/api/admin/ui-log` | none | DB-only | admin | admin | admin | |
| DELETE | `/api/admin/ui-log` | none | DB-only | admin | admin | admin | |
| POST | `/api/seed-universe` | none | providers (FMP profile per universe member) | admin | admin | admin (slowapi `seed_universe` 1/min) | not under `/api/admin` — listed explicitly in `PROTECTED_PREFIXES` |
| POST | `/api/admin/run-backfill` | none | providers (heavy: ~6 calls per ticker, whole universe by default) | admin | admin | admin (slowapi `admin_backfill` 1/5min) | |
| GET | `/api/admin/monitoring/status` | none | DB-only (`cron_loop_runs`) + process-local | admin | admin | admin | |
| GET | `/api/admin/llm-metrics` | none | DB-only (`llm_call_logs`) | admin | admin | admin | |
| GET | `/api/admin/sdk-traces` | none | DB-only (`sdk_traces`) | admin | admin | admin | |
| GET | `/api/admin/sdk-traces/{run_id}` | none | DB-only | admin | admin | admin | |
| GET | `/api/admin/calibration` | none | DB-only | admin | admin | admin | |
| GET | `/api/admin/per-agent-attribution` | none | DB-only | admin | admin | admin | |
| GET | `/api/admin/regime-accuracy` | none | DB-only | admin | admin | admin | |
| GET | `/api/admin/calibration-summary` | none | DB-only | admin | admin | admin | |
| POST | `/api/admin/run-postmortems` | none | **LLM** (`postmortem_service`, one call per memo) + providers (prices) | admin | admin | admin | |
| GET | `/api/admin/cron-health` | none | DB-only (`cron_loop_runs`) | admin | admin | admin | |
| GET | `/api/admin/universe-review` | none | providers only with `compare_feed=true` (FMP constituents) | admin | admin | admin | |
| POST | `/api/admin/run-weekly-digest` | none | **LLM** (`filing_memory` digest calls) + DB (vector store) | admin | admin | admin | |
| GET | `/api/admin/specialist-reliability` | none | DB-only | admin | admin | admin | |
| GET | `/api/admin/postmortems/{ticker}` | none | DB-only | admin | admin | admin | |
| POST | `/api/admin/rerun-memos` | none | DB-only in-request (enqueues `RegenJob`s; LLM + providers run in the worker) | admin | admin | admin | |
| POST | `/api/admin/mispricing-audit` | none | **LLM** (`mispricing_audit.run_audit`) + DB | admin | admin | admin | |
| GET | `/api/admin/update-queue` | none | process-local memory | admin | admin | admin | |
| POST | `/api/admin/news-domains/reload` | none | filesystem (`news_domains.json`) | admin | admin | admin | |
| GET | `/api/admin/auto-update` | none | DB-only (`companies`) | admin | admin | admin | |
| PUT | `/api/admin/auto-update/{ticker}` | none | DB-only | admin | admin | admin | |
| POST | `/api/admin/auto-update/check/{ticker}` | none | DB-only (`should_auto_regen`) | admin | admin | admin | |
| GET | `/api/admin/llm-breakers` | none | process-local (`agents.llm` breaker dicts) | admin | admin | admin | web process only |
| POST | `/api/admin/llm-breakers/reset` | none | process-local | admin | admin | admin | |
| GET | `/api/admin/llm-recent-failures` | none | DB-only (`llm_call_logs`) | admin | admin | admin | |
| GET | `/api/admin/regen-jobs` | none | DB-only (`regen_jobs`) | admin | admin | admin | |
| POST | `/api/admin/fix-sequences` | none | DB-only (Postgres sequences) | admin | admin | admin | |

### Routes FEAT-002 adds (not in `app.openapi()` today)

Listed so the allowlist is complete when the policy layer is written; they
are owned by slices S1/S3/S4 and are not pinned by `test_route_audit.py`
until they exist.

| Method | Path | External calls | Proposed policy | Rate scope |
|---|---|---|---|---|
| GET | `/api/public/config` | none | public | public_get (`Cache-Control: public, max-age=60`) |
| GET | `/api/public/samples` | DB-only (`public_samples`) | public | public_get |
| GET | `/api/public/samples/{ticker}` | DB-only (`public_samples`; allowlisted tickers only; ETag/304) | public | public_get |
| POST | `/api/public/events` | DB-only (`analytics_events`, allowlisted names/props) | public | public_get |
| GET | `/api/me` | DB-only | free (any valid JWT) | data |
| POST | `/api/me/bootstrap` | DB-only | free (valid JWT) | 3/hour/IP |
| GET | `/api/me/usage` | DB-only | free | data |
| POST | `/api/billing/checkout` | Stripe (httpx) | free, verified email, `BILLING_ENABLED` | 10/hour/user |
| POST | `/api/billing/portal` | Stripe (httpx) | free with `stripe_customer_id` | 10/hour/user |
| POST | `/api/billing/reconcile` | Stripe (httpx) | free | 5/hour/user |
| POST | `/api/billing/webhook` | none (signature verified locally) | Stripe signature only — `@limiter.exempt`, exempt from the customer policy | none |
| GET | `/api/admin/abuse-telemetry` | DB-only | admin | admin |
| GET | `/api/admin/billing/users/{external_id}` | DB-only | admin | admin |
| POST | `/api/admin/billing/overrides` | DB-only | admin | admin |
| POST | `/api/admin/samples/rebuild` | DB-only (enqueues; the worker builds) | admin | admin |

### Non-API paths

`/docs`, `/redoc`, `/openapi.json` are served by FastAPI and are not in
`app.openapi()["paths"]`; per decision they stay exactly as today. Everything
else not under `api/`, `health`, `docs`, `redoc`, `openapi.json` is the SPA
shell (`test_spa_fallback.py`).

## Frontend routes (`frontend/src/App.tsx`)

All routes render inside `Layout`; `RouteTracker` sits at the root.

| Route | Page | Backend calls (via `api/client.ts` unless noted) | After plan §6.1 |
|---|---|---|---|
| `/` | `pages/Dashboard.tsx` | `GET /api/providers/status`, `GET /api/screener` | becomes the public Landing; the dashboard moves to `/app` |
| `/chat` | `pages/Chat.tsx` | `POST /api/chat` | `/app/chat` (legacy path redirects, query preserved) |
| `/research` | `pages/Research.tsx` (+ `AnalyzeStockGate`, `MemoVersionTimeline`, `MemoryTrail`) | `GET /api/stocks`, `GET /api/stocks/{t}/memo`, `POST /api/stocks/{t}/analyze`, `GET /api/stocks/{t}/analyze/status`, `GET /api/stocks/{t}/memos`, `GET /api/stocks/{t}/memory` | `/app/research` |
| `/dcf` | `pages/DCFLab.tsx` (+ `DCFVersionHistory`) | `GET /api/stocks`, `GET /api/dcf/{t}/default-assumptions`, `GET /api/dcf/{t}/consensus`, `GET /api/dcf/{t}/saved`, `POST /api/dcf/{t}`, `GET /api/admin/dcf-versions/{t}` | `/app/dcf` |
| `/comps` | `pages/Comps.tsx` | `GET /api/stocks`, `GET /api/comps/{t}` | `/app/comps` |
| `/screener` | `pages/Screener.tsx` | `GET /api/screener`, `POST /api/screener/custom` | `/app/screener` |
| `/portfolio` | `pages/PortfolioBuilder.tsx` | `POST /api/portfolio/build` | `/app/portfolio` |
| `/macro` | `pages/Macro.tsx` | `GET /api/macro/series`, `POST /api/macro/analyze` | `/app/macro` |
| `/track-record` | `pages/TrackRecord.tsx` | `GET /api/admin/track-record`; raw `fetch` `POST /api/admin/evaluate-outcomes` | `/app/track-record` |
| `/settings` | `pages/Settings.tsx` | `GET /api/providers/status` | `/app/settings` |
| `*` | `pages/Dashboard.tsx` | as `/` | public `NotFound` |
| (every page) | `components/Layout.tsx` → `ProviderHealthBanner`; `lib/logger.ts` | `GET /api/providers/status`; `POST /api/admin/ui-log` | unchanged |

## Findings that affect other slices

- `pages/TrackRecord.tsx:52` calls `POST /api/admin/evaluate-outcomes` with a
  raw `fetch`, so the `client.ts` bearer header will never be attached; under
  the `pro` policy the "Evaluate now" button 401s unless S5 routes it through
  the client (or the owner chooses to make the endpoint admin-only — plan §9
  Q8).
- `GET /api/stocks/{ticker}`, `GET /api/stocks/{ticker}/prices`,
  `POST /api/screener/run`, `POST /api/screener/nl`, every
  `/api/data-catalog/*` route and `GET /api/admin/lopsidedness-audit` have no
  frontend caller today. They are classified anyway (default-deny means an
  unclassified route is a 401, not a hole), but none needs a UX affordance.
- Cost-bearing work that runs *inside* a request today and must not under
  `AUTH_ENABLED`: inline memo generation on `GET /memo`, `POST /analyze?sync=true`
  and `POST /api/chat`; the lazy-universe profile lookup + 5-year backfill on
  `GET /memo` and `POST /analyze`. Everything else that reaches an LLM does one
  bounded cheap-route call (`dcf`, `comps`, `macro/analyze`, `screener/nl`,
  `portfolio/build`) and is accepted as-is behind the `llm_light` / `series`
  scopes.
