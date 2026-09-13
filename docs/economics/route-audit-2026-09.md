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
| GET | `/api/public/config` | none yet (S5 `ConfigProvider` will read it at boot) | none (settings only) | public | public | public_get | `Cache-Control: public, max-age=60`; feature flags + Clerk publishable key + prices |
| GET | `/api/me` | none yet (S5 `useAccount`) | DB-only (`users`, `subscriptions`, `usage_counters`, `admin_overrides`) | public (auth off → anonymous principal) | free | data | requires a valid JWT once `AUTH_ENABLED` |
| POST | `/api/me/bootstrap` | none yet (S5 `RequireAuth`, once per session) | DB-only (`users` upsert; one trial per verified `email_hash`) | public (auth off → no-op) | free | data (slowapi 3/hour/IP keyed by `rate_limit.client_ip`) | idempotent; never resets a started trial |
| GET | `/api/me/usage` | none yet (S5 `Account.tsx`) | DB-only (`usage_counters`, `usage_events`) | public (auth off → anonymous) | free | data | |
| GET | `/api/public/samples` | none yet (S6 landing page) | DB-only (`public_samples`; allowlisted tickers only) | public | public | public_get | `Cache-Control: public, max-age=300`; build state per listed ticker, built or not |
| GET | `/api/public/samples/{ticker}` | none yet (S6 sample components) | DB-only (`public_samples`; allowlisted tickers only; never generates or backfills) | public | public | public_get | strong `ETag`, `If-None-Match` → 304; `Cache-Control: public, max-age=3600`; unlisted → 404 even with a memo; listed-but-unbuilt → 200 with nulls + `degraded`; `prices` is `[{date, close}]` |
| POST | `/api/public/events` | none yet (S6 analytics helper) | DB-only (`analytics_events`; allowlisted names/props, ≤50 events, 200-char strings, 64 KB body) | public | public | public_get | always 200 `{accepted, rejected}`; `user_id`/`plan` only from a verified bearer |
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
| POST | `/api/dcf/{ticker}` | `pages/DCFLab.tsx` (`runDCF`) | **LLM** (`build_full_dcf` → `scenario_assumptions.build_bull_bear`, one cheap-route call when a profile and key exist; deterministic fallback otherwise) + providers (`get_full_financials`, `get_estimates`) | public | dcf | series (60/min/user) | accepted as-is per plan §9 Q5; cost tracked via `LLMCallLog.feature`. The LLM call fires on every request with an assumptions body (`build_dcf` caches only `assumptions=None`), so for Free the only ceiling is the rate scope — see `unit-costs-2026-09.md` §3 owner question |
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
| GET | `/api/fundamentals/catalog` | none yet (FEAT-001 S6 `pages/Fundamentals.tsx`) | none (static metric catalog) | public (auth off → anonymous principal, Pro shape) | free | data (slowapi `fundamentals_series` 60/min IP) | FEAT-001 S2. `ENABLE_FUNDAMENTALS_EXPLORER=false` unmounts both fundamentals routers (the rollback switch) |
| POST | `/api/fundamentals/series` | none yet (FEAT-001 S6 `pages/Fundamentals.tsx`) | DB-only (`financial_periods`, `companies`); providers only when a market-derived metric is selected (`get_price_series`, provider-cached, the same read as `/api/stocks/{t}/prices`); never backfills | public (auth off → Pro shape 5×4×full history) | free | series (slowapi `fundamentals_series` 60/min IP) | FEAT-001 S2. Plan shape inside the route via `features.shape`: Free 2×2×5y — extra companies/metrics → 402 `plan_required` with `extra.limits/requested/upgrade`, extra years capped with `limits.capped_by_plan`; Pro 5×4×max. 422 `invalid_request` on an unknown metric; 404 `unknown_tickers` only when every ticker is unknown, else per-ticker `unavailable` (`not_backfilled`, remedy = run research) |
| POST | `/api/fundamentals/commentary` | none yet (FEAT-001 S6 `CommentaryPanel`) | **LLM** (one cheap-route call over the displayed series + bounded memo excerpts; none at all with `FUNDAMENTALS_ANON_COMMENTARY=false` and the wall off) + DB (`chart_commentaries` cache, 90-day GC in `llm_log_gc`) | public (auth off → deterministic degraded body unless `FUNDAMENTALS_ANON_COMMENTARY=true`) | metered:chart_commentary | llm_light (slowapi `fundamentals_commentary` 10/min IP; 2 concurrent via `active_actions`) | FEAT-001 S3 fills the route (S2 mounts the stub router). Free 5 / Pro 100 a month; charged before the LLM call, released when it returns nothing; cache hits are free |
| GET | `/api/scorecard` | none yet (Phase 6 F `pages/Scorecard.tsx`) | DB-only (`scorecard_runs`, `scorecard_scores`, `companies`) | public (auth off) | pro | data | Phase 6 C. Latest succeeded run's cross-section; `sort_by` whitelisted (422 otherwise); 404 before any run. `ENABLE_SCORECARD=false` → 404 `feature_disabled` on every scorecard read |
| GET | `/api/scorecard/spec` | none yet (Phase 6 F) | DB-only (`scorecard_versions`; the first scorecard read per web process lazily upserts the in-code spec's registry row, idempotent; in-code spec fallback) | public (auth off) | pro | data | Phase 6 C. Methodology + `spec_hash` |
| GET | `/api/scorecard/evaluation` | none yet (Phase 6 F evaluation tab) | DB-only (`scorecard_evaluations`) | public (auth off) | pro | data | Phase 6 C. Latest quintile / FF6 / double-LASSO rows, caveats always attached |
| GET | `/api/scorecard/export` | none yet (Phase 6 F export link, `contract=v1`) | DB-only (`scorecard_scores` streamed with `yield_per`) | public (auth off; bearer `SCORECARD_EXPORT_TOKEN` required when that setting is non-empty) | pro | data (slowapi `scorecard_export` 30/min IP) | Phase 6 C. Frozen v1 column order, `X-Scorecard-Contract: v1`; with `SCORECARD_EXPORT_TOKEN` set `auth/policy.lookup` classifies it public and the route enforces the token (separate from the admin token) |
| GET | `/api/scorecard/{ticker}` | none yet (Phase 6 F ticker view; FullInvestmentMemo reads the memo's own `scorecard` field) | DB-only (`scorecard_scores`, `scorecard_runs`, `companies`) | public (auth off) | pro | data | Phase 6 C. Latest row + observed/model-read features + month-end history |
| GET | `/api/industries/taxonomy` | none yet (FEAT-003 slice 6 `pages/Industries.tsx`) | DB-only (`gics_taxonomy_versions`, `gics_nodes` — immutable, cached per version id — `company_industry_classifications` counted in one grouped read, plus one bulk latest-edition read) | public (auth off) | public | data (slowapi `industry_read` 60/min IP) | FEAT-003 slice 5. Always answers: before the first import it is a 503 `taxonomy_not_imported` that still carries the access policy and the operator remedy, so the UI can explain a gate rather than render a 401. Every count (sectors, groups, industries, sub-industries, constituents) is read from the registry — no literal 24 / 25 / 74 / 163 may appear in the API |
| GET | `/api/industries/snapshot` | none yet (FEAT-003 slice 6; the PM block quotes the same row) | DB-only (`industry_snapshots`, latest or by `period_key`) | public (auth off) | pro | data (slowapi `industry_read` 60/min IP) | FEAT-003 slice 5. The cross-industry view rides the `pm_chat` tier (feature `industry_analysis`); the worker computes it, this only reads it |
| GET | `/api/industries/{code}/report` | none yet (FEAT-003 slice 6) | DB-only (`industry_reports`, `industry_stats`, `industry_report_jobs` for the failed-attempt block) | public (auth off) | public | data (slowapi `industry_read` 60/min IP) | FEAT-003 slice 5. The `latest` surface follows `INDUSTRY_ANALYSIS_ACCESS` (owner decision 1, default `public`); flipping that setting to `pro` moves this row and `/companies` to Pro through `industry_report_store.access_policy()`, which `auth/policy.py` resolves per call. Never generates — a page view cannot queue work; 404 `no_report` carries `last_attempt`, and a failed refresh leaves the previous edition in place marked `stale` |
| GET | `/api/industries/{code}/companies` | none yet (FEAT-003 slice 6) | DB-only (one membership join over `company_industry_classifications` × `companies`, one `industry_stats` read; sub-industry names come from the cached node index) | public (auth off) | public | data (slowapi `industry_read` 60/min IP) | FEAT-003 slice 5. Three queries, never one per constituent. Membership (`count`) and price coverage (`n_priced`) are separate fields, and every unpriced name carries a reason. Classification rows show their source label, author and as-of; security assignments taken from the research map are economic research examples, not licensed issuer GICS mapping |
| GET | `/api/industries/{code}/history` | none yet (FEAT-003 slice 6) | DB-only (`industry_reports` metadata, plus one COUNT so a capped list reports what it dropped) | public (auth off) | pro | data (slowapi `industry_read` 60/min IP) | FEAT-003 slice 5. Prior editions, newest first, no payloads. Pro whenever `AUTH_ENABLED` is on (feature `industry_analysis`) |
| GET | `/api/industries/{code}/changes` | none yet (FEAT-003 slice 6) | DB-only (two `industry_reports` rows and their `industry_stats`) | public (auth off) | pro | data (slowapi `industry_read` 60/min IP) | FEAT-003 slice 5. Pure arithmetic over the two stored statistics rows; a fact missing on either side is `null` with the side named, never differenced to zero. Pro (feature `industry_analysis`) |
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
| GET | `/api/admin/outcome-audit` | operations review | DB-only (one SELECT over all outcomes and snapshot metadata) | admin | admin | admin | Read-only, uncapped triage; no provider calls, evaluations, row mutations, or KPI changes. |
| GET | `/api/admin/market-data/plan` | operations | DB-only metadata | admin | admin | admin | All companies, benchmark and legacy memo symbols; minimum two years and older required memo dates. No full memo bodies. |
| GET | `/api/admin/market-data/coverage` | operations | DB-only prices and financial periods | admin | admin | admin | Full per-source date bounds and missing-session/period identities; no provider calls. |
| GET | `/api/admin/market-data/prices` | operations | DB-only daily prices | admin | admin | admin | One provider and close basis per returned series; no live fetch. |
| GET | `/api/admin/market-data/backfill-status` | operations | DB-only last company import result | admin | admin | admin | Complete saved report for one target, including interruptions and incomplete data. |
| POST | `/api/admin/market-data/backfill` | operations | price and fundamental provider APIs; durable DB upserts | admin | admin | admin | One planned ticker per request, atomic claim; no filings, transcripts, LLMs, memo generation, evaluation or historical outcome writes. |
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
| POST | `/api/admin/samples/rebuild` | none | DB-only (control row in `public_samples`; the worker builds on its next poll, ≤10 min) | admin | admin | admin | tickers ⊆ `SAMPLE_TICKERS` else 422; merges repeated requests |
| GET | `/api/admin/abuse-telemetry` | none | DB-only (`analytics_events`, `users.bootstrap_ip_hash`, `ui_logs`) | admin | admin | admin | FEAT-002 phase 6: 429s by scope/plan/route, trial creations per IP hash, share of authenticated requests refused with 429, trailing 24h |
| GET | `/api/admin/unit-economics` | none | DB-only (`llm_call_logs`: a newest-first SELECT capped at 20,000 rows plus a bounded boundary lookback, both on the `generated_at` index; and one COUNT that is capped only by the window, so its cost grows with the rows in the window — `feature` is unindexed — which is what the 90-day `window_days` ceiling bounds) | admin | admin | admin | Observed cost per metered operation (research_run, pm_chat, chart_commentary, industry_report) — median, p90 and the sample behind each — priced with `llm_metrics.estimate_cost_usd` and projected against the `auth/features.py` allowances. Read-only; a figure with too thin a sample is null with the count, never a number. Makes no provider or model call — `scripts/audit_unit_costs.py` is the tool that spends |
| POST | `/api/admin/scorecard/refresh` | none | DB-only in-request (enqueues a `scorecard_runs` row; the worker's `scorecard_loop` drains it at its next interval tick, minutes; the daily scoring is gated to 03:45 UTC inside that loop) | admin | admin | admin | Phase 6 C. 202; coalesces on `(version_key, as_of, kind)` |
| POST | `/api/admin/scorecard/evaluate` | none | DB-only in-request (enqueues an `evaluate` run; numpy work runs in the worker; Ken French monthly series via `provider_cache`) | admin | admin | admin | Phase 6 C. 202 |
| POST | `/api/admin/scorecard/backfill` | none | DB-only in-request (enqueues `pit_prepare` + one `backfill` run per month end; provider reads happen in the worker's `pit_prepare` via the cached 252-day series) | admin | admin | admin | Phase 6 C. 202; skips month ends that already have a succeeded run |
| GET | `/api/admin/scorecard/disagreements` | none | DB-only in-request (one bounded SELECT over `scorecard_disagreements`, default `status=open`, `limit` ≤ 500) | admin | admin | admin | Phase 6 D. Memo-vs-scorecard disagreement queue, newest first |
| POST | `/api/admin/scorecard/disagreements/{disagreement_id}/dismiss` | none | DB-only in-request (one row UPDATE) | admin | admin | admin | Phase 6 D. Idempotent; 404 unknown id; a `reviewed` row is left as it is |
| POST | `/api/admin/industries/taxonomy/import` | none | filesystem (the one bundled knowledge JSON) + DB (`gics_taxonomy_versions`, `gics_nodes`) | admin | admin | admin (slowapi `industry_admin` 10/min) | FEAT-003 slice 5. Idempotent by node checksum; the same version key with a different structure is a 409, never a silent rewrite of a version that reports and classifications already point at by id. 503 `knowledge_base_missing` when the JSON is absent — a missing knowledge base is a build defect, not an empty taxonomy |
| POST | `/api/admin/industries/classify` | none | DB-only (two bulk reads, one bulk insert over `companies` → `company_industry_classifications`) | admin | admin | admin (slowapi `industry_admin` 10/min) | FEAT-003 slice 5. No per-ticker loop, so it is safe on the web process. Returns capped `changed` / `unmapped_labels` lists, each with its total and the number dropped |
| POST | `/api/admin/industries/reports/regenerate` | none | DB-only in-request (enqueues `industry_report_jobs`; the worker's drainer does the generation, LLM included) | admin | admin | admin (slowapi `industry_admin` 10/min) | FEAT-003 slice 5. 202, the same job path the Sunday cron uses. Unknown codes are a 404 here rather than three failed attempts in the worker. While the drainer module is not on the build it answers 503 `report_queue_unavailable` and names what is missing instead of pretending a job was queued; 409 `taxonomy_drift` unless `accept_stale_taxonomy=true` |
| GET | `/api/admin/industries/jobs` | none | DB-only (one capped page of `industry_report_jobs` plus a grouped COUNT over the whole filtered queue) | admin | admin | admin | FEAT-003 slice 5. Status counts come from SQL, not from the returned page, so a capped list never understates the backlog |
| POST | `/api/billing/checkout` | none yet (S5 `Account.tsx` / pricing CTA → `api.checkout`) | Stripe (httpx: `POST /customers` once per user, `POST /checkout/sessions`; idempotency keys on both) | public (auth off → 404 `feature_disabled`) | free | checkout (10/hour/user) | 404 `feature_disabled` while `BILLING_ENABLED=false`; 403 `email_unverified`; 409 `already_subscribed` when an active/trialing/past_due row exists; 503 `billing_unavailable` when Stripe is unconfigured/unreachable. Mid-trial (≥48h left) passes `subscription_data[trial_end]=users.trial_ends_at` so Stripe starts `trialing` and the first charge lands when the local trial ends; otherwise `billing_starts:"now"`. The redirect is never proof of payment. |
| POST | `/api/billing/portal` | none yet (S5 `Account.tsx` "Manage billing") | Stripe (httpx: `POST /billing_portal/sessions`) | public (auth off → 404) | free | checkout (10/hour/user) | 409 `no_billing_account` before any checkout created a customer; return URL `/app/account` |
| POST | `/api/billing/reconcile` | none yet (S5 `BillingSuccess.tsx` after polling `/api/me`) | Stripe (httpx: `GET /subscriptions?customer=…&status=all`) | public (auth off → 404) | free | reconcile (5/hour/user) | applies fetched objects through the same state machine as webhooks, as of *now*; a fetch error returns `reconcile.ok=false` and changes nothing (never downgrades); returns the `/api/me` payload |
| POST | `/api/billing/webhook` | Stripe only | none (`Stripe-Signature` verified locally: HMAC-SHA256 over `t.body`, 300s tolerance, constant-time) | public | public | none (`@limiter.exempt`; exempt from the customer policy — the signature is the auth) | raw body ≤256 KB (413); 400 bad signature / non-event; 503 when `STRIPE_WEBHOOK_SECRET` unset (Stripe retries); 500 only on a DB write failure (retry is idempotent via `billing_webhook_events.stripe_event_id`); 200 `{received, outcome}` for `applied` / `duplicate` / `ignored_stale` / `ignored_unhandled` / `error` (ownership mismatch → no grant). Never writes `users.trial_*`. |
| GET | `/api/admin/billing/users/{external_id}` | none | DB-only (`users`, `subscriptions`, `admin_overrides`, last 20 `usage_events`) | admin | admin | admin | support view; no email (none is stored) |
| POST | `/api/admin/billing/overrides` | none | DB-only (`admin_overrides`; `trial_reset` also rewrites `users.trial_*` — an operator action, the one writer besides bootstrap) | admin | admin | admin | kinds `plan` (value `pro`), `quota` (a metered feature + `unlimited`/integer; replaces that feature's monthly limit in `entitlements.authorize` and on `/api/me` while active), `trial_reset`, `suspend`; `reason` mandatory |

### Routes FEAT-002 still adds (not in `app.openapi()` yet)

None — every FEAT-002 route is live and in the table above. (S4 landed the
billing routes; keep this heading so a future addition has a home.)

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
