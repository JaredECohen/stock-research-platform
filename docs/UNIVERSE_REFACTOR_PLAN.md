# Wave 9b — Universe refactor (kill demo, S&P 100 screener, custom screens)

The platform was built around a synthesized 33-ticker demo dataset.
Live providers (FMP, Alpha Vantage, FRED, SEC EDGAR) were bolted on
as a chain that falls back to demo on miss. That tradeoff made sense
during development; it doesn't anymore. Demo data is now actively
misleading: research runs return live numbers but DCF history reads
demo from `financial_periods`, the screener still defines the
universe via `COMPANY_PROFILES`, and `list_tickers()` hard-codes the
33-name set.

Wave 9b removes demo from the runtime entirely, makes any ticker
researchable on demand, defines a curated S&P 100 screener universe
that's pre-analyzed nightly, and adds factor-rank + rule-based
screening modes alongside the existing AI-first view.

## 1. Goals

1. **Demo retires from runtime.** `DemoProvider` and `COMPANY_PROFILES`
   become test fixtures only. Production never serves synthetic data.
2. **Any ticker researchable.** User types `PYPL` → backend lazy-loads
   profile + financials + filings + transcripts → runs agent graph →
   persists memo + DCF. Subsequent runs read from DB if cached data
   is still valid; force-refresh button bypasses cache.
3. **Curated S&P 100 screener universe.** Distinct concept from
   "researchable universe." Pre-loaded nightly so screening is instant.
4. **Three screener views**: AI-first (existing), single-factor rank
   (new), rule-based custom screen (new).
5. **Browser-session persistence** for custom screen state.

## 2. Decisions locked

These were the questions surfaced before kickoff. Recording the
answers so we don't relitigate.

| Question | Answer |
|---|---|
| Demo: keep or rip? | Keep, but only under `tests/` as a fixture. Zero runtime references. |
| S&P 100: static or auto-refresh? | Static `data/sp100.json`. Manual refresh later if index changes matter. |
| Lazy-research UX: eager or background? | Eager. User waits for backfill + agent run on first touch. Matches "research takes time" mental model. |
| Custom-screen metric vocabulary | ~15 standard ones for v1 (see §6.3). |
| Ticker researched but not in screener universe? | Memo/DCF saved per-ticker; does **not** appear in screener. Screener stays curated. |
| Cache invalidation policy | Three tiers (see §3). |

## 3. Cache architecture — three tiers

The big mental shift. Today there's one cache concept ("populated by
nightly backfill, read by some agents"). Going forward, three
explicit tiers based on data semantics.

### Tier A — Permanent / additive

Historical record. Append-only. No invalidation, ever.

| Table | Content | Refresh |
|---|---|---|
| `financial_periods` | Income/balance/cash by period | New rows arrive when new 10-Q/K lands |
| `filing_docs` | 10-K/Q text + sections | New rows on EDGAR poll |
| `earnings_transcripts` | Quarterly call transcripts | New rows after each call |
| `dcf_models` | Valuation per memo run | New rows per research run |
| `memo_snapshots` | Memo JSON per run | New rows per research run |

### Tier B — Event-driven (memos)

A memo doesn't expire on a clock. It expires when the inputs that
fed it changed enough to matter. Triggers for v1:

1. New 10-Q or 10-K filed for the ticker (caught by EDGAR poller →
   writes to `filing_docs` → fires invalidation hook)
2. New earnings transcript posted
3. User clicks "Re-run research"

Deferred for v2: material price move, high-impact news event. Both
are too noisy to start with.

Implementation: a `memo_freshness` view that joins `memo_snapshots`
with `filing_docs.filing_date` and `earnings_transcripts.call_date`.
Memo is "stale" iff `max(filing_date, call_date) > memo.created_at`.
The Research route checks this before returning a cached memo.

### Tier C — Short-TTL (transient snapshots)

"Now" data that's meaningless if stale. New tables introduced this
wave:

| Table | TTL | Notes |
|---|---|---|
| `cached_profiles` | 7 days | FMP `/profile`. Stable fields except market cap. |
| `cached_quotes` | 5 min market hours, ∞ off-hours | Latest price/volume |
| `cached_prices_eod` | ∞ for closed days | OHLCV per day |
| `cached_ratios` | 1 day | P/E, EV/EBITDA, margins, ROIC, debt/EBITDA |
| `cached_estimates` | 1 day | Sell-side consensus |
| `cached_news` | 1 hour | Article list per ticker |
| `cached_macro` | 1 day | FRED series values |

Schema: `(key TEXT PRIMARY KEY, payload_json JSON, fetched_at TIMESTAMP)`
plus capability-specific indexes where needed.

## 4. Phase 1 — Decouple demo from runtime

### 4.1 Move demo to test fixtures

- `backend/app/data/demo_dataset.py` → `backend/app/tests/fixtures/demo_dataset.py`
- `backend/app/data/demo_*.json` → `backend/app/tests/fixtures/demo/`
- `backend/app/providers/demo_provider.py` → `backend/app/tests/fixtures/demo_provider.py`
- All test imports updated.

### 4.2 Drop runtime references

- [data_service.py:264](../backend/app/services/data_service.py#L264)
  `list_tickers()` no longer queries `DemoProvider`. New behavior:
  return `companies` table contents (anything ever researched).
- [data_service.py:217-243](../backend/app/services/data_service.py#L217-L243)
  `_live_chain` returns providers with no demo fallback in the
  chain. On total miss, return `None` and let callers decide.
- [orchestrator.py:57](../backend/app/agents/orchestrator.py#L57)
  drop the `set(get_data_service().list_tickers())` universe gate.
  Any ticker the user submits is valid; the lazy-fetch flow
  (Phase 2c) handles introduction.
- `seed_demo_data.py` deleted. Replaced by `seed_universe.py` (Phase 3).

### 4.3 Test fixture wiring

- `tests/conftest.py` exposes a `demo_provider` fixture that injects
  `DemoProvider` into `data_service` for the duration of the test.
- Existing tests that implicitly relied on demo data switch to the
  fixture.

## 5. Phase 2 — Read-through cache + lazy research

### 5.1 New cache tables (Phase 2a)

Seven tables per §3 Tier C. SQLAlchemy models + `Base.metadata.create_all`
auto-migration (the project's existing pattern; no Alembic).

### 5.2 data_service becomes read-through (Phase 2b)

Per-capability wrapper:

```python
def get_company_profile(self, ticker: str, *, force_refresh: bool = False):
    if not force_refresh:
        cached = read_cache("profile", ticker, ttl_days=7)
        if cached:
            return cached
    live = self._try_chain("profile", "get_company_profile", ticker)
    if live:
        write_cache("profile", ticker, live)
        return live
    # serve stale rather than fail
    return read_cache("profile", ticker, ttl_days=None)
```

Same shape for prices, ratios, estimates, news, macro. Capabilities
backed by Tier A tables (`financials`, `filings`, `transcripts`)
already have their own DB-backed read path via `history_service` —
those wrappers will check `financial_periods` etc. before falling
through to FMP.

### 5.3 Lazy ticker introduction (Phase 2c)

`/api/research` route flow:

1. User submits ticker `T`.
2. If `T` not in `companies` table:
   - Hit FMP `/profile` (no fallback — fail loudly if unknown ticker)
   - Insert `companies` row. `universe_tier = "analyzed_on_demand"`.
3. Run `history_service.backfill_ticker(T)` — pulls 5yr financials,
   filings, transcripts. Idempotent.
4. Run agent graph (already working).
5. `save_memo` + `save_dcf_model` at the end.
6. Return memo to user.

Frontend: spinner with progress states (`Loading profile…`,
`Loading 5yr financials…`, `Running agent diligence…`,
`Generating valuation…`).

### 5.4 Memo invalidation (Phase 2d)

Per §3 Tier B. Memo read path:

```python
def get_or_run_memo(ticker: str, force_refresh: bool = False):
    cached = latest_memo_snapshot(ticker)
    if cached and not force_refresh and is_memo_fresh(cached):
        return cached
    return run_research_pipeline(ticker)

def is_memo_fresh(memo) -> bool:
    last_filing = max_filing_date(memo.ticker)
    last_transcript = max_transcript_date(memo.ticker)
    invalidation = max(last_filing, last_transcript)
    return invalidation <= memo.created_at
```

## 6. Phase 3 — S&P 100 universe

### 6.1 Static list

`backend/app/data/sp100.json` with the 100 tickers. Source: official
S&P 100 constituents.

### 6.2 New seeder

`backend/app/seed_universe.py`:

1. Read `sp100.json`.
2. For each ticker, hit FMP `/profile` to populate `companies` row
   (no synthetic data).
3. Mark `universe_tier = "auto_analysis"`.
4. Trigger `history_service.backfill_ticker` for each.
5. Run `seed_screener_scores` after backfill completes to populate
   the `screener_scores` and `screener_metrics` (Phase 4a) tables.

Cold-load: ~100 tickers × ~6 FMP endpoints = ~600 calls. ~3-5 min
within FMP rate limits. Run once via CLI; nightly job
([history_backfill.py](../backend/app/monitoring/history_backfill.py))
keeps it fresh.

### 6.3 Screener metric vocabulary (locked)

For Phase 4a `screener_metrics` columns:

```
pe_ttm, forward_pe, peg, ev_ebitda, ev_revenue,
gross_margin, op_margin, fcf_margin, roic, roe,
debt_to_ebitda, revenue_growth_yoy, dividend_yield,
market_cap, beta
```

15 metrics. All derivable from FMP `/ratios` + `/key-metrics` +
`/income-statement` + `/profile`. Sourced once nightly alongside
`seed_screener_scores`.

## 7. Phase 4 — Three screener views

### 7.1 AI-first (existing)

No changes. `GET /api/screener?theme=…&sector=…&limit=…` continues
to work.

### 7.2 Single-factor rank (Phase 4b)

Extend the existing endpoint:

```
GET /api/screener?sort_by=quality|growth|valuation|earnings_momentum
                  |risk|catalyst|macro_fit&order=desc
```

Backend: just an ORDER BY on `screener_scores`. UI: clickable column
headers on the existing table.

### 7.3 Custom rule-based screen (Phase 4b)

```
POST /api/screener/custom
{
  "rules": [
    {"metric": "gross_margin", "op": ">", "value": 0.7},
    {"metric": "pe_ttm", "op": "<", "value": 30}
  ],
  "sort_by": "market_cap",
  "order": "desc",
  "limit": 50
}
```

Backend: WHERE clause assembly against `screener_metrics`. Operators:
`>`, `<`, `>=`, `<=`, `=`, `between`. Tickers limited to the S&P 100
universe (no leakage to lazy-researched tickers; per locked decision).

### 7.4 Frontend (Phase 4c)

[Screener.tsx](../frontend/src/pages/Screener.tsx) currently 133
lines, single view. Refactor into tabbed structure:

```
<Tabs>
  <Tab label="AI-First">           // existing UI
  <Tab label="Factor Rank">        // existing table + click-to-sort columns
  <Tab label="Custom Screen">      // rule builder + result table
</Tabs>
```

Custom Screen UI sketch:

- Rule list (add / remove rows): `[metric ▾] [op ▾] [value]`
- Sort-by dropdown
- "Run" button → POST → results table
- "Clear" / "Reset" buttons

## 8. Phase 5 — localStorage persistence

Pure frontend. Keys:

```
screener:tab          // "ai" | "factor" | "custom"
screener:custom:v1    // serialized rule list + sort + order
screener:factor:v1    // selected sort column + direction
```

Hydrate on page mount. Save on every change (debounced 250ms).
Versioned keys so we can break the schema cleanly later.

## 9. Migration & rollout

### 9.1 DB migration

Two passes:

1. New tables (`cached_*`, `screener_metrics`) auto-create on next
   startup via SQLAlchemy `create_all`.
2. Existing 384 demo rows in `financial_periods` are **dropped**
   wholesale at the start of Phase 3 cold-load. Rationale: they're
   stamped `source=demo`, the values are synthetic, and S&P 100
   backfill replaces them with live FMP data.

### 9.2 Config flips

- `USE_DEMO_DATA` removed from `config.env` (no longer meaningful)
- `ENABLE_LIVE_DATA` removed (always live)
- `ENABLE_MONITORING` should flip to `true` so the nightly backfill
  + screener-score recompute job runs unattended

### 9.3 Test impact

Tests that imported `from app.data.demo_dataset import …` migrate
to `from app.tests.fixtures.demo_dataset import …`. Tests that
relied on `data_service` returning demo data wrap with the
`demo_provider` fixture from `conftest.py`.

## 10. Risks / things to watch

1. **FMP rate limits during S&P 100 cold-load.** ~600 calls in 3-5
   min should be within Starter-tier 300/min. Watch the
   `rate_limited` count from `history_backfill.run_once`.
2. **Tickers FMP doesn't cover.** S&P 100 should be 100% covered,
   but lazy-research can hit obscure tickers FMP rejects. Surface
   a clean error to the user, don't fall through to synthetic data.
3. **Memo freshness check correctness.** The `max(filing_date,
   transcript_date) > memo.created_at` rule is timezone-sensitive.
   Standardize on UTC for stored timestamps and the comparison.
4. **Backfill blocks the user on first research.** Eager UX = 30-90s
   wait on a brand-new ticker. If users complain, Phase 2c can
   move backfill to a background task with status polling later.

## 11. Sequencing

1. Phase 1 — Decouple demo (foundation, ~half-day)
2. Phase 3 — S&P 100 universe (depends on Phase 1, ~few hours)
3. Phase 2a/b — Read-through cache (~1 day)
4. Phase 2c — Lazy ticker introduction (~few hours)
5. Phase 2d — Memo invalidation (~few hours)
6. Phase 4a — `screener_metrics` snapshot (~few hours)
7. Phase 4b — Backend screener endpoints (~half-day)
8. Phase 4c — Frontend tabs + rule builder (~1 day)
9. Phase 5 — localStorage (~1 hour)

Estimated total: ~4-6 focused days. Each phase ships independently
behind feature flags where reasonable so the branch can be merged
incrementally rather than as one mega-PR.

## 12. Changelog

### 2026-05-04 — FMP `/stable/` migration (in-branch follow-up)

**What broke.** Mid-branch we discovered that every `/api/v3/...` request
returned HTTP 403 with `"Legacy Endpoint : Due to Legacy endpoints
being no longer supported"` — *not* a rate-limit, *not* a tier issue.
FMP retired both `/api/v3/` and `/api/v4/` on 2025-08-31. The keys are
fine; our URLs were stale.

**What we changed.** Rewrote
[`fmp_provider.py`](../backend/app/providers/fmp_provider.py) to hit the
current `/stable/` namespace. URL shape changed from
`/api/v3/{path}/{ticker}` to `/stable/{path}?symbol={ticker}`. Field
names mostly stable; the deltas:

- `mktCap` → `marketCap`
- `priceEarningsRatio` → `priceToEarningsRatio`
- `priceToFreeCashFlow` → `priceToFreeCashFlowRatio`
- `evToEbitda` → `evToEBITDA`
- `weightedAverageShsOutDil` (now camelCase consistent)
- profile no longer carries `sharesOutstanding` — pulled from
  `/stable/shares-float?symbol=…` and merged into the profile payload
  so consumers get the same shape they had before
- `/profile` now does include `sector`, `industry`, `country`, `ceo`
  (better than v3)

**New capabilities the migration unlocks.** With FMP back online:

- Live ratios (PE, EV/EBITDA, ROIC, gross/op/ebitda margins, debt/EBITDA)
- Analyst estimates → `forward_pe`, `peg`, eventually `price_target`
  (still NULL in `screener_metrics` until the metrics service consumes
  the estimates feed; the plumbing is ready)
- Real EOD prices for `cached_prices_eod`
- Earnings calendar with surprise %

**Re-run.** Cleared the provider cache, re-seeded all 100 S&P 100 names
with FMP `/stable/` profiles, re-ran `history_service.backfill_ticker`
across the universe, and re-snapshotted `screener_metrics`. Final
counts:

```
financial_periods : 25,824 rows / 100 tickers (all source=live, FMP)
filing_docs       :    993 rows / SEC EDGAR
screener_metrics  :    100 rows (full coverage)
```

Spot-checks: AAPL FY2025 revenue $416.2B (matches 10-K filed Oct 31).
NVDA ROIC 67%, gross margin 71%. AAPL P/E 36.7 / EV/EBITDA 28.8 — all
match published values.

**Bug fix in passing.** `history_service._ingest_statement_rows` would
hit `UNIQUE` constraint conflicts when a provider returned two rows for
the same fiscal period after a restatement (JNJ FY2023 was the
reproducer). Now dedupes by period before iterating; the later row
wins.
