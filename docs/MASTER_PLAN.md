# MarketMosaic — Master Implementation Plan

A consolidated roadmap covering the seven specs we're committing to, the
deferred Phase G–K items from README §20, and the remaining open items from
the multi-agent migration. Organized as six dependency-ordered waves so
individual PRs land in a buildable sequence.

> **Source of truth.** This file is the canonical roadmap. README §20
> tracks "deferred" items and is updated as each ships; this plan is the
> implementation-level sequencing.

---

## 1. Where we are today

### Shipped & merged on `main`
- **Phase 1–6** of the multi-agent migration: snapshot cache + lineage,
  per-agent models, Agents SDK runtime (real package + shim fallback),
  Gemini wrappers, monitoring scaffolding, sector cross-talk, safe-runner
  per-agent failure isolation.
- Frontend: cross-sector chips, macro regime banner, hot-news side panel.
- Real-token cost capture wired into cache writes.
- Long-term agent memory: `memory/companies/<TICKER>.md` + `memory/sectors/<slug>.md`
  with delta-only writes, condense-on-cap, cross-company patterns.
- Universe tiering (`data_only` / `auto_analysis` / `analyzed_on_demand`),
  versioned `MemoSnapshot` store with `parent_version` lineage.
- Per-agent model wiring fully threaded into the legacy graph.
- LLM-driven macro agent.
- Two-file env split: `config.env` (committed, model assignments + flags) +
  `.env` (gitignored, secrets).
- `ENABLE_LIVE_DATA=true` as production default; tests force-back to demo
  via conftest for determinism.

### End-state vision

Every US-listed equity + ADR has provider data ingested. A curated tier-1
list (currently 17 names) gets full memos auto-refreshed on EDGAR /
earnings deltas, with material news triggering incremental patches that
don't re-run the whole graph. Each memo is the synthesis of seven
specialist agents (sector w/ integrated bull/bear, earnings, filing,
valuation, comps, macro, technical) coordinated by a PM agent and
challenged by a cross-family critic. Every ticker has a long-term
memory file that grows with each delta event, condenses old entries on
cap, and records realized outcomes vs ratings so future calls can
reference past mistakes. DCF models are persisted versioned objects that
*roll forward* with each new earnings period rather than being rebuilt
from scratch. Backtest support via `as_of_date` lets every memo be
reproduced for any historical point. Operations cost is observable
per-call, per-agent, per-ticker.

---

## 2. Workstream summary

| Wave | Theme | PRs | Dev-days |
|---|---|---|---|
| **1** | Observability + on-demand UI + historical anchor | 3 PRs | ~5 |
| **2** | Financial-history depth | 1 PR | ~3 |
| **3** | Research-depth analytics | 4 PRs | ~6 |
| **4** | Eval + integration testing | 2 PRs | ~3 |
| **5** | Persistent DCF + incremental updates | 2 PRs | ~5 |
| **6** | Resilience + polish | 4 PRs | ~4 |
| **Total** | | **16 PRs** | **~26 dev-days** |

---

## 3. Per-wave detail

### Wave 1 — Observability, on-demand, historical anchor

Three small foundational PRs that unlock most downstream waves. All
independent of each other; ship in parallel.

#### PR-W1A — LLM call trace logging (Spec 6)

Every provider call leaves a `LLMCallLog` row tagged with `(run_id,
agent_name, model, tokens_in, tokens_out, duration_ms, success)`.
Foundational for cost accounting + slow-call audit + debugging "why did
this memo cost $0.40?". DB-only; no UI surfacing.

**Touches:** `models.py` (new `LLMCallLog`); `agents/llm.py::_record_usage`
(extend to write a row); `services/llm_metrics.py` (aggregation queries);
`api/routes_admin.py` (`/api/admin/llm-metrics?run_id=X`);
`scripts/llm_cost_report.py`; `contextvars` for agent-name propagation.

**Effort:** ~1 day.
**Exit criteria:** running a memo produces ≥10 LLMCallLog rows with
correct agent attribution. Cost-per-run query returns sensible
breakdowns. CLI cost report renders.

#### PR-W1B — On-demand "Analyze this stock" UI (Phase J)

UI affordance for the `data_only` universe tier. Backend already
supports `?ondemand=true`. UI just needs the button + clarity on the wait.

**Touches:** `frontend/src/pages/Research.tsx`, `MemoCard.tsx`, optional
new `frontend/src/components/AnalyzeStockGate.tsx` for the gating UX.

**UX:**
- Tier badges next to ticker list: `auto` / `cached` / `data-only`.
- Selecting a `data_only` ticker shows: "MarketMosaic hasn't analyzed
  this stock yet. Generating a memo takes ~30s and uses real provider
  data." → button: **Analyze**.
- Button hits `GET /api/stocks/{ticker}/memo?ondemand=true` with a
  loading state.

**Effort:** ~½ day.

#### PR-W1C — As-of-date selector (Spec 5)

Every memo can be reproduced for any historical date. All providers
respect `as_of`; cache key segregates live vs backtest; memory writes
skip on backtest runs.

**Touches:** `BaseProvider` interface gains `as_of: Optional[date]` on
every method; each provider class; `services/data_service.py` capability
chain; `cache/snapshots.py` (key segregation); `agents/graph.py::run_stock_memo`
(new param + skip memory writes when set); `MemoSnapshot.as_of_date` column;
API endpoint accepts `?as_of=2025-09-15`.

**Effort:** ~2-3 days. Lots of small wiring touchpoints.

**Exit criteria:** historical memo for NVDA at `as_of=2025-09-15`
returns sources_used with no post-2025-09-15 filings. Live-mode `MSFT`
cache and `MSFT@2025-09-15` cache don't collide. Memory file unchanged
after a backtest run.

---

### Wave 2 — Financial-history depth (Phase G)

#### PR-W2 — FinancialPeriod / FilingDoc / EarningsTranscript ORM + backfill

Three new tables persisting structured statements, full-text filings,
and structured transcripts. Backfill job seeds tier-1 names with whatever
depth the provider exposes.

**ORM additions:**

```python
class FinancialPeriod(Base):
    """One row per (ticker, period, statement_line). Long format so 10y
    of revenue is one SELECT."""
    id: int (PK); ticker: str (indexed); period: str (e.g. "2024Q4")
    period_end: date (indexed); fiscal_year: int; fiscal_quarter: int
    statement: Literal["income", "balance", "cash"]
    line_item: str; value: float
    currency: str = "USD"; source: str
    fetched_at: datetime
    # unique (ticker, period, statement, line_item)

class FilingDoc(Base):
    id: int (PK); ticker: str (indexed); accession_number: str (unique)
    filing_type: Literal["10-K", "10-Q", "8-K"]
    filing_date: date (indexed); period_end: date
    raw_text: Text
    sections: dict (JSON)   # {risk_factors, mda, business, ...}
    word_count: int
    fetched_at: datetime

class EarningsTranscript(Base):
    id: int (PK); ticker: str (indexed); period: str
    fiscal_year: int; fiscal_quarter: int; call_date: date
    blocks: list (JSON)     # [{speaker, role, segment, text}]
    full_text: Text
    word_count: int
    fetched_at: datetime
    # unique (ticker, period)
```

**Service:** `backend/app/services/history_service.py` exposing
`get_financial_history`, `get_filing_text`, `get_recent_filings`,
`get_transcript`, `backfill_ticker`.

**Backfill job:** `backend/app/monitoring/history_backfill.py`. Nightly
under `ENABLE_MONITORING=true`. Idempotent — uses `(ticker, period)`
unique constraints.

**Effort:** ~3 days.
**Exit criteria:** tier-1 NVDA backfill produces ≥40 quarters of
FinancialPeriod rows, ≥12 filings, ≥10 transcripts.
`get_financial_history("NVDA", ["revenue", "operating_income"], 40)`
returns structured ten-year series.

---

### Wave 3 — Research-depth analytics

Four PRs that materially upgrade analytical surface. PRs are independent;
ship in any order.

#### PR-W3A — Sector-integrated bull/bear (Spec 1)

Sector analyst produces both bear (first) + bull case + key disagreement
+ falsifiable tests + first-pass synthesis. PM uses it as a prior, not
a directive.

**Why this design over adversarial bull/bear:**
1. Sector analyst already has the deepest context (cohort, regime,
   peers); separate Bull/Bear agents would re-load that context twice.
2. MM is a research platform, not a trading desk. Surfacing both sides
   with synthesis is what readers want; adversarial transcripts are noisy.
3. Cost: sector-integrated keeps it at 1 enriched call vs ~4 for two
   debate rounds × two sides.

**Bias mitigations baked in:**
1. Force order: bear case FIRST. Hardest to write; doing it first with
   full attention prevents hand-waving at the end.
2. Require a falsifiable test on each side. If the analyst can't
   articulate "this side is wrong if X is observed" for the bear, the
   bull case isn't trustworthy either.

**Schema additions:**

```python
class FalsifiableTest(BaseModel):
    statement: str
    invalidates_side: Literal["bull", "bear"]

class BullBearAnalysis(BaseModel):
    bull_case: BullBearCase
    bear_case: BullBearCase
    key_disagreement: str
    falsifiable_tests: List[FalsifiableTest]
    sector_synthesis: str
    sector_lean: Literal["bull", "bear", "balanced"]
```

Stored under `sector_agent_view.data["bull_bear_analysis"]`. PM synthesis
prompt is updated to read sector synthesis as a prior + decide whether
other findings outvote it.

**Touches:** `prompts.py`, `schemas.py`, `sector_agents.py`, `graph.py`
(replace `_bull_case`/`_bear_case` template helpers), `MemoCard.tsx`
(render `key_disagreement` + falsifiable tests prominently).

**Effort:** ~1 day.

#### PR-W3B — Technical Analyst (Spec 3)

New specialist computing SMA/EMA/RSI/MACD/Bollinger/VWMA on 252+ days of
OHLCV. Pure-math indicators in `app/finance/technicals.py`; LLM narrative
pass in `agents/technical_agent.py`. Frames technicals as "positioning
signals in support of the long-term thesis", NOT as standalone trade
signals.

**Schema:**

```python
class TechnicalSignals(BaseModel):
    last_price: float; last_date: str
    sma_50: float; sma_200: float; sma_50_above_200: bool
    ema_10: float; ema_20: float
    rsi_14: float
    macd_line: float; macd_signal: float; macd_histogram: float
    bb_upper: float; bb_lower: float; bb_position: float
    vwma_20: float
    high_52w: float; low_52w: float; position_52w: float
    trend: Literal["up", "down", "sideways"]
    momentum: Literal["positive", "negative", "neutral"]
    notes: List[str]
```

`StockMemoOut.technical_agent_view: AgentFinding`. Model:
`OPENAI_TOOL_MODEL`. Tests cover indicator math properties (RSI bounded
0-100, MACD line - signal = histogram, SMA50 latest = mean of last 50).

**Effort:** ~1.5 days.

#### PR-W3C — Drill-down agent reports (Spec 7)

`AgentFinding.long_form_report: Optional[str]` (markdown). Each agent
prompt extended to emit a 6-8 paragraph report. Deterministic templated
fallback per agent. UI tab/drawer for full-report view.

**Token budget:** ~500-1000 extra output per agent × 7 agents =
+3500-7000 tokens per memo. At gpt-5.4 pricing roughly $0.02-0.04/memo.
Optionally gated by `ENABLE_LONG_FORM_REPORTS=true` flag.

**Frontend:** new `AgentReportDrawer.tsx`; "Read full report ▾" per tile;
top-of-memo "View all reports" tab.

**Effort:** ~1.5 days.

#### PR-W3D — Memory hooks for filings/transcripts (Phase K)

Depends on PR-W2. After new filings/transcripts land, the reflection
step extracts structured facts (segment performance, guidance changes,
capex commentary, M&A, leadership changes) and appends them to the
company memory file as structured trigger entries.

**Touches:** `agents/reflection_agent.py` (new
`_extract_structured_facts(filing_doc) -> Dict`); `memory/longterm.py`
(MemoryEntry gains optional `structured_facts: Dict`); fact-extraction
prompt; tests verifying extraction quality on demo filings.

**Effort:** ~2 days.

---

### Wave 4 — Eval + integration testing

#### PR-W4A — Realized outcome tracking (Spec 2)

`MemoOutcome` ORM table linked to `MemoSnapshot`; daily
`evaluate_all_due()` job computes forward returns at 30/90/180/365 days
vs SPY. Reflection writes outcome entries to company memory. API +
admin endpoint surface track-record stats.

**ORM:**

```python
class MemoOutcome(Base):
    id: int (PK)
    memo_snapshot_id: int (FK → memo_snapshots.id, indexed)
    ticker: str (indexed)
    rating_at_memo: str
    confidence_at_memo: float
    price_at_memo: float
    horizon_days: int           # 30 / 90 / 180 / 365
    evaluated_at: datetime
    forward_return: float
    benchmark_return: float
    alpha: float
    thesis_held: bool
    note: str
```

**Memory integration:** new outcome entry per (memo, horizon) appended to
`memory/companies/<TICKER>.md` with `trigger="outcome:90d"` etc. Becomes
trigger-eligible context for the *next* sector run on that ticker.

**Depends on:** PR-W1C (as-of-date) for clean backtest evaluation;
forward-return-only mode works without it.

**Effort:** ~1.5 days.

#### PR-W4B — Live integration tests (§20.3)

Opt-in `RUN_LIVE_TESTS=1` test path that exercises real EDGAR / Gemini /
Anthropic / FMP. Skipped by default. CI runs nightly with secrets.

**Effort:** ~1.5 days.

---

### Wave 5 — Persistent DCF + incremental updates

#### PR-W5A — Persistent DCF + LLM assumption updater (Phase H)

`DCFModel` ORM table — versioned with `parent_version`, lineage just
like `MemoSnapshot`. `dcf_updater.py` runs on each new earnings period:

1. Roll forward year 1 → "year 0 actual", shift the explicit forecast.
2. LLM compares prior assumptions vs actual; proposes adjustments
   (revenue growth, op margin, capex %, terminal growth, WACC).
3. Engine re-runs; new version stored as `v(N+1)` referencing v(N).

LLM-driven (no deterministic guard rails — assumption-change log shows
reasoning for review). Validation gate during initial rollout: log every
proposed change with rationale; v1 limits delta to ±20% of prior
assumption per cycle; manual review for first 10 tickers before fully
autonomous.

**Depends on:** PR-W2 (needs FinancialPeriod history to anchor
assumptions; needs EarningsTranscript for guidance reads).

**Effort:** ~3 days.

#### PR-W5B — Update orchestrator + news-impact-agent (Phase I)

Subscribes monitoring loops to memo refresh logic:

- New filing/earnings → enqueue `full_reanalysis(ticker)`.
- Material news → `news_impact_agent` reads prior memo + alert →
  returns `{material: bool, affected_fields: [...], delta_summary: str}`.
  If material, creates `v(N+1)` inheriting from `v(N)` with
  rating/confidence/risks patched. Critic skipped (per agreed design),
  `revision_log` flagged with `critic_skipped: true`.
- Per-ticker FIFO queue prevents two events on the same ticker racing.
- Patch frequency cap: max 2 patches per day per ticker.

**News-impact-agent model:** Anthropic Haiku 4.5 (cheap, cross-family
with PM's OpenAI synthesis).

**Depends on:** Versioned memo store (already shipped), PR-W5A (DCF
persistence — patches that touch valuation should leave DCF version
stable, only update memo's `dcf_summary` if explicitly affected).

**Effort:** ~2 days.

---

### Wave 6 — Resilience + polish

Independent; ship as time permits.

#### PR-W6A — Checkpoint resume (Spec 4)

`MemoRunCheckpoint` ORM table; `@checkpointed("step_name")` decorator on
each major step in `run_stock_memo`; `run_id` propagated via
`contextvars`; daily GC of expired checkpoints. TTL 24 hours.

**Effort:** ~1.5 days.

#### PR-W6B — Cohort-similarity invalidation tightening (§20.5)

Hash specific KPI inputs (revenue / op income / capex / shares) into
the warm snapshot's `sources_used` so an irrelevant peer-side 10-K
doesn't trigger sector_warm recompute.

**Effort:** ~½ day.

#### PR-W6C — News allow-list governance to JSON config (§20.6)

Move `_ALLOWED_DOMAINS` / `_BLOCKED_DOMAINS` from `news_agent.py`
constants into `app/data/news_domains.json`.

**Effort:** ~½ day.

#### PR-W6D — Schema-version migration path (§20.8)

`backend/app/cache/migrations.py` registering per-`(kind, from_version,
to_version)` upgrader functions. Runs on read when a stored snapshot's
version doesn't match the current code's version.

**Effort:** ~1 day.

---

## 4. Critical path / sequencing

```
Wave 0 (merge in-flight PRs)  [DONE]
    │
    ├──── Wave 1A (LLM trace)        ────┐
    ├──── Wave 1B (on-demand UI)     ────┤   No deps; parallel
    └──── Wave 1C (as-of-date)       ────┘
                │
                └──── Wave 2 (financial history)  ──┐
                            │                       │
                            ├── Wave 3D (mem hooks) │
                            └── Wave 5A (DCF)       │
                                                    │
        Wave 3A (bull/bear)  ──────────── (parallel; no deps)
        Wave 3B (technical)  ──────────── (parallel; no deps)
        Wave 3C (drill-down) ──────────── (parallel; no deps)
                                                    │
                                                    ▼
                                    Wave 5B (update orchestrator)
                                                    │
                                                    ▼
                                    Wave 4A (outcome tracking)
                                                    │
                                                    ▼
                                    Wave 4B (live integration tests)
                                                    │
                                                    ▼
                                    Wave 6 (resilience + polish, opportunistic)
```

**Parallelizable PRs:** all of Wave 1, all of Wave 3 (3D after Wave 2).
**Critical path:** Wave 1C → Wave 2 → Wave 5A → Wave 5B → Wave 4A.
**Total dev-days on critical path:** ~14. **Total work if every PR is
single-threaded:** ~26.

---

## 5. Decisions logged (already agreed)

These were settled during planning; recording so future contributors
don't re-debate.

- **Bull/bear architecture:** sector-integrated, NOT separate adversarial
  agents. Bias mitigated via prompt structure (bear-first + falsifiable
  tests). Devil's-advocate amplifier deferred — only add if memos start
  feeling one-sided in practice.
- **Technical analyst rating influence:** does NOT override the rating.
  Surfaces as separate "technical positioning" finding; rating remains
  fundamentals + valuation + thesis-driven.
- **DCF assumption updater:** LLM-driven, no deterministic guard rails
  beyond the ±20% delta cap. Assumption-change log captures reasoning
  for manual review during initial rollout.
- **Critic on incremental patches:** skipped. `revision_log` flagged
  with `critic_skipped: true` so reviewers know.
- **News-impact-agent model:** Anthropic Haiku 4.5 (cross-family with
  PM's OpenAI synthesis).
- **Patch frequency cap:** max 2 patches per ticker per day; further
  material news queues but doesn't fire until next refresh.
- **As-of-date depth limit:** bound by demo dataset's 4 years until FMP
  Ultimate's deeper history is wired (Wave 2).
- **LLM trace retention:** 90 days then GC. Useful for monthly cost
  reports without bloating SQLite.

---

## 6. Decisions still open

These need a call before the relevant wave kicks off.

- **Outcome tracker reflection cadence.** Default suggestion: only
  generate prose reflection for 90d / 365d horizons; 1m / 3m surfaced
  numerically but no narrative entry. Confirm before Wave 4A.
- **Long-form reports rollout.** Ship enabled by default or behind
  `ENABLE_LONG_FORM_REPORTS=true`? Token cost is small (~$0.02-0.04 per
  memo) but adds noticeable latency. Default-on recommended; revisit
  if smoke runtime spikes.
- **Backtest UI surface.** Add a date picker to `/research` for as-of
  runs, or keep it API-only behind `?as_of=` for power users? Adding
  to UI increases support surface. Recommended: API-only for v1.

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Wave 2 backfill blows past FMP rate limits | Med | High | Tier-1 only first (17 names × 40 quarters = 680 calls; well under FMP Ultimate's daily cap). Stagger over 2 hours. |
| Bull/bear sector prompt produces lopsided output | Med | Med | Structured 5-block format forces both sides; test with 5 tier-1 tickers, eyeball output before merging Wave 3A. |
| LLM-driven DCF updater proposes wild assumption swings | Med | High | Log every proposed change with rationale; v1 limits delta to ±20% per cycle; manual review for first 10 tickers. |
| Update orchestrator floods queue on volatile day | Low | Med | Per-ticker FIFO + max-2-patches-per-day cap. |
| Checkpoint resume serialization bugs | Med | Low | Tests verify identical output between fresh-run and resume-from-step-N for every step. |
| Outcome tracker noisy on 1m horizons | Med | Low | Default reflection only writes for 90d / 365d horizons. |

---

## 8. Already shipped / superseded

- §20.1 SDK package upgrade — DONE in PR #6.
- §20.2 Live token cost — superseded by Wave 1A (LLM trace logging).
- §20.4 Frontend wiring of cross-sector / macro / news fields — DONE in PR #2.
- §20.7 SDK-runtime LLM wiring — DONE in PR #6.

## 9. Out of scope (for now)

- Vector RAG with pgvector for filings/transcripts (alternative to BM25).
- Backtest *engine* (replay portfolios, Sharpe / drawdown). Spec 2 +
  README §18 evaluation lays the groundwork; the actual engine is a
  separate product surface.
- Multi-account / advisor features.
- Earnings preview mode (diff guidance vs consensus pre-print).
- Mobile UI.

---

## 10. Recommended next move

1. **Wave 1A (LLM trace logging)** — half-day, smallest possible
   foundational PR. Cost visibility for everything that follows.
2. **Wave 1B (on-demand UI) + Wave 1C (as-of-date)** in parallel.
3. **Wave 3A (sector bull/bear)** — biggest user-visible quality bump
   per dollar. Independent of Wave 1.

If you want to start with the flashiest single change instead of the
most foundational: skip directly to **Wave 3A**.
