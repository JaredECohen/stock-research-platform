# Autonomous Execution Prompt — MarketMosaic Wave 1–7

This document contains a self-contained instruction set for an agent
that will land the remaining 19 PRs in `docs/MASTER_PLAN.md` autonomously.

**How to use:** paste the section below titled "PROMPT" into a fresh
agent session in the repo root. The agent should read the master plan
and existing codebase, then execute waves in the documented dependency
order.

---

## PROMPT

You are an expert full-stack engineer working on **MarketMosaic**, a
multi-agent equity research platform. Repo path:
`/Users/jaredcohen/code/AgenticAI/stock-research-platform/`

### Mission

Land the 19 PRs documented in `docs/MASTER_PLAN.md` (Waves 1–7) in
dependency order. Run autonomously to completion. **DO NOT pause for
user input.** **DO NOT ask clarifying questions.** Make reasonable
implementation decisions and document them in code comments. When
genuinely blocked, produce a final report and stop cleanly.

### Existing state to build on (DO NOT BREAK)

- 88+ tests passing on `main`. `python -m pytest -q` from `backend/`
  must still pass after every PR. Tests force `ENABLE_LIVE_DATA=false`
  via `conftest.py` so they're zero-cost regardless of `.env`.
- `python -m scripts.smoke_test` from `backend/` — all 7 demo prompts
  must continue to pass.
- `cd frontend && npm run build` — must continue to succeed.
- Live mode is on (`ENABLE_LIVE_DATA=true` in `config.env`). FMP /
  OpenAI / Anthropic / FRED / SEC keys are configured. Vertex AI is
  configured for Gemini.
- 17-name tier-1 universe across all 11 GICS sectors.
- `safe_runner` produces typed fallback findings on agent failure.
- Long-term memory (`memory/companies/<TICKER>.md` + `memory/sectors/<slug>.md`)
  is wired with delta-only writes, condense-on-cap, cross-company patterns.
- Versioned memo store (`MemoSnapshot`) with `parent_version` lineage.
- Two-file env split: `config.env` (committed) + `.env` (gitignored).
- Real `openai-agents` SDK installed; `USE_AGENTS_SDK=true` activates it.

### PRs already merged (background context)

- Phase 1–6 multi-agent migration (cache, per-agent models, Gemini,
  monitoring, sector cross-talk, safe-runner).
- Frontend memo card with cross-sector chips, macro banner, hot-news panel.
- Real-token cost capture (Phase C).
- Long-term agent memory + cross-company learning.
- Universe tiering + versioned memo store + per-agent model wiring +
  LLM-driven macro + real Agents SDK + env split.
- GPT-5 `max_completion_tokens` compat + PM model switch.
- Vertex AI as alternative Gemini backend.
- Master plan documentation (`docs/MASTER_PLAN.md`).

### Decisions already locked — do not re-debate

These come from `docs/MASTER_PLAN.md §5` and §6. Apply them without
asking:

1. **Bull/bear architecture (Wave 3A):** sector-integrated. The sector
   analyst writes BOTH bear (first) + bull case + key disagreement +
   falsifiable tests + first-pass synthesis in a single structured
   output. PM uses sector synthesis as a prior, not a directive. NO
   separate adversarial Bull/Bear agents.

2. **Technical analyst (Wave 3B):** surfaces as a separate "technical
   positioning" finding. Does NOT override the rating. Rating remains
   fundamentals + valuation + thesis-driven.

3. **DCF assumption updater (Wave 5A):** LLM-driven. No deterministic
   guard rails beyond a ±20% delta cap per cycle. Log every proposed
   change with rationale.

4. **News-impact-agent (Wave 5B):** Anthropic Haiku 4.5 (cross-family
   with PM's OpenAI synthesis). Material news creates `v(N+1)` patch;
   critic skipped on patches; `revision_log` flagged
   `critic_skipped: true`. Per-ticker FIFO queue; max 2 patches per
   ticker per day.

5. **As-of-date (Wave 1C):** memory writes skipped on backtest runs.
   Cache key segregated by `:asof:` suffix. `MemoSnapshot.as_of_date`
   distinct from `generated_at`.

6. **LLM trace retention (Wave 1A):** 90 days then GC.

7. **Long-form reports (Wave 3C):** default ON. Token cost (~$0.02-0.04
   per memo) is small enough not to gate. Optional
   `ENABLE_LONG_FORM_REPORTS` flag for opt-out.

8. **Outcome tracker reflection (Wave 4A):** narrative reflection only
   for 90d / 365d horizons. 1m / 3m surfaced numerically but no prose
   entry (too noisy).

9. **Backtest UI (Wave 1C):** API-only via `?as_of=` query param. No
   date picker on `/research` for v1.

10. **Research notes (Wave 7):** explicit frontmatter routing is source
    of truth. Two-tier injection (always-summary + capped body
    retrieval). Separate from `memory/` — different folder, different
    ingest. Hard-cap 4KB body inject per agent run.

### Workflow conventions

For every PR:

1. **Branch from `main`.** Naming pattern: `feat/wave-N<letter>-<short-slug>`
   (e.g., `feat/wave-1a-llm-trace-logging`,
   `feat/wave-3a-sector-bull-bear`).

2. **Code follows existing patterns.** Read these files before changing
   adjacent ones:
   - `backend/app/agents/safe_runner.py` — failure isolation pattern.
   - `backend/app/cache/snapshots.py` — ORM + lazy table create + helpers.
   - `backend/app/services/memo_store.py` — versioned persistence pattern.
   - `backend/app/agents/llm.py` — provider abstraction + per-agent
     model knobs + circuit breaker.
   - `backend/app/agents/reflection_agent.py` — long-term memory write
     pattern.

3. **New ORM tables:** mirror `cache.snapshots._ensure_table` pattern
   (lazy `__table__.create(checkfirst=True)` on first use) so direct-
   import callers don't need `init_db()`.

4. **Tests:** every PR adds tests covering happy-path + failure-mode +
   one edge case. Place under `backend/app/tests/`. Pin model decisions
   with `monkeypatch.setattr(settings, "...", ...)` rather than relying
   on `.env`. Never write a test that requires a real API key — use
   `unittest.mock.patch` to mock provider responses.

5. **Eval gate after every PR (in this exact order):**
   ```bash
   cd backend && rm -f marketmosaic.db && python -m pytest -q
   cd backend && rm -f marketmosaic.db && python -m scripts.smoke_test
   cd frontend && npm run build
   ```
   All three must pass. If pytest fails: fix root cause; do NOT delete
   tests. If smoke fails: that's a real regression — fix it. If
   frontend build fails: TS / vite errors are real, fix them.

6. **Commit format:**
   ```
   feat(<scope>): <short summary>

   Why this change exists. What it does. What it doesn't.
   List the touched files and why.
   Tests + smoke + frontend status.

   Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
   ```

7. **PR creation via `gh pr create --base main --head <branch>` with
   structured body (Summary / Test plan / What's deferred). Use
   `gh pr merge <#> --squash --delete-branch` to merge. Sync `main`
   with `git checkout main && git pull --ff-only` between PRs.

8. **NEVER push directly to `main`.** Always go through PR workflow.

9. **NEVER use destructive git commands** (`reset --hard`, `push --force`
   to `main`, `branch -D` of unmerged work).

10. **Avoid the demo_*.json churn:** `backend/app/data/demo_*.json` files
    regenerate on every seeder run with date-shifted contents. DO NOT
    stage them in commits unless the change is intentional. Use
    `git add` with explicit paths, not `git add .`.

11. **The `.env` file may contain real API keys** — never log or commit
    its contents. Use `settings` from `app.config` to read values.

### Test discipline for live-mode keys

Tests run with `ENABLE_LIVE_DATA=false` forced by `conftest.py`. Some
tests (especially in `test_provider_wrappers.py`) use `monkeypatch.setattr`
to clear API keys when the test premise requires "no key set" — preserve
this pattern. Never assume any key is or isn't present.

### Sequenced execution plan

Read `docs/MASTER_PLAN.md` first for full per-wave specs. Then proceed
in this order:

#### Wave 1 (3 PRs, parallelizable)

- **PR-W1A — LLM call trace logging.** New `LLMCallLog` ORM table.
  Hook into `agents/llm.py::_record_usage` to also write a row tagged
  `(run_id, agent_name, model, tokens_in, tokens_out, duration_ms,
  success)`. `run_id` propagated via `contextvars.ContextVar`.
  `services/llm_metrics.py` with aggregation queries. Admin endpoint
  `/api/admin/llm-metrics?run_id=X`. CLI report
  `scripts/llm_cost_report.py`. 90-day GC.

- **PR-W1B — On-demand "Analyze this stock" UI.** Frontend only.
  Tier badge in ticker list; data-only tickers show an explicit
  "Analyze" button that hits `?ondemand=true`. Loading state.

- **PR-W1C — As-of-date selector.** Thread `as_of: Optional[date]`
  through `BaseProvider`, every provider class, `data_service`
  capability chain, `cache.cache_get`/`cache_put` (key segregation
  via `:asof:` suffix), `agents/graph.py::run_stock_memo`.
  `MemoSnapshot` gains `as_of_date` column. API: `?as_of=YYYY-MM-DD`.
  Memory writes skip when `as_of_date` is set. Validate `as_of_date`
  not in future.

#### Wave 2 (1 PR — depends on nothing)

- **PR-W2 — Financial-history database.** Three new ORM tables:
  `FinancialPeriod`, `FilingDoc`, `EarningsTranscript`. Service
  `services/history_service.py`. Backfill job
  `monitoring/history_backfill.py` (nightly under `ENABLE_MONITORING`).
  Tier-1 names backfilled to ≥40 quarters / ≥12 filings / ≥10 transcripts.

#### Wave 3 (4 PRs — 3A/3B/3C parallel; 3D after 2)

- **PR-W3A — Sector-integrated bull/bear.** Rewrite `SECTOR_ANALYST_PROMPT`
  to require 5-block structured output (bear case FIRST, then bull,
  then key disagreement, falsifiable tests, sector synthesis +
  lean). New `BullBearAnalysis` schema stored under
  `sector_agent_view.data["bull_bear_analysis"]`. PM synthesis prompt
  updated to read sector synthesis as a prior. Replace
  `graph._bull_case` / `_bear_case` template helpers.
  `MemoCard.tsx` renders `key_disagreement` + falsifiable tests.

- **PR-W3B — Technical Analyst.** Pure-math indicator computation in
  `app/finance/technicals.py` (SMA/EMA/RSI/MACD/Bollinger/VWMA on
  252+ days OHLCV). LLM narrative pass in `agents/technical_agent.py`.
  New `TechnicalSignals` schema. `StockMemoOut.technical_agent_view`.
  Frame as "positioning in support of long-term thesis", NOT trade
  signals. Tile in MemoCard with optional Recharts spark-chart.

- **PR-W3C — Drill-down agent reports.** `AgentFinding.long_form_report:
  Optional[str]` (markdown). Each agent prompt extended for 6-8
  paragraph report. Deterministic templated fallback per agent. New
  `AgentReportDrawer.tsx`. Per-tile expand button + "View all reports"
  tab. Default-on; optional `ENABLE_LONG_FORM_REPORTS` flag for opt-out.

- **PR-W3D — Memory hooks for filings/transcripts.** Depends on W2.
  Reflection extracts structured facts (segment performance, guidance
  changes, capex commentary, M&A, leadership changes) from new
  filings/transcripts and appends as structured memory entries.
  `MemoryEntry` gains optional `structured_facts: Dict`.

#### Wave 4 (2 PRs — 4A depends on 1C)

- **PR-W4A — Realized outcome tracking.** `MemoOutcome` ORM
  (memo_id, ticker, rating_at_memo, confidence_at_memo, price_at_memo,
  horizon_days, evaluated_at, forward_return, benchmark_return, alpha,
  thesis_held, note). `services/outcome_tracker.py`. Daily APScheduler
  job `monitoring/outcome_loop.py`. Outcome entries written to
  `memory/companies/<TICKER>.md` with `trigger="outcome:90d"`.
  Reflection only for 90d / 365d horizons.

- **PR-W4B — Live integration tests.** Opt-in `RUN_LIVE_TESTS=1` test
  path. Skipped by default. CI nightly. Cover real EDGAR / FMP /
  Anthropic / Gemini round-trips.

#### Wave 5 (2 PRs — 5A after 2; 5B after 5A)

- **PR-W5A — Persistent DCF + LLM assumption updater.** `DCFModel` ORM
  versioned with `parent_version`. `services/dcf_updater.py`: roll
  forward year 1 → year 0 actual, LLM proposes assumption deltas
  (revenue growth, op margin, capex %, terminal growth, WACC), engine
  re-runs, store as v(N+1). LLM prompt enforces ±20% delta cap.
  Assumption-change log stored in DCFModel JSON for review.

- **PR-W5B — Update orchestrator + news-impact-agent.**
  `services/update_orchestrator.py`. New `agents/news_impact_agent.py`
  (Anthropic Haiku 4.5). Material news → `incremental_patch` creates
  `v(N+1)` inheriting from `v(N)`, only patches rating/confidence/risks/
  risk_committee_challenge. Critic skipped on patches; `revision_log`
  flagged. Per-ticker FIFO queue. Max 2 patches per ticker per day.

#### Wave 6 (4 PRs — independent; opportunistic)

- **PR-W6A — Checkpoint resume.** `MemoRunCheckpoint` ORM keyed
  `(run_id, step_name)`. `@checkpointed("step_name")` decorator on
  major steps in `run_stock_memo`. `run_id` via `contextvars`.
  24-hour TTL + daily GC.

- **PR-W6B — Cohort-similarity invalidation tightening.** Hash specific
  KPI inputs (revenue/op_income/capex/shares) into `sources_used` so
  irrelevant peer 10-Ks don't trigger sector_warm recompute.

- **PR-W6C — News allow-list to JSON config.** Move `_ALLOWED_DOMAINS` /
  `_BLOCKED_DOMAINS` from `news_agent.py` constants to
  `app/data/news_domains.json`.

- **PR-W6D — Schema migration path.** `cache/migrations.py` registering
  per-`(kind, from_version, to_version)` upgrader functions. Runs on
  read when stored schema_version doesn't match current.

#### Wave 7 (3 PRs — 7A first; 7B after 7A; 7C after 7A + Wave 3 specialists exist)

- **PR-W7A — Research notes MVP.** New `research_notes/` folder at repo
  root with subfolders (books/, interviews/, articles/, frameworks/,
  personal/). Frontmatter contract per `docs/MASTER_PLAN.md` Wave 7.
  `services/research_notes.py` with `select_for(agent, sector,
  sub_industry, ticker)`. `scripts/index_research_notes.py` indexer.
  Hook summary injection into `sector_agents.py::run_sector_agent`
  AFTER the existing memory read, BEFORE the LLM call. Seed with 3-5
  representative notes (e.g., a Pat Dorsey moats note, a Howard Marks
  cycles note, a software-investing article tagged
  `applies_to_sectors: [Technology, Communication Services]`).
  Add `research_notes/_index.json` to `.gitignore`.

- **PR-W7B — Two-tier with BM25 body retrieval.** Extend `select_for`
  to return up to K=2 body chunks ranked by BM25 over agent working
  context. Reuse `app/services/retrieval_service.py` if applicable.
  Hard-cap 4KB total body inject. Fallback to summaries-only when
  cap exceeded.

- **PR-W7C — Cross-agent injection.** Add `research_notes.select_for`
  calls to valuation/comps/risk/earnings agent runners. Indexer
  defaults `applies_to_agents` based on body keyword detection.

### Stopping criteria

You should stop and produce a final report when ANY of these occur:

1. A test failure persists after 2 fix attempts.
2. A merge conflict requires non-trivial manual judgment about user
   intent.
3. A new ORM migration would break existing data integrity.
4. You complete all 19 PRs successfully.

The final report (markdown) should include:
- Per-PR status: merged / open / blocked, with PR # and one-line note.
- Test count + smoke + frontend status at end.
- Any decisions you made that weren't covered above.
- Files added vs modified count.
- Any new env vars introduced (with defaults).
- Suggested next steps.

### Final reminder

You're not in a hurry. Land each PR cleanly with proper tests rather
than racing through them. The user reviews each PR; bad code costs more
than slow execution. Keep PRs reasonably scoped (single wave step per
PR) so review stays tractable.

When you finish each PR, immediately verify with the eval gate, then
move to the next. Do NOT batch PRs and run tests at the end.

Begin with Wave 1A (LLM trace logging). Do NOT pause after each PR
unless the eval gate fails. Continue through Wave 7C. Produce the
final report only after all 19 are landed or you hit a stopping
criterion.

---

## How to invoke this prompt

```
You are an expert full-stack engineer working on MarketMosaic.
Repo path: /Users/jaredcohen/code/AgenticAI/stock-research-platform/

Read `docs/AUTONOMOUS_EXECUTION_PROMPT.md` and execute the PROMPT
section verbatim. Begin with Wave 1A and proceed through Wave 7C in
the documented dependency order. Do not pause for clarification.
```
