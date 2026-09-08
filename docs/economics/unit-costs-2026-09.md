# Unit costs — 2026-09 (FEAT-002 Phase 1)

**Status: NOT YET MEASURED.** This document holds the method, the formulas
and the go/no-go thresholds. The results section is empty on purpose — it is
filled by running `backend/scripts/audit_unit_costs.py` in the nightly-live
environment (real keys), never by estimation. Do not put numbers in §5 that
did not come from that script's JSON output.

Companion: [`route-audit-2026-09.md`](route-audit-2026-09.md) (which routes
reach an LLM or a provider, and the proposed policy per route).

## 1. What is being measured

One row per billable feature in the DEVPLAN FEAT-002 matrix:

| Feature | Representative action | Where the cost is incurred |
|---|---|---|
| `memo_view` | read the latest stored `MemoSnapshot` for a ticker | DB read only — the memo was paid for when it was generated. Expected LLM $0, provider calls 0. Measured anyway so a regression (a read path that starts generating) shows up as a non-zero number. |
| `research_run` | `POST /api/stocks/{t}/analyze` → `RegenJob` → worker `execute_job` → `run_stock_memo(force_refresh=True)` | Worker: ~26 LLM round-trips per memo (specialists + PM synthesis + risk committee), provider backfill on cold tickers, 5-9 minutes wall-clock. **The dominant term.** |
| `pm_chat` | one `POST /api/chat` turn with inline memo generation disabled (answers from the stored memo, as the gated orchestrator will) | Web: `classify_intent` (one cheap call) + SDK chat agent or memo-context answer (1+ calls, tool reads). |
| `dcf` | `POST /api/dcf/{t}` with default assumptions, `force_refresh=True` | Web: one cheap-route call for bull/bear scenario drivers; provider reads for financials + estimates. |
| `comps` | `GET /api/comps/{t}`, `force_refresh=True` | Web: one cheap-route call for exposure peers (cached 7d); provider reads for the target and every peer. |
| `chart_commentary` | placeholder — FEAT-001 is not built | Unknown. The script records `not_implemented`; the formulas below treat it as an unmeasured term (result `None`), never as zero. |

`POST /api/macro/analyze` and `POST /api/screener/nl` are charged as
`pm_chat` turns (one cheap-route call each); they are cheaper than a chat
turn, so the `pm_chat` figure is an upper bound for them.

## 2. Method

Script: `backend/scripts/audit_unit_costs.py`. Run from `backend/` in the
nightly-live environment (same env block as
`.github/workflows/nightly-live.yml`):

```
RUN_LIVE_TESTS=1 ENABLE_LIVE_DATA=true USE_DEMO_DATA=false \
  python -m scripts.audit_unit_costs --tickers NVDA,COST,JPM --n 3 --research-n 1
```

Guards: the script exits 0 with a message unless `RUN_LIVE_TESTS=1` **and**
an LLM key is configured. That check happens before `app.config` is imported,
so a developer `.env` with live keys cannot be picked up by accident.
`test_route_audit.py::test_unit_cost_script_refuses_to_spend_by_default` pins
the refusal path.

Per action the script:

1. Enters `llm_call_context(agent_name="unit_cost_audit", run_id="audit-<feature>-<i>-<ts>")`
   so every `LLMCallLog` row written during the action carries that
   `run_id` (for `research_run` the job's own `run_id` is used instead,
   exactly as a user-triggered run would).
2. Runs the action (see §1). `research_run` goes through
   `regen_worker.enqueue` → `claim_next_job` → `execute_job` synchronously in
   the script process, so worker seconds are attributable and the queue is
   not shared with a live worker thread.
3. Reads `llm_metrics.cost_per_run(run_id)`: call count, tokens in/out,
   `estimate_cost_usd` (from `MODEL_PRICES_PER_MTOK`; a model missing from
   the table falls back to the provider default — check the `models` list
   in each sample and update the price table before trusting a figure),
   failures, LLM duration.
4. Counts provider work in the action's time window: `provider_cache` rows
   written (`fetched_at >= start` — every miss writes a row) and
   `cache_cost_logs` snapshot writes vs hits. Neither ledger records misses
   directly, so the window count is the proxy; actions run sequentially so
   the window is attributable. Provider spend is not converted to dollars —
   FMP/Polygon/Tiingo are flat subscriptions; the number to watch is
   *calls per action on a cold ticker* against each plan's daily quota.
5. Records wall-clock seconds and, for `research_run`,
   `RegenJob.finished_at - started_at`.

Output: `docs/economics/unit-costs-measured-<date>.json` (per-sample rows +
per-feature means + the worst-case totals of §3) and a markdown table on
stdout for §5. Counts and dollars only — no prompts, answers, memo bodies or
provider payloads are printed or written.

Sample size: `--n 3` per feature (`--research-n 1` by default because each
research run is 5-9 minutes; raise it when a tighter estimate is wanted).
Three tickers across three sectors (`SAMPLE_TICKERS`, default NVDA/COST/JPM)
so a sector-specific analyst roster does not skew the mean.

Caveats the reader must carry into §5:

- `estimate_cost_usd` is list-price arithmetic on token counts, not an
  invoice. Reconcile against the provider dashboards for the run window
  before a pricing decision.
- Warm caches make `dcf`/`comps`/`memo_view` cheaper than a first-touch
  ticker. `force_refresh=True` is used for `dcf`/`comps` so the LLM call is
  always exercised; provider counts are still whatever the cache state was.
- `pm_chat` cost depends on the question class. The three representative
  questions (risk summary, rating rationale with history, moat comparison)
  are chosen to hit the SDK chat path with tool reads; a purely conversational
  turn is cheaper.

## 3. Worst-case monthly variable cost per user

Let `C(f)` be the mean LLM $/action for feature `f` from §5. Allowances are
the DEVPLAN FEAT-002 launch numbers (plan §4.7), UTC calendar month.

**Free Explorer, every allowance exhausted:**

```
Free = 3·C(memo_view) + 1·C(research_run) + 10·C(pm_chat) + 5·C(chart_commentary)
       + 3·C(dcf) + 3·C(comps)
```

The last two terms are the "DCF/comps follow the memo allowance" rule: a
Free user can run DCF and comps for each of the 3 memo tickers. Plan §4.7
omits them; they are included here because each is an LLM call and the
worst case must not understate.

**Pro, every allowance exhausted:**

```
Pro = 20·C(research_run) + 300·C(pm_chat) + 100·C(chart_commentary)
      + U_dcf·C(dcf) + U_comps·C(comps) + U_portfolio·C(portfolio) + U_macro·C(macro)
```

`U_*` are Pro's unmetered features, bounded only by rate limits
(`series` 60/min, `llm_light` 10/min with 2 concurrent). There is no
allowance to multiply by, so the script reports the first three terms as
`pro.total_usd` and the doc must state the assumed `U_*` from observed usage
once there is any; until then the go/no-go below is evaluated on the metered
terms plus a stated headroom.

**Go / no-go thresholds** (plan §4.7, owner decision §9 Q1 if failed):

| Plan | Threshold | Rationale |
|---|---|---|
| Pro | worst-case variable cost **< 50% of $29.99 = $14.995 / user / month** | leaves gross margin for infrastructure, payment fees (~3%+$0.30) and the unmetered `U_*` terms |
| Free | worst-case variable cost **< $1.50 / user / month** | a Free user must be cheap enough that the trial-to-paid funnel, not cost control, is the reason to limit Free |

If either threshold fails, the DEVPLAN allowances are revisited *before* the
flags flip (`USAGE_LIMITS_ENABLED` stays false until then). Likely levers, in
order: lower `research_run` allowances (the dominant term), route `pm_chat`
classification to the cheapest model, and precompute DCF/comps in the worker
instead of in-request.

A second, non-dollar gate: **provider calls per cold research run** against
the FMP/Polygon daily quotas, so 20 Pro research runs on unknown tickers by a
handful of users cannot exhaust a provider plan. Report it from the
`provider_cache_writes` column.

## 4. Provider and worker capacity (not in dollars)

- Worker: one memo at a time on the starter instance (512 MB). Queue time,
  not cost, is the user-visible limit; `mean_worker_seconds × runs/day` must
  stay under the worker's daily capacity with headroom for scheduled regens
  (`update_orchestrator`, EDGAR poller). Record the mean worker seconds here
  once measured and derive runs/day.
- Providers: FMP is the primary fundamentals source (see
  `docs/research/Framework_Update_2026-09-07.md` for the provider roles);
  `provider_cache` TTLs make repeat tickers free. Cold tickers introduced via
  research runs pay the 5-year backfill (~6 calls per ticker on
  `run-backfill`'s budget) — that is the number the `research_run` provider
  column captures.

## 5. Results — NOT YET MEASURED

> **NOT YET MEASURED — run the script in the nightly-live environment**
> (`RUN_LIVE_TESTS=1 ENABLE_LIVE_DATA=true USE_DEMO_DATA=false python -m scripts.audit_unit_costs`)
> and paste the stdout table and the worst-case lines below, with the date,
> the models used (`llm.models` in each sample) and the JSON file name.
> No number in this section may be an estimate.

| Feature | n ok | LLM $/action (mean) | LLM calls | provider writes | snapshot writes | seconds | worker s |
|---|---|---|---|---|---|---|---|
| memo_view | — | not measured | — | — | — | — | — |
| research_run | — | not measured | — | — | — | — | — |
| pm_chat | — | not measured | — | — | — | — | — |
| dcf | — | not measured | — | — | — | — | — |
| comps | — | not measured | — | — | — | — | — |
| chart_commentary | — | not implemented (FEAT-001) | — | — | — | — | — |

Free worst case: **not measured** (threshold < $1.50)
Pro worst case: **not measured** (threshold < $14.995)

Verdict: **pending measurement.** Owner question §9 Q1 cannot be answered
until this section is filled.

## 6. Ongoing measurement after launch

Once S1 adds `LLMCallLog.user_id` / `feature`, gross margin by plan and
feature is a join of `llm_call_logs` to `users`/`subscriptions` with no
prompt content (DEVPLAN "Product and conversion analytics"). The per-action
figures in §5 become the baseline that `GET /api/admin/llm-metrics` is
compared against; re-run the script after any change to the analyst roster,
model routing or `MODEL_PRICES_PER_MTOK`.
