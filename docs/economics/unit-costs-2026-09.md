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
| `pm_chat` | one `POST /api/chat` turn with inline memo generation disabled on **both** entry points — `orchestrator.run_stock_memo` and `sdk_runtime.run_stock_memo_via_sdk` (the `USE_AGENTS_SDK=true` branch, which is the committed `config.env` default and mints its own `run_id`) — so the sample answers from the stored memo, as the gated orchestrator will | Web: `classify_intent` (one cheap call) + SDK chat agent or memo-context answer (1+ calls, tool reads). |
| `dcf` | `POST /api/dcf/{t}` with default assumptions, `force_refresh=True` | Web: one cheap-route call for bull/bear scenario drivers (`build_bull_bear`) on **every** request that carries an assumptions body — `build_dcf` reads its 7-day cache only for `assumptions=None`, and the DCF Lab always posts a body; provider reads for financials + estimates. |
| `comps` | `GET /api/comps/{t}`, `force_refresh=True` | Web: one cheap-route call for exposure peers (cached **30d**; the comps snapshot itself 7d; the route exposes no refresh switch); provider reads for the target and every peer. |
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
   the table falls back to the provider default — it is logged once, the
   report names it under `fallback_priced_models`, and
   `test_llm_prices.py` fails if a model in `Settings` has no row),
   failures, LLM duration.

   **Price-table correction, 2026-09-19.** Every row was re-verified against
   the providers' pricing pages (`PRICES_VERIFIED_ON`). Any figure produced
   before that date used a table that was off for every routed model: Opus
   4.7/4.8 at 3x list ($15/$75 vs $5/$25), Haiku 4.5 at half ($0.50/$2.50 vs
   $1/$5), gpt-5.5 at 1.6x/1.07x, gpt-5.4 at 2x/1.33x, gpt-5 at 2.8x/1.4x,
   gemini-2.5-pro at 2.8x/1.4x, gemini-2.5-flash output at half, and the
   configured `gpt-4.1-mini` and `gemini-3.1-pro` absent (priced at the
   $3/$12 default, 7.5x and 1.5x high on input). Re-run the script before
   comparing against any pre-correction number.
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
  turn is cheaper. `USE_AGENTS_SDK` is left as configured (true in
  `config.env`) so the SDK chat agent is what gets measured; only the two
  inline-memo entry points are swapped, and
  `test_route_audit.py::test_pm_chat_sample_never_generates_a_memo` pins
  that a `single_stock_analysis` turn cannot start a memo run under either
  flag value. A sample whose `llm.n_calls` is in the twenties is a memo
  run leaking in — treat it as a script bug, not a chat cost.

## 3. Worst-case monthly variable cost per user

Let `C(f)` be the mean LLM $/action for feature `f` from §5. Allowances are
the DEVPLAN FEAT-002 launch numbers (plan §4.7), UTC calendar month.

Three kinds of term appear, and the reader must not confuse them:

| Kind | Meaning | Is it a ceiling? |
|---|---|---|
| **metered** | `allowance × C(f)` — the usage meter refuses the (n+1)th action | yes |
| **cache-bounded** | no meter, but the LLM call sits behind a cache whose TTL bounds the monthly count arithmetically | yes, as long as the cache stays in front of the call |
| **unmetered `U_*`** | no meter and no cache; the only limit is the per-user rate scope | **no** — `U_*` is a *declared usage assumption* for the go/no-go; the abuse ceiling is the rate limit, reported separately |

**Free Explorer, every allowance exhausted:**

```
Free = 3·C(memo_view) + 1·C(research_run) + 10·C(pm_chat) + 5·C(chart_commentary)   # metered
       + 6·C(comps)                                                                # cache-bounded
       + U_free_dcf·C(dcf)                                                         # unmetered, U_free_dcf = 60 (assumed)
```

- `6·C(comps)`: the "comps follows the memo allowance" rule gives a Free
  user 3 tickers; `GET /api/comps/{t}` exposes no refresh switch, the
  comps snapshot is cached 7 days and the exposure-peers LLM call 30 days,
  so a 31-day month can reach that call at most twice per ticker — 3 × 2.
  The bound disappears if a refresh switch is ever added to the route.
- `U_free_dcf·C(dcf)`: the "DCF follows the memo allowance" rule limits
  *which tickers*, not *how many runs*. `POST /api/dcf/{t}` calls
  `build_bull_bear` (one LLM call) on every request whose body carries
  assumptions — `build_dcf` reads its cache only for `assumptions=None`,
  and the DCF Lab always posts a body — so a Free user's DCF spend is
  bounded by the `series` scope (60/min/user ⇒ up to 3,600 LLM calls/hour),
  not by 3. Plan §4.7 omits this term; an earlier draft of this doc wrote it
  as `3·C(dcf)`, which understated it. **`U_free_dcf = 60`** (20 what-if
  runs on each of 3 tickers) is the declared assumption for the go/no-go;
  the script prints `3600·C(dcf)` next to it as the per-hour abuse ceiling.
  See the owner question at the end of this section.

**Pro, every allowance exhausted:**

```
Pro = 20·C(research_run) + 300·C(pm_chat) + 100·C(chart_commentary)   # metered
      + U_pro_dcf·C(dcf) + U_pro_comps·C(comps)                        # unmetered, assumed 200 / 20
      + U_portfolio·C(portfolio_build)                                 # unmetered, NOT sampled by the script
```

- `POST /api/macro/analyze` and `POST /api/screener/nl` are charged as
  `pm_chat` turns (orchestrator decision), so they sit inside
  `300·C(pm_chat)` and get no separate `U_macro` term.
- `U_pro_dcf = 200` and `U_pro_comps = 20` (10 tickers × 2 cache expiries)
  are declared assumptions, replaced by observed usage once there is any.
  Rate ceilings: `dcf` 3,600/hour (`series`), `comps` 7,200/hour (`data`).
- `portfolio_build` (one cheap-route call, `llm_light` 10/min with 2
  concurrent, Pro-only) is not sampled by the script; the script lists it
  under `unmeasured_terms` and leaves `pro.total_usd` `None` until a
  figure is supplied by hand from `LLMCallLog` for that route.

The script mirrors these constants (`ALLOWANCES`, `CACHE_BOUNDED`,
`ASSUMED_UNMETERED`, `RATE_CEILING_PER_HOUR`, `UNMEASURED_TERMS`) and
reports, per plan: `metered`, `cache_bounded`, `assumed_unmetered`,
`rate_ceiling_per_hour_usd`, `total_usd` (`None` while any term is
unmeasured), `total_measured_usd` and `unmeasured_terms`.
`test_route_audit.py::test_free_dcf_is_not_an_allowance_term` pins that
`dcf` never returns to the metered dict for Free.

**Owner question (extends plan §9 Q5).** Q5 accepted the in-request DCF
LLM call "bounded by 60/min/user" — that bound is 3,600 calls/hour, which
is not a bound a Free tier can carry. Before `USAGE_LIMITS_ENABLED` flips,
one of: (a) meter `dcf` runs for Free (e.g. 20 what-ifs per allowed ticker
per month, matching `U_free_dcf`), (b) run `build_bull_bear` only for the
default-assumption build and use the deterministic scenario fallback for
Free what-ifs, or (c) precompute scenarios in the worker (Q5's alternative).
Until decided, `U_free_dcf` is an assumption and the Free threshold below
is evaluated against an assumption, not a ceiling.

**Go / no-go thresholds** (plan §4.7, owner decision §9 Q1 if failed):

| Plan | Threshold | Rationale |
|---|---|---|
| Pro | worst-case variable cost **< 50% of $29.99 = $14.995 / user / month** | leaves gross margin for infrastructure, payment fees (~3%+$0.30) and error in the assumed `U_*` terms |
| Free | worst-case variable cost **< $1.50 / user / month** | a Free user must be cheap enough that the trial-to-paid funnel, not cost control, is the reason to limit Free. Evaluated with `U_free_dcf = 60` — an assumption; the DCF meter question above must be settled for this to become a ceiling |

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

Free worst case: **not measured** (threshold < $1.50; with `U_free_dcf = 60` assumed)
Pro worst case: **not measured** (threshold < $14.995; `portfolio_build` term to be supplied by hand)
Rate-limit ceilings (`dcf`, `comps` $/hour): **not measured**

Verdict: **pending measurement.** Owner question §9 Q1 cannot be answered
until this section is filled.

## 6. Ongoing measurement after launch

Once S1 adds `LLMCallLog.user_id` / `feature`, gross margin by plan and
feature is a join of `llm_call_logs` to `users`/`subscriptions` with no
prompt content (DEVPLAN "Product and conversion analytics"). The per-action
figures in §5 become the baseline that `GET /api/admin/llm-metrics` is
compared against; re-run the script after any change to the analyst roster,
model routing or `MODEL_PRICES_PER_MTOK`.
