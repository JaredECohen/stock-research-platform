# FEAT-003 — Industry Analysis: owner decisions and operations

GICS Industry Group analysts and the weekly Industry Analysis reports ship
**dark**: `ENABLE_INDUSTRY_REPORTS` and `ENABLE_INDUSTRY_ANALYST_ROUTING` are
`false`, so the taxonomy imports and the classification audit runs, but no
report is generated and the sector analyst remains the memo pipeline's
primary. Nothing below turns on until the owner decides §1 and flips the
flags in §4.

This file covers the API and ops surface (slice 5): the four admin
endpoints, the read routes' access policy, the cron loops to watch, and
the rollback. The analyst, writer, statistics and job internals are
documented in their own modules.

Everything a report says is **research and education**. Forward-looking
lines are scenarios, not recommendations, and the reports carry that
disclaimer on every response (`store.DISCLAIMER`).

---

## 1. Owner decisions — blockers, with the default now shipping

Each row is live today at the default. Changing one is an env-var edit in
the Render dashboard (and, where noted, a code change); none of them needs
a migration.

| # | Decision | Default now shipping | What changes it | Why it is a blocker |
|---|---|---|---|---|
| 1 | **Public or Pro?** Who can read the latest report for an industry group. | `INDUSTRY_ANALYSIS_ACCESS=public` — the latest edition and the constituent list answer to everyone; **history**, **changes-since-prior** and the PM's cross-industry block are Pro whenever `AUTH_ENABLED` is on. | Set `INDUSTRY_ANALYSIS_ACCESS=pro` on the **web** service. Both gates (`auth/policy.py` and the handler seam) read it from `industry_report_store.access_policy()`, so one edit moves both. | It decides whether this is an acquisition surface or a Pro feature, and the pricing page copy follows it. |
| 2 | **Benchmark set.** What a group's return is measured against. | `INDUSTRY_BENCHMARKS=universe_ew,sector_ew,KFR.MKT_RF.D` — our own universe equal-weight, the sector equal-weight cohort, and the Ken French daily market factor. | Changing it to SPY / sector ETFs needs **ETF price data rights we do not have**; do not set it until that is bought. | A benchmark we may not redistribute is a licensing exposure, not a config choice. |
| 3 | **Publication time.** | Sunday **06:30 UTC**, as-of the prior **Friday** close, `period_key` = the ISO week of that Friday (`INDUSTRY_REPORTS_CRON_DOW/HOUR/MINUTE`, `INDUSTRY_REPORTS_AS_OF_WEEKDAY`). | Env vars on the **worker**. An admin regenerate derives the same `period_key` from the same settings, so a manual refresh and the cron cannot create two edition series. | Moving it after launch changes what "this week's report" means to a reader mid-week. |
| 4 | **Auto-publish or review?** | `INDUSTRY_REPORTS_REQUIRE_REVIEW=false` — an edition that passes the validator publishes itself. The `pending_review` status value exists and the flag is honoured, but **there is no publish endpoint**: setting it to `true` today means nothing publishes until one is built. | Leave `false` until a review UI exists. | Turning it on without a publisher is an outage that looks like a quiet week. |
| 5 | **GICS display and data rights.** | Codes and names are displayed with the attribution string in `gics_registry.ATTRIBUTION`. Every company→group mapping is labelled *derived from provider classification, not licensed GICS security assignments*. Assignments taken from the research map are labelled *economic research examples; not official licensed issuer GICS mapping*. No S&P/MSCI constituent file is in the repo, and `internal_labels` display mode is reserved but **not built**. | A licence would let us drop the caveats and ship real constituent lists. | Shipping unlabelled mappings as licensed GICS is the one claim here that is not ours to make. |

### The sub-industry layer

The 8-digit layer and the symbol→sub-industry `security_reference` come from
`docs/research/Investment_Universe_163_Map.json`, which is **our own research
map**. Its symbol assignments are economic research examples, not licensed
issuer GICS mapping, and `GET /api/industries/{code}/companies` returns that
label on every row (`classification.source_label`, plus
`security_reference_caveat` on the response). Do not restate a research-map
assignment as an issuer's official GICS classification anywhere in the UI,
the marketing site, or an export.

Retired sub-industries import as **inactive** nodes rather than being
dropped, so an old report's codes still resolve to a name.

---

## 2. Operations

All four ops endpoints live under `/api/admin/*` and take the operator
bearer token (`ADMIN_API_TOKEN`). None of them is browser-called, so none is
in `admin_auth.EXEMPT_PREFIXES`. A customer JWT — Pro included — does not
open them.

```
export ADMIN=…            # ADMIN_API_TOKEN
export BASE=https://marketmosaic.onrender.com
auth=(-H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json')
```

### 2.1 Import the taxonomy (once per taxonomy edition)

```
curl -sS "${auth[@]}" -X POST "$BASE/api/admin/industries/taxonomy/import" \
     -d '{"activate": true}'
```

Reads the one bundled knowledge JSON
(`backend/app/data/industry_knowledge/gics_industries_2026.json`) and writes
the node rows. Idempotent by node checksum: re-running it is a no-op
(`"imported": false`). The same version key with a **different** structure is
a `409 taxonomy_checksum_mismatch` — a changed structure is a new version,
because existing reports and classifications point at the old one by id.
A missing JSON is `503 knowledge_base_missing`, by design: an absent
knowledge base is a build defect, and an empty taxonomy would only surface
weeks later.

What today's bundle imports (observed 2026-09-11 — **counts are never
hardcoded anywhere in the API**; every response reads them back from the
registry):

```
sector 11 · industry_group 25 · industry 74 · sub_industry 163 · inactive 7
```

The daily `industry_classification_loop` calls
`gics_registry.ensure_taxonomy(activate=True)` on every run, so a deployed
bundle imports itself within a day on the worker. This endpoint is for
importing one **now** — a mid-cycle taxonomy edition, or the first boot of a
deployment whose loop has not run yet.

### 2.2 Classify the universe

```
curl -sS "${auth[@]}" -X POST "$BASE/api/admin/industries/classify" -d '{}'
# one group only, forced:
curl -sS "${auth[@]}" -X POST "$BASE/api/admin/industries/classify" \
     -d '{"tickers": ["AAPL","MSFT"], "reclassify": true}'
```

Two bulk reads and one bulk insert whatever the size — no per-ticker loop, so
it is safe on the web process. The response reports `counts` by state
(`mapped` / `fallback` / `missing` / `stale` / `conflict`) and `sources`
(`research_map` / `provider_alias`). `changed` and `unmapped_labels` are
capped, and each cap reports its total and how many entries it dropped.

Read the states honestly:

* `fallback` knows the company's **sector** only. It is **not** a group
  member and is never counted in a cohort.
* `conflict` means the research map and the provider alias disagree about the
  group. The row **is** a member (the research map wins routing) and both
  codes are recorded.
* `missing` is a company no rule resolved. It is a gap to fix, not a zero.

The daily `industry_classification_loop` (03:40 UTC) audits this and records
`success=False` when the bundle has drifted from the active version or any
company is `missing`, so you normally do not run this by hand.

### 2.3 Regenerate reports

```
curl -sS "${auth[@]}" -X POST "$BASE/api/admin/industries/reports/regenerate" \
     -d '{"period_key": "2026-W37", "codes": ["4510"]}'      # 202
```

**Enqueues.** It never generates in the request: the worker's drainer owns
that, and a web process that generated a report would be exactly the
"expensive work in a page request" failure this repo has already paid for.
Omit `codes` for every group; omit `period_key` for the current week derived
from the same settings as the cron. An unknown code is a `404` here rather
than three failed attempts in the worker.

**Do not run a regenerate while `/api/admin/cron-health` shows
`taxonomy_drift=1`** on `industry_classification_loop`: the bundle and the
active version disagree, and the reports you would queue would be written
against the version that is about to be replaced. Import first (§2.1).

While the drainer module is not on the build the endpoint answers
`503 report_queue_unavailable` and names what is missing, rather than
returning 202 for a job nobody will ever run.

### 2.4 Watch the queue

```
curl -sS "${auth[@]}" "$BASE/api/admin/industries/jobs?status=failed&limit=50"
```

Newest first. `status_counts` is computed in SQL over the whole filtered
queue, so a capped page never makes a backlog look smaller than it is.
`drainer.module_present` and `drainer.enabled` say whether anything is
draining it — remember this endpoint is served by **web**, and the drainer
runs on **worker**.

### 2.5 Cron-health names to watch

`GET /api/admin/cron-health` (admin). Two loops belong to this feature:

| Loop | Cadence | Healthy looks like |
|---|---|---|
| `industry_classification_loop` | daily, 03:40 UTC | `success=true`, note carries `taxonomy_drift=0`. `success=false` means drift, or at least one `missing` company — both are real, both are visible in the note. |
| `industry_weekly_loop` | Sunday, 06:30 UTC | `success=true` with a note of `disabled (ENABLE_INDUSTRY_REPORTS=false)` while the feature is dark — that is the dark state, not a failure. **Known gap (slice 4):** it ticks only on Sundays and is not yet in the `weekly_loops` set in `routes_admin.cron_health_endpoint`, so cron-health calls it stale after 26 hours. Until the drainer lands (which heartbeats every 5 minutes) a stale reading here is expected, not an outage. |

Both are registered in `app/monitoring/__init__.py::KNOWN_LOOPS` and run on
the **worker** service only.

---

## 3. Reading the API honestly

Three distinctions the responses keep apart, and that any UI or report built
on them must keep apart too:

1. **Membership is not price coverage.** `/companies` returns `count`
   (classified members) and `n_priced` (members the latest statistics row
   could price) as separate fields, and every unpriced name carries a
   reason. A group with three members and no price history is
   `insufficient_sample` — a labelled state, not an error and not a zero.
2. **Numbers travel with their method.** Never show `benchmark_relative`
   without `method.benchmark_cohort_basis` (the cohort deliberately excludes
   constituents the group's own numbers include — they are named under
   `sample.benchmark_cohort.group_members_outside_cohort`), and never quote
   `breadth.above_50d_mean` without `method.breadth_mean_window.sessions`.
3. **Staleness is composed, not guessed.** `stale` is true when the as-of is
   older than `INDUSTRY_REPORT_STALE_AFTER_DAYS` **or** a later refresh
   attempt failed; `last_attempt` names the failure. A failed week leaves the
   previous edition in place, flagged — it never blanks the page.

Before the first import, every read answers `503 taxonomy_not_imported` with
the remedy, and `/api/industries/taxonomy` answers **in every
configuration** — including `INDUSTRY_ANALYSIS_ACCESS=pro` — carrying the
access policy, so the UI can explain a gate instead of rendering a 401.

---

## 4. Flag order at go-live

Each step is a dashboard edit plus the check beside it.

1. **Worker**: `ENABLE_INDUSTRY_REPORTS=true`. At the next tick
   `/api/admin/cron-health` should show `industry_weekly_loop` with a note
   that no longer says `disabled`. Leave it **false on web** — the drainer
   must not run in the process that serves pages.
2. Wait for one Sunday (or run §2.3 for a single group). Check
   `/api/admin/industries/jobs` drains to `succeeded`, then read
   `/api/industries/{code}/report` and confirm `degraded` is empty and the
   sources manifest is populated.
3. **Web**: decide §1 row 1. `INDUSTRY_ANALYSIS_ACCESS` stays `public` unless
   the owner moves it.
4. Only after a production A/B has been read: `ENABLE_INDUSTRY_ANALYST_ROUTING=true`
   on both services. Until then the sector analyst remains the memo
   pipeline's primary and the industry analyst is additive.

---

## 5. Rollback

In increasing order of blast radius — each step is reversible and none of
them drops data.

| Symptom | Action | Effect |
|---|---|---|
| Reports are wrong or embarrassing | `ENABLE_INDUSTRY_REPORTS=false` on the worker | Generation stops. Published editions stay readable; the weekly loop records `disabled`. |
| The memo pipeline regressed after routing | `ENABLE_INDUSTRY_ANALYST_ROUTING=false` | The industry analyst stops being routed; the sector analyst is primary again. No memo data changes. |
| The read surface must come down | `INDUSTRY_ANALYSIS_ACCESS=pro` with `AUTH_ENABLED=true` | Everything except `/api/industries/taxonomy` goes behind the wall in one edit. |
| A bad taxonomy edition was imported and activated | `POST /api/admin/industries/taxonomy/import` with the previous `version_key` and `activate: true` | Versions are additive rows; activating the old one restores it. Reports and classifications keep pointing at the version id they were written against, so nothing is orphaned. |
| The queue is wedged | Delete the `queued`/`running` rows for the period in `industry_report_jobs`, then re-run §2.3 | Jobs are durable rows, not in-process state; the drainer picks the new ones up on its next tick. |

There is no "unpublish" and no deletion path for an edition: history is the
product. Take the surface down instead.
