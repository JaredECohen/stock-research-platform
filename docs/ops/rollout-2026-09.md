# Rollout and rollback checklist — September 2026 completion branch

Branch: `work/marketmosaic-completion` (never pushed; no rollout without explicit
authorization). This checklist is the operator's sequence for taking the branch
to Render, verifying it, and backing it out. Every step that touches an external
system is marked **OWNER** and is not something an agent performs.

## 0. What ships

| Area | State on the branch | Runtime default |
| --- | --- | --- |
| Correctness/hardening backlog (Phase 2) | merged | always on |
| Refactor (stage split, registry roster, schema/model packages, typing) | merged | n/a |
| Accounts + billing (FEAT-002) | merged behind flags | `AUTH_ENABLED=false`, `USAGE_LIMITS_ENABLED=false`, no `STRIPE_*` |
| Fundamentals Explorer (FEAT-001) | merged | on (anonymous = Pro shape while the wall is off) |
| Fundamental Factor Scorecard (Phase 6) | merged | `ENABLE_SCORECARD=true`, `ENABLE_SCORECARD_DISAGREEMENT_REGEN=false`, `SCORECARD_EXPORT_TOKEN` unset |
| GICS registry + classification audit (FEAT-003 slice 1) | merged | `ENABLE_INDUSTRY_ANALYST_ROUTING=false` |
| Snapshot-cascade OOM fix + `snapshot_gc` retention (hotfix) | merged | always on; GC daily 04:15 UTC |
| Provider-key log leak fix (hotfix) | merged | always on |
| Industry analysts, analytics, weekly reports, API, UI (FEAT-003 slices 2–6) | merged | `ENABLE_INDUSTRY_REPORTS` false on web / true on worker; `ENABLE_INDUSTRY_ANALYST_ROUTING=false`; `INDUSTRY_REPORTS_REQUIRE_REVIEW=false`; `INDUSTRY_ANALYSIS_ACCESS=public` |

## 1. Pre-flight (agent-verifiable, all green before asking for authorization)

- [ ] `backend/`: `ruff check app`, the Python 3.11 AST/compile check, `python -m mypy`
      (configured scope), `make lock-check`, `git diff --check`.
- [ ] `backend/`: full suite with the CI flags and blank LLM keys; the only failure is
      `test_config_load_order` (key-blanking artifact). The harness refuses outbound
      sockets (`app/tests/netguard.py`) and lists any test that reached for the network.
- [ ] `frontend/`: `npm run lint`, `npm test`, `npm run build`.
- [ ] `test_deploy_config` passes: `ENABLE_MONITORING`/`ENABLE_REGEN_WORKER` false on web,
      true on the worker; `ENABLE_INDUSTRY_REPORTS` true on exactly one service.
- [ ] `KNOWN_LOOPS` count in `CLAUDE.md` matches `app/monitoring/__init__.py` (21).
- [ ] Route audit doc has a live row for every route (`test_route_audit`).

## 2. Deploy-visible canary (lock this first)

`/health` reports `build.git_commit` from Render's `RENDER_GIT_COMMIT`. After the deploy,
`curl https://<web-host>/health` must show the merged commit on **both** services (the
worker has no HTTP port; confirm its commit from the Render deploy log). Do not proceed to
step 4 until the web commit matches.

## 3. The deploy (**OWNER**)

1. Merge `work/marketmosaic-completion` into `main` (fast-forward or merge commit; never a
   history rewrite) and push — **only with explicit authorization**.
2. Render builds one image for `marketmosaic` (web) and `marketmosaic-worker`.
3. Keep the worker at **one instance**: the scorecard loop, the classification audit and the
   regen queue assume a single drainer per deployment (claims are DB-atomic, but two
   drainers double the provider budget).
4. Tables are created lazily on first use (`init_db()` at startup plus per-service
   `_ensure_tables`); there is no migration step. New tables are additive; nothing is
   dropped or altered.
5. Environment on Render — leave every new flag at its default listed in §0. In particular
   do **not** set `AUTH_ENABLED`, `STRIPE_*`, `CLERK_*`, `SCORECARD_EXPORT_TOKEN` or
   `ENABLE_SCORECARD_DISAGREEMENT_REGEN` in this rollout.
6. `PUBLIC_BASE_URL` / `marketmosaic.ai` on `:443` — owner decision recorded as a blocker;
   unchanged by this rollout.

## 4. Post-deploy verification

- [ ] `/health` → `status: ok`, `build.git_commit` = merged commit.
- [ ] `/api/providers/status` → `mode: live`, no unexpected `missing_api_keys`, `llm.degraded=false`.
- [ ] `/api/admin/cron-health` (admin bearer) → 21 loops registered, none stale beyond its
      class; `scorecard_loop` note shows `daily=…` and `review_queued=0`;
      `industry_classification_loop` note shows `taxonomy_drift=0` and `missing=0` (a
      non-zero `missing` is the alias map needing an entry, not an outage).
- [ ] Stored memos are cached snapshots: fixes only reach a ticker after regeneration.
      Trigger regen for the tickers that matter (`POST /api/admin/rerun-memos` or
      `POST /api/stocks/{ticker}/analyze`) and spot-check one memo for the
      `Fundamental Factor Scorecard` section (n/a with a soft degradation until the first
      scorecard run has rows) and unchanged rating logic.
- [ ] Scorecard first run: `POST /api/admin/scorecard/refresh` (202), then watch
      `scorecard_runs` via cron-health notes (`written=`) — the loop drains within minutes;
      `/api/scorecard` answers 404 until a run succeeds.
- [ ] Taxonomy: `python -m app.scripts.import_gics_taxonomy --status` from a shell on the
      worker (or the admin import route once slice 5 lands) shows the bundled version active.
- [ ] Frontend: `/app/scorecard`, `/app/research?ticker=…` memo section, Fundamentals
      Explorer, provider-health banner. No console errors.

## 4b. The production incident this branch also carries

`fix/snapshot-scan-oom` (commits `f66c86a`, `92b8a8a`) is merged here, but it
is **independently deployable from `main`** and should ship first: the worker
was OOM-killed hourly on 2026-09-10 and the fix does not need the rest of this
branch. After deploying it, confirm from the Render events API that
`reason.oomKilled` stops recurring (the email only fires once, so the inbox is
not the signal), and watch the first `snapshot_gc` note in cron-health — a run
that reports `capped=1` for several days means retention is not keeping up
with write volume.

Unrelated to the deploy and **owner-only**: four provider API keys
(FMP, Alpha Vantage, Polygon, Census) were written to Render's logs in
plaintext and still need rotating. The code fix stops the leak; it does not
undo it.

### The first Industry Analysis Sunday will look wrong, and is not

Most groups will report `insufficient_sample` until the weekly price warm-up budget
(`INDUSTRY_PRICE_WARMUP_BUDGET`, 150 calls) has filled coverage over several weeks. In the
demo universe only 5 of 25 groups clear the sample floor on a first run. That is the
labelled state working as designed — the alternative is a number computed from three
companies presented as an industry's return. Decide deliberately whether to raise the
budget for the first few weeks; do not read it as a failed deploy.

Also expect, and do not treat as errors, these labelled degradations on early editions:
`generation:deterministic_final_attempt:<n>` (the week was rescued by the no-LLM writer),
and `cross_industry:snapshot:prior_period:<key>` / `companies:events:prior_period:<key>` /
`outlook:macro_regime:prior_period:<key>` (a group report generated before its own period's
snapshot exists quotes the prior one, and says so).

## 5. Rollback

1. Render → redeploy the previous image on both services (web first, then worker).
2. Flags: nothing to unset — every new feature defaults off or degrades to n/a.
3. Data: new tables (`scorecard_*`, `price_month_ends`, `gics_*`, `company_industry_classifications`,
   `industry_*`) are additive and harmless to the previous image; leave them.
4. Memos regenerated on the new image carry a `scorecard` field the old frontend ignores.
5. If a regen storm was started in step 4, `POST /api/admin/regen-jobs` state can be
   inspected via `GET /api/admin/regen-jobs`; the worker drains at most its configured concurrency.

## 6. Owner decisions still open (recorded, not blocking this rollout)

- Clerk + Stripe credentials, legal review and pricing page copy before `AUTH_ENABLED`.
- `PUBLIC_BASE_URL` / `marketmosaic.ai :443`, `ABUSE_HASH_SALT`, `og.png`.
- Free-tier DCF meter, unit-cost measurement.
- GICS display/data rights; industry access tier, benchmarks, publication time, auto-publish vs review.
- Render API token rotation/scoping (never printed, never rotated by an agent).
