# CLAUDE.md — MarketMosaic

Guidance for AI agents working in this repo.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec

## Testing

Run the backend suite the way CI does, from `backend/`:

```
ENABLE_LIVE_DATA=false USE_DEMO_DATA=true python -m pytest -q
```

Two things to know before you trust a local run:

- **Blank the LLM keys.** A developer `.env` carries live `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY`, and the memo tests will spend real money against them.
  CI has no keys, so pass `OPENAI_API_KEY="" ANTHROPIC_API_KEY="" GEMINI_API_KEY=""`
  to match it. Blanking breaks no test: `test_config_load_order` used to fail
  under it, but since FIX-010 it takes `OPENAI_API_KEY` out of the environment
  itself, so a failure there is now a real regression.
- **The harness blocks outbound sockets.** `app/tests/netguard.py` (installed by
  `conftest.py`) refuses every non-loopback connect and DNS lookup and lists the
  tests that reached for the network in the terminal summary. Providers are
  already silent under the `USE_DEMO_DATA=true` / `ENABLE_LIVE_DATA=false` pair;
  the guard makes that a property. `RUN_LIVE_TESTS=1` (the `live` marker's
  opt-in) or `MM_ALLOW_NETWORK=1` lifts it.
- **Use an isolated database, and a NEW one each run.** The default sqlite file
  is shared, so two concurrent runs produce flaky memo-version failures. Pass
  `DATABASE_URL="sqlite:////tmp/<something-unique>.db"`.

  Re-using a file a previous run populated breaks the suite the same way, and
  the symptom is misleading: tests that assert a first write is version 1 fail
  with bare arithmetic (`assert 6 == 2`) in `test_memo_store`,
  `test_outcome_tracking` and `test_scorecard_memo_integration`, minutes into
  the run and nowhere near the cause. That cost a real investigation once. The
  harness now counts pre-existing rows at session start and says so in the
  terminal summary, so check for the "database was not empty" section before
  treating any of those failures as a regression.

CI installs from `requirements.txt`, which pins floors rather than ceilings, so
**CI resolves newer dependencies than a long-lived dev environment**. Run the
suite against those versions before trusting a green local run — on 2026-09-11
that difference was hiding a production defect in which the per-IP rate limiter
silently enforced nothing at all (see `rate_limit._find_route_handler`), and
every configuration-level test still passed. Tests that
introspect framework internals can pass locally and fail there. To reproduce CI's
versions cheaply:

```
python -m venv --system-site-packages /tmp/civenv
/tmp/civenv/bin/pip install fastapi==<ci-version> starlette==<ci-version>
```

## Production

Two Render services share one Docker image, differing only by entrypoint:

- **web** (`marketmosaic`) — uvicorn. Serves the API and the built frontend.
- **worker** (`marketmosaic-worker`) — `python -m app.worker`. Owns the memo-regen
  queue and every monitoring loop in `app/monitoring/__init__.py::KNOWN_LOOPS`
  (21: FEAT-003's `industry_classification_loop` + `industry_weekly_loop`,
  plus `snapshot_gc`. `test_worker_service` and
  `test_cron_health_cross_process` both pin the list to what `register_all`
  actually registers, so the count here is documentation, not a contract).

They coordinate only through Postgres. Keep `ENABLE_MONITORING` and
`ENABLE_REGEN_WORKER` **false** on web and **true** on the worker; flipping either
re-couples memory profiles that were deliberately separated.

Because state is split across two processes, **anything held in a module-level dict
is invisible to the other one**. That has caused real outages: `/api/admin/cron-health`
silently reported zero loops, and the LLM circuit-breaker endpoints describe only the
process that served the request. Cross-process state belongs in the database.

`/api/admin/*` requires `ADMIN_API_TOKEN` (bearer). Five endpoints are exempt because
the browser calls them — see `app/api/admin_auth.EXEMPT_PREFIXES`. **Check
`frontend/src/` before tightening anything under that prefix**; guarding a
browser-called endpoint breaks user-facing pages rather than securing anything.
