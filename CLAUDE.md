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
  to match it. Doing so makes `test_config_load_order` fail — that failure is an
  artifact of the blanking, not a regression.
- **Use an isolated database.** The default sqlite file is shared, so two
  concurrent runs produce flaky memo-version failures. Pass
  `DATABASE_URL="sqlite:////tmp/<something-unique>.db"`.

CI installs from `requirements.txt`, which pins floors rather than ceilings, so
**CI resolves newer dependencies than a long-lived dev environment**. Tests that
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
  queue and all 15 monitoring loops.

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
