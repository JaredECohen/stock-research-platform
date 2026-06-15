"""Theme 5 — nightly regen smoke test (live, opt-in).

Exercises the exact production regeneration path end-to-end: POST
/analyze enqueues a durable RegenJob, the worker body
(`process_next_job`, called directly so the test is deterministic —
same code the polling thread runs) executes a full live memo, and the
/analyze/status polling contract reports completion the way
Research.tsx consumes it.

This is the regression net for the failure class that dominated the
firefighting commits (OOM kills, silent save failures, stuck breaker):
if regen breaks anywhere between the HTTP request and the persisted
snapshot, this fails nightly with the job's error telemetry in the
assertion message instead of a user reporting an infinite spinner.

Gated like the rest of the live suite: skipped unless RUN_LIVE_TESTS=1;
runs one full live memo (~5-9 min, real LLM spend) in nightly-live.yml.
"""
from __future__ import annotations

import pytest

from app.config import settings

pytestmark = pytest.mark.live

# Cheap-ish, data-rich, always in the demo+live universe. Distinct from
# test_live_integration's AAPL run so a ticker-specific data issue
# doesn't take out both canaries at once.
SMOKE_TICKER = "MSFT"


def _require_key(name: str, attr: str) -> str:
    val = getattr(settings, attr, None) or ""
    if not val:
        pytest.skip(f"{name} not configured (set {name} in .env to enable)")
    return val


def test_regen_queue_end_to_end(live_settings):
    _require_key("FMP_API_KEY", "fmp_api_key")
    if not settings.has_llm:
        pytest.skip("no LLM key configured (OPENAI_API_KEY / ANTHROPIC_API_KEY)")

    from fastapi.testclient import TestClient
    from app.database import init_db
    from app.main import app
    from app.services import regen_worker

    init_db()
    c = TestClient(app)

    # 1. Enqueue via the public endpoint, exactly as the frontend does.
    r = c.post(f"/api/stocks/{SMOKE_TICKER}/analyze")
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] in {"started", "in_progress"}
    started_at = body["started_at"]
    job_id = body["job_id"]

    s = c.get(f"/api/stocks/{SMOKE_TICKER}/analyze/status").json()
    assert s["in_progress"] is True

    # 2. Drain the queue with the worker's own job body (synchronous so
    # the test doesn't poll a thread; identical code path otherwise).
    done = regen_worker.process_next_job()
    assert done is not None and done["id"] == job_id
    assert done["status"] == "succeeded", (
        f"regen job failed: {done['error_type']}: {done['error_message']}\n"
        f"progress trail: {[p['step'] for p in done['progress']]}\n"
        f"traceback tail:\n{done['traceback_tail']}"
    )
    assert done["memo_version"] is not None

    # 3. Completion is visible through the polling contract: the
    # persisted memo timestamp advanced past the job's started_at,
    # which is the frontend's stop-polling condition.
    s2 = c.get(f"/api/stocks/{SMOKE_TICKER}/analyze/status").json()
    assert s2["in_progress"] is False
    assert s2["last_failure"] is None
    assert s2["latest_memo_at"] is not None and s2["latest_memo_at"] > started_at
    assert s2["latest_version"] == done["memo_version"]
    # Per-step telemetry made it into the trace (worker waypoints +
    # checkpoint completions), so a future silent death is localizable.
    steps = " ".join(p["step"] for p in s2["last_progress"])
    assert "run_stock_memo_returned" in steps

    # 4. The memo itself is servable and fresh.
    m = c.get(f"/api/stocks/{SMOKE_TICKER}/memo")
    assert m.status_code == 200
    assert int(m.headers["X-Memo-Version"]) == done["memo_version"]
