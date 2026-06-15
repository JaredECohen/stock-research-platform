"""Pytest harness — register the DemoProvider fixture for the test session.

Wave 9b: production no longer falls back to demo data when a live
provider misses; tests that hit `data_service` would otherwise return
empty results. The autouse fixture here injects `DemoProvider` at the
head of every capability chain so tests get deterministic, network-free
responses without any per-test wiring.

To exercise the live provider chain in a specific test, opt out:

    @pytest.mark.live_only
    def test_real_fmp_call(): ...

(See `test_live_marker_gate.py` for the marker-handling logic.)
"""
from __future__ import annotations

import pytest

from app.services.data_service import get_data_service
from app.tests.fixtures.demo_provider import DemoProvider


@pytest.fixture(autouse=True, scope="session")
def _register_demo_provider():
    """Wire the in-memory demo provider into `data_service` for the entire
    test session. Cleared on teardown so subprocess pytests don't leak
    state."""
    ds = get_data_service()
    provider = DemoProvider()
    ds.register_test_provider(provider)
    yield provider
    ds.register_test_provider(None)


@pytest.fixture(autouse=True, scope="session")
def _drain_regen_queue_on_exit():
    """Tests enqueue durable `RegenJob` rows (POST /analyze) but the
    worker thread is deliberately not started under pytest, so queued
    rows would persist in the shared sqlite file. Purge unfinished jobs
    at session end so the next `uvicorn` boot doesn't pick up a test's
    leftover job and burn a real LLM regen on it."""
    yield
    try:
        from app.database import SessionLocal
        from app.models import RegenJob
        with SessionLocal() as db:
            db.query(RegenJob).filter(
                RegenJob.status.in_(("queued", "running"))
            ).delete(synchronize_session=False)
            db.commit()
    except Exception:
        pass  # table may not exist if no test touched the queue
