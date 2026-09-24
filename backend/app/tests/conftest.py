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

import os
import sys

import pytest

from app.services.data_service import get_data_service
from app.tests import netguard
from app.tests.fixtures.demo_provider import DemoProvider

# Installed at import so background threads started under a TestClient
# lifespan are covered too; a no-op under RUN_LIVE_TESTS=1 / MM_ALLOW_NETWORK=1.
netguard.install()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    netguard.set_current(item.nodeid)
    yield
    netguard.set_current(f"<after {item.nodeid}>")


_REUSED_DB_TABLES = ("memo_snapshots", "research_snapshots")
_reused_db: dict[str, int] = {}


def _check_database_is_fresh() -> None:
    """Record whether the target database already holds this suite's data.

    Several tests assert a first write lands at version 1, or that a query
    returns exactly the rows they just seeded. Those assertions are only
    true against a fresh database, which is why CLAUDE.md says to pass a
    unique `DATABASE_URL`. Re-run the suite against a database a previous
    run populated and four unrelated tests fail with arithmetic that looks
    like nondeterminism (`assert 2 == 1`) five minutes in — a real
    investigation once went looking for a race that was never there.

    So: detect it up front and say so, rather than letting the symptom
    masquerade as a flake. A warning rather than a hard error, because
    re-running against a populated development database is a legitimate
    thing to do deliberately.
    """
    try:
        from sqlalchemy import text

        from app.database import engine
        with engine.connect() as conn:
            for table in _REUSED_DB_TABLES:
                try:
                    n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
                except Exception:
                    continue          # table not created yet: a fresh DB
                if n:
                    _reused_db[table] = int(n)
    except Exception:                  # pragma: no cover — diagnostics only
        return


def pytest_sessionstart(session):
    _check_database_is_fresh()


def pytest_terminal_summary(terminalreporter):

    if _reused_db:
        terminalreporter.section("database was not empty when this run started")
        for table, n in sorted(_reused_db.items()):
            terminalreporter.line(f"{table}: {n} row(s) already present")
        terminalreporter.line(
            "Tests that assert a first write is version 1, or that count the rows they "
            "just seeded, fail against a reused database — most visibly test_memo_store, "
            "test_outcome_tracking and test_scorecard_memo_integration. Pass a unique "
            "DATABASE_URL (see CLAUDE.md) before treating those failures as real."
        )

    # A green run must not be read as "token budgets were checked against the
    # embedding model's vocabulary". The cl100k_base ranks are vendored
    # (FIX-012), so the encoder loads offline and the encoder-specific tests
    # assert rather than skip. If it still failed to load, `count_tokens`
    # degraded to its character heuristic for every other test too, and that
    # degradation is otherwise silent — so name the file that failed.
    embeddings = sys.modules.get("app.services.embeddings")
    if embeddings is not None and embeddings._encoding.cache_info().currsize:
        if embeddings._encoding() is None:
            vendored = embeddings.VENDORED_TIKTOKEN_FILE
            state = "present" if vendored.is_file() else "MISSING"
            terminalreporter.section("tiktoken encoding unavailable")
            terminalreporter.line(
                f"count_tokens ran on its character heuristic for this whole session: the "
                f"vendored cl100k_base ranks at {vendored} ({state}) did not load, and the "
                "network fallback failed too (the netguard refuses it). tiktoken deletes a cache file whose "
                "SHA-256 does not match its pin, so check `git status` for a deletion and "
                "run test_tiktoken_vendored.py, which says whether the bytes or the pin "
                f"moved. TIKTOKEN_CACHE_DIR={os.environ.get('TIKTOKEN_CACHE_DIR')!r} "
                "overrides the vendored directory when set."
            )

    offenders = netguard.hits()
    if not offenders:
        return
    terminalreporter.section("netguard: outbound network attempts (blocked)")
    for test_id, targets in sorted(offenders.items()):
        terminalreporter.line(f"{test_id}: {', '.join(targets)}")
    terminalreporter.line(f"{len(offenders)} test(s) reached for the network; each attempt was refused.")


@pytest.fixture(autouse=True, scope="session")
def _create_tables():
    """Create all ORM tables before any test runs.

    CI starts from a fresh DB (`rm -f marketmosaic.db`). Tables are
    otherwise only created lazily by `init_db()` at app startup — which
    fires for TestClient-based tests but NOT for tests that import a
    service module directly (e.g. `test_as_of_clipping` calling
    `data_service.get_filings`, whose path queries the `companies`
    table). Those failed with "no such table: companies" purely on
    collection order — whichever ran before the first TestClient boot.
    Creating the schema once at session start makes the suite
    order-independent.
    """
    from app.database import init_db
    init_db()
    yield


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
