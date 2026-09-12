"""The research-run charge lifecycle through the regen queue (FEAT-002, S2).

`POST /analyze` reserves; the worker commits on success and releases on
failure, `WorkerRestart` and `QueueExpired`; a coalesced second caller
is never charged; a Free user's second run of the month is 402 with no
job row. Lazy universe resolution happens in the worker, not the request.
The graph is stubbed throughout — these are charge-mechanics tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agents import llm
from app.api import routes_stocks
from app.auth import usage
from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import RegenJob, UsageEvent
from app.services import regen_worker
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth
from app.tests.gating_helpers import (
    assert_structured,
    free_user,
    pro_user,
    purge_jobs,
    seed_demo_universe,
    usage_events,
    user_id_for,
)

COLD = "ZZCOLD"  # never in `companies`


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def client():
    seed_demo_universe()
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_queue():
    purge_jobs()
    yield
    purge_jobs()


@pytest.fixture()
def stub_graph(monkeypatch):
    """A memo run that records the call context instead of thinking."""
    calls: list[dict] = []

    def fake_run(ticker, *, scenario="soft_landing", force_refresh=False, run_id=None, **_kw):
        calls.append({"ticker": ticker, "run_id": run_id, "ctx": dict(llm.current_call_context())})
        return SimpleNamespace(ticker=ticker, rating_label="Neutral")

    monkeypatch.setattr(regen_worker, "run_stock_memo", fake_run)
    import app.services.memo_store as memo_store
    monkeypatch.setattr(memo_store, "latest_memo",
                        lambda _t, **_kw: SimpleNamespace(version=3, generated_at=datetime.utcnow()))
    return calls


@pytest.fixture()
def no_provider_work(monkeypatch):
    """Provider-side universe work must not happen for tickers the DB
    already knows; the request path must not do it at all under the wall."""
    def boom(*_a, **_kw):
        raise AssertionError("provider work ran for a known ticker")

    monkeypatch.setattr(regen_worker, "ensure_company_in_universe", boom)
    monkeypatch.setattr(regen_worker, "backfill_ticker", boom)
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe",
                        lambda _t: (_ for _ in ()).throw(AssertionError("lazy universe ran in the request")))


def _event(event_id: int) -> UsageEvent:
    with SessionLocal() as db:
        ev = db.get(UsageEvent, event_id)
        db.expunge(ev)
        return ev


def _job(job_id: int) -> RegenJob:
    with SessionLocal() as db:
        row = db.get(RegenJob, job_id)
        db.expunge(row)
        return row


def _analyze(client, tok, ticker="MSFT"):
    return client.post(f"/api/stocks/{ticker}/analyze", headers=bearer(tok))


# ---------------------------------------------------------------------------
# Success / failure / orphan
# ---------------------------------------------------------------------------

def test_success_commits_the_charge_and_attributes_llm_calls(auth_on, client, stub_graph, no_provider_work):
    _sub, tok = pro_user(client, auth_on)
    uid = user_id_for(client, tok)
    resp = _analyze(client, tok)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "started" and body["charged"] is True
    job = _job(body["job_id"])
    assert job.requested_by_user_id == uid
    assert _event(job.usage_event_id).status == "reserved"

    done = regen_worker.process_next_job()
    assert done["status"] == "succeeded", done
    assert done["requested_by_user_id"] == uid and done["usage_event_id"] == job.usage_event_id
    assert _event(job.usage_event_id).status == "committed"
    ctx = stub_graph[0]["ctx"]
    assert ctx["user_id"] == uid and ctx["feature"] == "research_run" and ctx["run_id"] == job.run_id
    with SessionLocal() as db:
        assert usage.used(db, uid, "research_run", usage.period_key()) == 1


def test_failure_releases_the_charge(auth_on, client, monkeypatch, no_provider_work):
    def boom(*_a, **_kw):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(regen_worker, "run_stock_memo", boom)
    _sub, tok = pro_user(client, auth_on)
    uid = user_id_for(client, tok)
    job = _job(_analyze(client, tok).json()["job_id"])
    done = regen_worker.process_next_job()
    assert done["status"] == "failed" and done["error_type"] == "RuntimeError"
    assert _event(job.usage_event_id).status == "released"
    with SessionLocal() as db:
        assert usage.used(db, uid, "research_run", usage.period_key()) == 0


def test_repeat_orphan_releases_the_charge(auth_on, client, no_provider_work):
    _sub, tok = pro_user(client, auth_on)
    job = _job(_analyze(client, tok).json()["job_id"])
    assert regen_worker.claim_next_job() == job.id
    assert regen_worker.recover_orphans()["requeued"] == 1
    # The retry may still deliver: the reservation stays.
    assert _event(job.usage_event_id).status == "reserved"
    assert regen_worker.claim_next_job() == job.id
    assert regen_worker.recover_orphans()["failed"] == 1
    assert _job(job.id).error_type == "WorkerRestart"
    assert _event(job.usage_event_id).status == "released"


def test_expired_queue_entry_releases_the_charge(auth_on, client, no_provider_work):
    _sub, tok = pro_user(client, auth_on)
    job = _job(_analyze(client, tok).json()["job_id"])
    with SessionLocal() as db:
        row = db.get(RegenJob, job.id)
        row.enqueued_at = datetime.utcnow() - timedelta(hours=3)
        db.commit()
    assert regen_worker.recover_orphans()["expired"] == 1
    assert _job(job.id).error_type == "QueueExpired"
    assert _event(job.usage_event_id).status == "released"


def test_commit_and_release_are_idempotent_after_finish(auth_on, client, stub_graph, no_provider_work):
    _sub, tok = pro_user(client, auth_on)
    job = _job(_analyze(client, tok).json()["job_id"])
    regen_worker.process_next_job()
    with SessionLocal() as db:
        assert usage.release(db, job.usage_event_id) is False
        assert usage.commit(db, job.usage_event_id) is False
    assert _event(job.usage_event_id).status == "committed"


# ---------------------------------------------------------------------------
# Coalescing and allowances
# ---------------------------------------------------------------------------

def test_coalesced_second_user_is_not_charged(auth_on, client, stub_graph, no_provider_work):
    _a, tok_a = pro_user(client, auth_on)
    _b, tok_b = pro_user(client, auth_on)
    uid_a, uid_b = user_id_for(client, tok_a), user_id_for(client, tok_b)
    first = _analyze(client, tok_a).json()
    second = _analyze(client, tok_b)
    assert second.status_code == 202
    assert second.json()["status"] == "in_progress" and second.json()["charged"] is False
    assert second.json()["job_id"] == first["job_id"]
    # B's reservation was released; A's stands and is committed by the run.
    assert [e.status for e in usage_events(uid_b, "research_run")] == ["released"]
    with SessionLocal() as db:
        assert usage.used(db, uid_b, "research_run", usage.period_key()) == 0
    regen_worker.process_next_job()
    assert [e.status for e in usage_events(uid_a, "research_run")] == ["committed"]
    assert _job(first["job_id"]).requested_by_user_id == uid_a


def test_free_second_run_is_402_with_no_job_row(auth_on, client, stub_graph, no_provider_work):
    _sub, tok = free_user(auth_on)
    first = _analyze(client, tok, "MSFT")
    assert first.status_code == 202, first.text
    regen_worker.process_next_job()
    second = _analyze(client, tok, "NVDA")
    detail = assert_structured(second, code="quota_exceeded", status=402)
    assert detail["feature"] == "research_run" and detail["used"] == 1 and detail["limit"] == 1
    assert detail["plan"] == "free" and detail["resets_at"]
    with SessionLocal() as db:
        assert db.query(RegenJob).filter(RegenJob.ticker == "NVDA").count() == 0


def test_unverified_email_cannot_start_a_run(auth_on, client, no_provider_work):
    tok = auth_on.token(verified=False)
    assert_structured(_analyze(client, tok), code="email_unverified", status=403)
    with SessionLocal() as db:
        assert db.query(RegenJob).count() == 0


def test_anonymous_analyze_is_401(auth_on, client):
    assert_structured(client.post("/api/stocks/MSFT/analyze"), code="auth_required", status=401)


# ---------------------------------------------------------------------------
# Lazy universe resolution moves into the worker
# ---------------------------------------------------------------------------

def test_worker_introduces_a_cold_ticker_before_the_run(auth_on, client, stub_graph, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(regen_worker, "ensure_company_in_universe",
                        lambda t: order.append(f"introduce:{t}") or {"ticker": t})
    monkeypatch.setattr(regen_worker, "backfill_ticker", lambda t: order.append(f"backfill:{t}") or {})
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe",
                        lambda _t: (_ for _ in ()).throw(AssertionError("lazy universe ran in the request")))
    _sub, tok = pro_user(client, auth_on)
    job = _job(_analyze(client, tok, COLD).json()["job_id"])
    done = regen_worker.process_next_job()
    assert done["status"] == "succeeded", done
    assert order == [f"introduce:{COLD}", f"backfill:{COLD}"]
    assert stub_graph[0]["ticker"] == COLD
    steps = [p["step"] for p in done["progress"]]
    assert steps.index("ticker_introduced") < steps.index("calling_run_stock_memo")
    assert _event(job.usage_event_id).status == "committed"


def test_rejected_symbol_fails_the_job_and_releases(auth_on, client, stub_graph, monkeypatch):
    monkeypatch.setattr(regen_worker, "ensure_company_in_universe", lambda _t: None)
    monkeypatch.setattr(regen_worker, "backfill_ticker",
                        lambda _t: (_ for _ in ()).throw(AssertionError("backfill without a profile")))
    _sub, tok = pro_user(client, auth_on)
    job = _job(_analyze(client, tok, COLD).json()["job_id"])
    done = regen_worker.process_next_job()
    assert done["status"] == "failed" and done["error_type"] == "ValueError"
    assert "rejected this symbol" in done["error_message"]
    assert stub_graph == []
    assert _event(job.usage_event_id).status == "released"
    assert regen_worker.ticker_status(COLD)["last_failure"]["error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# Wall off: nothing changes
# ---------------------------------------------------------------------------

def test_wall_off_resolves_the_universe_in_the_request_and_charges_nothing(client, stub_graph, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    seen: list[str] = []
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe", lambda t: seen.append(t) or "auto_analysis")
    monkeypatch.setattr(regen_worker, "ensure_company_in_universe",
                        lambda _t: (_ for _ in ()).throw(AssertionError("worker introduced a known ticker")))
    with SessionLocal() as db:
        events_before = db.query(UsageEvent).count()
    resp = client.post("/api/stocks/MSFT/analyze")
    assert resp.status_code == 202
    assert resp.json()["charged"] is False
    assert seen == ["MSFT"]
    job = _job(resp.json()["job_id"])
    assert job.requested_by_user_id is None and job.usage_event_id is None
    done = regen_worker.process_next_job()
    assert done["status"] == "succeeded"
    assert stub_graph[0]["ctx"]["user_id"] is None
    with SessionLocal() as db:
        assert db.query(UsageEvent).count() == events_before  # no meter row written by either side


def test_wall_off_sync_still_runs_inline(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe", lambda _t: "auto_analysis")
    from app.tests.factories import make_memo
    monkeypatch.setattr(routes_stocks, "run_stock_memo",
                        lambda t, **_kw: make_memo(ticker=t, company_name="Microsoft"))
    resp = client.post("/api/stocks/MSFT/analyze?sync=true")
    assert resp.status_code == 200, resp.text
    assert resp.json()["ticker"] == "MSFT"
