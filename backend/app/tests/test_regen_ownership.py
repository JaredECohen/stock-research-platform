"""Ownership survives process overlap; stale attempts cannot spend or publish."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.agents import llm
from app.agents.safe_runner import safe_call
from app.database import Base
from app.models import MemoRunCheckpoint, MemoSnapshot, RegenJob
from app.services import checkpoint_store, memo_store
from app.services import regen_lease as lease
from app.services import regen_worker as worker


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'ownership.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    for module in (worker, lease, memo_store, checkpoint_store):
        monkeypatch.setattr(module, "SessionLocal", factory)
    monkeypatch.setattr(worker, "_introduce_ticker", lambda *args: None)
    yield factory, engine
    engine.dispose()


def claim(database):
    job, _ = worker.enqueue("LEASE")
    receipt = worker.claim_next_job()
    assert receipt.job_id == job["id"]
    return receipt


def expire(database, receipt):
    with database[0]() as db:
        db.get(RegenJob, receipt.job_id).lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
        db.commit()


def fake_memo(ticker="LEASE"):
    return SimpleNamespace(ticker=ticker, rating_label="Neutral",
        model_dump_json=lambda: json.dumps({"ticker": ticker, "rating_label": "Neutral"}))


def snapshot(database):
    with database[0]() as db:
        return [{c.name: getattr(row, c.name) for c in RegenJob.__table__.columns}
                for row in db.execute(select(RegenJob)).scalars()]


def test_new_process_cannot_recover_live_predecessor(database):
    receipt = claim(database)
    before = snapshot(database)
    env = {**os.environ, "DATABASE_URL": str(database[1].url), "ENABLE_LIVE_DATA": "false", "USE_DEMO_DATA": "true",
           "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": ""}
    code = "from app.services import regen_worker as w; import json; print(json.dumps({'recovery':w.recover_orphans(),'claim':w.claim_next_job()}))"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(result.stdout.splitlines()[-1]) == {"recovery": {"requeued": 0, "failed": 0, "expired": 0}, "claim": None}
    assert snapshot(database) == before
    assert lease.renew(receipt)


def test_expired_takeover_rejects_old_renew_execution_and_all_db_writes(database, monkeypatch):
    old = claim(database)
    with database[0]() as db:
        run_id = db.get(RegenJob, old.job_id).run_id
    with lease.claim_context(old):
        checkpoint_store.save_step(run_id, "done", payload={"owner": "old"})
    expire(database, old)
    assert not lease.renew(old)  # Even before another owner claims it.
    assert worker.recover_orphans()["requeued"] == 1
    new = worker.claim_next_job()
    assert new.job_id == old.job_id and new.owner_token != old.owner_token
    with lease.claim_context(new):
        checkpoint_store.save_step(run_id, "done", payload={"owner": "new"})
    before = snapshot(database)
    monkeypatch.setattr(worker, "run_stock_memo", lambda *a, **k: pytest.fail("stale graph executed"))
    monkeypatch.setattr(worker, "_finalize_charge", lambda *a, **k: pytest.fail("stale charge changed"))
    assert worker.execute_job(old)["status"] == "lease_lost"
    assert worker._finish_claim(old, RuntimeError("old failure")) is None
    with lease.claim_context(old):
        with pytest.raises(lease.LeaseLost):
            worker._append_progress(old.job_id, "old progress")
        with pytest.raises(lease.LeaseLost):
            checkpoint_store.save_step(run_id, "done", payload={"stale": True})
        with pytest.raises(lease.LeaseLost):
            memo_store.save_memo(fake_memo())
    assert snapshot(database) == before
    with database[0]() as db:
        assert not list(db.execute(select(MemoSnapshot)).scalars())
        rows = list(db.execute(select(MemoRunCheckpoint)).scalars())
        assert len(rows) == 1 and rows[0].payload == {"owner": "new"}


def test_stale_cancellation_bypasses_fallback_and_prevents_provider_dispatch(database, monkeypatch):
    receipt = claim(database)
    expire(database, receipt)
    monkeypatch.setattr(llm, "_openai_client", lambda: pytest.fail("provider dispatched"))
    monkeypatch.setattr(llm, "_anthropic_client", lambda: pytest.fail("provider dispatched"))
    with lease.claim_context(receipt):
        for fn, provider in ((llm._call_json, "openai"), (llm._call_text, "anthropic")):
            with pytest.raises(lease.LeaseLost):
                safe_call(fn, provider, prompt="x", system="x", route="cheap", max_tokens=1, model=None, fallback=None)
        with llm.llm_call_context(run_id="run"):
            @checkpoint_store.checkpointed("test")
            def step():
                pytest.fail("stale checkpoint step executed")
            with pytest.raises(lease.LeaseLost):
                safe_call(step, fallback=None)


def test_publication_and_receipt_share_transaction_and_recovery_does_not_repeat_graph(database, monkeypatch):
    receipt = claim(database)
    with lease.claim_context(receipt):
        saved = memo_store.save_memo(fake_memo())
        again = memo_store.save_memo(fake_memo())
        assert again.id == saved.id
    with database[0]() as db:
        assert db.get(RegenJob, receipt.job_id).memo_version == saved.version
        assert len(list(db.execute(select(MemoSnapshot)).scalars())) == 1
    expire(database, receipt)
    monkeypatch.setattr(worker, "run_stock_memo", lambda *a, **k: pytest.fail("published job repeated graph"))
    assert worker.recover_orphans()["published_recovered"] == 1
    assert worker.process_next_job() is None
    with database[0]() as db:
        assert db.get(RegenJob, receipt.job_id).status == "succeeded"
        assert len(list(db.execute(select(MemoSnapshot)).scalars())) == 1


def test_snapshot_and_receipt_both_roll_back_on_commit_failure(database):
    receipt = claim(database)
    factory, engine = database
    def fail_commit(session):
        raise RuntimeError("commit unavailable")
    event.listen(factory, "before_commit", fail_commit)
    try:
        with lease.claim_context(receipt), pytest.raises(RuntimeError, match="commit unavailable"):
            memo_store.save_memo(fake_memo())
    finally:
        event.remove(factory, "before_commit", fail_commit)
    with factory() as db:
        assert db.get(RegenJob, receipt.job_id).memo_version is None
        assert not list(db.execute(select(MemoSnapshot)).scalars())


def test_caller_owned_rollback_preserves_snapshot_and_checkpoint_boundaries(database):
    receipt = claim(database)
    with database[0]() as db:
        run_id = db.get(RegenJob, receipt.job_id).run_id
        with lease.claim_context(receipt):
            memo_store.save_memo(fake_memo(), db=db)
            checkpoint_store.save_step(run_id, "new", payload={"value": 1}, db=db)
        db.rollback()
    with database[0]() as db:
        assert db.get(RegenJob, receipt.job_id).memo_version is None
        assert not list(db.execute(select(MemoSnapshot)).scalars())
        assert not list(db.execute(select(MemoRunCheckpoint)).scalars())


def test_lease_keeper_renews_during_blocking_work(database, monkeypatch):
    receipt = claim(database)
    monkeypatch.setattr(lease, "RENEW_SECONDS", 0.01)
    with database[0]() as db:
        db.get(RegenJob, receipt.job_id).lease_expires_at = datetime.utcnow() + timedelta(seconds=5)
        db.commit()
    with lease.keep_alive(receipt):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with database[0]() as db:
                remaining = (db.get(RegenJob, receipt.job_id).lease_expires_at - datetime.utcnow()).total_seconds()
            if remaining > 100:
                break
            time.sleep(0.01)
        assert remaining > 100
        assert worker.recover_orphans()["requeued"] == 0


def test_legacy_unowned_running_jobs_are_named_and_never_recovered(database, caplog):
    receipt = claim(database)
    with database[0]() as db:
        row = db.get(RegenJob, receipt.job_id)
        row.owner_token = row.lease_expires_at = None
        row.started_at = datetime.utcnow() - timedelta(days=1)
        db.commit()
    before = snapshot(database)
    result = worker.recover_orphans()
    assert result["requeued"] == result["failed"] == 0
    assert [r["id"] for r in result["legacy_deferred"]] == [receipt.job_id]
    assert "LEASE" in caplog.text and "legacy unowned" in caplog.text
    assert snapshot(database) == before


def test_worker_finishes_its_exact_publication_and_keeps_success_after_postpublish_error(database, monkeypatch):
    job, _ = worker.enqueue("LEASE")
    def graph(*args, **kwargs):
        memo_store.save_memo(fake_memo())
        raise RuntimeError("after publication")
    monkeypatch.setattr(worker, "run_stock_memo", graph)
    done = worker.process_next_job()
    assert done["id"] == job["id"] and done["status"] == "succeeded" and done["memo_version"] == 1
    with database[0]() as db:
        assert len(list(db.execute(select(MemoSnapshot)).scalars())) == 1


def test_graph_return_without_owned_publication_is_failure(database, monkeypatch):
    worker.enqueue("LEASE")
    monkeypatch.setattr(worker, "run_stock_memo", lambda *a, **k: fake_memo())
    done = worker.process_next_job()
    assert done["status"] == "failed" and done["memo_version"] is None
    assert "publication receipt" in done["error_message"]


def test_stale_embedding_dispatch_is_cancelled_without_hash_fallback(database, monkeypatch):
    import openai

    from app.services import embeddings

    receipt = claim(database)
    expire(database, receipt)
    monkeypatch.setattr(embeddings, "_is_openai_available", lambda: True)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: SimpleNamespace(embeddings=SimpleNamespace(
        create=lambda **kw: pytest.fail("stale embedding request dispatched"))))
    monkeypatch.setattr(embeddings, "_hash_embed", lambda *a: pytest.fail("cancellation degraded to hash fallback"))
    with lease.claim_context(receipt), pytest.raises(lease.LeaseLost):
        embeddings.embed(["text"])


def test_expired_job_is_recovered_during_existing_idle_poll(database, monkeypatch):
    receipt = claim(database)
    assert worker.process_next_job() is None
    expire(database, receipt)
    def graph(*args, **kwargs):
        memo = fake_memo()
        memo_store.save_memo(memo)
        return memo
    monkeypatch.setattr(worker, "run_stock_memo", graph)
    done = worker.process_next_job()
    assert done["status"] == "succeeded" and done["attempts"] == 2
    assert done["memo_version"] == 1


def test_lease_columns_are_added_to_populated_legacy_table(database, monkeypatch):
    from sqlalchemy import inspect, text

    from app import database as database_module

    receipt = claim(database)
    engine = database[1]
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE regen_jobs DROP COLUMN owner_token"))
        connection.execute(text("ALTER TABLE regen_jobs DROP COLUMN lease_expires_at"))
    monkeypatch.setattr(database_module, "engine", engine)
    added = database_module.reconcile_missing_columns()
    assert {"regen_jobs.owner_token", "regen_jobs.lease_expires_at"} <= set(added)
    assert all(c["nullable"] for c in inspect(engine).get_columns("regen_jobs")
               if c["name"] in {"owner_token", "lease_expires_at"})
    with database[0]() as db:
        row = db.get(RegenJob, receipt.job_id)
        assert row.status == "running" and row.owner_token is None and row.lease_expires_at is None
    assert worker.recover_orphans()["legacy_deferred"][0]["id"] == receipt.job_id


def test_transient_renewal_error_retries_without_surrendering_live_claim(database, monkeypatch):
    import threading

    receipt = claim(database)
    original = lease.renew
    renewed = threading.Event()
    calls = []
    def transient(claim):
        calls.append(claim)
        if len(calls) == 1:
            raise RuntimeError("temporary connection failure")
        result = original(claim)
        renewed.set()
        return result
    monkeypatch.setattr(lease, "renew", transient)
    monkeypatch.setattr(lease, "RENEW_SECONDS", 0.01)
    with lease.keep_alive(receipt):
        assert renewed.wait(3)
        assert worker.recover_orphans()["requeued"] == 0
    assert len(calls) >= 2


def test_completion_uses_owned_snapshot_not_a_later_other_publication(database, monkeypatch):
    worker.enqueue("LEASE")
    def graph(*args, **kwargs):
        memo = fake_memo()
        own = memo_store.save_memo(memo)
        with database[0]() as db:
            db.add(MemoSnapshot(ticker="LEASE", version=own.version + 1, memo_json={"other": True}))
            db.commit()
        return memo
    monkeypatch.setattr(worker, "run_stock_memo", graph)
    done = worker.process_next_job()
    assert done["status"] == "succeeded" and done["memo_version"] == 1
    with database[0]() as db:
        rows = list(db.execute(select(MemoSnapshot).order_by(MemoSnapshot.version)).scalars())
        assert [row.version for row in rows] == [1, 2] and rows[1].memo_json == {"other": True}
