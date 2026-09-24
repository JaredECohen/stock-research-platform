"""Real DB overlap, expiry and publication fault tests; no providers or models."""
from __future__ import annotations

import contextvars
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.agents import llm
from app.agents.safe_runner import safe_call
from app.database import Base
from app.models import CrossIndustrySnapshot, IndustryReport, IndustryReportJob, IndustryStatSnapshot
from app.services import industry_analytics as analytics
from app.services import industry_lease as lease
from app.services import industry_legacy_recovery as legacy
from app.services import industry_report_store as store
from app.services import industry_report_worker as worker
from app.services import industry_snapshot as snapshots

PERIOD = "2026-W36"
AS_OF = worker.as_of_for_period(PERIOD)


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'industry-ownership.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    for module in (lease, worker, store, snapshots, analytics, legacy):
        monkeypatch.setattr(module, "SessionLocal", factory)
    monkeypatch.setattr(store.gics_registry, "resolve_version", lambda *a: SimpleNamespace(id=1))
    monkeypatch.setattr(store.gics_registry, "group", lambda *a, **k: None)
    yield factory, engine
    engine.dispose()


def queued(database, kind="group_report"):
    with database[0]() as db:
        row = IndustryReportJob(kind=kind, taxonomy_version_id=1,
            industry_group_code="4510" if kind == "group_report" else None,
            period_key=PERIOD, run_id="industry-run", status="queued", attempts=0)
        db.add(row)
        db.commit()
        return row.id


def claim(database, kind="group_report"):
    job_id = queued(database, kind)
    receipt = worker.claim_next_job()
    assert receipt.job_id == job_id
    return receipt


def expire(database, receipt):
    with database[0]() as db:
        db.get(IndustryReportJob, receipt.job_id).lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
        db.commit()


def capture(database):
    with database[0]() as db:
        return [{c.name: getattr(row, c.name) for c in IndustryReportJob.__table__.columns}
                for row in db.scalars(select(IndustryReportJob).order_by(IndustryReportJob.id))]


def report(receipt, generation=None):
    # An analyst-written edition by default: only those take the
    # latest-good flag (owner decision 1), and these tests are about the
    # publication receipt, not the display rule.
    return store.save_report(code="4510", period_key=PERIOD, as_of=AS_OF,
                             payload={"original": "facts"}, job_id=receipt.job_id,
                             generation=generation or {"generation_mode": "llm"})


def cross(payload=None):
    return snapshots._upsert(CrossIndustrySnapshot(taxonomy_version_id=1, period_key=PERIOD,
        as_of=AS_OF, payload=payload or {"original": "facts"}, schema_version=1,
        computed_at=datetime.utcnow(), stats_ids=[], report_versions={}))


def stats():
    return analytics._persist(IndustryStatSnapshot(taxonomy_version_id=1,
        industry_group_code="4510", period_key=PERIOD, as_of=AS_OF,
        inputs_hash="fixed", payload={"original": "facts"}))


def test_second_process_defers_live_predecessor_and_cannot_claim_it(database):
    receipt = claim(database)
    before = capture(database)
    env = {**os.environ, "DATABASE_URL": str(database[1].url), "ENABLE_LIVE_DATA": "false",
           "USE_DEMO_DATA": "true", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": ""}
    code = "from app.services import industry_report_worker as w; import json; print(json.dumps({'recovery':w.recover_orphans(),'claim':w.claim_next_job()}))"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                            timeout=60, check=True)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "recovery": {"requeued": 0, "failed": 0, "expired": 0}, "claim": None}
    assert capture(database) == before
    assert lease.renew(receipt)


@pytest.mark.parametrize("kind", ["group_report", "cross_snapshot"])
def test_expired_owner_cannot_renew_execute_write_or_finish_replacement(database, monkeypatch, kind):
    old = claim(database, kind)
    expire(database, old)
    assert not lease.renew(old)
    assert worker.recover_orphans()["requeued"] == 1
    with database[0]() as db:
        db.get(IndustryReportJob, old.job_id).not_before = None
        db.commit()
    new = worker.claim_next_job()
    assert new.job_id == old.job_id and new.owner_token != old.owner_token
    before = capture(database)
    monkeypatch.setattr(worker, "_run_group_report", lambda *a: pytest.fail("stale report executed"))
    monkeypatch.setattr(worker, "_run_cross_snapshot", lambda *a: pytest.fail("stale snapshot executed"))
    assert worker.execute_job(old)["status"] == "lease_lost"
    assert worker._finish_claim(old, error=RuntimeError("stale exception")) is None
    assert not lease.renew(old)
    with lease.claim_context(old):
        for write in (lambda: worker._append_progress(old.job_id, "old"), lambda: report(old), cross, stats):
            with pytest.raises(lease.LeaseLost):
                write()
    assert capture(database) == before
    with database[0]() as db:
        assert not list(db.scalars(select(IndustryReport)))
        assert not list(db.scalars(select(CrossIndustrySnapshot)))
        assert not list(db.scalars(select(IndustryStatSnapshot)))


@pytest.mark.parametrize("kind", ["group_report", "cross_snapshot"])
def test_publication_receipt_is_atomic_idempotent_and_prevents_post_crash_rerun(database, monkeypatch, kind):
    receipt = claim(database, kind)
    publish = (lambda: report(receipt)) if kind == "group_report" else cross
    with lease.claim_context(receipt):
        saved = publish()
        assert publish().id == saved.id
    field = "report_id" if kind == "group_report" else "snapshot_id"
    with database[0]() as db:
        assert getattr(db.get(IndustryReportJob, receipt.job_id), field) == saved.id
    expire(database, receipt)
    monkeypatch.setattr(worker, "_run_group_report", lambda *a: pytest.fail("published output repeated"))
    monkeypatch.setattr(worker, "_run_cross_snapshot", lambda *a: pytest.fail("published output repeated"))
    assert worker.recover_orphans()["published_recovered"] == 1
    assert worker.process_next_job() is None
    assert capture(database)[0]["status"] == "succeeded"


@pytest.mark.parametrize("kind", ["group_report", "cross_snapshot"])
def test_failed_publication_commit_rolls_back_output_receipt_and_prior_latest(database, kind):
    prior = report(SimpleNamespace(job_id=None)) if kind == "group_report" else cross({"before": True})
    receipt = claim(database, kind)
    def fail_commit(session):
        raise RuntimeError("commit unavailable")
    event.listen(database[0], "before_commit", fail_commit)
    try:
        with lease.claim_context(receipt), pytest.raises(RuntimeError, match="commit unavailable"):
            report(receipt) if kind == "group_report" else cross({"after": True})
    finally:
        event.remove(database[0], "before_commit", fail_commit)
    with database[0]() as db:
        job = db.get(IndustryReportJob, receipt.job_id)
        assert job.report_id is None and job.snapshot_id is None
        model = IndustryReport if kind == "group_report" else CrossIndustrySnapshot
        rows = list(db.scalars(select(model)))
        assert len(rows) == 1 and rows[0].id == prior.id
        if kind == "group_report":
            assert rows[0].is_latest_good
        else:
            assert rows[0].payload == {"before": True}


def test_error_after_committed_publication_finishes_without_retry(database):
    receipt = claim(database)
    with lease.claim_context(receipt):
        saved = report(receipt)
    final = worker._finish_claim(receipt, error=RuntimeError("later telemetry failed"))
    assert final["status"] == "succeeded" and final["report_id"] == saved.id
    assert worker.process_next_job() is None


def test_stale_cross_attempt_cannot_overwrite_replacement_snapshot(database):
    old = claim(database, "cross_snapshot")
    expire(database, old)
    worker.recover_orphans()
    with database[0]() as db:
        db.get(IndustryReportJob, old.job_id).not_before = None
        db.commit()
    new = worker.claim_next_job()
    with lease.claim_context(new):
        saved = cross({"owner": "replacement"})
    with lease.claim_context(old), pytest.raises(lease.LeaseLost):
        cross({"owner": "stale"})
    with database[0]() as db:
        assert db.get(CrossIndustrySnapshot, saved.id).payload == {"owner": "replacement"}


def test_paid_dispatch_and_parallel_context_cancel_without_fallback(database, monkeypatch):
    receipt = claim(database)
    expire(database, receipt)
    for client in ("_openai_client", "_anthropic_client", "_gemini_client"):
        monkeypatch.setattr(llm, client, lambda: pytest.fail("paid client dispatched"))
    with lease.claim_context(receipt):
        for fn, provider in ((llm._call_json, "openai"), (llm._call_text, "anthropic")):
            with pytest.raises(lease.LeaseLost):
                safe_call(fn, provider, prompt="x", system="x", route="cheap", max_tokens=1, model=None, fallback=None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(contextvars.copy_context().run, lease.assert_current)
            with pytest.raises(lease.LeaseLost):
                future.result()
        with pytest.raises(lease.LeaseLost):
            llm.gemini_chat_text(prompt="x")


def test_keeper_retries_transient_db_failure_but_cannot_resurrect_expiry(database, monkeypatch):
    receipt = claim(database)
    monkeypatch.setattr(lease, "RENEW_SECONDS", 0.01)
    real_renew, calls = lease.renew, []
    def flaky(claim):
        calls.append(claim)
        if len(calls) == 1:
            raise RuntimeError("transient database failure")
        return real_renew(claim)
    monkeypatch.setattr(lease, "renew", flaky)
    with lease.keep_alive(receipt):
        until = time.monotonic() + 2
        while len(calls) < 3 and time.monotonic() < until:
            time.sleep(0.01)
    assert len(calls) >= 3
    expire(database, receipt)
    assert not real_renew(receipt)


def test_idle_cadence_recovers_expiry_after_startup_and_respects_backoff(database):
    receipt = claim(database)
    assert worker.recover_orphans() == {"requeued": 0, "failed": 0, "expired": 0}
    expire(database, receipt)
    assert worker.process_next_job() is None
    row = capture(database)[0]
    assert row["status"] == "queued" and row["attempts"] == 1
    assert row["not_before"] > datetime.utcnow()


def test_default_recovery_reports_every_legacy_identity_without_touching_it(database, caplog):
    for i in range(7):
        job_id = queued(database)
        with database[0]() as db:
            row = db.get(IndustryReportJob, job_id)
            row.status, row.attempts, row.run_id = "running", 1, f"legacy-{i}"
            db.commit()
    before = capture(database)
    with caplog.at_level("WARNING"):
        result = worker.recover_orphans()
    assert len(result["legacy_deferred"]) == 7
    assert capture(database) == before
    assert all(f"legacy-{i}" in caplog.text for i in range(7))


def legacy_job(database, *, attempts=1, kind="group_report"):
    job_id = queued(database, kind)
    with database[0]() as db:
        row = db.get(IndustryReportJob, job_id)
        row.status, row.attempts = "running", attempts
        row.started_at = row.heartbeat_at = datetime.utcnow()
        db.commit()
        return {key: getattr(row, key) for key in ("id", "run_id", "attempts", "started_at", "heartbeat_at")}


@pytest.mark.parametrize("field", ["run_id", "attempts", "started_at", "heartbeat_at", "owner_token"])
def test_legacy_operator_recovery_rejects_any_changed_attempt(database, field):
    expected = legacy_job(database)
    with database[0]() as db:
        row = db.get(IndustryReportJob, expected["id"])
        setattr(row, field, (expected[field] + timedelta(seconds=1)) if field.endswith("_at")
                else 2 if field == "attempts" else "replacement")
        db.commit()
    before = capture(database)
    result = legacy.recover_legacy_jobs([expected], "Verified predecessor stopped")
    assert result["recovered"] == 0 and result["results"][0]["action"] == "rejected"
    assert capture(database) == before


def test_legacy_recovery_accepts_an_audit_only_receipt(database):
    """A final attempt that stored an audit-only template DID write its
    output; the receipt is as real as a published one and must not be
    rejected as inconsistent because of its status."""
    expected = legacy_job(database)
    saved = report(SimpleNamespace(job_id=expected["id"]), generation={"generation_mode": "deterministic"})
    assert saved.status == "audit_only"
    result = legacy.recover_legacy_jobs([expected], "Verified predecessor stopped")
    assert [r["action"] for r in result["results"]] == ["published_recovered"]


def test_legacy_recovery_requires_unique_exact_publication_and_never_guesses_cross_receipt(database):
    expected = legacy_job(database)
    saved = report(SimpleNamespace(job_id=expected["id"]))
    cross_expected = legacy_job(database, kind="cross_snapshot")
    cross()  # Same period alone cannot prove which legacy job produced it.
    result = legacy.recover_legacy_jobs([expected, cross_expected], "Stopped predecessor; api_key=secret-value")
    assert [r["action"] for r in result["results"]] == ["published_recovered", "rejected"]
    with database[0]() as db:
        first = db.get(IndustryReportJob, expected["id"])
        second = db.get(IndustryReportJob, cross_expected["id"])
        assert first.report_id == saved.id and first.status == "succeeded"
        assert second.snapshot_id is None and second.attempts == 1 and second.status == "running"
        assert result["results"][1]["reason"] == "ambiguous_legacy_snapshot"
        assert first.progress[-1]["expected"]["heartbeat_at"] == expected["heartbeat_at"].isoformat()
        assert "secret-value" not in json.dumps(first.progress)
    assert legacy.recover_legacy_jobs([expected], "same evidence")["recovered"] == 0


@pytest.mark.parametrize("problem", ["duplicate", "wrong_period"])
def test_ambiguous_or_inconsistent_legacy_reports_are_rejected_without_mutation(database, problem):
    expected = legacy_job(database)
    saved = report(SimpleNamespace(job_id=expected["id"]))
    if problem == "duplicate":
        report(SimpleNamespace(job_id=expected["id"]))
    else:
        with database[0]() as db:
            db.get(IndustryReport, saved.id).period_key = "2026-W35"
            db.commit()
    before = capture(database)
    result = legacy.recover_legacy_jobs([expected], "predecessor retired")
    assert result["results"][0]["reason"] == "ambiguous_or_inconsistent_publication"
    assert capture(database) == before


def test_legacy_final_attempt_is_failed_without_executing_or_resetting_attempts(database, monkeypatch):
    expected = legacy_job(database, attempts=3)
    monkeypatch.setattr(worker, "execute_job", lambda *a: pytest.fail("operator recovery executed job"))
    result = legacy.recover_legacy_jobs([expected], "all predecessor instances retired")
    assert result["results"][0]["action"] == "failed"
    assert capture(database)[0]["attempts"] == 3


@pytest.mark.parametrize("bad", [[], [{}], [{"id": True}], "all"])
def test_legacy_request_validation_happens_before_db_mutation(database, bad):
    legacy_job(database)
    before = capture(database)
    with pytest.raises(ValueError):
        legacy.recover_legacy_jobs(bad, "retired")
    assert capture(database) == before


@pytest.mark.parametrize("kind", ["group_report", "cross_snapshot"])
def test_live_receipt_resumption_does_not_compute_or_publish_twice(database, monkeypatch, kind):
    receipt = claim(database, kind)
    with lease.claim_context(receipt):
        report(receipt) if kind == "group_report" else cross()
    monkeypatch.setattr(worker, "_run_group_report", lambda *a: pytest.fail("publication repeated"))
    monkeypatch.setattr(worker, "_run_cross_snapshot", lambda *a: pytest.fail("publication repeated"))
    assert worker.execute_job(receipt)["status"] == "succeeded"
    assert worker.execute_job(receipt)["status"] == "lease_lost"


def test_embedding_dispatch_is_fenced_without_changing_unowned_callers(database, monkeypatch):
    from app.services import embeddings
    receipt = claim(database)
    expire(database, receipt)
    calls = []
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: calls.append(kw)))
    monkeypatch.setattr("openai.OpenAI", lambda **kw: client)
    monkeypatch.setattr(embeddings, "_is_openai_available", lambda: True)
    with lease.claim_context(receipt), pytest.raises(lease.LeaseLost):
        embeddings.embed(["must not dispatch"])
    assert calls == []
    # No claim: the same client entrypoint is still reached, then existing
    # fallback handles this deliberately incomplete mocked response.
    assert len(embeddings.embed(["ordinary caller"])) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("scenario", ["duplicate", "oversized", "bad_time", "no_evidence"])
def test_legacy_bounds_and_all_input_validation_precede_any_disposition(database, scenario):
    expected = legacy_job(database)
    before = capture(database)
    rows, evidence = [expected], "all predecessors retired"
    if scenario == "duplicate":
        rows *= 2
    elif scenario == "oversized":
        rows = [{**expected, "id": i + 1} for i in range(51)]
    elif scenario == "bad_time":
        rows.append({**expected, "id": expected["id"] + 1, "heartbeat_at": "unknown"})
    else:
        evidence = " "
    with pytest.raises(ValueError):
        legacy.recover_legacy_jobs(rows, evidence)
    assert capture(database) == before


def test_full_retirement_evidence_survives_beyond_log_limit_and_masks_tail_secret(database):
    expected = legacy_job(database)
    evidence = "Verified predecessor instance retired. " * 30 + "last-instance-928 stopped; api_key=private-tail-secret"
    result = legacy.recover_legacy_jobs([expected], evidence)
    assert len(result["retirement_evidence"]) > 1000
    assert "last-instance-928 stopped" in result["retirement_evidence"]
    assert "private-tail-secret" not in result["retirement_evidence"]
    with database[0]() as db:
        assert db.get(IndustryReportJob, expected["id"]).progress[-1]["retirement_evidence"] == result["retirement_evidence"]


def test_unexpected_batch_failure_rolls_back_every_legacy_disposition(database, monkeypatch):
    first, second = legacy_job(database), legacy_job(database)
    before = capture(database)
    calls = []
    real_clock = worker._utcnow
    def failed_second_clock():
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("later database step unavailable")
        return real_clock()
    monkeypatch.setattr(worker, "_utcnow", failed_second_clock)
    with pytest.raises(RuntimeError, match="later database"):
        legacy.recover_legacy_jobs([first, second], "verified retired predecessors")
    assert capture(database) == before


def test_malformed_stored_period_is_reported_without_losing_other_results(database):
    first, bad = legacy_job(database), legacy_job(database)
    saved = report(SimpleNamespace(job_id=bad["id"]))
    with database[0]() as db:
        db.get(IndustryReportJob, bad["id"]).period_key = "bad"
        db.get(IndustryReport, saved.id).period_key = "bad"
        db.commit()
    result = legacy.recover_legacy_jobs([first, bad], "predecessors retired")
    assert result["results"] == [{"id": first["id"], "action": "requeued"},
                                 {"id": bad["id"], "action": "rejected", "reason": "invalid_stored_period"}]
    assert capture(database)[1]["status"] == "running"


@pytest.mark.parametrize("change", ["attempt", "backoff"])
def test_delayed_claim_cannot_reset_a_new_attempt_or_bypass_new_backoff(database, change):
    job_id = queued(database)
    observed_attempts = 0
    # Another drainer claimed the selected job and failed it back to queued
    # before this delayed claimant reaches its UPDATE.
    with database[0]() as db:
        job = db.get(IndustryReportJob, job_id)
        if change == "attempt":
            job.attempts = 1
        else:
            job.not_before = datetime.utcnow() + timedelta(minutes=15)
        db.commit()
    before = capture(database)
    with database[0]() as db:
        assert worker._claim(db, job_id, observed_attempts) is None
    assert capture(database) == before


@pytest.mark.parametrize("change", ["attempt", "backoff"])
def test_cross_deferral_cannot_shorten_a_new_retry_backoff(database, monkeypatch, change):
    job_id = queued(database, "cross_snapshot")
    cutoff = datetime.utcnow()
    retry_at = cutoff + timedelta(minutes=30)
    def interleave(db, *args, **kwargs):
        # Run after claim_next_job selected attempts=0. Another claimant has
        # completed its failure transaction before this stale deferral UPDATE.
        row = db.get(IndustryReportJob, job_id)
        row.not_before = retry_at
        if change == "attempt":
            row.attempts = 1
        db.commit()
        return 1
    monkeypatch.setattr(worker, "_group_jobs_in_flight", interleave)
    assert worker.claim_next_job(now=cutoff) is None
    row = capture(database)[0]
    assert row["not_before"] == retry_at
    assert row["attempts"] == (1 if change == "attempt" else 0)


def test_legacy_cross_without_any_output_preserves_retry_policy(database):
    expected = legacy_job(database, kind="cross_snapshot")
    result = legacy.recover_legacy_jobs([expected], "predecessors retired")
    assert result["results"] == [{"id": expected["id"], "action": "requeued"}]
    row = capture(database)[0]
    assert row["attempts"] == 1 and row["snapshot_id"] is None
    assert row["not_before"] > datetime.utcnow()
