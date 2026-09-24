"""Owner decision 1 — the one-shot reclassification of legacy edition flags.

Reads never depend on it (every reader applies the display rule, see
`test_industry_report_store`); it normalises the ROW state the pre-rule
code left behind. What can go wrong with a production data write, and the
test that pins each:

* a dry run that writes — `test_dry_run_writes_nothing`;
* the wrong rows moved, or a pending_review row touched —
  `test_apply_fixes_the_flags_and_records_the_manifest`;
* a second run that changes something again — `test_apply_is_idempotent`;
* interleaving with a report job's own flip — `test_refuses_with_a_queued_or_running_job`;
* a partial write — `test_a_compare_and_set_miss_rolls_back_everything`;
* no way back — `test_reclassify_revert_restores_manifest` (and the
  revert refusing a row that has moved on since);
* needing a production shell at all — the deployed worker runs it once,
  recorded by the ledger, deferring while jobs are active:
  `test_the_worker_runs_it_once_and_only_once`.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.database import SessionLocal
from app.models import IndustryReport, IndustryReportJob
from app.scripts import reclassify_industry_editions as cli
from app.services import gics_registry as reg
from app.services import industry_report_store as rs
from app.services import industry_report_worker as jobs

AS_OF = datetime(2026, 9, 4, 21, 0)
LLM = {"generation_mode": "llm"}
TEMPLATE = {"generation_mode": "deterministic"}


@pytest.fixture(scope="module")
def info():
    version = reg.ensure_taxonomy(activate=True)
    assert version is not None
    yield version
    reg.activate_version(version.version_key)


def _wipe(version_id: int) -> None:
    # The reclassification spans EVERY row of the taxonomy version, so the
    # test owns all of them — a leftover from another module would appear
    # in the manifest.
    with SessionLocal() as db:
        db.execute(delete(IndustryReportJob).where(IndustryReportJob.taxonomy_version_id == version_id))
        db.execute(delete(IndustryReport).where(IndustryReport.taxonomy_version_id == version_id))
        db.commit()


@pytest.fixture(autouse=True)
def clean(info):
    _wipe(info.id)
    jobs._reclassify_settled = False
    yield
    _wipe(info.id)
    jobs._reclassify_settled = False


def _row(info, code: str, version: int, *, status: str, latest: bool, generation: dict,
         degraded: list[str] | None = None) -> int:
    with SessionLocal() as db:
        row = IndustryReport(
            taxonomy_version_id=info.id, industry_group_code=code, version=version,
            period_key=f"2026-W{30 + version}", as_of=AS_OF, status=status, is_latest_good=latest,
            payload={"sections": {}}, generation=generation, degraded=list(degraded or []), generated_at=AS_OF,
        )
        db.add(row)
        db.commit()
        return row.id


def _legacy(info) -> dict[str, int]:
    """Two groups as the pre-rule code left them.

    Group A: analyst v1 superseded by template v2, which holds the flag.
    Group B: analyst v1 and v2 (v2 latest), a legacy template v3 already
    superseded, and a pending_review analyst v4 the reclassification must
    leave alone."""
    a, b = [g.code for g in reg.industry_groups(version=info)][:2]
    return {
        "a1": _row(info, a, 1, status="superseded", latest=False, generation=LLM),
        "a2": _row(info, a, 2, status="succeeded", latest=True, generation=TEMPLATE),
        "b1": _row(info, b, 1, status="superseded", latest=False, generation=LLM),
        "b2": _row(info, b, 2, status="succeeded", latest=True, generation=LLM),
        "b3": _row(info, b, 3, status="superseded", latest=False, generation={"generation_mode": "llm_unavailable"}),
        "b4": _row(info, b, 4, status="pending_review", latest=False, generation=LLM),
    }


def _state(info) -> dict[int, tuple[str, bool]]:
    with SessionLocal() as db:
        return {rid: (s, bool(flag)) for rid, s, flag in db.execute(
            select(IndustryReport.id, IndustryReport.status, IndustryReport.is_latest_good)
            .where(IndustryReport.taxonomy_version_id == info.id))}


def _ledgers(info) -> list[IndustryReportJob]:
    with SessionLocal() as db:
        return list(db.execute(select(IndustryReportJob).where(
            IndustryReportJob.taxonomy_version_id == info.id,
            IndustryReportJob.kind == rs.RECLASSIFY_KIND)).scalars())


def test_dry_run_writes_nothing(info, capsys):
    ids = _legacy(info)
    before = _state(info)
    assert cli.main([]) == cli.EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "dry_run" and out["applied"] is False
    assert _state(info) == before and _ledgers(info) == []
    planned = {e["row_id"]: (e["before"], e["after"]) for e in out["manifest"]}
    assert set(planned) == {ids["a1"], ids["a2"], ids["b3"]}
    assert out["counts"] == {"changed": 3, "to_audit_only": 2, "to_succeeded": 1}


def test_apply_fixes_the_flags_and_records_the_manifest(info, capsys):
    ids = _legacy(info)
    assert cli.main(["--apply"]) == cli.EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is True and out["counts"]["changed"] == 3
    state = _state(info)
    assert state[ids["a1"]] == ("succeeded", True)
    assert state[ids["a2"]] == ("audit_only", False)
    assert state[ids["b2"]] == ("succeeded", True)
    assert state[ids["b1"]] == ("superseded", False)
    assert state[ids["b3"]] == ("audit_only", False)
    assert state[ids["b4"]] == ("pending_review", False), "pending_review is a human's queue — untouched"
    # Nothing deleted; one ledger row carrying exactly the printed manifest.
    assert len(state) == len(ids)
    [ledger] = _ledgers(info)
    assert ledger.status == "succeeded" and ledger.run_id == rs.RECLASSIFY_RUN_ID
    assert ledger.progress[-1]["manifest"] == out["manifest"]
    # The public reads were already right and stay right.
    assert rs.latest_good(reg.industry_groups(version=info)[0].code, version=info)["id"] == ids["a1"]


def test_apply_is_idempotent(info, capsys):
    _legacy(info)
    assert cli.main(["--apply"]) == cli.EXIT_OK
    capsys.readouterr()
    before = _state(info)
    assert cli.main(["--apply"]) == cli.EXIT_OK
    again = json.loads(capsys.readouterr().out)
    assert again["counts"] == {"changed": 0} and again["manifest"] == []
    assert _state(info) == before


@pytest.mark.parametrize("status", ["queued", "running"])
def test_refuses_with_a_queued_or_running_job(info, capsys, status):
    """`save_report`'s flip and this script must not interleave."""
    _legacy(info)
    before = _state(info)
    code = reg.industry_groups(version=info)[0].code
    with SessionLocal() as db:
        db.add(IndustryReportJob(kind="group_report", taxonomy_version_id=info.id, industry_group_code=code,
                                 period_key="2026-W39", run_id="r", status=status, attempts=0, max_attempts=3,
                                 enqueued_at=AS_OF))
        db.commit()
    assert cli.main(["--apply"]) == cli.EXIT_REFUSED
    assert "queued or running" in json.loads(capsys.readouterr().err)["refused"]
    assert _state(info) == before and _ledgers(info) == []


def test_a_compare_and_set_miss_rolls_back_everything(info, monkeypatch, capsys):
    """If any row is not in its planned before-state when the write lands,
    NOTHING is written — not the rows that did match, not the ledger."""
    ids = _legacy(info)
    before = _state(info)
    real_plan = rs._reclassification_plan
    calls = {"n": 0}

    def stale_plan(db, version_id, *, lock):
        plan = real_plan(db, version_id, lock=lock)
        calls["n"] += 1
        if calls["n"] == 1:  # the plan the writes use: one entry's before-state is wrong
            plan[-1] = {**plan[-1], "before": {"status": "pending_review", "is_latest_good": False}}
        return plan

    monkeypatch.setattr(rs, "_reclassification_plan", stale_plan)
    assert cli.main(["--apply"]) == cli.EXIT_ABORTED
    err = json.loads(capsys.readouterr().err)
    assert err["written"] is False and "compare-and-set" in err["aborted"]
    assert _state(info) == before and _ledgers(info) == []
    assert ids  # the rows are all still there


def test_reclassify_revert_restores_manifest(info, tmp_path, capsys):
    _legacy(info)
    before = _state(info)
    assert cli.main(["--apply"]) == cli.EXIT_OK
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(capsys.readouterr().out)
    assert _state(info) != before

    # Dry run of the revert writes nothing.
    assert cli.main(["--revert", str(manifest_file)]) == cli.EXIT_OK
    capsys.readouterr()
    assert _state(info) != before
    assert cli.main(["--revert", str(manifest_file), "--apply"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["applied"] is True
    assert _state(info) == before
    assert {row.run_id for row in _ledgers(info)} == {rs.RECLASSIFY_RUN_ID, rs.RECLASSIFY_REVERT_RUN_ID}


def test_revert_from_ledger_and_refusal_when_a_row_moved_on(info, capsys):
    ids = _legacy(info)
    before = _state(info)
    assert cli.main(["--apply"]) == cli.EXIT_OK
    capsys.readouterr()
    # A newer save moved one reclassified row on: the revert must not
    # overwrite it, and must write nothing at all.
    with SessionLocal() as db:
        db.get(IndustryReport, ids["a1"]).status = "superseded"
        db.commit()
    moved = _state(info)
    assert cli.main(["--revert-from-ledger", "--apply"]) == cli.EXIT_ABORTED
    capsys.readouterr()
    assert _state(info) == moved
    # Put it back and the ledger revert restores the original state exactly.
    with SessionLocal() as db:
        db.get(IndustryReport, ids["a1"]).status = "succeeded"
        db.commit()
    assert cli.main(["--revert-from-ledger", "--apply"]) == cli.EXIT_OK
    assert _state(info) == before


def test_the_worker_runs_it_once_and_only_once(info, monkeypatch):
    """No production shell needed: the deployed drainer applies it on a
    heartbeat, defers while a report job is active, records a ledger row,
    and never writes again — not after a restart, not after the owner's CLI."""
    ids = _legacy(info)
    code = reg.industry_groups(version=info)[0].code
    with SessionLocal() as db:
        job = IndustryReportJob(kind="group_report", taxonomy_version_id=info.id, industry_group_code=code,
                                period_key="2026-W39", run_id="r", status="running", attempts=1, max_attempts=3,
                                enqueued_at=AS_OF)
        db.add(job)
        db.commit()
        job_id = job.id
    deferred = jobs.reclassify_legacy_editions_once()
    assert deferred["status"] == "deferred" and _ledgers(info) == []

    with SessionLocal() as db:
        db.get(IndustryReportJob, job_id).status = "succeeded"
        db.commit()
    applied = jobs.reclassify_legacy_editions_once()
    assert applied["status"] == "applied" and applied["counts"]["changed"] == 3
    assert _state(info)[ids["a2"]] == ("audit_only", False)
    assert jobs.reclassify_legacy_editions_once() is None, "settled in this process"
    jobs._reclassify_settled = False  # a restart: the ledger is the authority
    assert jobs.reclassify_legacy_editions_once()["status"] == "already_done"
    assert len(_ledgers(info)) == 1
    [ledger] = _ledgers(info)
    assert ledger.source == "worker_drainer"


def test_the_worker_run_can_be_switched_off(info, monkeypatch):
    _legacy(info)
    before = _state(info)
    monkeypatch.setattr(settings, "industry_reclassify_legacy_editions", False)
    assert jobs.reclassify_legacy_editions_once() is None
    assert _state(info) == before and _ledgers(info) == []


def test_the_ledger_row_is_invisible_to_the_queue_and_the_report_reads(info):
    """The ledger is an `industry_report_jobs` row written already finished:
    no drainer can claim it and no report read counts it as an attempt."""
    _legacy(info)
    jobs.reclassify_legacy_editions_once()
    assert jobs.claim_next_job() is None
    code = reg.industry_groups(version=info)[0].code
    assert rs.last_attempt(code, version=info) is None
    assert rs.last_attempted_periods([code], version=info) == {}
