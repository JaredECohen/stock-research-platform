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
* a way back that silently does nothing — a later empty apply hiding the
  real manifest: `test_worker_apply_then_cli_apply_then_revert_from_ledger_restores`;
* rows nobody meant to touch — the pinned expected population
  (`test_*_aborts_the_whole_apply`);
* needing a production shell at all — the deployed worker runs it once,
  recorded by the ledger, deferring while jobs are active:
  `test_the_worker_runs_it_once_and_only_once`;
* housekeeping stopping the queue — `test_an_unexpected_error_never_stops_the_heartbeat_or_the_queue`.
"""
from __future__ import annotations

import json
import threading
import time
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
         degraded: list[str] | None = None, period_key: str | None = None) -> int:
    # Versions 1..4 land in W37..W40: the pinned legacy window.
    with SessionLocal() as db:
        row = IndustryReport(
            taxonomy_version_id=info.id, industry_group_code=code, version=version,
            period_key=period_key or f"2026-W{36 + version}", as_of=AS_OF, status=status, is_latest_good=latest,
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
    assert again["status"] == "already_done" and again["applied"] is False
    assert _state(info) == before
    # A second apply leaves no second ledger row: one would carry an empty
    # manifest and hide the real one from --revert-from-ledger.
    assert len(_ledgers(info)) == 1


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
    # That stale entry is also not a legacy transition, which the expected-
    # set check would catch first; switch that off to reach the CAS itself.
    monkeypatch.setattr(rs, "_plan_violations", lambda plan, **_: [])
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


# --- the ledger is the one way back -------------------------------------------


def test_worker_apply_then_cli_apply_then_revert_from_ledger_restores(info, capsys):
    """The production order: the worker applies on its first heartbeat
    after deploy, then the owner runs the documented `--apply`. That
    second apply must not write an (empty) ledger row, or
    `--revert-from-ledger` reverts nothing and still exits 0."""
    _legacy(info)
    before = _state(info)
    assert jobs.reclassify_legacy_editions_once()["status"] == "applied"
    applied = _state(info)

    assert cli.main(["--apply"]) == cli.EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "already_done" and out["applied"] is False
    assert _state(info) == applied
    assert [(r.run_id, len(r.progress[-1]["manifest"])) for r in _ledgers(info)] == [(rs.RECLASSIFY_RUN_ID, 3)]

    assert cli.main(["--revert-from-ledger", "--apply"]) == cli.EXIT_OK
    reverted = json.loads(capsys.readouterr().out)
    assert reverted["counts"] == {"reverted": 3} and reverted["applied"] is True
    assert _state(info) == before

    # Reverting the same apply twice is refused, not a silent no-op.
    assert cli.main(["--revert-from-ledger", "--apply"]) == cli.EXIT_REFUSED
    assert "already reverted" in json.loads(capsys.readouterr().err)["error"]
    assert _state(info) == before


def test_an_empty_ledger_row_from_an_earlier_build_never_shadows_the_manifest(info, capsys):
    """Belt and braces for rows already written: the revert target is the
    newest apply that CHANGED rows, not merely the newest row."""
    _legacy(info)
    before = _state(info)
    assert cli.main(["--apply"]) == cli.EXIT_OK
    capsys.readouterr()
    with SessionLocal() as db:
        rs._write_ledger(db, info.id, run_id=rs.RECLASSIFY_RUN_ID, source="old_build", step="reclassified",
                         manifest=[], counts={"changed": 0})
        db.commit()
    assert rs.reclassification_ledger(version=info)["counts"]["changed"] == 3
    assert cli.main(["--revert-from-ledger", "--apply"]) == cli.EXIT_OK
    assert _state(info) == before


def test_a_revert_refuses_an_empty_manifest(info, tmp_path, capsys):
    _legacy(info)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"manifest": []}))
    assert cli.main(["--revert", str(empty), "--apply"]) == cli.EXIT_REFUSED
    assert "nothing to revert" in json.loads(capsys.readouterr().err)["refused"]
    assert _ledgers(info) == []


def test_the_worker_never_reapplies_after_the_owner_reverted(info, capsys):
    """A revert is the owner saying "not this": a worker restart must not
    undo it. The owner may re-apply by hand."""
    _legacy(info)
    before = _state(info)
    assert jobs.reclassify_legacy_editions_once()["status"] == "applied"
    assert cli.main(["--revert-from-ledger", "--apply"]) == cli.EXIT_OK
    capsys.readouterr()
    jobs._reclassify_settled = False  # a restart
    assert jobs.reclassify_legacy_editions_once()["status"] == "already_done"
    assert _state(info) == before
    assert cli.main(["--apply"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == "applied"
    assert _state(info) != before


def test_the_revert_refuses_with_a_queued_job(info, capsys):
    _legacy(info)
    assert cli.main(["--apply"]) == cli.EXIT_OK
    capsys.readouterr()
    applied = _state(info)
    code = reg.industry_groups(version=info)[0].code
    with SessionLocal() as db:
        db.add(IndustryReportJob(kind="group_report", taxonomy_version_id=info.id, industry_group_code=code,
                                 period_key="2026-W41", run_id="r", status="queued", attempts=0, max_attempts=3,
                                 enqueued_at=AS_OF))
        db.commit()
    assert cli.main(["--revert-from-ledger", "--apply"]) == cli.EXIT_REFUSED
    assert "queued or running" in json.loads(capsys.readouterr().err)["refused"]
    assert _state(info) == applied
    assert {r.run_id for r in _ledgers(info)} == {rs.RECLASSIFY_RUN_ID}


# --- the pinned expected population -------------------------------------------


def test_a_row_outside_the_legacy_window_aborts_the_whole_apply(info, capsys):
    """A row the pre-rule code could not have written is not legacy state:
    the automatic run writes NOTHING (not even the in-window rows), leaves
    no ledger, and is not retried; the owner reads the dry run and widens
    the window by hand."""
    ids = _legacy(info)
    code = reg.industry_groups(version=info)[2].code
    late = _row(info, code, 1, status="succeeded", latest=True, generation=TEMPLATE, period_key="2026-W41")
    before = _state(info)

    aborted = jobs.reclassify_legacy_editions_once()
    assert aborted["status"] == "aborted" and "legacy window" in aborted["reason"]
    assert _state(info) == before and _ledgers(info) == []
    assert jobs.reclassify_legacy_editions_once() is None, "an abort is not retried by this process"

    assert cli.main([]) == cli.EXIT_ABORTED
    dry = json.loads(capsys.readouterr().out)
    assert any(f"row {late}" in v for v in dry["violations"])
    assert cli.main(["--apply"]) == cli.EXIT_ABORTED
    capsys.readouterr()
    assert _state(info) == before and _ledgers(info) == []

    assert cli.main(["--apply", "--last-legacy-period", "2026-W41"]) == cli.EXIT_OK
    capsys.readouterr()
    assert _state(info)[late] == ("audit_only", False)
    assert _state(info)[ids["a2"]] == ("audit_only", False)


def test_a_non_legacy_transition_aborts_the_whole_apply(info):
    """An audit-only row holding the flag is not something the pre-rule
    code wrote; the rule would "fix" it, the expected set refuses to."""
    ids = _legacy(info)
    code = reg.industry_groups(version=info)[2].code
    _row(info, code, 1, status="audit_only", latest=True, generation=TEMPLATE)
    before = _state(info)
    with pytest.raises(rs.ReclassifyAborted, match="not a legacy transition"):
        rs.reclassify_legacy_editions(apply=True, version=info)
    assert _state(info) == before and _ledgers(info) == []
    assert ids


def test_more_rows_than_the_pinned_ceiling_aborts_the_whole_apply(info, monkeypatch):
    _legacy(info)
    before = _state(info)
    monkeypatch.setattr(rs, "RECLASSIFY_MAX_CHANGES", 2)
    with pytest.raises(rs.ReclassifyAborted, match="pinned ceiling"):
        rs.reclassify_legacy_editions(apply=True, version=info)
    assert _state(info) == before and _ledgers(info) == []


# --- every guard aborts the whole transaction ---------------------------------


def test_rows_still_disagreeing_after_the_writes_roll_everything_back(info, monkeypatch, capsys):
    _legacy(info)
    before = _state(info)
    real_plan = rs._reclassification_plan
    calls = {"n": 0}

    def plan_that_survives_the_writes(db, version_id, *, lock):
        calls["n"] += 1
        plan = real_plan(db, version_id, lock=lock)
        if calls["n"] == 2:  # the post-write recomputation
            return [{"row_id": -1}]
        return plan

    monkeypatch.setattr(rs, "_reclassification_plan", plan_that_survives_the_writes)
    assert cli.main(["--apply"]) == cli.EXIT_ABORTED
    err = json.loads(capsys.readouterr().err)
    assert err["written"] is False and "still disagree" in err["aborted"]
    assert _state(info) == before and _ledgers(info) == []


def test_the_apply_compare_and_set_checks_the_flag_too(info, monkeypatch, capsys):
    """A row whose status matches the plan but whose flag does not has
    moved: that is a miss, and nothing is written."""
    ids = _legacy(info)
    before = _state(info)
    real_plan = rs._reclassification_plan
    calls = {"n": 0}

    def flag_drifted(db, version_id, *, lock):
        calls["n"] += 1
        plan = real_plan(db, version_id, lock=lock)
        if calls["n"] == 1:
            plan = [{**e, "before": {**e["before"], "is_latest_good": not e["before"]["is_latest_good"]}}
                    if e["row_id"] == ids["a2"] else e for e in plan]
        return plan

    monkeypatch.setattr(rs, "_reclassification_plan", flag_drifted)
    # The drifted entry is not a legacy transition either; isolate the CAS.
    monkeypatch.setattr(rs, "_plan_violations", lambda plan, **_: [])
    assert cli.main(["--apply"]) == cli.EXIT_ABORTED
    assert "compare-and-set" in json.loads(capsys.readouterr().err)["aborted"]
    assert _state(info) == before and _ledgers(info) == []


def test_the_revert_compare_and_set_checks_the_flag_too(info, capsys):
    ids = _legacy(info)
    assert cli.main(["--apply"]) == cli.EXIT_OK
    capsys.readouterr()
    with SessionLocal() as db:
        db.get(IndustryReport, ids["a2"]).is_latest_good = True  # status still audit_only
        db.commit()
    moved = _state(info)
    assert cli.main(["--revert-from-ledger", "--apply"]) == cli.EXIT_ABORTED
    capsys.readouterr()
    assert _state(info) == moved
    assert {r.run_id for r in _ledgers(info)} == {rs.RECLASSIFY_RUN_ID}


# --- the worker hook never stops the drainer ----------------------------------


def test_an_unexpected_error_never_stops_the_heartbeat_or_the_queue(info, monkeypatch):
    """A housekeeping step on the heartbeat must not be what silences the
    drainer: an exception there used to skip the heartbeat and every job,
    on every pass, for as long as it lasted."""
    calls = {"reclassify": 0, "heartbeat": 0, "process": 0}

    def boom(**_):
        calls["reclassify"] += 1
        raise RuntimeError("could not serialize access")

    def beat(now=None):
        calls["heartbeat"] += 1
        return "ok"

    def process(**_):
        calls["process"] += 1
        return None

    monkeypatch.setattr(rs, "run_legacy_reclassification_once", boom)
    # The direct call reports it and stays unsettled (retried next beat).
    assert jobs.reclassify_legacy_editions_once()["status"] == "error"
    assert jobs._reclassify_settled is False

    monkeypatch.setattr(jobs, "recover_orphans", lambda *a, **k: {})
    monkeypatch.setattr(jobs, "heartbeat", beat)
    monkeypatch.setattr(jobs, "process_next_job", process)
    monkeypatch.setattr(jobs, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(jobs, "HEARTBEAT_SECONDS", 0)
    jobs._stop_event.clear()
    thread = threading.Thread(target=jobs._worker_loop, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and min(calls.values()) < 3:
            time.sleep(0.01)
    finally:
        jobs._stop_event.set()
        thread.join(timeout=5)
        jobs._stop_event.clear()
    assert calls["reclassify"] >= 3 and calls["heartbeat"] >= 3 and calls["process"] >= 3, calls
