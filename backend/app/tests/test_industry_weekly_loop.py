"""FEAT-003 slice 4 — the Sunday Industry Analysis loop and its health row.

What can rot here: a loop that enqueues before warming the prices (so the
whole warm-up budget is dead code and week one is a page of `no_prices`),
a note that says `enqueued=0` without saying why, a tick that raises
before it reports, and a loop cron-health calls stale six days out of
seven because nobody added it to the weekly set.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import delete

from app.api import routes_admin
from app.config import settings
from app.database import SessionLocal
from app.models import CronLoopRun, IndustryReportJob
from app.monitoring import KNOWN_LOOPS
from app.monitoring import industry_weekly_loop as loop
from app.services import gics_registry as reg
from app.services import industry_analytics as ia
from app.services import industry_classification as ic
from app.services import industry_report_store as rs
from app.services import industry_report_worker as jobs
from app.tests.gating_helpers import seed_demo_universe

SUNDAY = datetime(2026, 9, 6, 6, 30)
AS_OF = datetime(2026, 9, 4, 21, 0)
PERIOD = "2026-W36"


@pytest.fixture(scope="module")
def info():
    seed_demo_universe()
    version = reg.ensure_taxonomy(activate=True)
    assert version is not None
    ic.classify_all(version=version)
    yield version
    reg.activate_version(version.version_key)


@pytest.fixture()
def runs(monkeypatch):
    """Capture what the loop reports instead of writing the shared row."""
    recorded: list[tuple[str, bool, str]] = []
    monkeypatch.setattr(
        loop, "record_run",
        lambda name, *, success=True, note="": recorded.append((name, success, note)),
    )
    return recorded


@pytest.fixture(autouse=True)
def _wipe(info):
    def clean() -> None:
        with SessionLocal() as db:
            db.execute(delete(IndustryReportJob).where(
                IndustryReportJob.taxonomy_version_id == info.id))
            db.commit()

    clean()
    yield
    clean()


@pytest.fixture()
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_industry_reports", True)


def _warm_stub(order: list[str], **extra):
    def warm(*, budget=None, version=None, loaders=None):
        order.append("warm_up")
        return {"budget": budget, "fetched": 3, "failed": ["ZZZ"], "remaining_missing": 7,
                "groups_touched": ["0000"], **extra}

    return warm


# --- the tick -----------------------------------------------------------------


def test_a_disabled_deployment_records_a_live_but_disabled_loop(runs, monkeypatch):
    """The web service runs this scheduler too. It must show as a loop that
    ran and chose not to act, never as one that has never reported."""
    monkeypatch.setattr(settings, "enable_industry_reports", False)
    result = loop.run_once()
    assert result == {"enabled": False, "enqueued": 0,
                      "note": "disabled (ENABLE_INDUSTRY_REPORTS=false)"}
    assert runs == [(loop.LOOP_NAME, True, "disabled (ENABLE_INDUSTRY_REPORTS=false)")]
    with SessionLocal() as db:
        assert db.query(IndustryReportJob).count() == 0


def test_prices_are_warmed_before_anything_is_enqueued(info, runs, enabled, monkeypatch):
    """Order is the whole point: warming after the enqueue would let the
    drainer compute the week off whatever happened to be cached, and the
    warm-up budget would never change a published number."""
    order: list[str] = []
    monkeypatch.setattr(ia, "warm_up_prices", _warm_stub(order))
    real_enqueue = jobs.enqueue_period

    def spy(*args, **kw):
        order.append("enqueue_period")
        return real_enqueue(*args, **kw)

    monkeypatch.setattr(jobs, "enqueue_period", spy)
    summary = loop.run_once(now=SUNDAY)

    assert order == ["warm_up", "enqueue_period"]
    assert summary["period_key"] == PERIOD and summary["as_of"] == AS_OF.isoformat()
    assert summary["groups"] == len(reg.industry_groups(version=info))
    assert summary["enqueued"] == summary["groups"]
    assert summary["cross_snapshot_job"] is not None
    assert summary["warm_up"]["budget"] == settings.industry_price_warmup_budget

    name, success, note = runs[-1]
    assert (name, success) == (loop.LOOP_NAME, True)
    # Counts, not a bare "ok": `success=True` with no numbers is what hid
    # an outage here before.
    for token in (f"period={PERIOD}", f"groups={summary['groups']}", "enqueued=", "coalesced=0",
                  "skipped_published=0", "warm_fetched=3", "warm_failed=1",
                  "warm_remaining_missing=7", "with_constituents="):
        assert token in note, note
    with SessionLocal() as db:
        assert db.query(IndustryReportJob).count() == summary["groups"] + 1


def test_a_second_tick_coalesces_instead_of_doubling_the_week(info, runs, enabled, monkeypatch):
    monkeypatch.setattr(ia, "warm_up_prices", _warm_stub([]))
    first = loop.run_once(now=SUNDAY)
    second = loop.run_once(now=SUNDAY + timedelta(minutes=5))
    assert second["enqueued"] == 0 and second["coalesced"] == first["enqueued"]
    assert runs[-1][1] is True, "a fully coalesced week is healthy, not a failure"
    with SessionLocal() as db:
        assert db.query(IndustryReportJob).count() == first["enqueued"] + 1


def test_a_published_week_is_skipped_unless_forced(info, runs, enabled, monkeypatch):
    monkeypatch.setattr(ia, "warm_up_prices", _warm_stub([]))
    code = reg.industry_groups(version=info)[0].code
    rs.save_report(code=code, period_key=PERIOD, as_of=AS_OF, payload={"sections": {}}, version=info)
    try:
        summary = loop.run_once(now=SUNDAY, codes=[code])
        assert summary["skipped_published"] == 1 and summary["enqueued"] == 0
        assert "skipped_published=1" in runs[-1][2]
        # No generation_mode: a template, stored audit-only. The note says
        # the skip is a withheld week, not a published one.
        assert summary["skipped_withheld"] == 1 and "skipped_withheld=1" in runs[-1][2]
        forced = loop.run_once(now=SUNDAY, codes=[code], force=True)
        assert forced["enqueued"] == 1
    finally:
        from app.models import IndustryReport
        with SessionLocal() as db:
            db.execute(delete(IndustryReport).where(
                IndustryReport.taxonomy_version_id == info.id,
                IndustryReport.industry_group_code == code))
            db.commit()


def test_note_carries_previous_period_outcome(info, runs, enabled, monkeypatch):
    """This tick's `record_run` clears the progress row the drainer wrote
    all last week, so last week's verdict has to ride in THIS note — every
    count, not just a boolean — while `success` stays about the enqueue."""
    from app.models import IndustryReport

    monkeypatch.setattr(ia, "warm_up_prices", _warm_stub([]))
    prev = "2026-W35"
    groups = [g.code for g in reg.industry_groups(version=info)][:3]
    try:
        agentic = rs.save_report(code=groups[0], period_key=prev, as_of=AS_OF - timedelta(days=7), payload={},
                                 version=info, generation={"generation_mode": "llm"})
        template = rs.save_report(code=groups[1], period_key=prev, as_of=AS_OF - timedelta(days=7), payload={},
                                  version=info, generation={"generation_mode": "deterministic"})
        with SessionLocal() as db:
            for code, status, rid in ((groups[0], "succeeded", agentic.id), (groups[1], "succeeded", template.id),
                                      (groups[2], "failed", None)):
                db.add(IndustryReportJob(kind="group_report", taxonomy_version_id=info.id,
                                         industry_group_code=code, period_key=prev, run_id="r", status=status,
                                         attempts=3, max_attempts=3, enqueued_at=SUNDAY, report_id=rid))
            db.commit()
        summary = loop.run_once(now=SUNDAY, codes=[])
        name, success, note = runs[-1]
        assert success is True, "success is about this week's enqueue, unchanged"
        for token in (f"prev_period={prev}", "prev_agentic=1", "prev_template=1", "prev_failed=1",
                      "prev_pending=0", "prev_not_updated_rate=0.6667", "prev_healthy=False",
                      f"prev_not_updated_codes={','.join(sorted(groups[1:3]))}"):
            assert token in note, note
        assert summary["previous_period"]["period_key"] == prev
    finally:
        with SessionLocal() as db:
            db.execute(delete(IndustryReport).where(IndustryReport.taxonomy_version_id == info.id,
                                                    IndustryReport.industry_group_code.in_(groups)))
            db.commit()


def test_a_first_run_with_nothing_classified_bootstraps_and_says_so(info, runs, enabled, monkeypatch):
    """The daily classification loop owns membership. On a database that
    has never run it, the weekly loop bootstraps once rather than
    publishing a week of empty groups — and reports that it did."""
    monkeypatch.setattr(ia, "warm_up_prices", _warm_stub([]))
    calls: list[str] = []
    seen: list[dict] = [{}, ic.constituents_by_group(version=info)]

    monkeypatch.setattr(ic, "constituents_by_group",
                        lambda **kw: seen.pop(0))
    monkeypatch.setattr(ic, "classify_all",
                        lambda **kw: calls.append("classify_all") or {})
    summary = loop.run_once(now=SUNDAY, codes=[])
    assert calls == ["classify_all"] and summary["bootstrapped_classification"] is True
    assert "bootstrapped_classification=1" in runs[-1][2]
    assert summary["groups_with_constituents"] > 0


def test_a_tick_that_raises_still_reports_its_failure(info, runs, enabled, monkeypatch):
    monkeypatch.setattr(ia, "warm_up_prices", _warm_stub([]))

    def boom(*a, **kw):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(jobs, "enqueue_period", boom)
    with pytest.raises(RuntimeError):
        loop.run_once(now=SUNDAY)
    name, success, note = runs[-1]
    assert (name, success) == (loop.LOOP_NAME, False)
    assert "error=RuntimeError" in note


# --- schedule + health --------------------------------------------------------


def test_the_loop_is_registered_where_the_owner_decision_says(monkeypatch):
    registered: list[dict] = []

    class FakeScheduler:
        def add_job(self, fn, trigger, **kw):
            registered.append({"fn": fn, "trigger": trigger, **kw})

    loop.register(FakeScheduler())
    assert loop.LOOP_NAME in KNOWN_LOOPS
    job = registered[0]
    assert job["trigger"] == "cron" and job["id"] == loop.LOOP_NAME
    assert (job["day_of_week"], job["hour"], job["minute"]) == (
        settings.industry_reports_cron_dow, settings.industry_reports_cron_hour,
        settings.industry_reports_cron_minute,
    )
    assert (job["day_of_week"], job["hour"], job["minute"]) == ("sun", 6, 30)


def test_cron_health_gives_the_weekly_loop_a_week_and_the_drainer_an_hour():
    """A Sunday loop judged on the 26h daily window is "stale" from Monday
    lunchtime onwards, which buries the daily loops that are genuinely
    late. The drainer is the opposite case: it reports every five minutes,
    so a day-old row means it is dead."""
    from app.monitoring import _LAST_RUNS

    now = datetime.utcnow()
    rows = {
        loop.LOOP_NAME: now - timedelta(days=3),
        jobs.HEARTBEAT_NAME: now - timedelta(minutes=90),
        "sector_digest_loop": now - timedelta(days=3),
    }
    # `status_snapshot` layers this process's in-memory records over the DB
    # rows, so an earlier test in the same session that heartbeated for
    # real would mask the rows written here.
    saved = {name: _LAST_RUNS.pop(name, None) for name in rows}
    with SessionLocal() as db:
        for name, when in rows.items():
            row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == name).one_or_none()
            if row is None:
                row = CronLoopRun(loop_name=name)
                db.add(row)
            row.last_run_at, row.success, row.note = when, True, "test"
        db.commit()
    try:
        payload = routes_admin.cron_health_endpoint()
        by_name = {r["loop"]: r for r in payload["loops"]}
        assert by_name[loop.LOOP_NAME]["stale"] is False, "a 3-day-old weekly loop is fine"
        assert by_name["sector_digest_loop"]["stale"] is False
        assert by_name[jobs.HEARTBEAT_NAME]["stale"] is True, (
            "a 90-minute-old drainer heartbeat means the thread is gone"
        )
    finally:
        for name, record in saved.items():
            if record is not None:
                _LAST_RUNS[name] = record
        with SessionLocal() as db:
            db.execute(delete(CronLoopRun).where(CronLoopRun.loop_name.in_(list(rows))))
            db.commit()
