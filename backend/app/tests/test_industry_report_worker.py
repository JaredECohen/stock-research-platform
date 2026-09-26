"""FEAT-003 slice 4 — the durable Industry Analysis report queue.

What can rot here: a double-enqueued week, a claim two drainers both win,
a retry that hammers instead of backing off, a failed refresh that blanks
the page by dropping the previous edition, a final attempt that still
needs an LLM, a restart that leaves jobs `running` forever, a two-week-old
job published as "this week", a drainer that is invisible to cron-health
between Sundays, and — the one that is a correctness bug rather than a
performance one — a period drained with one analytics context per group,
which makes the groups of one week incomparable. Each has a test.

The pipeline runs for real (registry → classification → stats → writer →
validator → store); only the *reads* are stubbed, through
``industry_analytics.default_loaders``, so no provider or LLM is touched
and the numbers are fixed by hand-built price series.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.database import SessionLocal
from app.models import CrossIndustrySnapshot, IndustryReport, IndustryReportJob, IndustryStatSnapshot
from app.services import gics_registry as reg
from app.services import industry_analytics as ia
from app.services import industry_classification as ic
from app.services import industry_report_store as rs
from app.services import industry_report_worker as jobs
from app.tests.fixtures import industry_analyst_stub
from app.tests.gating_helpers import seed_demo_universe

# A Sunday 06:30 UTC tick and the Friday close it publishes.
SUNDAY = datetime(2026, 9, 6, 6, 30)
AS_OF = datetime(2026, 9, 4, 21, 0)
PERIOD = "2026-W36"
CUTOFF = AS_OF.date()


def _series(drift: float) -> list[dict[str, Any]]:
    """A daily series rising steadily to the cutoff — enough history for
    every horizon, so no return is null for a reason the test did not ask
    for."""
    return [
        {"date": (CUTOFF - timedelta(days=i)).isoformat(),
         "close": round(100.0 * (1 + drift * (400 - i) / 400), 4)}
        for i in range(400, -1, -1)
    ]


@pytest.fixture(scope="module")
def env():
    """Two real industry groups with enough classified constituents to
    clear the sample floor, plus stub loaders that serve their prices.

    Codes come from the classification, never from a literal: which demo
    company lands in which group is the registry's business, not this
    test's.
    """
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    ic.classify_all(version=info)
    by_group = ic.constituents_by_group(version=info)
    floor = int(settings.industry_stats_min_sample)
    big = [c for c, t in sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0])) if len(t) >= floor]
    assert len(big) >= 2, f"demo universe classified too few groups above the sample floor: {big}"
    codes = big[:2]
    prices = {
        t: _series(0.1 + 0.01 * i)
        for i, t in enumerate(sorted({t for c in codes for t in by_group[c]}))
    }
    groups = {c: by_group[c] for c in codes}
    yield {"info": info, "codes": codes, "groups": groups, "prices": prices}
    reg.activate_version(info.version_key)


@pytest.fixture(autouse=True)
def _stub_reads(env, monkeypatch):
    """Every DB/provider read the analytics performs, replaced by the
    fixture's in-memory series. `default_loaders` is the seam: the
    dataclass captures the real functions as field defaults, so patching
    the factory is what actually redirects a context built inside a job."""
    def loaders() -> ia.Loaders:
        return ia.Loaders(
            constituents_by_group=lambda version: dict(env["groups"]),
            companies=lambda ts: {
                t: {"company_name": f"{t} Co", "market_cap": 1.0e9, "is_active": True,
                    "shares_outstanding": None, "last_price": None}
                for t in ts
            },
            metrics=lambda ts: {},
            cached_prices=lambda ts: {t: env["prices"][t] for t in ts if t in env["prices"]},
            cached_price_tickers=lambda ts: {t for t in ts if t in env["prices"]},
            fetch_prices=lambda t: env["prices"].get(t),
            market_factor=lambda: None,
            prior_stats=lambda code, version, key: None,
        )

    monkeypatch.setattr(ia, "default_loaders", loaders)


@pytest.fixture(autouse=True)
def _wipe(env):
    """Jobs, editions, stats and snapshots for the fixture's groups, before
    and after every test."""
    def clean() -> None:
        vid, codes = env["info"].id, env["codes"]
        with SessionLocal() as db:
            db.execute(delete(IndustryReportJob).where(IndustryReportJob.taxonomy_version_id == vid))
            db.execute(delete(IndustryReport).where(
                IndustryReport.taxonomy_version_id == vid, IndustryReport.industry_group_code.in_(codes)))
            db.execute(delete(IndustryStatSnapshot).where(
                IndustryStatSnapshot.taxonomy_version_id == vid,
                IndustryStatSnapshot.industry_group_code.in_(codes)))
            db.execute(delete(CrossIndustrySnapshot).where(
                CrossIndustrySnapshot.taxonomy_version_id == vid))
            db.commit()

    clean()
    yield
    clean()


@pytest.fixture()
def analyst(monkeypatch):
    """The deterministic stand-in analyst model (see
    `fixtures/industry_analyst_stub.py`). Without it the test environment
    has no LLM, every edition is a template, and templates are stored
    audit-only (owner decision 1) — a test that needs a PUBLISHED edition
    asks for this fixture."""
    industry_analyst_stub.install(monkeypatch)


def _job(job_id: int) -> dict[str, Any]:
    with SessionLocal() as db:
        row = db.get(IndustryReportJob, job_id)
        assert row is not None
        return jobs._job_dict(row)


def _boom(*_a, **_kw):
    raise RuntimeError("writer exploded")


# --- period arithmetic --------------------------------------------------------


def test_the_period_is_the_prior_friday_close_and_round_trips():
    """The jobs table stores a period key, not a timestamp, so the key has
    to name exactly one as-of — otherwise a retried job silently computes a
    different week from the one the loop enqueued."""
    assert jobs.period_for(SUNDAY) == (PERIOD, AS_OF)
    assert jobs.as_of_for_period(PERIOD) == AS_OF
    assert ia.period_key_for(jobs.as_of_for_period(PERIOD)) == PERIOD
    # A tick on the as-of weekday itself reaches back a whole week rather
    # than reading a session that has not closed.
    friday_key, friday_as_of = jobs.period_for(datetime(2026, 9, 4, 6, 30))
    assert friday_as_of == AS_OF - timedelta(days=7) and friday_key == "2026-W35"
    with pytest.raises(ValueError):
        jobs.as_of_for_period("not-a-week")


# --- enqueue ------------------------------------------------------------------


def test_enqueue_coalesces_an_active_job_for_the_same_group_and_period(env):
    code = env["codes"][0]
    first, created = jobs.enqueue(code, PERIOD, version=env["info"], source="weekly_cron")
    assert created and first["kind"] == "group_report" and first["priority"] == 100
    second, created_again = jobs.enqueue(code, PERIOD, version=env["info"], source="admin")
    assert not created_again and second["id"] == first["id"]
    # A different period is a different job; a finished job does not coalesce.
    other, created_other = jobs.enqueue(code, "2026-W35", version=env["info"])
    assert created_other and other["id"] != first["id"]


def test_the_job_kind_is_the_string_the_report_store_filters_on(env):
    """`report_store.last_attempt()` selects on kind == 'group_report'.
    Any other string makes a failed refresh report `last_attempt: null`
    next to a stale edition — the opposite of what it is for."""
    code = env["codes"][0]
    jobs.enqueue(code, PERIOD, version=env["info"])
    assert rs.last_attempt(code, version=env["info"])["status"] == "queued"


def test_enqueue_period_skips_a_published_week_unless_forced(env):
    code = env["codes"][0]
    rs.save_report(code=code, period_key=PERIOD, as_of=AS_OF, payload={"sections": {}},
                   version=env["info"])
    result = jobs.enqueue_period(PERIOD, [code], source="weekly_cron", force=False,
                                 version=env["info"])
    assert result["enqueued"] == 0 and result["skipped_published"] == 1
    assert result["skipped_published_codes"] == [code]
    # Nothing will write stats for the period, so no snapshot job is made —
    # and the result says why rather than reporting a silent zero.
    assert result["cross_snapshot"]["job_id"] is None
    assert "no group job" in result["cross_snapshot"]["reason"]

    forced = jobs.enqueue_period(PERIOD, [code], source="admin", force=True, version=env["info"])
    assert forced["enqueued"] == 1 and forced["skipped_published"] == 0
    assert forced["cross_snapshot"]["job_id"] is not None


def test_a_week_held_in_review_is_not_generated_a_second_time(env, analyst, monkeypatch):
    """With `INDUSTRY_REPORTS_REQUIRE_REVIEW` on, an edition lands as
    `pending_review` and never takes `is_latest_good`. A skip check that
    asks "is there a latest-good edition for this period" therefore sees
    none, re-enqueues the whole week and pays the model to reproduce
    reports that are already sitting in the review queue — every Sunday,
    forever. A week that has been *generated* is not generated again;
    whether a human has released it is a separate question."""
    monkeypatch.setattr(settings, "industry_reports_require_review", True)
    jobs.enqueue_period(PERIOD, env["codes"], version=env["info"], include_cross_snapshot=False)
    jobs.drain()

    edition = rs.latest_good(env["codes"][0], version=env["info"])
    assert edition is None, "a pending_review edition must not be published"
    assert rs.history(env["codes"][0], version=env["info"])[0]["status"] == "pending_review"

    again = jobs.enqueue_period(PERIOD, env["codes"], version=env["info"],
                                include_cross_snapshot=False)
    assert again["enqueued"] == 0
    assert again["skipped_published"] == len(env["codes"])
    # `force` is still the way to deliberately regenerate one.
    assert jobs.enqueue_period(PERIOD, env["codes"][:1], version=env["info"], force=True,
                               include_cross_snapshot=False)["enqueued"] == 1


def test_enqueue_period_counts_everything_it_did_not_enqueue(env):
    """A truncated run must say how much it dropped. `enqueued=1` with 24
    groups silently unserved is how a week goes missing."""
    codes = env["codes"]
    result = jobs.enqueue_period(PERIOD, [*codes, "9999"], source="weekly_cron",
                                 version=env["info"], max_jobs=1)
    assert result["enqueued"] == 1
    assert result["over_budget"] == 1 and result["over_budget_codes"] == [codes[1]]
    assert result["unknown_codes"] == ["9999"]
    assert result["n_groups"] == 2 and result["max_jobs_per_run"] == 1


def test_enqueue_period_defaults_to_every_active_group(env):
    result = jobs.enqueue_period(PERIOD, source="weekly_cron", version=env["info"], max_jobs=500)
    assert result["n_groups"] == len(reg.industry_groups(version=env["info"]))
    assert result["enqueued"] == result["n_groups"]


# --- claim --------------------------------------------------------------------


def test_a_claim_is_exclusive_and_the_snapshot_waits_for_the_group_jobs(env):
    result = jobs.enqueue_period(PERIOD, env["codes"], version=env["info"])
    cross_id = result["cross_snapshot"]["job_id"]
    first = jobs.claim_next_job(now=SUNDAY)
    second = jobs.claim_next_job(now=SUNDAY)
    assert {first.job_id, second.job_id} == set(result["job_ids"])
    assert _job(first.job_id)["status"] == "running" and _job(first.job_id)["attempts"] == 1
    # Both group jobs are in flight, so the snapshot is pushed out rather
    # than computed from a half-written period.
    assert jobs.claim_next_job(now=SUNDAY) is None
    deferred = _job(cross_id)
    assert deferred["status"] == "queued"
    assert deferred["not_before"] == (SUNDAY + timedelta(minutes=jobs.CROSS_DEFER_MINUTES)).isoformat()
    # A second drainer trying to claim a row that is already running loses.
    with SessionLocal() as db:
        assert jobs._claim(db, first.job_id, 1) is None
    # Once the group jobs finish, the snapshot is claimable.
    with SessionLocal() as db:
        for job_id in (first.job_id, second.job_id):
            db.get(IndustryReportJob, job_id).status = "succeeded"
        db.commit()
    assert jobs.claim_next_job(now=SUNDAY + timedelta(minutes=10)).job_id == cross_id


def test_a_deferred_job_is_not_claimed_before_its_not_before(env):
    job, _ = jobs.enqueue(env["codes"][0], PERIOD, version=env["info"])
    with SessionLocal() as db:
        db.get(IndustryReportJob, job["id"]).not_before = SUNDAY + timedelta(minutes=15)
        db.commit()
    assert jobs.claim_next_job(now=SUNDAY) is None
    assert jobs.claim_next_job(now=SUNDAY + timedelta(minutes=16)).job_id == job["id"]


# --- execution ----------------------------------------------------------------


def test_a_drained_period_publishes_one_edition_per_group_and_one_snapshot(env, analyst):
    result = jobs.enqueue_period(PERIOD, env["codes"], version=env["info"])
    assert result["enqueued"] == len(env["codes"])
    assert result["cross_snapshot"]["created"] is True
    done = jobs.drain()
    assert [j["status"] for j in done] == ["succeeded"] * 3
    assert [j["kind"] for j in done] == ["group_report", "group_report", "cross_snapshot"]
    for code in env["codes"]:
        edition = rs.latest_good(code, version=env["info"])
        assert edition["period_key"] == PERIOD and edition["version"] == 1
        assert edition["stats_id"] is not None
        # Only an analyst-written edition publishes; the stub analyst wrote
        # every section, and the edition says who it was.
        assert edition["payload"]["analyst_narrative"] == "llm"
        assert edition["generation"]["model"] == industry_analyst_stub.MODEL
        assert not [d for d in edition["degraded"] if d.startswith("analyst_narrative:")]
        assert edition["coverage"]["n_constituents"] >= settings.industry_stats_min_sample
        assert edition["freshness"]["data_as_of"] == AS_OF.isoformat()
        assert rs.freshness(code, version=env["info"])["last_attempt"]["status"] == "succeeded"
    assert jobs.recent_jobs(status="succeeded")[0]["kind"] == "cross_snapshot"
    with SessionLocal() as db:
        snap = db.execute(select(CrossIndustrySnapshot).where(
            CrossIndustrySnapshot.period_key == PERIOD)).scalars().first()
    assert snap is not None and snap.as_of == AS_OF
    assert jobs.drain() == []


def test_one_analytics_context_is_shared_across_a_period_and_dropped_between_them(env, analyst, monkeypatch):
    """The benchmark cohort is frozen when `load_context` returns. Two
    contexts for one period means two cohorts, so the groups of that week
    stop being comparable — and the whole universe's prices get re-read
    per group."""
    calls: list[str] = []
    real = ia.load_context

    def spy(as_of, **kw):
        calls.append(as_of.isoformat())
        return real(as_of, **kw)

    monkeypatch.setattr(ia, "load_context", spy)
    jobs.enqueue_period(PERIOD, env["codes"], version=env["info"], include_cross_snapshot=False)
    jobs.enqueue_period("2026-W35", env["codes"][:1], version=env["info"],
                        include_cross_snapshot=False)
    jobs.drain()
    assert calls == [AS_OF.isoformat()] * 1 + [(AS_OF - timedelta(days=7)).isoformat()], calls


def test_a_failure_backs_off_on_the_injected_clock_and_keeps_the_prior_edition(env, monkeypatch):
    code = env["codes"][0]
    # Both modules' clocks are injected: the store stamps `generated_at`
    # and decides staleness, the queue decides the backoff, and the point
    # of the test is the relation between the two.
    store_clock = {"t": SUNDAY - timedelta(days=1)}
    monkeypatch.setattr(rs, "_utcnow", lambda: store_clock["t"])
    prior = rs.save_report(code=code, period_key="2026-W35", as_of=AS_OF - timedelta(days=7),
                           payload={"sections": {}}, version=env["info"],
                           generation={"generation_mode": "llm"})
    monkeypatch.setattr(jobs.writer, "write_report", _boom)

    clock = {"t": SUNDAY}
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock["t"])
    monkeypatch.setattr(jobs.industry_lease, "utcnow", lambda: clock["t"])
    job, _ = jobs.enqueue(code, PERIOD, version=env["info"])
    assert jobs.process_next_job(now=clock["t"])["status"] == "queued"
    first = _job(job["id"])
    assert first["attempts"] == 1 and first["error_type"] == "RuntimeError"
    assert first["not_before"] == (SUNDAY + timedelta(minutes=jobs.BACKOFF_MINUTES)).isoformat()

    clock["t"] = SUNDAY + timedelta(minutes=20)
    jobs.process_next_job(now=clock["t"])
    second = _job(job["id"])
    assert second["attempts"] == 2
    assert second["not_before"] == (
        clock["t"] + timedelta(minutes=2 * jobs.BACKOFF_MINUTES)).isoformat()

    clock["t"] = clock["t"] + timedelta(minutes=61)
    jobs.process_next_job(now=clock["t"])
    final = _job(job["id"])
    assert final["status"] == "failed" and final["attempts"] == 3 and final["finished_at"]
    store_clock["t"] = clock["t"] + timedelta(minutes=1)

    # The page never blanks: the previous edition is still the latest good
    # one, and the reader is told a refresh failed.
    latest = rs.latest_good(code, version=env["info"])
    assert latest["version"] == prior.version and latest["period_key"] == "2026-W35"
    fresh = rs.freshness(code, version=env["info"])
    assert fresh["stale"] is True and fresh["last_attempt"]["status"] == "failed"
    assert fresh["last_attempt"]["error_type"] == "RuntimeError"
    assert "refresh attempt failed" in fresh["stale_reason"]


def test_the_final_attempt_runs_deterministic_and_is_stored_audit_only(env, monkeypatch):
    """Two model failures still end the job: the last attempt runs the
    writer that needs no LLM. Its edition is kept for audit and NEVER
    published (owner decision 1) — the group has no analyst edition, so
    nothing is the latest good one, and the attempt says what it produced."""
    code = env["codes"][0]
    seen: list[bool] = []
    real = jobs.writer.write_report

    def flaky(*args, deterministic: bool = False, **kw):
        seen.append(deterministic)
        if not deterministic:
            raise RuntimeError("model refused")
        return real(*args, deterministic=True, **kw)

    monkeypatch.setattr(jobs.writer, "write_report", flaky)
    clock = {"t": SUNDAY}
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock["t"])
    monkeypatch.setattr(jobs.industry_lease, "utcnow", lambda: clock["t"])
    job, _ = jobs.enqueue(code, PERIOD, version=env["info"])
    for _ in range(3):
        jobs.process_next_job(now=clock["t"])
        clock["t"] = clock["t"] + timedelta(hours=1)  # past each backoff
    assert seen == [False, False, True]
    assert _job(job["id"])["status"] == "succeeded"
    assert rs.latest_good(code, version=env["info"]) is None
    [edition] = rs.history(code, version=env["info"], include_withheld=True)
    assert edition["period_key"] == PERIOD and edition["stored_status"] == "audit_only"
    assert edition["is_latest_good"] is False
    assert "generation:deterministic_final_attempt:3" in edition["degraded"]
    assert rs.history(code, version=env["info"]) == []
    assert rs.last_attempt(code, version=env["info"])["outcome"] == "withheld_template"
    assert "published=audit_only" in _job(job["id"])["progress"][-1]["step"]


def test_non_final_template_attempt_is_retried_not_stored(env, monkeypatch):
    """No LLM (or an open breaker, which answers in milliseconds) makes the
    writer return the template. Storing it on attempt 1 would end the
    group's week before the backoff gave the model another chance: a
    non-final template attempt raises `AnalystUnavailable` before anything
    is saved, and only the final attempt stores the audit-only copy."""
    code = env["codes"][0]
    clock = {"t": SUNDAY}
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock["t"])
    monkeypatch.setattr(jobs.industry_lease, "utcnow", lambda: clock["t"])
    job, _ = jobs.enqueue(code, PERIOD, version=env["info"])

    for attempt in (1, 2):
        assert jobs.process_next_job(now=clock["t"])["status"] == "queued"
        row = _job(job["id"])
        assert row["attempts"] == attempt and row["error_type"] == "AnalystUnavailable"
        assert "only the final attempt stores an audit-only copy" in row["error_message"]
        assert row["not_before"] == (clock["t"] + timedelta(minutes=jobs.BACKOFF_MINUTES * attempt)).isoformat()
        assert rs.history(code, version=env["info"], include_withheld=True) == [], "nothing is stored"
        # Not claimable before the backoff; claimable just after it.
        assert jobs.claim_next_job(now=clock["t"] + timedelta(minutes=jobs.BACKOFF_MINUTES * attempt - 1)) is None
        clock["t"] = clock["t"] + timedelta(minutes=jobs.BACKOFF_MINUTES * attempt + 1)

    assert jobs.process_next_job(now=clock["t"])["status"] == "succeeded"
    stored = rs.history(code, version=env["info"], include_withheld=True)
    assert [e["stored_status"] for e in stored] == ["audit_only"]
    assert rs.latest_good(code, version=env["info"]) is None


def test_thin_agentic_attempt_is_retried_then_stored_audit_only(env, monkeypatch):
    """A model that answers but writes too little (here: everything except
    drivers and outlook) is not an analyst edition. Non-final → retried;
    the final attempt is the deterministic template → audit-only."""
    industry_analyst_stub.install(
        monkeypatch, sections=[s for s in rs.INTERPRETED_SECTIONS if s not in ("drivers", "outlook")])
    code = env["codes"][0]
    clock = {"t": SUNDAY}
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock["t"])
    monkeypatch.setattr(jobs.industry_lease, "utcnow", lambda: clock["t"])
    job, _ = jobs.enqueue(code, PERIOD, version=env["info"])
    jobs.process_next_job(now=clock["t"])
    row = _job(job["id"])
    assert row["error_type"] == "AnalystUnavailable" and "drivers, outlook" in row["error_message"]
    for _ in range(2):
        clock["t"] = clock["t"] + timedelta(hours=1)
        jobs.process_next_job(now=clock["t"])
    assert _job(job["id"])["status"] == "succeeded"
    assert rs.latest_good(code, version=env["info"]) is None
    assert [e["stored_status"] for e in rs.history(code, version=env["info"], include_withheld=True)] == ["audit_only"]


def test_template_week_is_not_reenqueued(env):
    """A week whose only product was an audit-only template has been
    GENERATED; re-enqueueing it every Sunday would silently re-spend three
    attempts. `force` is the deliberate retry."""
    code = env["codes"][0]
    template = rs.save_report(code=code, period_key=PERIOD, as_of=AS_OF, payload={"sections": {}},
                              version=env["info"], generation={"generation_mode": "deterministic"})
    assert template.status == "audit_only"
    again = jobs.enqueue_period(PERIOD, [code], version=env["info"], include_cross_snapshot=False)
    assert again["enqueued"] == 0 and again["skipped_published_codes"] == [code]
    # ...but it is not published, and the result says which skips those are.
    assert again["skipped_withheld"] == 1 and again["skipped_withheld_codes"] == [code]
    forced = jobs.enqueue_period(PERIOD, [code], version=env["info"], force=True, include_cross_snapshot=False)
    assert forced["enqueued"] == 1


# --- the week's outcome -------------------------------------------------------


def _outcome_job(code: str, info, *, status: str, report_id: int | None = None, period_key: str = PERIOD) -> None:
    with SessionLocal() as db:
        db.add(IndustryReportJob(
            kind="group_report", taxonomy_version_id=info.id, industry_group_code=code, period_key=period_key,
            run_id="r", status=status, attempts=1, max_attempts=3, enqueued_at=SUNDAY, report_id=report_id,
        ))
        db.commit()


def test_period_outcome_counts_and_health(env, monkeypatch):
    """Each group lands in exactly one bucket, and the verdict waits for the
    week to finish: `healthy` is None while anything is pending."""
    info = env["info"]
    groups = [g.code for g in reg.industry_groups(version=info)]
    agentic, template, failed, pending = groups[0], groups[1], groups[2], groups[3]
    extra = groups[4:6]
    with SessionLocal() as db:  # these groups are outside the fixture's wipe
        db.execute(delete(IndustryReport).where(IndustryReport.taxonomy_version_id == info.id,
                                                IndustryReport.industry_group_code.in_(groups[:6])))
        db.commit()
    try:
        a = rs.save_report(code=agentic, period_key=PERIOD, as_of=AS_OF, payload={}, version=info,
                           generation={"generation_mode": "llm"})
        t = rs.save_report(code=template, period_key=PERIOD, as_of=AS_OF, payload={}, version=info,
                           generation={"generation_mode": "deterministic"})
        _outcome_job(agentic, info, status="succeeded", report_id=a.id)
        _outcome_job(template, info, status="succeeded", report_id=t.id)
        _outcome_job(failed, info, status="failed")
        _outcome_job(pending, info, status="queued")
        out = jobs.period_outcome(PERIOD, info)
        assert (out["agentic"], out["template"], out["failed"], out["pending"], out["groups"]) == (1, 1, 1, 1, 4)
        assert out["complete"] is False and out["healthy"] is None
        assert out["template_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert out["not_updated_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert out["not_updated_codes"] == sorted([template, failed])

        with SessionLocal() as db:  # the pending group finishes with an analyst edition
            db.execute(delete(IndustryReportJob).where(IndustryReportJob.industry_group_code == pending))
            db.commit()
        for code in [pending, *extra]:
            rid = rs.save_report(code=code, period_key=PERIOD, as_of=AS_OF, payload={}, version=info,
                                 generation={"generation_mode": "llm"}).id
            _outcome_job(code, info, status="succeeded", report_id=rid)
        out = jobs.period_outcome(PERIOD, info)
        assert (out["agentic"], out["template"], out["failed"], out["pending"]) == (4, 1, 1, 0)
        assert out["not_updated_rate"] == pytest.approx(2 / 6, abs=1e-4)
        assert out["healthy"] is False, "2 of 6 not updated is above the 0.10 default"
        monkeypatch.setattr(settings, "industry_report_not_updated_unhealthy_rate", 0.5)
        assert jobs.period_outcome(PERIOD, info)["healthy"] is True
        note = jobs.period_outcome_note(out, prefix="prev_")
        for token in ("prev_period=2026-W36", "prev_agentic=4", "prev_template=1", "prev_failed=1",
                      "prev_pending=0", "prev_healthy=False", "prev_threshold=0.1"):
            assert token in note, note
    finally:
        with SessionLocal() as db:
            db.execute(delete(IndustryReport).where(IndustryReport.taxonomy_version_id == info.id,
                                                    IndustryReport.industry_group_code.in_(groups[:6])))
            db.commit()


@pytest.fixture()
def weekly_row():
    """The weekly loop's shared CronLoopRun row, cleared before and after."""
    from app.models import CronLoopRun
    from app.monitoring import _LAST_RUNS
    from app.monitoring.industry_weekly_loop import LOOP_NAME

    def clear() -> None:
        _LAST_RUNS.pop(LOOP_NAME, None)
        with SessionLocal() as db:
            db.execute(delete(CronLoopRun).where(CronLoopRun.loop_name == LOOP_NAME))
            db.commit()

    clear()
    yield LOOP_NAME
    clear()


def _weekly_progress(loop_name: str):
    """Read the row the way the web service's cron-health does: straight
    from the database, in a fresh session — not from this process's memory."""
    from app.models import CronLoopRun

    with SessionLocal() as db:
        row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == loop_name).one_or_none()
        return None if row is None else (row.progress_note, row.progress_success)


def test_group_job_completion_records_weekly_progress(env, analyst, weekly_row, monkeypatch):
    """The drainer (worker process) writes the week's outcome to the loop's
    DB row as progress; cron-health (web process) reads it from there. A
    module-level dict could not cross that boundary."""
    monkeypatch.setattr(jobs, "_utcnow", lambda: SUNDAY)
    monkeypatch.setattr(jobs.industry_lease, "utcnow", lambda: SUNDAY)
    jobs.enqueue_period(PERIOD, env["codes"], version=env["info"], include_cross_snapshot=False)
    jobs.process_next_job(now=SUNDAY)
    note, success = _weekly_progress(weekly_row)
    assert f"period={PERIOD}" in note and "pending=1" in note and success is None
    jobs.process_next_job(now=SUNDAY)
    note, success = _weekly_progress(weekly_row)
    assert "agentic=2" in note and "template=0" in note and "pending=0" in note
    assert "not_updated_rate=0.0" in note and success is True

    from app.api import routes_admin
    by_name = {r["loop"]: r for r in routes_admin.cron_health_endpoint()["loops"]}
    assert by_name[weekly_row]["progress_note"] == note


def test_progress_only_for_current_period(env, analyst, weekly_row, monkeypatch):
    """A forced regenerate of an OLD week finishing mid-week must not write
    that week's verdict over this week's."""
    later = SUNDAY + timedelta(days=14)  # the current period is two weeks on
    monkeypatch.setattr(jobs, "_utcnow", lambda: later)
    monkeypatch.setattr(jobs.industry_lease, "utcnow", lambda: later)
    assert jobs.period_for(later)[0] != PERIOD
    jobs.enqueue(env["codes"][0], PERIOD, version=env["info"], force=True, source="admin")
    assert jobs.process_next_job(now=later)["status"] == "succeeded"
    assert _weekly_progress(weekly_row) is None
    # …while a job for the current period does record.
    current = jobs.period_for(later)[0]
    jobs.enqueue(env["codes"][0], current, version=env["info"])
    jobs.process_next_job(now=later)
    note, _ = _weekly_progress(weekly_row)
    assert f"period={current}" in note


def test_a_rejected_edition_is_retried_and_counts_the_problems_it_cannot_show(env, monkeypatch):
    code = env["codes"][0]
    monkeypatch.setattr(jobs.validator, "validate", lambda payload, facts: [f"p{i}" for i in range(14)])
    job, _ = jobs.enqueue(code, PERIOD, version=env["info"])
    jobs.process_next_job(now=SUNDAY)
    row = _job(job["id"])
    assert row["status"] == "queued" and row["error_type"] == "ReportRejected"
    assert row["error_message"].startswith("14 validation problem(s): p0")
    assert "+4 more problems not shown" in row["error_message"]
    assert rs.latest_good(code, version=env["info"]) is None  # nothing published


def test_retry_passes_the_rejection_to_the_writer(env, monkeypatch):
    """A retry after a validator rejection is told what was rejected —
    otherwise attempt 2 is attempt 1's prompt again and fails the same
    way. The deterministic final attempt makes no model call and gets no
    notes; neither does a first attempt."""
    code = env["codes"][0]
    seen: list[tuple[bool, Any]] = []
    real = jobs.writer.write_report

    def recording(*args, deterministic: bool = False, repair_notes: Any = None, **kw):
        seen.append((deterministic, repair_notes))
        return real(*args, deterministic=deterministic, **kw)

    problems = {1: ["outlook: number '75%' is not in the facts or a registered assumption"],
                2: ["outlook: FA1 has no explicit horizon"]}
    monkeypatch.setattr(jobs.writer, "write_report", recording)
    monkeypatch.setattr(jobs.validator, "validate", lambda payload, facts: problems.get(len(seen), []))
    clock = {"t": SUNDAY}
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock["t"])
    monkeypatch.setattr(jobs.industry_lease, "utcnow", lambda: clock["t"])
    job, _ = jobs.enqueue(code, PERIOD, version=env["info"])
    messages = []
    for _ in range(3):
        jobs.process_next_job(now=clock["t"])
        messages.append(_job(job["id"])["error_message"])
        clock["t"] = clock["t"] + timedelta(hours=1)  # past each backoff

    assert messages[0] == "1 validation problem(s): " + problems[1][0]
    assert seen == [(False, ""), (False, messages[0]), (True, "")]
    assert _job(job["id"])["status"] == "succeeded"  # the final attempt stored its audit-only copy


def test_a_retry_after_a_crash_carries_no_repair_notes(env, monkeypatch):
    """Only a validator rejection is something the model can repair; a
    crash's message is not handed to it."""
    code = env["codes"][0]
    seen: list[Any] = []
    real = jobs.writer.write_report

    def crash_once(*args, repair_notes: Any = None, **kw):
        seen.append(repair_notes)
        if len(seen) == 1:
            raise RuntimeError("model timed out")
        return real(*args, **kw)

    monkeypatch.setattr(jobs.writer, "write_report", crash_once)
    clock = {"t": SUNDAY}
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock["t"])
    monkeypatch.setattr(jobs.industry_lease, "utcnow", lambda: clock["t"])
    jobs.enqueue(code, PERIOD, version=env["info"])
    for _ in range(2):
        jobs.process_next_job(now=clock["t"])
        clock["t"] = clock["t"] + timedelta(hours=1)
    assert seen == ["", ""]


def test_an_unwritable_report_never_creates_a_row(env, monkeypatch):
    code = env["codes"][0]
    monkeypatch.setattr(jobs.writer, "write_report", _boom)
    jobs.enqueue(code, PERIOD, version=env["info"])
    jobs.process_next_job(now=SUNDAY)
    with SessionLocal() as db:
        rows = db.execute(select(IndustryReport).where(
            IndustryReport.industry_group_code == code,
            IndustryReport.taxonomy_version_id == env["info"].id)).scalars().all()
    assert rows == []


# --- what the snapshot feeds, and what a zero means ---------------------------


def _snapshot(period_key: str, *, groups: list[dict[str, Any]],
              major: list[dict[str, Any]]) -> dict[str, Any]:
    """A cross-industry snapshot row shaped like `compute_cross_snapshot`'s."""
    return {
        "period_key": period_key,
        "payload": {
            "as_of": AS_OF.isoformat(),
            "groups": groups,
            "major_events": major,
            "events_window_days": 14,
            "n_events": 1, "n_news": 1,
            "events_sources": {"catalysts": {"n": 1, "reason": None},
                               "news": {"n": 1, "reason": None}},
        },
    }


def test_a_prior_period_snapshot_labels_every_section_it_feeds(env, monkeypatch):
    """One snapshot feeds three sections. The cross-industry spillovers are
    the obvious one, but the same payload also supplies the companies
    section's event window and the outlook's macro regime — so a single
    `cross_industry:…` label reads as if only that section were degraded
    while the other two quietly carry last week's window under this week's
    as-of."""
    prior = _snapshot("2026-W35", groups=[], major=[])
    monkeypatch.setattr(jobs.industry_snapshot, "snapshot_for_period", lambda k, **kw: None)
    monkeypatch.setattr(jobs.industry_snapshot, "latest_snapshot", lambda **kw: prior)
    row, reasons = jobs._snapshot_for(PERIOD, env["info"])
    assert row is prior
    assert reasons == ["cross_industry:snapshot:prior_period:2026-W35",
                       "companies:events:prior_period:2026-W35",
                       "outlook:macro_regime:prior_period:2026-W35"]

    # No snapshot at all is the same story with a different reason.
    monkeypatch.setattr(jobs.industry_snapshot, "latest_snapshot", lambda **kw: None)
    row, reasons = jobs._snapshot_for(PERIOD, env["info"])
    assert row is None
    assert reasons == ["cross_industry:snapshot:none_yet",
                       "companies:events:none_yet",
                       "outlook:macro_regime:none_yet"]

    # And the period's own snapshot degrades nothing.
    monkeypatch.setattr(jobs.industry_snapshot, "snapshot_for_period",
                        lambda k, **kw: _snapshot(k, groups=[], major=[]))
    assert jobs._snapshot_for(PERIOD, env["info"])[1] == []


def test_events_the_snapshots_global_cap_dropped_are_counted_not_published_as_zero():
    """`compute_cross_snapshot` keeps only the top `MAX_MAJOR_EVENTS` by
    materiality across the WHOLE universe. A group whose events ranked
    below that cut arrives here with an empty list, indistinguishable from
    a genuinely quiet week — and the report would print "Events in window:
    0" as an observed fact. The group's pre-cap counts are on the snapshot
    row, so the drop is countable exactly, and must be counted."""
    row = {"code": "4530", "events_14d": 2, "news_14d": 1}
    snapshot = _snapshot(PERIOD, groups=[row], major=[])

    events, prov = jobs._events_for(snapshot, "4530")
    assert events == []
    assert prov["n_in_window_for_group"] == 3
    assert prov["n_dropped_by_snapshot_global_cap"] == 3
    assert prov["reason"] == "all_dropped_by_snapshot_global_cap"
    assert prov["snapshot_period_key"] == PERIOD and prov["window_days"] == 14
    assert prov["snapshot_global_cap"] == jobs.industry_snapshot.MAX_MAJOR_EVENTS

    # Partly dropped: one of the three survived the cut.
    kept = [{"industry_group_code": "4530", "ticker": "AAA", "kind": "news"}]
    events, prov = jobs._events_for(_snapshot(PERIOD, groups=[row], major=kept), "4530")
    assert len(events) == 1
    assert prov["n_dropped_by_snapshot_global_cap"] == 2
    assert prov["reason"] == "partly_dropped_by_snapshot_global_cap"

    # A genuinely quiet window says so, and names the channels that were read.
    quiet = {"code": "4530", "events_14d": 0, "news_14d": 0}
    events, prov = jobs._events_for(_snapshot(PERIOD, groups=[quiet], major=[]), "4530")
    assert events == [] and prov["n_dropped_by_snapshot_global_cap"] == 0
    assert prov["reason"] == "none_stored_in_window"
    assert set(prov["channels"]) == {"catalysts", "news"}

    # The two "there was nothing to read" cases keep their own reasons.
    assert jobs._events_for(None, "4530")[1]["reason"] == "no_snapshot"
    assert jobs._events_for(_snapshot(PERIOD, groups=[], major=[]), "4530")[1]["reason"] == (
        "group_absent_from_snapshot")


def test_the_edition_carries_the_event_provenance_and_counts_what_the_cap_dropped(env, analyst, monkeypatch):
    """End to end: the provenance reaches the saved edition's coverage, and
    a drop is named in `degraded` where a reader (and the UI) will see it."""
    code = env["codes"][0]
    row = {"code": code, "events_14d": 4, "news_14d": 0}
    monkeypatch.setattr(jobs.industry_snapshot, "snapshot_for_period",
                        lambda k, **kw: _snapshot(k, groups=[row], major=[]))
    jobs.enqueue(code, PERIOD, version=env["info"])
    assert jobs.process_next_job(now=SUNDAY)["status"] == "succeeded"

    edition = rs.latest_good(code, version=env["info"])
    prov = edition["coverage"]["events"]
    assert prov["n_in_report"] == 0 and prov["n_dropped_by_snapshot_global_cap"] == 4
    assert prov["reason"] == "all_dropped_by_snapshot_global_cap"
    assert "companies:events:snapshot_global_cap_dropped:4" in edition["degraded"]


# --- the validator gets facts the payload did not supply ----------------------


def test_the_validator_checks_against_independently_built_facts(env, analyst, monkeypatch):
    """The validator's first duty is to prove the published `facts` are the
    server's — an LLM never writes into facts, and every number the
    interpretation quotes must appear in them. Handing it the facts read
    back out of the payload under review makes that guard compare an object
    with itself (`x != x`, never true) and derives the allowed-numbers
    whitelist from the very document it is meant to constrain."""
    code = env["codes"][0]
    seen: dict[str, Any] = {}
    real_validate = jobs.validator.validate

    def spy(payload, facts):
        seen["facts"] = facts
        seen["payload_facts"] = (payload.get("sections") or {})["overview"]["facts"]
        return real_validate(payload, facts)

    monkeypatch.setattr(jobs.validator, "validate", spy)
    jobs.enqueue(code, PERIOD, version=env["info"])
    assert jobs.process_next_job(now=SUNDAY)["status"] == "succeeded"
    # Same values, different object: a second copy built from the inputs.
    assert seen["facts"]["overview"] == seen["payload_facts"]
    assert seen["facts"]["overview"] is not seen["payload_facts"]


def test_facts_tampered_with_after_the_writer_are_rejected(env, monkeypatch):
    """The guard that could never fire, firing. A writer regression that
    merged a model's section dict wholesale — its own `facts` key included —
    must not publish; with payload-derived facts it validated clean and the
    invented number joined the allowed-numbers whitelist on the way."""
    code = env["codes"][0]
    real_write = jobs.writer.write_report

    def tamper(*a, **kw):
        result = real_write(*a, **kw)
        result.payload["sections"]["overview"]["facts"]["n_constituents"] = 424242
        return result

    monkeypatch.setattr(jobs.writer, "write_report", tamper)
    job, _ = jobs.enqueue(code, PERIOD, version=env["info"])
    jobs.process_next_job(now=SUNDAY)

    row = _job(job["id"])
    assert row["status"] == "queued" and row["error_type"] == "ReportRejected"
    assert "facts mutated: overview" in row["error_message"]
    assert rs.latest_good(code, version=env["info"]) is None  # nothing published


# --- recovery -----------------------------------------------------------------


def test_recover_orphans_requeues_once_then_fails_and_expires_a_missed_week(env):
    code = env["codes"][0]
    running, _ = jobs.enqueue(code, PERIOD, version=env["info"])
    final, _ = jobs.enqueue(code, "2026-W35", version=env["info"])
    stale, _ = jobs.enqueue(code, "2026-W20", version=env["info"])
    now = SUNDAY
    with SessionLocal() as db:
        r = db.get(IndustryReportJob, running["id"])
        r.status, r.attempts, r.started_at = "running", 1, now - timedelta(hours=3)
        f = db.get(IndustryReportJob, final["id"])
        f.status, f.attempts = "running", f.max_attempts
        for owned in (r, f):
            owned.owner_token = f"expired-{owned.id}"
            owned.lease_expires_at = now - timedelta(seconds=1)
        s = db.get(IndustryReportJob, stale["id"])
        s.enqueued_at = now - timedelta(days=jobs.QUEUE_MAX_AGE_DAYS + 1)
        db.commit()

    assert jobs.recover_orphans(now=now) == {"requeued": 1, "failed": 1, "expired": 1}
    requeued = _job(running["id"])
    assert requeued["status"] == "queued"
    assert requeued["not_before"] == (now + timedelta(minutes=jobs.BACKOFF_MINUTES)).isoformat()
    assert requeued["progress"][-1]["step"] == "requeued_after_lease_expired"
    assert _job(final["id"])["error_type"] == "WorkerRestart"
    expired = _job(stale["id"])
    assert expired["status"] == "failed" and expired["error_type"] == "QueueExpired"
    assert str(jobs.QUEUE_MAX_AGE_DAYS) in expired["error_message"]


# --- heartbeat / lifecycle ----------------------------------------------------


def test_the_drainer_heartbeats_queue_depth_for_cron_health(env):
    from app.monitoring import status_snapshot

    jobs.enqueue(env["codes"][0], PERIOD, version=env["info"])
    note = jobs.heartbeat(now=SUNDAY)
    assert "queued=1" in note and "failed_today=0" in note
    row = status_snapshot()[jobs.HEARTBEAT_NAME]
    assert row["success"] is True and row["last_run_at"]
    assert row["note"] == note


def test_the_heartbeat_reports_a_failed_report_rather_than_a_bare_ok(env, monkeypatch):
    code = env["codes"][0]
    monkeypatch.setattr(jobs.writer, "write_report", _boom)
    job, _ = jobs.enqueue(code, PERIOD, version=env["info"], max_attempts=1)
    jobs.process_next_job(now=SUNDAY)
    assert _job(job["id"])["status"] == "failed"
    note = jobs.heartbeat()
    assert "failed_today=1" in note
    from app.monitoring import status_snapshot
    assert status_snapshot()[jobs.HEARTBEAT_NAME]["success"] is False


def test_start_worker_no_ops_under_pytest_and_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_industry_reports", False)
    assert jobs.start_worker() is False and jobs.is_running() is False
    monkeypatch.setattr(settings, "enable_industry_reports", True)
    assert jobs.start_worker() is False, "a polling thread must not run during a test session"
    assert jobs.is_running() is False
    jobs.stop_worker()


def test_dates_come_from_the_clock_seam_not_the_wall_clock(monkeypatch):
    """Every test above pins `_utcnow`; this pins that there is exactly one
    seam to pin."""
    monkeypatch.setattr(jobs, "_utcnow", lambda: datetime(2031, 1, 5, 12, 0))
    assert jobs.close_day_before(jobs._utcnow()) == date(2031, 1, 3)
    assert jobs.period_for()[0] == ia.period_key_for(date(2031, 1, 3))


# --- attribution umbrella (slice B8-A2a) --------------------------------------

def _record_context_then_fail(seen: list[dict[str, Any]]):
    from app.agents.llm import current_call_context

    def write_report(*_a, **_kw):
        seen.append(current_call_context())
        raise RuntimeError("stop after recording")
    return write_report


def test_a_job_runs_under_its_origin_job_and_run(env, monkeypatch):
    """Every LLM call a job makes carries origin=worker:industry, the job
    and the run (attribution design §4.8): recorded where the writer is
    called, inside the job's umbrella."""
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(jobs.writer, "write_report", _record_context_then_fail(seen))
    job, _ = jobs.enqueue(env["codes"][0], PERIOD, version=env["info"])
    jobs.process_next_job(now=SUNDAY)
    assert seen, "the writer was not reached"
    ctx = seen[0]
    assert ctx["origin"] == "worker:industry"
    assert ctx["job_id"] == f"industry:{job['id']}"
    assert ctx["run_id"] == _job(job["id"])["run_id"] and ctx["run_id"]


def test_a_script_draining_the_queue_keeps_its_origin(env, monkeypatch):
    """capture_industry_ui_fixture drains jobs under its own origin: that
    script started the work, so the job keeps it rather than relabelling
    the rows as the generic worker."""
    from app.agents.llm import llm_call_context

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(jobs.writer, "write_report", _record_context_then_fail(seen))
    job, _ = jobs.enqueue(env["codes"][0], PERIOD, version=env["info"])
    with llm_call_context(origin="script:capture_industry_ui_fixture"):
        jobs.process_next_job(now=SUNDAY)
    assert seen and seen[0]["origin"] == "script:capture_industry_ui_fixture"
    assert seen[0]["job_id"] == f"industry:{job['id']}"
