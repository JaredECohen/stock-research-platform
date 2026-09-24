"""FEAT-003 slice 4 — the week end to end, and the week that half fails.

The unit tests in ``test_industry_report_worker.py`` pin each mechanism in
isolation against two hand-picked groups. This file runs the whole feature
the way Sunday runs it, across *every* classified group:

    seed → classify → weekly loop (warm-up, then enqueue) → drain

and then breaks it on purpose. The failure it reproduces is the one that
actually costs a user something: a refresh that dies for one industry
group while the rest of the week publishes fine. What must survive that is
narrow and easy to get wrong — the broken group's **previous edition stays
the latest good one** (the page shows last week's analysis, not a blank),
it is **flagged stale with the reason**, its **last attempt is on the
record** with the error, and **no other group is affected**.

Nothing here reaches a provider or an LLM. The reads are redirected
through ``industry_analytics.default_loaders``, and the analyst model is
the deterministic stand-in in ``fixtures/industry_analyst_stub.py``: only
an analyst-written edition is ever published (owner decision 1), so a
week run on the bare template would publish nothing at all. The last
test runs the model-outage week for real — every group falls back to the
audit-only template, the page keeps last week's analyst edition marked
"not updated", and the week's health says so.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.database import SessionLocal
from app.models import CrossIndustrySnapshot, IndustryReport, IndustryReportJob, IndustryStatSnapshot
from app.monitoring import industry_weekly_loop as loop
from app.services import gics_registry as reg
from app.services import industry_analytics as ia
from app.services import industry_classification as ic
from app.services import industry_report_store as rs
from app.services import industry_report_worker as jobs
from app.tests.fixtures import industry_analyst_stub
from app.tests.gating_helpers import seed_demo_universe

SUNDAY = datetime(2026, 9, 6, 6, 30)
AS_OF = datetime(2026, 9, 4, 21, 0)
PERIOD = "2026-W36"
NEXT_SUNDAY = SUNDAY + timedelta(days=7)
NEXT_PERIOD = "2026-W37"


def _series(seed: int) -> list[dict[str, Any]]:
    """A daily series with enough history for every horizon, so a null
    return can only mean something the test asked for."""
    cutoff = AS_OF.date()
    drift = 0.05 + 0.005 * (seed % 17)
    return [
        {"date": (cutoff - timedelta(days=i)).isoformat(),
         "close": round(50.0 * (1 + drift * (500 - i) / 500), 4)}
        for i in range(500, -1, -1)
    ]


@pytest.fixture(scope="module")
def universe():
    """Every demo company, classified for real, with prices for all but
    two tickers — those two are what the warm-up has to go and fetch."""
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    ic.classify_all(version=info)
    by_group = ic.constituents_by_group(version=info)
    assert by_group, "the demo universe classified into no industry group at all"
    tickers = sorted({t for ts in by_group.values() for t in ts})
    prices = {t: _series(i) for i, t in enumerate(tickers)}
    # Two tickers are deliberately absent from the "cache" so the warm-up
    # has real work to do and its coverage numbers mean something.
    cold = set(tickers[:2])
    yield {"info": info, "groups": by_group, "prices": prices, "cold": cold,
           "tickers": tickers}
    reg.activate_version(info.version_key)


@pytest.fixture(autouse=True)
def _stub_reads(universe, monkeypatch):
    """Every read the analytics performs, served from the fixture. The
    warm-up's ``fetch_prices`` moves a ticker out of ``cold``, which is
    what ``data_service.get_price_history`` does in production when it
    writes the cache row the later context reads."""
    cached = set(universe["tickers"]) - set(universe["cold"])

    def fetch(ticker: str) -> list[dict[str, Any]] | None:
        rows = universe["prices"].get(ticker)
        if rows:
            cached.add(ticker)
        return rows

    def loaders() -> ia.Loaders:
        return ia.Loaders(
            constituents_by_group=lambda version: dict(universe["groups"]),
            companies=lambda ts: {
                t: {"company_name": f"{t} Co", "market_cap": 5.0e8 + 1.0e6 * len(t),
                    "is_active": True, "shares_outstanding": None, "last_price": None}
                for t in ts
            },
            metrics=lambda ts: {},
            cached_prices=lambda ts: {t: universe["prices"][t] for t in ts if t in cached},
            cached_price_tickers=lambda ts: {t for t in ts if t in cached},
            fetch_prices=fetch,
            market_factor=lambda: None,
            prior_stats=lambda code, version, key: None,
        )

    monkeypatch.setattr(ia, "default_loaders", loaders)
    industry_analyst_stub.install(monkeypatch)
    # The initial report, later attempts and leases share the simulated week.
    # Real wall time would expire historical claims and can date the prior
    # report after the deliberately advanced failed refresh.
    monkeypatch.setattr(jobs, "_utcnow", lambda: SUNDAY)
    monkeypatch.setattr(rs, "_utcnow", lambda: jobs._utcnow())
    monkeypatch.setattr(jobs.industry_lease, "utcnow", lambda: jobs._utcnow())
    monkeypatch.setattr(settings, "enable_industry_reports", True)
    return {"cached": cached}


@pytest.fixture(autouse=True)
def _wipe(universe):
    def clean() -> None:
        vid = universe["info"].id
        with SessionLocal() as db:
            for model in (IndustryReportJob, IndustryReport, IndustryStatSnapshot,
                          CrossIndustrySnapshot):
                db.execute(delete(model).where(model.taxonomy_version_id == vid))
            db.commit()

    clean()
    yield
    clean()


def _eligible(universe) -> list[str]:
    """The groups whose constituent count clears the sample floor — the
    ones a reader is entitled to expect a published edition for."""
    floor = int(settings.industry_stats_min_sample)
    return sorted(c for c, ts in universe["groups"].items() if len(ts) >= floor)


def _reports(universe) -> dict[str, dict[str, Any]]:
    return rs.latest_good_many(sorted(universe["groups"]), version=universe["info"])


def test_a_whole_week_runs_from_the_loop_to_a_published_edition_per_group(universe, monkeypatch):
    """The happy path, end to end, for every group the loop touches."""
    order: list[str] = []
    real_warm, real_enqueue = ia.warm_up_prices, jobs.enqueue_period

    def warm(**kw):
        order.append("warm_up")
        return real_warm(**kw)

    def enqueue_period(*a, **kw):
        order.append("enqueue")
        return real_enqueue(*a, **kw)

    monkeypatch.setattr(ia, "warm_up_prices", warm)
    monkeypatch.setattr(jobs, "enqueue_period", enqueue_period)

    summary = loop.run_once(now=SUNDAY)

    # The warm-up is only worth its provider budget if it lands BEFORE the
    # jobs that read the prices it fetched.
    assert order == ["warm_up", "enqueue"]
    assert summary["period_key"] == PERIOD and summary["as_of"] == AS_OF.isoformat()
    assert summary["enqueued"] == summary["groups"] > 0
    assert summary["warm_up"]["fetched"] == len(universe["cold"])
    assert summary["warm_up"]["budget"] == settings.industry_price_warmup_budget
    # Coverage is reported, not assumed: after the warm-up nothing is missing.
    assert summary["warm_up"]["remaining_missing"] == 0

    done = jobs.drain()
    assert done, "the loop enqueued a week that the drainer then found nothing to do"
    assert {j["status"] for j in done} == {"succeeded"}
    kinds = [j["kind"] for j in done]
    assert kinds.count("cross_snapshot") == 1
    assert kinds[-1] == "cross_snapshot", "the snapshot must drain after the group reports"

    published = _reports(universe)
    for code in _eligible(universe):
        edition = published.get(code)
        assert edition is not None, f"group {code} cleared the sample floor but never published"
        assert edition["period_key"] == PERIOD and edition["version"] == 1
        assert edition["stats_id"] is not None
        # Only an analyst edition publishes; this one is the stub's, and
        # its generation says so.
        assert edition["payload"]["analyst_narrative"] == "llm"
        assert edition["generation"]["model"] == industry_analyst_stub.MODEL
        assert rs.freshness(code, version=universe["info"])["stale"] is False

    with SessionLocal() as db:
        snap = db.execute(select(CrossIndustrySnapshot).where(
            CrossIndustrySnapshot.taxonomy_version_id == universe["info"].id,
            CrossIndustrySnapshot.period_key == PERIOD)).scalars().first()
    assert snap is not None and snap.as_of == AS_OF
    assert jobs.drain() == [], "the queue did not come to rest"


def test_one_group_failing_leaves_its_prior_edition_up_and_the_rest_of_the_week_intact(
    universe, monkeypatch,
):
    """A refresh that dies for one group must cost that group its *new*
    edition and nothing else — not its page, and not its neighbours'."""
    # Week one publishes normally.
    loop.run_once(now=SUNDAY)
    jobs.drain()
    week_one = _reports(universe)
    eligible = _eligible(universe)
    broken = eligible[0]
    healthy = [c for c in eligible if c != broken]
    assert healthy, "need at least two eligible groups to prove the blast radius"
    prior = week_one[broken]

    # Week two, with the writer refusing for exactly one group. Deterministic
    # mode is refused too: this is a group whose data breaks the writer, not
    # a model outage, so the final-attempt fallback cannot rescue it either.
    real_write = jobs.writer.write_report

    def selective(analyst, *a, **kw):
        if analyst.code == broken:
            raise RuntimeError(f"writer exploded for {broken}")
        return real_write(analyst, *a, **kw)

    monkeypatch.setattr(jobs.writer, "write_report", selective)
    clock = {"t": NEXT_SUNDAY}
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock["t"])

    loop.run_once(now=NEXT_SUNDAY)
    # Three passes, each past the previous attempt's backoff, so the broken
    # group exhausts its attempts instead of sitting deferred.
    for _ in range(3):
        jobs.drain(now=clock["t"])
        clock["t"] = clock["t"] + timedelta(hours=2)

    week_two = _reports(universe)

    # The blast radius is one group.
    for code in healthy:
        assert week_two[code]["period_key"] == NEXT_PERIOD, f"{code} was collateral damage"
        assert week_two[code]["version"] == week_one[code]["version"] + 1
        assert rs.freshness(code, version=universe["info"])["stale"] is False

    # The broken group's page still has last week's analysis on it...
    latest = week_two[broken]
    assert latest["version"] == prior["version"]
    assert latest["period_key"] == PERIOD
    assert latest["id"] == prior["id"]
    # ...and exactly one edition, so the failed attempt wrote no row at all.
    with SessionLocal() as db:
        n_rows = db.execute(select(IndustryReport).where(
            IndustryReport.taxonomy_version_id == universe["info"].id,
            IndustryReport.industry_group_code == broken)).scalars().all()
    assert len(n_rows) == 1

    # ...and the reader is told, with the reason, rather than shown a
    # week-old edition dated as if it were current.
    fresh = rs.freshness(broken, version=universe["info"])
    assert datetime.fromisoformat(fresh["last_attempt"]["at"]) >= datetime.fromisoformat(prior["generated_at"])
    assert fresh["stale"] is True
    assert "refresh attempt failed" in fresh["stale_reason"]
    attempt = fresh["last_attempt"]
    assert attempt["status"] == "failed"
    assert attempt["period_key"] == NEXT_PERIOD
    assert attempt["error_type"] == "RuntimeError"
    assert attempt["attempts"] == attempt["max_attempts"] == 3
    assert attempt["report_id"] is None


def test_a_failed_group_does_not_hold_the_cross_industry_snapshot_hostage(universe, monkeypatch):
    """The snapshot waits for the group jobs to *finish*, not to succeed —
    otherwise one broken group would leave the PM with no cross-industry
    context for the whole week."""
    eligible = _eligible(universe)
    broken = eligible[0]
    real_write = jobs.writer.write_report

    def selective(analyst, *a, **kw):
        if analyst.code == broken:
            raise RuntimeError("writer exploded")
        return real_write(analyst, *a, **kw)

    monkeypatch.setattr(jobs.writer, "write_report", selective)
    clock = {"t": SUNDAY}
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock["t"])

    loop.run_once(now=SUNDAY)
    for _ in range(3):
        jobs.drain(now=clock["t"])
        clock["t"] = clock["t"] + timedelta(hours=2)

    with SessionLocal() as db:
        snap = db.execute(select(CrossIndustrySnapshot).where(
            CrossIndustrySnapshot.taxonomy_version_id == universe["info"].id,
            CrossIndustrySnapshot.period_key == PERIOD)).scalars().first()
        cross_job = db.execute(select(IndustryReportJob).where(
            IndustryReportJob.taxonomy_version_id == universe["info"].id,
            IndustryReportJob.kind == jobs.KIND_CROSS)).scalars().first()
    assert cross_job is not None and cross_job.status == "succeeded"
    assert snap is not None
    # And it counts every group in the registry — including the ones with
    # no constituents and the one whose report failed — rather than
    # quietly narrowing to the set it managed to compute.
    coverage = (snap.payload or {}).get("coverage") or {}
    n_registry = len(reg.industry_groups(version=universe["info"]))
    assert coverage.get("n_groups") == n_registry
    assert coverage.get("n_insufficient_sample") is not None


def _run_week(when: datetime, monkeypatch) -> None:
    """One Sunday, start to finish, on a pinned clock."""
    monkeypatch.setattr(jobs, "_utcnow", lambda: when)
    loop.run_once(now=when)
    jobs.drain(now=when)


def test_week_two_labels_every_section_that_quotes_last_weeks_snapshot(universe, monkeypatch):
    """The cross-industry snapshot for a period is computed from the very
    stats rows the group jobs write, so it is deliberately the last job of
    the week — and every group report of that week therefore quotes the
    PRIOR period's snapshot. That payload feeds three sections, not one:
    the spillovers, the companies section's event window, and the outlook's
    macro regime. All three get a label; labelling only the cross-industry
    one would let a reader scope the staleness to the section it is
    easiest to ignore."""
    _run_week(SUNDAY, monkeypatch)
    _run_week(NEXT_SUNDAY, monkeypatch)

    published = _reports(universe)
    for code in _eligible(universe):
        edition = published[code]
        assert edition["period_key"] == NEXT_PERIOD
        assert f"cross_industry:snapshot:prior_period:{PERIOD}" in edition["degraded"]
        assert f"companies:events:prior_period:{PERIOD}" in edition["degraded"], code
        assert f"outlook:macro_regime:prior_period:{PERIOD}" in edition["degraded"], code
        # The event window the reader is shown belongs to that snapshot,
        # and the coverage block says which one.
        assert edition["coverage"]["events"]["snapshot_period_key"] == PERIOD


def test_the_snapshot_the_drainer_hands_the_writer_carries_this_groups_spillovers(universe, monkeypatch):
    """This slice's half of the cross-industry contract: by week two the
    row `_snapshot_for` returns really does contain spillovers naming the
    group being written. Pinned separately from the edition assertion below
    so that, when that one fails, it is unambiguous which side broke."""
    _run_week(SUNDAY, monkeypatch)
    row, reasons = jobs._snapshot_for(NEXT_PERIOD, universe["info"])
    assert row is not None and row["period_key"] == PERIOD
    assert reasons and all(r.endswith(f"prior_period:{PERIOD}") for r in reasons)

    spillovers = (row["payload"] or {}).get("spillovers") or []
    assert spillovers, "the snapshot computed no dependency spillovers at all"
    named = {
        code: [s for s in spillovers if code in (s.get("codes") or [])]
        for code in _eligible(universe)
    }
    # Every spillover is keyed by `codes`; a link can name one group or
    # seven, so there is no origin/destination pair to filter on.
    assert all(isinstance(s.get("codes"), list) for s in spillovers)
    assert any(named.values()), f"no spillover names any eligible group: {sorted(named)}"


def test_a_published_edition_carries_the_spillovers_that_name_its_group(universe, monkeypatch):
    """The reader-facing end of the same contract: a group named by a
    dependency link must see it in its own cross-industry section."""
    _run_week(SUNDAY, monkeypatch)
    _run_week(NEXT_SUNDAY, monkeypatch)

    row, _ = jobs._snapshot_for(NEXT_PERIOD, universe["info"])
    spillovers = (row["payload"] or {}).get("spillovers") or []
    published = _reports(universe)
    linked = [c for c in _eligible(universe)
              if any(c in (s.get("codes") or []) for s in spillovers)]
    assert linked, "fixture problem: no eligible group is named by any link"
    for code in linked:
        facts = published[code]["payload"]["sections"]["cross_industry"]["facts"]
        assert facts["spillovers"], f"{code} is named by a dependency link but its edition reports none"


def test_a_model_outage_week_keeps_last_weeks_analysis_marked_not_updated(universe, monkeypatch):
    """Owner decision 1, end to end. Week two the model answers nothing
    (an open breaker returns in milliseconds), so every group retries and
    then ends on the deterministic template — stored audit-only, never
    shown. Every page keeps week one's analyst edition, says it was not
    updated and names the week; the week's health counts it as unhealthy
    rather than as a success."""
    from app.models import CronLoopRun
    from app.monitoring import _LAST_RUNS

    loop.run_once(now=SUNDAY)
    jobs.drain()
    week_one = _reports(universe)
    assert week_one

    monkeypatch.setattr(jobs.writer, "_llm_call", lambda *a, **kw: None)
    clock = {"t": NEXT_SUNDAY}
    monkeypatch.setattr(jobs, "_utcnow", lambda: clock["t"])
    summary = loop.run_once(now=NEXT_SUNDAY)
    # The loop's note carries the PREVIOUS week's verdict: week one was clean.
    assert f"prev_period={PERIOD}" in summary["note"] and "prev_template=0" in summary["note"]
    assert summary["previous_period"]["healthy"] is True
    try:
        for _ in range(3):
            jobs.drain(now=clock["t"])
            clock["t"] = clock["t"] + timedelta(hours=2)

        week_two = _reports(universe)
        assert week_two.keys() == week_one.keys()
        for code, edition in week_two.items():
            assert edition["id"] == week_one[code]["id"], f"{code} published a template"
            fresh = rs.freshness(code, version=universe["info"])
            assert fresh["stale"] is True
            assert fresh["not_updated"] == {"period_key": NEXT_PERIOD, "outcome": "withheld_template"}
            assert f"the {NEXT_PERIOD} refresh produced no validated analyst edition" in fresh["stale_reason"]

        outcome = jobs.period_outcome(NEXT_PERIOD, universe["info"])
        assert outcome["agentic"] == 0 and outcome["pending"] == 0
        assert outcome["template"] == outcome["groups"] > 0
        assert outcome["healthy"] is False
        with SessionLocal() as db:
            row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == loop.LOOP_NAME).one()
            assert row.progress_success is False and f"period={NEXT_PERIOD}" in row.progress_note
    finally:
        _LAST_RUNS.pop(loop.LOOP_NAME, None)
        with SessionLocal() as db:
            db.query(CronLoopRun).filter(CronLoopRun.loop_name == loop.LOOP_NAME).delete()
            db.commit()
