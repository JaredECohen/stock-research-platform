"""FEAT-003 slice 5 — the Industry Analysis read API and the ops surface.

What can rot here, and what each test pins:

* a route that answers 200 with a taxonomy that was never imported
  (503 with the remedy instead), or 500s on an unknown code (404);
* a count that drifts from the registry — no literal 24 / 25 / 74 / 163
  may appear in the API, so the assertions compare the response with the
  registry and the classification table, never with a number;
* the membership-vs-coverage collapse: `/companies` must list the
  classified members and mark which ones the statistics row priced, with
  a reason for every unpriced name;
* numbers served without the method that produced them (a benchmark
  without its cohort basis, breadth without its session window);
* a failed refresh that leaves the page looking fresh — `stale` and
  `last_attempt` are composed from the jobs table;
* a page view that queues or generates work, and an N+1 on `/companies`.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, event

from app.config import settings
from app.database import SessionLocal, engine
from app.main import app
from app.models import (
    Company,
    CompanyIndustryClassification,
    IndustryReport,
    IndustryReportJob,
    IndustryStatSnapshot,
)
from app.services import gics_registry as reg
from app.services import industry_analytics as ia
from app.services import industry_report_store as store

AS_OF = datetime(2026, 9, 4, 21, 0)
NOW = datetime(2026, 9, 6, 12, 0)
TICKERS = ("ZQIND1", "ZQIND2", "ZQIND3")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def taxonomy():
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None, "the bundled knowledge JSON must import for these tests"
    yield info
    reg.activate_version(info.version_key)


@pytest.fixture()
def group(taxonomy):
    """A real active group with a real sub-industry layer, wiped of
    editions, jobs, stats and synthetic constituents on both sides."""
    node = next(
        (g for g in reg.industry_groups(version=taxonomy) if reg.sub_industries_of(g.code, version=taxonomy)),
        reg.industry_groups(version=taxonomy)[0],
    )

    def _wipe() -> None:
        with SessionLocal() as db:
            for model in (IndustryReportJob, IndustryReport, IndustryStatSnapshot):
                db.execute(delete(model).where(
                    model.taxonomy_version_id == taxonomy.id, model.industry_group_code == node.code))
            db.execute(delete(CompanyIndustryClassification).where(
                CompanyIndustryClassification.ticker.in_(TICKERS)))
            db.execute(delete(Company).where(Company.ticker.in_(TICKERS)))
            db.commit()

    _wipe()
    yield node
    _wipe()


# ---------------------------------------------------------------------------
# Fixtures that write rows
# ---------------------------------------------------------------------------


def _seed_members(node, taxonomy, *, priced: tuple[str, ...] = TICKERS[:2]) -> dict:
    """Three constituents: two the statistics row prices, one it does not."""
    industries = reg.industries_of_group(node.code, version=taxonomy)
    subs = reg.sub_industries_of(node.code, version=taxonomy)
    industry_code = industries[0].code if industries else None
    sub = subs[0] if subs else None
    with SessionLocal() as db:
        for i, ticker in enumerate(TICKERS):
            db.add(Company(
                ticker=ticker, company_name=f"{ticker} Corp", sector="Technology",
                industry="Semiconductors", sub_industry="Semiconductors",
                market_cap=1_000_000_000.0 * (i + 1), is_active=True,
            ))
            db.add(CompanyIndustryClassification(
                ticker=ticker, taxonomy_version_id=taxonomy.id,
                sector_code=node.code[:2], industry_group_code=node.code,
                industry_code=industry_code,
                sub_industry_code=sub.code if sub else None,
                sub_industry_codes=[sub.code] if sub else [],
                state="mapped", source="research_map", method="security_reference",
                author="Investment_Universe_163_Map.json@2026-09-08", source_as_of="2026-09-08",
                confidence=0.9, source_sector="Technology", source_industry="Semiconductors",
                is_current=True, classified_at=AS_OF,
            ))
        db.commit()
    return {"industry_code": industry_code, "sub": sub, "priced": priced}


def _seed_stats(node, taxonomy, *, period_key: str, ret_1m: float | None,
                priced: tuple[str, ...] = TICKERS[:2]) -> int:
    with SessionLocal() as db:
        row = IndustryStatSnapshot(
            taxonomy_version_id=taxonomy.id, industry_group_code=node.code,
            period_key=period_key, as_of=AS_OF,
            method={
                "benchmark_cohort_basis": "a classified constituent whose price series was present",
                "breadth_mean_window": {"sessions": 50, "basis": "trading sessions, not calendar days"},
                "weighting": ["equal", "market_cap"],
            },
            sample={
                "n_constituents": len(TICKERS), "n_with_prices": len(priced),
                "excluded": [{"ticker": t, "reason": "no_prices"} for t in TICKERS if t not in priced],
            },
            payload={
                "returns": {"1m": {"equal_weight": ret_1m, "market_cap_weight": ret_1m}},
                "breadth": {"1m": {"pct_positive": 0.5}},
                "benchmark_relative": {"universe_ew": {"1m": {"value": 0.01}}},
                "leaders": [{"ticker": priced[0]}], "laggards": [{"ticker": priced[-1]}],
            },
            per_ticker={
                t: {
                    "company_name": f"{t} Corp", "market_cap": 1e9, "weight_mcw": 0.5,
                    "last_close": 10.0, "last_date": "2026-09-04", "price_source": "cache",
                    "returns": {"1m": 0.02}, "above_50d_mean": True, "exclusion": None,
                }
                for t in priced
            },
            inputs_hash=f"h-{period_key}-{ret_1m}",
        )
        db.add(row)
        db.commit()
        return row.id


def _seed_report(node, taxonomy, *, period_key: str, stats_id: int | None, **kw):
    return store.save_report(
        code=node.code, period_key=period_key, as_of=AS_OF, version=taxonomy,
        payload={"sections": {"overview": {"facts": {"n_constituents": 3}, "interpretation": {"text": "x"}}}},
        stats_id=stats_id, generation={"cost_usd": 0.12, "llm_calls": 2}, **kw,
    )


def _seed_failed_job(node, taxonomy, *, at: datetime) -> None:
    with SessionLocal() as db:
        db.add(IndustryReportJob(
            kind="group_report", taxonomy_version_id=taxonomy.id, industry_group_code=node.code,
            period_key="2026-W37", run_id="run-fail", status="failed", attempts=3, max_attempts=3,
            enqueued_at=at, started_at=at, finished_at=at, source="weekly_cron",
            error_type="ValidatorRejected", error_message="drivers section opened with a KPI forecast",
        ))
        db.commit()


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_taxonomy_answers_from_the_registry_not_a_literal(client, taxonomy):
    body = client.get("/api/industries/taxonomy").json()
    counts = reg.counts(version=taxonomy)
    groups = [g for s in body["sectors"] for g in s["industry_groups"]]

    assert len(body["sectors"]) == counts["sector"]
    assert len(groups) == counts["industry_group"]
    assert sum(g["industry_count"] for g in groups) == counts["industry"]
    assert sum(g["sub_industry_count"] for g in groups) == counts["sub_industry"]
    assert body["node_counts"] == counts
    assert body["taxonomy_version"]["key"] == taxonomy.version_key
    assert body["attribution"] and body["mapping_caveat"]


def test_taxonomy_sub_industry_counts_match_the_registry_per_group(client, taxonomy):
    body = client.get("/api/industries/taxonomy").json()
    groups = {g["code"]: g for s in body["sectors"] for g in s["industry_groups"]}
    for code, entry in list(groups.items())[:6]:
        assert entry["sub_industry_count"] == len(reg.sub_industries_of(code, version=taxonomy)), code


def test_taxonomy_names_the_groups_this_universe_can_never_cover(client, taxonomy, group):
    """The structural count, in one place, and agreeing with every row.

    A group whose membership is below the sample floor is short of
    COMPANIES: it will report `insufficient_sample` every week no matter
    how long the price warm-up runs, and an operator deciding whether to
    widen the universe needs that number without reading 25 rows.
    """
    _seed_members(group, taxonomy)  # three constituents — at the floor
    body = client.get("/api/industries/taxonomy").json()
    floor = ia.sample_floor()
    groups = {g["code"]: g for s in body["sectors"] for g in s["industry_groups"]}
    summary = body["universe_coverage"]

    assert summary["min_sample"] == floor
    assert summary["setting"] == "INDUSTRY_STATS_MIN_SAMPLE"
    assert summary["groups"] == len(groups)
    assert summary["coverable"] + summary["not_coverable"] == summary["groups"]
    assert summary["basis"]

    # Every per-group verdict is the same arithmetic on the count beside
    # it, and the summary is exactly the groups that failed it.
    short = {c for c, g in groups.items() if not g["universe_coverage"]["coverable"]}
    assert short == set(summary["not_coverable_codes"])
    assert summary["not_coverable"] == len(short)
    assert summary["constituents_needed"] == sum(
        groups[c]["universe_coverage"]["constituents_short_by"] for c in short
    )
    for g in groups.values():
        cov = g["universe_coverage"]
        assert cov["min_sample"] == floor
        assert cov["constituent_count"] == g["constituent_count"]
        assert cov["coverable"] is (g["constituent_count"] >= floor)
        assert cov["constituents_short_by"] == max(floor - g["constituent_count"], 0)
        assert cov["explanation"], g["code"]

    # The seeded group sits exactly at the floor: coverable, and nothing
    # about it says "short".
    assert groups[group.code]["universe_coverage"]["coverable"] is True
    assert groups[group.code]["universe_coverage"]["constituents_short_by"] == 0

    # And the point of the whole field: this universe leaves groups the
    # weekly warm-up can never rescue, and they are named rather than
    # left to render as "not ready yet".
    assert short, "expected at least one group below the floor in this test universe"
    thin = groups[sorted(short)[0]]["universe_coverage"]
    assert "warm-up" in thin["explanation"] and str(floor) in thin["explanation"]


UNCOUNTED_TICKERS = ("ZQUNC1", "ZQUNC2", "ZQUNC3", "ZQUNC4")


def _a_group_with_no_constituents(client) -> str:
    """A real group this database has classified nothing into — read off
    the response, because the suite shares a database and another module's
    fixtures may have populated any given group."""
    body = client.get("/api/industries/taxonomy").json()
    empty = [g["code"] for s in body["sectors"] for g in s["industry_groups"] if g["constituent_count"] == 0]
    assert empty, "no empty group to seed into; this test needs one and the taxonomy is fully populated"
    return sorted(empty)[0]


def _seed_two_members(code: str, taxonomy) -> None:
    """Two `mapped` constituents — one short of the sample floor of 3."""
    with SessionLocal() as db:
        for i, ticker in enumerate(TICKERS[:2]):
            db.add(Company(
                ticker=ticker, company_name=f"{ticker} Corp", sector="Technology",
                industry="Semiconductors", is_active=True, market_cap=1e9 * (i + 1),
            ))
            db.add(CompanyIndustryClassification(
                ticker=ticker, taxonomy_version_id=taxonomy.id,
                sector_code=code[:2], industry_group_code=code,
                state="mapped", source="research_map", method="security_reference",
                author="test", source_as_of="2026-09-08", confidence=0.9,
                source_sector="Technology", source_industry="Semiconductors",
                is_current=True, classified_at=AS_OF,
            ))
        db.commit()


@pytest.fixture()
def uncounted_rows(taxonomy, group):
    """Seeds four active companies that no group's constituent count
    includes, into whichever group the test names.

    Two `fallback` rows (the provider label is not in the alias map, so
    the row knows the sector and not the group — the normal outcome of
    `resolve()`) and two `stale` rows that still name the group.
    `constituents_by_group` filters to mapped+conflict, so all four are
    invisible to it. Yields the seeder; the fixture owns the cleanup.
    """
    def seed(code: str) -> None:
        with SessionLocal() as db:
            for i, ticker in enumerate(UNCOUNTED_TICKERS):
                stale = i >= 2
                db.add(Company(
                    ticker=ticker, company_name=f"{ticker} Corp", sector="Technology",
                    industry="Widget Fabrication", is_active=True, market_cap=1e9,
                ))
                db.add(CompanyIndustryClassification(
                    ticker=ticker, taxonomy_version_id=taxonomy.id,
                    sector_code=code[:2],
                    industry_group_code=code if stale else None,
                    state="stale" if stale else "fallback",
                    source="provider_alias", method="provider_label", author="test",
                    source_as_of="2026-09-08", confidence=0.5,
                    source_sector="Technology", source_industry="Widget Fabrication",
                    is_current=True, classified_at=AS_OF,
                ))
            db.commit()

    yield seed
    with SessionLocal() as db:
        db.execute(delete(CompanyIndustryClassification).where(
            CompanyIndustryClassification.ticker.in_(UNCOUNTED_TICKERS)))
        db.execute(delete(Company).where(Company.ticker.in_(UNCOUNTED_TICKERS)))
        db.commit()


def test_a_group_short_of_constituents_is_not_reported_as_a_universe_short_of_companies(
    client, taxonomy, group, uncounted_rows,
):
    """The conflation this endpoint exists to remove, one level up.

    "The universe would have to add N more companies" is only true when
    every company already here counts towards some group. A `fallback`
    row knows its sector and not its group, and a `stale` row is waiting
    to be re-classified — both are companies in this universe that a
    single alias-map entry could put in the short group, and an operator
    told to widen the universe would be solving the wrong problem.
    """
    code = _a_group_with_no_constituents(client)
    _seed_two_members(code, taxonomy)
    uncounted_rows(code)

    body = client.get("/api/industries/taxonomy").json()
    groups = {g["code"]: g for s in body["sectors"] for g in s["industry_groups"]}
    cov = groups[code]["universe_coverage"]
    floor = ia.sample_floor()

    assert groups[code]["constituent_count"] == 2
    assert cov["coverable"] is False and cov["constituents_short_by"] == floor - 2
    # The four rows this universe holds and no group counts, two of which
    # already name this very group.
    assert cov["uncounted_in_universe"] >= len(UNCOUNTED_TICKERS)
    assert cov["uncounted_for_group"] >= 2

    text = cov["explanation"]
    assert "the universe would have to add" not in text, (
        "the response told the reader to add companies when companies already in this universe are "
        "uncounted — the shortfall is of CLASSIFIED constituents, not of companies"
    )
    assert "more classified constituent" in text
    assert "that no group counts" in text and "already name this group" in text
    assert "stale" in text and "fallback" in text

    summary = body["universe_coverage"]
    assert summary["uncounted"]["by_state"]["fallback"] >= 2
    assert summary["uncounted"]["by_state"]["stale"] >= 2
    assert summary["uncounted"]["by_group_code"][code] >= 2
    assert summary["uncounted"]["total"] == sum(summary["uncounted"]["by_state"].values())
    assert summary["uncounted"]["note"]
    # The one-place sentence an operator reads before deciding to widen
    # the universe names the other remedy too.
    assert "classifying those counts towards the gap without widening the universe" in summary["explanation"]
    assert "CLASSIFIED CONSTITUENTS" in summary["basis"]


def test_the_remedy_named_depends_on_whether_anything_is_uncounted():
    """With nothing uncounted, "add companies" IS the whole remedy and the
    response says so; with rows nobody counts, it must not. Checked on the
    composers directly, so the sentence does not depend on what another
    module happened to leave in the shared database."""
    from app.api import routes_industries as ri

    floor = ia.sample_floor()
    nothing = {"total": 0, "by_state": {}, "by_group_code": {}, "without_group_code": 0}
    pool = {"total": 4, "by_state": {"fallback": 2, "stale": 2}, "by_group_code": {"1010": 2},
            "without_group_code": 2}

    bare = ri._group_universe_coverage(floor - 1, floor, nothing, "1010")
    assert bare["uncounted_in_universe"] == 0 and bare["uncounted_for_group"] == 0
    assert "only adding companies can supply them" in bare["explanation"]

    with_pool = ri._group_universe_coverage(floor - 1, floor, pool, "1010")
    assert with_pool["uncounted_in_universe"] == 4 and with_pool["uncounted_for_group"] == 2
    assert "only adding companies can supply them" not in with_pool["explanation"]
    assert "2 fallback, 2 stale" in with_pool["explanation"]
    assert "2 of them already name this group" in with_pool["explanation"]

    # And the same split in the one-place summary.
    assert "only adding companies can" in ri._universe_coverage_explanation(
        groups=25, not_coverable=20, needed=43, floor=floor, uncounted=nothing,
    )
    assert "without widening the universe" in ri._universe_coverage_explanation(
        groups=25, not_coverable=20, needed=43, floor=floor, uncounted=pool,
    )
    # Nothing short: the reader is told the floor is not the universe's
    # problem, rather than being shown a zero.
    healthy = ri._universe_coverage_explanation(
        groups=25, not_coverable=0, needed=0, floor=floor, uncounted=nothing,
    )
    assert "waiting on prices, not on the universe" in healthy


def test_taxonomy_carries_the_access_policy(client):
    access = client.get("/api/industries/taxonomy").json()["access"]
    assert access["surface"] == "latest"
    assert access["tier"] == store.access_policy()["setting"]
    assert access["enforced"] is bool(settings.auth_enabled)
    assert set(access["surfaces"]) >= {"latest", "history", "changes", "pm_chat"}


def test_taxonomy_not_imported_is_503_with_the_remedy_and_the_policy(client, monkeypatch):
    monkeypatch.setattr(reg, "active_version", lambda: None)
    resp = client.get("/api/industries/taxonomy")
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["code"] == "taxonomy_not_imported"
    assert "import" in detail["remedy"]
    # Still answers with the policy, so the UI can explain the state.
    assert detail["access"]["surfaces"]["history"] == "pro"


@pytest.mark.parametrize("path", [
    "/api/industries/{code}/report", "/api/industries/{code}/companies",
    "/api/industries/{code}/history", "/api/industries/{code}/changes",
])
def test_reads_are_503_before_the_taxonomy_is_imported(client, monkeypatch, group, path):
    monkeypatch.setattr(reg, "active_version", lambda: None)
    resp = client.get(path.format(code=group.code))
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"]["code"] == "taxonomy_not_imported"


# ---------------------------------------------------------------------------
# Unknown codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", ["report", "companies", "history", "changes"])
@pytest.mark.parametrize("code", ["9999", "nope", "45"])
def test_unknown_group_code_is_404(client, code, suffix):
    resp = client.get(f"/api/industries/{code}/{suffix}")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"]["code"] == "unknown_industry_group"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_404_names_the_group_and_the_last_attempt(client, group, taxonomy):
    _seed_failed_job(group, taxonomy, at=NOW)
    resp = client.get(f"/api/industries/{group.code}/report")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "no_report"
    assert group.name in detail["message"]
    assert detail["last_attempt"]["status"] == "failed"
    assert detail["last_attempt"]["error_type"] == "ValidatorRejected"


def test_report_serves_the_edition_with_its_statistics_and_method(client, group, taxonomy, monkeypatch):
    monkeypatch.setattr(store, "_utcnow", lambda: NOW)
    stats_id = _seed_stats(group, taxonomy, period_key="2026-W36", ret_1m=0.03)
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=stats_id)

    body = client.get(f"/api/industries/{group.code}/report").json()
    assert body["code"] == group.code and body["name"] == group.name
    assert body["version"] == 1 and body["is_latest_good"] is True
    assert body["stale"] is False and body["stale_reason"] is None
    assert body["llm_cost_usd"] == 0.12
    assert body["disclaimer"] and body["attribution"]
    # Numbers never travel without the method that produced them.
    assert body["stats"]["payload"]["benchmark_relative"]["universe_ew"]["1m"]["value"] == 0.01
    assert body["stats"]["method"]["benchmark_cohort_basis"]
    assert body["stats"]["method"]["breadth_mean_window"]["sessions"] == 50
    assert body["stats"]["sample"]["n_constituents"] == 3
    # Per-ticker rows belong to /companies, and the response says so.
    assert "per_ticker" not in body["stats"]
    assert "companies" in body["stats"]["per_ticker_note"]


def _seed_floor_state(node, taxonomy, *, n_constituents: int, n_priced: int) -> str:
    """A stats row and its edition for a group in one sample-floor state.

    The `sample_floor` block is built by the producer
    (`industry_analytics.classify_sample_floor`) rather than typed out
    here: a hand-written copy would agree with this test and drift from
    the service, which is exactly the failure this repo has had before.
    `coverage` is composed the way the worker composes it — from the
    stats row's own `sample`.
    """
    floor = ia.sample_floor()
    sample = {
        "n_constituents": n_constituents,
        "n_with_prices": n_priced,
        "min_sample": floor,
        "excluded": [],
        "sample_floor": ia.classify_sample_floor(
            n_constituents=n_constituents, n_with_prices=n_priced, min_sample=floor,
        ),
    }
    state = str(sample["sample_floor"]["state"])
    with SessionLocal() as db:
        row = IndustryStatSnapshot(
            taxonomy_version_id=taxonomy.id, industry_group_code=node.code,
            period_key="2026-W36", as_of=AS_OF, method={"weighting": ["equal"]},
            sample=sample,
            payload={"status": "ok" if state == ia.FLOOR_MET else ia.REASON_INSUFFICIENT},
            per_ticker={},
            inputs_hash=f"floor-{n_constituents}-{n_priced}",
        )
        db.add(row)
        db.commit()
        stats_id = row.id
    _seed_report(node, taxonomy, period_key="2026-W36", stats_id=stats_id, coverage=sample)
    return state


@pytest.mark.parametrize(("n_constituents", "n_priced", "expected", "structural"), [
    (9, 1, ia.FLOOR_PRICES_NOT_WARMED, False),
    (2, 2, ia.FLOOR_UNIVERSE_TOO_SMALL, True),
    (9, 5, ia.FLOOR_MET, False),
])
def test_report_says_which_kind_of_short_the_group_is(
    client, group, taxonomy, n_constituents, n_priced, expected, structural,
):
    """`insufficient_sample` alone reads as "not ready yet". The response
    has to distinguish a warm-up that will catch up from a universe that
    never can — on both the statistics row and the coverage block the
    header renders."""
    assert _seed_floor_state(group, taxonomy, n_constituents=n_constituents, n_priced=n_priced) == expected
    body = client.get(f"/api/industries/{group.code}/report").json()

    for where in (body["stats"]["sample"]["sample_floor"], body["coverage"]["sample_floor"]):
        assert where["state"] == expected
        assert where["structural"] is structural
        assert where["clears_with_warm_up"] is (expected == ia.FLOOR_PRICES_NOT_WARMED)
        assert where["min_sample"] == ia.sample_floor()
        assert where["n_constituents"] == n_constituents
        assert where["n_with_prices"] == n_priced
        assert where["explanation"]

    # The words differ too, not just the enum: this is what the page
    # prints, and "not ready yet" for a structural shortfall is the bug.
    text = body["coverage"]["sample_floor"]["explanation"]
    if structural:
        assert "warm-up can cover it" in text
    elif expected == ia.FLOOR_PRICES_NOT_WARMED:
        assert "without changing the universe" in text


def test_report_without_a_statistics_row_says_why(client, group, taxonomy):
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=None,
                 degraded=["analyst_narrative:llm_unavailable"])
    body = client.get(f"/api/industries/{group.code}/report").json()
    assert body["stats"] is None
    assert "insufficient sample" in body["stats_unavailable_reason"]
    assert body["degraded"] == ["analyst_narrative:llm_unavailable"]


def test_failed_refresh_makes_the_edition_stale_and_names_the_attempt(client, group, taxonomy, monkeypatch):
    monkeypatch.setattr(store, "_utcnow", lambda: NOW)
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=None)
    _seed_failed_job(group, taxonomy, at=NOW + timedelta(days=7))

    body = client.get(f"/api/industries/{group.code}/report").json()
    assert body["stale"] is True
    assert "failed" in body["stale_reason"]
    assert body["last_attempt"]["error_type"] == "ValidatorRejected"
    # The last good edition is still served — a failed week is never blank.
    assert body["version"] == 1 and body["payload"]["sections"]


def test_an_old_as_of_is_stale_by_age_with_the_reason(client, group, taxonomy, monkeypatch):
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=None)
    monkeypatch.setattr(store, "_utcnow", lambda: AS_OF + timedelta(days=int(settings.industry_report_stale_after_days) + 2))
    body = client.get(f"/api/industries/{group.code}/report").json()
    assert body["stale"] is True
    assert str(settings.industry_report_stale_after_days) in body["stale_reason"]


def test_the_picker_and_the_page_agree_about_staleness_inside_a_day(
    client, group, taxonomy, monkeypatch,
):
    """`/taxonomy`'s `stale_by_age` is the report endpoint's `stale`.

    `_age_days` floored the age to whole days before comparing it with
    the same threshold the store compares as a full timedelta, so an
    edition N days and twelve hours old was `stale_by_age: false` in the
    picker and `stale: true` on its own page — for every edition in the
    [N, N+1) day window.
    """
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=None)
    stale_after = int(settings.industry_report_stale_after_days)
    half_past = AS_OF + timedelta(days=stale_after, hours=12)
    monkeypatch.setattr(store, "_utcnow", lambda: half_past)

    report = client.get(f"/api/industries/{group.code}/report").json()
    pointer = _pointer(client, group)
    assert report["stale"] is True and str(stale_after) in report["stale_reason"]
    assert pointer["stale_by_age"] is True
    # `age_days` stays the floored whole number a UI renders — the verdict
    # is what had to stop being computed from it.
    assert pointer["age_days"] == stale_after

    # And the boundary still reads fresh on both, rather than over-flagging.
    monkeypatch.setattr(store, "_utcnow", lambda: AS_OF + timedelta(days=stale_after))
    assert client.get(f"/api/industries/{group.code}/report").json()["stale"] is False
    assert _pointer(client, group)["stale_by_age"] is False


def _pointer(client, group) -> dict:
    """This group's `latest_report` pointer out of the taxonomy tree."""
    body = client.get("/api/industries/taxonomy").json()
    entry = next(
        g for s in body["sectors"] for g in s["industry_groups"] if g["code"] == group.code)
    assert entry["latest_report"] is not None
    return entry["latest_report"]


def test_report_version_number_and_bad_version(client, group, taxonomy):
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=None)
    _seed_report(group, taxonomy, period_key="2026-W37", stats_id=None)
    assert client.get(f"/api/industries/{group.code}/report?version=1").json()["version"] == 1
    assert client.get(f"/api/industries/{group.code}/report?version=latest").json()["version"] == 2
    assert client.get(f"/api/industries/{group.code}/report?version=9").status_code == 404
    bad = client.get(f"/api/industries/{group.code}/report?version=abc")
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "bad_version"


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_is_newest_first_and_counts_what_it_dropped(client, group, taxonomy):
    for week in ("2026-W35", "2026-W36", "2026-W37"):
        _seed_report(group, taxonomy, period_key=week, stats_id=None)
    body = client.get(f"/api/industries/{group.code}/history?limit=2").json()
    assert [item["version"] for item in body["items"]] == [3, 2]
    assert body["count"] == 2 and body["limit"] == 2 and body["truncated"] == 1
    assert body["items"][0]["llm_cost_usd"] == 0.12
    # Metadata only: history must not carry three report payloads.
    assert "payload" not in body["items"][0]


# ---------------------------------------------------------------------------
# Changes
# ---------------------------------------------------------------------------


def test_changes_is_arithmetic_over_the_two_statistics_rows(client, group, taxonomy):
    a = _seed_stats(group, taxonomy, period_key="2026-W35", ret_1m=0.01)
    b = _seed_stats(group, taxonomy, period_key="2026-W36", ret_1m=-0.03)
    _seed_report(group, taxonomy, period_key="2026-W35", stats_id=a)
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=b)

    body = client.get(f"/api/industries/{group.code}/changes").json()
    assert body["from"]["version"] == 1 and body["to"]["version"] == 2
    assert body["adjacent"] is True
    delta = body["facts_delta"]["returns.1m.ew"]
    assert delta["from"] == 0.01 and delta["to"] == -0.03
    assert delta["delta"] == pytest.approx(-0.04)
    # A fact neither edition carries is null with a reason, never zero.
    missing = body["facts_delta"]["valuation.pe_ttm.median"]
    assert missing["delta"] is None and missing["reason"]


def test_changes_on_the_first_edition_says_there_is_no_prior(client, group, taxonomy):
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=None)
    resp = client.get(f"/api/industries/{group.code}/changes")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "no_prior_edition"


def test_changes_accepts_explicit_non_adjacent_versions(client, group, taxonomy):
    for week in ("2026-W35", "2026-W36", "2026-W37"):
        _seed_report(group, taxonomy, period_key=week, stats_id=None)
    body = client.get(f"/api/industries/{group.code}/changes?from=1&to=3").json()
    assert body["from"]["version"] == 1 and body["to"]["version"] == 3
    assert body["adjacent"] is False


# --- review fixes: the default basis is the edition that was REPLACED ---


def test_changes_defaults_to_the_parent_edition_not_version_minus_one(
    client, group, taxonomy, monkeypatch,
):
    """With review required, `version - 1` diffs against an edition no
    reader has ever seen.

    `save_report` increments `version` on every save but points
    `parent_report_id` at the latest *good* edition, so a group can hold
    v1 (published), v2 (pending_review) and v3 (published, parent v1).
    The default basis must be v1 — the edition v3 actually replaced.
    """
    a = _seed_stats(group, taxonomy, period_key="2026-W35", ret_1m=0.01)
    c = _seed_stats(group, taxonomy, period_key="2026-W37", ret_1m=0.04)
    _seed_report(group, taxonomy, period_key="2026-W35", stats_id=a)
    monkeypatch.setattr(settings, "industry_reports_require_review", True)
    held = _seed_report(group, taxonomy, period_key="2026-W36", stats_id=None)
    assert held.status == "pending_review" and held.version == 2
    monkeypatch.setattr(settings, "industry_reports_require_review", False)
    published = _seed_report(group, taxonomy, period_key="2026-W37", stats_id=c)
    assert published.version == 3

    body = client.get(f"/api/industries/{group.code}/changes").json()
    assert body["from"]["version"] == 1, "defaulted to the unpublished v2"
    assert body["to"]["version"] == 3
    assert body["adjacent"] is True  # v3's parent IS v1
    assert body["facts_delta"]["returns.1m.ew"]["delta"] == pytest.approx(0.03)


def test_changes_without_a_parent_counts_the_editions_it_is_not_first_over(
    client, group, taxonomy, monkeypatch,
):
    """The other half of the same bug: `parent_report_id is None` means
    nothing was published before, NOT that this is the first edition on
    file. Claiming "first on file" over two unpublished editions is a
    false statement about the data."""
    monkeypatch.setattr(settings, "industry_reports_require_review", True)
    _seed_report(group, taxonomy, period_key="2026-W35", stats_id=None)
    monkeypatch.setattr(settings, "industry_reports_require_review", False)
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=None)

    resp = client.get(f"/api/industries/{group.code}/changes")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "no_prior_edition"
    assert detail["earlier_editions"] == 1
    assert "none was published" in detail["message"]
    assert "first on file" not in detail["message"]
    # …and the basis it refused to guess is still reachable explicitly.
    body = client.get(f"/api/industries/{group.code}/changes?from=1&to=2").json()
    assert body["from"]["version"] == 1 and body["to"]["version"] == 2


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


def _baseline(client, group) -> dict:
    """This group's membership BEFORE the test seeds into it.

    The suite shares one sqlite file and other modules classify real demo
    tickers, so "the group is empty" is an assumption, not a fact — it
    held when this module ran alone and broke the moment
    `test_industry_classification` ran first. Every membership count
    below is therefore a delta against this, and every row assertion
    picks its row by ticker rather than by position.
    """
    body = client.get(f"/api/industries/{group.code}/companies").json()
    return {
        "count": body["count"],
        "n_priced": body["n_priced"],
        "states": dict(body["membership_states"]),
    }


def test_companies_separates_membership_from_price_coverage(client, group, taxonomy):
    base = _baseline(client, group)
    _seed_members(group, taxonomy)
    _seed_stats(group, taxonomy, period_key="2026-W36", ret_1m=0.02)

    body = client.get(f"/api/industries/{group.code}/companies").json()
    assert body["count"] == base["count"] + len(TICKERS)
    # The statistics row seeded here prices two of ITS three names and has
    # never heard of anything else classified into the group, so every
    # other member is unpriced with a reason — which is the point.
    assert body["n_priced"] == 2
    assert body["membership_source"].startswith("company_industry_classifications")
    assert body["membership_states"]["mapped"] == base["states"].get("mapped", 0) + len(TICKERS)

    rows = {r["ticker"]: r for r in body["items"]}
    assert rows[TICKERS[0]]["priced"] is True
    assert rows[TICKERS[0]]["last_close"] == 10.0
    unpriced = rows[TICKERS[2]]
    assert unpriced["priced"] is False
    assert unpriced["unpriced_reason"] == "not in the latest statistics row"
    assert unpriced["last_close"] is None  # never a zero
    assert {e["ticker"] for e in body["excluded"]} == {TICKERS[2]}


def test_companies_counters_describe_the_membership_not_the_page(client, group, taxonomy):
    """`count` and `n_priced` are one coverage figure; paging must not move it.

    Counting `n_priced` and `membership_states` inside the `rows[:limit]`
    loop made a capped call report `0 of 4 priced` for a group whose full
    page said `2 of 4` — two numbers in one object describing different
    populations. The whole-membership counters are what a reader is told
    the coverage is, so they are what a truncated call must still report.
    """
    base = _baseline(client, group)
    _seed_members(group, taxonomy)
    _seed_stats(group, taxonomy, period_key="2026-W36", ret_1m=0.02)

    full = client.get(f"/api/industries/{group.code}/companies").json()
    capped = client.get(f"/api/industries/{group.code}/companies?limit=1").json()

    assert capped["count"] == full["count"] == base["count"] + len(TICKERS)
    assert capped["n_priced"] == full["n_priced"] == 2
    assert capped["membership_states"] == full["membership_states"]
    # Only the page and its drop-count move.
    assert len(capped["items"]) == 1
    assert capped["truncated"] == capped["count"] - 1
    assert full["truncated"] == 0
    assert capped["counts_basis"]


def test_companies_carries_the_sub_industry_layer_and_its_provenance(client, group, taxonomy):
    seeded = _seed_members(group, taxonomy)
    body = client.get(f"/api/industries/{group.code}/companies").json()
    row = next(r for r in body["items"] if r["ticker"] == TICKERS[0])
    if seeded["sub"] is not None:
        assert row["sub_industry_code"] == seeded["sub"].code
        assert row["sub_industry_name"] == seeded["sub"].name
    assert row["classification"]["source"] == "research_map"
    assert "research map" in row["classification"]["source_label"]
    assert row["classification"]["as_of"] == "2026-09-08"
    assert row["classification"]["mapping_caveat"]
    assert "not official licensed issuer GICS mapping" in body["security_reference_caveat"]


def test_companies_without_a_statistics_row_says_why(client, group, taxonomy):
    base = _baseline(client, group)
    _seed_members(group, taxonomy)
    body = client.get(f"/api/industries/{group.code}/companies").json()
    assert body["stats"] is None
    assert "no statistics row" in body["stats_unavailable_reason"]
    assert body["n_priced"] == 0 and body["count"] == base["count"] + len(TICKERS)
    assert all(r["unpriced_reason"] for r in body["items"])


def test_companies_is_three_queries(client, group, taxonomy):
    """The active version, the membership join, the statistics read — and
    nothing per constituent. Counted with the caches warm, because a cold
    node cache costs one extra (immutable, cached per version id) read."""
    _seed_members(group, taxonomy)
    _seed_stats(group, taxonomy, period_key="2026-W36", ret_1m=0.02)
    path = f"/api/industries/{group.code}/companies"
    assert client.get(path).status_code == 200  # warm the node cache

    seen: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", _record)
    try:
        assert client.get(path).status_code == 200
    finally:
        event.remove(engine, "before_cursor_execute", _record)
    selects = [s for s in seen if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) <= 3, "\n\n".join(selects)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_snapshot_404_when_none_has_been_computed(client, taxonomy):
    with SessionLocal() as db:
        from app.models import CrossIndustrySnapshot
        db.execute(delete(CrossIndustrySnapshot).where(
            CrossIndustrySnapshot.taxonomy_version_id == taxonomy.id))
        db.commit()
    resp = client.get("/api/industries/snapshot")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "no_snapshot"


def test_snapshot_serves_the_stored_row(client, taxonomy):
    from app.models import CrossIndustrySnapshot
    with SessionLocal() as db:
        db.add(CrossIndustrySnapshot(
            taxonomy_version_id=taxonomy.id, period_key="2026-W36", as_of=AS_OF,
            schema_version=1, payload={"groups": [], "missing_groups": ["4530"]},
            stats_ids=[1], report_versions={"4530": 2}, computed_at=AS_OF,
        ))
        db.commit()
    try:
        body = client.get("/api/industries/snapshot").json()
        assert body["period_key"] == "2026-W36"
        assert body["payload"]["missing_groups"] == ["4530"]
        assert body["report_versions"] == {"4530": 2}
        assert body["access"]["surface"] == "pm_chat"
    finally:
        with SessionLocal() as db:
            db.execute(delete(CrossIndustrySnapshot).where(
                CrossIndustrySnapshot.taxonomy_version_id == taxonomy.id))
            db.commit()


# ---------------------------------------------------------------------------
# Reads never write
# ---------------------------------------------------------------------------


def test_no_read_enqueues_a_job_or_writes_a_report(client, group, taxonomy):
    _seed_members(group, taxonomy)
    stats_id = _seed_stats(group, taxonomy, period_key="2026-W36", ret_1m=0.02)
    _seed_report(group, taxonomy, period_key="2026-W36", stats_id=stats_id)

    def _counts() -> tuple[int, int]:
        with SessionLocal() as db:
            return (
                db.query(IndustryReportJob).count(),
                db.query(IndustryReport).count(),
            )

    before = _counts()
    for path in ("taxonomy", f"{group.code}/report", f"{group.code}/companies",
                 f"{group.code}/history", "snapshot"):
        client.get(f"/api/industries/{path}")
    assert _counts() == before


# ---------------------------------------------------------------------------
# Admin surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def admin_token(monkeypatch):
    token = "admin-token-industries-tests"
    monkeypatch.setattr(settings, "admin_api_token", token)
    return {"Authorization": f"Bearer {token}"}


ADMIN_CALLS = [
    ("POST", "/api/admin/industries/taxonomy/import", {}),
    ("POST", "/api/admin/industries/classify", {"tickers": []}),
    ("POST", "/api/admin/industries/reports/regenerate", {"codes": []}),
    ("GET", "/api/admin/industries/jobs", None),
    ("POST", "/api/admin/industries/jobs/recover-legacy", {}),
]


@pytest.mark.parametrize("method,path,body", ADMIN_CALLS)
def test_admin_routes_refuse_without_the_token(client, admin_token, method, path, body):
    from app.api import admin_auth
    assert admin_auth.is_protected(method, path)
    assert not admin_auth.is_exempt(method, path)
    resp = client.request(method, path, json=body)
    assert resp.status_code == 401, resp.text


def test_admin_import_is_idempotent_by_checksum(client, admin_token, taxonomy):
    body = client.post("/api/admin/industries/taxonomy/import", json={}, headers=admin_token).json()
    assert body["version_key"] == taxonomy.version_key
    assert body["imported"] is False and body["nodes_inserted"] == 0
    assert body["node_counts"]["industry_group"] == reg.counts(version=taxonomy)["industry_group"]
    assert body["drift"] is None


def test_admin_import_refuses_a_changed_structure_under_the_same_key(client, admin_token, monkeypatch):
    monkeypatch.setattr(reg, "checksum_for", lambda nodes: "0" * 64)
    resp = client.post("/api/admin/industries/taxonomy/import", json={}, headers=admin_token)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "taxonomy_checksum_mismatch"


def test_admin_classify_is_bulk_and_counts_truncation(client, admin_token, group, taxonomy):
    _seed_members(group, taxonomy)
    body = client.post("/api/admin/industries/classify",
                       json={"tickers": list(TICKERS)}, headers=admin_token).json()
    assert body["taxonomy_version"] == taxonomy.version_key
    assert body["classified"] == len(TICKERS)
    assert sum(body["counts"].values()) == len(TICKERS)
    assert body["changed_truncated"] == 0 and body["changed_total"] == len(body["changed"])
    assert body["mapping_caveat"]


def test_admin_regenerate_reports_an_unavailable_queue_rather_than_pretending(client, admin_token, group):
    from app.api import routes_industries_admin as admin_routes
    monkey = admin_routes._worker
    resp = client.post("/api/admin/industries/reports/regenerate",
                       json={"codes": [group.code]}, headers=admin_token)
    if monkey() is None:
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["code"] == "report_queue_unavailable"
        assert detail["requested"] == 1 and detail["period_key"]
    else:  # the drainer has landed — the same call must queue, not 503
        assert resp.status_code == 202, resp.text
        assert resp.json()["period_key"]


def test_admin_regenerate_speaks_the_queue_s_real_contract(client, admin_token, group):
    """Cross-slice contract, pinned because it already broke once.

    `industry_report_worker.enqueue_period` returns COUNTS under
    `enqueued`/`coalesced`/`skipped_published` and the code lists under the
    matching `_codes` keys. The admin route was written against an imagined
    contract where the count keys held lists, so `list(3)` raised and the
    endpoint 500'd the first time it met the real queue — which only
    happened after both slices merged, because each was tested against its
    own idea of the other.

    This asserts the two halves agree on the actual keys, in both
    directions, so neither side can drift alone.
    """
    from app.services import industry_report_worker as jobs

    resp = client.post("/api/admin/industries/reports/regenerate",
                       json={"codes": [group.code]}, headers=admin_token)
    assert resp.status_code == 202, resp.text
    body = resp.json()

    # The endpoint reports the group it moved, not a bare count.
    moved = [row["code"] for row in body["enqueued"] + body["coalesced"]]
    assert moved == [group.code], body

    # And the producer's own shape is what the consumer assumed: counts are
    # integers, `_codes` are lists, and they describe the same thing.
    result = jobs.enqueue_period(body["period_key"], codes=[group.code], source="test")
    for count_key, codes_key in (("enqueued", "enqueued_codes"),
                                 ("coalesced", "coalesced_codes"),
                                 ("skipped_published", "skipped_published_codes")):
        assert isinstance(result[count_key], int), count_key
        assert isinstance(result[codes_key], list), codes_key
        assert result[count_key] == len(result[codes_key]), (count_key, codes_key)


def test_admin_regenerate_refuses_a_drifted_taxonomy_unless_told_otherwise(client, admin_token, group, monkeypatch):
    """Queueing a week against a node set the deploy has already replaced
    publishes a stale structure as this week's edition, so it is a 409
    rather than a silent success. Overridable on purpose: re-running the
    structure that IS active is a legitimate thing to want."""
    from app.api import routes_industries_admin as admin_routes

    drift = {
        "kind": "same_key_changed_structure",
        "active_version_key": "gics-2026-04",
        "remedy": "import under a new --version-key",
    }
    monkeypatch.setattr(admin_routes.gics_registry, "bundled_drift", lambda *a, **k: drift)

    refused = client.post("/api/admin/industries/reports/regenerate",
                          json={"codes": [group.code]}, headers=admin_token)
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]
    assert detail["code"] == "taxonomy_drift"
    assert detail["drift"]["remedy"], "the refusal has to say how to resolve it"

    accepted = client.post("/api/admin/industries/reports/regenerate",
                           json={"codes": [group.code], "accept_stale_taxonomy": True},
                           headers=admin_token)
    assert accepted.status_code == 202, accepted.text


def test_admin_regenerate_rejects_an_unknown_code(client, admin_token):
    resp = client.post("/api/admin/industries/reports/regenerate",
                       json={"codes": ["9999"]}, headers=admin_token)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "unknown_industry_group"


def test_admin_jobs_counts_the_whole_queue_not_the_page(client, admin_token, group, taxonomy):
    with SessionLocal() as db:
        for i in range(3):
            db.add(IndustryReportJob(
                kind="group_report", taxonomy_version_id=taxonomy.id,
                industry_group_code=group.code, period_key="2026-W36", run_id=f"r{i}",
                status="queued" if i else "failed", attempts=i, max_attempts=3,
                enqueued_at=NOW, source="weekly_cron",
                owner_token="private-industry-claim" if i == 2 else None,
                lease_expires_at=NOW + timedelta(seconds=120) if i == 2 else None,
            ))
        db.commit()
    body = client.get(f"/api/admin/industries/jobs?code={group.code}&limit=1",
                      headers=admin_token).json()
    assert body["count"] == 1 and body["truncated"] == 2
    assert body["status_counts"] == {"queued": 2, "failed": 1}
    assert body["jobs"][0]["code"] == group.code
    assert body["jobs"][0]["ownership_tracked"] is True
    assert body["jobs"][0]["lease_expires_at"] == (NOW + timedelta(seconds=120)).isoformat()
    assert "owner_token" not in body["jobs"][0]
    assert "private-industry-claim" not in str(body)
    assert "drainer" in body and "enabled" in body["drainer"]


def _legacy_recovery_payload():
    return {
        "expected_jobs": [{
            "id": 42, "run_id": "legacy-run", "attempts": 1,
            "started_at": "2026-09-13T06:00:00", "heartbeat_at": None,
        }],
        "retirement_evidence": "Render predecessor retirement verified from shutdown and replacement evidence.",
    }


def test_legacy_recovery_route_preserves_expected_state_and_all_results(
    client, admin_token, monkeypatch,
):
    from app.services import industry_legacy_recovery

    seen = []
    expected = {
        "requested": 1, "recovered": 0,
        "results": [{"id": 42, "action": "rejected", "reason": "row_changed"}],
        "retirement_evidence": _legacy_recovery_payload()["retirement_evidence"],
    }

    def recover(rows, evidence):
        seen.append((rows, evidence))
        return expected

    monkeypatch.setattr(industry_legacy_recovery, "recover_legacy_jobs", recover)
    response = client.post(
        "/api/admin/industries/jobs/recover-legacy",
        json=_legacy_recovery_payload(), headers=admin_token,
    )
    assert response.status_code == 200 and response.json() == expected
    rows, evidence = seen[0]
    assert rows == [{"id": 42, "run_id": "legacy-run", "attempts": 1,
                     "started_at": datetime(2026, 9, 13, 6), "heartbeat_at": None}]
    assert evidence == expected["retirement_evidence"]


@pytest.mark.parametrize("invalid", [
    "empty_jobs", "too_many_jobs", "missing_timestamp", "empty_evidence",
    "unknown_expected_field", "boolean_id", "numeric_timestamp",
])
def test_legacy_recovery_invalid_request_never_reaches_service(
    client, admin_token, monkeypatch, invalid,
):
    from app.services import industry_legacy_recovery

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid recovery input must not reach the mutation service")

    monkeypatch.setattr(industry_legacy_recovery, "recover_legacy_jobs", forbidden)
    body = _legacy_recovery_payload()
    if invalid == "empty_jobs":
        body["expected_jobs"] = []
    elif invalid == "too_many_jobs":
        body["expected_jobs"] = [dict(body["expected_jobs"][0], id=n + 1) for n in range(51)]
    elif invalid == "missing_timestamp":
        del body["expected_jobs"][0]["heartbeat_at"]
    elif invalid == "empty_evidence":
        body["retirement_evidence"] = " " * 30
    elif invalid == "unknown_expected_field":
        body["expected_jobs"][0]["ignored_owner"] = "must-not-be-ignored"
    elif invalid == "numeric_timestamp":
        body["expected_jobs"][0]["started_at"] = 1789280000
    else:
        body["expected_jobs"][0]["id"] = True
    response = client.post(
        "/api/admin/industries/jobs/recover-legacy", json=body, headers=admin_token,
    )
    assert response.status_code == 422


def test_legacy_recovery_service_refusal_is_structured(client, admin_token, monkeypatch):
    from app.services import industry_legacy_recovery

    def refuse(*args, **kwargs):
        raise ValueError("Duplicate job identities")

    monkeypatch.setattr(industry_legacy_recovery, "recover_legacy_jobs", refuse)
    response = client.post(
        "/api/admin/industries/jobs/recover-legacy",
        json=_legacy_recovery_payload(), headers=admin_token,
    )
    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "invalid_legacy_recovery", "message": "Duplicate job identities",
    }


def test_default_period_key_follows_the_configured_as_of_weekday():
    from app.api.routes_industries_admin import default_period_key
    # Sunday 2026-09-06 with as-of weekday Friday → the week of 2026-09-04.
    assert default_period_key(datetime(2026, 9, 6, 6, 30)) == "2026-W36"
    assert default_period_key(datetime(2026, 9, 4, 23, 0)) == "2026-W36"
