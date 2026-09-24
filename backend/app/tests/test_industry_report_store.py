"""FEAT-003 slice 3 — the versioned Industry Analysis report store.

What can rot here: two latest-good editions for one group, a new edition
that forgets its parent, a review-mode save that publishes anyway, a diff
that invents a delta from a missing fact, and a stale flag that stays
green after a failed refresh. Each has a test.

Owner decision 1 (2026-09-24) adds the display rule: a template edition
(or an analyst edition too thin to stand) is stored audit-only and no
reader ever serves it — including legacy rows the pre-rule code flagged
latest-good, read correctly with no backfill.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.database import SessionLocal
from app.models import IndustryReport, IndustryReportJob, IndustryStatSnapshot
from app.services import gics_registry as reg
from app.services import industry_report_store as rs

AS_OF = datetime(2026, 9, 4, 21, 0)


@pytest.fixture(scope="module", autouse=True)
def _taxonomy():
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    yield info
    reg.activate_version(info.version_key)


@pytest.fixture()
def code(_taxonomy):
    """A real active group, wiped of editions, jobs and stats before and after."""
    group = reg.industry_groups(version=_taxonomy)[-1].code

    def _wipe() -> None:
        with SessionLocal() as db:
            db.execute(delete(IndustryReportJob).where(
                IndustryReportJob.taxonomy_version_id == _taxonomy.id, IndustryReportJob.industry_group_code == group))
            db.execute(delete(IndustryReport).where(
                IndustryReport.taxonomy_version_id == _taxonomy.id, IndustryReport.industry_group_code == group))
            db.execute(delete(IndustryStatSnapshot).where(
                IndustryStatSnapshot.taxonomy_version_id == _taxonomy.id, IndustryStatSnapshot.industry_group_code == group))
            db.commit()

    _wipe()
    yield group
    _wipe()


def _stats(code: str, version_id: int, period_key: str, *, ret_1m: float | None, tickers: tuple[str, ...]) -> int:
    with SessionLocal() as db:
        row = IndustryStatSnapshot(
            taxonomy_version_id=version_id, industry_group_code=code, period_key=period_key, as_of=AS_OF,
            method={}, sample={"n_with_prices": len(tickers), "n_constituents": len(tickers)},
            payload={
                "returns": {"1m": {"equal_weight": ret_1m, "market_cap_weight": None}},
                "breadth": {"1m": {"pct_positive": 0.5}},
                "valuation": {"ev_ebitda": {"median": 10.0}},
                "leaders": [{"ticker": tickers[0]}], "laggards": [{"ticker": tickers[-1]}],
            },
            per_ticker={t: {"last_close": 1.0} for t in tickers},
            inputs_hash=f"h-{period_key}-{ret_1m}",
        )
        db.add(row)
        db.commit()
        return row.id


def _payload(view: str, changed: str = "") -> dict:
    return {"sections": {
        "outlook": {"interpretation": {"analyst_view": view}},
        "what_changed": {"interpretation": {"text": changed}},
    }}


AGENTIC = {"generation_mode": "llm"}
TEMPLATE = {"generation_mode": "deterministic"}


def _save(**kw):
    """`save_report` for an ANALYST-written edition — the only kind the
    store publishes (owner decision 1). `generation_mode` is merged into
    whatever generation the test passes."""
    kw["generation"] = {**AGENTIC, **(kw.get("generation") or {})}
    return rs.save_report(**kw)


def _thin_markers(*sections: str) -> list[str]:
    """The writer's own markers for sections a template filled."""
    return [f"analyst_narrative:{s}:deterministic" for s in sections]


def _insert_legacy(code: str, version_id: int, *, version: int, status: str, latest: bool,
                   generation: dict, degraded: list[str] | None = None, period_key: str = "2026-W36",
                   payload: dict | None = None, parent_id: int | None = None) -> int:
    """A row exactly as the pre-rule code left it — written directly, so
    the store's rule is not what put it there."""
    with SessionLocal() as db:
        row = IndustryReport(
            taxonomy_version_id=version_id, industry_group_code=code, version=version,
            parent_report_id=parent_id, period_key=period_key, as_of=AS_OF, status=status,
            is_latest_good=latest, payload=payload or _payload(f"legacy v{version}"),
            generation=generation, degraded=list(degraded or []), generated_at=AS_OF,
        )
        db.add(row)
        db.commit()
        return row.id


def _latest_flags(code: str, version_id: int) -> list[tuple[int, str, bool]]:
    with SessionLocal() as db:
        rows = db.execute(
            select(IndustryReport.version, IndustryReport.status, IndustryReport.is_latest_good).where(
                IndustryReport.taxonomy_version_id == version_id, IndustryReport.industry_group_code == code,
            ).order_by(IndustryReport.version)
        ).all()
        return [(int(v), str(s), bool(flag)) for v, s, flag in rows]


# --- save / flip --------------------------------------------------------------


def test_save_assigns_versions_links_the_parent_and_keeps_one_latest_good(code, _taxonomy):
    v1 = _save(code=code, period_key="2026-W35", as_of=AS_OF - timedelta(days=7), payload=_payload("first"),
               version=_taxonomy, degraded=["cross_industry:snapshot:none_yet"])
    assert (v1.version, v1.parent_report_id, v1.is_latest_good, v1.status) == (1, None, True, "succeeded")
    v2 = _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("second", "moved"), version=_taxonomy,
               generation={"cost_usd": 0.01, "llm_calls": 2})
    assert v2.version == 2 and v2.parent_report_id == v1.id and v2.is_latest_good
    assert _latest_flags(code, _taxonomy.id) == [(1, "superseded", False), (2, "succeeded", True)]
    latest = rs.latest_good(code, version=_taxonomy)
    assert latest["version"] == 2 and latest["payload"]["sections"]["outlook"]["interpretation"]["analyst_view"] == "second"
    assert latest["disclaimer"] == rs.DISCLAIMER and latest["llm_cost_usd"] == 0.01
    assert rs.get(code, 1, version=_taxonomy)["status"] == "superseded"
    assert rs.get(code, "latest", version=_taxonomy)["version"] == 2
    assert rs.get(code, 9, version=_taxonomy) is None
    hist = rs.history(code, version=_taxonomy)
    assert [h["version"] for h in hist] == [2, 1] and "payload" not in hist[0]
    assert hist[1]["degraded"] == ["cross_industry:snapshot:none_yet"]
    assert rs.latest_versions(version=_taxonomy)[code] == 2


def test_review_mode_saves_pending_without_flipping_the_latest_good(code, _taxonomy, monkeypatch):
    v1 = _save(code=code, period_key="2026-W35", as_of=AS_OF, payload=_payload("first"), version=_taxonomy)
    monkeypatch.setattr(settings, "industry_reports_require_review", True)
    v2 = _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("second"), version=_taxonomy)
    assert v2.status == "pending_review" and not v2.is_latest_good and v2.parent_report_id == v1.id
    assert _latest_flags(code, _taxonomy.id) == [(1, "succeeded", True), (2, "pending_review", False)]
    assert rs.latest_good(code, version=_taxonomy)["version"] == 1
    # An explicit succeeded status is the reviewed path and does publish.
    v3 = _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("third"), version=_taxonomy,
               status="succeeded")
    assert v3.is_latest_good and v3.parent_report_id == v1.id
    assert rs.latest_good(code, version=_taxonomy)["version"] == 3


def test_unknown_group_or_status_is_refused_before_any_write(_taxonomy):
    with pytest.raises(reg.UnknownNode):
        rs.save_report(code="0000", period_key="2026-W36", as_of=AS_OF, payload={}, version=_taxonomy)
    group = reg.industry_groups(version=_taxonomy)[0].code
    with pytest.raises(ValueError):
        rs.save_report(code=group, period_key="2026-W36", as_of=AS_OF, payload={}, version=_taxonomy,
                       status="superseded", generation=AGENTIC)


def test_latest_good_is_none_for_an_unpublished_group_and_without_a_taxonomy(code, _taxonomy, monkeypatch):
    assert rs.latest_good(code, version=_taxonomy) is None
    monkeypatch.setattr(reg, "resolve_version", lambda v=None: (_ for _ in ()).throw(reg.TaxonomyNotImported("x")))
    assert rs.latest_good(code) is None


# --- diff --------------------------------------------------------------------------


def test_diff_between_non_adjacent_versions_is_arithmetic_over_stored_facts(code, _taxonomy):
    s1 = _stats(code, _taxonomy.id, "2026-W34", ret_1m=0.02, tickers=("AAA", "BBB"))
    s2 = _stats(code, _taxonomy.id, "2026-W35", ret_1m=0.05, tickers=("AAA", "BBB"))
    s3 = _stats(code, _taxonomy.id, "2026-W36", ret_1m=None, tickers=("BBB", "CCC"))
    _save(code=code, period_key="2026-W34", as_of=AS_OF - timedelta(days=14), payload=_payload("v1"), stats_id=s1, version=_taxonomy)
    _save(code=code, period_key="2026-W35", as_of=AS_OF - timedelta(days=7), payload=_payload("v2"), stats_id=s2, version=_taxonomy)
    _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("v3", "CCC joined"), stats_id=s3, version=_taxonomy)

    out = rs.diff(code, 1, 2, version=_taxonomy)
    assert out["adjacent"] is True
    assert out["facts_delta"]["returns.1m.ew"] == {"from": 0.02, "to": 0.05, "delta": pytest.approx(0.03)}
    assert out["facts_delta"]["returns.1m.mcw"]["delta"] is None
    # market_cap_weight is absent from BOTH editions in this fixture; the
    # old chain fell through and blamed the from-edition for it.
    assert out["facts_delta"]["returns.1m.mcw"]["reason"] == "missing in both editions"
    assert out["constituents"] == {"added": [], "removed": [], "n_from": 2, "n_to": 2}

    out = rs.diff(code, 1, "latest", version=_taxonomy)
    assert out["adjacent"] is False and out["to"]["version"] == 3
    delta = out["facts_delta"]["returns.1m.ew"]
    assert delta["from"] == 0.02 and delta["to"] is None and delta["delta"] is None
    assert delta["reason"] == "missing in to-edition"
    assert out["constituents"] == {"added": ["CCC"], "removed": ["AAA"], "n_from": 2, "n_to": 2}
    assert out["analyst_view"] == {"from": "v1", "to": "v3", "what_changed": "CCC joined",
                                   "reasons": {"from": None, "to": None, "what_changed": None}}
    assert out["leaders_laggards"]["to"]["leaders"] == [{"ticker": "BBB"}]
    with pytest.raises(rs.ReportNotFound):
        rs.diff(code, 1, 42, version=_taxonomy)


# --- last attempt / freshness --------------------------------------------------------


def _job(code: str, version_id: int, *, status: str, at: datetime, error: str = "") -> None:
    with SessionLocal() as db:
        db.add(IndustryReportJob(
            kind="group_report", taxonomy_version_id=version_id, industry_group_code=code, period_key="2026-W37",
            run_id="r", status=status, attempts=3, max_attempts=3, enqueued_at=at, started_at=at, finished_at=at,
            error_type=error, error_message="boom" if error else "",
        ))
        db.commit()


def test_freshness_flags_a_failed_refresh_and_an_old_as_of(code, _taxonomy, monkeypatch):
    monkeypatch.setattr(rs, "_utcnow", lambda: AS_OF + timedelta(days=2))
    assert rs.last_attempt(code, version=_taxonomy) is None
    assert rs.freshness(code, version=_taxonomy) == {"stale": None, "stale_reason": "no_report", "last_attempt": None,
                                                     "not_updated": None, "stale_codes": []}
    _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("v1"), version=_taxonomy)
    fresh = rs.freshness(code, version=_taxonomy)
    assert fresh["stale"] is False and fresh["stale_reason"] is None

    _job(code, _taxonomy.id, status="failed", at=AS_OF + timedelta(days=9), error="ValidationError")
    monkeypatch.setattr(rs, "_utcnow", lambda: AS_OF + timedelta(days=9, hours=1))
    attempt = rs.last_attempt(code, version=_taxonomy)
    assert attempt["status"] == "failed" and attempt["error_type"] == "ValidationError" and attempt["attempts"] == 3
    flagged = rs.freshness(code, version=_taxonomy)
    assert flagged["stale"] is True and "latest refresh attempt failed (ValidationError)" in flagged["stale_reason"]
    assert flagged["last_attempt"]["job_id"] == attempt["job_id"]
    # The edition itself still stands: failures never touch the report table.
    assert rs.latest_good(code, version=_taxonomy)["version"] == 1

    monkeypatch.setattr(rs, "_utcnow", lambda: AS_OF + timedelta(days=settings.industry_report_stale_after_days + 1))
    aged = rs.freshness(code, version=_taxonomy)
    assert aged["stale"] is True and f"older than {settings.industry_report_stale_after_days} days" in aged["stale_reason"]


def _count_report_selects():
    """A context manager that counts SELECTs against `industry_reports`."""
    import contextlib

    from sqlalchemy import event

    @contextlib.contextmanager
    def counting():
        selects: list[str] = []

        def _count(conn, cursor, statement, params, context, executemany):
            if statement.lstrip().upper().startswith("SELECT") and "industry_reports" in statement:
                selects.append(statement)

        engine = SessionLocal.kw["bind"]
        event.listen(engine, "before_cursor_execute", _count)
        try:
            yield selects
        finally:
            event.remove(engine, "before_cursor_execute", _count)

    return counting()


def test_latest_good_many_is_two_queries_whatever_the_group_count(code, _taxonomy):
    """The chat tool and the PM block ask about a portfolio's groups at
    once. The display rule is applied to the candidates' small metadata
    columns and then exactly the winners are loaded: two SELECTs for one
    group or for twenty, never one per group. A group without an edition is
    absent from the map rather than present with an empty report."""
    others = [g.code for g in reg.industry_groups(version=_taxonomy) if g.code != code][:3]
    _save(code=code, period_key="2026-W36", as_of=AS_OF, version=_taxonomy, payload={"sections": {}})

    with _count_report_selects() as one:
        out = rs.latest_good_many([code, "9999"], version=_taxonomy)
    with _count_report_selects() as many:
        wide = rs.latest_good_many([code, *others, "9999"], version=_taxonomy)
    assert set(out) == set(wide) == {code}
    assert out[code]["version"] == 1 and out[code]["is_latest_good"] is True
    assert len(one) == len(many) == 2, (one, many)
    assert rs.latest_good_many([], version=_taxonomy) == {}


# --- the display rule (owner decision 1, contract C8) ---------------------------


def test_the_store_and_the_validator_agree_on_the_interpreted_sections():
    """The store keeps its own copy (it must not import `app.agents` on the
    web path); this is the pin that keeps the copy honest."""
    from app.agents.industry_report_validator import INTERPRETED_SECTIONS

    assert rs.INTERPRETED_SECTIONS == INTERPRETED_SECTIONS
    assert set(rs.REQUIRED_MODEL_SECTIONS) <= set(INTERPRETED_SECTIONS)
    assert rs.MIN_MODEL_SECTIONS <= len(INTERPRETED_SECTIONS)


def test_template_edition_is_stored_audit_only_and_never_latest(code, _taxonomy):
    first = _save(code=code, period_key="2026-W35", as_of=AS_OF, payload=_payload("analyst"), version=_taxonomy)
    template = rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("template"),
                              version=_taxonomy, generation=TEMPLATE,
                              degraded=["analyst_narrative:deterministic_mode"])
    assert template.status == "audit_only" and template.is_latest_good is False
    # Linked to the edition a reader was looking at, and replacing nothing.
    assert template.parent_report_id == first.id
    assert _latest_flags(code, _taxonomy.id) == [(1, "succeeded", True), (2, "audit_only", False)]
    latest = rs.latest_good(code, version=_taxonomy)
    assert latest["version"] == 1 and latest["is_latest_good"] is True
    # `llm_unavailable` and a missing mode are templates too — fail closed.
    for generation in ({"generation_mode": "llm_unavailable"}, {}):
        row = rs.save_report(code=code, period_key="2026-W37", as_of=AS_OF, payload=_payload("t"),
                             version=_taxonomy, generation=generation)
        assert row.status == "audit_only"
    assert rs.latest_good(code, version=_taxonomy)["version"] == 1
    assert rs.latest_versions(version=_taxonomy)[code] == 1


def test_explicit_status_cannot_publish_a_template(code, _taxonomy):
    for status in ("succeeded", "pending_review", "audit_only"):
        with pytest.raises(ValueError, match="audit only"):
            rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("t"),
                           version=_taxonomy, generation=TEMPLATE, status=status)
    assert _latest_flags(code, _taxonomy.id) == [], "a refused save must write nothing"


def test_thin_agentic_edition_is_audit_only(code, _taxonomy):
    """An analyst edition in which the model did not write the causal chain
    (drivers) or the forward view (outlook), or wrote fewer than
    `MIN_MODEL_SECTIONS` sections, is a template with some model prose on
    top — stored for audit, not published as the analyst's."""
    no_drivers = rs.save_report(code=code, period_key="2026-W35", as_of=AS_OF, payload=_payload("x"),
                                version=_taxonomy, generation=AGENTIC, degraded=_thin_markers("drivers"))
    assert no_drivers.status == "audit_only"
    templated = [s for s in rs.INTERPRETED_SECTIONS if s not in rs.REQUIRED_MODEL_SECTIONS]
    too_few = len(rs.INTERPRETED_SECTIONS) - rs.MIN_MODEL_SECTIONS + 1
    thin = rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("x"),
                          version=_taxonomy, generation=AGENTIC, degraded=_thin_markers(*templated[:too_few]))
    assert thin.status == "audit_only"
    # One template section fewer is exactly the minimum, and publishes.
    enough = _save(code=code, period_key="2026-W37", as_of=AS_OF, payload=_payload("x"), version=_taxonomy,
                   degraded=_thin_markers(*templated[:too_few - 1]))
    assert enough.status == "succeeded" and enough.is_latest_good
    # The payload's own per-section record is read too: a section it calls
    # a template is one, even when the row's markers missed it.
    by_payload = rs.save_report(
        code=code, period_key="2026-W38", as_of=AS_OF, version=_taxonomy, generation=AGENTIC,
        payload={**_payload("x"), "narrative_by_section": {"outlook": "deterministic"}},
    )
    assert by_payload.status == "audit_only"
    assert rs.latest_good(code, version=_taxonomy)["version"] == enough.version
    assert "not model-written: drivers" in rs.withheld_reason(AGENTIC, _thin_markers("drivers"))


def test_latest_good_ignores_legacy_flags(code, _taxonomy):
    """Rows the pre-rule code wrote: the analyst edition `superseded` by a
    template that took the latest-good flag. Every reader must find the
    analyst edition the moment the code deploys — before any backfill."""
    v1 = _insert_legacy(code, _taxonomy.id, version=1, status="superseded", latest=False, generation=AGENTIC,
                        period_key="2026-W35")
    _insert_legacy(code, _taxonomy.id, version=2, status="succeeded", latest=True, generation=TEMPLATE,
                   period_key="2026-W36", parent_id=v1)
    latest = rs.latest_good(code, version=_taxonomy)
    assert latest["id"] == v1 and latest["period_key"] == "2026-W35"
    assert set(rs.latest_good_many([code], version=_taxonomy)) == {code}
    assert rs.latest_good_many([code], version=_taxonomy)[code]["id"] == v1
    assert rs.latest_versions(version=_taxonomy)[code] == 1
    with pytest.raises(rs.EditionWithheld) as exc:
        rs.get(code, 2, version=_taxonomy)
    assert exc.value.version == 2 and "template edition" in exc.value.reason
    assert rs.get(code, 2, version=_taxonomy, include_withheld=True)["status"] == "audit_only"
    # A new analyst edition links to the analyst edition, not the template.
    v3 = _save(code=code, period_key="2026-W37", as_of=AS_OF, payload=_payload("new"), version=_taxonomy)
    assert v3.parent_report_id == v1


def test_public_flags_follow_publishable_rule_before_backfill(code, _taxonomy):
    """`status` and `is_latest_good` on every public projection are derived
    from the rule, not copied from the row: a legacy analyst edition marked
    `superseded` by a template reads as the published latest, and the
    template reads `audit_only` wherever an admin path can see it."""
    v1 = _insert_legacy(code, _taxonomy.id, version=1, status="superseded", latest=False, generation=AGENTIC,
                        period_key="2026-W35")
    _insert_legacy(code, _taxonomy.id, version=2, status="succeeded", latest=True, generation=TEMPLATE,
                   period_key="2026-W36", parent_id=v1)
    for edition in (rs.latest_good(code, version=_taxonomy), rs.get(code, 1, version=_taxonomy),
                    rs.history(code, version=_taxonomy)[0]):
        assert (edition["status"], edition["is_latest_good"]) == ("succeeded", True), edition
        assert edition["stored_status"] == "superseded"
    admin = rs.history(code, version=_taxonomy, include_withheld=True)
    assert [(h["version"], h["status"], h["is_latest_good"]) for h in admin] == [
        (2, "audit_only", False), (1, "succeeded", True)]


def test_query_and_python_edition_kind_agree_on_legacy_rows(code, _taxonomy):
    """The candidate filter runs in SQL (`agentic_clause`) and the rule
    runs in Python (`edition_kind`). If they ever disagreed, a row would be
    publishable to one reader and not another; every generation shape a
    legacy row can carry is pinned here."""
    shapes = [
        {"generation_mode": "llm"}, {"generation_mode": "deterministic"},
        {"generation_mode": "llm_unavailable"}, {"generation_mode": ""}, {"generation_mode": None},
        {"generation_mode": "LLM"}, {"generation_mode": ["llm"]}, {"generation_mode": {"mode": "llm"}},
        {}, {"model": "llm"}, None,
    ]
    ids = {}
    for i, generation in enumerate(shapes, start=1):
        ids[i] = _insert_legacy(code, _taxonomy.id, version=i, status="succeeded", latest=False,
                                generation=generation)
    with SessionLocal() as db:
        by_sql = set(db.execute(select(IndustryReport.id).where(
            IndustryReport.taxonomy_version_id == _taxonomy.id, IndustryReport.industry_group_code == code,
            rs.agentic_clause())).scalars())
        rows = db.execute(select(IndustryReport.id, IndustryReport.generation).where(
            IndustryReport.taxonomy_version_id == _taxonomy.id,
            IndustryReport.industry_group_code == code)).all()
    by_python = {rid for rid, generation in rows if rs.edition_kind(generation) == rs.EDITION_AGENTIC}
    assert by_sql == by_python == {ids[1]}


def test_history_excludes_withheld_by_default(code, _taxonomy):
    _save(code=code, period_key="2026-W35", as_of=AS_OF, payload=_payload("a"), version=_taxonomy)
    rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("t"), version=_taxonomy,
                   generation=TEMPLATE)
    _save(code=code, period_key="2026-W37", as_of=AS_OF, payload=_payload("b"), version=_taxonomy)
    page = rs.history_page(code, version=_taxonomy, limit=1)
    assert [h["version"] for h in page["items"]] == [3]
    assert page["total"] == 2 and page["withheld"] == 1
    assert [h["version"] for h in rs.history(code, version=_taxonomy)] == [3, 1]
    assert [h["version"] for h in rs.history(code, version=_taxonomy, include_withheld=True)] == [3, 2, 1]
    assert rs.withheld_count(code, version=_taxonomy) == 1


def _attempt(code: str, version_id: int, *, period_key: str, status: str, at: datetime,
             report_id: int | None = None, error_type: str = "", error_message: str = "") -> int:
    with SessionLocal() as db:
        job = IndustryReportJob(
            kind="group_report", taxonomy_version_id=version_id, industry_group_code=code, period_key=period_key,
            run_id="r", status=status, attempts=3, max_attempts=3, enqueued_at=at, started_at=at,
            finished_at=at if status in ("succeeded", "failed") else None, report_id=report_id,
            error_type=error_type, error_message=error_message,
        )
        db.add(job)
        db.commit()
        return job.id


def test_freshness_reports_not_updated_for_a_withheld_week(code, _taxonomy, monkeypatch):
    monkeypatch.setattr(rs, "_utcnow", lambda: AS_OF + timedelta(days=3))
    _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("a"), version=_taxonomy)
    template = rs.save_report(code=code, period_key="2026-W37", as_of=AS_OF + timedelta(days=7),
                              payload=_payload("t"), version=_taxonomy, generation=TEMPLATE)
    _attempt(code, _taxonomy.id, period_key="2026-W37", status="succeeded", at=AS_OF + timedelta(days=2),
             report_id=template.id)
    fresh = rs.freshness(code, version=_taxonomy)
    assert fresh["stale"] is True
    assert fresh["not_updated"] == {"period_key": "2026-W37", "outcome": "withheld_template"}
    assert "not updated this week: the 2026-W37 refresh produced no validated analyst edition" in fresh["stale_reason"]
    assert fresh["stale_codes"] == ["not_updated"]
    assert fresh["last_attempt"]["outcome"] == "withheld_template"
    # The public form never names the audit-only row.
    assert rs.public_attempt(fresh["last_attempt"])["report_id"] is None
    display = rs.display_block(rs.latest_good(code, version=_taxonomy), fresh)
    assert display["not_updated"] == {"period_key": "2026-W37", "outcome": "withheld_template"}


def test_freshness_reports_not_updated_for_a_failed_week(code, _taxonomy, monkeypatch):
    monkeypatch.setattr(rs, "_utcnow", lambda: AS_OF + timedelta(days=3))
    _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("a"), version=_taxonomy)
    _attempt(code, _taxonomy.id, period_key="2026-W37", status="failed", at=AS_OF + timedelta(days=2),
             error_type="RuntimeError", error_message="boom")
    fresh = rs.freshness(code, version=_taxonomy)
    assert fresh["stale"] is True and "latest refresh attempt failed (RuntimeError)" in fresh["stale_reason"]
    assert fresh["not_updated"] == {"period_key": "2026-W37", "outcome": "failed"}
    assert fresh["stale_codes"] == ["refresh_failed", "not_updated"]


def test_freshness_in_progress_is_not_stale(code, _taxonomy, monkeypatch):
    monkeypatch.setattr(rs, "_utcnow", lambda: AS_OF + timedelta(days=3))
    _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("a"), version=_taxonomy)
    _attempt(code, _taxonomy.id, period_key="2026-W37", status="running", at=AS_OF + timedelta(days=2))
    fresh = rs.freshness(code, version=_taxonomy)
    assert fresh["stale"] is False and fresh["stale_codes"] == []
    assert fresh["not_updated"] == {"period_key": "2026-W37", "outcome": "in_progress"}
    # Reported, but not news yet: the page shows no "not updated" banner.
    assert rs.display_block(rs.latest_good(code, version=_taxonomy), fresh)["not_updated"] is None


def test_an_older_version_opened_explicitly_is_never_marked_not_updated(code, _taxonomy, monkeypatch):
    """v1 (W35) was superseded by a PUBLISHED v2 (W36); W37 came out
    withheld. The banner belongs on v2 — on v1 it would say W35 was left
    standing because W37 failed, when W36 is on the site."""
    monkeypatch.setattr(rs, "_utcnow", lambda: AS_OF + timedelta(days=3))
    _save(code=code, period_key="2026-W35", as_of=AS_OF, payload=_payload("a"), version=_taxonomy)
    _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("b"), version=_taxonomy)
    template = rs.save_report(code=code, period_key="2026-W37", as_of=AS_OF, payload=_payload("t"),
                              version=_taxonomy, generation=TEMPLATE)
    _attempt(code, _taxonomy.id, period_key="2026-W37", status="succeeded", at=AS_OF + timedelta(days=2),
             report_id=template.id)
    old = rs.get(code, "1", version=_taxonomy)
    assert old["is_latest_good"] is False
    fresh = rs.freshness(code, version=_taxonomy, report=old)
    assert fresh["not_updated"] is None and "not_updated" not in fresh["stale_codes"]
    assert "not updated this week" not in (fresh["stale_reason"] or "")
    # The latest edition still carries it.
    assert rs.freshness(code, version=_taxonomy)["not_updated"] == {
        "period_key": "2026-W37", "outcome": "withheld_template"}


def test_a_queued_later_week_does_not_hide_a_withheld_one(code, _taxonomy, monkeypatch):
    """W37 finished withheld, W38 is queued (retry backoff): the page must
    still say W37 was not updated, as the picker pointer does."""
    monkeypatch.setattr(rs, "_utcnow", lambda: AS_OF + timedelta(days=3))
    _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("a"), version=_taxonomy)
    template = rs.save_report(code=code, period_key="2026-W37", as_of=AS_OF, payload=_payload("t"),
                              version=_taxonomy, generation=TEMPLATE)
    _attempt(code, _taxonomy.id, period_key="2026-W37", status="succeeded", at=AS_OF + timedelta(days=2),
             report_id=template.id)
    _attempt(code, _taxonomy.id, period_key="2026-W38", status="queued", at=AS_OF + timedelta(days=9))
    fresh = rs.freshness(code, version=_taxonomy)
    assert fresh["not_updated"] == {"period_key": "2026-W37", "outcome": "withheld_template"}
    assert "not_updated" in fresh["stale_codes"]
    assert rs.last_attempted_periods([code], version=_taxonomy) == {code: "2026-W37"}
    display = rs.display_block(rs.latest_good(code, version=_taxonomy), fresh)
    assert display["not_updated"] == {"period_key": "2026-W37", "outcome": "withheld_template"}


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_a_same_week_attempt_is_not_not_updated(code, _taxonomy, monkeypatch, status):
    """W37's analyst edition is up; an admin re-run of W37 then ended
    withheld (or failed). W37 WAS updated — the banner is only for a LATER
    week, never over the week's own analyst edition."""
    monkeypatch.setattr(rs, "_utcnow", lambda: AS_OF + timedelta(days=3))
    _save(code=code, period_key="2026-W37", as_of=AS_OF, payload=_payload("a"), version=_taxonomy)
    report_id = None
    if status == "succeeded":
        report_id = rs.save_report(code=code, period_key="2026-W37", as_of=AS_OF, payload=_payload("t"),
                                   version=_taxonomy, generation=TEMPLATE).id
    _attempt(code, _taxonomy.id, period_key="2026-W37", status=status, at=AS_OF + timedelta(days=2),
             report_id=report_id, error_type="" if report_id else "RuntimeError")
    fresh = rs.freshness(code, version=_taxonomy)
    assert fresh["not_updated"] is None
    assert "not_updated" not in fresh["stale_codes"]


def test_public_attempt_never_quotes_rejected_model_prose():
    raw = {"job_id": 1, "status": "queued", "outcome": "in_progress", "error_type": "ReportRejected",
           "error_message": "3 validation problem(s): drivers: unsupported causal claim 'The model said so'",
           "report_id": None}
    public = rs.public_attempt(raw)
    assert public["error_message"] == "the analyst draft did not pass validation (3 problems)"
    assert "model said" not in str(public)
    assert raw["error_message"].startswith("3 validation"), "the raw attempt must not be mutated"
    one = rs.public_attempt({**raw, "error_message": "1 validation problem(s): x"})
    assert one["error_message"] == "the analyst draft did not pass validation (1 problem)"
    other = {"error_type": "RuntimeError", "error_message": "boom", "outcome": "failed", "report_id": None}
    assert rs.public_attempt(other) == other
    assert rs.public_attempt(None) is None


def test_hidden_sections_are_nulled_in_the_response_only(code, _taxonomy):
    payload = {"sections": {
        "overview": {"facts": {"n": 1}, "interpretation": {"text": "model"}},
        "themes": {"facts": {"n": 2}, "interpretation": {"text": "template prose"}},
    }}
    row = _save(code=code, period_key="2026-W36", as_of=AS_OF, payload=payload, version=_taxonomy,
                degraded=_thin_markers("themes"))
    edition = rs.latest_good(code, version=_taxonomy)
    public = rs.public_payload(edition)
    assert public["sections"]["themes"] == {"facts": {"n": 2}, "interpretation": None}
    assert public["sections"]["overview"]["interpretation"] == {"text": "model"}
    assert rs.display_block(edition)["hidden_sections"] == ["themes"]
    # Stored payload untouched.
    with SessionLocal() as db:
        stored = db.get(IndustryReport, row.id).payload
    assert stored["sections"]["themes"]["interpretation"] == {"text": "template prose"}


def test_diff_refuses_a_withheld_edition(code, _taxonomy):
    _save(code=code, period_key="2026-W35", as_of=AS_OF, payload=_payload("a"), version=_taxonomy)
    rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("t"), version=_taxonomy,
                   generation=TEMPLATE)
    with pytest.raises(rs.EditionWithheld):
        rs.diff(code, 1, 2, version=_taxonomy)
    with pytest.raises(rs.EditionWithheld):
        rs.diff(code, 2, "latest", version=_taxonomy)


# --- access policy (owner decision 1) ------------------------------------------


def test_access_policy_reads_the_setting_and_keeps_the_tier_apart_from_enforcement(monkeypatch):
    """One place answers "what does this surface cost" so the routes, the
    UI and the chat tool cannot each invent a different answer. The tier
    is the policy; ``enforced`` is whether the login wall is on — a caller
    that conflated them would tell a user they paid for an open surface."""
    monkeypatch.setattr(settings, "industry_analysis_access", "public", raising=False)
    monkeypatch.setattr(settings, "auth_enabled", False, raising=False)
    policy = rs.access_policy()
    assert policy["surfaces"] == {"latest": "public", "history": "pro", "changes": "pro", "pm_chat": "pro"}
    assert policy["enforced"] is False and policy["setting"] == "public"
    assert rs.surface_tier("history") == "pro" and rs.surface_tier("latest") == "public"

    monkeypatch.setattr(settings, "auth_enabled", True, raising=False)
    assert rs.access_policy()["enforced"] is True

    # The setting moves `latest` only — the deeper reads stay Pro.
    monkeypatch.setattr(settings, "industry_analysis_access", "pro", raising=False)
    assert rs.access_policy()["surfaces"]["latest"] == "pro"


def test_an_unreadable_access_setting_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "industry_analysis_access", "everyone", raising=False)
    policy = rs.access_policy()
    assert policy["setting"] == "pro" and policy["surfaces"]["latest"] == "pro"
    assert rs.surface_tier("something_new") == "pro"
