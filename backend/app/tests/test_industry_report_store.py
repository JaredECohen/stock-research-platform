"""FEAT-003 slice 3 — the versioned Industry Analysis report store.

What can rot here: two latest-good editions for one group, a new edition
that forgets its parent, a review-mode save that publishes anyway, a diff
that invents a delta from a missing fact, and a stale flag that stays
green after a failed refresh. Each has a test.
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
    v1 = rs.save_report(code=code, period_key="2026-W35", as_of=AS_OF - timedelta(days=7), payload=_payload("first"),
                        version=_taxonomy, degraded=["analyst_narrative:llm_unavailable"])
    assert (v1.version, v1.parent_report_id, v1.is_latest_good, v1.status) == (1, None, True, "succeeded")
    v2 = rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("second", "moved"), version=_taxonomy,
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
    assert hist[1]["degraded"] == ["analyst_narrative:llm_unavailable"]
    assert rs.latest_versions(version=_taxonomy)[code] == 2


def test_review_mode_saves_pending_without_flipping_the_latest_good(code, _taxonomy, monkeypatch):
    v1 = rs.save_report(code=code, period_key="2026-W35", as_of=AS_OF, payload=_payload("first"), version=_taxonomy)
    monkeypatch.setattr(settings, "industry_reports_require_review", True)
    v2 = rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("second"), version=_taxonomy)
    assert v2.status == "pending_review" and not v2.is_latest_good and v2.parent_report_id == v1.id
    assert _latest_flags(code, _taxonomy.id) == [(1, "succeeded", True), (2, "pending_review", False)]
    assert rs.latest_good(code, version=_taxonomy)["version"] == 1
    # An explicit succeeded status is the reviewed path and does publish.
    v3 = rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("third"), version=_taxonomy,
                        status="succeeded")
    assert v3.is_latest_good and v3.parent_report_id == v1.id
    assert rs.latest_good(code, version=_taxonomy)["version"] == 3


def test_unknown_group_or_status_is_refused_before_any_write(_taxonomy):
    with pytest.raises(reg.UnknownNode):
        rs.save_report(code="0000", period_key="2026-W36", as_of=AS_OF, payload={}, version=_taxonomy)
    group = reg.industry_groups(version=_taxonomy)[0].code
    with pytest.raises(ValueError):
        rs.save_report(code=group, period_key="2026-W36", as_of=AS_OF, payload={}, version=_taxonomy, status="superseded")


def test_latest_good_is_none_for_an_unpublished_group_and_without_a_taxonomy(code, _taxonomy, monkeypatch):
    assert rs.latest_good(code, version=_taxonomy) is None
    monkeypatch.setattr(reg, "resolve_version", lambda v=None: (_ for _ in ()).throw(reg.TaxonomyNotImported("x")))
    assert rs.latest_good(code) is None


# --- diff --------------------------------------------------------------------------


def test_diff_between_non_adjacent_versions_is_arithmetic_over_stored_facts(code, _taxonomy):
    s1 = _stats(code, _taxonomy.id, "2026-W34", ret_1m=0.02, tickers=("AAA", "BBB"))
    s2 = _stats(code, _taxonomy.id, "2026-W35", ret_1m=0.05, tickers=("AAA", "BBB"))
    s3 = _stats(code, _taxonomy.id, "2026-W36", ret_1m=None, tickers=("BBB", "CCC"))
    rs.save_report(code=code, period_key="2026-W34", as_of=AS_OF - timedelta(days=14), payload=_payload("v1"), stats_id=s1, version=_taxonomy)
    rs.save_report(code=code, period_key="2026-W35", as_of=AS_OF - timedelta(days=7), payload=_payload("v2"), stats_id=s2, version=_taxonomy)
    rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("v3", "CCC joined"), stats_id=s3, version=_taxonomy)

    out = rs.diff(code, 1, 2, version=_taxonomy)
    assert out["adjacent"] is True
    assert out["facts_delta"]["returns.1m.ew"] == {"from": 0.02, "to": 0.05, "delta": pytest.approx(0.03)}
    assert out["facts_delta"]["returns.1m.mcw"]["delta"] is None
    assert out["facts_delta"]["returns.1m.mcw"]["reason"] == "missing in from-edition"
    assert out["constituents"] == {"added": [], "removed": [], "n_from": 2, "n_to": 2}

    out = rs.diff(code, 1, "latest", version=_taxonomy)
    assert out["adjacent"] is False and out["to"]["version"] == 3
    delta = out["facts_delta"]["returns.1m.ew"]
    assert delta["from"] == 0.02 and delta["to"] is None and delta["delta"] is None
    assert delta["reason"] == "missing in to-edition"
    assert out["constituents"] == {"added": ["CCC"], "removed": ["AAA"], "n_from": 2, "n_to": 2}
    assert out["analyst_view"] == {"from": "v1", "to": "v3", "what_changed": "CCC joined"}
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
    assert rs.freshness(code, version=_taxonomy) == {"stale": None, "stale_reason": "no_report", "last_attempt": None}
    rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, payload=_payload("v1"), version=_taxonomy)
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


def test_latest_good_many_reads_every_group_in_one_query(code, _taxonomy):
    """The chat tool and the PM block ask about a portfolio's groups at
    once; one SELECT, and a group without an edition is absent from the
    map rather than present with an empty report."""
    from sqlalchemy import event

    other = next(g.code for g in reg.industry_groups(version=_taxonomy) if g.code != code)
    rs.save_report(code=code, period_key="2026-W36", as_of=AS_OF, version=_taxonomy, payload={"sections": {}})

    selects: list[str] = []

    def _count(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT") and "industry_reports" in statement:
            selects.append(statement)

    engine = SessionLocal.kw["bind"]
    event.listen(engine, "before_cursor_execute", _count)
    try:
        out = rs.latest_good_many([code, other, "9999"], version=_taxonomy)
    finally:
        event.remove(engine, "before_cursor_execute", _count)
    assert set(out) == {code}
    assert out[code]["version"] == 1 and out[code]["is_latest_good"] is True
    assert len(selects) == 1, selects
    assert rs.latest_good_many([], version=_taxonomy) == {}


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
