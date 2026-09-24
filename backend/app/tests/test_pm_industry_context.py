"""FEAT-003 slice 3 — the PM's industry context and the chat tool.

What can rot here: the PM block appearing (or a DB read costing a memo)
when no snapshot exists, the block outgrowing its budget, the chat tool
computing something instead of reading stored artifacts, and a portfolio
question answered for one ticker only. Each has a test.

The chat tool is exercised the way ``test_chat_sdk_structure`` does it:
against a stand-in ``agents`` module whose ``function_tool`` is the
identity, so the closure stays a plain callable.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import delete

from app.agents import chat_sdk, pm_context
from app.config import settings
from app.database import SessionLocal
from app.models import CrossIndustrySnapshot, IndustryReport
from app.services import gics_registry as reg
from app.services import industry_analytics as ia
from app.services import industry_classification as ic
from app.services import industry_labels as il
from app.services import industry_report_store as rs
from app.services import industry_snapshot as isn
from app.tests.fixtures.demo_dataset import COMPANY_PROFILES
from app.tests.gating_helpers import seed_demo_universe

AS_OF = datetime(2026, 9, 4, 21, 0)
PERIOD = "2026-W36"
# Only analyst-written editions reach the PM (owner decision 1).
AGENTIC = {"generation_mode": "llm"}
TEMPLATE = {"generation_mode": "deterministic"}


@pytest.fixture(scope="module", autouse=True)
def _taxonomy():
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    ic.classify_all(tickers=sorted(COMPANY_PROFILES), version=info)
    yield info
    _wipe(info.id)
    reg.activate_version(info.version_key)


def _wipe(version_id: int) -> None:
    with SessionLocal() as db:
        db.execute(delete(CrossIndustrySnapshot).where(CrossIndustrySnapshot.taxonomy_version_id == version_id))
        db.execute(delete(IndustryReport).where(IndustryReport.taxonomy_version_id == version_id))
        db.commit()


@pytest.fixture()
def clean(_taxonomy):
    _wipe(_taxonomy.id)
    yield _taxonomy
    _wipe(_taxonomy.id)


def _stats_row(code: str, sid: int, ret_1m: float) -> dict[str, Any]:
    return {
        "id": sid, "code": code, "period_key": PERIOD, "as_of": AS_OF.isoformat(),
        "sample": {"n_with_prices": 4}, "per_ticker": {"NVDA": {}},
        "payload": {
            "status": "ok",
            "returns": {"1w": {"equal_weight": 0.01}, "1m": {"equal_weight": ret_1m}, "ytd": {"equal_weight": None, "reason": "history_window"}},
            "benchmark_relative": {"universe_ew": {"1m": {"value": ret_1m}}},
            "breadth": {"1m": {"pct_positive": 0.75}}, "dispersion": {"stdev": 0.03},
            "valuation": {"ev_ebitda": {"median": 14.0}}, "fundamental_momentum": {"value": None, "reason": "no_prior_period"},
        },
    }


def _snapshot(info, codes: list[str]) -> CrossIndustrySnapshot:
    loaders = isn.SnapshotLoaders(
        stats=lambda period_key, version: {c: _stats_row(c, i + 1, 0.04 * (i + 1)) for i, c in enumerate(codes)},
        macro=lambda: {"regime": "Soft landing", "favored_sectors": [], "pressured_sectors": [], "as_of": None},
        events=lambda tickers, cutoff, days: [],
        report_versions=lambda version: {},
    )
    return isn.compute_cross_snapshot(PERIOD, AS_OF, version=info, loaders=loaders)


def _own_group(ticker: str, info) -> str:
    return ic.current_for([ticker], version=info)[ticker]["industry_group_code"]


# --- PM context block ----------------------------------------------------------


def test_no_snapshot_means_no_block(clean):
    assert pm_context.industry_context_block(ticker="NVDA") == ""
    assert "Cross-industry snapshot" not in pm_context.build_pm_context(ticker="NVDA", sector="Technology")


def test_block_appears_with_a_snapshot_and_stays_within_budget(clean):
    nvda = _own_group("NVDA", clean)
    _snapshot(clean, [nvda] + [g.code for g in reg.industry_groups(version=clean) if g.code != nvda])
    block = pm_context.industry_context_block(ticker="NVDA")
    assert block.startswith("## Cross-industry snapshot")
    rendered = block.split("\n\n")[1]
    assert len(rendered) <= isn.PM_BLOCK_MAX_CHARS
    assert f"Industry group {il.label(nvda)}: no published Industry Analysis edition yet." in block
    assert "scenario input, not recommendations" in block
    ctx = pm_context.build_pm_context(ticker="NVDA", sector="Technology")
    assert "## Cross-industry snapshot" in ctx
    # A caller's smaller budget bounds the render, never the other way round.
    small = pm_context.industry_context_block(ticker="NVDA", max_chars=600)
    assert len(small.split("\n\n")[1]) <= 600
    assert len(pm_context.build_pm_context(ticker="NVDA", max_chars_each=600)) < 6000


def test_block_quotes_the_own_groups_edition_and_names_linked_and_unmapped(clean):
    nvda = _own_group("NVDA", clean)
    _snapshot(clean, [nvda])
    rs.save_report(
        code=nvda, period_key=PERIOD, as_of=AS_OF, version=clean, generation=AGENTIC,
        degraded=["cross_industry:snapshot:none_yet"],
        payload={"sections": {
            "outlook": {"interpretation": {"analyst_view": "Capacity additions " + "x" * 900}},
            "what_changed": {"interpretation": {"text": "Breadth narrowed."}},
        }},
    )
    block = pm_context.industry_context_block(ticker="NVDA", tickers=["ZZNOPE"])
    assert f"Industry group {il.label(nvda)} — edition of {PERIOD} (degraded edition: cross_industry:snapshot:none_yet)" in block
    assert "Analyst view: Capacity additions" in block and "What changed: Breadth narrowed." in block
    assert "No industry-group mapping for: ZZNOPE (unclassified)" in block
    if isn.linked_group_codes(nvda):
        assert "Linked groups (dependency graph, analyst hypotheses):" in block
    excerpt = pm_context._report_excerpt(nvda)
    assert len(excerpt["analyst_view"]) <= pm_context.INDUSTRY_EXCERPT_MAX_CHARS // 2
    assert excerpt["what_changed"] == "Breadth narrowed."


# --- payload / chat tool ------------------------------------------------------------


@pytest.fixture
def tools(monkeypatch) -> dict[str, Any]:
    fake = types.ModuleType("agents")
    fake.Agent = lambda **kwargs: types.SimpleNamespace(name=kwargs["name"], tools=kwargs["tools"])
    fake.function_tool = lambda fn: fn
    monkeypatch.setitem(sys.modules, "agents", fake)
    monkeypatch.setattr(chat_sdk, "_can_use_sdk", lambda: True)
    agent = chat_sdk._build_chat_agent()
    assert agent is not None
    return {t.__name__: t for t in agent.tools}


def test_chat_tool_answers_for_a_portfolio_of_tickers_from_stored_artifacts_only(clean, tools, monkeypatch):
    nvda, jpm = _own_group("NVDA", clean), _own_group("JPM", clean)
    _snapshot(clean, [nvda, jpm])
    rs.save_report(code=jpm, period_key=PERIOD, as_of=AS_OF, version=clean, generation=AGENTIC,
                   payload={"sections": {"outlook": {"interpretation": {"analyst_view": "Banks steady."}}}})

    def _boom(*a, **k):
        raise AssertionError("the chat tool must not compute or fetch")

    monkeypatch.setattr(ia, "compute_group_stats", _boom)
    monkeypatch.setattr(ia, "load_context", _boom)
    monkeypatch.setattr(isn, "compute_cross_snapshot", _boom)
    from app.services import data_service
    monkeypatch.setattr(data_service.DataService, "get_price_history", _boom)

    out = tools["get_industry_context"](tickers=["NVDA", "jpm", "ZZNOPE"])
    assert out["status"] == "ok" and out["taxonomy_version"] == il.public_version_key(clean.version_key)
    assert out["by_ticker"] == {"NVDA": {"code": il.slug(nvda), "state": "mapped"}, "JPM": {"code": il.slug(jpm), "state": "mapped"}}
    assert out["unmapped_tickers"] == [{"ticker": "ZZNOPE", "state": "unclassified"}]
    assert out["snapshot"]["period_key"] == PERIOD and out["snapshot"]["macro_regime"] == "Soft landing"
    by_code = {g["code"]: g for g in out["groups"]}
    assert by_code[il.slug(nvda)]["relation"] == "own" and by_code[il.slug(nvda)]["snapshot_row"]["ret_1m_ew"] == 0.04
    assert by_code[il.slug(nvda)]["report"] is None
    assert by_code[il.slug(jpm)]["report"]["analyst_view"] == "Banks steady." and by_code[il.slug(jpm)]["report"]["version"] == 1
    linked = [g for g in out["groups"] if g["relation"] == "linked"]
    # Linked groups had no stats this period: the snapshot names them as
    # such rather than carrying numbers for them.
    assert all(g["via"] and g["snapshot_row"]["status"] == "no_stats" for g in linked)
    assert "scenarios, not recommendations" in out["note"] and out["mapping_caveat"]

    explicit = tools["get_industry_context"](code=jpm)
    assert explicit["code"] == {"code": il.slug(jpm), "name": il.label(jpm)}
    assert [g["relation"] for g in explicit["groups"]] == ["requested"]
    # The answer names groups by slug, so the model's follow-up passes a
    # slug back: it must resolve to the same group.
    by_slug = tools["get_industry_context"](code=il.slug(jpm))
    assert by_slug["code"] == explicit["code"] and [g["code"] for g in by_slug["groups"]] == [il.slug(jpm)]
    bad = tools["get_industry_context"](tickers=[], code="9999")
    assert "no industry group" in bad["code"]["error"] and bad["groups"] == []
    assert "not found" in tools["get_industry_context"](code="no-such-group")["code"]["error"]
    # A sector's code is not a group, and the refusal does not quote it back.
    sector = tools["get_industry_context"](code=jpm[:2])
    assert sector["groups"] == [] and f"'{jpm[:2]}'" not in sector["code"]["error"]


def test_chat_tool_reports_missing_snapshot_and_taxonomy_honestly(clean, tools, monkeypatch):
    out = tools["get_industry_context"](tickers=["NVDA"])
    assert out["status"] == "ok" and out["snapshot"] == {"status": "no_snapshot"}
    assert out["groups"][0]["snapshot_row"] is None
    monkeypatch.setattr(reg, "active_version", lambda: None)
    assert tools["get_industry_context"](tickers=["NVDA"])["status"] == "taxonomy_not_imported"


def test_chat_tool_carries_the_access_policy_it_was_served_under(clean, tools, monkeypatch):
    """Owner decision 1: the PM integration is a Pro surface when the login
    wall is on, the latest report follows INDUSTRY_ANALYSIS_ACCESS. The
    answer carries the tier so the UI can explain gating from one source
    rather than re-deriving it from the setting — and it carries the
    taxonomy-missing answer too, where the tier is the same."""
    monkeypatch.setattr(settings, "industry_analysis_access", "public", raising=False)
    monkeypatch.setattr(settings, "auth_enabled", False, raising=False)
    out = tools["get_industry_context"](tickers=["NVDA"])
    assert out["access"] == {"surface": "pm_chat", "tier": "pro", "enforced": False, "latest_report_tier": "public"}
    assert out["access"]["tier"] == rs.surface_tier("pm_chat")

    monkeypatch.setattr(settings, "auth_enabled", True, raising=False)
    monkeypatch.setattr(settings, "industry_analysis_access", "pro", raising=False)
    monkeypatch.setattr(reg, "active_version", lambda: None)
    missing = tools["get_industry_context"](tickers=["NVDA"])
    assert missing["status"] == "taxonomy_not_imported"
    assert missing["access"] == {"surface": "pm_chat", "tier": "pro", "enforced": True, "latest_report_tier": "pro"}




def _count_selects():
    import contextlib

    from sqlalchemy import event

    @contextlib.contextmanager
    def counting():
        seen: dict[str, list[str]] = {"reports": [], "jobs": []}

        def _count(conn, cursor, statement, params, context, executemany):
            if not statement.lstrip().upper().startswith("SELECT"):
                return
            if "industry_reports" in statement:
                seen["reports"].append(statement)
            elif "industry_report_jobs" in statement:
                seen["jobs"].append(statement)

        engine = SessionLocal.kw["bind"]
        event.listen(engine, "before_cursor_execute", _count)
        try:
            yield seen
        finally:
            event.remove(engine, "before_cursor_execute", _count)

    return counting()


def test_chat_tool_reads_every_groups_edition_in_a_constant_number_of_queries(clean, tools):
    """A portfolio question puts several groups in scope; this runs on a
    web request, so the editions come back in the store's two queries
    (candidates' metadata, then the winners) and the attempted periods in
    one — never a query per group."""
    codes = [g.code for g in reg.industry_groups(version=clean)]
    _snapshot(clean, codes)
    for c in codes[:3]:
        rs.save_report(code=c, period_key=PERIOD, as_of=AS_OF, version=clean, generation=AGENTIC,
                       payload={"sections": {"outlook": {"interpretation": {"analyst_view": "x"}}}})

    with _count_selects() as one:
        tools["get_industry_context"](tickers=["NVDA"])
    with _count_selects() as many:
        out = tools["get_industry_context"](tickers=["NVDA", "JPM", "AAPL", "MSFT"])
    assert out["status"] == "ok" and len(out["groups"]) >= 2
    assert len(one["reports"]) == len(many["reports"]) == 2, many["reports"]
    assert len(one["jobs"]) == len(many["jobs"]) == 1, many["jobs"]


def test_pm_reads_analyst_editions_only(clean, tools):
    """A template edition never reaches the PM or the chat tool — not the
    newest one, and not a legacy one still flagged latest-good. The group
    reads as having no edition, or as its last ANALYST edition."""
    nvda, jpm = _own_group("NVDA", clean), _own_group("JPM", clean)
    _snapshot(clean, [nvda, jpm])
    rs.save_report(code=nvda, period_key=PERIOD, as_of=AS_OF, version=clean, generation=TEMPLATE,
                   payload={"sections": {"outlook": {"interpretation": {"analyst_view": "TEMPLATE VIEW"}}}})
    with SessionLocal() as db:  # legacy: analyst v1 superseded by a template flagged latest-good
        db.add(IndustryReport(taxonomy_version_id=clean.id, industry_group_code=jpm, version=1,
                              period_key="2026-W35", as_of=AS_OF, status="superseded", is_latest_good=False,
                              generation=AGENTIC, degraded=[], generated_at=AS_OF,
                              payload={"sections": {"outlook": {"interpretation": {"analyst_view": "Analyst view."}}}}))
        db.add(IndustryReport(taxonomy_version_id=clean.id, industry_group_code=jpm, version=2,
                              period_key=PERIOD, as_of=AS_OF, status="succeeded", is_latest_good=True,
                              generation=TEMPLATE, degraded=[], generated_at=AS_OF,
                              payload={"sections": {"outlook": {"interpretation": {"analyst_view": "TEMPLATE VIEW"}}}}))
        db.commit()

    block = pm_context.industry_context_block(ticker="NVDA", tickers=["JPM"])
    assert "TEMPLATE VIEW" not in block
    assert f"Industry group {il.label(nvda)}: no published Industry Analysis edition yet." in block
    assert f"Industry group {il.label(jpm)} — edition of 2026-W35" in block and "Analyst view." in block
    out = tools["get_industry_context"](tickers=["NVDA", "JPM"])
    by_code = {g["code"]: g for g in out["groups"]}
    assert by_code[il.slug(nvda)]["report"] is None
    assert by_code[il.slug(jpm)]["report"]["version"] == 1 and "TEMPLATE" not in str(out)


def test_diff_and_pm_excerpt_hide_template_what_changed(clean):
    """`what_changed` (and the outlook view) written by a template inside an
    analyst edition is template prose. The diff returns null with the
    reason; the PM excerpt returns "" (rendered "n/a"). Neither quotes it."""
    nvda = _own_group("NVDA", clean)
    _snapshot(clean, [nvda])
    base = {"sections": {
        "outlook": {"interpretation": {"analyst_view": "Model view one."}},
        "what_changed": {"interpretation": {"text": "Model change one."}},
    }}
    rs.save_report(code=nvda, period_key="2026-W35", as_of=AS_OF, version=clean, generation=AGENTIC, payload=base)
    templated = {"sections": {
        "outlook": {"interpretation": {"analyst_view": "Model outlook two."}},
        "what_changed": {"interpretation": {"text": "TEMPLATE CHANGE"}},
    }}
    # One template-filled section of ten leaves a publishable analyst
    # edition; the marker is what says which one the template wrote.
    rs.save_report(code=nvda, period_key=PERIOD, as_of=AS_OF, version=clean, generation=AGENTIC,
                   payload=templated, degraded=["analyst_narrative:what_changed:deterministic"])

    delta = rs.diff(nvda, 1, 2, version=clean)
    view = delta["analyst_view"]
    assert view["what_changed"] is None and "template-filled" in view["reasons"]["what_changed"]
    assert view["to"] == "Model outlook two." and view["reasons"]["to"] is None
    assert view["from"] == "Model view one."

    excerpt = pm_context._report_excerpt(nvda)
    assert excerpt["what_changed"] == "" and excerpt["hidden_sections"] == ["what_changed"]
    block = pm_context.industry_context_block(ticker="NVDA")
    assert "TEMPLATE CHANGE" not in block and "What changed: n/a" in block

    # A template-filled outlook is hidden the same way on both surfaces
    # (such an edition is below the publication minimum, so it can only
    # arrive through an explicit comparison or a legacy row — hence the
    # direct calls).
    outlook_templated = {
        "payload": {"sections": {
            "outlook": {"interpretation": {"analyst_view": "TEMPLATE OUTLOOK"}},
            "what_changed": {"interpretation": {"text": "Model change three."}},
        }},
        "degraded": ["analyst_narrative:outlook:deterministic"], "version": 3, "period_key": PERIOD,
    }
    ex = pm_context._excerpt_from(nvda, outlook_templated)
    assert ex["analyst_view"] == "" and "TEMPLATE" not in str(ex)
    both = rs._analyst_view({"payload": base, "degraded": []}, outlook_templated)
    assert both["to"] is None and "template-filled" in both["reasons"]["to"]


def test_block_marks_a_group_not_updated_this_week(clean):
    """The newest week finished without an analyst edition: the PM reads
    last week's analysis and is told it was not updated, and which week."""
    from app.models import IndustryReportJob

    nvda = _own_group("NVDA", clean)
    _snapshot(clean, [nvda])
    rs.save_report(code=nvda, period_key=PERIOD, as_of=AS_OF, version=clean, generation=AGENTIC,
                   payload={"sections": {"outlook": {"interpretation": {"analyst_view": "Old view."}}}})
    template = rs.save_report(code=nvda, period_key="2026-W37", as_of=AS_OF, version=clean, generation=TEMPLATE,
                              payload={"sections": {}})
    with SessionLocal() as db:
        db.add(IndustryReportJob(kind="group_report", taxonomy_version_id=clean.id, industry_group_code=nvda,
                                 period_key="2026-W37", run_id="r", status="succeeded", attempts=3,
                                 max_attempts=3, enqueued_at=AS_OF, report_id=template.id))
        db.commit()
    try:
        block = pm_context.industry_context_block(ticker="NVDA")
        assert (f"(not updated this week; newest analyst edition is {PERIOD}, the 2026-W37 refresh produced none)"
                in block)
        assert pm_context._report_excerpt(nvda)["not_updated"] == "2026-W37"
    finally:
        with SessionLocal() as db:
            db.query(IndustryReportJob).filter(IndustryReportJob.industry_group_code == nvda).delete()
            db.commit()
