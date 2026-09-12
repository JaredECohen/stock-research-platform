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
from app.services import industry_report_store as rs
from app.services import industry_snapshot as isn
from app.tests.fixtures.demo_dataset import COMPANY_PROFILES
from app.tests.gating_helpers import seed_demo_universe

AS_OF = datetime(2026, 9, 4, 21, 0)
PERIOD = "2026-W36"


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
    assert f"Industry group {nvda}: no published Industry Analysis edition yet." in block
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
        code=nvda, period_key=PERIOD, as_of=AS_OF, version=clean, degraded=["analyst_narrative:llm_unavailable"],
        payload={"sections": {
            "outlook": {"interpretation": {"analyst_view": "Capacity additions " + "x" * 900}},
            "what_changed": {"interpretation": {"text": "Breadth narrowed."}},
        }},
    )
    block = pm_context.industry_context_block(ticker="NVDA", tickers=["ZZNOPE"])
    assert f"Industry group {nvda} — edition v1 {PERIOD} (degraded edition: analyst_narrative:llm_unavailable)" in block
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
    rs.save_report(code=jpm, period_key=PERIOD, as_of=AS_OF, version=clean,
                   payload={"sections": {"outlook": {"interpretation": {"analyst_view": "Banks steady."}}}})

    def _boom(*a, **k):
        raise AssertionError("the chat tool must not compute or fetch")

    monkeypatch.setattr(ia, "compute_group_stats", _boom)
    monkeypatch.setattr(ia, "load_context", _boom)
    monkeypatch.setattr(isn, "compute_cross_snapshot", _boom)
    from app.services import data_service
    monkeypatch.setattr(data_service.DataService, "get_price_history", _boom)

    out = tools["get_industry_context"](tickers=["NVDA", "jpm", "ZZNOPE"])
    assert out["status"] == "ok" and out["taxonomy_version"] == clean.version_key
    assert out["by_ticker"] == {"NVDA": {"code": nvda, "state": "mapped"}, "JPM": {"code": jpm, "state": "mapped"}}
    assert out["unmapped_tickers"] == [{"ticker": "ZZNOPE", "state": "unclassified"}]
    assert out["snapshot"]["period_key"] == PERIOD and out["snapshot"]["macro_regime"] == "Soft landing"
    by_code = {g["code"]: g for g in out["groups"]}
    assert by_code[nvda]["relation"] == "own" and by_code[nvda]["snapshot_row"]["ret_1m_ew"] == 0.04
    assert by_code[nvda]["report"] is None
    assert by_code[jpm]["report"]["analyst_view"] == "Banks steady." and by_code[jpm]["report"]["version"] == 1
    linked = [g for g in out["groups"] if g["relation"] == "linked"]
    # Linked groups had no stats this period: the snapshot names them as
    # such rather than carrying numbers for them.
    assert all(g["via"] and g["snapshot_row"]["status"] == "no_stats" for g in linked)
    assert "scenarios, not recommendations" in out["note"] and out["mapping_caveat"]

    explicit = tools["get_industry_context"](code=jpm)
    assert explicit["code"] == {"code": jpm, "name": reg.group(jpm, version=clean).name}
    assert [g["relation"] for g in explicit["groups"]] == ["requested"]
    bad = tools["get_industry_context"](tickers=[], code="9999")
    assert "not found" in bad["code"]["error"] and bad["groups"] == []


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


def test_chat_tool_reads_every_groups_edition_in_one_query(clean, tools):
    """A portfolio question puts several groups in scope; the editions
    must come back in one SELECT, not one per group, because this runs on
    a web request."""
    from sqlalchemy import event

    codes = [g.code for g in reg.industry_groups(version=clean)]
    _snapshot(clean, codes)
    for c in codes[:3]:
        rs.save_report(code=c, period_key=PERIOD, as_of=AS_OF, version=clean,
                       payload={"sections": {"outlook": {"interpretation": {"analyst_view": "x"}}}})

    selects: list[str] = []

    def _count(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT") and "industry_reports" in statement:
            selects.append(statement)

    engine = SessionLocal.kw["bind"]
    event.listen(engine, "before_cursor_execute", _count)
    try:
        out = tools["get_industry_context"](tickers=["NVDA", "JPM", "AAPL", "MSFT"])
    finally:
        event.remove(engine, "before_cursor_execute", _count)
    assert out["status"] == "ok" and len(out["groups"]) >= 2
    assert len(selects) == 1, f"{len(selects)} report queries for {len(out['groups'])} groups"
