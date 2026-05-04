"""Wave 3C tests — drill-down long-form agent reports.

Covers:
- Deterministic build always succeeds and produces structured markdown
  (no LLM required), with all expected sections when fields are populated.
- LLM enrichment is gated by `ENABLE_LONG_FORM_REPORTS`. With the flag
  off, the deterministic body alone is returned (no enrichment header).
- `attach_long_form` is a no-op on None and never raises on partial
  findings.
- A full memo run populates `long_form_report` on every specialist
  finding (sector / earnings / filing / valuation / comps / macro / risk
  + the optional technical).
"""
from __future__ import annotations

from app.agents.long_form import (
    attach_long_form,
    build_long_form_report,
    deterministic_long_form,
)
from app.config import settings
from app.schemas import AgentFinding


def _stub_finding(
    *, agent="Sector Analyst", headline="h", summary="s",
    key_points=None, sources=None, data=None,
) -> AgentFinding:
    return AgentFinding(
        agent=agent, headline=headline, summary=summary,
        key_points=key_points or ["bullet a", "bullet b"],
        sources=sources or ["src:1"],
        data=data or {},
    )


def test_deterministic_long_form_includes_all_expected_sections():
    f = _stub_finding(
        data={
            "kpi_placements": {
                "revenue_growth": {
                    "target": 0.50, "distribution": {"median": 0.18}, "quartile": 1,
                },
            },
            "cross_sector_relevance": ["NEE", "CAT"],
            "bull_bear_analysis": {
                "sector_synthesis": "Cohort placement supports premium.",
                "key_disagreement": "Bulls vs bears on cycle.",
            },
        },
    )
    md = deterministic_long_form(f, ticker="NVDA", agent_name="Sector Analyst")
    assert "## Sector Analyst — NVDA drill-down" in md
    assert "Key points" in md
    assert "Cohort placement" in md
    assert "Cross-sector pull-through" in md
    assert "Sector synthesis" in md
    assert "Sources" in md
    # Wave 8O renamed the trailer to "Analyst confidence" + 0-100 scale.
    assert "confidence" in md.lower()


def test_deterministic_long_form_handles_empty_data():
    f = _stub_finding(data={})
    md = deterministic_long_form(f, ticker="MSFT", agent_name="Risk Analyst")
    assert "## Risk Analyst — MSFT drill-down" in md
    # Should not crash on missing optional sections.
    assert "Cohort placement" not in md


def test_build_long_form_report_returns_deterministic_body_when_flag_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_long_form_reports", False)
    f = _stub_finding()
    out = build_long_form_report(f, ticker="MSFT", agent_name="Sector Analyst")
    assert "## Sector Analyst — MSFT drill-down" in out
    # No enrichment marker when the flag is off.
    assert "Analyst expansion" not in out


def test_build_long_form_report_appends_enrichment_when_flag_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_long_form_reports", True)

    # Stub the LLM enrichment so the test is offline.
    from app.agents import long_form as lf
    monkeypatch.setattr(
        lf, "_enriched_long_form",
        lambda finding, *, ticker, agent_name, profile=None: "Extra paragraph.",
    )
    f = _stub_finding()
    out = lf.build_long_form_report(f, ticker="NVDA", agent_name="Macro Analyst")
    assert "Analyst expansion" in out
    assert "Extra paragraph." in out


def test_build_long_form_report_enrichment_failure_falls_back(monkeypatch):
    """When LLM enrichment returns None, the deterministic body alone is returned."""
    monkeypatch.setattr(settings, "enable_long_form_reports", True)
    from app.agents import long_form as lf
    monkeypatch.setattr(
        lf, "_enriched_long_form",
        lambda finding, *, ticker, agent_name, profile=None: None,
    )
    f = _stub_finding()
    out = lf.build_long_form_report(f, ticker="NVDA", agent_name="Comps Analyst")
    assert "Analyst expansion" not in out
    assert "## Comps Analyst" in out


def test_attach_long_form_noop_on_none():
    assert attach_long_form(None, ticker="X", agent_name="Y") is None


def test_attach_long_form_mutates_in_place_and_returns_finding():
    f = _stub_finding()
    out = attach_long_form(f, ticker="MSFT", agent_name="Sector Analyst")
    assert out is f  # same object
    assert f.long_form_report is not None
    assert "## Sector Analyst — MSFT drill-down" in f.long_form_report


# ---------------------------------------------------------------------------
# Graph integration
# ---------------------------------------------------------------------------

def test_full_memo_run_populates_long_form_on_every_specialist():
    from app.agents.graph import run_stock_memo
    memo = run_stock_memo("MSFT")
    for view in (
        memo.sector_agent_view,
        memo.earnings_agent_view,
        memo.filing_agent_view,
        memo.valuation_agent_view,
        memo.comps_agent_view,
        memo.macro_sensitivity,
    ):
        assert view.long_form_report, (
            f"{view.agent} missing long_form_report"
        )
    # Technical is optional but present when wired up (Wave 3B).
    if memo.technical_agent_view is not None:
        assert memo.technical_agent_view.long_form_report
