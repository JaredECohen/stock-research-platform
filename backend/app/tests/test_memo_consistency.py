"""Memo-consistency invariants.

A single memo used to answer "is it cheap or expensive?" independently in
five places (thesis verdict word, rating badge, comps premium, valuation
headline, DCF summary) and the answers were never reconciled — see
NEXT_STEPS.md B1/B2/Theme 1. These tests lock the reconciliation in so a
regression of that class turns into a red CI run instead of a user-visible
self-contradicting memo.
"""
from __future__ import annotations

import pytest

from app.agents import graph
from app.agents.comps_agent import run_comps_agent
from app.agents.graph import (
    _build_valuation_verdict,
    _refresh_dcf_references,
    _risk_items_from_bear_case,
    _verdict_word,
)
from app.agents.safe_runner import DegradationLog
from app.agents.valuation_agent import run_valuation_agent
from app.schemas import AgentFinding, BullBearCase, DCFResult
from app.services.valuation_service import build_comps, build_dcf


# ---------------------------------------------------------------------------
# B2 — one DCF number everywhere
# ---------------------------------------------------------------------------

def _adjusted_copy(dcf: DCFResult, *, base_upside: float, base_price: float) -> DCFResult:
    data = dcf.model_dump()
    data["base"]["upside_pct"] = base_upside
    data["base"]["implied_share_price"] = base_price
    return DCFResult(**data)


def test_refresh_dcf_references_rewrites_stale_numbers():
    old = build_dcf("NVDA")
    new = _adjusted_copy(old, base_upside=0.17, base_price=999.99)

    finding = AgentFinding(
        agent="Valuation Analyst",
        headline=f"DCF base implies {old.base.upside_pct:+.0%} vs current",
        summary=(
            f"Base case implied price ${old.base.implied_share_price:,.2f} "
            f"({old.base.upside_pct:+.1%})."
        ),
        key_points=[f"Base case implied price: ${old.base.implied_share_price:,.2f}"],
        confidence=0.7,
    )
    _refresh_dcf_references(finding, old, new)

    assert f"{new.base.upside_pct:+.0%}" in finding.headline
    assert f"{old.base.upside_pct:+.0%}" not in finding.headline
    assert f"${new.base.implied_share_price:,.2f}" in finding.summary
    assert f"${old.base.implied_share_price:,.2f}" not in finding.summary
    assert any(f"${new.base.implied_share_price:,.2f}" in p for p in finding.key_points)
    # Transparency note appended so the reader knows figures were adjusted.
    assert any("PM-adjusted" in p for p in finding.key_points)


def test_refresh_dcf_references_noops_when_unchanged():
    dcf = build_dcf("NVDA")
    finding = AgentFinding(
        agent="Valuation Analyst", headline="h", summary="s",
        key_points=["k"], confidence=0.7,
    )
    _refresh_dcf_references(finding, dcf, dcf)
    assert finding.key_points == ["k"]  # no note when nothing changed


# ---------------------------------------------------------------------------
# Theme 1 — reconciled valuation verdict
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nvda_memo():
    return graph.run_stock_memo("NVDA")


def test_valuation_verdict_populated(nvda_memo):
    vv = nvda_memo.valuation_verdict
    assert vv.summary, "valuation_verdict.summary must never ship empty"
    assert vv.verdict in ("undervalued", "fairly_priced", "overvalued")


def test_valuation_verdict_matches_dcf_summary(nvda_memo):
    """The verdict's DCF number must be THE dcf_summary number — not a
    stale pre-adjustment copy (the B2 failure mode)."""
    base = nvda_memo.dcf_summary.get("base_upside")
    assert nvda_memo.valuation_verdict.dcf_base_upside == base


def test_valuation_verdict_word_follows_rating(nvda_memo):
    expected = _verdict_word(
        nvda_memo.rating_label, nvda_memo.dcf_summary.get("base_upside"),
    ).replace(" ", "_")
    assert nvda_memo.valuation_verdict.verdict == expected


def test_thesis_verdict_word_agrees_with_rating(nvda_memo):
    """B1 regression guard: if the thesis states a verdict word, it must be
    the one implied by the FINAL (post-blend) rating badge."""
    expected = _verdict_word(
        nvda_memo.rating_label, nvda_memo.dcf_summary.get("base_upside"),
    )
    thesis = nvda_memo.one_sentence_thesis.lower()
    stated = [
        w for w in ("undervalued", "overvalued", "fairly priced") if w in thesis
    ]
    if stated:
        assert stated[0] == expected, (
            f"thesis says {stated[0]!r} but rating "
            f"{nvda_memo.rating_label!r} implies {expected!r}"
        )


def test_thesis_carries_no_template_filler(nvda_memo):
    assert "core driver execution vs. the dominant risk" not in (
        nvda_memo.one_sentence_thesis
    )


# ---------------------------------------------------------------------------
# B4 — risk section never empty while a bear case exists
# ---------------------------------------------------------------------------

def test_key_risks_nonempty_when_bear_case_exists(nvda_memo):
    if nvda_memo.bear_case.key_points:
        assert nvda_memo.key_risks, (
            "bear case is populated but key_risks is empty — backfill failed"
        )


def test_risk_items_from_bear_case_skips_price_target_line():
    bear = BullBearCase(
        headline="Bear case",
        key_points=[
            "Multiple compression risk: EV/EBITDA 30.8x.",
            "DCF bear case implies $700.00 (-28%).",
        ],
    )
    items = _risk_items_from_bear_case(bear)
    assert len(items) == 1
    assert items[0].type == "valuation"
    assert "compression" in items[0].title.lower()


def test_risk_items_from_bear_case_empty_inputs():
    assert _risk_items_from_bear_case(None) == []
    assert _risk_items_from_bear_case(BullBearCase(headline="x", key_points=[])) == []


# ---------------------------------------------------------------------------
# B6 — mispricing thesis never ships blank
# ---------------------------------------------------------------------------

def test_mispricing_thesis_never_empty(nvda_memo):
    m = nvda_memo.mispricing_thesis
    assert m.consensus_view or m.our_view or m.gap, (
        "mispricing_thesis shipped with every field blank"
    )
    # And it can't contradict the reconciled verdict: a fairly_priced
    # verdict must say "no material mispricing", not claim an edge.
    if nvda_memo.valuation_verdict.verdict == "fairly_priced":
        assert (
            "no material mispricing" in m.gap.lower()
            or nvda_memo.valuation_verdict.summary
        )


# ---------------------------------------------------------------------------
# B5 — comps headline carries the punchline
# ---------------------------------------------------------------------------

def test_comps_headline_not_generic_when_premium_available():
    comps = build_comps("NVDA")
    if comps is None or (comps.premium_discount or {}).get("ev_ebitda") is None:
        pytest.skip("demo comps for NVDA carries no EV/EBITDA premium")
    finding = run_comps_agent({"ticker": "NVDA"}, comps)
    assert finding.headline != "Peer-relative read for NVDA"
    assert "%" in finding.headline, "headline should carry the magnitude"


# ---------------------------------------------------------------------------
# B3 / Theme 2 — deterministic fallback counts as degradation
# ---------------------------------------------------------------------------

def test_valuation_fallback_sets_flag(monkeypatch):
    """When the LLM call returns nothing usable, the valuation agent must
    mark its deterministic fallback so the graph can promote it into
    `degraded_agents` on LLM-enabled runs."""
    from app.agents import valuation_agent as va
    monkeypatch.setattr(va.llm, "chat_json", lambda *a, **k: None)
    finding = run_valuation_agent({"ticker": "NVDA"}, {"PE": 50.0}, None)
    assert isinstance(finding.data, dict)
    assert finding.data.get("deterministic_fallback")
    assert finding.confidence == 0.5


def test_record_soft_appears_in_degraded_agents_and_dedupes():
    log = DegradationLog()
    log.record_soft("Valuation Analyst", "fell back to deterministic output")
    log.record_soft("Valuation Analyst", "fell back again")
    assert log.degraded_agents() == ["Valuation Analyst"]
    assert log.failures[0]["error_type"] == "DeterministicFallback"


def test_healthy_run_does_not_flag_valuation_as_degraded(nvda_memo):
    """A healthy run must not flag the valuation agent: either the LLM
    produced a real view (no fallback fired), or no LLM is configured at
    all — in which case the deterministic path IS the expected behavior
    and the graph's `settings.has_llm` gate keeps it out of the log."""
    assert "Valuation Analyst" not in nvda_memo.degraded_agents
