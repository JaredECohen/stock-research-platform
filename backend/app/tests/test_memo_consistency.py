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


def test_healthy_run_carries_no_degradation_events(nvda_memo):
    """RP-001: with no keys every deterministic path is the design, so a
    healthy demo run must not report PM Synthesis / PM DCF Adjuster /
    Thesis Builder as degraded — `note_soft` is gated on `has_llm` where
    the deterministic path is expected."""
    assert nvda_memo.degradation_events == []
    assert nvda_memo.degraded_agents == []
    assert nvda_memo.extra_agent_views == {}


def test_degradation_events_agree_with_degraded_agents(nvda_memo_unpriced):
    """The two fields are views of one accumulator; whatever the run
    recorded, their agent lists must be identical and in order."""
    memo = nvda_memo_unpriced
    assert [e["agent"] for e in memo.degradation_events] == memo.degraded_agents


# ---------------------------------------------------------------------------
# DCF unavailable — None is "n/a", never "+0.0%" / "$0.00" / a neutral signal
# ---------------------------------------------------------------------------

def _unpriced_copy(dcf: DCFResult) -> DCFResult:
    """The shape `run_dcf` emits when the share count / quote never reached
    the model: every scenario's implied price and upside are None."""
    data = dcf.model_dump()
    for k in ("base", "bull", "bear"):
        data[k]["implied_share_price"] = None
        data[k]["upside_pct"] = None
    data["current_price"] = None
    return DCFResult(**data)


_ZERO_LIES = ("+0.0%", "+0%", "$0.00")


def test_refresh_dcf_references_tolerates_none_numbers():
    """Neither side of the PM adjustment may crash the rewrite when a
    number is None, and the transparency note must read n/a rather than
    fabricate a figure. Nothing is substituted for a None side — there is
    no printed variant of "n/a" that is safe to rewrite."""
    old = build_dcf("NVDA")
    new = _unpriced_copy(old)
    finding = AgentFinding(
        agent="Valuation Analyst",
        headline=f"DCF base implies {old.base.upside_pct:+.0%} vs current",
        summary="s", key_points=["k"], confidence=0.7,
    )
    _refresh_dcf_references(finding, old, new)
    # Old numbers are left alone (no pairs), so only the note is appended.
    assert f"{old.base.upside_pct:+.0%}" in finding.headline
    assert finding.key_points == ["k"] or any(
        "PM-adjusted" in p and "n/a" in p for p in finding.key_points
    )
    # And the reverse direction (old None → new real) must not raise either.
    finding2 = AgentFinding(agent="Valuation Analyst", headline="h", summary="s",
                            key_points=[], confidence=0.7)
    _refresh_dcf_references(finding2, new, old)
    assert not any(z in p for p in finding2.key_points for z in _ZERO_LIES)


def test_valuation_verdict_treats_none_dcf_as_unavailable(nvda_memo):
    memo = nvda_memo.model_copy(
        update={"dcf_summary": {**nvda_memo.dcf_summary, "base_upside": None}},
    )
    vv = _build_valuation_verdict(memo, build_comps("NVDA"))
    assert vv.dcf_base_upside is None
    assert "DCF unavailable" in vv.summary
    assert not any(z in vv.summary for z in _ZERO_LIES)
    # The word still follows the rating badge — None is not a 0% neutral.
    assert vv.verdict == _verdict_word(memo.rating_label, None).replace(" ", "_")


def test_verdict_word_accepts_none_upside():
    assert _verdict_word(None, None) == "fairly priced"
    assert _verdict_word("Bullish", None) == "undervalued"
    assert _verdict_word("Bearish", None) == "overvalued"
    assert _verdict_word("Neutral", None) == "fairly priced"


def test_valuation_agent_fallback_renders_na_for_unpriced_dcf(monkeypatch):
    from app.agents import valuation_agent as va
    monkeypatch.setattr(va.llm, "chat_json", lambda *a, **k: None)
    dcf = _unpriced_copy(build_dcf("NVDA"))
    finding = run_valuation_agent({"ticker": "NVDA"}, {"PE": 50.0}, dcf)
    assert "DCF upside n/a" in finding.headline
    assert any(p == "Base case implied price: n/a" for p in finding.key_points)
    assert any(p == "Bull case: n/a | Bear case: n/a" for p in finding.key_points)
    for text in [finding.headline, finding.summary, *finding.key_points]:
        assert not any(z in text for z in _ZERO_LIES)


@pytest.fixture(scope="module")
def nvda_memo_unpriced():
    """A full memo run whose DCF could not price the shares. Module-scoped
    like `nvda_memo` (a memo run is the expensive part of this file);
    `pytest.MonkeyPatch` because the function-scoped `monkeypatch` fixture
    can't back a module fixture."""
    unpriced = _unpriced_copy(build_dcf("NVDA"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(graph, "build_dcf", lambda ticker, **kw: unpriced)
        yield graph.run_stock_memo("NVDA")


def test_unpriced_memo_dcf_summary_carries_none(nvda_memo_unpriced):
    m = nvda_memo_unpriced
    assert m.dcf_summary, "the DCF ran — the summary must not vanish"
    assert m.dcf_summary["base_upside"] is None
    assert m.dcf_summary["base_implied_price"] is None
    assert m.dcf_summary["current_price"] is None
    assert m.dcf_summary["tv_clamped"] is False


def test_unpriced_memo_verdict_names_dcf_unavailable(nvda_memo_unpriced):
    vv = nvda_memo_unpriced.valuation_verdict
    assert vv.dcf_base_upside is None
    assert "DCF unavailable" in vv.summary
    assert vv.verdict == _verdict_word(nvda_memo_unpriced.rating_label, None).replace(" ", "_")


def test_unpriced_memo_keeps_consistency_invariants(nvda_memo_unpriced):
    """The B1/B6 invariants must survive a None DCF: the thesis verdict
    word agrees with the rating, and the mispricing card still ships."""
    m = nvda_memo_unpriced
    expected = _verdict_word(m.rating_label, None)
    stated = [
        w for w in ("undervalued", "overvalued", "fairly priced")
        if w in m.one_sentence_thesis.lower()
    ]
    if stated:
        assert stated[0] == expected
    assert m.mispricing_thesis.consensus_view or m.mispricing_thesis.our_view or m.mispricing_thesis.gap
    assert m.key_risks or not m.bear_case.key_points


def test_unpriced_memo_prose_never_prints_zero_lies(nvda_memo_unpriced):
    """No section may render the missing number as a real one."""
    m = nvda_memo_unpriced
    texts = [
        m.one_sentence_thesis, m.final_pm_view, m.final_verdict,
        m.valuation_verdict.summary, m.mispricing_thesis.gap,
        m.bull_case.headline, m.bear_case.headline,
        *m.bull_case.key_points, *m.bear_case.key_points,
        m.valuation_agent_view.headline, m.valuation_agent_view.summary,
        *m.valuation_agent_view.key_points,
        str(m.dcf_summary.get("summary", "")),
    ]
    for t in texts:
        assert not any(z in t for z in _ZERO_LIES), t
    assert any("n/a" in p for p in m.bull_case.key_points + m.bear_case.key_points)
