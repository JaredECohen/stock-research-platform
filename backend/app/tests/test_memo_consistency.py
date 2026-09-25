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

from app.agents import graph, memo_quality
from app.agents.comps_agent import run_comps_agent
from app.agents.graph import (
    _refresh_dcf_references,
    _risk_items_from_bear_case,
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
# FIX-008 — DCF-derived arithmetic is recomputed, not left from the old model
# ---------------------------------------------------------------------------

def _with_scenarios(dcf: DCFResult, *, current: float | None, **scenarios) -> DCFResult:
    """Copy `dcf` with (implied_price, upside) set per named scenario."""
    data = dcf.model_dump()
    data["current_price"] = current
    for name, (price, upside) in scenarios.items():
        data[name]["implied_share_price"] = price
        data[name]["upside_pct"] = upside
    return DCFResult(**data)


# The saved META v1 memo (docs/reviews/2026-09-13-META-v1.json): the
# valuation analyst wrote against dcf_initial_summary, the PM DCF Adjuster
# shipped dcf_summary.
def _meta_initial_and_final() -> tuple[DCFResult, DCFResult]:
    dcf = build_dcf("NVDA")
    initial = _with_scenarios(
        dcf, current=648.03,
        base=(794.9500804392044, 0.22671802299153498),
        bull=(1454.9509556865448, 1.2451907406856857),
        bear=(373.3253526686917, -0.42390729955605183),
    )
    final = _with_scenarios(
        dcf, current=648.03,
        base=(708.9303286367508, 0.0939776378203954),
        bull=(1162.6842269331853, 0.7941827182895628),
        bear=(366.0839443044916, -0.4350817951260102),
    )
    return initial, final


_META_KP0 = (
    "Base DCF $794.95 (+22.7%); bull $1,454.95 (+124.5%); bear $373.33 (-42.4%). "
    "The 3.9x bull/bear ratio is the tell — this is a terminal-growth-fragile name."
)


def test_refresh_recomputes_bull_bear_ratio_meta_counterexample():
    """META v1 kept "3.9x" (1,454.95 / 373.33) beside final prices whose
    ratio is 3.176x. The ratio must come from the final model."""
    initial, final = _meta_initial_and_final()
    finding = AgentFinding(agent="Valuation Analyst", headline="h", summary="s",
                           key_points=[_META_KP0], confidence=0.7)
    _refresh_dcf_references(finding, initial, final)
    kp = finding.key_points[0]
    assert kp.startswith(
        "Base DCF $708.93 (+9.4%); bull $1,162.68 (+79.4%); bear $366.08 (-43.5%). "
        "The 3.2x bull/bear ratio"
    )
    assert "3.9x" not in kp


def test_refresh_rewrites_long_form_report():
    """The drill-down is rendered before the PM adjustment; it must not
    keep the initial model's numbers while the card shows the final ones."""
    initial, final = _meta_initial_and_final()
    long_form = (
        "**META screens cheap on our DCF (+22.7% base)**\n\n"
        "### Key points\n- " + _META_KP0 + "\n\n"
        "The 3.9x bull-to-bear spread ($373–$1,455) underscores fragility; "
        "27.5x earnings, 16.4x EV/EBITDA."
    )
    finding = AgentFinding(agent="Valuation Analyst", headline="h", summary="s",
                           key_points=[], confidence=0.7, long_form_report=long_form)
    _refresh_dcf_references(finding, initial, final)
    out = finding.long_form_report or ""
    assert "(+9.4% base)" in out
    assert "The 3.2x bull/bear ratio" in out
    assert "3.2x bull-to-bear spread ($366–$1,163)" in out
    assert "27.5x earnings, 16.4x EV/EBITDA" in out
    for stale in ("3.9x", "$794.95", "+22.7%", "$1,454.95", "$373", "$1,455"):
        assert stale not in out


def test_refresh_ratio_ambiguity_guard_and_precision():
    """Only a multiple tied to bull/bear wording AND equal to the old ratio
    at its printed precision is rewritten, at that same precision."""
    initial, final = _meta_initial_and_final()
    finding = AgentFinding(
        agent="Valuation Analyst", headline="h", key_points=[], confidence=0.7,
        summary=("A bull/bear ratio of 3.90x. Trades at 3.9x EV/Revenue, "
                 "27.5x earnings; a peer's 4.4x bull/bear ratio; 13.9x bull/bear."),
    )
    _refresh_dcf_references(finding, initial, final)
    assert finding.summary == (
        "A bull/bear ratio of 3.18x. Trades at 3.9x EV/Revenue, "
        "27.5x earnings; a peer's 4.4x bull/bear ratio; 13.9x bull/bear."
    )


def test_refresh_is_single_pass_without_cascades():
    """A new value equal to another scenario's old value must not be
    rewritten a second time, and a figure must not match inside a longer
    number ("9.0%" in "19.0%", "$109" in "$109.50")."""
    dcf = build_dcf("NVDA")
    old = _with_scenarios(dcf, current=100.0, base=(109.0, 0.09),
                          bull=(117.0, 0.17), bear=(80.0, -0.20))
    new = _with_scenarios(dcf, current=100.0, base=(117.0, 0.17),
                          bull=(130.0, 0.30), bear=(80.0, -0.20))
    finding = AgentFinding(
        agent="Valuation Analyst", headline="Base +9%, bull +17%", confidence=0.7,
        summary="Base $109.00, bull $117.00; margin 19.0%; peer at $109.50.",
        key_points=[],
    )
    _refresh_dcf_references(finding, old, new)
    assert finding.headline == "Base +17%, bull +30%"
    assert finding.summary == "Base $117.00, bull $130.00; margin 19.0%; peer at $109.50."


def test_refresh_recomputes_worded_downside_and_upside():
    """Unsigned magnitudes are DCF figures only when "downside"/"upside"
    follows; the same digits elsewhere are left alone."""
    dcf = build_dcf("NVDA")
    old = _with_scenarios(dcf, current=100.0, base=(137.0, -0.63),
                          bull=(122.0, 0.22), bear=(20.0, -0.80))
    new = _with_scenarios(dcf, current=100.0, base=(155.0, -0.55),
                          bull=(118.0, 0.18), bear=(20.0, -0.80))
    finding = AgentFinding(
        agent="Valuation Analyst", headline="h", key_points=[], confidence=0.7,
        summary="Base: 63% downside and 63.0% downside; bull 22% upside; 63% of revenue.",
    )
    _refresh_dcf_references(finding, old, new)
    assert finding.summary == (
        "Base: 55% downside and 55.0% downside; bull 18% upside; 63% of revenue."
    )


def test_refresh_integer_magnitude_needs_a_dcf_clause():
    """An integer "N% upside" is a DCF figure only in a clause that names a
    scenario or an old DCF price; the Street's target or a drawdown that
    happens to share the digits is left alone."""
    initial, final = _meta_initial_and_final()
    finding = AgentFinding(
        agent="Valuation Analyst", headline="h", key_points=[], confidence=0.7,
        summary=("The Street's $800 target implies 23% upside; a 42% downside "
                 "to the 2022 low. Base case: 23% upside. Bear at $373 is a 42% downside."),
    )
    _refresh_dcf_references(finding, initial, final)
    assert finding.summary == (
        "The Street's $800 target implies 23% upside; a 42% downside "
        "to the 2022 low. Base case: 9% upside. Bear at $366 is a 44% downside."
    )


def test_refresh_negative_to_positive_flip_keeps_the_signed_form():
    """A scenario that crosses zero upward: "-5.0%" is both the signed and
    the one-decimal form of the old base, so it must map to the signed new
    figure, not be dropped as ambiguous and left beside the new price."""
    dcf = build_dcf("NVDA")
    old = _with_scenarios(dcf, current=100.0, base=(95.0, -0.05),
                          bull=(140.0, 0.40), bear=(70.0, -0.30))
    new = _with_scenarios(dcf, current=100.0, base=(103.2, 0.032),
                          bull=(145.0, 0.45), bear=(102.0, 0.02))
    finding = AgentFinding(
        agent="Valuation Analyst", headline="DCF base -5.0% vs spot", key_points=[],
        confidence=0.7,
        summary=("Base case implied price $95.00 vs current $100.00 (-5.0%). "
                 "Bull $140.00 (+40.0%) | Bear $70.00 (-30.0%)."),
    )
    _refresh_dcf_references(finding, old, new)
    assert finding.headline == "DCF base +3.2% vs spot"
    assert finding.summary == (
        "Base case implied price $103.20 vs current $100.00 (+3.2%). "
        "Bull $145.00 (+45.0%) | Bear $102.00 (+2.0%)."
    )


def test_refresh_sign_flip_rewrites_magnitude_and_direction():
    """A scenario whose sign the PM adjustment flipped: the worded figure is
    rebuilt from the new value, direction word included, so neither the old
    model's magnitude nor an inverted direction survives."""
    dcf = build_dcf("NVDA")
    old = _with_scenarios(dcf, current=100.0, base=(37.0, -0.63),
                          bull=(122.0, 0.22), bear=(20.0, -0.80))
    new = _with_scenarios(dcf, current=100.0, base=(110.0, 0.10),
                          bull=(95.0, -0.05), bear=(20.0, -0.80))
    finding = AgentFinding(
        agent="Valuation Analyst", headline="h", key_points=[], confidence=0.7,
        summary=("Base implies a 63% downside; bull 22.0% upside. "
                 "If rates fall, the bull case's 22.0% Upside evaporates. "
                 "Bull $122.00 (+22.0%); bull 22% upside."),
    )
    _refresh_dcf_references(finding, old, new)
    assert finding.summary == (
        "Base implies a 10% upside; bull 5.0% downside. "
        "If rates fall, the bull case's 5.0% Downside evaporates. "
        "Bull $95.00 (-5.0%); bull 5% downside."
    )
    for wrong in ("63%", "22.0%", "22%", "10% downside", "-5.0% upside", "5.0% upside"):
        assert wrong not in finding.summary


def test_refresh_rewrites_figures_after_a_comma():
    """The standalone guard only matters for digit-led figures; a "$" or
    signed figure right after a comma is still a DCF figure."""
    initial, final = _meta_initial_and_final()
    finding = AgentFinding(
        agent="Valuation Analyst", headline="h", key_points=[], confidence=0.7,
        summary="Base/bull/bear: +22.7%,+124.5%,-42.4% ($794.95,$1,454.95,$373.33).",
    )
    _refresh_dcf_references(finding, initial, final)
    assert finding.summary == (
        "Base/bull/bear: +9.4%,+79.4%,-43.5% ($708.93,$1,162.68,$366.08)."
    )


@pytest.mark.parametrize("swap", [False, True])
def test_refresh_leaves_a_string_two_scenarios_share(swap):
    """Two old scenarios that print the same string but move to different
    new values: which one the prose meant is unknowable, so it is left as
    written (in either scenario order), while unshared figures still move."""
    dcf = build_dcf("NVDA")
    a, b = (1454.6, 0.101), (1455.2, 0.099)
    na, nb = (708.93, 0.05), (1162.68, 0.20)
    if swap:
        a, b, na, nb = b, a, nb, na
    old = _with_scenarios(dcf, current=1322.0, base=a, bull=b, bear=(900.0, -0.32))
    new = _with_scenarios(dcf, current=1322.0, base=na, bull=nb, bear=(880.0, -0.33))
    finding = AgentFinding(
        agent="Valuation Analyst", headline="h", key_points=[], confidence=0.7,
        summary="Fair value near $1,455 (+10%) on both base and bull; bear $900.00.",
    )
    _refresh_dcf_references(finding, old, new)
    assert finding.summary == (
        "Fair value near $1,455 (+10%) on both base and bull; bear $880.00."
    )


def test_refresh_ratio_left_alone_when_a_side_is_unpriced():
    """Guard (passes before and after FIX-008): no ratio exists for a None
    price, and nothing is invented for it — matching the pct/USD contract."""
    initial, final = _meta_initial_and_final()
    unpriced_bear = _with_scenarios(final, current=648.03, bear=(None, None))
    finding = AgentFinding(agent="Valuation Analyst", headline="h", summary="s",
                           key_points=[_META_KP0], confidence=0.7)
    _refresh_dcf_references(finding, initial, unpriced_bear)
    assert "3.9x bull/bear ratio" in finding.key_points[0]
    assert "$708.93" in finding.key_points[0]
    finding2 = AgentFinding(agent="Valuation Analyst", headline="h", summary="s",
                            key_points=["The 3.9x bull/bear ratio"], confidence=0.7)
    _refresh_dcf_references(finding2, unpriced_bear, initial)
    assert finding2.key_points[0] == "The 3.9x bull/bear ratio"


def test_refresh_noops_on_equal_copies():
    """Guard (passes before and after FIX-008): an equal-valued copy (not
    the same object) changes nothing and appends no note."""
    initial, _ = _meta_initial_and_final()
    finding = AgentFinding(agent="Valuation Analyst", headline="h", summary="s",
                           key_points=[_META_KP0], confidence=0.7,
                           long_form_report=_META_KP0)
    _refresh_dcf_references(finding, initial, initial.model_copy(deep=True))
    assert finding.key_points == [_META_KP0]
    assert finding.long_form_report == _META_KP0


# ---------------------------------------------------------------------------
# Theme 1 — reconciled valuation verdict
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nvda_memo():
    return graph.run_stock_memo("NVDA")


def test_valuation_verdict_populated(nvda_memo):
    vv = nvda_memo.valuation_verdict
    assert vv.summary, "valuation_verdict.summary must never ship empty"
    assert vv.verdict in ("undervalued", "fairly_priced", "overvalued", "mixed")


def _thesis_word_is_defensible(memo) -> None:
    """W2b 7(b): the thesis may state the rating's word or the evidence
    verdict's word (the guard rewrites only a word that contradicts both),
    and an accepted divergence keeps the PM's own word."""
    thesis = memo.one_sentence_thesis.lower()
    stated = [w for w in ("undervalued", "overvalued", "fairly priced") if w in thesis]
    if not stated:
        return
    allowed = {memo_quality.rating_word(memo.rating_label)}
    evidence = memo_quality.verdict_word(memo.valuation_verdict.verdict)
    if evidence is not None:
        allowed.add(evidence)
    assert stated[0] in allowed, (
        f"thesis says {stated[0]!r}; rating {memo.rating_label!r} and evidence "
        f"{memo.valuation_verdict.verdict!r} allow {sorted(allowed)}"
    )


def test_valuation_verdict_matches_dcf_summary(nvda_memo):
    """The verdict's DCF number must be THE dcf_summary number — not a
    stale pre-adjustment copy (the B2 failure mode)."""
    base = nvda_memo.dcf_summary.get("base_upside")
    assert nvda_memo.valuation_verdict.dcf_base_upside == base


def test_valuation_verdict_is_evidence_not_the_rating(nvda_memo):
    """W2b 7(b): the verdict is the evidence read computed before the PM
    wrote (`basis="evidence"`), and its display factor is the same number
    the scores carry — one computation, not two that can drift."""
    vv = nvda_memo.valuation_verdict
    assert vv.basis == "evidence"
    assert vv.signals.get("method") == memo_quality.METHOD
    assert vv.factor_valuation == nvda_memo.scores["factor_valuation"]
    # The rule applied to the stored votes reproduces the stored verdict.
    votes = vv.signals["votes"]
    rich = sum(1 for v in votes.values() if v < 0)
    cheap = sum(1 for v in votes.values() if v > 0)
    if vv.verdict == "overvalued":
        assert rich >= 2 and cheap == 0
    elif vv.verdict == "undervalued":
        assert cheap >= 2 and rich == 0


def test_thesis_verdict_word_agrees_with_rating_or_evidence(nvda_memo):
    """B1 regression guard, W2b form: a stated verdict word must agree with
    the FINAL rating or with the evidence verdict."""
    _thesis_word_is_defensible(nvda_memo)


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
    the deterministic path is expected.

    Phase 6: the one entry a healthy run MAY carry is the scorecard's soft
    "no row on file" note — CI has no worker and therefore no score rows,
    and by design that absence is reported (the section says n/a), not
    hidden. Everything else must still be empty.
    """
    from app.agents.scorecard_context import AGENT_NAME
    events = [e for e in nvda_memo.degradation_events if e["agent"] != AGENT_NAME]
    agents = [a for a in nvda_memo.degraded_agents if a != AGENT_NAME]
    assert events == []
    assert agents == []
    assert nvda_memo.extra_agent_views == {}
    for e in nvda_memo.degradation_events:
        assert e["error_type"] == "DataUnavailable", e


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


def test_valuation_verdict_treats_none_dcf_as_unavailable():
    comps = build_comps("NVDA")
    prem = (comps.premium_discount or {}).get("ev_ebitda") if comps is not None else None
    vv = memo_quality.valuation_evidence_verdict(
        family_pct=None, family_coverage=None, comps_premium=prem,
        dcf_initial_upside=None, dcf_final_upside=None,
    )
    assert vv.dcf_base_upside is None
    assert "DCF unavailable" in vv.summary
    assert not any(z in vv.summary for z in _ZERO_LIES)
    # None is an absent signal, not a 0% neutral vote.
    assert "dcf_initial" not in vv.signals["votes"]


def test_rating_and_verdict_words():
    assert memo_quality.rating_word(None) == "fairly priced"
    assert memo_quality.rating_word("Bullish") == "undervalued"
    assert memo_quality.rating_word("Very Bearish") == "overvalued"
    assert memo_quality.rating_word("Neutral") == "fairly priced"
    assert memo_quality.verdict_word("fairly_priced") == "fairly priced"
    assert memo_quality.verdict_word("overvalued") == "overvalued"
    assert memo_quality.verdict_word("mixed") is None


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
    # An unpriced DCF casts no vote; it is not a 0% neutral signal.
    assert "dcf_initial" not in vv.signals["votes"]


def test_unpriced_memo_keeps_consistency_invariants(nvda_memo_unpriced):
    """The B1/B6 invariants must survive a None DCF: the thesis verdict
    word agrees with the rating or the evidence, and the mispricing card
    still ships."""
    m = nvda_memo_unpriced
    _thesis_word_is_defensible(m)
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


# ---------------------------------------------------------------------------
# RP-002 — `_build_verdict` exercised directly on a fixture memo
#
# The verdict stage is pure: it reads the post-review memo and returns a
# `VerdictOutcome` the orchestrator applies. That is what lets these tests
# target the reconciliation logic without the module-scoped pipeline run
# above (which stays for the end-to-end invariants).
# ---------------------------------------------------------------------------

from app.agents.graph import _build_verdict  # noqa: E402
from app.agents.memo_context import VerdictOutcome  # noqa: E402
from app.schemas import MispricingThesis, ValuationVerdict  # noqa: E402
from app.tests.factories import make_findings, make_memo, make_profile  # noqa: E402


def _verdict_for(memo, *, dcf=None, comps=None, findings=None, profile=None) -> VerdictOutcome:
    return _build_verdict(
        memo, comps=comps, dcf=dcf,
        profile=profile if profile is not None else make_profile(memo.ticker),
        findings=findings if findings is not None else make_findings(),
        ticker=memo.ticker,
    )


def _evidence(**kw):
    base = dict(family_pct=None, family_coverage=None, comps_premium=None,
                dcf_initial_upside=None, dcf_final_upside=None)
    base.update(kw)
    return memo_quality.valuation_evidence_verdict(**base)


@pytest.mark.parametrize("rating", ["Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish"])
def test_build_verdict_word_follows_evidence(rating):
    """W2b 7(b): the verdict stage passes the compose-stage evidence verdict
    through; the rating never rewrites it (it used to derive it)."""
    vv = _evidence(comps_premium=0.44, dcf_initial_upside=-0.54, dcf_final_upside=-0.30)
    memo = make_memo(rating_label=rating, dcf_summary={"base_upside": -0.30}, valuation_verdict=vv)
    out = _verdict_for(memo)
    assert out.valuation_verdict is vv
    assert out.valuation_verdict.verdict == "overvalued"
    assert out.render(memo).startswith(f"PM final view: {rating} (confidence 60)")


def test_evidence_dcf_number_is_the_dcf_summary_number():
    """The verdict's printed DCF is the final (dcf_summary) number; the
    initial one is what votes."""
    vv = _evidence(dcf_initial_upside=0.05, dcf_final_upside=0.173)
    assert vv.dcf_base_upside == 0.173
    assert "+17%" in vv.summary and "+5%" in vv.summary


def test_evidence_none_dcf_upside_says_unavailable():
    """Phase 2: an unpriced DCF is "n/a", never a 0% neutral signal."""
    vv = _evidence(comps_premium=-0.30)
    assert vv.dcf_base_upside is None
    assert "DCF unavailable" in vv.summary
    assert not any(z in vv.summary for z in _ZERO_LIES)
    # One cheap vote is not a verdict.
    assert vv.verdict == "fairly_priced"


def test_build_verdict_mispricing_fallback_is_nonempty_and_quotes_final_thesis():
    memo = make_memo(
        rating_label="Neutral", one_sentence_thesis="TEST is fairly priced — steady compounder.",
    )
    assert memo.mispricing_thesis == MispricingThesis()
    out = _verdict_for(memo)
    m = out.mispricing_thesis
    assert m.consensus_view and m.our_view and m.gap
    assert "no material mispricing" in m.gap.lower()
    assert m.our_view == out.one_sentence_thesis
    assert m.falsifiers == ["Cloud slowdown"]  # thesis breakers lead the falsifier list


def test_build_verdict_keeps_a_populated_mispricing_thesis():
    pm_thesis = MispricingThesis(consensus_view="Street sees 10%.", our_view="We see 15%.", gap="5pp.")
    memo = make_memo(rating_label="Bullish", mispricing_thesis=pm_thesis)
    out = _verdict_for(memo)
    assert out.mispricing_thesis is pm_thesis


def test_build_verdict_rewrites_the_anti_pattern_thesis():
    anti = "TEST Corp — Technology / Software, AI hook; DCF base case +25% suggests material upside."
    memo = make_memo(rating_label="Bullish", one_sentence_thesis=anti)
    out = _verdict_for(memo)
    assert out.thesis_rewrite_fired
    assert out.one_sentence_thesis != anti
    assert not graph._looks_like_anti_pattern_thesis(out.one_sentence_thesis)
    assert "undervalued" in out.one_sentence_thesis
    assert out.one_sentence_thesis in out.final_verdict_body


def test_build_verdict_rewrites_a_thesis_whose_verdict_word_contradicts_the_rating():
    """The thesis was written pre-blend; a risk-rec downgrade or the
    factor blend can move the rating after it, so the stated word must
    be re-checked against the FINAL rating."""
    memo = make_memo(rating_label="Bearish", one_sentence_thesis="TEST is undervalued — great franchise.")
    out = _verdict_for(memo)
    assert out.thesis_rewrite_fired
    assert "overvalued" in out.one_sentence_thesis
    assert "undervalued" not in out.one_sentence_thesis


def test_build_verdict_leaves_a_consistent_thesis_alone():
    memo = make_memo(rating_label="Bullish", one_sentence_thesis="TEST is undervalued — cloud share gains.")
    out = _verdict_for(memo)
    assert not out.thesis_rewrite_fired
    assert out.one_sentence_thesis == memo.one_sentence_thesis
    assert out.degradations == []


def test_build_verdict_is_pure():
    """Nothing on the memo changes until `VerdictOutcome.apply` runs."""
    memo = make_memo(
        rating_label="Bullish",
        one_sentence_thesis="TEST — Tech / Software, hook; DCF base case +25%.",
    )
    before = memo.model_dump()
    out = _verdict_for(memo)
    assert memo.model_dump() == before
    assert memo.valuation_verdict == ValuationVerdict()
    assert memo.final_verdict == ""
    out.apply(memo, DegradationLog())
    assert memo.valuation_verdict is out.valuation_verdict
    assert memo.one_sentence_thesis == out.one_sentence_thesis
    assert memo.mispricing_thesis is out.mispricing_thesis
    # The confidence-bearing verdict is rendered once, after the quality
    # stage (`graph._render_final_texts`), never by `apply`.
    assert memo.final_verdict == ""
    assert out.render(memo) == f"PM final view: Bullish (confidence 60). {out.final_verdict_body}"


def test_build_verdict_cross_sector_relevance_rides_on_scores():
    findings = make_findings()
    findings["sector"].data = {"cross_sector_relevance": ["AMD", "AVGO"], "kpi_placements": {"x": 1}}
    memo = make_memo(rating_label="Neutral")
    out = _verdict_for(memo, findings=findings)
    assert out.extra_scores == {"cross_sector_relevance_count": 2.0}
    assert "Cross-sector pull-through: AMD, AVGO." in out.final_verdict_body
    assert "Cohort placement" in out.final_verdict_body
    out.apply(memo, DegradationLog())
    assert memo.scores["cross_sector_relevance_count"] == 2.0
    assert memo.scores["factor_pm_score"] == 55.0  # existing scores kept


def test_compose_reports_a_valuation_verdict_crash_instead_of_hiding_it(monkeypatch):
    """W2b: the verdict is computed in the compose stage now (before the
    PM), so its crash guard lives there. A crash is a hard banner entry and
    a placeholder card that says so — never a silent empty card, and never
    a card that reads as a rating-derived verdict."""
    from app.agents.intake import IntakeDecision
    from app.agents.memo_context import AnalystRound, DCFStage
    from app.tests.factories import make_inputs

    def boom(*a, **k):
        raise RuntimeError("verdict exploded")

    monkeypatch.setattr(memo_quality, "valuation_evidence_verdict", boom)
    inputs = make_inputs()
    memo = graph._compose_memo(
        inputs, AnalystRound(findings=make_findings(), intake=IntakeDecision()),
        DCFStage(dcf=None, initial_dcf=None),
    )
    assert memo.valuation_verdict.basis == "evidence"
    assert memo.valuation_verdict.summary == memo_quality.VERDICT_UNAVAILABLE_SUMMARY
    assert "Valuation Verdict" in inputs.degradation.degraded_agents()
    event = next(e for e in inputs.degradation.events() if e["agent"] == "Valuation Verdict")
    assert event["error_type"] == "RuntimeError" and "verdict exploded" in event["message"]


def test_build_verdict_soft_notes_dedupe_on_apply(monkeypatch):
    """A second "Thesis Builder" note must not double the banner entry —
    the same guarantee `record_soft` gave when the stage wrote to the log
    directly."""
    def boom(*a, **k):
        raise RuntimeError("gap clause exploded")

    monkeypatch.setattr(graph, "_market_gap_clause", boom)
    memo = make_memo(rating_label="Bullish", one_sentence_thesis="TEST is undervalued — fine.")
    out = _verdict_for(memo)
    assert [(n.agent, n.soft) for n in out.degradations] == [("Thesis Builder", True)]
    log = DegradationLog()
    log.record_soft("Thesis Builder", "earlier note from PM synthesis")
    out.apply(memo, log)
    assert log.degraded_agents() == ["Thesis Builder"]


def test_verdict_outcome_records_rewrite_and_mispricing_provenance(monkeypatch):
    """W2a §6: the two facts no stored text can recover are written at
    verdict time. `thesis_rewritten` is set only when the builder's text
    actually replaced the PM's (the guard firing is not enough: a rejected
    rewrite leaves the PM's words), and `mispricing_fallback` only when the
    fallback built the card."""
    # The saved anti-pattern counterexample: rewrite fires and replaces.
    anti = "TEST Corp — Technology / Software, AI hook; DCF base case +25% suggests material upside."
    memo = make_memo(rating_label="Bullish", one_sentence_thesis=anti,
                     section_provenance={"v": 1, "llm_configured": True})
    out = _verdict_for(memo)
    assert out.thesis_rewrite_fired and out.thesis_rewritten
    assert out.mispricing_fallback  # the PM left the card blank
    out.apply(memo, DegradationLog())
    assert memo.section_provenance == {
        "v": 1, "llm_configured": True, "thesis": "rewrite", "mispricing": "fallback",
    }

    # A consistent PM thesis and a populated PM card: nothing template.
    pm_card = MispricingThesis(consensus_view="Street sees 10%.", our_view="We see 15%.", gap="5pp.")
    clean = make_memo(rating_label="Bullish", mispricing_thesis=pm_card,
                      one_sentence_thesis="TEST is undervalued — cloud share gains.")
    out = _verdict_for(clean)
    assert not out.thesis_rewrite_fired and not out.thesis_rewritten
    assert not out.mispricing_fallback
    out.apply(clean, DegradationLog())
    assert clean.section_provenance == {"thesis": "pm", "mispricing": "pm"}

    # The guard fires but the rewrite is rejected (it would itself be the
    # anti-pattern): the PM's thesis stands, so it is not a rewrite.
    monkeypatch.setattr(graph, "_build_thesis_from_findings", lambda *a, **k: anti)
    fired = make_memo(rating_label="Bearish", one_sentence_thesis="TEST is undervalued — great franchise.")
    out = _verdict_for(fired)
    assert out.thesis_rewrite_fired and not out.thesis_rewritten
    assert out.one_sentence_thesis.startswith("TEST is undervalued — great franchise")


# ---------------------------------------------------------------------------
# Phase 6 — the scorecard informs the memo; it does not move the rating
# ---------------------------------------------------------------------------

from app.schemas import StockMemoOut  # noqa: E402


def _scorecard_fixture(pct: float = 8.0):
    from datetime import date

    from app.schemas import ScorecardCategory, ScorecardContribution, ScorecardSummary
    return ScorecardSummary(
        version_key="fs-v1", as_of=date(2026, 6, 30), overall_z=-1.2, overall_score=26.0,
        universe_percentile=pct, sector_percentile=12.0, coverage=0.9,
        categories={"valuation": ScorecardCategory(z=-1.4, score=22.0, percentile=6.0, weight=0.125)},
        top_negative=[ScorecardContribution(feature="accruals_ratio", family="earnings_quality", z=-1.8, contribution=-0.05)],
        profiles={"compounder": -0.4, "inflection": None},
    )


def test_memo_validates_with_scorecard_none_and_with_a_summary():
    memo = make_memo()
    assert memo.scorecard is None
    payload = memo.model_dump(mode="json")
    assert payload["scorecard"] is None
    assert StockMemoOut.model_validate(payload).scorecard is None
    # An older snapshot without the key at all.
    payload.pop("scorecard")
    assert StockMemoOut.model_validate(payload).scorecard is None
    with_row = make_memo(scorecard=_scorecard_fixture())
    round_trip = StockMemoOut.model_validate(with_row.model_dump(mode="json"))
    assert round_trip.scorecard is not None
    assert round_trip.scorecard.universe_percentile == 8.0


@pytest.mark.parametrize("rating", ["Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish"])
@pytest.mark.parametrize("factor_pm", [15.0, 40.0, 62.0, 88.0])
def test_rating_blend_ignores_the_scorecard(rating, factor_pm):
    """Behavior preservation: `_blend_rating` reads only `rating_label` and
    `scores["factor_pm_score"]`. A bottom-decile scorecard on a Very Bullish
    memo changes nothing about the blended rating or its audit fields."""
    without = make_memo(rating_label=rating, scores={"factor_pm_score": factor_pm})
    with_row = make_memo(rating_label=rating, scores={"factor_pm_score": factor_pm}, scorecard=_scorecard_fixture())
    graph._blend_rating(without)
    graph._blend_rating(with_row)
    assert with_row.rating_label == without.rating_label
    assert with_row.scores == without.scores
    assert {"llm_rating_score", "llm_rating_weight", "blended_pm_score"} <= set(with_row.scores)


def test_attach_scorecard_disagreement_does_not_touch_the_rating():
    memo = make_memo(rating_label="Very Bullish", scores={"factor_pm_score": 88.0}, scorecard=_scorecard_fixture())
    graph._blend_rating(memo)
    rating_after_blend, scores_after_blend = memo.rating_label, dict(memo.scores)
    graph._attach_scorecard_disagreement(memo)
    assert memo.scorecard is not None and memo.scorecard.disagreement is not None
    assert memo.scorecard.disagreement.severity == "material"
    assert memo.rating_label == rating_after_blend
    assert memo.scores == scores_after_blend
    # A finding, not an outage: nothing lands on the banner.
    assert memo.degraded_agents == []
