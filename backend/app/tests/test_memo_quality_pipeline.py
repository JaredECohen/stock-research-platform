"""W2b 7(a) through the memo pipeline: source ledger -> number check ->
flag / withhold -> confidence caps, on demo runs (no keys, no network).

The registry these tests judge against is the one the REAL ledger builds
during the run (`_capture_registry`), never a hand-built stand-in, so a
registration site that goes missing shows up here as untraceable figures.
"""
from __future__ import annotations

import json
import re

import pytest

from app.agents import graph, number_check, prompts
from app.agents import llm as llm_mod
from app.config import Settings, settings
from app.schemas import BullBearCase, RiskItem

FLAGGED = number_check.FLAGGED_STATUSES


def _capture_registry(monkeypatch) -> dict:
    """Keep the registry `_check_numbers` judged the memo against."""
    box: dict = {}
    real = graph._check_numbers

    def spy(memo, inputs, notes):
        box["registry"] = inputs.ledger.snapshot()
        return real(memo, inputs, notes)

    monkeypatch.setattr(graph, "_check_numbers", spy)
    return box


def _live_looking(monkeypatch, *, pm: dict | None = None, answers=None) -> None:
    """has_llm on (keys stay blank), every model call answers nothing unless
    `answers(prompt)` returns something, and the PM says `pm`."""
    monkeypatch.setattr(Settings, "has_llm", property(lambda self: True))

    def chat(prompt, **kw):
        return answers(prompt) if answers else None

    monkeypatch.setattr(llm_mod, "chat_json", chat)
    if pm is not None:
        monkeypatch.setattr(graph, "_pm_synthesis", lambda *a, **kw: dict(pm))


def _assert_offsets(memo) -> None:
    nc = memo.quality.number_check
    for c in nc.claims:
        text = number_check.resolve_field(memo, c.field)
        assert isinstance(text, str) and text[c.start:c.end] == c.raw, c
    for w in nc.withheld:
        for c in w.claims:
            assert w.text[c.start:c.end] == c.raw, (w, c)


# ---------------------------------------------------------------------------
# The CI invariant: code-generated prose has no untraceable figure
# ---------------------------------------------------------------------------

# The primary kinds each demo memo cites. A registration site that goes
# missing (the filing analyst's MD&A and chunks are the only source of 10-K
# figures in live runs) loses its kind here even when no figure goes
# untraceable, because the financials often carry the same number. XOM's
# demo run has no transcript or filing text.
PRIMARY_KINDS_BY_TICKER = {
    "NVDA": ["filing", "financials", "transcript"], "JPM": ["filing", "financials", "transcript"],
    "XOM": ["financials"], "AMT": ["filing", "financials", "transcript"],
    "COST": ["filing", "financials", "transcript"],
}


@pytest.mark.parametrize("ticker", sorted(PRIMARY_KINDS_BY_TICKER))
def test_deterministic_memo_has_no_untraceable_figures(ticker):
    """Every figure a keyless demo memo prints is written by code from
    registered data. A new deterministic sentence quoting an unregistered
    number fails here — the registry-completeness invariant."""
    memo = graph.run_stock_memo(ticker)
    nc = memo.quality.number_check
    assert nc is not None and nc.checked is True
    c = nc.counts
    assert c["claims_total"] >= 30
    assert c["untraceable"] == c["mis_anchored"] == c["weak"] == 0, nc.claims
    assert nc.withheld == [] and nc.lists_not_withheld == []
    assert nc.primary_kinds_cited == PRIMARY_KINDS_BY_TICKER[ticker]
    assert nc.sources_cited and all(":" in s for s in nc.sources_cited)
    # An untraceable figure is a finding about the memo, not an outage.
    assert "Number Check" not in memo.degraded_agents


# ---------------------------------------------------------------------------
# Known-bad figures and false support, on the real ledger's registry
# ---------------------------------------------------------------------------

def _status(text: str, registry, raw: str) -> str:
    got = [c for c in number_check.check_text(text, registry) if c.claim.raw == raw]
    assert got, (text, raw)
    return got[0].status


def test_known_bad_figures_flagged_on_real_registry(monkeypatch):
    """The critique's three demonstrated errors, in their original shapes,
    against registries the real ledger built on demo runs: GOOGL's "$66
    gap" (the bull/base gap was $92), META's "6.7 percentage points" (a
    6.7% RELATIVE gap, 2.6 points) and a stale "3.9x" bull/bear ratio."""
    box = _capture_registry(monkeypatch)
    googl = graph.run_stock_memo("GOOGL")
    reg = box["registry"]
    d = googl.dcf_summary
    bull, base = d["bull_implied_price"], d["base_implied_price"]
    true_gap = bull - base
    assert abs(true_gap - 66) > 0.5
    # The sentence as GOOGL v6 printed it, with this registry's bull and base.
    text = (f"The bull ${bull:.0f} vs base ${base:.0f} spread is driven almost entirely by terminal "
            "growth: moving g from ~3% to ~4-4.5% and shaving WACC to ~8.5% roughly explains the $66 gap.")
    assert _status(text, reg, "$66") in FLAGGED
    assert _status(text.replace("$66", f"${true_gap:.0f}"), reg, f"${true_gap:.0f}") == "traced"

    meta = graph.run_stock_memo("META")
    reg = box["registry"]
    text = "Its operating margin sits 6.7 percentage points above its peer median."
    assert _status(text, reg, "6.7 percentage points") in FLAGGED
    d = meta.dcf_summary
    ratio = d["bull_implied_price"] / d["bear_implied_price"]
    assert abs(ratio - 3.9) > 0.05
    stale = "The 3.9x bull/bear ratio frames the discount-rate sensitivity."
    assert _status(stale, reg, "3.9x") in FLAGGED
    assert _status(stale.replace("3.9x", f"{ratio:.1f}x"), reg, f"{ratio:.1f}x") == "traced"


def test_value_first_multiples_trace_on_real_registry(monkeypatch):
    """Review regression: the commonest ways of writing multiples and
    margins read mis_anchored against the real NVDA registry ("x earnings"
    anchored to nothing, "x sales" to revenue, "46.8x P/E, 34.4x EV/EBITDA"
    gave each value the PREVIOUS metric), and a correct P/E point was
    withheld from the bull case."""
    box = _capture_registry(monkeypatch)
    graph.run_stock_memo("NVDA")
    reg = box["registry"]
    for text in (
        "Shares trade at 22.3x sales.", "Shares trade at 22.3x revenue.",
        "The stock trades at 34.4x EBITDA.", "Trading at 46.8x earnings, a premium to peers.",
        "At 46.8x earnings and 22.3x sales, NVDA is priced for perfection.",
        "Valuation: 46.8x P/E, 34.4x EV/EBITDA, 49.6x P/FCF.",
        "NVDA runs a 61% operating margin and 97.6% gross margins.",
        "NVDA trades at 46.8x earnings.",
    ):
        got = [(c.claim.raw, c.status) for c in number_check.check_text(text, reg)]
        assert got and all(st == "traced" for _, st in got), (text, got)

    from app.tests.factories import make_memo
    points = ["Trades at 46.8x earnings, a premium to peers, justified by growth.",
              "Operating margin of 61% underlines pricing power.", "Tailwind: networking attach"]
    memo = make_memo(bull_case=BullBearCase(headline="Bull", key_points=points))
    result = number_check.check_memo(memo, reg, withhold=True)
    assert "bull_case.key_points" not in result.plan.items
    bull = [c.status for fr in result.fields if fr.spec.path.startswith("bull_case.key_points")
            for c in fr.claims]
    assert bull == ["traced", "traced"]


def test_false_support_rate_on_real_registry(monkeypatch):
    """Critique delta 2: fabricate values at the printed precision for every
    figure a demo memo traces and re-check them in place. Measured on the
    real ledger at 3-7% per memo (5.5% over eight tickers; bare value
    matching was 60-87% on the prototype); the bound guards a regression."""
    box = _capture_registry(monkeypatch)
    memo = graph.run_stock_memo("NVDA")
    reg = box["registry"]
    n = still = 0
    for spec in number_check.iter_fields(memo):
        facts = [c for c in number_check.extract_claims(spec.text) if c.cls == "fact"]
        for c, ctx in zip(facts, number_check.claim_contexts(spec.text, facts)):
            if c.value == 0 or number_check.classify(c, ctx, reg)[0] != "traced":
                continue
            printed = c.value / (0.01 if c.unit == "pp" and c.scale == 0.01 else c.scale or 1.0)
            m = re.search(r"\d[\d,]*(?:\.\d+)?", c.raw)
            for f in (0.62, 0.78, 1.27, 1.45, 1.9):
                fake = f"{printed * f:.{c.decimals}f}"
                if float(fake) == round(printed, c.decimals):
                    continue
                new_text = spec.text[:c.start] + c.raw.replace(m.group(0), fake, 1) + spec.text[c.end:]
                again = [x for x in number_check.extract_claims(new_text) if x.cls == "fact"]
                target = next(x for x in again if x.start == c.start)
                ctx2 = number_check.claim_contexts(new_text, again)[again.index(target)]
                n += 1
                still += number_check.classify(target, ctx2, reg)[0] == "traced"
    assert n >= 150
    assert still / n <= 0.10, f"false support {still}/{n}"


# ---------------------------------------------------------------------------
# Flag, withhold, guard
# ---------------------------------------------------------------------------

PM_FAKE = {
    "rating_label": "Neutral", "confidence_score": 88,
    "final_pm_view": ("Data-center revenue of $123.45B, a 37.7% operating margin and 88.8x EV/EBITDA "
                      "frame the call."),
    "one_sentence_thesis": "NVDA is fairly priced — share gains are in the multiple.",
    "mispricing_thesis": {"consensus_view": "The market pays up for share gains.",
                          "our_view": "Fairly priced on our work.", "gap": "No actionable gap.",
                          "falsifiers": ["Revenue growth below 12.3% for two quarters"]},
}


def test_fabricated_pm_figures_are_flagged(monkeypatch):
    _live_looking(monkeypatch, pm=PM_FAKE)
    memo = graph.run_stock_memo("NVDA")
    nc = memo.quality.number_check
    flagged = {c.raw: c for c in nc.claims if c.field == "final_pm_view" and c.status in FLAGGED}
    assert set(flagged) == {"$123.45B", "37.7%", "88.8x"}
    assert nc.counts["flagged_distinct"] >= 3
    codes = {c.code: c.cap for c in memo.quality.confidence.caps}
    assert codes["untraceable_figures"] == 70.0
    # Recorded in quality, not on the degraded-agents banner.
    assert "Number Check" not in memo.degraded_agents
    assert not any(e.get("error_type") == "UntraceableNumbers" for e in memo.degradation_events)
    # Falsifiers are forward conditions: counted, never checked or stored.
    assert nc.counts["threshold"] >= 1
    assert not any(c.field.startswith("mispricing_thesis.falsifiers") for c in nc.claims)


def test_claim_offsets_survive_prefix(monkeypatch):
    """The review stages move confidence, so the PM view gains its "Final
    rating after ..." preface AFTER the check located the figures."""
    _live_looking(monkeypatch, pm=PM_FAKE)
    memo = graph.run_stock_memo("NVDA")
    assert memo.final_pm_view.startswith("Final rating after")
    _assert_offsets(memo)
    c = next(c for c in memo.quality.number_check.claims if c.raw == "$123.45B")
    assert c.start > len("Final rating after")


def _bull(points):
    return lambda *a, **k: BullBearCase(headline="Bull case: durable execution.", key_points=list(points))


def _bear(points):
    return lambda *a, **k: BullBearCase(headline="Bear case: the cycle turns.", key_points=list(points))


BULL = ["Tailwind: Hyperscaler AI capex", "Services attach reached 91.7% last year",
        "Tailwind: Sovereign AI projects", "Tailwind: Networking + software stack"]
# All five bad: the review stage appends up to two risk-lens points, so
# this list is still mostly bad when the check sees it.
BEAR = ["Channel inventory hit 94.1 days", "Pricing fell 93.2% in China",
        "Share loss of 95.3 points to custom silicon", "Gross margin fell to 31.9%",
        "Backlog shrank to $87.6B"]


def test_withhold_supporting_point_and_half_list_guard(monkeypatch):
    monkeypatch.setattr(graph, "_bull_case", _bull(BULL))
    monkeypatch.setattr(graph, "_bear_case", _bear(BEAR))
    memo = graph.run_stock_memo("NVDA")
    nc = memo.quality.number_check
    # One bad point in four: withheld, verbatim, with its original index.
    (w,) = nc.withheld
    assert (w.field, w.index, w.text) == ("bull_case.key_points", 1, BULL[1])
    assert [c.raw for c in w.claims] == ["91.7%"]
    assert memo.bull_case.key_points == [BULL[0], BULL[2], BULL[3]]
    # Three bad points in four: likelier a registry gap — flag, keep all.
    assert memo.bear_case.key_points[:5] == BEAR
    assert nc.lists_not_withheld == ["bear_case.key_points"]
    assert {c.field for c in nc.claims if c.status in FLAGGED} >= {
        f"bear_case.key_points[{i}]" for i in range(5)}
    assert nc.counts["withheld"] == 1
    _assert_offsets(memo)


def test_withhold_kill_switch(monkeypatch):
    monkeypatch.setattr(settings, "number_check_withhold", False)
    monkeypatch.setattr(graph, "_bull_case", _bull(BULL))
    memo = graph.run_stock_memo("NVDA")
    nc = memo.quality.number_check
    assert nc.withheld == [] and memo.bull_case.key_points == BULL
    assert any(c.field == "bull_case.key_points[1]" and c.status in FLAGGED for c in nc.claims)


def test_risks_never_withheld(monkeypatch):
    """Downside information is flagged, never removed."""
    risks = [RiskItem(title="Leverage", detail="Debt rose to 77.7x EBITDA.", severity="high"),
             RiskItem(title="Export rules", detail="Restrictions tighten.", severity="medium"),
             RiskItem(title="Customer concentration", detail="Top buyers dominate.", severity="medium")]
    monkeypatch.setattr(graph, "derive_risk_items", lambda profile: list(risks))
    memo = graph.run_stock_memo("NVDA")
    nc = memo.quality.number_check
    assert memo.key_risks[0].detail == "Debt rose to 77.7x EBITDA."
    assert not [w for w in nc.withheld if w.field in ("key_risks", "thesis_breakers")]
    assert any(c.field == "key_risks[0].detail" and c.status in FLAGGED for c in nc.claims)


def _valuation_answer(prompt: str):
    if prompt.startswith(prompts.VALUATION_ANALYST_PROMPT):
        return {"headline": "Valuation view", "summary": "Multiples look full against the DCF.",
                "key_points": ["P/E 46.8x", "EV/EBITDA 34.4x", "Hidden optionality worth $987.65B",
                               "FCF yield 2.0%"],
                "confidence": 0.7}
    return None


def test_withheld_text_appears_only_in_quality(monkeypatch):
    """Critique delta 4: a withheld point must not survive in the analyst's
    long-form "Key points" block or in the deep-research audit trail."""
    _live_looking(monkeypatch, answers=_valuation_answer)
    memo = graph.run_stock_memo("NVDA")
    nc = memo.quality.number_check
    bad = "Hidden optionality worth $987.65B"
    assert [(w.field, w.text) for w in nc.withheld] == [("valuation_agent_view.key_points", bad)]
    dump = memo.model_dump(mode="json")
    assert "987.65" in json.dumps(dump["quality"])
    dump.pop("quality")
    assert "987.65" not in json.dumps(dump)
    view = memo.valuation_agent_view
    assert view.key_points == ["P/E 46.8x", "EV/EBITDA 34.4x", "FCF yield 2.0%"]
    assert "- EV/EBITDA 34.4x" in view.long_form_report and "### Key points" in view.long_form_report
    _assert_offsets(memo)


def test_withheld_point_scrubbed_from_a_separate_round_copy(monkeypatch):
    """Round 0 keeps the analyst's ORIGINAL finding object when a deep-
    research re-fire replaces the view (deep_research.py), so the audit
    trail can hold a copy the view-level removal never touches."""
    _live_looking(monkeypatch, answers=_valuation_answer)
    real = graph._check_numbers
    copies = []

    def spy(memo, inputs, notes):
        r0 = memo.round_findings[0].findings
        r0["valuation"] = r0["valuation"].model_copy(deep=True)
        copies.append(r0["valuation"])
        return real(memo, inputs, notes)

    monkeypatch.setattr(graph, "_check_numbers", spy)
    memo = graph.run_stock_memo("NVDA")
    bad = "Hidden optionality worth $987.65B"
    (copy,) = copies
    assert copy is not memo.valuation_agent_view
    assert bad not in copy.key_points
    assert "987.65" not in (copy.long_form_report or "")
    dump = memo.model_dump(mode="json")
    dump.pop("quality")
    assert "987.65" not in json.dumps(dump)


def test_research_notes_are_registered():
    """The one choke point every note reaches an agent through registers
    the rendered block as a non-primary `research_note` source."""
    from app.agents.source_ledger import SourceLedger
    from app.services import research_notes as rn

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(rn, "select_for", lambda *a, **k: ["n"])
        mp.setattr(rn, "select_bodies", lambda *a, **k: [])
        mp.setattr(rn, "render_summary_block", lambda s: "Note: cloud margin reached 43.21% last year.")
        mp.setattr(rn, "render_body_block", lambda e: "")
        ledger = SourceLedger()
        with ledger.activate():
            block = rn.build_notes_block_for_agent("valuation", {"ticker": "T"})
    finally:
        mp.undo()
    assert "43.21%" in block
    reg = ledger.snapshot()
    assert reg.resolves("notes:valuation")
    (c,) = number_check.check_text("Cloud margin reached 43.21% last year.", reg)
    assert c.status == "traced" and c.kinds == ("research_note",)


def test_forecast_assumptions_capped_at_five():
    from app.agents import memo_quality
    good = [{"value": 10 + i, "unit": "pct", "basis_ref": "financials:T", "horizon": "FY2027"}
            for i in range(7)]
    raw = [{"value": "x", "unit": "pct", "basis_ref": "r", "horizon": "h"},
           {"value": 50, "unit": "bps", "basis_ref": "financials:T", "horizon": "FY2027"},
           {"value": 5, "unit": "furlongs", "basis_ref": "r", "horizon": "h"},
           {"value": 5, "unit": "pct", "basis_ref": "", "horizon": "h"}, *good]
    got = memo_quality.parse_forecast_assumptions(raw)
    assert memo_quality.MAX_FORECAST_ASSUMPTIONS == 5 and len(got) == 5
    assert got[0] == {"value": 0.5, "unit": "pp", "basis_ref": "financials:T", "horizon": "FY2027"}
    assert [a["value"] for a in got[1:]] == [10, 11, 12, 13]


def test_reflection_sees_the_checked_memo(monkeypatch):
    seen = {}
    monkeypatch.setattr(graph, "_run_reflection_step", lambda memo: seen.setdefault("memo", memo) and ([], []))
    monkeypatch.setattr(graph, "_bull_case", _bull(BULL))
    memo = graph.run_stock_memo("NVDA")
    reflected = seen["memo"]
    assert BULL[1] not in reflected.bull_case.key_points
    assert reflected.confidence_score == memo.quality.confidence.final


# ---------------------------------------------------------------------------
# Declared assumptions, sources, resume
# ---------------------------------------------------------------------------

def test_declared_assumption_labelled_not_untraceable(monkeypatch):
    pm = dict(PM_FAKE, final_pm_view=(
        "We assume revenue growth of 18.5% through FY2027 and a 41.5x exit multiple."),
        forecast_assumptions=[
            {"value": 18.5, "unit": "pct", "basis_ref": "financials:NVDA", "horizon": "FY2027"},
            {"value": 41.5, "unit": "multiple", "basis_ref": "made_up:ref", "horizon": "FY2027"},
            "not an object",
        ])
    real = graph._compose_memo

    def compose(inputs, analysts, dcf_stage):
        memo = real(inputs, analysts, dcf_stage)
        return memo

    _live_looking(monkeypatch)
    monkeypatch.setattr(graph, "_pm_synthesis", lambda *a, **kw: dict(pm))
    monkeypatch.setattr(graph, "_compose_memo", compose)
    memo = graph.run_stock_memo("NVDA")
    nc = memo.quality.number_check
    by_raw = {c.raw: c.status for c in nc.claims if c.field == "final_pm_view"}
    assert by_raw["18.5%"] == "assumption"
    assert by_raw["41.5x"] in FLAGGED
    assert nc.counts["assumption"] == 1
    assert [(a["value"], a["basis_ref"], a["status"]) for a in nc.assumptions] == [
        (18.5, "financials:NVDA", "assumption")]
    assert any("made_up:ref" in n for n in nc.notes)


def test_pm_prompt_lists_source_refs_and_asks_for_assumptions(monkeypatch):
    seen: list[str] = []

    def answers(prompt):
        if prompt.startswith(prompts.PM_SYNTHESIS_PROMPT):
            seen.append(prompt)
        return None

    _live_looking(monkeypatch, answers=answers)
    graph.run_stock_memo("NVDA")
    (prompt,) = seen
    volatile = prompt[len(prompts.PM_SYNTHESIS_PROMPT):]
    assert "## Source refs (for forecast_assumptions basis_ref)" in volatile
    assert "financials:NVDA" in volatile.split("Findings:")[0]
    text = " ".join(prompts.PM_SYNTHESIS_PROMPT.split())
    assert "FORECAST ASSUMPTIONS." in text and "forecast_assumptions" in text
    assert "at most 5" in text


def test_quality_contains_no_gics_code(monkeypatch):
    """Owner decision 10: no industry codes on public surfaces. The group
    analyst's inputs are registered under `industry_group:{slug}`, and
    nothing in `quality` (refs, claims, notes) carries a code."""
    from app.agents import industry_analysts as ia
    from app.services import gics_registry as reg
    from app.services import industry_classification as ic
    from app.tests.gating_helpers import seed_demo_universe
    from app.tests.test_memo_industry_labels import _leaks

    seed_demo_universe()
    assert reg.ensure_taxonomy(activate=True) is not None
    ic.classify_all(tickers=["MSFT"])
    ia.clear_cache()
    monkeypatch.setattr(settings, "enable_industry_analyst_routing", True)
    row = ic.current_for(["MSFT"])["MSFT"]
    analyst = ia.analyst_for_classification(row)
    assert analyst is not None
    box = _capture_registry(monkeypatch)
    try:
        memo = graph.run_stock_memo("MSFT")
    finally:
        ia.clear_cache()
    refs = set(box["registry"].sources)
    assert f"industry_group:{analyst.slug}" in refs
    codes = {str(row.get(k)) for k in ("sector_code", "industry_group_code", "industry_code",
                                       "sub_industry_code") if row.get(k)}
    assert codes
    for ref in refs:
        assert not any(re.search(rf"(?<!\d){c}(?!\d)", ref) for c in codes), ref
    assert _leaks(memo.quality.model_dump(mode="json"), row, analyst) == []


def test_prior_extraction_supports_qoq_figures(monkeypatch):
    """The QoQ tile compares two structured extractions; they are its
    inputs, registered as the labelled non-primary `prior_extraction`: a
    figure traced there never counts as a filing or a transcript."""
    from app.agents import earnings_qoq
    from app.agents.source_ledger import SourceLedger
    from app.tests.factories import make_memo

    prior = {"period": "2025Q2", "guidance_changes": [{"metric": "gross margin", "direction": "raised",
                                                       "detail": "gross margin guide raised to 45.5%"}]}
    current = {"period": "2025Q3", "guidance_changes": [{"metric": "gross margin", "direction": "lowered",
                                                         "detail": "gross margin guide cut to 42.5%"}]}
    monkeypatch.setattr(earnings_qoq, "_prior_structured_extraction", lambda t, p: prior)
    monkeypatch.setattr(earnings_qoq, "_llm_delta", lambda c, p, t: {
        "reversals": ["Gross margin guide cut to 42.5% after being raised to 45.5%"],
        "net_signal": "worse", "one_line_takeaway": "Guidance reversed."})
    ledger = SourceLedger()
    with ledger.activate():
        finding = earnings_qoq.run_earnings_qoq_delta("NVDA", current)
    assert finding is not None
    memo = make_memo(earnings_qoq_delta=finding)
    result = number_check.check_memo(memo, ledger.snapshot(), withhold=True)
    qoq = [c for fr in result.fields if fr.spec.path.startswith("earnings_qoq_delta") for c in fr.claims]
    assert {c.claim.raw: c.status for c in qoq} == {"42.5%": "traced", "45.5%": "traced"}
    assert all(set(c.kinds) == {"prior_extraction"} for c in qoq)
    nc = number_check.summarize(result, assumptions=[], notes=[])
    assert "prior_extraction" not in nc.primary_kinds_cited
    assert {"prior_extraction:2025Q2", "prior_extraction:2025Q3"} <= set(nc.sources_cited)


def test_resume_replays_sources_and_unchecked_when_missing():
    """A resumed run (same run_id) loads each analyst's finding from its
    checkpoint and replays the facts it registered: the check is complete.
    With a roster row missing its sources, figures are reported unchecked
    (cap `figures_unchecked`), never flagged against a partial registry."""
    from app.database import SessionLocal
    from app.models import MemoRunCheckpoint

    first = graph.run_stock_memo("NVDA", run_id="w2b-resume-1")
    resumed = graph.run_stock_memo("NVDA", run_id="w2b-resume-1")
    a, b = first.quality.number_check, resumed.quality.number_check
    assert a.checked and b.checked
    assert b.counts["untraceable"] == 0 and b.counts["traced"] == a.counts["traced"]

    with SessionLocal() as db:
        row = db.query(MemoRunCheckpoint).filter_by(
            run_id="w2b-resume-1", step_name="graph.valuation_finding").one()
        row.sources = None
        db.commit()
    stale = graph.run_stock_memo("NVDA", run_id="w2b-resume-1")
    nc = stale.quality.number_check
    assert nc.checked is False and "graph.valuation_finding" in nc.notes[0]
    assert nc.claims == [] and nc.withheld == []
    assert "figures_unchecked" in {c.code for c in stale.quality.confidence.caps}


def test_number_check_crash_is_unchecked_and_on_the_banner(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(number_check, "check_memo", boom)
    memo = graph.run_stock_memo("NVDA")
    nc = memo.quality.number_check
    assert nc.checked is False and "crashed" in nc.notes[0]
    assert "Number Check" in memo.degraded_agents
    assert "figures_unchecked" in {c.code for c in memo.quality.confidence.caps}


def test_apply_withholding_renumbers_later_claims():
    from app.schemas import NumberCheck, NumberClaim
    from app.tests.factories import make_memo
    memo = make_memo(bull_case=BullBearCase(headline="h", key_points=["a 1.1%", "b 2.2%", "c 3.3%"]))
    nc = NumberCheck(checked=True, claims=[
        NumberClaim(field="bull_case.key_points[0]", start=2, end=6, raw="1.1%", status="untraceable"),
        NumberClaim(field="bull_case.key_points[2]", start=2, end=6, raw="3.3%", status="weak"),
    ])
    plan = number_check.WithholdPlan(items={"bull_case.key_points": [0]})
    assert number_check.apply_withholding(memo, nc, plan) == ["a 1.1%"]
    assert memo.bull_case.key_points == ["b 2.2%", "c 3.3%"]
    assert [(c.field, c.raw) for c in nc.claims] == [("bull_case.key_points[1]", "3.3%")]
    assert nc.withheld[0].claims[0].raw == "1.1%"
    assert number_check.drop_stale_claims(memo, nc) == 0
