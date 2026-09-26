"""Chat's bounded read of the bull/bear debate (slice B8-C1; design
`design-bullbear-final.md` §12.4).

`debate_outcome` (≤800 chars: status, the PM crux, rulings per side, the
unanswered count) reaches every chat exit that describes a memo — the LLM
context (legacy follow-up and the SDK `get_memo` tool) and the rendered
inline answer — and is built from the PRESENTED memo, so a section the
memo page hides stays hidden in chat. A memo with no debate (every memo
written with DEBATE_MODE off) sends exactly what it always did.
"""
from __future__ import annotations

from app.agents import orchestrator as orch_mod
from app.schemas import DebateRecord, SectionAvailability, StockMemoOut
from app.schemas.agents import DebateClaim, DebateResolution, DebateRuling
from app.services import memo_sections
from app.tests.factories import make_memo

LONG_CRUX = "Whether data-centre demand outlasts the capex cycle. " * 60   # ~3,300 chars


def _claim(cid: str, side: str, **kw) -> DebateClaim:
    base = dict(id=cid, side=side, claim=f"claim {cid}", grade="sourced")
    base.update(kw)
    return DebateClaim(**base)


def _debate(**overrides) -> DebateRecord:
    base = dict(
        status="complete",
        claims=[
            _claim("B1", "bull"), _claim("B2", "bull"),
            _claim("R1", "bear"),
            _claim("R2", "bear", dropped=True, drop_reason="duplicate"),
            _claim("R3", "bear", grade="unsupported"),
        ],
        # R2 (dropped) and R3 (unsupported) are withheld from the page, so
        # they must not count: the displayed count is 2 (B2, R1).
        unanswered=["B2", "R1", "R2", "R3"],
        resolution=DebateResolution(
            status="ruled", crux=LONG_CRUX,
            rulings=[
                DebateRuling(dispute="D1", claim="B1", ruling="bull"),
                DebateRuling(dispute="D2", claim="R1", ruling="bear"),
                DebateRuling(dispute="D3", claim="B2", ruling="bull"),
                DebateRuling(dispute="D4", claim="R1", ruling="split"),
                DebateRuling(dispute="D5", claim="B2", ruling="not_ruled"),
            ],
        ),
    )
    base.update(overrides)
    return DebateRecord(**base)


def _debated_memo(ticker: str = "ZZDEB", **debate_overrides) -> StockMemoOut:
    """A memo as the D6 writer stores it: the debate record plus the bull and
    bear cases projected from its displayable claims. The presenter shows a
    claim only when the presented case of its side still carries its text
    (B8-D4 `_present_debate`), so a memo whose cases do not quote the claims
    would hide all of them."""
    rec = _debate(**debate_overrides)
    return make_memo(
        ticker=ticker, debate=rec,
        bull_case=memo_sections.debate_case_projection(rec, "bull"),
        bear_case=memo_sections.debate_case_projection(rec, "bear"),
    )


def _presented(memo: StockMemoOut, **hidden: str) -> StockMemoOut:
    shown = memo_sections.present_memo(memo)
    for key, reason in hidden.items():
        shown.section_availability[key] = SectionAvailability(status="unavailable", reason=reason)
    return shown


def test_chat_debate_outcome_bounded_and_respects_hidden_sections():
    memo = _presented(_debated_memo())
    outcome = orch_mod._debate_outcome(memo)
    assert outcome is not None
    assert len(outcome) <= orch_mod.DEBATE_OUTCOME_MAX_CHARS
    assert outcome.startswith("Bull/bear debate complete.")
    assert "PM crux: Whether data-centre demand outlasts the capex cycle." in outcome
    assert "bull 2, bear 1, split 1, unresolved 1" in outcome
    assert outcome.endswith("Claims left unanswered by the other side: 2.")

    # It reaches the LLM context and the rendered inline answer.
    ctx = orch_mod._memo_for_chat_context(memo)
    assert ctx["debate_outcome"] == outcome
    assert f"**Debate:** {outcome}" in orch_mod._render_memo_answer(memo)

    # A debate section the page hides is not described in chat at all.
    hidden = _presented(_debated_memo(), debate="agent_failed")
    assert orch_mod._debate_outcome(hidden) is None
    assert "debate_outcome" not in orch_mod._memo_for_chat_context(hidden)
    assert "**Debate:**" not in orch_mod._render_memo_answer(hidden)

    # A hidden PM view cannot be quoted as having ruled.
    no_pm = _presented(_debated_memo(),
                       final_pm_view="pm_view_unavailable")
    text = orch_mod._debate_outcome(no_pm)
    assert text is not None and "PM crux" not in text and "rulings" not in text
    assert "The PM did not adjudicate this debate." in text


def test_debate_outcome_states_for_partial_unavailable_and_unruled():
    partial = _presented(_debated_memo(status="partial"))
    assert orch_mod._debate_outcome(partial).startswith("Bull/bear debate partly complete")

    unavailable = _presented(_debated_memo(status="unavailable", reason="refused:cyber"))
    text = orch_mod._debate_outcome(unavailable)
    # Plain words, no internal codes (product principle 2).
    assert text == "The bull/bear debate was unavailable for this memo."

    unruled = _presented(_debated_memo(
        resolution=DebateResolution(status="pm_unavailable", crux="should not show")))
    text = orch_mod._debate_outcome(unruled)
    assert "should not show" not in text and "did not adjudicate" in text


def test_memo_without_a_debate_sends_the_same_context_as_before():
    for debate in (None, DebateRecord(status="not_run")):
        memo = _presented(make_memo(ticker="ZZDEB", debate=debate))
        assert orch_mod._debate_outcome(memo) is None
        ctx = orch_mod._memo_for_chat_context(memo)
        assert "debate_outcome" not in ctx
        assert "**Debate:**" not in orch_mod._render_memo_answer(memo)
