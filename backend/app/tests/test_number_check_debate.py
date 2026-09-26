"""D4 (2026-09-25) — the number check reads the debate texts a reader sees.

Design-bullbear-final §12.2, for a debate that is shown (complete or
partial): rebuttal arguments are FLAG (never withheld: that would orphan the
claim id the matrix points at), the cruxes and the PM's resolution crux are
paragraphs (the resolution is the PM's, so its declared assumptions apply),
and a displayed claim's falsifier is a threshold. Claim texts are not read
from the record: a shown claim is its side's case key point, already
checked there, and reading it twice would double its weight in the ratio cap.

Critique #7: registering debate passages widens the ledger for the whole
memo, so the result must say which refs support each figure (R1 counts the
figures traced via debate-registered refs alone).
"""
from __future__ import annotations

from app.agents import number_check as nc
from app.agents.source_ledger import FactRegistry, SourceLedger
from app.schemas import (
    BullBearCase,
    DebateClaim,
    DebateRecord,
    DebateResolution,
    DebateResponse,
)
from app.tests.factories import make_memo

REVENUE = {"income": [{"period": "2025", "revenue": 1_203_000_000}]}


def _registry(*registrations: tuple[str, str, object]) -> FactRegistry:
    ledger = SourceLedger()
    for kind, ref, obj in registrations:
        ledger.register(kind, ref, obj)
    return ledger.snapshot()


def _claim(cid: str, side: str, text: str, **over) -> DebateClaim:
    return DebateClaim.model_validate({"id": cid, "side": side, "claim": text, "grade": "sourced", **over})


def _debate(**over) -> DebateRecord:
    base = dict(
        status="complete",
        headlines={"bull": "Share gains compound", "bear": "Margins mean-revert"},
        cruxes={"bull": "Whether revenue of $1.2B keeps growing.", "bear": "Whether $9.99B of backlog is real."},
        claims=[
            _claim("BULL-1", "bull", "Revenue reached $1.2B on share gains.", falsifier="Revenue < $1.0B"),
            _claim("BEAR-1", "bear", "Competition is intensifying.", falsifier="Churn > 5%"),
            _claim("BEAR-2", "bear", "Hidden liabilities of $7.77B.", grade="unsupported",
                   falsifier="Liabilities > $8B"),
            _claim("BULL-2", "bull", "A dropped claim.", dropped=True, falsifier="Price > $500"),
        ],
        responses=[
            DebateResponse(side="bear", target="BULL-1", stance="rebut",
                           argument="Revenue was really $3.33B lower once one-offs go."),
            DebateResponse(side="bull", target="BEAR-2", stance="rebut",
                           argument="The $6.66B figure is not in any filing."),
        ],
        resolution=DebateResolution(status="ruled", crux="We expect 18.5% growth to hold."),
        deterministic_checks=["Rating Bullish vs valuation verdict overvalued: diverges by 2 steps."],
    )
    base.update(over)
    return DebateRecord(**base)  # type: ignore[arg-type]


def _debated_memo(**over) -> object:
    debate = over.pop("debate", _debate())
    return make_memo(
        bull_case=BullBearCase(headline="Share gains compound",
                               key_points=["Revenue reached $1.2B on share gains."]),
        bear_case=BullBearCase(headline="Margins mean-revert", key_points=["Competition is intensifying."]),
        key_risks=[], thesis_breakers=[], debate=debate, **over,
    )


def _by_field(result: nc.MemoCheck) -> dict[str, list[tuple[str, str]]]:
    return {fr.spec.path: [(c.claim.raw, c.status) for c in fr.claims] for fr in result.fields if fr.claims}


def test_debate_paths_and_policies():
    specs = {s.path: s.policy for s in nc.iter_fields(_debated_memo()) if s.path.startswith("debate.")}
    assert specs == {
        # The response to the unsupported BEAR-2 is never shown, so it is not read.
        "debate.responses[0].argument": nc.POLICY_FLAG,
        "debate.cruxes.bull": nc.POLICY_PARAGRAPH,
        "debate.cruxes.bear": nc.POLICY_PARAGRAPH,
        # Only the displayed claims' falsifiers (not the unsupported or dropped ones).
        "debate.claims[0].falsifier": nc.POLICY_THRESHOLD,
        "debate.claims[1].falsifier": nc.POLICY_THRESHOLD,
        "debate.resolution.crux": nc.POLICY_PARAGRAPH,
    }


def test_debate_fields_only_when_the_debate_is_shown():
    """Legacy memos (no debate) and a debate that is not shown read exactly
    the fields they read before D4."""
    legacy = make_memo()
    assert not [s for s in nc.iter_fields(legacy) if s.path.startswith("debate.")]
    for status in ("not_run", "unavailable"):
        memo = _debated_memo(debate=_debate(status=status))
        assert not [s for s in nc.iter_fields(memo) if s.path.startswith("debate.")], status
    partial = _debated_memo(debate=_debate(status="partial"))
    assert [s for s in nc.iter_fields(partial) if s.path.startswith("debate.")]


def test_number_check_flags_rebuttal_figures_without_removing_items():
    reg = _registry(("financials", "financials:T", REVENUE))
    memo = _debated_memo()
    result = nc.check_memo(memo, reg, withhold=True)
    by_field = _by_field(result)
    assert by_field["debate.responses[0].argument"] == [("$3.33B", "untraceable")]
    assert by_field["debate.cruxes.bear"] == [("$9.99B", "untraceable")]
    # The falsifier is a threshold: counted, never checked or flagged.
    assert by_field["debate.claims[0].falsifier"] == [("$1.0B", "threshold")]
    # FLAG: nothing in the debate is planned for withholding, and applying
    # the plan leaves every response in place.
    assert not any(lp.startswith("debate.") for lp in result.plan.items)
    summary = nc.summarize(result, assumptions=[], notes=[])
    nc.apply_withholding(memo, summary, result.plan)
    assert len(memo.debate.responses) == 2
    assert {c.field for c in summary.claims} >= {"debate.responses[0].argument", "debate.cruxes.bear"}
    # A stored claim indexes its field's text (the renderer contract).
    for c in summary.claims:
        text = nc.resolve_field(memo, c.field)
        assert isinstance(text, str) and text[c.start:c.end] == c.raw, c


def test_resolution_crux_is_pm_field():
    assert nc.is_pm_field("debate.resolution.crux")
    assert not nc.is_pm_field("debate.cruxes.bull")
    assert not nc.is_pm_field("debate.responses[0].argument")
    declared = [{"value": 18.5, "unit": "pct", "basis_ref": "financials:T", "horizon": "FY2027"}]
    memo = _debated_memo(debate=_debate(cruxes={"bull": "Growth of 18.5% holds."}))
    result = nc.check_memo(memo, _registry(("financials", "financials:T", REVENUE)), withhold=True,
                           assumptions=declared)
    by_field = _by_field(result)
    # The PM's own declaration labels the PM's ruling, not an advocate's crux.
    assert by_field["debate.resolution.crux"] == [("18.5%", "assumption")]
    assert by_field["debate.cruxes.bull"] == [("18.5%", "untraceable")]


def test_no_double_count_of_claim_text():
    """A shown claim's figure is read once, as its case key point; the
    record's claim texts (shown or not) are never read."""
    memo = _debated_memo()
    specs = list(nc.iter_fields(memo))
    assert not [s for s in specs if s.path.startswith("debate.claims[") and not s.path.endswith(".falsifier")]
    texts_with_figure = [s.path for s in specs if "$1.2B" in s.text]
    assert texts_with_figure == ["bull_case.key_points[0]", "debate.cruxes.bull"]
    # The unsupported claim's figure appears nowhere the check reads.
    assert not [s.path for s in specs if "$7.77B" in s.text]
    result = nc.check_memo(memo, _registry(("financials", "financials:T", REVENUE)), withhold=True)
    summary = nc.summarize(result, assumptions=[], notes=[])
    # One traced $1.2B per displayed text: the case point and the bull crux.
    raws = [(fr.spec.path, c.claim.raw) for fr in result.fields for c in fr.claims if c.status == "traced"]
    assert raws.count(("bull_case.key_points[0]", "$1.2B")) == 1
    assert summary.counts["claims_total"] == sum(
        1 for fr in result.fields for c in fr.claims if c.status in nc.FACT_STATUSES)


def test_number_check_exposes_support_refs():
    """Every checked figure carries ALL the refs whose facts carry its
    value (uncapped, unlike the eight stored credits), so a figure traced
    only through debate-registered passages can be counted (critique #7)."""
    reg = _registry(
        ("financials", "financials:T", REVENUE),
        ("research_note", "chunk:filing:0007", {"revenue": 1_203_000_000}),
        ("research_note", "chunk:filing:0009", {"revenue_backlog": 9_990_000_000}),
    )
    memo = _debated_memo(debate=_debate(cruxes={"bull": "Revenue of $1.2B.", "bear": "Backlog of $9.99B."}))
    result = nc.check_memo(memo, reg, withhold=True)
    support = {(f.field, f.raw): (f.status, f.refs) for f in nc.figure_support(result)}
    assert support[("bull_case.key_points[0]", "$1.2B")] == (
        "traced", ("chunk:filing:0007", "financials:T"))
    assert support[("debate.cruxes.bear", "$9.99B")] == ("traced", ("chunk:filing:0009",))
    assert support[("debate.claims[0].falsifier", "$1.0B")] == ("threshold", ())
    assert support[("debate.responses[0].argument", "$3.33B")] == ("untraceable", ())
    # Offsets index the field's text, as the stored claims do.
    for f in nc.figure_support(result):
        text = nc.resolve_field(memo, f.field)
        assert text[f.start:f.end] == f.raw, f
    only_debate = nc.traced_only_via(result, {"chunk:filing:0007", "chunk:filing:0009"})
    # The $1.2B also traces to the financials, so only the backlog figure
    # would not have traced without the debate's passages.
    assert [(f.field, f.raw) for f in only_debate] == [("debate.cruxes.bear", "$9.99B")]
