"""Memo schemas — mispricing thesis, valuation verdict, `StockMemoOut`.

Every class referenced by a `StockMemoOut` annotation is imported here
explicitly: under `from __future__ import annotations` pydantic resolves
forward references from this module's globals, so a name that only lives
in the package `__init__` would fail to resolve at model build time.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .agents import (
    AgentFinding,
    BullBearCase,
    CatalystItem,
    CriticReview,
    DebateRecord,
    DivergenceAssessment,
    RiskItem,
    RoundFindings,
)
from .common import RatingLabel
from .scorecard import ScorecardSummary


class MispricingThesis(BaseModel):
    """The 'why is the market wrong' field — required output of the PM.

    Wave 10. A serious retail user pays for the *mispricing* call, not
    a metric recap. PM is required to fill these on every memo. Empty
    strings are allowed (the PM can say 'no mispricing — fairly priced
    on our work') but the structure must be present.
    """
    consensus_view: str = ""
    our_view: str = ""
    gap: str = ""
    falsifiers: list[str] = Field(default_factory=list)


class ValuationVerdict(BaseModel):
    """Single source of truth for "is it cheap or expensive?".

    Historically the memo answered that question independently in five
    places (thesis verdict word, rating badge, comps premium, valuation
    headline, DCF summary) and the answers could contradict each other.
    This object is computed once per memo — after the PM DCF adjustment
    and the final rating blend — from the three signals that diverge in
    practice. The thesis, the valuation card, and the mispricing fallback
    all read from it.
    """
    # "mixed" (W2b 7(b): opposing evidence votes) is accepted here one
    # deploy wave before anything writes it. Expand before write is what
    # keeps a rollback safe: if the writer is reverted, memos it already
    # stored still validate instead of turning into `memo_unreadable`.
    verdict: Literal["undervalued", "fairly_priced", "overvalued", "mixed"] = "fairly_priced"
    # How `verdict` was reached. Every verdict `graph._build_valuation_verdict`
    # produced before W2b came from the rating (`graph._verdict_word`), so the
    # default is the truth for those; W2b's evidence-only verdict records
    # "evidence". Not every verdict was produced, though: the guarded fallback
    # in graph.py (`_guarded(..., fallback=ValuationVerdict())`) and memos
    # stored before this object existed (the `default_factory` on
    # StockMemoOut) both carry the bare placeholder, which also reads as
    # basis="rating". That placeholder is only recognisable by its empty
    # `summary`, so consumers must treat an empty summary as "not produced"
    # (W2a §4.3) whatever `basis` says.
    basis: Literal["rating", "evidence"] = "rating"
    # The evidence votes and inputs behind an "evidence" verdict (W2b).
    # Free-form so the vote set can evolve without a schema bump.
    signals: dict[str, Any] = Field(default_factory=dict)
    dcf_base_upside: float | None = None
    comps_ev_ebitda_premium: float | None = None
    factor_valuation: float | None = None
    summary: str = ""


# ---------------------------------------------------------------------------
# Memo contract C1 (S2, 2026-09-24): every StockMemoOut change W2a and W2b
# need, shipped expand-only in one schema bump. No writer sets any of these
# in the slice that adds them; later slices only write them.
# ---------------------------------------------------------------------------

SectionStatus = Literal["available", "degraded", "unavailable"]
# Closed vocabulary; the frontend maps each value to one sentence. This is
# the union of the integration plan's C1 list and the reasons W2a §4.3's
# per-section rules emit, so the presenter (a later slice that must not
# touch this schema) never needs a value the contract lacks.
SectionReason = Literal[
    "template_fallback", "template_always", "derived_from_hidden", "skipped_by_intake",
    "critic_not_run", "rule_based", "llm_patched", "reduced_inputs", "not_produced",
    "unclassified",
    "agent_failed", "no_source_data", "pm_view_unavailable", "partial_template",
    "follow_up_unanswered", "templated_scenarios",
    # D4 (2026-09-25, design-bullbear-final §12.1): the debate's own states.
    # D2 shipped the debate record without them, and the table names them,
    # so they are added here expand-only (read-time values; never stored).
    # `debate_unavailable` hides both cases and the debate section,
    # `not_run` is a debate the run had no model for (no LLM, a backtest),
    # and `rebuttals_unavailable` is the note on a partial debate.
    "debate_unavailable", "not_run", "rebuttals_unavailable",
]


class SectionAvailability(BaseModel):
    """Read-time verdict on one memo section (W2a). Never stored.

    Computed by the presenter from the stored payload on every read, so a
    change to the classification rules applies to old memos without
    rewriting them. `memo_store.save_memo` refuses a memo that carries one.
    """
    status: SectionStatus = "available"
    reason: SectionReason | None = None
    hidden_items: int = 0           # list sections: items the presenter removed
    headline_hidden: bool = False   # bull/bear: headline blanked, real items kept
    # Evidence for the verdict, e.g. "event:PM Synthesis/DeterministicFallback",
    # "signature:pm_view_tail", "provenance:thesis=rewrite".
    basis: list[str] = Field(default_factory=list)


NumberClaimStatus = Literal[
    "traced", "weak", "untraceable", "mis_anchored", "assumption", "threshold", "unchecked",
]


class NumberClaim(BaseModel):
    """One figure in memo prose and how it traced to the source ledger (W2b 7a).

    `start`/`end` are offsets into the field's text; `raw` is the exact
    slice, so a renderer can detect a stale offset instead of mis-marking.
    """
    field: str
    start: int
    end: int
    raw: str
    value: float | None = None
    unit: str = ""
    status: NumberClaimStatus
    source_refs: list[str] = Field(default_factory=list)


class WithheldItem(BaseModel):
    """A list item removed from the memo because its figures did not trace."""
    field: str
    index: int
    text: str
    claims: list[NumberClaim] = Field(default_factory=list)


class NumberCheck(BaseModel):
    """W2b 7(a) number-to-source check. `checked=False` means not run."""
    checked: bool = False
    method_version: str = "1"
    # Per-status tallies ("traced", "untraceable", ...). A dict rather than
    # one field per status so a new status does not need a schema bump.
    counts: dict[str, int] = Field(default_factory=dict)
    claims: list[NumberClaim] = Field(default_factory=list)
    withheld: list[WithheldItem] = Field(default_factory=list)
    lists_not_withheld: list[str] = Field(default_factory=list)
    # Fields whose figures were not checked (e.g. text a news patch added),
    # which the UI labels rather than presenting as verified.
    unchecked_fields: list[str] = Field(default_factory=list)
    sources_cited: list[str] = Field(default_factory=list)
    primary_kinds_cited: list[str] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RatingReconciliation(BaseModel):
    """W2b 7(b): how the PM rating was squared with the valuation evidence."""
    outcome: Literal["not_applicable", "consistent", "accepted", "downgraded"] = "not_applicable"
    pm_rating: str = ""
    pm_confidence: float | None = None
    blended_rating: str = ""
    final_rating: str = ""
    valuation_verdict: str = ""
    divergence: bool = False
    reason: str = ""
    reason_checks: dict[str, bool] = Field(default_factory=dict)
    critic_assessment: DivergenceAssessment = "not_assessed"
    note: str = ""


class ConfidenceCap(BaseModel):
    """One ceiling on earned confidence (W2b 7c), e.g. `critic_not_live` 60."""
    code: str
    cap: float
    detail: str = ""


class ConfidenceAssessment(BaseModel):
    """PM confidence (`raw`) and what the caps left of it (`final`)."""
    raw: float
    final: float
    caps: list[ConfidenceCap] = Field(default_factory=list)
    binding: str | None = None


class MemoQuality(BaseModel):
    """W2b research-quality record. None on every memo that pre-dates it."""
    v: int = 1
    number_check: NumberCheck | None = None
    rating_reconciliation: RatingReconciliation | None = None
    confidence: ConfidenceAssessment | None = None


class StockMemoOut(BaseModel):
    ticker: str
    company_name: str
    sector: str
    final_pm_view: str
    rating_label: RatingLabel
    confidence_score: float
    one_sentence_thesis: str
    # Wave 10 — mispricing-first synthesis. Empty Mispricing() means the
    # memo predates the schema or the PM declined to commit a view.
    mispricing_thesis: MispricingThesis = Field(default_factory=MispricingThesis)
    # Reconciled valuation call — the one place that answers "cheap or
    # expensive?". Empty default on memos that pre-date the field.
    valuation_verdict: ValuationVerdict = Field(default_factory=ValuationVerdict)
    # Wave 10 — memo-time price snapshot. Frozen at memo creation so a
    # later live-overlay can show drift ("memo wrote DCF vs $145; current
    # $158, +9% since memo"). Null on memos that pre-date the field or
    # ran without quote chain access.
    price_at_memo: float | None = None
    price_at_memo_at: datetime | None = None
    business_summary: str
    sector_agent_view: AgentFinding
    earnings_agent_view: AgentFinding
    filing_agent_view: AgentFinding
    valuation_agent_view: AgentFinding
    comps_agent_view: AgentFinding
    macro_sensitivity: AgentFinding
    # Wave 3B — Technical Analyst. Optional so older memos that pre-date
    # this addition still validate; the graph populates it on every run.
    technical_agent_view: AgentFinding | None = None
    bull_case: BullBearCase
    bear_case: BullBearCase
    # D2 (2026-09-25): the bull/bear debate as it ran, behind DEBATE_MODE.
    # None on every memo written with the mode off and on every memo that
    # pre-dates the field. `bull_case`/`bear_case` above are unchanged, so
    # every existing consumer of the cases keeps working either way.
    debate: DebateRecord | None = None
    catalysts: list[CatalystItem]
    key_risks: list[RiskItem]
    thesis_breakers: list[RiskItem]
    dcf_summary: dict[str, Any] = Field(default_factory=dict)
    # Wave 10 — initial (consensus-anchored) DCF kept alongside the
    # PM-adjusted view in `dcf_summary`. Empty when the PM made no
    # adjustments or no LLM was available. Lets the UI show "what the
    # team's research changed about the model".
    dcf_initial_summary: dict[str, Any] = Field(default_factory=dict)
    dcf_pm_adjustments: list[dict[str, Any]] = Field(default_factory=list)
    dcf_pm_adjustment_headline: str = ""
    portfolio_fit: str = ""
    risk_committee_challenge: CriticReview
    final_verdict: str
    scores: dict[str, float] = Field(default_factory=dict)
    sources_used: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_mode: Literal["demo", "live"] = "demo"
    # Wave 9 — full diligence-dialog audit trail. `round=0` is the
    # initial parallel fan-out; rounds 1+ are PM↔specialist
    # critique-and-revise turns. Empty list when deep_research is off.
    round_findings: list[RoundFindings] = Field(default_factory=list)
    # Wave 10 — forward catalysts (next 90d) populated from
    # `catalyst_events`. Each item: {ticker, event_type, event_date,
    # title, description, materiality, source}.
    forward_catalysts: list[dict[str, Any]] = Field(default_factory=list)
    # Wave 10 — earnings quarter-over-quarter delta (separate finding;
    # rendered as its own UI tile). Optional — None when prior-quarter
    # data isn't available yet.
    earnings_qoq_delta: AgentFinding | None = None
    # Wave 10 — PM intake decision (which specialists were skipped and
    # why). `{skipped: List[str], rationale: str}`. Empty dict when
    # all 8 ran (the default). Audit trail for cost-aware memo runs.
    intake_decision: dict[str, Any] = Field(default_factory=dict)
    # Wave 10 — per-agent influence on the rating. Computed post-PM
    # synthesis from each agent's confidence + tone; values are signed
    # contributions (positive = bullish pull, negative = bearish pull),
    # roughly normalized so the largest |value| is the most-influential
    # agent on this memo. Empty on memos that pre-date the field.
    agent_influence: dict[str, float] = Field(default_factory=dict)
    # Wave 10 — macro context frozen at memo creation. Lets the
    # postmortem regime-conditional dashboards bucket memos by the
    # regime that was active when they were written, even if the macro
    # broadcast cache has rolled over by the time outcomes evaluate.
    macro_snapshot_at_memo: dict[str, float] = Field(default_factory=dict)
    macro_regime_at_memo: str = ""
    # List of agents that failed during this memo's generation. Empty when
    # everything ran normally; populated by the safe-runner so the UI can
    # show "X analyst was unavailable" rather than dropping the memo.
    degraded_agents: list[str] = Field(default_factory=list)
    # RP-001 — the *reason* behind each `degraded_agents` entry, in the
    # same order: `{agent, error_type, message}` (the shape of
    # `DegradationLog.failures`). Lets the guard tests and a future UI
    # tooltip say why a section is thin instead of only that it is.
    # Empty on memos that pre-date the field.
    degradation_events: list[dict[str, Any]] = Field(default_factory=list)
    # RP-003 landing zone — a roster agent with no dedicated memo field
    # (a future Industry Group analyst) lands here keyed by its roster key,
    # so adding an analyst does not require a schema edit. Empty for the
    # current eight-agent roster.
    extra_agent_views: dict[str, AgentFinding] = Field(default_factory=dict)
    # Phase 6 — the Fundamental Factor Scorecard read the memo was written
    # against: observed rank (percentiles, coverage, fiscal period) beside
    # the model read (z, contributions) under a named version, plus the
    # disagreement flag when the narrative contradicts it. None when no
    # succeeded run has scored the ticker (the section renders n/a and
    # `degraded_agents` carries a soft "Fundamental Scorecard" entry), on
    # memos that pre-date the field, and when `ENABLE_SCORECARD=false`.
    # It informs the memo; it does not enter the rating blend.
    scorecard: ScorecardSummary | None = None
    # W2a write-time facts the payload cannot otherwise recover, e.g.
    # {"v": 1, "llm_configured": bool, "thesis": "pm"|"rewrite",
    # "mispricing": "pm"|"fallback"}; W2b may add "confidence": "earned".
    # {} on memos that pre-date it. Persisted with the memo.
    section_provenance: dict[str, Any] = Field(default_factory=dict)
    # W2a read-time map keyed by section key. Filled only by the presenter
    # on the way out; empty on raw and stored memos. `save_memo` refuses a
    # memo that carries one (a presented memo must never become the stored
    # truth) and also excludes the key as a backstop.
    section_availability: dict[str, SectionAvailability] = Field(default_factory=dict)
    # W2b research-quality record (number check, rating reconciliation,
    # earned confidence). None on memos that pre-date it; the UI then
    # renders nothing new.
    quality: MemoQuality | None = None
    disclaimer: str = (
        "MarketMosaic is for investment research and education only. "
        "It does not provide personalized financial, investment, legal, or tax advice."
    )
