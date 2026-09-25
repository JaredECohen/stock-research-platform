"""Typed context passed between the stages of a memo run (RP-002).

`graph._run_stock_memo_inner` is a sequence of stage calls:

    _gather_inputs -> MemoInputs
    _run_analyst_round(inputs) -> AnalystRound
    _adjust_dcf(inputs, round) -> DCFStage
    _compose_memo(inputs, round, dcf_stage) -> StockMemoOut
    _review_memo(memo, inputs, round) -> StockMemoOut   (critic, blend, 7(b))
    _build_verdict(memo, ...) -> VerdictOutcome   (pure; orchestrator applies)
    _assess_quality(memo, inputs, round) -> QualityOutcome   (pure; 7(c))
    _render_final_texts(memo, verdict, initial)   (the confidence-bearing text)
    _run_reflection(memo, inputs)
    _persist(memo, inputs) -> StockMemoOut

The dataclasses here are the contracts between those stages. This module
imports schemas and the safe-runner only — never `graph` — so the roster
and the stages can both depend on it without an import cycle.

Mutation contract (read this before touching a stage)
-----------------------------------------------------
Two objects are SHARED and MUTATED IN PLACE across stages 2-5, on purpose:

* ``MemoInputs.profile`` — the company profile dict. The compose stage
  backfills ``profile["risks"]`` from the bear case when the profile
  carries none, and the verdict stage's thesis builder reads that list
  for its lever clause. A stage that copied the profile would silently
  starve the thesis of a real risk.
* ``AnalystRound.findings`` — the per-analyst findings dict, keyed by
  ``AgentSpec.key`` in roster order. The deep-research loop replaces
  entries, long-form enrichment mutates each finding's
  ``long_form_report``, ``_adjust_dcf`` rewrites the valuation finding's
  DCF numbers in place, and ``_review_memo`` stashes the applied risk
  recommendations on ``findings["risk"].data``. The memo built by the
  compose stage holds the *same* finding objects, so those later edits
  are visible on the memo without re-assignment.

Stages must therefore never copy ``profile`` or ``findings``; they pass
the same objects through. The verdict stage is the exception in the
other direction: it reads the memo and returns a ``VerdictOutcome`` and
the orchestrator applies it — nothing inside ``_build_verdict`` writes
to the memo, which is what lets `test_memo_consistency` exercise it on a
fixture memo without running the pipeline. The quality stage is pure the
same way (``QualityOutcome``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from ..schemas import (
    AgentFinding,
    CompsResult,
    ConfidenceAssessment,
    CritiqueQuestion,
    DCFResult,
    MemoQuality,
    MispricingThesis,
    NumberCheck,
    RoundFindings,
    ScorecardSummary,
    StockMemoOut,
    ValuationVerdict,
)
from .safe_runner import DegradationLog

if TYPE_CHECKING:  # intake -> roster -> memo_context; keep the runtime edge one-way
    from .intake import IntakeDecision
    from .number_check import WithholdPlan
    from .source_ledger import SourceLedger


@dataclass
class MemoInputs:
    """Everything the analyst roster and the later stages read.

    `AgentSpec.needs` names attributes of this class; `test_roster` checks
    every declared need exists here so a spec cannot ask for an input the
    gather stage never produces.
    """
    ticker: str
    run_id: str
    scenario: str
    force_refresh: bool
    as_of_date: date | None
    fin: dict[str, Any]
    profile: dict[str, Any]
    ratios: dict[str, Any]
    earnings: dict[str, Any]
    transcript: dict[str, Any] | None
    filings: list[dict[str, Any]]
    dcf: DCFResult | None
    comps: CompsResult | None
    degradation: DegradationLog
    # Consensus estimates. Reserved for the expectations-ledger work; the
    # gather stage does NOT fetch it today — the consensus lookup stays
    # lazy inside `graph._market_gap_clause`, which only fires when a DCF
    # with a growth path exists. Lifting it here would add a provider
    # round-trip to every memo, so it stays None until a stage needs it.
    estimates: dict[str, Any] | None = None
    # Phase 6 — the latest Fundamental Factor Scorecard row for the ticker
    # (point-in-time at `as_of_date`), or None when no succeeded run has
    # scored it / `ENABLE_SCORECARD=false`. Read once in the gather stage;
    # the valuation analyst and PM synthesis receive it as a <= 600-char
    # block (`scorecard_context.prompt_block`), and the compose stage
    # attaches it to `StockMemoOut.scorecard`. Never feeds the rating blend.
    scorecard: ScorecardSummary | None = None
    # Seed questions from `scorecard_disagreements` rows queued for review.
    # Non-empty only when a flag-gated review regen asked for this memo;
    # the deep-research loop re-fires them on round 1 regardless of the PM
    # critique. `scorecard_seeds_consumed` is the one stage flag written
    # after the gather stage (by `_run_analyst_round`, once the loop ran
    # with them) so the persist stage knows to mark the rows reviewed.
    scorecard_seeds: list[CritiqueQuestion] = field(default_factory=list)
    scorecard_seeds_consumed: bool = False
    # FEAT-003 — the company's current industry-group classification row
    # (`industry_classification.row_dict` shape), or None. Read once in the
    # gather stage, and only when ENABLE_INDUSTRY_ANALYST_ROUTING is on; the
    # Industry Group Analyst spec's `applies_to` and the sector analyst's
    # `{industry_group_block}` both read it here, so a memo never looks the
    # mapping up twice. With routing off it stays None and nothing reads it.
    industry_group: dict[str, Any] | None = None
    # W2b 7(a) — the run's source ledger (activated by `run_stock_memo`;
    # None for a direct stage call, in which case the number check does not
    # run). Analysts register their payloads through the context var, not
    # this handle; the quality stage reads the snapshot from here.
    ledger: SourceLedger | None = None
    # W2b 7(a) — forward figures the PM declared (`forecast_assumptions`),
    # shape-checked. Written by the compose stage (the PM speaks there), read
    # by the quality stage; like `scorecard_seeds_consumed`, a stage output
    # carried on the inputs because no memo field holds it before the check.
    forecast_assumptions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalystRound:
    """Output of the fan-out (+ deep-research dialog + long-form pass)."""
    # Roster order, keyed by AgentSpec.key — over the specs that applied to
    # this run (`roster.applicable`); a spec whose predicate said no has no
    # entry, so consumers index by key only after checking membership.
    findings: dict[str, AgentFinding]
    intake: IntakeDecision
    round_findings: list[RoundFindings] = field(default_factory=list)


@dataclass
class DCFStage:
    """The working DCF after the PM adjuster, plus the audit trail.

    `initial_dcf` is the consensus-anchored model the analysts ran on;
    `dcf` is what every later stage reads. They are the same object when
    no PM adjustment fired.
    """
    dcf: DCFResult | None
    initial_dcf: DCFResult | None
    pm_adjustments: list[dict[str, Any]] = field(default_factory=list)
    pm_headline: str = ""


@dataclass(frozen=True)
class DegradationNote:
    """One degradation the verdict stage observed, in program order.

    `soft=True` entries replay through `DegradationLog.record_soft` (which
    dedupes per agent — a second "Thesis Builder" note must not double the
    banner entry); hard entries came from a `safe_call` and replay as-is.
    """
    agent: str
    error_type: str
    message: str
    soft: bool


@dataclass(frozen=True)
class PMOpinion:
    """The PM synthesis's own call, captured before review moves anything.

    `_render_final_texts` compares the published rating and confidence
    with it and, when either moved, keeps this text as the PM's pre-review
    rationale instead of presenting an obsolete call as the final one."""
    rating: str
    confidence: float
    text: str


def final_verdict_lead(memo: StockMemoOut) -> str:
    """The rating/confidence lead of `final_verdict`, from the memo's
    CURRENT values. One formatter so no stage can print a stale number."""
    return f"PM final view: {memo.rating_label} (confidence {int(memo.confidence_score)}). "


@dataclass
class VerdictOutcome:
    """What `_build_verdict` decided; the orchestrator writes it onto the memo.

    `final_verdict_body` is the verdict WITHOUT its rating/confidence lead:
    the confidence is not final until the quality stage has capped it, so
    the lead is rendered once, afterwards (`render`, called from
    `graph._render_final_texts`). Rendering it here and patching it later is
    how the 5aa1b74 stale-string class happens."""
    valuation_verdict: ValuationVerdict
    one_sentence_thesis: str
    mispricing_thesis: MispricingThesis
    final_verdict_body: str
    extra_scores: dict[str, float] = field(default_factory=dict)  # cross_sector_relevance_count
    thesis_rewrite_fired: bool = False
    # W2a write-time provenance. `thesis_rewritten`: the thesis builder's text
    # replaced the PM's (not merely that the guard fired). `mispricing_fallback`:
    # the mispricing card is `_build_mispricing_fallback`'s template. Neither
    # fact is recoverable from the stored text alone.
    thesis_rewritten: bool = False
    mispricing_fallback: bool = False
    # Failures the stage swallowed on the reader's behalf. Returned rather
    # than recorded so the function stays pure; the orchestrator applies
    # them to the run's DegradationLog in this order.
    degradations: list[DegradationNote] = field(default_factory=list)

    def render(self, memo: StockMemoOut) -> str:
        """The full `final_verdict` against `memo`'s current rating/confidence."""
        return final_verdict_lead(memo) + self.final_verdict_body

    def apply(self, memo: StockMemoOut, degradation: DegradationLog) -> None:
        """Write the outcome onto `memo` and replay the degradations.

        Lives here rather than in graph.py so the one place that knows the
        outcome's field-to-memo mapping is next to the fields themselves.
        `final_verdict` is NOT written here (see the class docstring).
        """
        memo.valuation_verdict = self.valuation_verdict
        memo.one_sentence_thesis = self.one_sentence_thesis
        memo.mispricing_thesis = self.mispricing_thesis
        memo.section_provenance = {
            **(memo.section_provenance or {}),
            "thesis": "rewrite" if self.thesis_rewritten else "pm",
            "mispricing": "fallback" if self.mispricing_fallback else "pm",
        }
        if self.extra_scores and isinstance(memo.scores, dict):
            memo.scores = {**memo.scores, **self.extra_scores}
        for note in self.degradations:
            if note.soft:
                degradation.record_soft(note.agent, note.message, kind=note.error_type)
            else:
                # Already redacted by the scratch log's `record`; append the
                # same shape without re-running redaction on it.
                degradation.failures.append({
                    "agent": note.agent,
                    "error_type": note.error_type,
                    "message": note.message,
                })


@dataclass
class QualityOutcome:
    """What `_assess_quality` decided: 7(a) number check, 7(c) confidence.

    Pure stage output, applied by the orchestrator. `apply` is the one
    place that writes confidence after the quality stage, and it writes it
    everywhere at once: `confidence_score == scores["confidence"] ==
    quality.confidence.final`. It also carries out the number check's
    withholding plan (`number_check.apply_withholding`), which is the only
    edit the quality stage makes to memo prose."""
    confidence: ConfidenceAssessment
    degradations: list[DegradationNote] = field(default_factory=list)
    # None when the check did not run (no ledger) — the memo then carries no
    # `number_check` and no number-based cap.
    number_check: NumberCheck | None = None
    withhold: WithholdPlan | None = None

    def apply(self, memo: StockMemoOut, degradation: DegradationLog) -> None:
        final = float(self.confidence.final)
        quality = memo.quality or MemoQuality()
        nc = self.number_check.model_copy(deep=True) if self.number_check is not None else None
        if nc is not None:
            from . import number_check as _nc
            _nc.apply_withholding(memo, nc, self.withhold)
            nc.counts = {**nc.counts, "withheld": len(nc.withheld)}
            _nc.drop_stale_claims(memo, nc)
        memo.quality = quality.model_copy(update={"confidence": self.confidence, "number_check": nc})
        memo.confidence_score = final
        if isinstance(memo.scores, dict):
            memo.scores = {**memo.scores, "confidence": final}
        # W2a contract C2: an earned confidence is shown even when the PM view
        # was a template (the caps already say how little it is worth).
        memo.section_provenance = {**(memo.section_provenance or {}), "confidence": "earned"}
        for note in self.degradations:
            if note.soft:
                degradation.record_soft(note.agent, note.message, kind=note.error_type)
            else:
                degradation.failures.append({
                    "agent": note.agent, "error_type": note.error_type, "message": note.message,
                })
