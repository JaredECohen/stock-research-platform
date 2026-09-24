"""Typed context passed between the stages of a memo run (RP-002).

`graph._run_stock_memo_inner` is a sequence of stage calls:

    _gather_inputs -> MemoInputs
    _run_analyst_round(inputs) -> AnalystRound
    _adjust_dcf(inputs, round) -> DCFStage
    _compose_memo(inputs, round, dcf_stage) -> StockMemoOut
    _review_memo(memo, inputs, round) -> StockMemoOut
    _build_verdict(memo, ...) -> VerdictOutcome   (pure; orchestrator applies)
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
fixture memo without running the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from ..schemas import (
    AgentFinding,
    CompsResult,
    CritiqueQuestion,
    DCFResult,
    MispricingThesis,
    RoundFindings,
    ScorecardSummary,
    StockMemoOut,
    ValuationVerdict,
)
from .safe_runner import DegradationLog

if TYPE_CHECKING:  # intake -> roster -> memo_context; keep the runtime edge one-way
    from .intake import IntakeDecision


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


@dataclass
class VerdictOutcome:
    """What `_build_verdict` decided; the orchestrator writes it onto the memo."""
    valuation_verdict: ValuationVerdict
    one_sentence_thesis: str
    mispricing_thesis: MispricingThesis
    final_verdict: str
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

    def apply(self, memo: StockMemoOut, degradation: DegradationLog) -> None:
        """Write the outcome onto `memo` and replay the degradations.

        Lives here rather than in graph.py so the one place that knows the
        outcome's field-to-memo mapping is next to the fields themselves.
        """
        memo.valuation_verdict = self.valuation_verdict
        memo.one_sentence_thesis = self.one_sentence_thesis
        memo.mispricing_thesis = self.mispricing_thesis
        memo.final_verdict = self.final_verdict
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
