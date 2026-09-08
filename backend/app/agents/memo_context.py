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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..schemas import (
    AgentFinding,
    CompsResult,
    DCFResult,
    MispricingThesis,
    RoundFindings,
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
    as_of_date: Optional[date]
    fin: Dict[str, Any]
    profile: Dict[str, Any]
    ratios: Dict[str, Any]
    earnings: Dict[str, Any]
    transcript: Optional[Dict[str, Any]]
    filings: List[Dict[str, Any]]
    dcf: Optional[DCFResult]
    comps: Optional[CompsResult]
    degradation: DegradationLog
    # Consensus estimates. Reserved for the expectations-ledger work; the
    # gather stage does NOT fetch it today — the consensus lookup stays
    # lazy inside `graph._market_gap_clause`, which only fires when a DCF
    # with a growth path exists. Lifting it here would add a provider
    # round-trip to every memo, so it stays None until a stage needs it.
    estimates: Optional[Dict[str, Any]] = None


@dataclass
class AnalystRound:
    """Output of the fan-out (+ deep-research dialog + long-form pass)."""
    findings: Dict[str, AgentFinding]        # roster order, keyed by AgentSpec.key
    intake: "IntakeDecision"
    round_findings: List[RoundFindings] = field(default_factory=list)


@dataclass
class DCFStage:
    """The working DCF after the PM adjuster, plus the audit trail.

    `initial_dcf` is the consensus-anchored model the analysts ran on;
    `dcf` is what every later stage reads. They are the same object when
    no PM adjustment fired.
    """
    dcf: Optional[DCFResult]
    initial_dcf: Optional[DCFResult]
    pm_adjustments: List[Dict[str, Any]] = field(default_factory=list)
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
    extra_scores: Dict[str, float] = field(default_factory=dict)  # cross_sector_relevance_count
    thesis_rewrite_fired: bool = False
    # Failures the stage swallowed on the reader's behalf. Returned rather
    # than recorded so the function stays pure; the orchestrator applies
    # them to the run's DegradationLog in this order.
    degradations: List[DegradationNote] = field(default_factory=list)

    def apply(self, memo: StockMemoOut, degradation: DegradationLog) -> None:
        """Write the outcome onto `memo` and replay the degradations.

        Lives here rather than in graph.py so the one place that knows the
        outcome's field-to-memo mapping is next to the fields themselves.
        """
        memo.valuation_verdict = self.valuation_verdict
        memo.one_sentence_thesis = self.one_sentence_thesis
        memo.mispricing_thesis = self.mispricing_thesis
        memo.final_verdict = self.final_verdict
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
