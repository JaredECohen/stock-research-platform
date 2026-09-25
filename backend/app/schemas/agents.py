"""Agent-level structured outputs and the multi-agent message contracts.

Everything an individual specialist emits (`AgentFinding`, critique,
bull/bear analysis, earnings extraction, technical signals) plus the
PM<->sector<->tool interchange shapes. Depends only on `common`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class CatalystItem(BaseModel):
    title: str
    detail: str = ""
    horizon: Literal["near_term", "medium_term", "long_term"] = "medium_term"
    impact: Literal["low", "medium", "high"] = "medium"


class RiskItem(BaseModel):
    title: str
    detail: str = ""
    severity: Literal["low", "medium", "high"] = "medium"
    type: Literal["company", "valuation", "macro", "regulatory", "thesis_breaker"] = "company"


class BullBearCase(BaseModel):
    headline: str
    key_points: list[str] = Field(default_factory=list)


class CritiqueQuestion(BaseModel):
    """Wave 9 — one follow-up question the PM emits during deep research.

    `target_agent` names which specialist's runner re-fires with this
    question as additional prompt context. `why_it_matters` is captured
    in the audit trail so reviewers can see *why* the PM dug in.

    `target_agent` is a plain roster key, deliberately NOT a `Literal`:
    the roster is the single authority on which specialists exist, and
    `schemas` cannot import `agents.roster` (roster imports these models,
    so the Literal would be a cycle). Freezing the eight legacy keys here
    is what made the Industry Group Analyst unreachable — a critique aimed
    at it failed validation after the loop had already accepted it.
    `deep_research._addressable` does the checking against the live roster,
    before this model is constructed.
    """
    target_agent: str
    question: str
    why_it_matters: str = ""


class CritiqueOutput(BaseModel):
    """PM critique step's structured output. `no_further_questions`
    is the explicit early-exit signal the loop respects so the PM can
    end the dialog before the round budget runs out."""
    questions: list[CritiqueQuestion] = Field(default_factory=list)
    no_further_questions: bool = False
    rationale: str = ""


class RoundFindings(BaseModel):
    """Wave 9 — one round of the deep-research dialog.

    `round=0` is the initial parallel fan-out (no PM questions). Rounds
    1+ each carry the PM's questions (`pm_questions`) and the
    re-fired agents' new findings (`findings`, keyed by agent_name).
    `early_exit` is set when the PM declared no further questions on
    THIS round, so reviewers know whether the loop terminated by
    consensus or by hitting the round cap.
    """
    round: int
    pm_questions: list[CritiqueQuestion] = Field(default_factory=list)
    findings: dict[str, AgentFinding] = Field(default_factory=dict)
    early_exit: bool = False
    pm_rationale: str = ""


class RiskRecommendation(BaseModel):
    """Wave 8H — actionable rec from the risk analyst.

    The graph applies a deterministic enforcement step that REALLY
    moves these into the memo (confidence cap, rating downshift, bear
    augmentation, thesis-breaker propagation) so risk findings aren't
    just notes — they shape the final memo. The PM synthesis prompt
    also receives them so the LLM-written narrative acknowledges what
    the risk lens demanded.

    `target` — what part of the memo this rec touches.
    `direction` — which way to push.
    `magnitude` — how hard. Confidence deltas: small=5, medium=10, large=15.
    `detail` — short title shown in UI / prose.
    `rationale` — why; required so reviewers can audit why a rec moved.
    """
    target: Literal[
        "confidence", "rating", "thesis_breakers", "sizing", "bear_case",
    ]
    direction: Literal["raise", "lower", "flag", "neutral"]
    magnitude: Literal["small", "medium", "large"] = "medium"
    detail: str
    rationale: str


class FalsifiableTest(BaseModel):
    """A concrete observation that would invalidate one side of the thesis.

    Required by the sector-integrated bull/bear (Wave 3A): a side without a
    falsifiable test is hand-waving. Forcing the analyst to articulate
    "this is wrong if X is observed" disciplines the case construction.
    """
    statement: str
    invalidates_side: Literal["bull", "bear"]


class BullBearAnalysis(BaseModel):
    """Wave 3A — sector-integrated bull/bear with bias mitigations.

    Stored under `sector_agent_view.data["bull_bear_analysis"]`. PM
    synthesis treats `sector_synthesis` as a prior, not a directive: the
    PM rating may diverge from `sector_lean` when other findings outvote
    the sector view, and the PM should explain that divergence.
    """
    bull_case: BullBearCase
    bear_case: BullBearCase
    key_disagreement: str
    falsifiable_tests: list[FalsifiableTest] = Field(default_factory=list)
    sector_synthesis: str
    sector_lean: Literal["bull", "bear", "balanced"] = "balanced"


class Citation(BaseModel):
    """Wave 10 — typed citation for an evidence chain entry.

    `kind` is the source type — filing, transcript, ratio, peer,
    macro, dcf — so the UI can render an appropriate link / drawer.
    `ref` is the stable identifier (accession, period, ticker, etc.).
    `excerpt` is an optional short quote (≤300 chars) that the agent
    pulled into its prompt; we cap to keep the memo payload compact.
    """
    kind: Literal[
        "filing", "transcript", "ratio", "peer", "macro", "dcf",
        "news", "research_note", "other",
    ] = "other"
    ref: str = ""
    excerpt: str = ""
    section: str | None = None


class AgentFinding(BaseModel):
    """One agent's contribution to a memo.

    `data` is an optional structured payload — agents that do real research
    (e.g. sector cohort analysis) attach distributional stats, peer placements,
    and trend tables here so the frontend can render rich evidence beyond
    prose.

    `long_form_report` (Wave 3C) is an optional 4-8 paragraph markdown
    drill-down. Always at least a deterministic build from the structured
    fields above; when `ENABLE_LONG_FORM_REPORTS=true`, enriched with an
    LLM expansion per agent. Frontend renders it in a collapsible drawer
    on each agent tile.

    Wave 10 — `evidence` is the typed citation list (one entry per
    citable claim or chunk). Empty when the agent didn't emit
    citations; backwards-compatible with memos that pre-date the
    field. Frontend renders this as "View sources" expander on the
    agent tile.
    """
    agent: str
    headline: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    sources: list[str] = Field(default_factory=list)
    evidence: list[Citation] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    long_form_report: str | None = None


# Wave 10 — earnings structured extraction.

class GuidanceChange(BaseModel):
    metric: str  # e.g. "FY revenue", "Q4 op margin", "FY FCF"
    prior: str | None = None
    current: str | None = None
    direction: Literal["raised", "lowered", "reaffirmed", "introduced", "withdrawn", "unclear"] = "unclear"
    rationale: str = ""


class ToneSignal(BaseModel):
    speaker: str = ""  # e.g. "CEO", "CFO", "VP Finance"
    segment: str = ""  # e.g. "AWS", "Search", "Auto"
    classification: Literal["constructive", "measured", "cautious", "defensive", "evasive"] = "measured"
    evidence: str = ""  # short transcript quote


class QAThemeAnalysis(BaseModel):
    theme: str
    analyst: str = ""
    response_quality: Literal["clear", "partial", "deflected", "evasive"] = "clear"


class EarningsStructured(BaseModel):
    """Wave 10 — typed extraction over the transcript.

    Replaces the freeform `key_points` with a structure the UI can
    render as cards (guidance timeline, tone trends, Q&A heatmap).
    All fields default to empty so a partial LLM response still
    validates.
    """
    period: str = ""
    overall_tone: Literal["constructive", "measured", "cautious"] = "measured"
    guidance_changes: list[GuidanceChange] = Field(default_factory=list)
    tone_signals: list[ToneSignal] = Field(default_factory=list)
    qa_themes: list[QAThemeAnalysis] = Field(default_factory=list)
    most_defended_segment: dict[str, str] = Field(default_factory=dict)  # {name, why}
    most_pressed_segment: dict[str, str] = Field(default_factory=dict)
    forward_catalysts: list[dict[str, str]] = Field(default_factory=list)  # [{event, expected_quarter, materiality}]


# W2b 7(b): the critic's read of a PM rating that diverges from the
# valuation evidence. Shared with `memo.RatingReconciliation.critic_assessment`
# so the two can never disagree on vocabulary.
DivergenceAssessment = Literal["supported", "unsupported", "not_assessed"]


# ---------------------------------------------------------------------------
# Bull/bear debate record (D2, 2026-09-25; design-bullbear-final §11).
#
# Expand-only: nothing writes these yet. The debate engine (D3) and graph
# wiring (D6) fill them behind `DEBATE_MODE`; until then every memo carries
# `debate=None`, and every stored memo reads back the same way. The record
# is the whole debate as it ran, not a presentation: the presenter (D4)
# decides what a reader sees, so dropped and unsupported claims stay here
# for the audit and the number check.
# ---------------------------------------------------------------------------

DebateSideName = Literal["bull", "bear"]
# The side-blind grader's verdict on a claim's sourcing (design §4.4).
ClaimGrade = Literal["sourced", "partially_sourced", "analyst_only", "unsupported"]

# Quotes are verified against the excerpt, so the bound is part of the
# contract: a longer excerpt would let a "verified" quote come from text the
# reader is never shown. The engine clamps; the schema refuses.
DEBATE_EXCERPT_MAX_CHARS = 650


class DebateEvidence(BaseModel):
    """One passage or news item in the shared evidence pool (E01..E16).

    Both sides review the same pool; `found_by` records whose research plan
    retrieved it, which the symmetry audit reads, not the grader.
    """
    id: str
    kind: str
    ref: str
    title: str = ""
    date: str = ""
    excerpt: str = Field(default="", max_length=DEBATE_EXCERPT_MAX_CHARS)
    found_by: list[DebateSideName] = Field(default_factory=list)
    query: str = ""


class DebateClaim(BaseModel):
    """One opening claim (≤5 per side), graded side-blind in code."""
    id: str
    side: DebateSideName
    pillar: str = ""
    claim: str
    category: str = "other"
    materiality: Literal["high", "medium", "low"] = "medium"
    evidence: list[str] = Field(default_factory=list)
    # {evidence, text, verified}: a verbatim quote and whether the grader
    # found it in that evidence item's excerpt.
    quote: dict[str, Any] | None = None
    analyst_refs: list[str] = Field(default_factory=list)
    contests_analyst: str | None = None
    falsifier: str = ""
    grade: ClaimGrade = "unsupported"
    dropped: bool = False
    drop_reason: str = ""
    figures: dict[str, int] = Field(default_factory=dict)
    status: Literal["conceded", "partial", "contested", "unanswered"] = "unanswered"


class DebateResponse(BaseModel):
    """One side's rebuttal to one of the other side's claims."""
    side: DebateSideName
    target: str
    stance: Literal["rebut", "concede", "partial", "unanswered"]
    argument: str = ""
    evidence: list[str] = Field(default_factory=list)
    grade: ClaimGrade = "unsupported"


class DebateRuling(BaseModel):
    """The PM's ruling on one decisive dispute."""
    dispute: str
    claim: str
    ruling: Literal["bull", "bear", "split", "unresolved", "not_ruled"] = "not_ruled"
    basis: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class DebateResolution(BaseModel):
    """How the PM resolved the debate. `pm_unavailable` when the keyword
    (no-LLM) PM wrote the memo, so no ruling can be claimed."""
    status: Literal["ruled", "pm_unavailable", "not_applicable"] = "not_applicable"
    crux: str = ""
    rulings: list[DebateRuling] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    relied_unsupported: list[str] = Field(default_factory=list)


class DebateRecord(BaseModel):
    """The bull/bear debate as it ran for one memo (`StockMemoOut.debate`)."""
    protocol_version: int = 1
    status: Literal["complete", "partial", "unavailable", "not_run"] = "not_run"
    # Why a debate is partial or unavailable, e.g. "refused:<category>"
    # when the model declined and pair failover could not recover (P16).
    reason: str = ""
    rebuttal_status: str = ""
    research_status: str = "directed"
    presentation_order: Literal["bull_first", "bear_first"] = "bull_first"
    # {provider, model, effort, failed_over, phases: [...]}. Free-form on
    # purpose: each `phases[]` entry records one phase's outcome, including
    # a refusal and the pair-failover hop, and the engine's typed step
    # payloads (D3) own that shape. The record only has to carry it.
    route: dict[str, Any] = Field(default_factory=dict)
    headlines: dict[str, str] = Field(default_factory=dict)
    cruxes: dict[str, str] = Field(default_factory=dict)
    research: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    evidence: list[DebateEvidence] = Field(default_factory=list)
    claims: list[DebateClaim] = Field(default_factory=list)
    responses: list[DebateResponse] = Field(default_factory=list)
    disputes: list[str] = Field(default_factory=list)
    unanswered: list[str] = Field(default_factory=list)
    resolution: DebateResolution = Field(default_factory=DebateResolution)
    deterministic_checks: list[str] = Field(default_factory=list)
    # Integer tallies (conceded/contested counts, ...) plus the L1
    # counterfactual PM call's `cf_rating_score` / `cf_confidence` /
    # `cf_shift`, which are not integers. The design's `dict[str, int]`
    # would reject (or, in lax mode, truncate) those, so the value type is
    # widened here, before anything writes it. Stored, never displayed.
    outcome: dict[str, float | int] = Field(default_factory=dict)
    # Set by news patches (D5): {at, headline, alert_ref}, capped at 5.
    # Patches never re-run the debate; they note what arrived after it.
    news_since: list[dict[str, Any]] = Field(default_factory=list)
    # calls, tokens_in/out, usd, cap_usd.
    usage: dict[str, float] = Field(default_factory=dict)


class DebateReview(BaseModel):
    """The reviewer's read of how the PM handled the debate (item 8 (3))."""
    dispute_views: list[dict[str, Any]] = Field(default_factory=list)
    unaddressed: list[str] = Field(default_factory=list)
    one_sided: str = ""


# ---------------------------------------------------------------------------
# Full-report reviewer (owner decision 2026-09-25 item 8; D2 contract).
#
# The reviewer judges the whole published report, not only risk: thesis
# logic, evidence quality (with the number-check results), debate handling,
# risks and blind spots, and it returns a verdict with specific issues.
# Every field below is defaulted so the critic reviews already stored — and
# every review written while `REVIEWER_MODE=legacy` — read back unchanged,
# with "" / [] / None meaning "this review did not produce it".
# ---------------------------------------------------------------------------

ReviewVerdict = Literal["", "sound", "sound_with_issues", "unsound"]
ReviewIssueCategory = Literal[
    "thesis_logic", "evidence_quality", "debate_handling", "risk_blind_spot",
    "valuation_consistency",
]
# Which way the issue says the published call leans wrong (L5). A revision
# request must name a direction, so "neutral" is for issues that do not
# bear on the rating (a contradiction in the prose, a missing source).
ReviewIssueDirection = Literal["too_high", "too_low", "neutral"]
REVIEW_ISSUE_TEXT_MAX_CHARS = 300
REVIEW_FIX_REQUEST_MAX_CHARS = 200


class ReviewIssue(BaseModel):
    """One specific issue the reviewer raised.

    The vocabularies are closed (unknown values are refused, not coerced):
    `severity` decides whether an issue triggers the PM revision pass and an
    earned-confidence cap, so an off-list value must never be read as either.
    The writer (D7) downgrades a `material` issue that cites no resolvable
    evidence or deterministic-check id to `minor` before building this.
    """
    id: str
    category: ReviewIssueCategory
    severity: Literal["material", "minor"]
    direction: ReviewIssueDirection = "neutral"
    text: str = Field(max_length=REVIEW_ISSUE_TEXT_MAX_CHARS)
    # Evidence ids (debate E-ids, source-ledger refs) or deterministic-check ids.
    evidence: list[str] = Field(default_factory=list)
    fix_request: str = Field(default="", max_length=REVIEW_FIX_REQUEST_MAX_CHARS)
    # open -> addressed_by_pm (the PM revised) -> resolved (the re-check
    # agreed); rejected_by_pm when the PM kept its call and said why. Only
    # `resolved` stops an issue counting as open.
    status: Literal["open", "addressed_by_pm", "resolved", "rejected_by_pm"] = "open"


class ReviewRatingCase(BaseModel):
    """The reviewer's best case that the rating is too high, or too low (L5).
    Both are asked for on every review so a one-sided review is visible."""
    text: str = ""
    evidence: list[str] = Field(default_factory=list)


class ReviewRecheck(BaseModel):
    """The one bounded re-check of the issues the PM revision addressed.
    `not_run` / `failed` / `skipped_budget` leave every issue open, shown as
    "revised by the PM, not re-reviewed"."""
    status: Literal["not_run", "complete", "failed", "skipped_budget"] = "not_run"
    resolved: list[str] = Field(default_factory=list)
    open: list[str] = Field(default_factory=list)


class ReviewRevision(BaseModel):
    """The one bounded PM revision pass that material issues trigger."""
    status: Literal["not_needed", "revised", "failed", "skipped_budget"] = "not_needed"
    rating_before: str = ""
    rating_after: str = ""
    confidence_before: float | None = None
    confidence_after: float | None = None
    notes: list[str] = Field(default_factory=list)
    recheck: ReviewRecheck = Field(default_factory=ReviewRecheck)


class CriticReview(BaseModel):
    overall_assessment: str
    # Older stored reviews lack provenance; do not relabel them as live.
    review_mode: Literal["unknown", "pending", "live", "rule_based", "unavailable"] = "unknown"
    challenges: list[str] = Field(default_factory=list)
    underweighted_risks: list[str] = Field(default_factory=list)
    suggested_revisions: list[str] = Field(default_factory=list)
    advice_compliance_check: str = "Output framed as research/education only."
    # W2b 7(b). Expand-only (S2): nothing writes it yet. "not_assessed" is
    # the truth for every stored review and for any run with no divergence,
    # so the default reads old snapshots without relabelling them.
    valuation_divergence_assessment: DivergenceAssessment = "not_assessed"
    # Item-8 full-report reviewer (D2, expand-only; D7/R1 write them).
    # "provider:model" that actually served the review, after any failover,
    # so a review is never attributed to a model that did not write it.
    reviewer_model: str = ""
    verdict: ReviewVerdict = ""
    issues: list[ReviewIssue] = Field(default_factory=list)
    rating_too_high: ReviewRatingCase | None = None
    rating_too_low: ReviewRatingCase | None = None
    # Separate from `review_mode` (which says whether a live model answered):
    # this says whether the review counts as INDEPENDENT. A reviewer failure
    # publishes "not independently reviewed" (W2a) rather than canned text.
    review_status: Literal["independent", "not_independent", "rule_based", ""] = ""
    revision: ReviewRevision | None = None
    debate_review: DebateReview | None = None


# ---------------------------------------------------------------------------
# Multi-agent message contracts (Phase 4+)
# Lightweight Pydantic shapes used as the interchange between PM, sectors,
# tool agents, and the monitoring (news/social/macro) loops.
# ---------------------------------------------------------------------------

NewsSeverity = Literal["advisory", "material", "breaking"]


class NewsAlert(BaseModel):
    """Single news/social/macro item pushed into the hot cache."""
    ticker: str | None = None
    sector: str | None = None
    title: str
    summary: str = ""
    url: str = ""
    severity: NewsSeverity = "advisory"
    published_at: str | None = None
    source: str = "news_service"


class MacroBroadcast(BaseModel):
    """Macro snapshot + regime label broadcast to PM and sector agents."""
    snapshot: dict[str, float] = Field(default_factory=dict)
    regime: str = "mixed"
    favored_sectors: list[str] = Field(default_factory=list)
    pressured_sectors: list[str] = Field(default_factory=list)
    note: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SectorQuery(BaseModel):
    """PM → sector: ask a sector for a structured view on `target_ticker`."""
    sector: str
    target_ticker: str
    question: str = ""
    include_peer_relevance: bool = True


class SectorReport(BaseModel):
    """Sector → PM: structured response. Phase 6 populates cross-sector
    relevance so the PM can pull through tickers in *other* sectors."""
    sector: str
    target_ticker: str
    finding: AgentFinding
    cross_sector_relevance: list[str] = Field(default_factory=list)
    macro_alignment: str | None = None


class ToolFinding(BaseModel):
    """Tool agent → sector: an `AgentFinding` plus the tool name used."""
    tool: str
    finding: AgentFinding


class TechnicalSignals(BaseModel):
    """Wave 3B — pure-math indicators surfaced by the Technical Analyst.

    All numeric fields are optional because the indicator may need more
    bars than the price history exposes (e.g., SMA200 needs 200 bars).
    `trend` and `momentum` are best-effort buckets derived from whichever
    indicators came back populated.
    """
    last_price: float | None = None
    last_date: str | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    sma_50_above_200: bool | None = None
    ema_10: float | None = None
    ema_20: float | None = None
    rsi_14: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    bb_middle: float | None = None
    bb_position: float | None = None
    vwma_20: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    position_52w: float | None = None
    trend: Literal["up", "down", "sideways"] = "sideways"
    momentum: Literal["positive", "negative", "neutral"] = "neutral"
    notes: list[str] = Field(default_factory=list)
