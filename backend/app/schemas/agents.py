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
    """
    target_agent: Literal[
        "sector", "earnings", "valuation", "comps",
        "risk", "filing", "macro", "technical",
    ]
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


class CriticReview(BaseModel):
    overall_assessment: str
    challenges: list[str] = Field(default_factory=list)
    underweighted_risks: list[str] = Field(default_factory=list)
    suggested_revisions: list[str] = Field(default_factory=list)
    advice_compliance_check: str = "Output framed as research/education only."


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
