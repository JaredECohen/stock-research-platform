"""Memo schemas — mispricing thesis, valuation verdict, `StockMemoOut`.

Every class referenced by a `StockMemoOut` annotation is imported here
explicitly: under `from __future__ import annotations` pydantic resolves
forward references from this module's globals, so a name that only lives
in the package `__init__` would fail to resolve at model build time.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .agents import (
    AgentFinding,
    BullBearCase,
    CatalystItem,
    CriticReview,
    RiskItem,
    RoundFindings,
)
from .common import RatingLabel


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
    falsifiers: List[str] = Field(default_factory=list)


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
    verdict: Literal["undervalued", "fairly_priced", "overvalued"] = "fairly_priced"
    dcf_base_upside: Optional[float] = None
    comps_ev_ebitda_premium: Optional[float] = None
    factor_valuation: Optional[float] = None
    summary: str = ""


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
    price_at_memo: Optional[float] = None
    price_at_memo_at: Optional[datetime] = None
    business_summary: str
    sector_agent_view: AgentFinding
    earnings_agent_view: AgentFinding
    filing_agent_view: AgentFinding
    valuation_agent_view: AgentFinding
    comps_agent_view: AgentFinding
    macro_sensitivity: AgentFinding
    # Wave 3B — Technical Analyst. Optional so older memos that pre-date
    # this addition still validate; the graph populates it on every run.
    technical_agent_view: Optional[AgentFinding] = None
    bull_case: BullBearCase
    bear_case: BullBearCase
    catalysts: List[CatalystItem]
    key_risks: List[RiskItem]
    thesis_breakers: List[RiskItem]
    dcf_summary: Dict[str, Any] = Field(default_factory=dict)
    # Wave 10 — initial (consensus-anchored) DCF kept alongside the
    # PM-adjusted view in `dcf_summary`. Empty when the PM made no
    # adjustments or no LLM was available. Lets the UI show "what the
    # team's research changed about the model".
    dcf_initial_summary: Dict[str, Any] = Field(default_factory=dict)
    dcf_pm_adjustments: List[Dict[str, Any]] = Field(default_factory=list)
    dcf_pm_adjustment_headline: str = ""
    portfolio_fit: str = ""
    risk_committee_challenge: CriticReview
    final_verdict: str
    scores: Dict[str, float] = Field(default_factory=dict)
    sources_used: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_mode: Literal["demo", "live"] = "demo"
    # Wave 9 — full diligence-dialog audit trail. `round=0` is the
    # initial parallel fan-out; rounds 1+ are PM↔specialist
    # critique-and-revise turns. Empty list when deep_research is off.
    round_findings: List[RoundFindings] = Field(default_factory=list)
    # Wave 10 — forward catalysts (next 90d) populated from
    # `catalyst_events`. Each item: {ticker, event_type, event_date,
    # title, description, materiality, source}.
    forward_catalysts: List[Dict[str, Any]] = Field(default_factory=list)
    # Wave 10 — earnings quarter-over-quarter delta (separate finding;
    # rendered as its own UI tile). Optional — None when prior-quarter
    # data isn't available yet.
    earnings_qoq_delta: Optional[AgentFinding] = None
    # Wave 10 — PM intake decision (which specialists were skipped and
    # why). `{skipped: List[str], rationale: str}`. Empty dict when
    # all 8 ran (the default). Audit trail for cost-aware memo runs.
    intake_decision: Dict[str, Any] = Field(default_factory=dict)
    # Wave 10 — per-agent influence on the rating. Computed post-PM
    # synthesis from each agent's confidence + tone; values are signed
    # contributions (positive = bullish pull, negative = bearish pull),
    # roughly normalized so the largest |value| is the most-influential
    # agent on this memo. Empty on memos that pre-date the field.
    agent_influence: Dict[str, float] = Field(default_factory=dict)
    # Wave 10 — macro context frozen at memo creation. Lets the
    # postmortem regime-conditional dashboards bucket memos by the
    # regime that was active when they were written, even if the macro
    # broadcast cache has rolled over by the time outcomes evaluate.
    macro_snapshot_at_memo: Dict[str, float] = Field(default_factory=dict)
    macro_regime_at_memo: str = ""
    # List of agents that failed during this memo's generation. Empty when
    # everything ran normally; populated by the safe-runner so the UI can
    # show "X analyst was unavailable" rather than dropping the memo.
    degraded_agents: List[str] = Field(default_factory=list)
    # RP-001 — the *reason* behind each `degraded_agents` entry, in the
    # same order: `{agent, error_type, message}` (the shape of
    # `DegradationLog.failures`). Lets the guard tests and a future UI
    # tooltip say why a section is thin instead of only that it is.
    # Empty on memos that pre-date the field.
    degradation_events: List[Dict[str, Any]] = Field(default_factory=list)
    # RP-003 landing zone — a roster agent with no dedicated memo field
    # (a future Industry Group analyst) lands here keyed by its roster key,
    # so adding an analyst does not require a schema edit. Empty for the
    # current eight-agent roster.
    extra_agent_views: Dict[str, AgentFinding] = Field(default_factory=dict)
    disclaimer: str = (
        "MarketMosaic is for investment research and education only. "
        "It does not provide personalized financial, investment, legal, or tax advice."
    )
