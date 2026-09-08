"""Builders for the backend test suite.

`StockMemoOut` has eight required `AgentFinding` fields plus a
`CriticReview`, and until RP-002 no backend test could build one without
running the whole pipeline (`frontend/src/test/fixtures/memo.ts` is the
only fixture, and it is TypeScript). These helpers build a *valid* memo /
finding / stage input with minimal content so a stage function can be
exercised directly. Every builder takes `**overrides` that replace fields
after the defaults are laid down.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.agents.memo_context import MemoInputs
from app.agents.safe_runner import DegradationLog
from app.schemas import (
    AgentFinding,
    BullBearCase,
    CriticReview,
    RiskItem,
    StockMemoOut,
)


def make_finding(agent: str = "Sector Analyst", **overrides: Any) -> AgentFinding:
    base: Dict[str, Any] = dict(
        agent=agent,
        headline=f"{agent} headline",
        summary=f"{agent} summary.",
        key_points=[],
        confidence=0.7,
        sources=[],
        data={},
    )
    base.update(overrides)
    return AgentFinding(**base)


def make_findings(**overrides: AgentFinding) -> Dict[str, AgentFinding]:
    """One plain finding per roster analyst, keyed by `AgentSpec.key` in
    roster order — the shape `AnalystRound.findings` carries."""
    from app.agents.roster import AGENTS
    findings = {spec.key: make_finding(spec.display_name) for spec in AGENTS}
    findings.update(overrides)
    return findings


def make_profile(ticker: str = "TEST", **overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "ticker": ticker,
        "company_name": f"{ticker} Corp",
        "sector": "Technology",
        "industry": "Software",
        "business_description": f"{ticker} makes software.",
        "drivers": ["Cloud migration"],
        "risks": ["Pricing pressure"],
    }
    base.update(overrides)
    return base


def make_memo(**overrides: Any) -> StockMemoOut:
    """A valid memo with one plain finding per view and no verdict fields set.

    `rating_label` defaults to "Neutral"; `valuation_verdict`,
    `mispricing_thesis` and `final_verdict` default to empty so a test can
    assert what `_build_verdict` fills in.
    """
    base: Dict[str, Any] = dict(
        ticker="TEST",
        company_name="TEST Corp",
        sector="Technology",
        final_pm_view="Research view: Neutral.",
        rating_label="Neutral",
        confidence_score=60.0,
        one_sentence_thesis="",
        business_summary="TEST makes software.",
        sector_agent_view=make_finding("Sector Analyst"),
        earnings_agent_view=make_finding("Earnings Analyst"),
        filing_agent_view=make_finding("Filing Analyst"),
        valuation_agent_view=make_finding("Valuation Analyst"),
        comps_agent_view=make_finding("Comps Analyst"),
        macro_sensitivity=make_finding("Macro Analyst"),
        technical_agent_view=make_finding("Technical Analyst"),
        bull_case=BullBearCase(headline="Bull case", key_points=["Tailwind: cloud"]),
        bear_case=BullBearCase(headline="Bear case", key_points=["Pricing pressure"]),
        catalysts=[],
        key_risks=[RiskItem(title="Pricing pressure", detail="Competitors discount.",
                            severity="medium", type="company")],
        thesis_breakers=[RiskItem(title="Cloud slowdown", detail="Migration stalls.",
                                  severity="high", type="thesis_breaker")],
        dcf_summary={},
        risk_committee_challenge=CriticReview(overall_assessment="No objections."),
        final_verdict="",
        scores={"factor_pm_score": 55.0},
    )
    base.update(overrides)
    return StockMemoOut(**base)


def make_inputs(ticker: str = "TEST", *, profile: Optional[Dict[str, Any]] = None,
                **overrides: Any) -> MemoInputs:
    """A `MemoInputs` with demo-shaped dicts and no DCF / comps."""
    base: Dict[str, Any] = dict(
        ticker=ticker, run_id="test-run", scenario="soft_landing", force_refresh=False,
        as_of_date=None, fin={}, profile=profile if profile is not None else make_profile(ticker),
        ratios={}, earnings={}, transcript=None, filings=[], dcf=None, comps=None,
        degradation=DegradationLog(),
    )
    base.update(overrides)
    return MemoInputs(**base)
