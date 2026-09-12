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

from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.agents.memo_context import MemoInputs
from app.agents.safe_runner import DegradationLog
from app.models import FinancialPeriod
from app.schemas import (
    AgentFinding,
    BullBearCase,
    CriticReview,
    RiskItem,
    StockMemoOut,
)


def make_finding(agent: str = "Sector Analyst", **overrides: Any) -> AgentFinding:
    base: dict[str, Any] = dict(
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


def make_findings(**overrides: AgentFinding) -> dict[str, AgentFinding]:
    """One plain finding per roster analyst, keyed by `AgentSpec.key` in
    roster order — the shape `AnalystRound.findings` carries."""
    from app.agents.roster import AGENTS
    findings = {spec.key: make_finding(spec.display_name) for spec in AGENTS}
    findings.update(overrides)
    return findings


def make_profile(ticker: str = "TEST", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
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
    base: dict[str, Any] = dict(
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


_STATEMENT_BY_LINE: dict[str, str] | None = None


def _statement_for(line_item: str) -> str:
    """Which statement `history_service` files a line under — the unique
    index is `(ticker, period, statement, line_item)`, so a seeded row
    must land on the same statement the backfill would use."""
    global _STATEMENT_BY_LINE
    if _STATEMENT_BY_LINE is None:
        from app.services import history_service as hs
        _STATEMENT_BY_LINE = {}
        for stmt, lines in (("income", hs._INCOME_LINES), ("balance", hs._BALANCE_LINES),
                            ("cash", hs._CASH_LINES)):
            for line in lines:
                _STATEMENT_BY_LINE[line] = stmt
    return _STATEMENT_BY_LINE.get(line_item, "income")


def make_financial_period(
    ticker: str, fiscal_year: int, line_item: str, value: float | None, *,
    period: str | None = None, period_end: date | None = None,
    fiscal_quarter: int | None = None, statement: str | None = None,
    source: str = "test", fetched_at: datetime | None = None, currency: str = "USD",
) -> FinancialPeriod:
    """One long-format `FinancialPeriod` row, unsaved.

    Defaults to an annual row labelled `FY<year>` with no `period_end`
    (what a demo backfill writes). Pass `period_end` for a dated period
    or `fiscal_quarter` + a `2024Q3` period for a quarterly one.
    """
    return FinancialPeriod(
        ticker=ticker.upper(),
        period=period or f"FY{fiscal_year}",
        period_end=period_end,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        statement=statement or _statement_for(line_item),
        line_item=line_item,
        value=value,
        currency=currency,
        source=source,
        fetched_at=fetched_at or datetime.utcnow(),
    )


def seed_annual_periods(
    db: Session, ticker: str, years: dict[int, dict[str, float | None]], *,
    period_end_month: int | None = 12, source: str = "test",
    fetched_at: datetime | None = None, currency: str = "USD",
) -> int:
    """Seed `{fiscal_year: {line_item: value}}` as annual rows on `db`
    (not committed). `period_end_month=None` leaves `period_end` null.
    Returns the number of rows added."""
    from calendar import monthrange
    n = 0
    for fy, lines in years.items():
        pe = None
        if period_end_month is not None:
            pe = date(fy, period_end_month, monthrange(fy, period_end_month)[1])
        for line, value in lines.items():
            db.add(make_financial_period(
                ticker, fy, line, value, period_end=pe, source=source,
                fetched_at=fetched_at, currency=currency,
            ))
            n += 1
    return n


def make_inputs(ticker: str = "TEST", *, profile: dict[str, Any] | None = None,
                **overrides: Any) -> MemoInputs:
    """A `MemoInputs` with demo-shaped dicts and no DCF / comps."""
    base: dict[str, Any] = dict(
        ticker=ticker, run_id="test-run", scenario="soft_landing", force_refresh=False,
        as_of_date=None, fin={}, profile=profile if profile is not None else make_profile(ticker),
        ratios={}, earnings={}, transcript=None, filings=[], dcf=None, comps=None,
        degradation=DegradationLog(),
    )
    base.update(overrides)
    return MemoInputs(**base)
