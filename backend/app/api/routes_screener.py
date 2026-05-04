"""Screener endpoints — AI-first, factor-rank, and rule-based custom screen."""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from ..database import SessionLocal
from ..models import Company, ScreenerMetric, ScreenerScore
from ..schemas import (
    CustomScreenRequest,
    CustomScreenResult,
    CustomScreenRow,
    ScreenerRequest,
    ScreenerResult,
)
from ..services.screener_service import compute_universe_scores

router = APIRouter()


# Whitelisted score / metric columns we'll sort by. Anything else is
# rejected to keep this endpoint from accidentally exposing internals.
_AI_SORT_COLUMNS = {
    "pm_score", "quality", "growth", "valuation",
    "earnings_momentum", "risk", "macro_fit",
}


def _apply_sort(rows, sort_by: Optional[str], order: str = "desc"):
    if not sort_by or sort_by not in _AI_SORT_COLUMNS:
        sort_by = "pm_score"
    rev = order != "asc"
    rows.sort(key=lambda r: getattr(r, sort_by, 0) or 0, reverse=rev)
    for i, r in enumerate(rows, start=1):
        r.rank = i


@router.get("/api/screener", response_model=ScreenerResult)
def get_screener(
    theme: Optional[str] = None,
    sector: Optional[str] = None,
    sort_by: Optional[str] = "pm_score",
    order: str = "desc",
    limit: int = 50,
) -> ScreenerResult:
    """AI-first screen + factor-rank.

    `sort_by` accepts any of the seven score columns
    (pm_score, quality, growth, valuation, earnings_momentum, risk,
    macro_fit). Anything else falls back to `pm_score`. Pass
    `?sort_by=quality&order=desc` to render the "Factor Rank" view.
    """
    result = compute_universe_scores(theme=theme)
    if sector:
        result.rows = [r for r in result.rows if sector.lower() in r.sector.lower()]
    _apply_sort(result.rows, sort_by, order)
    if limit:
        result.rows = result.rows[:limit]
    return result


@router.post("/api/screener/run", response_model=ScreenerResult)
def run_screener(req: ScreenerRequest) -> ScreenerResult:
    result = compute_universe_scores(theme=req.theme)
    if req.sectors:
        wanted = {s.lower() for s in req.sectors}
        result.rows = [r for r in result.rows if r.sector.lower() in wanted]
    _apply_sort(result.rows, req.sort_by, "desc")
    if req.limit:
        result.rows = result.rows[: req.limit]
    return result


# ---------------------------------------------------------------------------
# Custom rule-based screen (Wave 9b Phase 4)
# ---------------------------------------------------------------------------

_OP_TO_FN = {
    ">":  lambda col, v: col > v,
    "<":  lambda col, v: col < v,
    ">=": lambda col, v: col >= v,
    "<=": lambda col, v: col <= v,
    "=":  lambda col, v: col == v,
}


@router.post("/api/screener/custom", response_model=CustomScreenResult)
def run_custom_screen(req: CustomScreenRequest) -> CustomScreenResult:
    """Filter the curated S&P 100 against a user-defined rule set.

    Each rule is `{metric, op, value}` (or `{op: "between", value, value2}`).
    Rules are AND-combined. Tickers are restricted to the curated
    `auto_analysis` universe so research-on-demand names don't leak in
    (per the locked decision in `docs/UNIVERSE_REFACTOR_PLAN.md`).

    Rows with NULL metrics fail the rule (rather than being dropped or
    treated as 0). Sort order is configurable; default = market_cap desc.
    """
    metric_names: List[str] = list({r.metric for r in req.rules})
    metric_names.append(req.sort_by)

    with SessionLocal() as db:
        # Join on Company so we can return company_name/sector and gate
        # on universe_tier in one round trip.
        query = (
            select(ScreenerMetric, Company.company_name, Company.sector)
            .join(Company, Company.ticker == ScreenerMetric.ticker)
            .where(Company.universe_tier == "auto_analysis")
        )
        if req.sectors:
            wanted = [s.lower() for s in req.sectors]
            from sqlalchemy import func
            query = query.where(func.lower(Company.sector).in_(wanted))

        for rule in req.rules:
            col = getattr(ScreenerMetric, rule.metric, None)
            if col is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown metric: {rule.metric}",
                )
            if rule.op == "between":
                if rule.value2 is None:
                    raise HTTPException(
                        status_code=422,
                        detail="`between` requires both `value` and `value2`",
                    )
                lo, hi = sorted([rule.value, rule.value2])
                query = query.where(col.is_not(None)).where(col >= lo).where(col <= hi)
            else:
                fn = _OP_TO_FN.get(rule.op)
                if fn is None:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Unsupported op: {rule.op}",
                    )
                # NULL fails the rule — explicit IS NOT NULL guard so SQL
                # tristate doesn't silently drop the row from comparison.
                query = query.where(col.is_not(None)).where(fn(col, rule.value))

        sort_col = getattr(ScreenerMetric, req.sort_by)
        if req.order == "asc":
            query = query.order_by(sort_col.asc().nulls_last())
        else:
            query = query.order_by(sort_col.desc().nulls_last())

        results = db.execute(query.limit(req.limit)).all()

        # Pull AI scores in one query so we can render PM conviction
        # alongside the raw metrics.
        tickers = [m.ticker for m, _, _ in results]
        score_lookup: Dict[str, ScreenerScore] = {}
        if tickers:
            score_rows = db.execute(
                select(ScreenerScore).where(
                    ScreenerScore.ticker.in_(tickers),
                    ScreenerScore.theme.is_(None),
                )
            ).scalars().all()
            score_lookup = {s.ticker: s for s in score_rows}

    rows: List[CustomScreenRow] = []
    for m, company_name, sector in results:
        score = score_lookup.get(m.ticker)
        # Surface every metric the user filtered on plus the sort column;
        # frontend renders these as columns in the result grid.
        metrics: Dict[str, Optional[float]] = {}
        for name in metric_names:
            metrics[name] = getattr(m, name, None)
        rows.append(CustomScreenRow(
            ticker=m.ticker,
            company_name=company_name,
            sector=sector,
            pm_score=score.pm_conviction if score else None,
            rating_label=None,  # rating lives on memo, not screener_score
            metrics=metrics,
        ))

    return CustomScreenResult(
        rows=rows,
        rule_count=len(req.rules),
        matched=len(rows),
    )
