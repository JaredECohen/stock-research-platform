"""Screener endpoints."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter

from ..schemas import ScreenerRequest, ScreenerResult
from ..services.screener_service import compute_universe_scores

router = APIRouter()


@router.get("/api/screener", response_model=ScreenerResult)
def get_screener(theme: Optional[str] = None, sector: Optional[str] = None, limit: int = 50) -> ScreenerResult:
    result = compute_universe_scores(theme=theme)
    if sector:
        result.rows = [r for r in result.rows if sector.lower() in r.sector.lower()]
    if limit:
        result.rows = result.rows[: limit]
    return result


@router.post("/api/screener/run", response_model=ScreenerResult)
def run_screener(req: ScreenerRequest) -> ScreenerResult:
    result = compute_universe_scores(theme=req.theme)
    if req.sectors:
        wanted = {s.lower() for s in req.sectors}
        result.rows = [r for r in result.rows if r.sector.lower() in wanted]
    if req.limit:
        result.rows = result.rows[: req.limit]
    return result
