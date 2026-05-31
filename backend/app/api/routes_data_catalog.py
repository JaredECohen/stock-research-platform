"""Data catalog endpoints.

Surfaces the curated sector-overlay catalog (FRED housing + regional,
EIA energy, BLS CPI components + regional labor, Census retail trade)
so the frontend can:

  - Browse what data the sector analyst has access to
  - Filter by sector / sub-industry / region / source / keyword
  - Fetch a snapshot for a specific series with derived stats
  - Inspect the per-ticker sector context the analyst sees

The agent itself uses the same functions directly via
`app.services.data_catalog_service` and `app.agents.sector_tools`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..agents.sector_tools import prepare_sector_context
from ..data_catalog import SERIES_REGISTRY, by_id, list_categories, list_regions, list_sector_tags
from ..services import data_catalog_service, company_geography, sector_overlays
from ..services.data_service import get_data_service

router = APIRouter()


@router.get("/api/data-catalog/series")
def list_series(
    sector: Optional[str] = None,
    sub_industry: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    keyword: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Browse the full catalog, optionally filtered."""
    if any([sector, sub_industry, category, region, source, keyword]):
        return data_catalog_service.discover_by_query(
            sector=sector,
            sub_industry=sub_industry,
            categories=[category] if category else None,
            region=region,
            sources=[source] if source else None,
            keywords=[keyword] if keyword else None,
        )
    return [s.to_dict() for s in SERIES_REGISTRY]


@router.get("/api/data-catalog/series/{series_id}")
def get_series(series_id: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Fetch a single series snapshot with derived stats."""
    spec = by_id(series_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown series_id: {series_id}")
    snap = data_catalog_service.fetch_series(series_id, force_refresh=force_refresh)
    if snap is None:
        raise HTTPException(status_code=502, detail="No provider responded.")
    return snap.to_dict()


@router.get("/api/data-catalog/meta")
def catalog_meta() -> Dict[str, Any]:
    """Enumerate the discrete values that can be filtered on."""
    return {
        "categories": list_categories(),
        "regions": list_regions(),
        "sector_tags": list_sector_tags(),
        "sources": sorted({s.source for s in SERIES_REGISTRY}),
        "total_series": len(SERIES_REGISTRY),
    }


@router.get("/api/data-catalog/ticker/{ticker}/context")
def ticker_context(
    ticker: str,
    overlays: Optional[List[str]] = Query(default=None),
) -> Dict[str, Any]:
    """Return the same context payload the sector analyst sees for a ticker.

    Pass `?overlays=energy&overlays=credit` to override the default sector-
    driven overlay choice.
    """
    sym = ticker.upper().strip()
    # Try to grab the company profile so discovery routes correctly.
    profile: Dict[str, Any] = {}
    try:
        p = get_data_service().get_company_profile(sym)
        if isinstance(p, dict):
            profile = {
                "ticker": sym,
                "sector": p.get("sector"),
                "sub_industry": p.get("sub_industry") or p.get("industry"),
            }
        else:
            profile = {"ticker": sym}
    except Exception:
        profile = {"ticker": sym}
    return prepare_sector_context(sym, profile=profile, overlay_names=overlays)


@router.get("/api/data-catalog/ticker/{ticker}/geography")
def ticker_geography(ticker: str, allow_llm: bool = False) -> Dict[str, Any]:
    """Return the resolved geographic footprint for a ticker."""
    geo = company_geography.get_geography(ticker, allow_llm_fallback=allow_llm)
    if geo is None:
        return {"ticker": ticker.upper(), "available": False, "source": None}
    return {"available": True, **geo}


@router.get("/api/data-catalog/ticker/{ticker}/overlay/{name}")
def ticker_overlay(ticker: str, name: str) -> Dict[str, Any]:
    """Compute a single named overlay for a ticker."""
    sym = ticker.upper().strip()
    profile: Dict[str, Any] = {"ticker": sym}
    try:
        p = get_data_service().get_company_profile(sym)
        if isinstance(p, dict):
            profile.update({
                "sector": p.get("sector"),
                "sub_industry": p.get("sub_industry") or p.get("industry"),
            })
    except Exception:
        pass
    fn = sector_overlays._OVERLAY_FUNCS.get(name)
    if fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown overlay '{name}'. Valid: {list(sector_overlays._OVERLAY_FUNCS)}",
        )
    return fn(sym, profile=profile)
