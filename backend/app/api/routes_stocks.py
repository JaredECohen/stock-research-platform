"""Stock endpoints — list, detail, memo generation."""
from __future__ import annotations

from datetime import date as _date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import select

from ..agents.graph import run_stock_memo
from ..database import SessionLocal
from ..models import Company
from ..schemas import CompanyOut, StockMemoOut
from ..services import memo_store
from ..services.data_service import get_data_service
from ..services.fundamentals_service import get_full_financials
from ..services.market_data_service import get_basic_stats, get_price_series

router = APIRouter()


def _company_tier(ticker: str) -> str:
    """Look up the universe tier for a ticker; returns 'data_only' if unknown."""
    with SessionLocal() as db:
        row = db.execute(
            select(Company.universe_tier).where(Company.ticker == ticker.upper())
        ).first()
    return (row[0] if row else "data_only") or "data_only"


@router.get("/api/stocks", response_model=List[CompanyOut])
def list_stocks() -> List[CompanyOut]:
    ds = get_data_service()
    # Single bulk-fetch of tier values so we don't run N queries.
    tiers: Dict[str, str] = {}
    with SessionLocal() as db:
        for tkr, tier in db.execute(select(Company.ticker, Company.universe_tier)).all():
            tiers[tkr] = tier or "data_only"
    out: List[CompanyOut] = []
    for ticker in ds.list_tickers():
        profile = ds.get_company_profile(ticker)
        if profile:
            payload = {k: profile.get(k) for k in CompanyOut.model_fields.keys() if k in profile}
            payload["universe_tier"] = tiers.get(ticker, "data_only")
            out.append(CompanyOut(**payload))
    return out


@router.get("/api/stocks/{ticker}")
def get_stock(ticker: str) -> Dict[str, Any]:
    fin = get_full_financials(ticker.upper())
    if not fin.get("profile"):
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")
    stats = get_basic_stats(ticker.upper())
    return dict(
        profile=fin["profile"],
        ratios=fin["ratios"],
        income=fin["income"],
        balance=fin["balance"],
        cash=fin["cash"],
        earnings=fin["earnings"],
        market_stats=stats,
    )


@router.get("/api/stocks/{ticker}/prices")
def get_stock_prices(ticker: str, days: int = 252) -> List[Dict[str, Any]]:
    rows = get_price_series(ticker.upper(), days)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No prices for {ticker}")
    return rows


def _parse_as_of(as_of: Optional[str]) -> Optional[_date]:
    """Parse + validate `?as_of=YYYY-MM-DD`. Future dates are rejected."""
    if not as_of:
        return None
    try:
        d = _date.fromisoformat(as_of)
    except ValueError:
        raise HTTPException(status_code=422,
                            detail=f"as_of must be YYYY-MM-DD; got {as_of!r}")
    if d > _date.today():
        raise HTTPException(status_code=422,
                            detail=f"as_of {d} is in the future")
    return d


@router.get("/api/stocks/{ticker}/memo", response_model=StockMemoOut)
def get_stock_memo(
    ticker: str,
    response: Response,
    scenario: str = "soft_landing",
    ondemand: bool = False,
    as_of: Optional[str] = Query(None, description="YYYY-MM-DD; backtest mode"),
) -> StockMemoOut:
    """Return the latest memo for `ticker`.

    Behavior:
    - If a stored snapshot exists, return it (cheap path) and stamp
      `X-Memo-Version` / `X-Memo-Trigger` / `X-Memo-Generated-At` headers
      so the UI can show "updated 2 days ago because of Q1 2026 earnings".
    - If no snapshot exists, run a full memo synchronously. For tickers in
      the `data_only` tier this requires `ondemand=true` to avoid
      surprise-charging the user; without the flag we 409 so the UI can
      surface an explicit "Analyze this stock" affordance.
    - When `as_of=YYYY-MM-DD` is passed (Wave 1C), the memo is reproduced
      as of that historical date. Backtest results are stored separately
      (won't shadow live memos) and skip long-term memory writes.
    """
    t = ticker.upper()
    as_of_date = _parse_as_of(as_of)

    # Backtest path: skip the cached-snapshot shortcut so we always
    # generate a fresh historical memo.
    if as_of_date is None:
        snap = memo_store.latest_memo(t)
        if snap is not None:
            response.headers["X-Memo-Version"] = str(snap.version)
            response.headers["X-Memo-Trigger"] = snap.trigger
            response.headers["X-Memo-Generated-At"] = snap.generated_at.isoformat()
            return memo_store.memo_to_pydantic(snap)

    tier = _company_tier(t)
    if as_of_date is None and tier == "data_only" and not ondemand:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{t} is in the data-only universe; pass ondemand=true to "
                "trigger the first deep analysis."
            ),
        )
    try:
        memo = run_stock_memo(t, scenario=scenario, as_of_date=as_of_date)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Promote `data_only` → `analyzed_on_demand` so subsequent calls use
    # the cached memo and don't re-trigger an expensive run automatically.
    # Skip promotion when running a backtest — that's diagnostic and
    # shouldn't change the universe state.
    if as_of_date is None and tier == "data_only":
        with SessionLocal() as db:
            row = db.get(Company, t)
            if row:
                row.universe_tier = "analyzed_on_demand"
                db.commit()

    fresh = memo_store.latest_memo(t, include_backtests=as_of_date is not None)
    if fresh is not None:
        response.headers["X-Memo-Version"] = str(fresh.version)
        response.headers["X-Memo-Trigger"] = fresh.trigger
        response.headers["X-Memo-Generated-At"] = fresh.generated_at.isoformat()
        if fresh.as_of_date:
            response.headers["X-Memo-As-Of"] = fresh.as_of_date.date().isoformat() if hasattr(fresh.as_of_date, "date") else str(fresh.as_of_date)
    return memo


@router.get("/api/stocks/{ticker}/memos")
def get_stock_memo_history(ticker: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Memo timeline for `ticker`, newest-first.

    Returns the metadata only (version / trigger / parent_version /
    revision_log / generated_at). Use `?version=N` on the singular memo
    endpoint to fetch a specific version's full body.
    """
    rows = memo_store.memo_history(ticker.upper(), limit=limit)
    return [
        {
            "version": r.version,
            "trigger": r.trigger,
            "parent_version": r.parent_version,
            "generated_at": r.generated_at.isoformat(),
            "revision_log": r.revision_log,
            "rating_label": (r.memo_json or {}).get("rating_label"),
            "confidence_score": (r.memo_json or {}).get("confidence_score"),
        }
        for r in rows
    ]


@router.post("/api/stocks/{ticker}/analyze", response_model=StockMemoOut)
def analyze_stock(
    ticker: str,
    response: Response,
    scenario: Optional[str] = None,
) -> StockMemoOut:
    """Force a fresh full reanalysis. Always creates a new memo version."""
    t = ticker.upper()
    try:
        memo = run_stock_memo(t, scenario=scenario or "soft_landing", force_refresh=True)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    snap = memo_store.latest_memo(t)
    if snap is not None:
        response.headers["X-Memo-Version"] = str(snap.version)
        response.headers["X-Memo-Trigger"] = snap.trigger
        response.headers["X-Memo-Generated-At"] = snap.generated_at.isoformat()
    return memo
