"""Wave 10 — forward catalyst calendar.

Surfaces upcoming events that could move a name: earnings, FDA dates,
investor days, industry conferences. Today's implementation pulls
earnings dates from FMP via the existing data service; the schema is
ready for the FDA / conference / investor-day sources called out in
the design review (Phase F follow-up).

Surfaced on:
- The memo via a new "Forward catalysts (next 90d)" section.
- The PM chat via a `get_catalysts(ticker)` tool (registered alongside
  the existing get_memo / get_dcf_summary).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..database import SessionLocal
from ..models import CatalystEvent

log = logging.getLogger(__name__)


def _materiality_for_event(event_type: str) -> str:
    """Default materiality per event type — overridable later by an
    LLM-judged pass."""
    if event_type == "earnings":
        return "high"
    if event_type == "fda":
        return "high"
    if event_type == "investor_day":
        return "medium"
    if event_type == "conference":
        return "low"
    return "medium"


def upsert_earnings_dates_for(ticker: str) -> int:
    """Pull the upcoming earnings date for a ticker from FMP and write
    it to `catalyst_events`. Idempotent on
    (ticker, event_type, event_date, title).

    Returns the count of new + updated rows. 0 when no date is
    available (provider miss, ticker outside coverage).
    """
    try:
        from .data_service import get_data_service
        ds = get_data_service()
        earnings = ds.get_earnings(ticker.upper()) or {}
    except Exception as exc:  # pragma: no cover
        log.debug("get_earnings failed for %s: %s", ticker, exc)
        return 0
    next_date_str = earnings.get("next_earnings_date") or ""
    if not next_date_str:
        return 0
    try:
        d = date.fromisoformat(str(next_date_str)[:10])
    except Exception:
        return 0
    title = f"{ticker.upper()} earnings"
    description = "Quarterly earnings call (date per FMP calendar)."
    written = 0
    with SessionLocal() as db:
        existing = db.execute(
            select(CatalystEvent).where(
                CatalystEvent.ticker == ticker.upper(),
                CatalystEvent.event_type == "earnings",
                CatalystEvent.event_date == d,
                CatalystEvent.title == title,
            )
        ).scalars().first()
        if existing is None:
            db.add(CatalystEvent(
                ticker=ticker.upper(),
                event_type="earnings",
                event_date=d,
                title=title,
                description=description,
                materiality=_materiality_for_event("earnings"),
                source="fmp",
                fetched_at=datetime.utcnow(),
            ))
            written += 1
        else:
            existing.description = description
            existing.materiality = _materiality_for_event("earnings")
            existing.fetched_at = datetime.utcnow()
        db.commit()
    return written


def get_upcoming(
    ticker: str, *, days_ahead: int = 90,
) -> List[Dict[str, Any]]:
    """Forward calendar for a single ticker — `days_ahead` window.

    Returns dicts ready for memo / chat consumption.
    """
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    with SessionLocal() as db:
        rows = db.execute(
            select(CatalystEvent)
            .where(
                CatalystEvent.ticker == ticker.upper(),
                CatalystEvent.event_date >= today,
                CatalystEvent.event_date <= cutoff,
            )
            .order_by(CatalystEvent.event_date)
        ).scalars().all()
    return [
        {
            "ticker": r.ticker,
            "event_type": r.event_type,
            "event_date": r.event_date.isoformat() if r.event_date else None,
            "title": r.title,
            "description": r.description,
            "materiality": r.materiality,
            "source": r.source,
        }
        for r in rows
    ]


def refresh_universe(*, limit: Optional[int] = None) -> Dict[str, int]:
    """Pull earnings dates for the screener universe. Suitable for a
    daily cron. Returns a small summary dict."""
    from ..models import Company
    with SessionLocal() as db:
        q = db.query(Company.ticker).filter(Company.universe_tier == "auto_analysis")
        tickers = [t for (t,) in q.all()]
    if limit is not None:
        tickers = tickers[:limit]
    written = 0
    for t in tickers:
        try:
            written += upsert_earnings_dates_for(t)
        except Exception as exc:  # pragma: no cover
            log.debug("catalyst refresh failed for %s: %s", t, exc)
    return {"tickers_checked": len(tickers), "rows_written": written}
