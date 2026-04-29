"""News monitoring loop.

For each ticker, throttled to once per hour, calls `news_agent.run` and
pushes the resulting `NewsAlert` records into the hot cache. If any alert
has severity `material` or `breaking`, we ping the relevant sector by
invalidating the sector's warm snapshot — the next sector run will pick up
the fresh news context.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable, List, Optional

from ..cache import cache_get, cache_put, invalidate
from ..services.data_service import get_data_service
from ..agents import news_agent
from . import record_run

log = logging.getLogger(__name__)

_THROTTLE_SECONDS = 60 * 60  # 1 hour per ticker


def _last_run_for(ticker: str) -> Optional[datetime]:
    snap = cache_get(f"news_loop_throttle:{ticker}", "loop_throttle")
    if not snap or not isinstance(snap.payload, dict):
        return None
    ts = snap.payload.get("last_run_at")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _record_run_for(ticker: str) -> None:
    cache_put(
        f"news_loop_throttle:{ticker}", "loop_throttle",
        payload={"last_run_at": datetime.utcnow().isoformat()},
        sources_used=[f"throttle:{ticker}"],
        generated_by="news_loop", cost_tokens=0,
        ttl_seconds=_THROTTLE_SECONDS * 4,
    )


def run_once(tickers: Optional[Iterable[str]] = None) -> List[dict]:
    """Run the news agent for each (un-throttled) ticker. Returns triggered events."""
    if tickers is None:
        ds = get_data_service()
        tickers = list(ds.list_tickers())[:10]  # demo universe sample

    events: List[dict] = []
    for t in tickers:
        last = _last_run_for(t)
        if last and (datetime.utcnow() - last).total_seconds() < _THROTTLE_SECONDS:
            continue
        try:
            alerts = news_agent.run(t, force_refresh=True)
        except Exception as exc:
            log.warning("news_agent failed for %s: %s", t, exc)
            continue
        _record_run_for(t)

        # Material or breaking → invalidate the sector warm snapshot for that
        # ticker's sector so the next sector pass re-incorporates the news.
        material = [a for a in alerts if a.severity in ("material", "breaking")]
        if material:
            ds = get_data_service()
            profile = ds.get_company_profile(t) or {}
            sector = profile.get("sector", "")
            sub_industry = profile.get("sub_industry") or profile.get("industry") or ""
            cache_key = f"{sector}:{sub_industry}:{t}"
            invalidate(cache_key, kind="sector_warm")
            events.append({"ticker": t, "severity_count": len(material)})

    record_run("news_loop", note=f"{len(events)} material events")
    return events


def register(scheduler) -> None:
    scheduler.add_job(run_once, "interval", hours=1, id="news_loop", replace_existing=True)
