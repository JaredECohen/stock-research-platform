"""News monitoring loop.

For each ticker, throttled to once per hour, calls `news_agent.run` and
pushes the resulting `NewsAlert` records into the hot cache. If any alert
has severity `material` or `breaking`, we ping the relevant sector by
invalidating the sector's warm snapshot — the next sector run will pick up
the fresh news context.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

from ..agents import news_agent
from ..cache import cache_get, cache_put, invalidate
from ..services.data_service import get_data_service
from . import record_run
from .research_focus import select_focus

log = logging.getLogger(__name__)

_THROTTLE_SECONDS = 60 * 60  # 1 hour per ticker

# How many tickers one run may cover.
#
# Cost math: this loop runs hourly, and `news_agent.run(force_refresh=True)`
# makes one Gemini call per ticker (`settings.gemini_news_model`). So the
# steady-state spend is budget x 24 calls/day — at 10, that is 240 Gemini
# calls a day. Raising it to ~25 (600/day) would cover every ticker that
# carries any research signal at all today: the 10 pins plus the 17 that
# have ever had a memo generated, minus the overlap.
#
# Deliberately left at 10, which is exactly what the old arbitrary
# `list_tickers()[:10]` slice spent. This change is about WHICH ten, not
# how many; raising it is a spend decision for the owner, not a side effect
# of a relevance fix.
NEWS_FOCUS_BUDGET = 10


def _last_run_for(ticker: str) -> datetime | None:
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


def run_once(tickers: Iterable[str] | None = None) -> list[dict]:
    """Run the news agent for each (un-throttled) ticker. Returns triggered events."""
    selection = None
    if tickers is None:
        # Relevance-ranked rather than an arbitrary universe slice — see
        # `research_focus` for why the old `list_tickers()[:10]` was wrong.
        #
        # On the rotation vs. `_THROTTLE_SECONDS`: the band-3 tail window
        # advances one position per hour while the loop also runs hourly,
        # so a stale ticker is selected on `budget - len(head)` CONSECUTIVE
        # hourly runs before it rotates out. That matters because the
        # throttle and the interval are both exactly one hour: a run that
        # fires a few seconds early finds `elapsed < 3600` and skips the
        # ticker. With a window wider than one slot, the next hour's run
        # picks it up — a ticker that appears for only a single hour could
        # be throttled out of existence forever.
        selection = select_focus(budget=NEWS_FOCUS_BUDGET)
        tickers = selection.tickers

    events: list[dict] = []
    assessment_failures = 0
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

            # Wave 5B: hand each material/breaking alert to the update
            # orchestrator. It gates on prior-memo presence + daily patch
            # cap + news_impact_agent's materiality verdict, and only
            # writes a patched snapshot when all gates pass.
            try:
                from ..services.update_orchestrator import on_news_alert
                for alert in material:
                    res = on_news_alert(t, alert)
                    # A dead news-impact LLM used to read as "no material
                    # news"; the handler now reports it and the note counts it.
                    if isinstance(res, dict) and res.get("reason") == "assessment_error":
                        assessment_failures += 1
            except Exception as exc:  # pragma: no cover — diagnostic only
                log.warning("update_orchestrator failed for %s: %s", t, exc)

    note = f"{len(events)} material events"
    if assessment_failures:
        note += f"; {assessment_failures} assessments failed"
    if selection is not None:
        # Folded in so cron-health shows what this run covered and, more
        # importantly, which qualifying tickers the budget could not reach.
        note += f"; {selection.note()}"
    record_run("news_loop", success=assessment_failures == 0, note=note)
    return events


def register(scheduler) -> None:
    scheduler.add_job(run_once, "interval", hours=1, id="news_loop", replace_existing=True)
