"""Earnings transcript poller — daily cron.

Mirrors `edgar_poller.py` for transcripts. Iterates every ticker in
`Company.universe_tier == 'auto_analysis'`, calls
`transcripts_service.latest_transcript`, and detects new periods by
comparing against a per-ticker `transcripts_seen_periods` cache row.

When a new transcript is observed:
  - The cache_put updates the seen set so we don't re-fire.
  - `update_orchestrator.on_transcript_event(ticker)` is invoked.
    That handler applies the same gating as filings — only the pinned
    auto-update tickers + recently-viewed memos trigger a full memo
    regen; everyone else just gets the raw transcript persisted
    (already in provider_cache).

Why daily (not 30-min like EDGAR): transcripts publish weeks after
quarter-end, on a schedule that's known well in advance. Polling
every 30 minutes burns FMP rate-limit budget for no signal. Daily at
06:00 UTC catches typical pre-market US releases.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Set

from ..cache import cache_get, cache_put
from ..monitoring import record_run
from ..services.data_service import get_data_service
from ..services.transcripts_service import get_transcripts

log = logging.getLogger(__name__)


def _seen_periods(ticker: str) -> Set[str]:
    snap = cache_get(ticker, "transcripts_seen_periods")
    if snap is None or not isinstance(snap.payload, dict):
        return set()
    return set(snap.payload.get("periods", []))


def _save_seen_periods(ticker: str, periods: Set[str]) -> None:
    cache_put(
        ticker, "transcripts_seen_periods",
        payload={"periods": sorted(periods)},
        sources_used=[f"transcripts:{ticker}:bookkeeping"],
        generated_by="transcripts_poller",
    )


def run_once(tickers: Optional[Iterable[str]] = None) -> List[dict]:
    """Poll for new earnings transcripts once. Returns
    `[{ticker, new_periods, regenerated}]` events.

    `regenerated` is the truthy result of
    `update_orchestrator.on_transcript_event` — `None` when the
    gating decided to skip memo regen (transcript persisted, no memo
    burned), or a dict with the regenerated memo's rating when the
    ticker is pinned / actively viewed.
    """
    if tickers is None:
        ds = get_data_service()
        tickers = ds.list_tickers()

    events: List[dict] = []
    for t in tickers:
        try:
            transcripts = get_transcripts(t) or []
        except Exception as exc:
            log.warning("transcript poll failed for %s: %s", t, exc)
            continue

        # Period is the canonical key — providers return e.g.
        # "2025Q4". We track the set of periods seen per ticker.
        periods: Set[str] = set()
        for tr in transcripts:
            p = tr.get("period") or tr.get("date") or ""
            if isinstance(p, str) and p:
                periods.add(p)

        seen = _seen_periods(t)
        new = periods - seen
        if new and seen:  # skip first-run init
            for period in sorted(new):
                regenerated = None
                try:
                    from ..services.update_orchestrator import on_transcript_event
                    regenerated = on_transcript_event(t, period=period)
                except Exception as exc:  # pragma: no cover — diagnostic
                    log.warning(
                        "update_orchestrator transcript handler failed for %s/%s: %s",
                        t, period, exc,
                    )
                events.append({
                    "ticker": t,
                    "period": period,
                    "regenerated": regenerated,
                })
        if periods:
            _save_seen_periods(t, periods | seen)

    record_run("transcripts_poller", note=f"{len(events)} new transcripts")
    return events


def register(scheduler) -> None:
    """Hook into APScheduler — daily at 06:00 UTC.

    Picked 06:00 UTC because most US-large-cap earnings calls release
    pre-market 13:00-14:00 UTC (8-9 AM ET); polling well before that
    lets the next-day cron pick up the entire prior evening's after-
    market release window."""
    scheduler.add_job(
        run_once, "cron", hour=6, minute=0,
        id="transcripts_poller", replace_existing=True,
    )
