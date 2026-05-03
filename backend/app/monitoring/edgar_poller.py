"""SEC EDGAR submissions poller.

Runs every 30 minutes (in production). For each ticker in the demo universe,
it asks the EDGAR provider for the latest 10-K / 10-Q / 8-K. When a new
accession number is observed, we invalidate the ticker's `company_cold`
snapshot so downstream warm caches (sector_warm, dcf, comps) auto-stale via
their `parent_snapshot_ids` chain.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Set

from ..cache import cache_get, cache_put, invalidate
from ..services.data_service import get_data_service
from ..services.filings_service import get_filings
from . import record_run

log = logging.getLogger(__name__)

_FILING_TYPES = {"10-K", "10-Q", "8-K"}


def _seen_accessions(ticker: str) -> Set[str]:
    snap = cache_get(ticker, "edgar_seen_accessions")
    if not snap or not isinstance(snap.payload, dict):
        return set()
    return set(snap.payload.get("accessions") or [])


def _save_seen_accessions(ticker: str, accessions: Set[str]) -> None:
    cache_put(
        ticker, "edgar_seen_accessions",
        payload={"accessions": sorted(accessions)},
        sources_used=[f"edgar:{ticker}:bookkeeping"],
        generated_by="edgar_poller",
        cost_tokens=0,
        ttl_seconds=365 * 24 * 3600,
    )


def run_once(tickers: Optional[Iterable[str]] = None) -> List[dict]:
    """Poll EDGAR once. Returns a list of `{ticker, new_accessions}` events.

    No-op for tickers without filings. The EDGAR provider returns an empty
    list in demo mode, so this loop becomes a quiet bookkeeping pass.
    """
    if tickers is None:
        ds = get_data_service()
        tickers = ds.list_tickers()

    events: List[dict] = []
    for t in tickers:
        try:
            filings = get_filings(t) or []
        except Exception as exc:
            log.warning("EDGAR poll failed for %s: %s", t, exc)
            continue

        accessions: Set[str] = set()
        for f in filings:
            if f.get("type") not in _FILING_TYPES:
                continue
            acc = f.get("accession_number") or ""
            if acc:
                accessions.add(acc)

        seen = _seen_accessions(t)
        new = accessions - seen
        if new and seen:  # Skip first-run, when seen is empty (initialization)
            invalidate(t, kind="company_cold")
            events.append({"ticker": t, "new_accessions": sorted(new)})
            # Wave 5B: hand the new-filing event to the update orchestrator
            # which enqueues a `full_reanalysis(ticker)`. Wrapped so a memo
            # failure here doesn't block the next ticker's poll.
            try:
                from ..services.update_orchestrator import on_filing_event
                on_filing_event(t)
            except Exception as exc:  # pragma: no cover — diagnostic only
                log.warning("update_orchestrator filing handler failed for %s: %s", t, exc)
        if accessions:
            _save_seen_accessions(t, accessions | seen)

    record_run("edgar_poller", note=f"{len(events)} new filings")
    return events


def register(scheduler) -> None:
    """Hook into APScheduler — every 30 minutes."""
    scheduler.add_job(run_once, "interval", minutes=30, id="edgar_poller", replace_existing=True)
