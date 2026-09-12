"""SEC EDGAR submissions poller.

Runs every 30 minutes (in production). For each ticker it polls, it asks the
EDGAR provider for the latest 10-K / 10-Q / 8-K. When a new accession number
is observed, we invalidate the ticker's `company_cold` snapshot so downstream
warm caches (sector_warm, dcf, comps) auto-stale via their
`parent_snapshot_ids` chain.

Which tickers it polls: the `auto_analysis` tier only (`AUTO_PULL_TIERS`),
not the whole `companies` table.

One provider call per ticker every 30 minutes is a standing cost, and it has
to stay bounded by a curated list rather than by how many names users have
happened to look at. Every manual search on a new ticker inserts an
`analyzed_on_demand` row, so polling the unfiltered table means the automatic
pull universe grows monotonically and never shrinks — 172 companies today,
whatever the search box produced by next quarter.

This constrains ingestion only. Manual research on ANY ticker is untouched:
an on-demand name still gets filings, a memo, and everything else the moment
a user asks for it — it just doesn't earn a permanent slot in the 30-minute
cron. An explicit `tickers=` argument bypasses the tier filter entirely,
which is how tests and admin re-runs drive a specific name through here.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from ..cache import cache_get, cache_put, invalidate
from ..services.data_service import curated_poll_universe
from ..services.filings_service import get_filings
from . import record_run

log = logging.getLogger(__name__)

_FILING_TYPES = {"10-K", "10-Q", "8-K"}


def _seen_accessions(ticker: str) -> set[str]:
    snap = cache_get(ticker, "edgar_seen_accessions")
    if not snap or not isinstance(snap.payload, dict):
        return set()
    return set(snap.payload.get("accessions") or [])


def _save_seen_accessions(ticker: str, accessions: set[str]) -> None:
    cache_put(
        ticker, "edgar_seen_accessions",
        payload={"accessions": sorted(accessions)},
        sources_used=[f"edgar:{ticker}:bookkeeping"],
        generated_by="edgar_poller",
        cost_tokens=0,
        ttl_seconds=365 * 24 * 3600,
    )


def run_once(tickers: Iterable[str] | None = None) -> list[dict]:
    """Poll EDGAR once. Returns a list of `{ticker, new_accessions}` events.

    With no argument, polls the `AUTO_PULL_TIERS` slice of the universe
    (see the module docstring). An explicit `tickers=` is taken as given
    and skips the tier filter — an admin re-run for one name should not
    have to care what tier it is in.

    No-op for tickers without filings. The EDGAR provider returns an empty
    list in demo mode, so this loop becomes a quiet bookkeeping pass.
    """
    excluded: int | None = None
    if tickers is None:
        tickers, excluded = curated_poll_universe()
    tickers = list(tickers)

    events: list[dict] = []
    gate_errors: list[str] = []
    for t in tickers:
        try:
            filings = get_filings(t) or []
        except Exception as exc:
            log.warning("EDGAR poll failed for %s: %s", t, exc)
            continue

        accessions: set[str] = set()
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
            # Wave 5B: hand the new-filing event to the update orchestrator,
            # which enqueues a `full_reanalysis` job on the durable
            # `regen_jobs` queue (memo failures land there, not here).
            # Wrapped so an enqueue failure doesn't block the next
            # ticker's poll.
            try:
                from ..services.update_orchestrator import on_filing_event
                res = on_filing_event(t)
                # `kind="gate_error"` means the auto-regen gate crashed
                # (e.g. a DB error) rather than deciding to skip; count it
                # so the note stops reading as "nothing to do".
                if isinstance(res, dict) and res.get("kind") == "gate_error":
                    gate_errors.append(t)
            except Exception as exc:  # pragma: no cover — diagnostic only
                log.warning("update_orchestrator filing handler failed for %s: %s", t, exc)
        if accessions:
            _save_seen_accessions(t, accessions | seen)

    note = f"{len(events)} new filings"
    # Say the constraint out loud. `excluded is None` means the caller named
    # the tickers, so no tier filter ran and there is nothing to report —
    # printing "0 out-of-tier" there would claim a filter that never fired.
    if excluded is not None:
        note += f"; polled {len(tickers)} in-tier, skipped {excluded} out-of-tier"
    if gate_errors:
        note += f"; gate errors on {len(gate_errors)}: {', '.join(gate_errors[:5])}"
    record_run("edgar_poller", success=not gate_errors, note=note)
    return events


def register(scheduler) -> None:
    """Hook into APScheduler — every 30 minutes."""
    scheduler.add_job(run_once, "interval", minutes=30, id="edgar_poller", replace_existing=True)
