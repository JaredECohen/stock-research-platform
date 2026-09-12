"""SEC EDGAR submissions poller.

Runs every 30 minutes (in production). For each ticker it polls, it reads the
**filing index** — accession numbers, form types and dates, no document text
(`filings_service.get_filings_index`). When a new accession number is observed,
we invalidate the ticker's `company_cold` snapshot so downstream warm caches
(sector_warm, dcf, comps) auto-stale via their `parent_snapshot_ids` chain, and
drop the cached filing *bodies* for that ticker so the next full read fetches
the new document rather than serving the previous one.

The index / bodies split is what makes a 30-minute cadence affordable.
`data_service.get_filings` fetches the full text of every form it returns — up
to ten per ticker, a few MB each, paced against SEC's ~10 req/s limit — so
polling the curated universe through it would mean ~1,700 document downloads
every pass on a 512 MiB worker that has already been OOM-killed twice. The
index is one submissions.json read per ticker.

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

Handing an event to the orchestrator is the expensive half, and it is capped
per pass — see `MAX_FILING_EVENTS_PER_PASS`.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from ..cache import cache_get, cache_put, invalidate
from ..services.data_service import curated_poll_universe
from ..services.filings_service import get_filings_index, invalidate_filings_text
from . import note_names, record_run

log = logging.getLogger(__name__)

_FILING_TYPES = {"10-K", "10-Q", "8-K"}

# How many tickers one pass may hand to `on_filing_event`.
#
# An event is not a cheap notification. `on_filing_event` calls
# `_persist_raw_data_only` BEFORE its auto-regen gate, which reads every
# filing body for the ticker, writes them to `filing_docs`, embeds the new
# chunks, and runs `filing_memory.post_pass` — whose own docstring says it
# uses the LLM for the diff bullets. Call it ~$0.02 of embeddings plus an LLM
# round-trip and up to ten document downloads per ticker.
#
# That matters most on the first pass after the cache fix: the accession list
# had been frozen since the deployment's first read, so ~166 tickers each
# surface a month or more of filings at once. Unbounded, that is one
# simultaneous burst of downloads, embeddings and LLM calls on the worker that
# Render has already OOM-killed twice.
#
# 15 per pass drains that backlog in ~12 passes — under six hours at the
# 30-minute cadence — while keeping the per-pass cost in the tens of LLM calls
# rather than the hundreds. Steady state is far below the cap: a normal pass
# sees a handful of 8-Ks across the whole universe.
MAX_FILING_EVENTS_PER_PASS = 15

# Upper bound on the per-ticker `edgar_seen_accessions` bookkeeping set.
#
# It used to be `accessions | seen`, unioned on every pass and never pruned, so
# the row grew for the life of the deployment. The set only has to remember
# enough history that a filing which has scrolled out of the provider's window
# is not re-offered as new, and the SEC provider returns at most 10 filings per
# ticker. 50 is five of those windows — years of filing history for a typical
# large cap, a couple of KB on disk, and impossible to overflow in one pass.
MAX_SEEN_ACCESSIONS = 50


def _seen_accessions(ticker: str) -> set[str]:
    snap = cache_get(ticker, "edgar_seen_accessions")
    if not snap or not isinstance(snap.payload, dict):
        return set()
    return set(snap.payload.get("accessions") or [])


def _bounded_seen(accessions: set[str], seen: set[str]) -> set[str]:
    """`accessions | seen`, pruned deterministically to MAX_SEEN_ACCESSIONS.

    Everything in `accessions` — the provider's current window — is kept
    unconditionally, whatever the cap says: dropping one of those would make
    the very next pass see it as new and re-fire the event. Only the older
    remainder is pruned, newest-first by accession number, which sorts
    chronologically within a filer.
    """
    merged = accessions | seen
    if len(merged) <= MAX_SEEN_ACCESSIONS:
        return merged
    keep = set(accessions)
    for acc in sorted(seen - accessions, reverse=True):
        if len(keep) >= MAX_SEEN_ACCESSIONS:
            break
        keep.add(acc)
    return keep


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

    At most `MAX_FILING_EVENTS_PER_PASS` tickers are handed to the
    orchestrator. A ticker over the cap is *deferred*, not dropped: its
    bookkeeping is left untouched, so the next pass still sees its
    accessions as new and picks it up. Deferrals are named in the run note.

    No-op for tickers without filings. The EDGAR provider returns an empty
    list in demo mode, so this loop becomes a quiet bookkeeping pass.
    """
    excluded: int | None = None
    if tickers is None:
        tickers, excluded = curated_poll_universe()
    tickers = list(tickers)

    events: list[dict] = []
    gate_errors: list[str] = []
    deferred: list[str] = []
    for t in tickers:
        try:
            filings = get_filings_index(t) or []
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
            if len(events) >= MAX_FILING_EVENTS_PER_PASS:
                # Over the cap. Everything below this point — including the
                # bookkeeping write at the bottom of the loop — is skipped on
                # purpose. Recording these accessions as seen without having
                # processed them would lose the event permanently: the next
                # pass would compute an empty diff and nothing would ever fire
                # for this filing again.
                deferred.append(t)
                continue
            invalidate(t, kind="company_cold")
            # Detection reads the index; every downstream reader reads the
            # bodies. Forget the cached bodies for this ticker so the full
            # read that follows fetches the new document instead of serving
            # the one that was cached before it existed.
            try:
                invalidate_filings_text(t)
            except Exception as exc:  # pragma: no cover — diagnostic only
                log.warning("filings cache invalidation failed for %s: %s", t, exc)
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
            _save_seen_accessions(t, _bounded_seen(accessions, seen))

    note = f"{len(events)} new filings"
    # Say the constraint out loud. `excluded is None` means the caller named
    # the tickers, so no tier filter ran and there is nothing to report —
    # printing "0 out-of-tier" there would claim a filter that never fired.
    if excluded is not None:
        note += f"; polled {len(tickers)} in-tier, skipped {excluded} out-of-tier"
    if deferred:
        note += f"; deferred {len(deferred)} over the {MAX_FILING_EVENTS_PER_PASS}"
        note += f"-event cap: {note_names(deferred)}"
    if gate_errors:
        note += f"; gate errors on {len(gate_errors)}: {', '.join(gate_errors[:5])}"
    record_run("edgar_poller", success=not gate_errors, note=note)
    return events


def register(scheduler) -> None:
    """Hook into APScheduler — every 30 minutes."""
    scheduler.add_job(run_once, "interval", minutes=30, id="edgar_poller", replace_existing=True)
