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

The pass itself is capped too. Better chunking makes the indexing cheaper but
does not make a 5 MB download faster, and a pass that outruns its 30-minute
interval eats the next tick silently (`max_instances=1`). So `run_once` has a
wall-clock budget (`MAX_PASS_SECONDS`), reports progress while it is still
running (`PROGRESS_INTERVAL_SECONDS`) rather than only on completion, and
rotates where it starts so the tail of the universe is covered across
consecutive passes instead of never. A ticker the budget did not reach keeps
its bookkeeping untouched, exactly like one deferred by the event cap.
"""
from __future__ import annotations

import logging
import time
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

# Wall-clock budget for one pass, in seconds.
#
# The loop is registered `interval, minutes=30` and APScheduler defaults to
# `max_instances=1`, so a pass that outlives its interval does not overlap —
# it silently eats the next tick, and the one after that. Measured in
# production on 2026-09-12: the last COMPLETED pass was logged at 20:52:34Z
# and it was still running 95 minutes later, having fired AAPL at 21:29Z and
# AMZN at 22:05Z — 36 minutes apart, in alphabetical order, so the pass was
# working the whole time. Nothing was wrong except that it could not finish,
# and `record_run` fires only at the END of `run_once`, so cron-health showed
# one stale timestamp for the entire hour and a half. An operator could not
# tell a slow pass from a wedged loop.
#
# 20 minutes leaves a third of the interval as headroom for the ticker that
# is in flight when the budget runs out (`on_filing_event` downloads
# documents, embeds them and calls the LLM, and is never interrupted
# mid-ticker — see `run_once`). A pass that stops here is not an error: it is
# the same bounded-work contract as MAX_FILING_EVENTS_PER_PASS, and the
# unvisited tickers keep their untouched bookkeeping, so the next tick — a
# rotated one, see `_rotated` — picks them up.
MAX_PASS_SECONDS = 20 * 60

# How often a long pass reports progress to `record_run`.
#
# `record_run` used to fire once, at the end. Until it did, `/api/admin/
# cron-health` reported the *previous* pass's timestamp, so a pass that ran
# 95 minutes looked identical to a loop that had died 95 minutes ago. Calling
# it periodically mid-pass costs one upsert every two minutes and makes the
# difference legible: a progress note names how far the pass has got.
PROGRESS_INTERVAL_SECONDS = 120


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


# Cache key for the rotation cursor. A pseudo-ticker rather than a real one
# because the cursor is a property of the pass, not of any company.
_CURSOR_KEY = "__edgar_poller_pass__"


def _resume_from() -> str | None:
    """The ticker the previous budget-capped pass stopped before, if any."""
    snap = cache_get(_CURSOR_KEY, "edgar_pass_cursor")
    if not snap or not isinstance(snap.payload, dict):
        return None
    value = snap.payload.get("resume_from")
    return str(value) if value else None


def _save_resume_from(ticker: str | None) -> None:
    cache_put(
        _CURSOR_KEY, "edgar_pass_cursor",
        payload={"resume_from": ticker or ""},
        sources_used=["edgar:pass:bookkeeping"],
        generated_by="edgar_poller",
        cost_tokens=0,
        ttl_seconds=365 * 24 * 3600,
    )


def _rotated(tickers: list[str], resume: str | None) -> list[str]:
    """Start the pass where the last one ran out of budget.

    Without this the budget would be a starvation device rather than a
    bound: a pass that only ever gets through the first 90 names in
    alphabetical order polls the same 90 forever, and the tail is never
    read again. Rotating means the whole universe is covered across
    consecutive passes — which is the only reason stopping early is
    acceptable at all. Falls back to the head when the cursor names a
    ticker that has since left the universe.
    """
    if not resume or resume not in tickers:
        return tickers
    i = tickers.index(resume)
    return tickers[i:] + tickers[:i]


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
    (see the module docstring), starting where the previous pass ran out of
    wall clock. An explicit `tickers=` is taken as given: it skips the tier
    filter, the rotation and the cursor, because an admin re-run for one
    name should not have to care what tier it is in and must not move the
    scheduled pass's place in the universe.

    Two bounds, and the same discipline behind both. At most
    `MAX_FILING_EVENTS_PER_PASS` tickers are handed to the orchestrator,
    and the pass stops after `MAX_PASS_SECONDS`. A ticker over either bound
    is *deferred* or *unvisited*, never dropped: its bookkeeping is left
    untouched, so the next pass still sees its accessions as new and picks
    it up. Both are named in the run note.

    Progress is reported to `record_run` every
    `PROGRESS_INTERVAL_SECONDS`, so cron-health distinguishes a long pass
    from a dead loop while it is still running.

    No-op for tickers without filings. The EDGAR provider returns an empty
    list in demo mode, so this loop becomes a quiet bookkeeping pass.
    """
    excluded: int | None = None
    scheduled = tickers is None
    if scheduled:
        tickers, excluded = curated_poll_universe()
        tickers = _rotated(list(tickers), _resume_from())
    else:
        tickers = list(tickers)

    started = time.monotonic()
    last_progress = started
    events: list[dict] = []
    gate_errors: list[str] = []
    deferred: list[str] = []
    unvisited: list[str] = []
    polled = 0

    for index, t in enumerate(tickers):
        # Checked before the ticker is touched, never during it.
        # `on_filing_event` downloads documents, embeds them and calls the
        # LLM; abandoning that half-done would leave `filing_docs` written
        # and `seen` unadvanced, which is a worse state than not starting.
        if time.monotonic() - started >= MAX_PASS_SECONDS:
            unvisited = list(tickers[index:])
            break

        now = time.monotonic()
        if now - last_progress >= PROGRESS_INTERVAL_SECONDS:
            last_progress = now
            record_run(
                "edgar_poller", success=True,
                note=(
                    f"in progress: polled {polled}/{len(tickers)}, "
                    f"{len(events)} new filings, {int(now - started)}s elapsed"
                ),
            )

        polled += 1
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

    # Where the next scheduled pass starts. The first unvisited ticker when
    # the budget bit, otherwise back to the head of the universe. Written
    # only for the scheduled pass — an admin re-run naming one ticker must
    # not move the cursor.
    if scheduled:
        try:
            _save_resume_from(unvisited[0] if unvisited else None)
        except Exception as exc:  # pragma: no cover — diagnostic only
            log.warning("edgar pass cursor write failed: %s", exc)

    note = f"{len(events)} new filings"
    # Say the constraint out loud. `excluded is None` means the caller named
    # the tickers, so no tier filter ran and there is nothing to report —
    # printing "0 out-of-tier" there would claim a filter that never fired.
    if excluded is not None:
        note += f"; polled {polled}/{len(tickers)} in-tier, skipped {excluded} out-of-tier"
    if deferred:
        note += f"; deferred {len(deferred)} over the {MAX_FILING_EVENTS_PER_PASS}"
        note += f"-event cap: {note_names(deferred)}"
    if unvisited:
        note += (
            f"; stopped at the {MAX_PASS_SECONDS}s pass budget with "
            f"{len(unvisited)} unvisited: {note_names(unvisited)}"
        )
    if gate_errors:
        note += f"; gate errors on {len(gate_errors)}: {', '.join(gate_errors[:5])}"
    record_run("edgar_poller", success=not gate_errors, note=note)
    return events


def register(scheduler) -> None:
    """Hook into APScheduler — every 30 minutes."""
    scheduler.add_job(run_once, "interval", minutes=30, id="edgar_poller", replace_existing=True)
