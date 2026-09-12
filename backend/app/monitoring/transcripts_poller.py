"""Earnings transcript poller — daily cron.

Mirrors `edgar_poller.py` for transcripts. Iterates the `auto_analysis`
tier of the universe (`AUTO_PULL_TIERS`), calls
`transcripts_service.latest_transcript`, and detects new periods by
comparing against a per-ticker `transcripts_seen_periods` cache row.

The tier restriction is the point, not a detail. One FMP call per ticker
per day is a standing cost that has to be bounded by a curated list, and
the other two tiers are the ones that grow on their own: every manual
search on a new name inserts an `analyzed_on_demand` row, so polling the
unfiltered `companies` table means the automatic pull universe only ever
gets bigger. Restricting ingestion here takes nothing away from manual
research — any ticker, at any tier, still gets its transcript pulled and
its memo written the moment a user asks. It just doesn't buy a permanent
slot in the daily cron by having been searched once.

This docstring claimed the restriction for some time before the code
implemented it (`run_once` called `list_tickers()` with no filter), which
is what `test_auto_pull_universe_is_tier_scoped.py` now guards against.
An explicit `tickers=` argument still bypasses the filter entirely — that
is how tests and admin re-runs drive a specific name through here.

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

Handing an event to the orchestrator is the expensive half, and it is
capped per pass — see `MAX_EVENT_TICKERS_PER_PASS`.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from ..cache import cache_get, cache_put
from ..monitoring import note_names, record_run
from ..services.data_service import curated_poll_universe
from ..services.transcripts_service import get_transcripts

log = logging.getLogger(__name__)

# How many tickers one pass may hand to `on_transcript_event`.
#
# Same reasoning as `edgar_poller.MAX_FILING_EVENTS_PER_PASS`, and the same
# cost: `on_transcript_event` calls `_persist_raw_data_only` before its gate,
# which re-reads the ticker's filings AND transcripts, writes them, and embeds
# the new chunks. It is a burst risk for the same reason too — `transcripts`
# was one of the capabilities cached forever, so the first pass after that fix
# surfaces every period that published in the meantime, for the whole curated
# universe at once.
#
# Twice the filing cap because this loop runs daily rather than every 30
# minutes: 30 events have 24 hours to work through, which is a far lower
# sustained rate than the filing poller's 15-per-half-hour, and it drains a
# ~166-ticker backlog in under a week instead of eleven days. Earnings arrive
# in two-week clusters, so a slower drain would still be catching up on Q1
# when Q2 starts.
MAX_EVENT_TICKERS_PER_PASS = 30

# Upper bound on the per-ticker `transcripts_seen_periods` bookkeeping set.
# Ten years of quarters. Periods are labels like "2025Q4", so this is
# genuinely chronological, and the union used to grow without limit.
MAX_SEEN_PERIODS = 40


def _seen_periods(ticker: str) -> set[str]:
    snap = cache_get(ticker, "transcripts_seen_periods")
    if snap is None or not isinstance(snap.payload, dict):
        return set()
    return set(snap.payload.get("periods", []))


def _bounded_seen(periods: set[str], seen: set[str]) -> set[str]:
    """`periods | seen`, pruned deterministically to MAX_SEEN_PERIODS.

    The provider's current window is kept whatever the cap says: dropping a
    period that is still being returned would make the next pass read it as
    new and re-fire the event. Only older periods are pruned, newest first.
    """
    merged = periods | seen
    if len(merged) <= MAX_SEEN_PERIODS:
        return merged
    keep = set(periods)
    for period in sorted(seen - periods, reverse=True):
        if len(keep) >= MAX_SEEN_PERIODS:
            break
        keep.add(period)
    return keep


def _save_seen_periods(ticker: str, periods: set[str]) -> None:
    cache_put(
        ticker, "transcripts_seen_periods",
        payload={"periods": sorted(periods)},
        sources_used=[f"transcripts:{ticker}:bookkeeping"],
        generated_by="transcripts_poller",
    )


def run_once(tickers: Iterable[str] | None = None) -> list[dict]:
    """Poll for new earnings transcripts once. Returns
    `[{ticker, new_periods, regenerated}]` events.

    `regenerated` is the truthy result of
    `update_orchestrator.on_transcript_event` — `None` when the handler
    raised, a `kind="skipped"` dict when the gating decided to skip
    memo regen (transcript persisted, no memo burned), or a
    `kind="full_reanalysis"` dict with the enqueued `regen_jobs` job id
    when the ticker is pinned / actively viewed (the worker thread runs
    the memo asynchronously; outcome lands in the job row).

    With no argument, polls the `AUTO_PULL_TIERS` slice of the universe
    (see the module docstring). An explicit `tickers=` is taken as given
    and skips the tier filter.

    At most `MAX_EVENT_TICKERS_PER_PASS` tickers are handed to the
    orchestrator. A ticker over the cap is *deferred*, not dropped: its
    bookkeeping is left untouched, so the next pass still sees its periods
    as new and picks it up. Deferrals are named in the run note.
    """
    excluded: int | None = None
    if tickers is None:
        tickers, excluded = curated_poll_universe()
    tickers = list(tickers)

    events: list[dict] = []
    deferred: list[str] = []
    processed = 0
    for t in tickers:
        try:
            transcripts = get_transcripts(t) or []
        except Exception as exc:
            log.warning("transcript poll failed for %s: %s", t, exc)
            continue

        # Period is the canonical key — providers return e.g.
        # "2025Q4". We track the set of periods seen per ticker.
        periods: set[str] = set()
        for tr in transcripts:
            p = tr.get("period") or tr.get("date") or ""
            if isinstance(p, str) and p:
                periods.add(p)

        seen = _seen_periods(t)
        new = periods - seen
        if new and seen:  # skip first-run init
            if processed >= MAX_EVENT_TICKERS_PER_PASS:
                # Over the cap. The `continue` also skips the bookkeeping
                # write below, and must: recording these periods as seen
                # without having processed them would lose the event
                # permanently — the next pass would compute an empty diff.
                deferred.append(t)
                continue
            processed += 1
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
            _save_seen_periods(t, _bounded_seen(periods, seen))

    # `kind="gate_error"` = the auto-regen gate crashed rather than
    # decided; surface it in the note instead of letting it pass as a skip.
    gate_errors = [
        e["ticker"] for e in events
        if isinstance(e.get("regenerated"), dict)
        and e["regenerated"].get("kind") == "gate_error"
    ]
    note = f"{len(events)} new transcripts"
    # Say the constraint out loud. `excluded is None` means the caller named
    # the tickers, so no tier filter ran and there is nothing to report.
    if excluded is not None:
        note += f"; polled {len(tickers)} in-tier, skipped {excluded} out-of-tier"
    if deferred:
        note += f"; deferred {len(deferred)} over the {MAX_EVENT_TICKERS_PER_PASS}"
        note += f"-ticker cap: {note_names(deferred)}"
    if gate_errors:
        note += f"; gate errors on {len(gate_errors)}: {', '.join(gate_errors[:5])}"
    record_run("transcripts_poller", success=not gate_errors, note=note)
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
