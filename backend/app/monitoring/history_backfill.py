"""Wave 2 — nightly history-table backfill for tier-1 names.

Fans out `history_service.backfill_ticker` over every ticker tagged
`auto_analysis` (the curated tier-1 watch list). Idempotent — re-running
against unchanged provider data is a no-op. Safe to run as the data
source for downstream agents because:

- `backfill_ticker` upserts on `(ticker, period, statement, line_item)`,
  on `accession_number`, and on `(ticker, period)` for transcripts.
- Per-ticker exceptions are caught + logged so one bad provider response
  doesn't poison the rest of the run.

This is a **reconciliation** pass, not a freshness driver, and the
distinction is the whole reason it is affordable. It reads filings and
transcripts with `prefer_cached=True`, so a cached row is used at whatever
age it has and only a ticker with no row at all costs a provider call.
Freshness belongs to the two pollers: `edgar_poller` detects a new
accession through the cheap index and invalidates that one ticker's filing
bodies, `transcripts_poller` detects a new period, and both hand the work
to `update_orchestrator` under a per-pass event cap.

Without that split this loop is the largest uncapped consumer in the
system. It touches all ~166 curated names in one job, `get_filings`
fetches up to ten multi-megabyte document bodies per ticker, and every
filing newly ingested fires `filing_memory.post_pass` — embeddings plus an
LLM diff. Before `filings` and `transcripts` were given TTLs they never
expired, so this loop made zero provider calls after its first night and
the cost was invisible; giving them real TTLs is exactly what would arm it.
The seven-day `filings` TTL would also have expired the whole universe
inside one window, because the pollers refresh every ticker within hours of
each other, so the herd would have arrived at a single 03:15 job.

The residue — a ticker genuinely cold, on a fresh deployment or newly
promoted into the tier — is bounded by `MAX_COLD_TICKERS_PER_PASS` and
named in the run note.

Wired in only when `ENABLE_MONITORING=true`; rolling history into local
SQLite is overkill for the demo loop but essential when the curated
universe is being driven against live providers.
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select

from ..database import SessionLocal
from ..models import Company
from ..services.history_service import backfill_hits_provider, backfill_ticker
from . import note_names, record_run

log = logging.getLogger(__name__)

# How many tickers one nightly pass may read cold — i.e. actually consult a
# provider for filings or transcripts rather than reconcile a cached row.
#
# A cold ticker is up to ten SEC document bodies plus four AlphaVantage
# requests, and each filing it newly inserts runs `filing_memory.post_pass`,
# which its own docstring says uses the LLM. So the ceiling is roughly 200
# LLM calls and embedding batches for a pass at this cap — the same order as
# `edgar_poller.MAX_FILING_EVENTS_PER_PASS` (15 events, up to 10 post-passes
# each) allows every 30 minutes, and here it is once a night.
#
# The cost of the cap is drain time: a universe that is cold end to end
# takes ceil(166 / 20) = 9 nights to warm. That is the right trade. Nothing
# is broken meanwhile — a ticker a user opens is fetched on demand, and the
# pollers keep detecting filings through the cheap index — whereas the
# uncapped alternative is ~1,660 document downloads and as many LLM calls in
# one job, on a 512 MiB worker Render has already OOM-killed twice.
MAX_COLD_TICKERS_PER_PASS = 20


def _tier1_tickers() -> list[str]:
    with SessionLocal() as db:
        rows = db.execute(
            select(Company.ticker)
            .where(Company.universe_tier == "auto_analysis")
            .order_by(Company.ticker)
        ).all()
    return [r[0] for r in rows]


def _rotated(tickers: list[str], day: int) -> list[str]:
    """The same list, entered at a different point each day.

    The cold-read budget is spent by the first tickers in iteration order
    that need a provider, and `_tier1_tickers` returns the same order every
    night. Without a rotation the head of the list would consume the budget
    every single night and the tail would never be reached at all — a cap
    that drains nothing is worse than no cap, because the note would show
    steady progress while a fixed set of tickers stayed permanently cold.
    Stepping by the cap means consecutive nights pick up where the previous
    one stopped, and the whole universe is covered in ceil(n / cap) nights.
    """
    if not tickers:
        return tickers
    offset = (day * MAX_COLD_TICKERS_PER_PASS) % len(tickers)
    return tickers[offset:] + tickers[:offset]


def run_once(ticker: str | None = None, *, day: int | None = None) -> dict[str, int]:
    """Backfill `ticker` (one) or every tier-1 name. Returns aggregate counts.

    Wave 8E: classify per-ticker failures so a wedged provider (rate-
    limit, auth, network) shows up in the loop status note rather than
    being silently absorbed. Counts surfaced:
      - `errors`: total per-ticker failures (any reason).
      - `rate_limited`: rows where the exception text mentions 429 / rate-limit.
      - `auth_errors`: rows where the exception text mentions 401 / 403 / forbidden.

    Loop status (`status_snapshot()`) reports `success=False` when ANY
    error fires so the admin endpoint flags the loop as unhealthy.

    Reading the note: `fp=`, `filings=` and `transcripts=` are counts of
    rows that actually changed, so a quiet night is `fp=0 filings=0
    transcripts=0` and a non-zero number means new data landed. `filings=`
    used to count every re-ingest of an unchanged row, so it read 1660 every
    night — 166 tickers x the provider's 10-filing cap — whether or not a
    single filing had been fetched. It was the only telemetry pointing at
    the filings pipeline, and it said "healthy" for the system's whole life
    while nothing was being ingested at all.

    `cold=` counts the tickers this pass had to consult a provider for, and
    `deferred=` the ones it would have had to and refused, over
    `MAX_COLD_TICKERS_PER_PASS`. Deferred names are listed: a cap nobody can
    see the shape of is a silent cap. A steady non-zero `deferred=` across
    nights means the universe is warming more slowly than names are being
    added and the cap wants raising.

    An explicit `ticker=` is one name asked for deliberately, so it bypasses
    both the budget and the rotation — an admin re-run should not be told to
    come back in nine days. `day=` is the rotation ordinal, injectable so a
    test can advance nights without touching the clock.
    """
    single = ticker is not None
    if single:
        tickers = [ticker.upper()]
    else:
        if day is None:
            day = datetime.utcnow().toordinal()
        tickers = _rotated(_tier1_tickers(), day)
    totals = {"financial_periods": 0, "filings": 0, "transcripts": 0}
    errors = 0
    rate_limited = 0
    auth_errors = 0
    cold = 0
    deferred: list[str] = []
    for t in tickers:
        try:
            # Ask before spending. A warm ticker reconciles cached rows
            # against the history tables for free and is never deferred; a
            # cold one is the expensive case the budget exists for.
            is_cold = not single and backfill_hits_provider(t)
            if is_cold:
                if cold >= MAX_COLD_TICKERS_PER_PASS:
                    deferred.append(t)
                    continue
                cold += 1
            res = backfill_ticker(t, prefer_cached=not single)
            for k, v in res.items():
                totals[k] = totals.get(k, 0) + v
        except Exception as exc:  # pragma: no cover — diagnostic only
            errors += 1
            msg = str(exc).lower()
            if "429" in msg or "rate limit" in msg or "rate-limit" in msg:
                rate_limited += 1
            if "401" in msg or "403" in msg or "forbidden" in msg or "unauthorized" in msg:
                auth_errors += 1
            log.warning("history_backfill failed for %s: %s", t, exc)
    note_parts = [
        f"tickers={len(tickers) - len(deferred)}",
        f"fp={totals['financial_periods']}",
        f"filings={totals['filings']}",
        f"transcripts={totals['transcripts']}",
        f"errors={errors}",
    ]
    if not single:
        note_parts.append(f"cold={cold}")
    if rate_limited:
        note_parts.append(f"rate_limited={rate_limited}")
    if auth_errors:
        note_parts.append(f"auth_errors={auth_errors}")
    note = " ".join(note_parts)
    if deferred:
        note += (
            f"; deferred {len(deferred)} over the {MAX_COLD_TICKERS_PER_PASS}"
            f"-cold-read cap: {note_names(deferred)}"
        )
    record_run("history_backfill", success=errors == 0, note=note)
    totals["errors"] = errors
    totals["rate_limited"] = rate_limited
    totals["auth_errors"] = auth_errors
    totals["cold_reads"] = cold
    totals["deferred"] = len(deferred)
    totals["tickers_processed"] = len(tickers) - len(deferred)
    return totals


def register(scheduler) -> None:
    # Daily, off-peak. Cron at 03:00 UTC keeps it clear of EDGAR poller and
    # the LLM log GC, both of which run at top-of-hour by default.
    scheduler.add_job(
        run_once, "cron", hour=3, minute=15,
        id="history_backfill", replace_existing=True,
    )
