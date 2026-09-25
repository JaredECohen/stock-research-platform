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

Fundamentals are the exception (FIX-005): after the reconciliation pass
this loop drains the filing-driven refreshes `fundamental_refresh`
scheduled (a period EDGAR shows as filed that FMP-primary storage lacks,
plus calendar and sweep checks), under their own cap of 30 per night and a
15-minute budget. Filings and transcripts stay reconciliation-only, and a
provider-owned ticker's statements are never read here any more.

Last, on the worker's scheduled run only and when OpenAI embeddings are
live, a bounded re-index retry (W7): the
newest zero-chunk in-scope filings and transcripts of the last 14 days are
indexed again — at most 10 sources, $0.10 and 20 MB a night, never a
post-pass — so a post-pass that failed on an embedding outage is not lost.
The note carries `reindexed=` / `reindex_deferred=` and every identity.

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


def run_once(ticker: str | None = None, *, day: int | None = None, reindex: bool = False) -> dict[str, int]:
    """Backfill `ticker` (one) or every tier-1 name. Returns aggregate counts.

    `reindex=True` adds the bounded W7 re-index retry, and only the worker's
    scheduled job passes it (`register`). `POST /api/admin/run-backfill`
    calls this function on the *web* process: there the retry would be
    embedding spend and `doc_chunks` writes outside the worker, per call
    rather than per night, racing the worker's own 03:15 run with nothing
    serialising the two.

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
    error_names: list[str] = []
    fetch_failures: list[dict] = []
    rate_limited = 0
    auth_errors = 0
    cold = 0
    deferred: list[str] = []
    post_pass_failures: list[dict] = []
    truncated_filings: list[dict] = []
    fund = None
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
            for k in ("financial_periods", "filings", "transcripts"):
                totals[k] += res.get(k, 0)
            fetch_failures.extend(res.get("filing_fetch_failures") or [])
            post_pass_failures.extend(res.get("post_pass_failures") or [])
            truncated_filings.extend(res.get("truncated_filings") or [])
        except Exception as exc:  # pragma: no cover — diagnostic only
            errors += 1
            msg = str(exc).lower()
            if "429" in msg or "rate limit" in msg or "rate-limit" in msg:
                rate_limited += 1
            if "401" in msg or "403" in msg or "forbidden" in msg or "unauthorized" in msg:
                auth_errors += 1
            error_names.append(f"{t}:{type(exc).__name__}")
            log.warning("history_backfill failed ticker=%s error_type=%s", t, type(exc).__name__)
    if not single:
        # FIX-005: drain the filing-driven fundamentals refreshes under their
        # own cap (30/night, 15 min). A single-ticker admin run skips it; the
        # fundamentals admin sync exists for one name.
        from ..services import fundamental_refresh
        try:
            fund = fundamental_refresh.nightly()
        except Exception as exc:  # nightly() never raises; this keeps record_run below
            fund = {**fundamental_refresh._empty_result(), "errors": [f"nightly:{type(exc).__name__}"]}
    retry = _reindex_retry() if reindex and not single else None
    note_parts = [
        f"tickers={len(tickers) - len(deferred)}",
        f"fp={totals['financial_periods']}",
        f"filings={totals['filings']}",
        f"transcripts={totals['transcripts']}",
        f"errors={errors}",
    ]
    if not single:
        note_parts.append(f"cold={cold}")
    if fund is not None:
        note_parts.append(f"fund_refreshed={fund['refreshed']} fund_pending={fund['pending']}")
    if retry is not None:
        note_parts.append(f"reindexed={retry.get('sources_indexed', 0)} "
                          f"reindex_deferred={len(retry.get('deferred') or [])}")
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
    if error_names:
        note += "; failed tickers: " + ", ".join(error_names)
    if fetch_failures:
        from ..services.history_service import filing_fetch_failure_note
        note += f"; filing fetch failures={len(fetch_failures)}: " + filing_fetch_failure_note(fetch_failures)
    if post_pass_failures:
        from ..services.history_service import post_pass_failure_note
        note += (
            f"; post-pass failures={len(post_pass_failures)}: "
            + post_pass_failure_note(post_pass_failures)
        )
    if truncated_filings:
        from ..services.history_service import truncated_filing_note
        note += f"; bounded filing sources={len(truncated_filings)}: " + truncated_filing_note(truncated_filings)
    if fund is not None:
        note += _fundamentals_note(fund)
    if retry is not None:
        note += _reindex_note(retry)
    log.info("history_backfill: %s", note)
    # A filed period still missing after its secondary attempt (≈ night 9)
    # is the FIX-005 regression signal: it fails the loop on the night the
    # ticker becomes stuck, and is named without failing afterwards. A
    # period merely lagging inside its retry window is named, not a failure.
    fund_failed = fund is not None and bool(fund["errors"] or fund["stuck"])
    # The re-index retry fails the loop only if it crashed outright. A source
    # it could not index already failed its own post-pass night, and an
    # OpenAI outage is the case this retry exists to absorb.
    reindex_failed = retry is not None and bool(retry.get("crashed"))
    record_run("history_backfill",
               success=(errors == 0 and not post_pass_failures and not fetch_failures
                        and not fund_failed and not reindex_failed), note=note)
    totals["errors"] = errors
    totals["rate_limited"] = rate_limited
    totals["auth_errors"] = auth_errors
    totals["cold_reads"] = cold
    totals["deferred"] = len(deferred)
    totals["tickers_processed"] = len(tickers) - len(deferred)
    totals["filing_fetch_errors"] = len(fetch_failures)
    totals["post_pass_errors"] = len(post_pass_failures)
    totals["truncated_filings"] = len(truncated_filings)
    if fund is not None:
        totals["fund_refreshed"] = fund["refreshed"]
        totals["fund_pending"] = fund["pending"]
        totals["fund_over_cap"] = len(fund["over_cap"])
        totals["fund_errors"] = len(fund["errors"])
        totals["fund_stuck"] = len(fund["stuck"])
        totals["fund_held"] = len(fund.get("held") or [])
    if retry is not None:
        totals["reindexed"] = int(retry.get("sources_indexed", 0))
        totals["reindex_deferred"] = len(retry.get("deferred") or [])
    return totals


# W7 §8.3 / critique: the bounded automatic re-index retry. A post-pass whose
# embedding call failed (an OpenAI outage, now a raised `EmbeddingUnavailable`
# rather than a silent hash vector) leaves its source with zero chunks, and
# `run_ingest_post_passes` never retries. Nightly, re-index the newest such
# sources inside this existing loop — no new loop, `KNOWN_LOOPS` unchanged —
# under caps that bound the only automatic corpus spend: at most 10 sources,
# $0.10 and 20 MB a night, over sources dated in the last 14 days. Never a
# post-pass: no LLM diff, no memory writes.
REINDEX_RECENT_DAYS = 14
REINDEX_MAX_SOURCES = 10
REINDEX_MAX_USD = 0.10
REINDEX_MAX_ADDED_MB = 20


def _reindex_retry() -> dict | None:
    """Run the bounded retry, or None when embeddings are not live here.

    Demo mode, CI and a keyless process skip it: their vectors would be hash
    vectors, which is exactly the unusable class a repair must not create.
    """
    from ..services import corpus_repair
    from ..services import embeddings as emb_svc
    if not emb_svc.semantic_available():
        return None
    now = datetime.utcnow()
    try:
        return corpus_repair.index_missing(
            recent_days=REINDEX_RECENT_DAYS, max_sources=REINDEX_MAX_SOURCES,
            max_usd=REINDEX_MAX_USD, max_added_mb=REINDEX_MAX_ADDED_MB,
            receipt=f"history_backfill:{now:%Y-%m-%d}", now=now,
        )
    except Exception as exc:
        log.warning("history_backfill re-index retry crashed error_type=%s", type(exc).__name__)
        return {"sources_indexed": 0, "deferred": [], "crashed": type(exc).__name__}


def _reindex_note(reindex: dict) -> str:
    """Every source the retry indexed, deferred or failed on, by identity.

    "deferred" here is prefixed with "reindex", keeping it distinct from the
    cold-read cap's own "deferred N over the …" segment.
    """
    note = ""
    if reindex.get("crashed"):
        note += f"; reindex crashed: {reindex['crashed']}"
    if reindex.get("stopped_reason"):
        note += f"; reindex stopped: {reindex['stopped_reason']}"
    if reindex.get("indexed"):
        note += f"; reindexed sources: {note_names(reindex['indexed'])}"
    if reindex.get("deferred"):
        note += f"; reindex deferred: {note_names(reindex['deferred'])}"
    if reindex.get("failures"):
        note += f"; reindex failures: {note_names(reindex['failures'])}"
    if reindex.get("empty"):
        note += f"; reindex wrote no chunks: {note_names(reindex['empty'])}"
    if reindex.get("in_flight"):
        # Written within the settle window: their own post-pass may be running.
        note += f"; reindex waiting on a fresh post-pass: {note_names(reindex['in_flight'])}"
    return note


def _fundamentals_note(fund: dict) -> str:
    """Every name the fundamentals drain skipped, deferred or failed on.

    None of these segments contains the word "deferred": the cold-read cap
    owns that word in this note.
    """
    from ..services.fundamental_refresh import MAX_DRAIN_SECONDS, MAX_FUNDAMENTAL_REFRESHES_PER_PASS
    note = ""
    if fund.get("skipped_reason"):
        note += f"; fundamentals refresh skipped: {fund['skipped_reason']}"
    if fund["over_cap"]:
        note += (f"; fundamentals over the {MAX_FUNDAMENTAL_REFRESHES_PER_PASS}-refresh cap: "
                 f"{note_names(fund['over_cap'])}")
    if fund["over_budget"]:
        note += f"; fundamentals over the {MAX_DRAIN_SECONDS}s drain budget: {note_names(fund['over_budget'])}"
    if fund["leased"]:
        note += f"; fundamentals leased elsewhere: {note_names(fund['leased'])}"
    if fund.get("held"):
        note += f"; fundamentals held for the FMP re-pull execution: {note_names(fund['held'])}"
    if fund["missing"]:
        note += f"; fundamentals missing filed periods: {note_names(fund['missing'])}"
    if fund["stuck"]:
        note += f"; fundamentals stuck (filed period not published after retries): {note_names(fund['stuck'])}"
    if fund["still_missing"]:
        note += f"; fundamentals still missing: {note_names(fund['still_missing'])}"
    if fund["rows_quarantined"] or fund["repair_ids"]:
        note += f"; fundamentals quarantined={fund['rows_quarantined']}: {note_names(fund['repair_ids'])}"
    if fund["entitlement_denied"]:
        note += f"; fmp entitlement denied: {note_names(fund['entitlement_denied'])}"
    if fund["errors"]:
        note += f"; fundamentals failed: {note_names(fund['errors'])}"
    return note


def register(scheduler) -> None:
    # Daily, off-peak. Cron at 03:00 UTC keeps it clear of EDGAR poller and
    # the LLM log GC, both of which run at top-of-hour by default.
    scheduler.add_job(
        run_once, "cron", hour=3, minute=15,
        # The re-index retry belongs to this scheduled run only (see run_once).
        kwargs={"reindex": True},
        id="history_backfill", replace_existing=True,
    )
