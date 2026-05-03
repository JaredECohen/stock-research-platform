"""Wave 2 — nightly history-table backfill for tier-1 names.

Fans out `history_service.backfill_ticker` over every ticker tagged
`auto_analysis` (the curated tier-1 watch list). Idempotent — re-running
against unchanged provider data is a no-op. Safe to run as the data
source for downstream agents because:

- `backfill_ticker` upserts on `(ticker, period, statement, line_item)`,
  on `accession_number`, and on `(ticker, period)` for transcripts.
- Per-ticker exceptions are caught + logged so one bad provider response
  doesn't poison the rest of the run.

Wired in only when `ENABLE_MONITORING=true`; rolling history into local
SQLite is overkill for the demo loop but essential when the curated
universe is being driven against live providers.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sqlalchemy import select

from ..database import SessionLocal
from ..models import Company
from ..services.history_service import backfill_ticker
from . import record_run

log = logging.getLogger(__name__)


def _tier1_tickers() -> List[str]:
    with SessionLocal() as db:
        rows = db.execute(
            select(Company.ticker).where(Company.universe_tier == "auto_analysis")
        ).all()
    return [r[0] for r in rows]


def run_once(ticker: Optional[str] = None) -> Dict[str, int]:
    """Backfill `ticker` (one) or every tier-1 name. Returns aggregate counts.

    Wave 8E: classify per-ticker failures so a wedged provider (rate-
    limit, auth, network) shows up in the loop status note rather than
    being silently absorbed. Counts surfaced:
      - `errors`: total per-ticker failures (any reason).
      - `rate_limited`: rows where the exception text mentions 429 / rate-limit.
      - `auth_errors`: rows where the exception text mentions 401 / 403 / forbidden.

    Loop status (`status_snapshot()`) reports `success=False` when ANY
    error fires so the admin endpoint flags the loop as unhealthy.
    """
    tickers = [ticker.upper()] if ticker else _tier1_tickers()
    totals = {"financial_periods": 0, "filings": 0, "transcripts": 0}
    errors = 0
    rate_limited = 0
    auth_errors = 0
    for t in tickers:
        try:
            res = backfill_ticker(t)
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
        f"tickers={len(tickers)}",
        f"fp={totals['financial_periods']}",
        f"filings={totals['filings']}",
        f"transcripts={totals['transcripts']}",
        f"errors={errors}",
    ]
    if rate_limited:
        note_parts.append(f"rate_limited={rate_limited}")
    if auth_errors:
        note_parts.append(f"auth_errors={auth_errors}")
    record_run("history_backfill", success=errors == 0, note=" ".join(note_parts))
    totals["errors"] = errors
    totals["rate_limited"] = rate_limited
    totals["auth_errors"] = auth_errors
    totals["tickers_processed"] = len(tickers)
    return totals


def register(scheduler) -> None:
    # Daily, off-peak. Cron at 03:00 UTC keeps it clear of EDGAR poller and
    # the LLM log GC, both of which run at top-of-hour by default.
    scheduler.add_job(
        run_once, "cron", hour=3, minute=15,
        id="history_backfill", replace_existing=True,
    )
