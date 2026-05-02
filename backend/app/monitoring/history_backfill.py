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
    """Backfill `ticker` (one) or every tier-1 name. Returns aggregate counts."""
    tickers = [ticker.upper()] if ticker else _tier1_tickers()
    totals = {"financial_periods": 0, "filings": 0, "transcripts": 0}
    errors = 0
    for t in tickers:
        try:
            res = backfill_ticker(t)
            for k, v in res.items():
                totals[k] = totals.get(k, 0) + v
        except Exception as exc:  # pragma: no cover — diagnostic only
            errors += 1
            log.warning("history_backfill failed for %s: %s", t, exc)
    note = (
        f"tickers={len(tickers)} fp={totals['financial_periods']} "
        f"filings={totals['filings']} transcripts={totals['transcripts']} "
        f"errors={errors}"
    )
    record_run("history_backfill", success=errors == 0, note=note)
    totals["errors"] = errors
    totals["tickers_processed"] = len(tickers)
    return totals


def register(scheduler) -> None:
    # Daily, off-peak. Cron at 03:00 UTC keeps it clear of EDGAR poller and
    # the LLM log GC, both of which run at top-of-hour by default.
    scheduler.add_job(
        run_once, "cron", hour=3, minute=15,
        id="history_backfill", replace_existing=True,
    )
