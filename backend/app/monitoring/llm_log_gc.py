"""Daily GC for LLMCallLog rows older than 90 days.

Also sweeps `chart_commentaries` (FEAT-001) on the same 90-day policy:
cached commentary is prose derived from data and memos that go stale, so
keeping it longer than the call log that paid for it buys nothing.

And `sdk_traces` (G22): the model's docstring always promised a 90-day GC
here, but nothing deleted them, and each row carries the SDK's raw item
stream, so they were the one unbounded LLM table.

Wired only when ENABLE_MONITORING=true. Idempotent — safe to run repeatedly.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import delete
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models.fundamentals import ChartCommentary
from ..models.telemetry import SDKTrace
from ..services.llm_metrics import gc_old
from . import record_run

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Clock seam so tests can pin the retention boundary."""
    return datetime.utcnow()


def gc_chart_commentaries(*, max_age_days: int = 90, db: Session | None = None) -> int:
    """Delete `chart_commentaries` rows older than `max_age_days`; returns
    the count. One DELETE — the table is bounded (one row per distinct
    chart × memo version) so there is nothing to batch."""
    own = db is None
    session = db or SessionLocal()
    try:
        cutoff = _utcnow() - timedelta(days=max_age_days)
        result = session.execute(delete(ChartCommentary).where(ChartCommentary.created_at < cutoff))
        session.commit()
        return int(result.rowcount or 0)
    finally:
        if own:
            session.close()


def gc_sdk_traces(*, max_age_days: int = 90, db: Session | None = None) -> int:
    """Delete `sdk_traces` rows older than `max_age_days`; returns the count.

    One set-based DELETE, never a load-then-delete: a row's `new_items` JSON
    can be large, and the worker runs on 512 MiB."""
    own = db is None
    session = db or SessionLocal()
    try:
        # A database init_db() never touched has no table: nothing to delete,
        # and not a reason to fail the whole GC run.
        if not sa_inspect(session.get_bind()).has_table(SDKTrace.__tablename__):
            return 0
        cutoff = _utcnow() - timedelta(days=max_age_days)
        result = session.execute(delete(SDKTrace).where(SDKTrace.generated_at < cutoff))
        session.commit()
        return int(getattr(result, "rowcount", 0) or 0)
    finally:
        if own:
            session.close()


def run_once(max_age_days: int = 90) -> dict[str, int]:
    n = gc_old(max_age_days=max_age_days)
    m = gc_chart_commentaries(max_age_days=max_age_days)
    t = gc_sdk_traces(max_age_days=max_age_days)
    record_run(
        "llm_log_gc",
        note=(f"deleted {n} llm_call_logs rows, {m} chart_commentaries rows and "
              f"{t} sdk_traces rows >{max_age_days}d old"),
    )
    return {"deleted": n, "deleted_commentaries": m, "deleted_sdk_traces": t,
            "max_age_days": max_age_days}


def register(scheduler) -> None:
    scheduler.add_job(run_once, "interval", days=1, id="llm_log_gc", replace_existing=True)
