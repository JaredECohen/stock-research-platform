"""Daily retention GC for `research_snapshots`.

`cache_put` is insert-only: each write appends a row and leaves the
previous one behind, live and un-stale. Nothing reaped the table, so it
grew monotonically — the EDGAR poller's bookkeeping writes alone add ~8k
rows a day — until the unbounded lineage scan in `cache.invalidate` could
no longer fit in the worker's memory and SIGKILLed it hourly (2026-09-10).

That scan is bounded now. This loop is the other half: it keeps the table
from growing back into the next unbounded thing. Deliberately conservative
— the newest row per `(subject, kind)` is never deleted, so a GC pass can
never turn a cache hit into a recompute.

Idempotent, and safe to run repeatedly.
"""
from __future__ import annotations

import logging

from ..cache.snapshots import gc_snapshots
from . import record_run

log = logging.getLogger(__name__)


def run_once() -> dict[str, int]:
    try:
        stats = gc_snapshots()
    except Exception as exc:
        # Never raise into the scheduler, but do not report success either:
        # a GC that silently stops working is how the table grew unnoticed
        # the first time.
        log.warning("snapshot_gc failed: %s", type(exc).__name__)
        record_run("snapshot_gc", success=False, note=f"error={type(exc).__name__}")
        return {"scanned": 0, "deleted": 0, "capped": 0}
    note = f"scanned={stats['scanned']} deleted={stats['deleted']} capped={stats['capped']}"
    # `capped` means the table still has more to reap than one pass removes;
    # the next run continues, but a run that stays capped for days means
    # retention is not keeping up with write volume.
    record_run("snapshot_gc", success=True, note=note)
    return stats


def register(scheduler) -> None:
    # Daily at 04:15 UTC — after checkpoint_gc (04:00), before
    # mispricing_audit_loop (04:30), so the nightly GCs do not overlap.
    scheduler.add_job(
        run_once, "cron", hour=4, minute=15,
        id="snapshot_gc", replace_existing=True,
    )
