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

It also reaps W7's SHADOW `learning_renders` audit rows older than 180
days, in bounded batches: one row per consumer per memo run adds up, and a
shadow row only matters while the renderer soaks before promotion. Inject
rows (what a memo was actually shown) are kept.

Idempotent, and safe to run repeatedly.
"""
from __future__ import annotations

import logging

from ..cache.snapshots import gc_snapshots
from . import record_run

log = logging.getLogger(__name__)


def _prune_learning_renders() -> dict[str, int]:
    from ..learning.context import prune_shadow_renders
    return prune_shadow_renders()


def run_once() -> dict[str, int]:
    # The two reapers are independent: one failing never stops the other.
    renders_error: str | None = None
    try:
        renders = _prune_learning_renders()
    except Exception as exc:
        renders_error = type(exc).__name__
        log.warning("snapshot_gc: learning_renders prune failed: %s", renders_error)
        renders = {"deleted": 0, "capped": 0}
    try:
        stats = gc_snapshots()
    except Exception as exc:
        # Never raise into the scheduler, but do not report success either:
        # a GC that silently stops working is how the table grew unnoticed
        # the first time.
        log.warning("snapshot_gc failed: %s", type(exc).__name__)
        record_run("snapshot_gc", success=False, note=f"error={type(exc).__name__}")
        return {"scanned": 0, "deleted": 0, "capped": 0, "ledger_deleted": 0,
                "renders_deleted": renders["deleted"]}
    stats = {**stats, "renders_deleted": renders["deleted"]}
    note = (
        f"scanned={stats['scanned']} deleted={stats['deleted']} "
        f"capped={stats['capped']} ledger_deleted={stats.get('ledger_deleted', 0)} "
        f"renders_deleted={renders['deleted']} renders_capped={renders['capped']}"
        + (f" renders_error={renders_error}" if renders_error else "")
    )
    # `capped` means the table still has more to reap than one pass removes;
    # the next run continues, but a run that stays capped for days means
    # retention is not keeping up with write volume. A failed render prune
    # is not success either, for the same reason.
    record_run("snapshot_gc", success=renders_error is None, note=note)
    return stats


def register(scheduler) -> None:
    # Daily at 04:15 UTC — after checkpoint_gc (04:00), before
    # mispricing_audit_loop (04:30), so the nightly GCs do not overlap.
    scheduler.add_job(
        run_once, "cron", hour=4, minute=15,
        id="snapshot_gc", replace_existing=True,
    )
