"""Phase 2 integration test: re-running an NVDA memo reuses the cache.

History worth keeping, because this test has now been wrong twice:

  1. It originally asserted an absolute wall-clock budget (warm < 0.5s),
     which tripped at ~1.2s on shared CI runners.
  2. That was replaced with a relative comparison (warm < cold), on the
     theory that the relationship holds even if the absolute numbers
     don't. It doesn't. Measured over repeated local trials in demo mode:

         COLD hits=20 writes=3  5.89s | WARM hits=11 writes=0  6.20s
         COLD hits=20 writes=3  4.80s | WARM hits=11 writes=0 15.59s

     The warm run is routinely *slower*. Nothing is wrong with the cache
     — it saves exactly the work it should — but the saving is three
     snapshot writes against a fixed orchestration cost that dominates
     the runtime, so the wall-clock difference is pure scheduler noise.
     This failed CI on two unrelated PRs and passed on re-run with no
     code change.

The fix is to stop timing anything. The property actually worth pinning
is "the second run doesn't redo the work", and the cache's own telemetry
states that directly: `CacheCostLog` records a `<kind>:hit` row per cache
read and a bare `<kind>` row per snapshot *write*. A warm run that reuses
the cache performs strictly fewer writes than the cold run that populated
it — deterministic, machine-independent, and a stronger claim than any
timing comparison, since a cache that got slower but still avoided the
work was never what this test was defending against.

Log rows are windowed by id rather than timestamp: ids are monotonic,
whereas two rows written inside the same microsecond would be
indistinguishable by `generated_at`.
"""
from __future__ import annotations

from sqlalchemy import func, select

from app.agents.graph import run_stock_memo
from app.cache import invalidate
from app.cache.snapshots import CacheCostLog
from app.database import SessionLocal


def _max_log_id() -> int:
    with SessionLocal() as db:
        return db.execute(
            select(func.coalesce(func.max(CacheCostLog.id), 0))
        ).scalar_one()


def _counts_after(log_id: int) -> tuple[int, int]:
    """Return (hits, writes) recorded after `log_id`."""
    with SessionLocal() as db:
        rows = db.execute(
            select(CacheCostLog).where(CacheCostLog.id > log_id)
        ).scalars().all()
    hits = sum(1 for r in rows if r.kind.endswith(":hit"))
    return hits, len(rows) - hits


def test_second_run_reuses_cache_instead_of_redoing_work():
    # Clean slate so the first run is genuinely a miss.
    invalidate("NVDA")
    invalidate("Technology:Semiconductors:NVDA", kind="sector_warm")

    before_cold = _max_log_id()
    memo1 = run_stock_memo("NVDA")
    cold_hits, cold_writes = _counts_after(before_cold)

    before_warm = _max_log_id()
    memo2 = run_stock_memo("NVDA")
    warm_hits, warm_writes = _counts_after(before_warm)

    assert memo1.ticker == "NVDA"
    assert memo2.ticker == "NVDA"

    # Guards the setup itself: if invalidate() ever stops working, the cold
    # run writes nothing and the comparison below would pass vacuously.
    assert cold_writes > 0, (
        "cold run wrote no snapshots — invalidate() did not clear the cache, "
        "so this test is not measuring what it claims"
    )
    assert warm_writes < cold_writes, (
        f"warm run wrote {warm_writes} snapshots vs cold {cold_writes}; "
        f"the cache did not prevent the second run from redoing work"
    )
    assert warm_hits >= 5, f"expected >=5 cache hits on the warm run, got {warm_hits}"
