"""Phase 2 integration test: re-running NVDA memo hits cache and is faster."""
from __future__ import annotations

import time
from datetime import datetime

from sqlalchemy import select

from app.agents.graph import run_stock_memo
from app.cache import invalidate
from app.cache.snapshots import CacheCostLog
from app.database import SessionLocal


def _count_hits_since(t0: datetime) -> int:
    with SessionLocal() as db:
        rows = db.execute(select(CacheCostLog).where(CacheCostLog.generated_at >= t0)).scalars().all()
        return sum(1 for r in rows if r.kind.endswith(":hit"))


def test_second_run_is_fast_and_cache_hits():
    # Make sure NVDA has a clean cache slate so the first run is genuinely a miss.
    invalidate("NVDA")
    invalidate("Technology:Semiconductors:NVDA", kind="sector_warm")

    t0 = datetime.utcnow()

    start_cold = time.perf_counter()
    memo1 = run_stock_memo("NVDA")
    cold_seconds = time.perf_counter() - start_cold

    start_warm = time.perf_counter()
    memo2 = run_stock_memo("NVDA")
    warm_seconds = time.perf_counter() - start_warm

    assert memo1.ticker == "NVDA"
    assert memo2.ticker == "NVDA"
    # Second call must be under 0.5s on demo (no LLM), and at least 5 hits logged.
    assert warm_seconds < 0.5, f"warm run too slow: {warm_seconds:.2f}s"
    hits = _count_hits_since(t0)
    assert hits >= 5, f"expected >=5 cache hits, got {hits}"
