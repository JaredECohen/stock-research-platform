"""Copy your local SQLite DB into a Postgres target (e.g. Render's
production database) so the deploy starts fully populated instead of
empty.

Usage
-----
Run locally with the source SQLite path + target Postgres URL set in
env. The Postgres URL must be the **External** connection string from
Render (the internal one is only reachable from inside Render's VPC).

    cd backend
    SOURCE_SQLITE=./marketmosaic.db \
    TARGET_POSTGRES_URL='postgresql+psycopg2://user:pass@host/dbname?sslmode=require' \
    python -m scripts.migrate_sqlite_to_postgres

What it does
------------
1. Connects to both DBs.
2. Creates the schema on Postgres via SQLAlchemy `Base.metadata.create_all`.
3. Copies every row from each table in dependency order, batching
   inserts at 500/row to keep memory + Postgres roundtrips reasonable.
4. Skips ephemeral / log tables you don't want to bring over (UILog,
   LLMCallLog) so the new instance starts with a clean audit trail.
5. Prints per-table progress + final row counts on both sides.

Idempotency
-----------
The script DELETEs from each target table before copying. Re-running
is safe and gives you a faithful snapshot of your local state — but
it WILL clobber any rows that exist on the target. Don't run against
a production DB after real users have written to it.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Type

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

# Tables we DON'T migrate. Each has a reason; keep in sync as we add
# new persisted state.
_SKIP_TABLES = {
    "ui_logs",          # audit log; starts fresh on the new instance
    "llm_call_logs",    # cost trail; same
    "cache_cost_logs",  # snapshot cost ledger; same
    "sdk_traces",       # OpenAI SDK exchange traces; same
    "memo_run_checkpoints",  # per-step run cache; rebuilds itself
    "provider_cache",   # transient TTL cache; will repopulate
}


def _bootstrap_paths() -> None:
    """Make `app.*` importable without requiring `pip install -e .`."""
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(here)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)


def _ordered_models() -> List[Type]:
    """Return ORM classes in roughly dependency-safe insert order.

    `Company` first because everything else FK-references it logically;
    even where there's no DB FK, ordering keeps the user-visible state
    consistent if the migration is interrupted halfway."""
    from app import models  # noqa: F401  (registers tables on Base)
    from app.database import Base
    from app.cache.snapshots import CacheCostLog, ResearchSnapshot  # noqa

    explicit_order = [
        "companies",
        "financial_periods",
        "filing_docs",
        "earnings_transcripts",
        "screener_metrics",
        "screener_scores",
        "stock_memos",
        "memo_snapshots",
        "memo_outcomes",
        "dcf_models",
        "research_snapshots",
        "cached_documents",
        "portfolio_runs",
    ]
    name_to_cls = {t.name: t for t in Base.metadata.sorted_tables}
    out: List[Type] = []
    seen: set[str] = set()
    for name in explicit_order:
        if name in name_to_cls and name not in _SKIP_TABLES:
            out.append(name_to_cls[name])
            seen.add(name)
    # Pick up anything else (defensive — keeps the script working when
    # new tables land without the script being updated).
    for name, table in name_to_cls.items():
        if name in seen or name in _SKIP_TABLES:
            continue
        out.append(table)
    return out


def _copy_table(src_session: Session, dst_session: Session, table) -> int:
    """Copy every row from `table` (source) into the same table on
    `dst_session`. Returns rows copied."""
    rows = src_session.execute(table.select()).mappings().all()
    if not rows:
        return 0
    # Wipe target so re-runs are deterministic.
    dst_session.execute(table.delete())
    dst_session.commit()
    # Batch insert. SQLAlchemy core insert() handles dialect quirks
    # (e.g., SQLite booleans → Postgres booleans) automatically.
    BATCH = 500
    n = 0
    for i in range(0, len(rows), BATCH):
        chunk = [dict(r) for r in rows[i:i + BATCH]]
        dst_session.execute(table.insert(), chunk)
        dst_session.commit()
        n += len(chunk)
    return n


def main() -> int:
    _bootstrap_paths()

    src_path = os.environ.get("SOURCE_SQLITE", "./marketmosaic.db")
    dst_url = os.environ.get("TARGET_POSTGRES_URL")
    if not dst_url:
        print("error: TARGET_POSTGRES_URL must be set "
              "(use Render's external connection string)", file=sys.stderr)
        return 1
    if not os.path.exists(src_path):
        print(f"error: SOURCE_SQLITE not found at {src_path}", file=sys.stderr)
        return 1

    src_engine = create_engine(f"sqlite:///{src_path}", future=True)
    dst_engine = create_engine(dst_url, future=True)

    # Ensure target schema exists. Equivalent to what `init_db()` does
    # on app startup, but we want it now before we copy.
    from app.database import Base
    print("creating schema on target…")
    Base.metadata.create_all(bind=dst_engine)

    SrcSession = sessionmaker(bind=src_engine, autoflush=False)
    DstSession = sessionmaker(bind=dst_engine, autoflush=False)
    tables = _ordered_models()

    totals = {"copied": 0, "tables": 0}
    started = time.perf_counter()
    with SrcSession() as src, DstSession() as dst:
        for table in tables:
            t0 = time.perf_counter()
            n = _copy_table(src, dst, table)
            dur = time.perf_counter() - t0
            totals["copied"] += n
            totals["tables"] += 1
            print(f"  {table.name:<30} {n:>8,} rows · {dur:5.2f}s")

    # Sanity: target row counts match source.
    print()
    print("verification — source vs target row counts:")
    insp_src = inspect(src_engine)
    insp_dst = inspect(dst_engine)
    with src_engine.connect() as src_conn, dst_engine.connect() as dst_conn:
        for table in tables:
            if not (insp_src.has_table(table.name) and insp_dst.has_table(table.name)):
                continue
            src_n = src_conn.execute(text(f'SELECT COUNT(*) FROM "{table.name}"')).scalar()
            dst_n = dst_conn.execute(text(f'SELECT COUNT(*) FROM "{table.name}"')).scalar()
            marker = "✓" if src_n == dst_n else "✗"
            print(f"  {marker} {table.name:<30} src={src_n:>8,}  dst={dst_n:>8,}")

    elapsed = time.perf_counter() - started
    print()
    print(f"done — {totals['copied']:,} rows across {totals['tables']} tables in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
