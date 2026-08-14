"""One-shot backfill of `doc_chunks.embedding_vec` from the JSON column.

Run once after the deploy that adds pgvector, from the backend directory:

    python -m scripts.backfill_pgvector

`vector_store.upsert_source` keeps the column current for newly indexed
filings, but it is deliberately capped at ~1000 rows per call so it can't
stall a memo run. That means the *existing* corpus would otherwise only
converge as chunks get re-indexed, which could take weeks. This script
does the bulk pass explicitly.

Idempotent and resumable — it only touches rows where `embedding_vec IS
NULL`, so re-running after an interruption picks up where it left off.
No-ops (exit 0, with an explanation) when pgvector isn't active or the
backend isn't Postgres, so it's safe to run anywhere.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from app.database import engine, ensure_pgvector  # noqa: E402
from app.services import vector_store  # noqa: E402


def main() -> int:
    from sqlalchemy import text

    if engine.dialect.name != "postgresql":
        print(f"backend is {engine.dialect.name}, not postgresql — nothing to do")
        return 0

    ensure_pgvector()
    vector_store.reset_pgvector_probe()
    if not vector_store.pgvector_available():
        print("pgvector column unavailable — search will use the streaming "
              "fallback (correct, just slower). Nothing to backfill.")
        return 0

    def remaining() -> int:
        with engine.connect() as conn:
            return conn.execute(text(
                "SELECT count(*) FROM doc_chunks "
                "WHERE embedding_vec IS NULL AND embedding IS NOT NULL"
            )).scalar_one()

    before = remaining()
    print(f"rows needing backfill: {before}")
    total = 0
    while True:
        n = vector_store.sync_pgvector_column(batch=2000, max_batches=25)
        total += n
        left = remaining()
        print(f"  populated {total} rows, {left} remaining")
        if n == 0 or left == 0:
            break

    after = remaining()
    print(f"done: {total} rows populated, {after} still NULL")
    if after:
        # Rows whose `embedding_dim` isn't the current model's dimension are
        # skipped by design — a 256-dim hash-fallback vector cannot cast into
        # a vector(1536) column. They stay searchable via the streaming path.
        print("  (remaining rows are dimension mismatches from a prior "
              "embedding model — expected, and still searchable)")

    _build_index()
    return 0


def _build_index() -> None:
    """Build the HNSW index, after the rows exist.

    Not done at startup on purpose: this takes minutes on a large table
    and `init_db()` runs before the service can answer `/health`, so
    building it there would fail the deploy. Building it *after* the
    backfill is also substantially faster than building on an empty table
    and inserting into it row by row.

    `maintenance_work_mem` drives how much of the graph pgvector can keep
    in memory during the build; when it doesn't fit, pgvector falls back
    to a slower on-disk build rather than failing. 128MB is a deliberate
    compromise for a small (basic-256mb) database instance — raise it if
    the database plan is larger and the build is slow.

    Failure here is non-fatal: search without the index does an exact
    scan, which is slower but returns identical results and still keeps
    the corpus out of Python memory.
    """
    from sqlalchemy import text

    print("building HNSW index (minutes on a large table; safe to interrupt "
          "and re-run — CREATE INDEX IF NOT EXISTS is idempotent)...")
    try:
        with engine.begin() as conn:
            conn.execute(text("SET maintenance_work_mem = '128MB'"))
            conn.execute(text(
                # Cosine opclass must match the `<=>` operator that
                # vector_store orders by, or the planner ignores the index.
                "CREATE INDEX IF NOT EXISTS ix_doc_chunks_embedding_vec "
                "ON doc_chunks USING hnsw (embedding_vec vector_cosine_ops)"
            ))
        print("  HNSW index ready")
    except Exception as exc:
        print(f"  index build failed ({exc}); search still works via exact "
              f"scan — retry later or on a larger DB plan")


if __name__ == "__main__":
    raise SystemExit(main())
