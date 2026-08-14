"""Manual backfill of `doc_chunks.embedding_vec` + HNSW index build.

    python -m scripts.backfill_pgvector    # from the backend directory

Normally you do NOT need this: `app.worker` runs the same
`vector_store.backfill_and_index()` on every boot, in its background
startup thread. This script exists for the cases that bypass the worker —
running the backfill against a database by hand, forcing it without
waiting for a redeploy, or checking how many rows are still outstanding.

Why it can't just live in `init_db()`: the index build takes minutes on a
large table and startup runs before the service can answer /health, so it
would fail the deploy. And why the inline sync in
`vector_store.upsert_source` isn't enough: that one is capped at ~1000
rows per call so it can't stall a memo run, which means the historical
corpus would take weeks to converge on its own.

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
    result = vector_store.backfill_and_index()
    total = result["populated"]

    after = remaining()
    print(f"done: {total} rows populated, {after} still NULL")
    print(f"HNSW index: {'ready' if result['indexed'] else 'not built (see logs)'}")
    if after:
        # Rows whose `embedding_dim` isn't the current model's dimension are
        # skipped by design — a 256-dim hash-fallback vector cannot cast into
        # a vector(1536) column. They stay searchable via the streaming path.
        print("  (remaining rows are dimension mismatches from a prior "
              "embedding model — expected, and still searchable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
