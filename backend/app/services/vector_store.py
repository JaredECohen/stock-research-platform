"""Wave 10 — chunk index + vector retrieval.

Sits on top of `doc_chunks` and the `embeddings` service. Two
operations: **upsert** (a list of chunks for a given source doc) and
**search** (top-K by cosine similarity, optionally filtered by ticker /
source_type / section).

When the corpus grows large enough to need it, an out-of-band
migration converts `doc_chunks.embedding` to pgvector and adds an HNSW
index; until then we score in Python over the filtered subset (small N
makes this fine).

This module is *off* the memo's critical path — failures log + return
None / empty rather than blocking a memo run.
"""
from __future__ import annotations

import heapq
import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select, text as sa_text

from ..database import SessionLocal, engine
from ..models import DocChunk
from . import embeddings as emb_svc
from . import memory_probe

log = logging.getLogger(__name__)

# Hard ceiling on candidate rows a single search will consider. Sized
# well above a realistic per-ticker chunk count (~900 filing chunks for
# 10 filings at MAX_TEXT_BYTES) so normal retrieval is never truncated,
# while still bounding the pathological case. At SCORE_BATCH_SIZE=256
# the peak working set is the batch, not this number.
MAX_SCAN_CANDIDATES = int(os.getenv("VECTOR_MAX_SCAN_CANDIDATES", "20000"))

# Rows scored per batch on the streaming path. 256 × 1536 float32 is a
# ~1.5 MB numpy matrix; the transient cost is the batch's decoded JSON
# lists (~12 MB), which is released before the next batch is fetched.
SCORE_BATCH_SIZE = int(os.getenv("VECTOR_SCORE_BATCH", "256"))

# Log a search that grew RSS by more than this. Keeps the common case
# quiet while leaving a breadcrumb for the expensive outliers.
_RSS_LOG_THRESHOLD_MB = 25.0

# Postgres column + index added alongside the JSON `embedding` column.
# The JSON column stays the source of truth; this is a derived index for
# fast ranking, so a pgvector failure degrades to the streaming path
# instead of losing data.
PGVECTOR_COLUMN = "embedding_vec"

# Tri-state probe cache: None = not yet probed, True/False = resolved.
_pgvector_state: Optional[bool] = None


def _numpy():
    """Lazily import numpy. Returns None if unavailable.

    Deliberately lazy: numpy costs ~28 MB RSS on import and nothing in
    the request path needs it until the first streaming vector search
    (the pgvector path never does). Keeping it out of module import
    preserves the process's idle footprint.
    """
    try:
        import numpy
        return numpy
    except Exception:  # pragma: no cover — numpy is a hard requirement
        return None


def pgvector_available() -> bool:
    """True when the `embedding_vec` column + pgvector extension are usable.

    Probed once per process and cached. Any failure — non-Postgres
    backend, extension not installed, column missing because the
    migration has not run, insufficient privileges — resolves to False
    and permanently routes searches to the streaming fallback. That
    makes this a pure optimisation: correctness never depends on it.
    """
    global _pgvector_state
    if _pgvector_state is not None:
        return _pgvector_state
    if engine.dialect.name != "postgresql":
        _pgvector_state = False
        return False
    try:
        with engine.connect() as conn:
            ok = conn.execute(sa_text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'doc_chunks' AND column_name = :col"
            ), {"col": PGVECTOR_COLUMN}).first()
        _pgvector_state = ok is not None
    except Exception as exc:  # pragma: no cover — needs Postgres
        log.warning("pgvector probe failed, using streaming search: %s", exc)
        _pgvector_state = False
    if _pgvector_state:
        log.info("vector_store: pgvector ranking enabled")
    return _pgvector_state


def reset_pgvector_probe() -> None:
    """Clear the cached probe result (tests, and after a migration run)."""
    global _pgvector_state
    _pgvector_state = None


def _search_pgvector(
    q_vec: Sequence[float],
    *,
    ticker: Optional[str],
    source_types: Optional[Sequence[str]],
    sections: Optional[Sequence[str]],
    top_k: int,
) -> Optional[List[Dict[str, Any]]]:
    """Rank inside Postgres; return exactly `top_k` rows, or None to fall back.

    `<=>` is pgvector's cosine *distance*, so `1 - distance` reproduces
    the cosine *similarity* the streaming path and the old pure-Python
    implementation returned — callers comparing scores across paths see
    the same scale.

    Returning None (rather than raising or returning []) on any error is
    what makes this safe to ship without a Postgres test rig: a bad
    query degrades to the streaming path instead of breaking retrieval.
    """
    where = [f"{PGVECTOR_COLUMN} IS NOT NULL"]
    params: Dict[str, Any] = {
        "q": "[" + ",".join(repr(float(x)) for x in q_vec) + "]",
        "k": int(top_k),
    }
    if ticker:
        where.append("ticker = :ticker")
        params["ticker"] = ticker.upper()
    if source_types:
        where.append("source_type = ANY(:source_types)")
        params["source_types"] = list(source_types)
    if sections:
        where.append("section = ANY(:sections)")
        params["sections"] = list(sections)

    sql = sa_text(
        "SELECT id, ticker, source_type, source_id, section, period_end, "
        f"       text, meta, 1 - ({PGVECTOR_COLUMN} <=> CAST(:q AS vector)) AS score "
        "FROM doc_chunks "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY {PGVECTOR_COLUMN} <=> CAST(:q AS vector) "
        "LIMIT :k"
    )
    try:
        with SessionLocal() as db:
            rows = db.execute(sql, params).mappings().all()
    except Exception as exc:  # pragma: no cover — needs Postgres
        log.warning("pgvector search failed, falling back to streaming: %s", exc)
        return None
    return [
        {
            "id": r["id"],
            "ticker": r["ticker"],
            "source_type": r["source_type"],
            "source_id": r["source_id"],
            "section": r["section"],
            "period_end": r["period_end"],
            "text": r["text"],
            "score": float(r["score"]),
            "meta": r["meta"] or {},
        }
        for r in rows
    ]


def sync_pgvector_column(*, batch: int = 2000, max_batches: int = 50) -> int:
    """Copy JSON `embedding` values into the pgvector column.

    Idempotent and incremental — only touches rows where the vector
    column is still NULL, so it is safe to call after every upsert and
    again as a catch-up sweep for rows written before the migration.
    Bounded by `batch * max_batches` per call so a first run over a large
    corpus can't hold a long transaction open.

    Returns the number of rows populated (0 on any failure or when
    pgvector isn't active).
    """
    if not pgvector_available():
        return 0
    total = 0
    sql = sa_text(
        f"UPDATE doc_chunks SET {PGVECTOR_COLUMN} = CAST(embedding::text AS vector) "
        "WHERE id IN ("
        "  SELECT id FROM doc_chunks "
        f"  WHERE {PGVECTOR_COLUMN} IS NULL AND embedding IS NOT NULL "
        "    AND embedding_dim = :dim "
        "  LIMIT :batch"
        ")"
    )
    try:
        for _ in range(max_batches):
            with engine.begin() as conn:
                n = conn.execute(
                    sql, {"dim": emb_svc.EMBEDDING_DIM, "batch": int(batch)}
                ).rowcount or 0
            total += n
            if n < batch:
                break
    except Exception as exc:  # pragma: no cover — needs Postgres
        log.warning("pgvector backfill failed: %s", exc)
    if total:
        log.info("pgvector backfill populated %d rows", total)
    return total


def ensure_hnsw_index(*, maintenance_work_mem: str = "128MB") -> bool:
    """Build the HNSW index over `embedding_vec` if it doesn't exist.

    Deliberately NOT called from `init_db()`: on a large table this takes
    minutes and holds its transaction, and init_db runs on the startup
    path before the service can answer /health — building it there would
    fail the deploy. It belongs anywhere that can afford to block:
    `scripts/backfill_pgvector`, or the worker's background boot thread.

    Also correct to call *after* the rows are populated — building on an
    empty table and inserting row-by-row is substantially slower.

    `maintenance_work_mem` drives how much of the graph pgvector keeps in
    memory during the build. When it doesn't fit, pgvector falls back to a
    slower on-disk build rather than failing, so a conservative value is
    safe on a small database instance.

    Returns True when the index exists afterwards. Failure is non-fatal:
    search without the index does an exact scan — slower, identical
    results, and still no corpus in Python memory.
    """
    if not pgvector_available():
        return False
    try:
        with engine.begin() as conn:
            conn.execute(sa_text(f"SET maintenance_work_mem = '{maintenance_work_mem}'"))
            conn.execute(sa_text(
                # Cosine opclass must match the `<=>` operator that search
                # orders by, or the planner ignores the index entirely.
                "CREATE INDEX IF NOT EXISTS ix_doc_chunks_embedding_vec "
                "ON doc_chunks USING hnsw (embedding_vec vector_cosine_ops)"
            ))
        log.info("pgvector HNSW index ready on doc_chunks")
        return True
    except Exception as exc:  # pragma: no cover — needs Postgres
        log.warning(
            "pgvector HNSW index build failed (%s); search falls back to an "
            "exact scan — same results, slower", exc,
        )
        return False


def backfill_and_index(*, batch: int = 2000, max_batches: int = 25) -> Dict[str, Any]:
    """Populate `embedding_vec` for every eligible row, then build the index.

    The full catch-up pass, as opposed to the deliberately tiny sync that
    `upsert_source` does inline (which is capped so it can't stall a memo
    run). Loops until no rows remain rather than trusting a single call to
    drain the whole corpus.

    Idempotent and resumable — only NULL `embedding_vec` rows are touched,
    so an interrupted run picks up where it left off. Safe to call on every
    boot: once the corpus is populated this is a single cheap COUNT.
    """
    if not pgvector_available():
        return {"skipped": True, "populated": 0, "indexed": False}
    populated = 0
    # Bounded rather than `while True`. sync_pgvector_column returns 0 both
    # when it's done and when it fails, so this terminates on its own — but
    # this runs unattended on worker boot, and a spin loop there would be
    # far worse than a backfill that finishes on the next deploy. At the
    # default bounds this ceiling is 2.5M rows per boot.
    for _ in range(50):
        n = sync_pgvector_column(batch=batch, max_batches=max_batches)
        populated += n
        if n == 0:
            break
    else:
        log.warning(
            "pgvector backfill hit its per-run ceiling after %d rows; "
            "remaining rows will be picked up on the next run", populated,
        )
    indexed = ensure_hnsw_index()
    return {"skipped": False, "populated": populated, "indexed": indexed}


def upsert_source(
    *,
    ticker: Optional[str],
    source_type: str,
    source_id: Optional[int],
    chunks: Sequence[Dict[str, Any]],
    section: Optional[str] = None,
    period_end: Optional[date] = None,
) -> int:
    """Insert chunks for a (source_type, source_id). Replaces any prior
    chunks for the same source so re-ingesting a filing doesn't
    duplicate.

    Each chunk dict supports: text (required), section, period_end,
    meta. Embeddings are computed server-side using the configured
    embedding model.

    Returns the count of chunks written. Returns 0 on any failure so
    the calling agent flow keeps moving.
    """
    if not chunks:
        return 0
    texts = [c.get("text", "") for c in chunks]
    if not any(t.strip() for t in texts):
        return 0
    try:
        vectors = emb_svc.embed(texts)
    except Exception as exc:  # pragma: no cover
        log.warning("embedding batch failed for %s/%s: %s", source_type, source_id, exc)
        return 0

    written = 0
    try:
        with SessionLocal() as db:
            # Replace prior chunks for this source so re-ingest is idempotent.
            if source_id is not None:
                db.query(DocChunk).filter(
                    DocChunk.source_type == source_type,
                    DocChunk.source_id == source_id,
                ).delete(synchronize_session=False)
            for c, vec in zip(chunks, vectors):
                row = DocChunk(
                    ticker=(ticker or None),
                    source_type=source_type,
                    source_id=source_id,
                    section=c.get("section") or section,
                    period_end=c.get("period_end") or period_end,
                    text=c.get("text", ""),
                    token_count=len(c.get("text", "").split()),
                    embedding_model=emb_svc.EMBEDDING_MODEL if len(vec) == emb_svc.EMBEDDING_DIM else "hash-fallback",
                    embedding_dim=len(vec),
                    embedding=list(vec),
                    meta=c.get("meta") or {},
                )
                db.add(row)
                written += 1
            db.commit()
    except Exception as exc:  # pragma: no cover
        log.warning("upsert chunks failed: %s", exc)
        return 0
    # Mirror the freshly-written JSON embeddings into the pgvector column
    # so the rows are immediately searchable on the fast path. No-ops off
    # Postgres, and a failure only costs these rows their index entry
    # (the streaming path still finds them via the JSON column).
    #
    # Deliberately narrow bounds: this runs inline during filing indexing,
    # which happens inside a memo run. A filing is ~90 chunks, so 1000 rows
    # covers what was just written plus some catch-up drift. Calling the
    # default (up to 100k rows) here would make the first upsert after a
    # deploy backfill the entire historical corpus synchronously, inside a
    # user-facing memo. The rest converges over subsequent upserts; call
    # `sync_pgvector_column()` directly to force a full backfill.
    if written:
        sync_pgvector_column(batch=500, max_batches=2)
    return written


def search(
    query: str,
    *,
    ticker: Optional[str] = None,
    source_types: Optional[Sequence[str]] = None,
    sections: Optional[Sequence[str]] = None,
    top_k: int = 8,
    allow_global_scan: bool = False,
) -> List[Dict[str, Any]]:
    """Top-K chunks by cosine similarity, optionally filtered.

    Returns dicts: id, ticker, source_type, source_id, section,
    period_end, text, score, meta.

    **`ticker` is required unless `allow_global_scan=True`.** A falsy
    ticker used to fall through the `if ticker:` filter and scan the
    whole `doc_chunks` table — see `_reject_global_scan` for why that
    was a process-killer rather than merely slow. No caller wants a
    corpus-wide semantic search today; the flag exists so that a future
    one has to say so explicitly (and still gets bounded by
    `MAX_SCAN_CANDIDATES`).

    Two execution paths, picked at runtime:
      - **pgvector** (Postgres, extension available): the database does
        the ranking and returns exactly `top_k` rows.
      - **streaming fallback** (SQLite, or Postgres pre-backfill): scores
        in batches of `SCORE_BATCH_SIZE` with numpy and keeps only a
        top-k heap, so peak memory is O(batch) not O(matched rows).
    """
    if not query.strip():
        return []
    # Normalize before the guard: a whitespace-only ticker is truthy in
    # Python, so `if not ticker` would wave it through and then filter on
    # `"   "`, which matches nothing. Harmless here but the same class of
    # bug as the original — treat blank as absent, once, at the boundary.
    ticker = (ticker or "").strip() or None
    if not ticker and not allow_global_scan:
        _reject_global_scan(source_types, sections)
        return []
    try:
        q_vec = emb_svc.embed_one(query)
    except Exception as exc:  # pragma: no cover
        log.warning("embed query failed: %s", exc)
        return []

    started = memory_probe.rss_mb()

    hits: Optional[List[Dict[str, Any]]] = None
    if pgvector_available():
        hits = _search_pgvector(
            q_vec, ticker=ticker, source_types=source_types,
            sections=sections, top_k=top_k,
        )
    if hits is None:
        hits = _search_streaming(
            q_vec, ticker=ticker, source_types=source_types,
            sections=sections, top_k=top_k,
        )

    # Only pay for a second RSS read when the search was expensive enough
    # to matter — this fires on every specialist agent round.
    if started is not None:
        now = memory_probe.rss_mb()
        if now is not None and now - started > _RSS_LOG_THRESHOLD_MB:
            log.info(
                "vector_store.search grew RSS by %.1f MB (ticker=%s types=%s hits=%d)",
                now - started, ticker or "<global>",
                ",".join(source_types or []) or "*", len(hits),
            )
    return hits


def _reject_global_scan(
    source_types: Optional[Sequence[str]], sections: Optional[Sequence[str]],
) -> None:
    """Log a missing-ticker search loudly and return nothing.

    This is the guard for the 2026-08-12 Render OOM. `search()` treated a
    falsy ticker as "no filter", so an empty/None ticker turned a
    ~900-row per-ticker query into a full-table scan — every filing chunk
    in the S&P 500 corpus, at ~48 KB per embedding once SQLAlchemy
    deserialized the JSON into a list of boxed Python floats. That is
    multiple GB of allocation for a call whose contract is "return 4
    passages", and the resulting SIGKILL left no traceback because it
    bypassed the caller's `try/except` entirely.

    Callers reach this when a profile lookup missed and they passed
    `profile.get("ticker")` through anyway. Returning `[]` puts them on
    their existing BM25 / deterministic fallback, which is the same
    behaviour they got from an empty result set before — so this costs
    no functionality. WARNING (not DEBUG) because a caller hitting it is
    a bug in the caller, and we want it in production logs.
    """
    log.warning(
        "vector_store.search called without a ticker (types=%s sections=%s); "
        "refusing corpus-wide scan — pass allow_global_scan=True to opt in",
        ",".join(source_types or []) or "*",
        ",".join(sections or []) or "*",
    )


def _filtered_stmt(
    stmt,
    *,
    ticker: Optional[str],
    source_types: Optional[Sequence[str]],
    sections: Optional[Sequence[str]],
):
    """Apply the shared ticker / source_type / section predicates."""
    if ticker:
        stmt = stmt.where(DocChunk.ticker == ticker.upper())
    if source_types:
        stmt = stmt.where(DocChunk.source_type.in_(list(source_types)))
    if sections:
        stmt = stmt.where(DocChunk.section.in_(list(sections)))
    return stmt


def _search_streaming(
    q_vec: Sequence[float],
    *,
    ticker: Optional[str],
    source_types: Optional[Sequence[str]],
    sections: Optional[Sequence[str]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Score in bounded batches, keeping only a top-k heap.

    Three things keep peak memory flat regardless of corpus size:

    1. The candidate query selects `(id, embedding)` — *not* the ORM
       entity — so a chunk's `text` (~3.8 KB each) never loads for rows
       that lose. Text is fetched in a second query for the `top_k`
       winners only.
    2. `partitions(SCORE_BATCH_SIZE)` streams the result rather than
       calling `.all()`, and each batch's Python float lists are dropped
       before the next batch is fetched.
    3. `MAX_SCAN_CANDIDATES` caps rows considered even when the filters
       match more, so a pathological ticker cannot blow the ceiling.

    Scoring uses numpy (float32: ~6 KB per vector vs ~48 KB as a Python
    list) and a single matrix-vector product per batch, replacing the
    pure-Python `cosine` loop that ran ~4M float ops per search.
    """
    dim = len(q_vec)
    np = _numpy()
    heap: List[tuple] = []  # min-heap of (score, id) — smallest score at [0]
    truncated = False
    scanned = 0

    with SessionLocal() as db:
        stmt = _filtered_stmt(
            select(DocChunk.id, DocChunk.embedding),
            ticker=ticker, source_types=source_types, sections=sections,
        )
        # Newest chunks first so that if the cap truncates, it drops the
        # oldest material — the sane bias for equity research.
        stmt = stmt.order_by(DocChunk.id.desc()).limit(MAX_SCAN_CANDIDATES + 1)
        # `yield_per` is what makes this actually stream. Without it,
        # `db.execute()` buffers every row up front and `partitions()`
        # merely slices an already-materialized list — which measured
        # *worse* than the code it replaced, because the full result set
        # was resident AND numpy had been imported on top of it. With it,
        # SQLAlchemy fetches `SCORE_BATCH_SIZE` rows at a time from the
        # cursor and only that batch is ever live.
        result = db.execute(stmt.execution_options(yield_per=SCORE_BATCH_SIZE))

        if np is not None:
            q_arr = np.asarray(q_vec, dtype="float32")
            q_norm = float(np.linalg.norm(q_arr)) or 1.0

        for part in result.partitions(SCORE_BATCH_SIZE):
            ids: List[int] = []
            vecs: List[Any] = []
            for chunk_id, emb in part:
                scanned += 1
                if scanned > MAX_SCAN_CANDIDATES:
                    truncated = True
                    break
                # Skip rows whose embedding dim doesn't match the query
                # (embedding-model swap leaves mixed dims in the table).
                if not emb or len(emb) != dim:
                    continue
                ids.append(chunk_id)
                vecs.append(emb)
            if ids:
                if np is not None:
                    # errstate: the vectorized BLAS path raises spurious
                    # divide-by-zero / overflow FPE flags on some backends
                    # (macOS Accelerate is one) because SIMD lanes set the
                    # status register for masked elements. The inputs here
                    # are finite by construction and zero norms are handled
                    # on the next line, so there is no real condition to
                    # report — and this runs per specialist per memo, so
                    # the warnings would be pure log noise.
                    with np.errstate(all="ignore"):
                        mat = np.asarray(vecs, dtype="float32")
                        norms = np.linalg.norm(mat, axis=1)
                        norms[norms == 0] = 1.0
                        scores = (mat @ q_arr) / (norms * q_norm)
                    for chunk_id, score in zip(ids, scores.tolist()):
                        _heap_push(heap, score, chunk_id, top_k)
                    del mat, norms, scores
                else:  # pragma: no cover — numpy is a hard dependency
                    for chunk_id, emb in zip(ids, vecs):
                        _heap_push(heap, emb_svc.cosine(q_vec, emb), chunk_id, top_k)
            # Drop this batch's Python float lists before fetching the next.
            del ids, vecs
            if truncated:
                break

        if truncated:
            log.warning(
                "vector_store.search hit the %d-candidate cap (ticker=%s types=%s); "
                "results are drawn from the newest chunks only",
                MAX_SCAN_CANDIDATES, ticker or "<global>",
                ",".join(source_types or []) or "*",
            )

        if not heap:
            return []
        # Second query: hydrate text/meta for the winners only.
        ranked = sorted(heap, key=lambda t: t[0], reverse=True)
        score_by_id = {chunk_id: score for score, chunk_id in ranked}
        rows = db.execute(
            select(DocChunk).where(DocChunk.id.in_(list(score_by_id.keys())))
        ).scalars().all()
        by_id = {r.id: r for r in rows}

    out: List[Dict[str, Any]] = []
    for _score, chunk_id in ranked:
        row = by_id.get(chunk_id)
        if row is None:  # deleted between the two queries
            continue
        out.append(_row_to_hit(row, score_by_id[chunk_id]))
    return out


def _heap_push(heap: List[tuple], score: float, chunk_id: int, top_k: int) -> None:
    """Keep `heap` as the running top-`top_k` by score."""
    if len(heap) < top_k:
        heapq.heappush(heap, (score, chunk_id))
    elif score > heap[0][0]:
        heapq.heapreplace(heap, (score, chunk_id))


def _row_to_hit(row: DocChunk, score: float) -> Dict[str, Any]:
    return {
        "id": row.id,
        "ticker": row.ticker,
        "source_type": row.source_type,
        "source_id": row.source_id,
        "section": row.section,
        "period_end": row.period_end,
        "text": row.text,
        "score": float(score),
        "meta": row.meta or {},
    }
