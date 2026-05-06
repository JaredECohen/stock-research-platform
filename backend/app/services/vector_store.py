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

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select

from ..database import SessionLocal
from ..models import DocChunk
from . import embeddings as emb_svc

log = logging.getLogger(__name__)


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
    return written


def search(
    query: str,
    *,
    ticker: Optional[str] = None,
    source_types: Optional[Sequence[str]] = None,
    sections: Optional[Sequence[str]] = None,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Top-K chunks by cosine similarity, optionally filtered.

    Returns dicts: id, ticker, source_type, source_id, section,
    period_end, text, score, meta.
    """
    if not query.strip():
        return []
    try:
        q_vec = emb_svc.embed_one(query)
    except Exception as exc:  # pragma: no cover
        log.warning("embed query failed: %s", exc)
        return []

    with SessionLocal() as db:
        stmt = select(DocChunk)
        if ticker:
            stmt = stmt.where(DocChunk.ticker == ticker.upper())
        if source_types:
            stmt = stmt.where(DocChunk.source_type.in_(list(source_types)))
        if sections:
            stmt = stmt.where(DocChunk.section.in_(list(sections)))
        rows = db.execute(stmt).scalars().all()

    scored: List[Dict[str, Any]] = []
    for row in rows:
        if not row.embedding:
            continue
        # Skip rows whose embedding dim doesn't match the query (model swap).
        if len(row.embedding) != len(q_vec):
            continue
        score = emb_svc.cosine(q_vec, row.embedding)
        scored.append({
            "id": row.id,
            "ticker": row.ticker,
            "source_type": row.source_type,
            "source_id": row.source_id,
            "section": row.section,
            "period_end": row.period_end,
            "text": row.text,
            "score": score,
            "meta": row.meta or {},
        })
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]
