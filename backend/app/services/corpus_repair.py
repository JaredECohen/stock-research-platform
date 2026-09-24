"""W7 §8.3 — targeted, capped repair of the vector corpus.

Two repairs, both driven by `corpus_inventory` and nothing broader:

* **reembed** — re-embed *unusable* chunks (no embedding, 256-dim hash,
  3072-dim legacy, any other dim) **in place**. Chunk ids, text, source and
  section are unchanged; only the embedding columns and a provenance stamp
  in `meta.embedding_repair` are written, through an UPDATE guarded on the
  row still being unusable, so a concurrent re-index (which replaces the
  source's rows) is skipped rather than overwritten.
* **index_missing** — index in-scope filings and transcripts that have zero
  chunks, newest first, through `filing_memory.index_filing` /
  `index_transcript`. **Never** `post_pass`: no LLM diff, no memory writes.

Out of scope, by decision: sources with valid 1536 vectors are never
re-chunked (no blanket re-index, no chunker A/B inferred from one run), and
orphan and demo rows are reported by the census, never re-embedded and never
deleted.

Caps. Every run is bounded by dollars, rows or sources, and added bytes,
checked *before* each batch or source against the projected cost. Spend is
the provider's `usage.total_tokens` (`embeddings.usage_meter`); a chunk's
`token_count` (a source's `word_count`) is only the pre-batch estimate, so
the one overshoot the USD cap allows is the first batch or source the
provider bills above that estimate. From then on each projection is scaled
by the billed/estimated ratio observed so far (never below 1), so the run
stops before a second. Bytes are gross (`corpus_inventory`: nothing is
credited for superseded values, which only VACUUM releases). `max_db_mb` is
an absolute ceiling on the whole database, measured (`pg_database_size`)
before every batch and source — `basic-256mb` Postgres has unknown free
storage, and storage, not money, is what a repair can exhaust, so this
measured ceiling, not the estimate, is the backstop. Every cap must be a
finite non-negative number (`max_db_mb` positive); NaN or inf would make a
comparison that never binds.

Callers: the operator CLI (`scripts/corpus_repair.py`, dry-run `--plan`
first) and one bounded automatic retry inside `history_backfill` (≤10 recent
sources, ≤$0.10, ≤20 MB a night). Nothing here runs on import.
"""
from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult

from ..database import SessionLocal
from ..models import DocChunk, EarningsTranscript, FilingDoc
from . import corpus_inventory as inv_svc
from . import embeddings as emb_svc
from . import vector_store

log = logging.getLogger(__name__)

REEMBED_BATCH = 64
# How long after a source row is written before `index_missing` may touch
# it (see there). A 10-K's post-pass takes a minute or two.
SETTLE = timedelta(hours=1)
CLASSES = ("reembed", "index-missing")

# Why a run ended early. None means it ran out of work.
STOP_MAX_USD = "max_usd"
STOP_MAX_ROWS = "max_rows"
STOP_MAX_SOURCES = "max_sources"
STOP_MAX_ADDED_MB = "max_added_mb"
STOP_MAX_DB_MB = "max_db_mb"
STOP_DB_SIZE_UNKNOWN = "db_size_unknown"
STOP_EMBEDDING_UNAVAILABLE = "embedding_unavailable"


class RepairRefused(RuntimeError):
    """The run cannot start safely (CLI exit 2)."""


def new_receipt_id(now: datetime | None = None) -> str:
    now = now or datetime.utcnow()
    return f"cr-{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


def _valid_cap(value: object, *, positive: bool = False) -> bool:
    # `nan < 0` is False and `x > inf` never true: a NaN or inf cap would
    # pass a sign check and then never stop anything.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and (value > 0 if positive else value >= 0)


def _require_caps(**caps: float | int | None) -> None:
    for name, value in caps.items():
        if not _valid_cap(value):
            raise ValueError(f"corpus repair cap {name} must be a finite non-negative number, got {value!r}")


def _require_db_ceiling(max_db_mb: float | None) -> None:
    """`None` (no ceiling) is for the nightly retry only; `run` requires one."""
    if max_db_mb is not None and not _valid_cap(max_db_mb, positive=True):
        raise ValueError(f"corpus repair cap max_db_mb must be a finite positive number, got {max_db_mb!r}")


def _billing_ratio(billed: int, estimated: int) -> float:
    """How far the provider has billed above our pre-batch estimates so far.
    Never below 1: an over-estimate must not loosen the cap."""
    return max(1.0, billed / estimated) if estimated > 0 else 1.0


def _require_semantic() -> None:
    # A process that may emit hash vectors would "repair" unusable rows into
    # more unusable rows.
    if not emb_svc.semantic_available():
        raise RepairRefused("no semantic embedding provider (OpenAI key missing, or demo-only mode)")


def _ceiling_stop(db_bytes: int | None, projected_bytes: int, max_db_mb: float | None) -> str | None:
    if max_db_mb is None:
        return None
    if db_bytes is None:
        return STOP_DB_SIZE_UNKNOWN
    if db_bytes + max(projected_bytes, 0) > max_db_mb * inv_svc.MB:
        return STOP_MAX_DB_MB
    return None


def _measure_db() -> int | None:
    with SessionLocal() as db:
        return inv_svc.database_bytes(db)


def _id_ranges(ids: Sequence[int]) -> list[list[int]]:
    """Touched ids as inclusive [start, end] runs: exact, and receipt-sized
    even when a run repairs 100k rows in id order."""
    out: list[list[int]] = []
    for i in sorted(ids):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def plan(classes: Sequence[str] = CLASSES, *, now: datetime | None = None) -> dict[str, Any]:
    """What a repair would do. Writes nothing."""
    unknown = set(classes) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown corpus repair class: {sorted(unknown)}")
    inv = inv_svc.build_inventory(now=now)
    keys = {"reembed": "reembed", "index-missing": "index_missing"}
    return {
        "targets": {c: inv["repair_plan"][keys[c]] for c in classes},
        "inventory": inv,
    }


# ---------------------------------------------------------------------------
# reembed
# ---------------------------------------------------------------------------

def reembed(
    *,
    max_usd: float,
    max_rows: int,
    max_added_mb: float,
    max_db_mb: float | None = None,
    receipt: str | None = None,
    now: datetime | None = None,
    batch_size: int = REEMBED_BATCH,
) -> dict[str, Any]:
    """Re-embed unusable, non-orphan, non-demo chunks in place, in id order.

    Resumable: batches commit one at a time, and a stopped run leaves every
    committed batch repaired, so the next run picks up the remainder.
    """
    _require_caps(max_usd=max_usd, max_rows=max_rows, max_added_mb=max_added_mb)
    _require_db_ceiling(max_db_mb)
    _require_semantic()
    receipt = receipt or new_receipt_id(now)
    max_added = max_added_mb * inv_svc.MB

    with SessionLocal() as db:
        avg_ok = inv_svc.avg_ok_embedding_bytes(db)
    # Gross per row: the new JSON value and vector are written beside the
    # old ones (MVCC), so the old row's size is never credited back.
    per_row = avg_ok + inv_svc.vector_row_bytes(vector_store.pgvector_available())
    # The `embedding` column itself is never selected.
    stmt = select(
        DocChunk.id, DocChunk.text, DocChunk.token_count, DocChunk.meta,
        DocChunk.embedding_dim, DocChunk.embedding_model,
    ).where(inv_svc.reembed_target_clause()).order_by(DocChunk.id)

    res: dict[str, Any] = {
        "class": "reembed", "receipt": receipt, "rows_repaired": 0, "skipped_changed": 0,
        "skipped_empty": 0, "batches": 0, "tokens": 0, "tokens_estimated": 0, "usd": 0.0,
        "added_bytes_est": 0, "stopped_reason": None,
    }
    touched: list[int] = []
    attempted = 0
    estimated_done = 0  # pre-batch estimates of the batches already billed
    last_id = 0
    while True:
        with SessionLocal() as db:
            rows = db.execute(stmt.where(DocChunk.id > last_id).limit(batch_size)).all()
        if not rows:
            break
        remaining = max_rows - attempted
        if remaining <= 0:
            res["stopped_reason"] = STOP_MAX_ROWS
            break
        rows = rows[:remaining]
        empty = [r for r in rows if not (r.text or "").strip()]
        work = [r for r in rows if (r.text or "").strip()]
        est_tokens = sum(int(r.token_count or 0) or emb_svc.count_tokens(r.text) for r in work)
        est_bytes = len(work) * per_row
        projected = est_tokens * _billing_ratio(res["tokens"], estimated_done)
        if work and emb_svc.tokens_to_usd(res["tokens"] + projected) > max_usd:
            res["stopped_reason"] = STOP_MAX_USD
            break
        if work and res["added_bytes_est"] + est_bytes > max_added:
            res["stopped_reason"] = STOP_MAX_ADDED_MB
            break
        ceiling = _ceiling_stop(_measure_db(), est_bytes, max_db_mb) if work else None
        if ceiling:
            res["stopped_reason"] = ceiling
            break
        res["skipped_empty"] += len(empty)
        attempted += len(rows)
        last_id = rows[-1].id
        if not work:
            continue
        with emb_svc.usage_meter() as meter:
            try:
                vectors = emb_svc.embed([r.text for r in work])
            except emb_svc.EmbeddingUnavailable:
                vectors = None
        res["tokens"] += meter.total_tokens
        res["tokens_estimated"] += meter.estimated_tokens
        estimated_done += est_tokens
        if vectors is None or any(len(v) != emb_svc.EMBEDDING_DIM for v in vectors):
            res["stopped_reason"] = STOP_EMBEDDING_UNAVAILABLE
            break
        stamp_at = (now or datetime.utcnow()).isoformat()
        with SessionLocal() as db:
            for r, vec in zip(work, vectors):
                meta = dict(r.meta or {})
                meta["embedding_repair"] = {
                    "from_model": r.embedding_model, "from_dim": r.embedding_dim,
                    "at": stamp_at, "receipt": receipt,
                }
                # Guarded: only a row that is still unusable is overwritten. A
                # concurrent re-index deletes and re-inserts the source's rows,
                # so this id is gone (0 rows) and nothing good is clobbered.
                done = cast(CursorResult, db.execute(
                    update(DocChunk)
                    .where(DocChunk.id == r.id, inv_svc.unusable_clause())
                    .values(
                        embedding=list(vec), embedding_model=emb_svc.EMBEDDING_MODEL,
                        embedding_dim=emb_svc.EMBEDDING_DIM, meta=meta,
                    )
                    .execution_options(synchronize_session=False)
                )).rowcount
                if done == 1:
                    res["rows_repaired"] += 1
                    touched.append(int(r.id))
                else:
                    res["skipped_changed"] += 1
            db.commit()
        res["batches"] += 1
        res["added_bytes_est"] += est_bytes
        del vectors
    res["usd"] = round(emb_svc.tokens_to_usd(res["tokens"]), 8)
    res["chunk_id_ranges"] = _id_ranges(touched)
    if touched:
        res["pgvector_sync"] = vector_store.backfill_and_index()
    log.info(
        "corpus_repair reembed receipt=%s repaired=%d skipped_changed=%d tokens=%d stopped=%s",
        receipt, res["rows_repaired"], res["skipped_changed"], res["tokens"], res["stopped_reason"],
    )
    return res


# ---------------------------------------------------------------------------
# index_missing
# ---------------------------------------------------------------------------

def _load_detached(kind: str, source_id: int):
    """The source row, detached, or None when it gained chunks or vanished.

    Re-checked here rather than trusted from the census: a poller may have
    indexed it since, and indexing it again would replace good chunks.
    """
    model = FilingDoc if kind == "filing" else EarningsTranscript
    with SessionLocal() as db:
        has_chunks = db.execute(
            select(DocChunk.id).where(DocChunk.source_type == kind, DocChunk.source_id == source_id).limit(1)
        ).first()
        if has_chunks is not None:
            return None
        row = db.get(model, source_id)
        if row is None:
            return None
        db.expunge(row)
        return row


def index_missing(
    *,
    max_sources: int,
    max_usd: float,
    max_added_mb: float,
    max_db_mb: float | None = None,
    recent_days: int | None = None,
    receipt: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Index zero-chunk `indexable` sources, newest first, under the caps.

    `recent_days` narrows to sources dated within that many days of `now` —
    the nightly retry's window, which catches a post-pass that failed on an
    OpenAI outage without walking the whole history every night.
    """
    _require_caps(max_sources=max_sources, max_usd=max_usd, max_added_mb=max_added_mb)
    _require_db_ceiling(max_db_mb)
    if recent_days is not None and recent_days < 0:
        raise ValueError(f"recent_days must be non-negative, got {recent_days!r}")
    _require_semantic()
    now = now or datetime.utcnow()
    receipt = receipt or new_receipt_id(now)
    max_added = max_added_mb * inv_svc.MB
    since: date | None = (now - timedelta(days=recent_days)).date() if recent_days is not None else None

    from . import filing_memory

    with SessionLocal() as db:
        candidates = [
            s for s in inv_svc.missing_sources(db, now=now, since=since) if s.category == "indexable"
        ]
        avg_ok = inv_svc.avg_ok_embedding_bytes(db)
    # A source written in the last `SETTLE` may have its own post-pass in
    # flight (`_ingest_filings` commits the row, then indexes it). Indexing
    # it here too would race that post-pass: `doc_chunks` has no unique key
    # and `upsert_source` takes no lock, so under READ COMMITTED both chunk
    # sets survive. A post-pass that failed is not in flight, so waiting out
    # the window costs at most one night. Wall clock, not `now`: `fetched_at`
    # is stamped from it.
    settled_before = datetime.utcnow() - SETTLE
    in_flight = [s for s in candidates if s.fetched_at is not None and s.fetched_at > settled_before]
    in_flight_keys = {(s.kind, s.id) for s in in_flight}
    candidates = [s for s in candidates if (s.kind, s.id) not in in_flight_keys]
    candidates.sort(key=lambda s: (s.date or date.min, s.id), reverse=True)
    per_chunk = avg_ok + inv_svc.vector_row_bytes(vector_store.pgvector_available()) + inv_svc.CHUNK_TEXT_BYTES

    res: dict[str, Any] = {
        "class": "index-missing", "receipt": receipt, "candidates": len(candidates),
        "sources_indexed": 0, "chunks_written": 0, "tokens": 0, "tokens_estimated": 0,
        "usd": 0.0, "added_bytes_est": 0, "stopped_reason": None,
        "indexed": [], "empty": [], "skipped_changed": [], "failures": [], "deferred": [],
        "in_flight": [s.identity for s in in_flight],
    }
    # `max_sources` counts sources that did indexing work (indexed, or
    # failed trying). One that turned out to have nothing to embed, or that
    # gained chunks since the census, costs no slot: a slot spent on it is
    # a real source deferred for nothing.
    attempted = 0
    estimated_done = 0
    for i, src in enumerate(candidates):
        if attempted >= max_sources:
            res["stopped_reason"] = STOP_MAX_SOURCES
        elif emb_svc.tokens_to_usd(
            res["tokens"] + src.est_tokens * _billing_ratio(res["tokens"], estimated_done)
        ) > max_usd:
            res["stopped_reason"] = STOP_MAX_USD
        elif res["added_bytes_est"] + src.est_chunks * per_chunk > max_added:
            res["stopped_reason"] = STOP_MAX_ADDED_MB
        else:
            res["stopped_reason"] = _ceiling_stop(_measure_db(), src.est_chunks * per_chunk, max_db_mb)
        if res["stopped_reason"]:
            res["deferred"] = [s.identity for s in candidates[i:]]
            break
        row = _load_detached(src.kind, src.id)
        if row is None:
            res["skipped_changed"].append(src.identity)
            continue
        indexer = filing_memory.index_filing if src.kind == "filing" else filing_memory.index_transcript
        with emb_svc.usage_meter() as meter:
            try:
                written = indexer(row)
                error: str | None = None
            except emb_svc.EmbeddingUnavailable:
                written, error = 0, STOP_EMBEDDING_UNAVAILABLE
            except Exception as exc:
                # Named by type only (exception text may carry secrets), and
                # isolated: one bad source must not stop the rest.
                written, error = 0, type(exc).__name__
        del row
        res["tokens"] += meter.total_tokens
        res["tokens_estimated"] += meter.estimated_tokens
        if meter.total_tokens:
            estimated_done += src.est_tokens
        if error == STOP_EMBEDDING_UNAVAILABLE:
            # The provider is down: every later source would fail the same way.
            res["stopped_reason"] = STOP_EMBEDDING_UNAVAILABLE
            res["deferred"] = [s.identity for s in candidates[i:]]
            break
        if error:
            attempted += 1
            res["failures"].append(f"{src.identity}:{error}")
            continue
        if not written:
            res["empty"].append(src.identity)
            continue
        attempted += 1
        res["sources_indexed"] += 1
        res["chunks_written"] += int(written)
        res["added_bytes_est"] += int(written) * per_chunk
        res["indexed"].append(src.identity)
    res["usd"] = round(emb_svc.tokens_to_usd(res["tokens"]), 8)
    log.info(
        "corpus_repair index_missing receipt=%s indexed=%d chunks=%d tokens=%d deferred=%d stopped=%s",
        receipt, res["sources_indexed"], res["chunks_written"], res["tokens"],
        len(res["deferred"]), res["stopped_reason"],
    )
    return res


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def run(
    *,
    mode: str,
    klass: str = "all",
    max_usd: float = 3.0,
    max_rows: int = 100_000,
    max_sources: int = 200,
    max_added_mb: float = 300.0,
    max_db_mb: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One CLI invocation → one receipt. `exit_code`: 0 done, 1 stopped
    early (a cap, the DB ceiling, or the provider), 2 refused."""
    now = now or datetime.utcnow()
    caps: dict[str, Any] = {"max_usd": max_usd, "max_rows": max_rows, "max_sources": max_sources,
                            "max_added_mb": max_added_mb, "max_db_mb": max_db_mb}
    receipt: dict[str, Any] = {
        "receipt_id": new_receipt_id(now), "mode": mode, "class": klass,
        "openai_configured": emb_svc._is_openai_available(),
        "semantic_embeddings": emb_svc.semantic_available(),
        # A non-finite cap is echoed as text: `json.dumps` would print a bare
        # NaN/Infinity token, which strict JSON parsers reject.
        "caps": {k: (v if v is None or math.isfinite(v) else str(v)) for k, v in caps.items()},
        "stopped_reason": None,
    }
    classes = CLASSES if klass == "all" else (klass,)
    if mode == "inventory":
        receipt["inventory"] = inv_svc.build_inventory(now=now)
        receipt["exit_code"] = 0
        return receipt
    if mode == "plan":
        planned = plan(classes, now=now)
        receipt["before"] = inv_svc.summary(planned["inventory"])
        receipt["targets"] = planned["targets"]
        receipt["exit_code"] = 0
        return receipt
    if mode != "apply":
        raise ValueError(f"unknown corpus repair mode: {mode!r}")

    refusal = None
    if klass not in CLASSES:
        refusal = "--apply needs one --class: reembed or index-missing"
    elif not emb_svc.semantic_available():
        refusal = "no semantic embedding provider (OpenAI key missing, or demo-only mode)"
    elif max_db_mb is None:
        refusal = "--apply needs --max-db-mb: an absolute database size ceiling"
    else:
        # Before the census: a bad cap must refuse (exit 2) with a receipt,
        # not escape as a traceback (exit 1, which means "stopped, resumable").
        try:
            _require_caps(max_usd=max_usd, max_added_mb=max_added_mb,
                          **({"max_rows": max_rows} if klass == "reembed" else {"max_sources": max_sources}))
            _require_db_ceiling(max_db_mb)
        except ValueError as exc:
            refusal = str(exc)
    if refusal:
        receipt["refused"] = refusal
        receipt["exit_code"] = 2
        return receipt

    before = inv_svc.build_inventory(now=now)
    receipt["before"] = inv_svc.summary(before)
    receipt["targets"] = {klass: before["repair_plan"]["reembed" if klass == "reembed" else "index_missing"]}
    try:
        if klass == "reembed":
            result = reembed(max_usd=max_usd, max_rows=max_rows, max_added_mb=max_added_mb,
                             max_db_mb=max_db_mb, receipt=receipt["receipt_id"], now=now)
        else:
            result = index_missing(max_sources=max_sources, max_usd=max_usd, max_added_mb=max_added_mb,
                                   max_db_mb=max_db_mb, receipt=receipt["receipt_id"], now=now)
            if result["chunks_written"]:
                result["pgvector_sync"] = vector_store.backfill_and_index()
    except RepairRefused as exc:
        receipt["refused"] = str(exc)
        receipt["exit_code"] = 2
        return receipt
    receipt["results"] = result
    receipt["stopped_reason"] = result["stopped_reason"]
    receipt["after"] = inv_svc.summary(inv_svc.build_inventory(now=now))
    receipt["exit_code"] = 1 if result["stopped_reason"] else 0
    return receipt
