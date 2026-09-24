"""W7 §8.2 — read-only census of the vector corpus (`doc_chunks`).

Nothing has ever counted which chunks carry a usable vector, and the local
database is not representative of production, so a repair sized from a guess
would be sized wrong. This module answers, in aggregates only:

* how many chunks fall in each embedding class (`CHUNK_CLASSES`), with
  their token totals;
* where the unusable ones are (by month and by ticker), and how many are
  orphans (their source row is gone) or demo rows;
* which filings and transcripts have no chunks at all, and why
  (`SOURCE_CATEGORIES`);
* what a targeted repair would cost in tokens, dollars and bytes;
* how large the database is, because on `basic-256mb` Postgres storage, not
  money, is the binding constraint.

**No statement here selects `doc_chunks.embedding` values.** A 1536-dim JSON
embedding is ~31 KB, a census over the whole table would pull the corpus
into a 512 MiB worker, and nothing here needs a vector — only whether one is
there (`IS [NOT] NULL`) and how big it is stored (`pg_column_size`).
`test_corpus_inventory.test_census_never_selects_embedding_values` enforces
this with a SQL hook. `filing_docs.sections` bodies are MBs, so the fetch
error flag and whether any section holds text are computed by JSON path in
the database, for candidate ids only, and come back as flags.

Bytes are *gross*: what a run adds to the database before any VACUUM. An
in-place re-embed does not shrink Postgres — MVCC keeps the superseded value
until VACUUM, which only makes the space reusable — so the old value's size
is reported separately (`superseded_bytes`) and never credited.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, exists, func, literal_column, or_, select
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from ..database import SessionLocal
from ..models import Company, DocChunk, EarningsTranscript, FilingDoc, MemoSnapshot
from . import embeddings as emb_svc
from . import vector_store

# Unusable classes: a query vector of `EMBEDDING_DIM` can never reach them.
UNUSABLE_CLASSES = ("no_embedding", "hash_fallback", "legacy_dim", "other_dim")
# `awaiting_sync` is usable JSON that pgvector has not mirrored yet; the
# sync repairs it for free, so it is not a re-embed target.
CHUNK_CLASSES = (*UNUSABLE_CLASSES, "awaiting_sync", "ok")
SOURCE_CATEGORIES = ("indexable", "fetch_error", "empty_body", "out_of_scope", "demo")
SOURCE_KINDS = ("filing", "transcript")

LEGACY_DIM = 3072  # text-embedding-3-large, 2026-05-06 → 2026-05-30

# Storage model (W7 §8.2). A 1536-dim row is ~31 KB of JSON plus ~6 KB of
# pgvector; a new chunk also stores ~2.5 KB of text.
SQLITE_OK_EMBEDDING_BYTES = 31_000
VECTOR_COLUMN_BYTES = 6_160
CHUNK_TEXT_BYTES = 2_500
# With pgvector, every usable row also gets an HNSW entry: another copy of
# the 1536 float4s plus its layer-0 neighbour links (m=16 → 32 tids).
HNSW_ROW_BYTES = 6_400
# sqlite has no `pg_column_size`; a stored JSON float is ~20 characters, so
# a row's current embedding is estimated from its recorded dimension.
SQLITE_JSON_BYTES_PER_DIM = 20
AVG_SAMPLE_ROWS = 200

# Index-missing estimates: a filing's `word_count` is words, the chunker
# budgets model tokens (~1.3 per word on filing prose), at ~500 per chunk.
TOKENS_PER_WORD = 1.3
CHUNK_TARGET_TOKENS = 500

# Scope for sources worth indexing: recent enough to be read by an analyst.
FILING_WINDOW_DAYS = 3 * 365
TRANSCRIPT_WINDOW_DAYS = 2 * 365
_PHASE2_BATCH = 500

MB = 1024 * 1024


def vector_row_bytes(pgvector: bool) -> int:
    """Bytes a usable vector adds beyond its JSON embedding: the pgvector
    column, and its HNSW entry when pgvector is live."""
    return VECTOR_COLUMN_BYTES + (HNSW_ROW_BYTES if pgvector else 0)


# ---------------------------------------------------------------------------
# Chunk classes
# ---------------------------------------------------------------------------

def classify_chunk(dim: int | None, embedding_null: bool, vec_null: bool, pgvector: bool) -> str:
    """The one place a chunk's class is decided (census and tests share it)."""
    if embedding_null or dim is None:
        return "no_embedding"
    if dim == emb_svc.FALLBACK_DIM:
        return "hash_fallback"
    if dim == LEGACY_DIM:
        return "legacy_dim"
    if dim != emb_svc.EMBEDDING_DIM:
        return "other_dim"
    if pgvector and vec_null:
        return "awaiting_sync"
    return "ok"


def unusable_clause():
    """SQL twin of `classify_chunk(...) in UNUSABLE_CLASSES`.

    Shared with `corpus_repair`'s concurrent-change guard, so "what the census
    counts" and "what the repair may overwrite" cannot drift apart.
    """
    return or_(
        DocChunk.embedding.is_(None),
        DocChunk.embedding_dim.is_(None),
        DocChunk.embedding_dim != emb_svc.EMBEDDING_DIM,
    )


def _demo_accession(col):
    # `UPPER ... LIKE` rather than ILIKE: sqlite has no ILIKE.
    return func.upper(col).like("%DEMO%")


def _in_companies():
    return exists(select(Company.ticker).where(Company.ticker == DocChunk.ticker))


def reembed_target_clause():
    """Unusable chunks whose source row exists, is not DEMO, and whose ticker
    is a company. Orphan and demo rows are reported, never re-embedded."""
    filing_ok = and_(
        DocChunk.source_type == "filing",
        exists(select(FilingDoc.id).where(
            FilingDoc.id == DocChunk.source_id,
            ~_demo_accession(FilingDoc.accession_number),
        )),
    )
    transcript_ok = and_(
        DocChunk.source_type == "transcript",
        exists(select(EarningsTranscript.id).where(EarningsTranscript.id == DocChunk.source_id)),
    )
    return and_(unusable_clause(), or_(filing_ok, transcript_ok), _in_companies())


def current_embedding_bytes_expr(dialect: str):
    """Stored size of a row's embedding, without reading it."""
    if dialect == "postgresql":
        return func.coalesce(func.pg_column_size(DocChunk.embedding), 0)
    return case(
        (DocChunk.embedding.is_(None), 0),
        else_=func.coalesce(DocChunk.embedding_dim, 0) * SQLITE_JSON_BYTES_PER_DIM,
    )


def _dialect(db: Session) -> str:
    return db.get_bind().dialect.name


def avg_ok_embedding_bytes(db: Session) -> int:
    """Mean stored bytes of a good 1536-dim embedding (Postgres: measured)."""
    if _dialect(db) != "postgresql":
        return SQLITE_OK_EMBEDDING_BYTES
    sample = (
        select(func.pg_column_size(DocChunk.embedding).label("b"))
        .where(DocChunk.embedding_dim == emb_svc.EMBEDDING_DIM, DocChunk.embedding.is_not(None))
        .limit(AVG_SAMPLE_ROWS)
        .subquery()
    )
    avg = db.execute(select(func.avg(sample.c.b))).scalar()
    return int(avg) if avg else SQLITE_OK_EMBEDDING_BYTES


def database_bytes(db: Session) -> int | None:
    """Current size of the whole database, or None when it cannot be measured."""
    dialect = _dialect(db)
    if dialect == "postgresql":
        return int(db.execute(sa_text("SELECT pg_database_size(current_database())")).scalar() or 0)
    if dialect == "sqlite":
        pages = db.execute(sa_text("PRAGMA page_count")).scalar() or 0
        size = db.execute(sa_text("PRAGMA page_size")).scalar() or 0
        return int(pages) * int(size)
    return None


def _doc_chunks_bytes(db: Session) -> int | None:
    if _dialect(db) != "postgresql":
        return None
    return int(db.execute(sa_text("SELECT pg_total_relation_size('doc_chunks')")).scalar() or 0)


def _month_expr(dialect: str):
    # Literal formats, not bound parameters: the expression is repeated in
    # GROUP BY, and a server-side-binding driver would see two different
    # parameters there and refuse the query.
    if dialect == "postgresql":
        return func.to_char(DocChunk.created_at, literal_column("'YYYY-MM'"))
    return func.strftime(literal_column("'%Y-%m'"), DocChunk.created_at)


def _chunk_classes(db: Session, pgvector: bool) -> dict[str, dict[str, Any]]:
    # Literal 1/0 rather than bound parameters: the same CASE appears in the
    # SELECT list and the GROUP BY, and must compare equal in both.
    one: ColumnElement[Any] = literal_column("1")
    zero: ColumnElement[Any] = literal_column("0")
    emb_null = case((DocChunk.embedding.is_(None), one), else_=zero)
    group: list[Any] = [DocChunk.source_type, DocChunk.embedding_dim, emb_null]
    vec_null: ColumnElement[Any] = zero
    if pgvector:
        vec_null = case((literal_column("doc_chunks.embedding_vec").is_(None), one), else_=zero)
        group.append(vec_null)
    rows = db.execute(
        select(
            DocChunk.source_type, DocChunk.embedding_dim, emb_null.label("emb_null"),
            vec_null.label("vec_null"), func.count().label("n"),
            func.coalesce(func.sum(DocChunk.token_count), 0).label("tokens"),
        ).group_by(*group)
    ).all()
    classes: dict[str, dict[str, Any]] = {
        c: {"rows": 0, "tokens": 0, "by_source_type": {}} for c in CHUNK_CLASSES
    }
    for source_type, dim, e_null, v_null, n, tokens in rows:
        cls = classify_chunk(dim, bool(e_null), bool(v_null), pgvector)
        bucket = classes[cls]
        bucket["rows"] += int(n)
        bucket["tokens"] += int(tokens or 0)
        key = source_type or "unknown"
        bucket["by_source_type"][key] = bucket["by_source_type"].get(key, 0) + int(n)
    return classes


# ---------------------------------------------------------------------------
# Sources without chunks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissingSource:
    """A filing or transcript with zero chunks, and why it has none."""
    kind: str
    id: int
    ticker: str
    date: date | None
    word_count: int
    category: str
    # When the row was last written. A source written moments ago may have
    # its own post-pass in flight; `corpus_repair.index_missing` leaves it.
    fetched_at: datetime | None = None

    @property
    def identity(self) -> str:
        return f"{self.ticker}:{self.kind}:{self.id}"

    @property
    def est_tokens(self) -> int:
        return int(math.ceil(max(self.word_count, 0) * TOKENS_PER_WORD))

    @property
    def est_chunks(self) -> int:
        return max(1, int(math.ceil(self.est_tokens / CHUNK_TARGET_TOKENS)))


def _truthy_flag(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "0", "false", "null", "none")


def _section_has_text(key: str):
    """SQL: the section `key` holds something `index_filing` could chunk.

    Evaluated in the database; only the boolean comes back. A JSON list is
    returned as its JSON text, so an empty list (`[]`) is excluded here too.
    """
    value = FilingDoc.sections[key].as_string()
    trimmed = func.trim(value)
    return and_(value.is_not(None), func.length(trimmed) > 0, trimmed.not_in(("[]", '""', "{}", "null")))


def _filing_body_flags(db: Session, ids: list[int]) -> tuple[set[int], set[int]]:
    """(ids whose stored body is a failed fetch, ids with any text section).

    Phase 2 of the anti-join: JSON-path projections over the candidate ids
    only, never the `sections` body itself.

    The second set is what separates an indexable filing from one that will
    always index to zero chunks. `index_filing` chunks `sections` only, and
    an 8-K's items map to none of the section keys, so a fetched 8-K stores
    `{"_source_metadata": ...}` with its whole body in `raw_text` — a
    positive `word_count` and nothing to embed. Counted as indexable, each
    one took a nightly retry slot, newest first, until a real 10-Q whose
    post-pass hit an outage was never reached.
    """
    from .history_service import FILING_SECTION_KEYS

    one: ColumnElement[Any] = literal_column("1")
    zero: ColumnElement[Any] = literal_column("0")
    flag = FilingDoc.sections[("_source_metadata", "text_fetch_error")].as_string()
    has_text = case((or_(*(_section_has_text(k) for k in FILING_SECTION_KEYS)), one), else_=zero)
    fetch_errors: set[int] = set()
    with_text: set[int] = set()
    for start in range(0, len(ids), _PHASE2_BATCH):
        chunk = ids[start:start + _PHASE2_BATCH]
        rows = db.execute(select(FilingDoc.id, flag, has_text).where(FilingDoc.id.in_(chunk))).all()
        for fid, value, text_flag in rows:
            if _truthy_flag(value):
                fetch_errors.add(int(fid))
            if text_flag:
                with_text.add(int(fid))
    return fetch_errors, with_text


def missing_sources(db: Session, *, now: datetime, since: date | None = None) -> list[MissingSource]:
    """Every filing and transcript with no chunks, categorised.

    `since` narrows to sources dated on or after it (the nightly retry's
    recent window); undated sources are then excluded, since they cannot be
    shown to be recent.

    Categories, first match wins:
      * `demo` — DEMO accession, or a ticker that is not a company;
      * `fetch_error` — the stored filing body is a failed fetch (skipped
        on purpose at ingest; indexing it would index an error page);
      * `empty_body` — nothing to embed: no words, or (filings) no text in
        any section `index_filing` chunks — an 8-K's body is `raw_text` only;
      * `out_of_scope` — not an auto-pull-tier ticker and no memo, or older
        than the analysts' window (3 years for filings, 2 for transcripts);
      * `indexable` — everything else.
    """
    from .data_service import AUTO_PULL_TIERS

    tiers: dict[str, str] = {
        t: tier for t, tier in db.execute(select(Company.ticker, Company.universe_tier)).all()
    }
    memo_tickers = {t for (t,) in db.execute(select(MemoSnapshot.ticker).distinct()).all()}
    today = now.date()
    filing_floor = today - timedelta(days=FILING_WINDOW_DAYS)
    transcript_floor = today - timedelta(days=TRANSCRIPT_WINDOW_DAYS)

    filing_stmt = select(
        FilingDoc.id, FilingDoc.ticker, FilingDoc.filing_date, FilingDoc.word_count,
        FilingDoc.accession_number, FilingDoc.fetched_at,
    ).where(~exists(select(DocChunk.id).where(
        DocChunk.source_type == "filing", DocChunk.source_id == FilingDoc.id,
    )))
    transcript_stmt = select(
        EarningsTranscript.id, EarningsTranscript.ticker, EarningsTranscript.call_date,
        EarningsTranscript.word_count, EarningsTranscript.fetched_at,
    ).where(~exists(select(DocChunk.id).where(
        DocChunk.source_type == "transcript", DocChunk.source_id == EarningsTranscript.id,
    )))
    if since is not None:
        filing_stmt = filing_stmt.where(FilingDoc.filing_date >= since)
        transcript_stmt = transcript_stmt.where(EarningsTranscript.call_date >= since)

    filings = db.execute(filing_stmt).all()
    transcripts = db.execute(transcript_stmt).all()

    def in_scope(ticker: str) -> bool:
        return tiers.get(ticker) in AUTO_PULL_TIERS or ticker in memo_tickers

    out: list[MissingSource] = []
    non_demo_filings: list[int] = []
    staged: list[tuple[int, str, date | None, int, bool, datetime | None]] = []
    for fid, ticker, filed, wc, accession, fetched in filings:
        demo = ticker not in tiers or "DEMO" in (accession or "").upper()
        staged.append((int(fid), ticker, filed, int(wc or 0), demo, fetched))
        if not demo:
            non_demo_filings.append(int(fid))
    fetch_errors, with_text = _filing_body_flags(db, non_demo_filings)
    for fid, ticker, filed, wc, demo, fetched in staged:
        if demo:
            cat = "demo"
        elif fid in fetch_errors:
            cat = "fetch_error"
        elif wc <= 0 or fid not in with_text:
            cat = "empty_body"
        elif not in_scope(ticker) or filed is None or filed < filing_floor:
            cat = "out_of_scope"
        else:
            cat = "indexable"
        out.append(MissingSource("filing", fid, ticker, filed, wc, cat, fetched))
    for tid, ticker, called, wc, fetched in transcripts:
        wc = int(wc or 0)
        if ticker not in tiers:
            cat = "demo"
        elif wc <= 0:
            cat = "empty_body"
        elif not in_scope(ticker) or called is None or called < transcript_floor:
            cat = "out_of_scope"
        else:
            cat = "indexable"
        out.append(MissingSource("transcript", int(tid), ticker, called, wc, cat, fetched))
    return out


def index_missing_bytes(sources: Iterable[MissingSource], avg_ok: int, pgvector: bool = False) -> int:
    return sum(s.est_chunks for s in sources) * (avg_ok + vector_row_bytes(pgvector) + CHUNK_TEXT_BYTES)


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

def build_inventory(*, now: datetime | None = None) -> dict[str, Any]:
    """The full census. Read-only; aggregates and identities only."""
    now = now or datetime.utcnow()
    pgvector = vector_store.pgvector_available()
    with SessionLocal() as db:
        dialect = _dialect(db)
        classes = _chunk_classes(db, pgvector)
        total = sum(c["rows"] for c in classes.values())

        month = _month_expr(dialect)
        by_month = {
            str(m or "unknown"): int(n)
            for m, n in db.execute(
                select(month, func.count()).where(unusable_clause()).group_by(month).order_by(month)
            ).all()
        }
        by_ticker = [
            {"ticker": t, "rows": int(n)}
            for t, n in db.execute(
                select(DocChunk.ticker, func.count().label("n")).where(unusable_clause())
                .group_by(DocChunk.ticker).order_by(func.count().desc(), DocChunk.ticker).limit(20)
            ).all()
        ]

        orphan_rows = db.execute(
            select(DocChunk.source_type, func.count())
            .select_from(DocChunk)
            .outerjoin(FilingDoc, and_(DocChunk.source_type == "filing", FilingDoc.id == DocChunk.source_id))
            .outerjoin(EarningsTranscript, and_(
                DocChunk.source_type == "transcript", EarningsTranscript.id == DocChunk.source_id,
            ))
            .where(or_(
                and_(DocChunk.source_type == "filing", FilingDoc.id.is_(None)),
                and_(DocChunk.source_type == "transcript", EarningsTranscript.id.is_(None)),
            ))
            .group_by(DocChunk.source_type)
        ).all()
        orphans = {k: 0 for k in SOURCE_KINDS}
        orphans.update({str(k): int(n) for k, n in orphan_rows})
        orphans_total = sum(orphans.values())

        demo_accession = and_(DocChunk.source_type == "filing", exists(select(FilingDoc.id).where(
            FilingDoc.id == DocChunk.source_id, _demo_accession(FilingDoc.accession_number),
        )))
        not_company = ~_in_companies()
        demo_counts = db.execute(select(
            func.coalesce(func.sum(case((not_company, 1), else_=0)), 0),
            func.coalesce(func.sum(case((demo_accession, 1), else_=0)), 0),
            func.coalesce(func.sum(case((or_(not_company, demo_accession), 1), else_=0)), 0),
        ).select_from(DocChunk)).one()

        current = current_embedding_bytes_expr(dialect)
        re_rows, re_tokens, re_current = db.execute(
            select(
                func.count(), func.coalesce(func.sum(DocChunk.token_count), 0),
                func.coalesce(func.sum(current), 0),
            ).where(reembed_target_clause())
        ).one()
        avg_ok = avg_ok_embedding_bytes(db)

        missing = missing_sources(db, now=now)
        storage = {"database_bytes": database_bytes(db) if dialect == "postgresql" else None,
                   "doc_chunks_bytes": _doc_chunks_bytes(db)}

    categories: dict[str, dict[str, int]] = {
        k: {c: 0 for c in SOURCE_CATEGORIES} for k in (*SOURCE_KINDS, "total")
    }
    for s in missing:
        categories[s.kind][s.category] += 1
        categories["total"][s.category] += 1
    indexable = [s for s in missing if s.category == "indexable"]
    ix_tokens = sum(s.est_tokens for s in indexable)

    re_rows, re_tokens = int(re_rows), int(re_tokens)
    reembed_plan = {
        "rows": re_rows,
        "tokens": re_tokens,
        "usd": round(emb_svc.tokens_to_usd(re_tokens), 8),
        # Gross: the new value is written beside the old one, which only
        # VACUUM releases (and then only for reuse). Never net of it.
        "added_bytes": re_rows * (avg_ok + vector_row_bytes(pgvector)),
        "superseded_bytes": int(re_current),
    }
    index_plan = {
        "sources": len(indexable),
        "est_tokens": ix_tokens,
        "usd": round(emb_svc.tokens_to_usd(ix_tokens), 8),
        "added_bytes": index_missing_bytes(indexable, avg_ok, pgvector),
    }
    return {
        "generated_at": now.isoformat(),
        "dialect": dialect,
        "openai_configured": emb_svc._is_openai_available(),
        "semantic_embeddings": emb_svc.semantic_available(),
        "pgvector": pgvector,
        "chunks": {"total": total, "classes": classes,
                   "unusable": sum(classes[c]["rows"] for c in UNUSABLE_CLASSES)},
        "unusable_by_month": by_month,
        "unusable_by_ticker_top20": by_ticker,
        "orphans": {**orphans, "total": orphans_total},
        "demo_source_rows": {
            "ticker_not_in_companies": int(demo_counts[0]),
            "demo_accession": int(demo_counts[1]),
            "total": int(demo_counts[2]),
        },
        "sources_without_chunks": categories,
        "avg_ok_embedding_bytes": avg_ok,
        "repair_plan": {
            "reembed": reembed_plan,
            "index_missing": index_plan,
            "total_usd": round(reembed_plan["usd"] + index_plan["usd"], 8),
        },
        "storage": storage,
    }


def summary(inv: dict[str, Any]) -> dict[str, Any]:
    """The receipt-sized view: class counts, source categories, storage."""
    return {
        "chunks": {c: inv["chunks"]["classes"][c]["rows"] for c in CHUNK_CLASSES},
        "unusable": inv["chunks"]["unusable"],
        "orphans": inv["orphans"]["total"],
        "demo_source_rows": inv["demo_source_rows"]["total"],
        "sources_without_chunks": inv["sources_without_chunks"]["total"],
        "repair_plan": inv["repair_plan"],
        "storage": inv["storage"],
    }
