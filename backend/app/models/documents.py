"""Document + raw-data stores — cached chunks, provider cache, financial
history, filings, transcripts, vector chunks."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class CachedDocument(Base):
    """Generic cache for retrieved chunks (filings, transcripts, news)."""
    __tablename__ = "cached_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    source_id: Mapped[str] = mapped_column(String(128))
    section: Mapped[str] = mapped_column(String(128), default="")
    title: Mapped[str] = mapped_column(String(512), default="")
    url: Mapped[str] = mapped_column(String(1024), default="")
    text: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


Index("ix_doc_ticker_source", CachedDocument.ticker, CachedDocument.source_type)


class ProviderCache(Base):
    """Read-through cache for raw provider responses (Wave 9b).

    A capability-keyed JSON store that sits between `data_service` and
    the live provider chain. Each row caches one response (`profile`
    for AAPL, `prices:252` for NVDA, `news` for MSFT, etc.) with a
    fetched_at timestamp; consumers apply per-capability TTLs at read
    time.

    Stale rows are kept after expiry — `data_service` will serve them
    when a refetch fails so the platform degrades gracefully when
    providers are unavailable. A separate GC job can prune very-old
    rows once we have history depth.
    """

    __tablename__ = "provider_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capability: Mapped[str] = mapped_column(String(32), index=True)
    key: Mapped[str] = mapped_column(String(128), index=True)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


Index(
    "ix_provider_cache_capability_key",
    ProviderCache.capability, ProviderCache.key,
    unique=True,
)


# ---------------------------------------------------------------------------
# Wave 2 — Financial history depth
# ---------------------------------------------------------------------------

class FinancialPeriod(Base):
    """One row per (ticker, period, statement, line_item).

    Long format so 10y of revenue is one indexed SELECT instead of unpacking
    a JSON blob per quarter. The unique constraint on (ticker, period,
    statement, line_item) makes the backfill job idempotent — re-running it
    upserts existing rows rather than duplicating them.

    `period` is a free-form string ("2024Q4", "FY2024") so we accept both
    quarterly and annual cadences from upstream providers; `period_end` is
    the canonical date for ordering and as-of-date queries.
    """
    __tablename__ = "financial_periods"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    period: Mapped[str] = mapped_column(String(16))
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer, nullable=True)
    statement: Mapped[str] = mapped_column(String(16), index=True)  # income | balance | cash
    line_item: Mapped[str] = mapped_column(String(64))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    source: Mapped[str] = mapped_column(String(32), default="demo")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # Phase 6 (scorecard) — point-in-time availability. `available_at` is
    # the first date this figure could have been known: the provider's
    # filing date when FMP supplies one, else a matching `filing_docs`
    # row, else a documented lag after `period_end` (see
    # `services/scorecard_pit.derive_available_at`, which also names the
    # rule used in `available_at_source`: provider | filing_doc | lag_rule
    # | assumed_fye). Set on INSERT only — a restatement overwrites
    # `value` but never moves `available_at`, so history reads as
    # "restated values at original availability". Both nullable so
    # `reconcile_missing_columns` can add them to a live table; NULL means
    # "not yet derived" and the PIT snapshot excludes (and counts) the row.
    available_at: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    available_at_source: Mapped[str | None] = mapped_column(String(16), nullable=True)


Index(
    "ix_finperiod_unique",
    FinancialPeriod.ticker, FinancialPeriod.period,
    FinancialPeriod.statement, FinancialPeriod.line_item,
    unique=True,
)


class FilingDoc(Base):
    """SEC filing — raw text + parsed sections for retrieval.

    `accession_number` is the SEC's globally unique key, so it doubles as
    our idempotency token. `sections` holds the parsed cuts (risk_factors,
    mda, business, ...); `raw_text` is the full body for full-text search /
    BM25 over the corpus.
    """
    __tablename__ = "filing_docs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    accession_number: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    filing_type: Mapped[str] = mapped_column(String(16), index=True)  # 10-K | 10-Q | 8-K
    filing_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    sections: Mapped[dict] = mapped_column(JSON, default=dict)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    url: Mapped[str] = mapped_column(String(1024), default="")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EarningsTranscript(Base):
    """Quarterly earnings call — structured speaker blocks + full text.

    `(ticker, period)` is the natural key. `blocks` is the speaker-segmented
    list ([{speaker, role, segment, text}]) so retrieval can target prepared
    remarks vs. Q&A independently; `full_text` is the concatenated body.
    """
    __tablename__ = "earnings_transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    period: Mapped[str] = mapped_column(String(16), index=True)
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer, nullable=True)
    call_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    blocks: Mapped[list] = mapped_column(JSON, default=list)
    full_text: Mapped[str] = mapped_column(Text, default="")
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "period", name="uq_earnings_transcript_ticker_period"),
    )


# ---------------------------------------------------------------------------
# Wave 10 — vector chunks, postmortems, theme exposure, catalysts
# ---------------------------------------------------------------------------

class DocChunk(Base):
    """A retrievable chunk from a filing / transcript / memo.

    Wave 10. The vector index for RAG over the corpus. Embeddings are
    stored as JSON-serialized lists of floats (Postgres + sqlite both
    support this) — when pgvector is enabled in production, an out-of-
    band migration converts the column to `vector(<dim>)` and adds an
    HNSW index. Until then `vector_store.search` scores the JSON embeddings
    in Python; it never falls back to BM25 itself — BM25 over filing text
    (`retrieval_service`) is the filing analyst's own fallback when the
    vector search returns nothing, fails, or is skipped for lack of a ticker.

    `source_type` ∈ {filing, transcript, memo, news}. `source_id` is
    the foreign-key into the originating table (FilingDoc.id,
    EarningsTranscript.id, MemoSnapshot.id) — kept loose (no FK) so a
    chunk survives source deletion (we'd rather have an orphan than a
    broken constraint). `meta` holds source-specific keys (section,
    accession, period, etc.).
    """
    __tablename__ = "doc_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str | None] = mapped_column(String(16), index=True, nullable=True)
    source_type: Mapped[str] = mapped_column(String(16), index=True)
    source_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    section: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    text: Mapped[str] = mapped_column(Text, default="")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_doc_chunks_ticker_source", "ticker", "source_type"),
    )
