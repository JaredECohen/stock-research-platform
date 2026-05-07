"""SQLAlchemy ORM models for MarketMosaic.

We persist the master security universe, daily price snapshots, generated
agent memos and screener score history. Detailed financial statements are
loaded from JSON-backed providers rather than the DB to keep the schema lean
in demo mode; in live mode the same provider interface returns identical
shapes from external APIs and can be cached into ORM tables if desired.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base
# Import the cache models so init_db()'s create_all picks them up. Imported
# for the side-effect of registering on Base.metadata; the symbols themselves
# are re-exported via app.cache.
from .cache.snapshots import CacheCostLog, ResearchSnapshot  # noqa: F401


class Company(Base):
    __tablename__ = "companies"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    company_name: Mapped[str] = mapped_column(String(256))
    exchange: Mapped[str] = mapped_column(String(32), default="NASDAQ")
    sector: Mapped[str] = mapped_column(String(64))
    industry: Mapped[str] = mapped_column(String(128))
    sub_industry: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    country: Mapped[str] = mapped_column(String(8), default="US")
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cik: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    isin: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    cusip: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    business_description: Mapped[str] = mapped_column(Text, default="")
    # String(16) — covers month names ("September" is 9 chars, longest is
    # "September"). Earlier String(8) fit the demo dataset's abbreviated
    # values but overflowed in Postgres for AAPL/V/SBUX (live FMP
    # profiles emit the full month name). SQLite was tolerant; Postgres
    # rejects with "value too long for type character varying(8)".
    fiscal_year_end: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_etf: Mapped[bool] = mapped_column(Boolean, default=False)
    beta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shares_outstanding: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_price_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Universe tiering (Phase F). Three states:
    #   data_only          — provider data ingested; no memo generated unless
    #                        the UI explicitly asks for it. Default for the
    #                        long tail of the universe.
    #   auto_analysis      — full memo refreshes automatically on EDGAR /
    #                        earnings deltas. Reserved for the curated
    #                        tier-1 watch list (e.g., 11 sectors × 2 names).
    #   analyzed_on_demand — first promoted out of `data_only` when the UI
    #                        called for a deep analysis. The memo is kept,
    #                        but auto-refresh is off; the next refresh
    #                        happens only on a manual request.
    universe_tier: Mapped[str] = mapped_column(
        String(24), default="data_only", index=True
    )

    memos: Mapped[list["StockMemo"]] = relationship(back_populates="company")


class StockMemo(Base):
    __tablename__ = "stock_memos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), ForeignKey("companies.ticker"), index=True)
    rating_label: Mapped[str] = mapped_column(String(32))
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    one_sentence_thesis: Mapped[str] = mapped_column(Text)
    body: Mapped[dict] = mapped_column(JSON, default=dict)
    scores: Mapped[dict] = mapped_column(JSON, default=dict)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    generation_mode: Mapped[str] = mapped_column(String(16), default="demo")

    company: Mapped[Company] = relationship(back_populates="memos")


class ScreenerScore(Base):
    __tablename__ = "screener_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    quality: Mapped[float] = mapped_column(Float, default=0.0)
    growth: Mapped[float] = mapped_column(Float, default=0.0)
    valuation: Mapped[float] = mapped_column(Float, default=0.0)
    earnings_momentum: Mapped[float] = mapped_column(Float, default=0.0)
    catalyst: Mapped[float] = mapped_column(Float, default=0.0)
    macro_fit: Mapped[float] = mapped_column(Float, default=0.0)
    risk: Mapped[float] = mapped_column(Float, default=0.0)
    pm_conviction: Mapped[float] = mapped_column(Float, default=0.0)
    one_line_thesis: Mapped[str] = mapped_column(Text, default="")
    main_catalyst: Mapped[str] = mapped_column(Text, default="")
    main_risk: Mapped[str] = mapped_column(Text, default="")
    theme: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
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
    payload_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


Index(
    "ix_provider_cache_capability_key",
    ProviderCache.capability, ProviderCache.key,
    unique=True,
)


class ScreenerMetric(Base):
    """Per-ticker raw metrics for rule-based screening (Wave 9b Phase 4).

    Snapshot-style table — one row per ticker, refreshed nightly along
    with `screener_scores`. Columns are deliberately concrete (P/E, EV/
    EBITDA, gross margin, …) so the custom-screen endpoint can WHERE
    against them with simple SQL rather than reaching into long-format
    `financial_periods` for every rule.

    All numeric values may be NULL when underlying data is missing
    (e.g. forward_pe before estimates land); callers must handle None.
    """

    __tablename__ = "screener_metrics"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    pe_ttm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    forward_pe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ev_ebitda: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ev_revenue: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gross_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    op_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fcf_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roic: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    debt_to_ebitda: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    revenue_growth_yoy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dividend_yield: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    beta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MemoSnapshot(Base):
    """Versioned, lineage-aware persistence of every generated memo.

    Each `(ticker, version)` is one immutable snapshot. A new version is
    created either by a `full_reanalysis` (after a filing or earnings
    delta) or an `incremental_patch` (driven by material news) — the
    latter case stores `parent_version` so reviewers can trace what was
    inherited from the prior memo and what was patched.

    The `memo_json` column stores the entire `StockMemoOut` so we don't
    have to re-derive its shape; ad-hoc fields evolve without schema
    migrations as long as the Pydantic model stays additive.

    Wave 1C: `as_of_date` distinguishes a backtest run (memo reproduced
    as of an earlier date) from a live memo (`generated_at` only). When
    set, the memo SHOULD reflect only data observable on or before that
    date and is excluded from the default `latest_memo` lookup.
    """
    __tablename__ = "memo_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    trigger: Mapped[str] = mapped_column(String(48), default="full_reanalysis")
    memo_json: Mapped[dict] = mapped_column(JSON, default=dict)
    revision_log: Mapped[list] = mapped_column(JSON, default=list)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True,
    )
    as_of_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, index=True,
    )


Index("ix_memo_snap_ticker_version", MemoSnapshot.ticker, MemoSnapshot.version, unique=True)


class PortfolioRun(Base):
    __tablename__ = "portfolio_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LLMCallLog(Base):
    """Append-only audit log of every provider LLM call (Wave 1A).

    One row per real LLM call with attribution to the agent that made it
    and the memo run it belongs to. Used for cost-per-run / cost-per-agent
    breakdowns + slow-call audit. Distinct from `CacheCostLog` (which
    tracks snapshot-write savings) — this one tracks actual provider
    spend regardless of caching.

    GC: rows older than 90 days are deleted by a daily monitoring job
    when ENABLE_MONITORING is on; otherwise they accumulate harmlessly
    (rows are small).
    """
    __tablename__ = "llm_call_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    ticker: Mapped[Optional[str]] = mapped_column(String(16), index=True, nullable=True)
    agent_name: Mapped[str] = mapped_column(String(64), index=True, default="unknown")
    provider: Mapped[str] = mapped_column(String(16), default="openai")
    model: Mapped[str] = mapped_column(String(64), default="")
    route: Mapped[str] = mapped_column(String(16), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str] = mapped_column(Text, default="")
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True,
    )


Index("ix_llm_call_run", LLMCallLog.run_id, LLMCallLog.generated_at)


class SDKTrace(Base):
    """Wave 10 — persisted OpenAI Agents SDK exchange trace.

    When `USE_AGENTS_SDK=true` and the official `openai-agents` package +
    `OPENAI_API_KEY` are present, every memo run kicks off a parallel
    SDK exchange (real LLM-driven handoffs / tool calls). The trace is
    informational — the canonical `StockMemoOut` still comes from the
    legacy graph — but it's load-bearing for evals + debugging memos
    that look wrong + tool-gap discovery.

    One row per memo run (or per chat turn that routes through the SDK).
    `new_items` carries the raw SDK item stream (handoffs, tool calls,
    reasoning steps); the admin trace viewer joins this against
    `LLMCallLog` rows by `run_id` so reviewers can see SDK reasoning + the
    deterministic graph's calls in the same timeline.

    GC: rows older than 90 days are deleted by the same daily monitoring
    job that cleans `LLMCallLog`. Rows are bigger (`new_items` JSON), so
    don't keep them forever.
    """
    __tablename__ = "sdk_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ticker: Mapped[Optional[str]] = mapped_column(String(16), index=True, nullable=True)
    # Source surface: "memo" (run_stock_memo path) or "chat" (freeform Q&A).
    surface: Mapped[str] = mapped_column(String(16), default="memo", index=True)
    final_output: Mapped[str] = mapped_column(Text, default="")
    # Raw SDK item stream serialized to JSON. Format is provider-defined; we
    # don't constrain it here so model upgrades that add fields don't require
    # a migration.
    new_items: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[str] = mapped_column(Text, default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True,
    )


Index("ix_sdk_trace_run", SDKTrace.run_id, SDKTrace.generated_at)


class UILog(Base):
    """Append-only trace of UI activity + backend HTTP requests.

    Frontend posts route changes, API calls (with duration + status),
    button clicks, and uncaught errors here via `POST /api/admin/ui-log`.
    Backend middleware also writes a row per request so server-side
    traces and client-side actions sit in one queryable timeline.

    Schema is intentionally loose — `payload` carries arbitrary JSON
    so new event kinds can ship without migrations.
    """
    __tablename__ = "ui_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    source: Mapped[str] = mapped_column(String(16), default="frontend")  # frontend | backend
    kind: Mapped[str] = mapped_column(String(32), index=True)
    # Common dimensions surfaced as columns for cheap filtering; everything
    # else (request body, error stack, etc.) rides in `payload`.
    path: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    method: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


Index("ix_ui_log_ts_kind", UILog.ts, UILog.kind)


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
    period_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
    fiscal_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fiscal_quarter: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    statement: Mapped[str] = mapped_column(String(16), index=True)  # income | balance | cash
    line_item: Mapped[str] = mapped_column(String(64))
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    source: Mapped[str] = mapped_column(String(32), default="demo")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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
    filing_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
    period_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    sections: Mapped[dict] = mapped_column(JSON, default=dict)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    url: Mapped[str] = mapped_column(String(1024), default="")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MemoRunCheckpoint(Base):
    """Wave 6A — per-step checkpoint for resumable memo runs.

    Each `(run_id, step_name)` tuple stores the JSON-serializable result
    of that step, the timestamp, and an `expires_at` (default 24h).
    `run_stock_memo` populates `run_id` (Wave 1A); a `@checkpointed(step)`
    decorator wraps each major step in `graph.py` so a crash mid-memo
    doesn't force a full rerun — the next call with the same `run_id`
    skips already-completed steps and resumes from the next.

    The store is intentionally simple: read-modify-write on every step
    (no pickling, no compression). At memo scale (~10 steps × ~tens of
    KB each) this is fine; the daily GC keeps the table bounded.
    """
    __tablename__ = "memo_run_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    step_name: Mapped[str] = mapped_column(String(64))
    ticker: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("run_id", "step_name", name="uq_run_step"),
    )


class DCFModel(Base):
    """Wave 5A — versioned, persistent DCF with assumption lineage.

    Today every memo run rebuilds the DCF from scratch using the default
    assumption derivation. With this table we can ROLL the model forward
    each earnings period: year 1 forecast becomes "year 0 actual", the
    explicit forecast shifts, and an LLM-driven updater proposes
    adjustments to revenue growth / margins / capex / WACC / terminal
    growth based on what the period actually delivered. New version
    stored as `v(N+1)` referencing `v(N)`.

    `assumption_changes` captures the per-version delta + rationale so
    reviewers can audit assumption drift over time. `change_log` is a
    rolling audit trail across versions (similar to `MemoSnapshot.revision_log`).
    """
    __tablename__ = "dcf_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    trigger: Mapped[str] = mapped_column(String(32), default="initial")
    # Full DCFAssumptions payload (JSON); decoupled from any specific
    # `DCFAssumptions` shape so future fields don't require a migration.
    assumptions: Mapped[dict] = mapped_column(JSON, default=dict)
    # Full DCFResult payload at this version (so we don't have to rebuild
    # to render historical snapshots). Optional — initial seeds may skip.
    dcf_result: Mapped[dict] = mapped_column(JSON, default=dict)
    # Each entry: {"field": "revenue_growth[0]", "from": 0.10, "to": 0.12,
    # "rationale": "..."}.
    assumption_changes: Mapped[list] = mapped_column(JSON, default=list)
    change_log: Mapped[list] = mapped_column(JSON, default=list)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True,
    )


Index("ix_dcf_model_ticker_version", DCFModel.ticker, DCFModel.version, unique=True)


class MemoOutcome(Base):
    """Wave 4A — realized-outcome scoring for a memo at a forward horizon.

    One row per `(memo_snapshot_id, horizon_days)`. The daily evaluator
    computes forward returns at 30 / 90 / 180 / 365 days vs. SPY, lays
    them down here, and (for the longer horizons) writes a reflection
    entry into the company's long-term memory file. Used for:
      - admin track-record stats (rating accuracy / alpha by sector / etc.),
      - reflection feedback loops (sector agent reads its own past calls).
    """
    __tablename__ = "memo_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memo_snapshot_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("memo_snapshots.id"), index=True,
    )
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    rating_at_memo: Mapped[str] = mapped_column(String(32), default="")
    confidence_at_memo: Mapped[float] = mapped_column(Float, default=0.0)
    price_at_memo: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    horizon_days: Mapped[int] = mapped_column(Integer, index=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    forward_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    benchmark_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    alpha: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thesis_held: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")

    __table_args__ = (
        UniqueConstraint(
            "memo_snapshot_id", "horizon_days",
            name="uq_memo_outcome_snapshot_horizon",
        ),
    )


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
    fiscal_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fiscal_quarter: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    call_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
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
    HNSW index. Until then, retrieval falls back to BM25 over `text`.

    `source_type` ∈ {filing, transcript, memo, news}. `source_id` is
    the foreign-key into the originating table (FilingDoc.id,
    EarningsTranscript.id, MemoSnapshot.id) — kept loose (no FK) so a
    chunk survives source deletion (we'd rather have an orphan than a
    broken constraint). `meta` holds source-specific keys (section,
    accession, period, etc.).
    """
    __tablename__ = "doc_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[Optional[str]] = mapped_column(String(16), index=True, nullable=True)
    source_type: Mapped[str] = mapped_column(String(16), index=True)
    source_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    period_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    text: Mapped[str] = mapped_column(Text, default="")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    embedding_dim: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_doc_chunks_ticker_source", "ticker", "source_type"),
    )


class MemoPostmortem(Base):
    """Wave 10 — what a memo actually got right or wrong.

    Two cadences fire per memo: a 30-day "early read" (drift signal)
    and a 90-day "full postmortem" (calibration lesson). Each carries
    a per-agent attribution dict so per-specialist accuracy can be
    tracked over time. The `lesson` is the markdown body that gets
    appended to the company / sector / PM memory files.
    """
    __tablename__ = "memo_postmortems"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memo_snapshot_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("memo_snapshots.id"), index=True,
    )
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, index=True)
    verdict: Mapped[str] = mapped_column(String(32), default="")  # right / wrong / mixed / pending
    lesson: Mapped[str] = mapped_column(Text, default="")
    agent_attribution: Mapped[dict] = mapped_column(JSON, default=dict)
    realized_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    benchmark_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    regime_at_memo: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    written_to_memory: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "memo_snapshot_id", "horizon_days",
            name="uq_memo_postmortem_snapshot_horizon",
        ),
    )


class ThemeExposure(Base):
    """Wave 10 — per-company exposure to investable themes.

    Drives the natural-language screener and the cross-sector exposure
    peers in the comps agent. Refreshed monthly from a corpus pass over
    business descriptions + earnings transcripts. `evidence` carries
    short citations so the user can audit a score.
    """
    __tablename__ = "theme_exposure"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    theme: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100
    evidence: Mapped[list] = mapped_column(JSON, default=list)  # list[str]
    refreshed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "theme", name="uq_theme_exposure_ticker_theme"),
    )


class CatalystEvent(Base):
    """Wave 10 — known forward catalysts (earnings, FDA, conferences).

    Surfaced on the memo + chat. Sources: FMP earnings calendar (always
    populated), plus optional FDA calendar / conference scrapes (Phase F
    of the design review). `materiality` ∈ {low, medium, high} — set by
    the source or by an LLM-judged pass.
    """
    __tablename__ = "catalyst_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    event_type: Mapped[str] = mapped_column(String(32))  # earnings / fda / conference / investor_day / other
    event_date: Mapped[date] = mapped_column(Date, index=True)
    title: Mapped[str] = mapped_column(String(256), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    materiality: Mapped[str] = mapped_column(String(16), default="medium")
    source: Mapped[str] = mapped_column(String(32), default="fmp")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "ticker", "event_type", "event_date", "title",
            name="uq_catalyst_event_natkey",
        ),
        Index("ix_catalyst_events_ticker_date", "ticker", "event_date"),
    )


class MispricingAudit(Base):
    """Wave 10 — periodic LLM-judged audit of PM mispricing theses.

    Each row is one audit run. The latest row's `pattern_observation`
    is fed into the PM synthesis prompt as a self-improvement signal —
    the PM reads "your most common failure mode lately is X" and is
    expected to avoid it on the next memo. Closes the loop on PM
    self-improvement.
    """
    __tablename__ = "mispricing_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audited_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    n_memos: Mapped[int] = mapped_column(Integer, default=0)
    pattern_observation: Mapped[str] = mapped_column(Text, default="")
    per_memo_scores: Mapped[list] = mapped_column(JSON, default=list)
    aggregate_means: Mapped[dict] = mapped_column(JSON, default=dict)
    weak_memo_count: Mapped[int] = mapped_column(Integer, default=0)
