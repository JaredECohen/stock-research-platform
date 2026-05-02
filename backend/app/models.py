"""SQLAlchemy ORM models for MarketMosaic.

We persist the master security universe, daily price snapshots, generated
agent memos and screener score history. Detailed financial statements are
loaded from JSON-backed providers rather than the DB to keep the schema lean
in demo mode; in live mode the same provider interface returns identical
shapes from external APIs and can be cached into ORM tables if desired.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
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
    fiscal_year_end: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
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
