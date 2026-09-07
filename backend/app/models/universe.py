"""Security universe — companies, legacy memos, screener scores + metrics."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


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
    # When True, new EDGAR filings / earnings transcripts trigger an
    # automatic memo regeneration (`full_reanalysis(ticker)`) for this
    # ticker. When False, the polling jobs still ingest + persist the
    # raw data, but memo regen waits for a user request. Seeded `True`
    # only for the top-10-by-market-cap list defined in sp500.json
    # (`_top_10_by_market_cap_*`). Orthogonal to `universe_tier` —
    # a ticker can be `auto_analysis` (in the screener) without being
    # `auto_update_memo` (no auto-regen). Combined with the recency
    # window in `update_orchestrator.should_auto_regen`, this controls
    # the marginal LLM spend of the SP500 expansion.
    auto_update_memo: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default="0",
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
