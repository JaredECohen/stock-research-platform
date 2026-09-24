"""Durable daily market data; independent of expiring provider caches."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class DailyPrice(Base):
    __tablename__ = "daily_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    price_date: Mapped[date] = mapped_column(Date, index=True)
    source: Mapped[str] = mapped_column(String(32))
    provider_symbol: Mapped[str] = mapped_column(String(32))
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float)
    adjusted_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    close_basis: Mapped[str] = mapped_column(String(64), default="provider_reported")
    adjusted_close_basis: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_daily_price_identity", "ticker", "price_date", "source", "close_basis", unique=True),
    )


class MarketDataSync(Base):
    """Last explicit company backfill result, including incomplete coverage."""
    __tablename__ = "market_data_syncs"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    requested_start: Mapped[date] = mapped_column(Date)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")
    report: Mapped[dict] = mapped_column(JSON, default=dict)


class FinancialDataRepair(Base):
    """Durable, reviewable before/after plan for one fenced financial repair."""
    __tablename__ = "financial_data_repairs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    digest: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="planned")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    plan: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)


class FundamentalRefreshState(Base):
    """What each issuer has reported, and when we next ask FMP for it (FIX-005).

    Owner decision 2026-09-24: fundamentals load as often as they change. The
    trigger is the filing EDGAR lists, not a timer. The row is shared by the
    web process (first-contact requests, admin sync) and the worker (poller,
    nightly drain, regen pull-through), so it is the only place this state
    may live. Every writer of the scheduling fields does a compare-and-set on
    `row_version` and re-reads on conflict, so a filing observed while a
    drain is recording its result is never lost.
    """
    __tablename__ = "fundamental_refresh_state"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    # Newest period of report EDGAR lists (`reportDate`) per cadence.
    filed_quarter_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    filed_quarter_form: Mapped[str | None] = mapped_column(String(16), nullable=True)
    filed_quarter_accession: Mapped[str | None] = mapped_column(String(32), nullable=True)
    filed_quarter_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    quarter_observed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    filed_annual_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    filed_annual_form: Mapped[str | None] = mapped_column(String(16), nullable=True)
    filed_annual_accession: Mapped[str | None] = mapped_column(String(32), nullable=True)
    filed_annual_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    annual_observed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_amendment_accession: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Evidence only: a 25/25-NSE can delist one class of notes while the
    # issuer keeps filing, so these never end the reporting expectation.
    # Only the evidence-backed `KNOWN_REPORTING_ENDED` registry does.
    last_deregistration_form: Mapped[str | None] = mapped_column(String(16), nullable=True)
    last_deregistration_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    # Scheduling (see services/fundamental_refresh.py).
    status: Mapped[str] = mapped_column(String(16), default="idle")
    trigger: Mapped[str] = mapped_column(String(24), default="")
    due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    first_due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    requested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    stuck_since: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # The nightly loop fails once per stuck episode: set when it has done so.
    stuck_reported_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_calendar_check_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_result: Mapped[dict] = mapped_column(JSON, default=dict)
    row_version: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
