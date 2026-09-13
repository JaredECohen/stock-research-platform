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
