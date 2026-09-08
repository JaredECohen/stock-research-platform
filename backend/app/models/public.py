"""FEAT-002 — tables behind the logged-out surface.

`public_samples` is what the marketing site renders for the curated
tickers: prebuilt by the worker (`sample_build_loop`, slice S3), read by
`GET /api/public/samples/*`. Public routes never build anything — the
point of the login wall is that anonymous traffic cannot cause LLM or
provider spend, and a public route that could fall through to
generation would defeat it.

`analytics_events` is the product funnel (landing → sample → signup →
trial → first value → checkout). Event names and prop keys are
allowlisted in `auth/analytics.py`; rows older than 90 days are GC'd by
the billing loop so anonymous traffic cannot grow the table forever.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class PublicSample(Base):
    """One row per (ticker, kind). `kind` ∈ memo / dcf / comps /
    fundamentals / prices / screener_row / commentary. `etag` is a hash of
    `payload` for conditional GETs; `degraded` lists which inputs were
    unavailable when the row was built so the page can say so instead of
    rendering a zero as if it were data.
    """
    __tablename__ = "public_samples"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    kind: Mapped[str] = mapped_column(String(24), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    source_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)  # e.g. memo_snapshot:123
    etag: Mapped[str] = mapped_column(String(40), default="")
    built_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    built_by: Mapped[str] = mapped_column(String(32), default="")
    degraded: Mapped[list] = mapped_column(JSON, default=list)


class AnalyticsEvent(Base):
    """Funnel + abuse telemetry. `source` is `fe` (posted by the browser
    through `/api/public/events`) or `be` (emitted server-side). `anon_id`
    is a random client uuid, not a fingerprint. `props` carries only
    allowlisted keys with bounded string values.
    """
    __tablename__ = "analytics_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    event_name: Mapped[str] = mapped_column(String(48), index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    anon_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    plan: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(String(8), default="be")
    props: Mapped[dict] = mapped_column(JSON, default=dict)


Index("ix_analytics_events_name_ts", AnalyticsEvent.event_name, AnalyticsEvent.ts)
