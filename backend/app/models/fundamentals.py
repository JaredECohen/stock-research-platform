"""FEAT-001 — the Fundamentals Explorer's one table: cached chart commentary.

`chart_commentaries` is the cross-process cache for `POST
/api/fundamentals/commentary`. A hit is served without an LLM call and
without a charge, so the cache is a cost control as much as a speed-up;
that is why it is a table and not a module dict — the web replica that
generated a commentary is not necessarily the one asked for it again.

Keying: `cache_key` (unique) is what the commentary service computes over
everything that changes the answer — the series `fingerprint`, the
selection, `CATALOG_VERSION` and the memo versions quoted — so a memo
regeneration or a catalog formula change never serves stale prose.
`fingerprint` is indexed separately so a chart can ask "is there any
commentary for exactly this data" without recomputing the memo part.

Every column is nullable or defaulted: the table is created by
`create_all` on a fresh database, and any column added later reaches a
live database through `database.reconcile_missing_columns`. Retention is
90 days via the daily `monitoring/llm_log_gc` loop, the same policy as
`llm_call_logs` (the prose is derived from data that itself goes stale).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class ChartCommentary(Base):
    __tablename__ = "chart_commentaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # sha256 hex over the full cache input (see module docstring).
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # The series response's fingerprint the commentary was written against.
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    # Who asked (FEAT-002 `users.id`); NULL for the anonymous default and
    # for rows written before accounts existed.
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # The selection, echoed so the row is self-describing in the database.
    tickers: Mapped[list] = mapped_column(JSON, default=list)
    metrics: Mapped[list] = mapped_column(JSON, default=list)
    years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    catalog_version: Mapped[str] = mapped_column(String(16), default="")
    # {"AAPL": {"version": 12, "generated_at": "..."}} — the memo versions
    # quoted in `memo_view`, so a regenerated memo invalidates the row.
    memo_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    # The `CommentaryOut` body as served.
    output: Mapped[dict] = mapped_column(JSON, default=dict)
    provider: Mapped[str] = mapped_column(String(16), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    # A degraded row is the deterministic observed-only body; it is stored
    # so the reason is auditable, and it was never charged.
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    degraded_reason: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


# "Any commentary for this exact chart, newest first" — the lookup a
# cache-hit check and the GC sweep both make.
Index("ix_chart_commentaries_fingerprint_created", ChartCommentary.fingerprint, ChartCommentary.created_at)
