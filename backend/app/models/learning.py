"""W7 learning ledger — learned priors, their evidence, render audit, mode control.

Owner decision 9 (2026-09-24): "backfill and enable memory but it shouldn't
over index on memory, it must update priors and learn as evidence comes in".
So a lesson is a testable hypothesis, its credibility is computed at read
time from append-only evidence rows (`app.learning.ledger.posterior`), and
nothing here is ever rewritten by an LLM. Design:
`.claude/memory/proposals/design-w7-learning-final.md` §3, plus the accepted
critique in the integration plan (slice S18).

All four tables are new, so `create_all` makes them; no existing table is
altered. They are append-mostly and never dropped: an item's status changes
are appended to `status_history`, never overwritten silently.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class LearningItem(Base):
    """One learned prior.

    `kind="lesson"`: a hypothesis. Live postmortems write the grounded form
    `{condition, observable}` — a situation a later memo's own content can
    show (<= 160 chars) and a benchmark-relative direction over the horizon.
    Its verdicts are computed from realized alpha, never taken from a model.
    Lessons backfilled from pre-ledger postmortems carry no observable: they
    are narrative, stay "untested" and are never judged.

    `kind="observation"`: a dated filing fact with no posterior; superseded
    by the next filing of the same type, or expired.

    `scope_key` is an internal key (ticker, industry-group code, sector
    slug). It is rendered through the public labels, never shown raw.
    """
    __tablename__ = "learning_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)            # lesson | observation
    scope_type: Mapped[str] = mapped_column(String(16))                   # company | industry_group | sector
    scope_key: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)                               # rendered text (lesson <=240, obs <=320)
    detail: Mapped[str] = mapped_column(Text, default="")                 # full narrative; audit only, never injected
    condition: Mapped[str | None] = mapped_column(String(200), nullable=True)
    observable: Mapped[str | None] = mapped_column(String(16), nullable=True)  # outperform | underperform
    # postmortem | postmortem_sector | postmortem_backfill | filing_delta | filing_pattern
    origin_kind: Mapped[str] = mapped_column(String(24))
    origin_ref: Mapped[str] = mapped_column(String(64))                   # postmortem id | accession
    origin_ticker: Mapped[str] = mapped_column(String(16), index=True)
    # Loose (no FK), like doc_chunks.source_id: an item outlives nothing it
    # points at, and the integrity check retires it if the origin snapshot
    # stops being eligible.
    origin_snapshot_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    source_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # information date (look-ahead guard)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)  # active|suppressed|retired|superseded
    status_history: Mapped[list] = mapped_column(JSON, default=list)      # [{at, from, to, reason, actor}]
    supersede_key: Mapped[str | None] = mapped_column(String(96), nullable=True, index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("origin_kind", "origin_ref", "scope_type", "scope_key", name="uq_learning_item_origin"),
        Index("ix_learning_items_scope", "scope_type", "scope_key", "status"),
    )


class LearningEvidence(Base):
    """Append-only judgment of one lesson against one later, eligible outcome.

    `independence_key` is `"{scope_key}:{horizon}:{bucket}"`: one row per
    lesson per scope window, so re-issues of one company inside a window
    (MSFT is at v108) and three peers of one group in the same window each
    count once (n_eff <= 1 per window).
    """
    __tablename__ = "learning_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(Integer, ForeignKey("learning_items.id"), index=True)
    verdict: Mapped[str] = mapped_column(String(12))                      # held | failed | mixed | irrelevant
    applies: Mapped[str] = mapped_column(String(8), default="")           # the judge's answer: yes | no | unclear
    postmortem_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    memo_snapshot_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16))
    horizon_days: Mapped[int] = mapped_column(Integer)
    independence_key: Mapped[str] = mapped_column(String(96))
    alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
    rationale: Mapped[str] = mapped_column(Text, default="")              # <= 500 chars
    observed_at: Mapped[datetime] = mapped_column(DateTime)               # outcome.evaluated_at (decay clock)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("item_id", "independence_key", name="uq_learning_evidence_window"),)


class LearningRender(Base):
    """Audit: one row per consumer render in shadow or inject mode.

    Written BEFORE a block is injected; a prior that cannot be audited is
    never shown. `error_type` names a render that failed (type only), and
    the promotion gate G3 requires none in the soak window.
    """
    __tablename__ = "learning_renders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    consumer: Mapped[str] = mapped_column(String(24))                     # pm_memo | sector | industry_group | critic
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    mode: Mapped[str] = mapped_column(String(8))                          # shadow | inject
    memo_snapshot_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    chars: Mapped[int] = mapped_column(Integer, default=0)
    items: Mapped[list] = mapped_column(JSON, default=list)               # [{ref, item_id, kind, scope_type, stance, rank, chars}]
    dropped: Mapped[list] = mapped_column(JSON, default=list)             # [{ref, reason}]
    considered: Mapped[list | None] = mapped_column(JSON, nullable=True)  # PM priors_considered (validated)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class LearningControlEvent(Base):
    """Append-only mode switch. The latest MODE row (reason other than
    `ledger_epoch`) is the DB mode; absence means "shadow".

    The one `ledger_epoch` row marks when the live ledger started writing,
    so the historical backfill never re-learns a postmortem the live path
    already saw (and deliberately declined).
    """
    __tablename__ = "learning_control_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(8))                          # off | shadow | inject
    actor: Mapped[str] = mapped_column(String(16))                        # admin | auto | system
    reason: Mapped[str] = mapped_column(Text, default="")
    gates: Mapped[dict] = mapped_column(JSON, default=dict)
    forced: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
