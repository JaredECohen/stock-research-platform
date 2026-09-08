"""Versioned DCF model persistence."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


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
    parent_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
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
