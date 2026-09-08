"""Memo lifecycle — snapshots, run checkpoints, outcomes, postmortems and
the mispricing audit."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


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
    # Wave 10 — macro regime at memo creation, copied from
    # `MemoSnapshot.memo_json["macro_regime_at_memo"]` at evaluate
    # time. Lets calibration's regime-conditional dashboards bucket
    # outcomes without joining through the snapshot blob.
    regime_at_memo: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)

    __table_args__ = (
        UniqueConstraint(
            "memo_snapshot_id", "horizon_days",
            name="uq_memo_outcome_snapshot_horizon",
        ),
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
