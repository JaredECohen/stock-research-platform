"""Background work + ops — theme exposure, catalyst calendar, the durable
regen queue and the cross-process cron liveness record."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


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


class RegenJob(Base):
    """Theme 5 — durable queue for background memo regeneration.

    One row per regen request. Replaces the in-memory `_REGEN_JOBS` /
    `_REGEN_FAILURES` dicts that lived in `routes_stocks` — those
    vanished with the process on OOM kills / deploys, which made
    failures invisible (the status endpoint just stopped reporting
    in_progress with no explanation). A DB row survives the process,
    so a killed regen shows up as a `failed` job with an explicit
    error instead of silently disappearing.

    Lifecycle: queued → running → succeeded | failed. A single worker
    thread (`services/regen_worker.py`) claims queued jobs oldest-first.
    On worker startup, orphaned `running` jobs (the process died
    mid-run) are requeued once with the same `run_id` — the Wave 8A
    checkpoint store then skips already-completed steps — and marked
    failed on the second orphaning so a crash-looping ticker can't
    wedge the queue.

    `run_id` joins against `MemoRunCheckpoint` (per-step progress) and
    `LLMCallLog` (per-call cost/failure) so one id links the job to its
    full telemetry trail. `progress` carries the worker's own coarse
    waypoints (claimed / calling graph / persisted / failed) for the
    gaps those tables can't see.
    """
    __tablename__ = "regen_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    scenario: Mapped[str] = mapped_column(String(32), default="soft_landing")
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    enqueued_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Version of the MemoSnapshot the job produced (success only).
    memo_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str] = mapped_column(String(64), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    traceback_tail: Mapped[str] = mapped_column(Text, default="")
    # Coarse waypoint trace: [{"step": ..., "at": iso-ts}, ...], capped.
    progress: Mapped[list] = mapped_column(JSON, default=list)
    # FEAT-002: who asked, and which `usage_events` reservation the worker
    # commits on success / releases on failure. Both nullable so rows from
    # sweeps and pre-accounts deployments read as "system" — and so
    # `reconcile_missing_columns` can add them to a live table.
    requested_by_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    usage_event_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


Index("ix_regen_jobs_ticker_status", RegenJob.ticker, RegenJob.status)


class CronLoopRun(Base):
    """Last-run record per monitoring loop, readable across processes.

    Added 2026-08-14, after the worker split (#44) silently broke
    `/api/admin/cron-health`. `monitoring._LAST_RUNS` is a module-level
    dict, so it only ever described the process that happened to be
    serving the request. Once the loops moved to `marketmosaic-worker`,
    the endpoint — running on the *web* service — began reporting
    `{"loops": [], "stale_count": 0}`.

    That failure mode is worse than an outage: `stale_count: 0` reads as
    "every loop is healthy" when it actually means "I cannot see any
    loops". The endpoint exists specifically to surface silent cron
    failures, so a version of it that silently reports success is exactly
    backwards.

    One row per loop, upserted on each run. Deliberately not append-only:
    this is a liveness signal, not an audit log, and 15 loops on intervals
    as short as 30 minutes would grow unbounded for no benefit.
    `CacheCostLog` already covers the append-only telemetry case.
    """
    __tablename__ = "cron_loop_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    loop_name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    last_run_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[str] = mapped_column(Text, default="")
    # Which process reported it. Distinguishes the worker's loops from
    # anything still running in-process on the web service, which is the
    # first question to ask when cron-health looks wrong.
    reported_by: Mapped[str] = mapped_column(String(32), default="")
    # In-flight activity is not a completed result. Nullable additions are
    # reconciled safely onto the deployed table without changing old rows.
    progress_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    progress_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
