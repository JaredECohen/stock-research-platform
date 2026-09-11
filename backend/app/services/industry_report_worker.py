"""FEAT-003 slice 4 — the durable queue behind the weekly Industry Analysis.

Structurally this is ``regen_worker`` with the domain swapped, and that is
deliberate: the memo queue's shape (enqueue-coalesce, an atomic conditional
UPDATE claim, restart recovery, an expiry for a backlog nobody is waiting
for) was arrived at through production failures, and a second queue that
invents its own shape would have to learn them again.

    weekly loop  →  enqueue_period() inserts one `group_report` row per
                    active industry group plus one `cross_snapshot` row
                    for the period, coalescing against active jobs
    drainer      →  claims the next eligible job (queued→running, atomic
                    conditional UPDATE), computes stats → writes the
                    edition → validates it → saves it, or records the
                    failure with a backoff
    cron-health  →  a `record_run("industry_report_worker")` heartbeat on
                    every tick, so the drainer is visibly alive between
                    Sundays rather than "stale" six days out of seven

Three properties this file is responsible for, each of which has a test:

* **One analytics context per period.** The benchmark cohort is frozen
  when ``load_context`` returns, so draining each group with its own
  context would make the groups of one period incomparable — and would
  re-read the whole universe's prices once per group. The drainer keeps
  one context alive for as long as consecutive jobs share a period.
* **A failed refresh never blanks the page.** Nothing writes a report row
  unless it validated; the prior edition keeps ``is_latest_good`` and the
  read API composes staleness from this table's ``last_attempt``. The job
  kind string is therefore exactly ``group_report`` — ``report_store``
  filters on it.
* **A week never ends blank because of the model.** The final attempt runs
  the deterministic writer, which needs no LLM, and the edition is
  published labelled ``degraded`` rather than not at all.

Single-replica by design, like the memo queue. The claim is still an
atomic conditional UPDATE, so a second replica would degrade safely (each
job runs once) rather than corrupt state.
"""
from __future__ import annotations

import logging
import sys
import threading
import traceback
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update

from ..agents import industry_report_validator as validator
from ..agents import industry_report_writer as writer
from ..agents.industry_analysts import get_industry_analyst
from ..agents.llm import llm_call_context
from ..agents.log_safety import redact, safe_exc
from ..config import settings
from ..database import SessionLocal
from ..models import IndustryReport, IndustryReportJob
from ..monitoring import record_run
from . import gics_registry, industry_analytics, industry_report_store, industry_snapshot

log = logging.getLogger(__name__)

# Job kinds. `group_report` is a frozen string: `industry_report_store`
# filters `last_attempt()` / `freshness()` on it, so writing anything else
# would make a failed refresh report "no attempt" next to a stale edition.
KIND_GROUP = "group_report"
KIND_CROSS = "cross_snapshot"
KINDS: tuple[str, ...] = (KIND_GROUP, KIND_CROSS)

# Claim order is (priority, id): the cross-industry snapshot is computed
# from the period's stats rows, so it drains after the group reports.
PRIORITY: dict[str, int] = {KIND_GROUP: 100, KIND_CROSS: 200}

_ACTIVE_STATUSES = ("queued", "running")
_FINISHED_STATUSES = ("succeeded", "failed")

# Retry backoff: 15 minutes × attempts so far (15, 30, …).
BACKOFF_MINUTES = 15
# When a cross_snapshot job is reached while its period's group jobs are
# still in flight, it is pushed out by this much rather than run early.
CROSS_DEFER_MINUTES = 5
# A queued job older than this missed its week entirely; the next cron
# enqueues the current period instead of publishing a stale one.
QUEUE_MAX_AGE_DAYS = 14

POLL_SECONDS = 5.0
HEARTBEAT_SECONDS = 300
HEARTBEAT_NAME = "industry_report_worker"

# The as-of is the prior Friday's close; 21:00 UTC is after every US
# session close, including the DST edge, and is the timestamp the rest of
# the feature's fixtures use.
AS_OF_HOUR_UTC = 21

_MAX_PROGRESS_ENTRIES = 50
_MAX_ERROR_CHARS = 500
_MAX_VALIDATION_PROBLEMS = 10

# Every report section that reads the cross-industry snapshot payload. When
# the period's own snapshot does not exist yet, ALL THREE quote a prior
# period, so all three are labelled — see `_snapshot_for`.
_SNAPSHOT_FED_SECTIONS: tuple[str, ...] = (
    "cross_industry:snapshot", "companies:events", "outlook:macro_regime",
)

# Serializes enqueue's check-then-insert within the process, exactly as the
# memo queue does: an admin regenerate racing the Sunday cron must not
# insert two jobs for the same group and period.
_ENQUEUE_LOCK = threading.Lock()


def _utcnow() -> datetime:
    """Clock seam — tests pin it instead of freezing time."""
    return datetime.utcnow()


class ReportRejected(RuntimeError):
    """The validator refused the edition. Retried like any other failure;
    the message names the problems (and counts any it had to drop)."""


# ---------------------------------------------------------------------------
# Period arithmetic — the one place the week's identity is decided
# ---------------------------------------------------------------------------


def _as_of_weekday() -> int:
    """Monday=0 … Sunday=6; Friday (4) by default."""
    return max(0, min(6, int(settings.industry_reports_as_of_weekday)))


def close_day_before(now: datetime) -> date:
    """The most recent as-of weekday *strictly before* ``now``'s date.

    Strictly before on purpose: a run that happens on the as-of weekday
    itself would be reading a session that has not closed, so it reaches
    back a full week rather than publishing a half-day.
    """
    day = now.date()
    delta = (day.weekday() - _as_of_weekday()) % 7
    return day - timedelta(days=delta or 7)


def period_for(now: datetime | None = None) -> tuple[str, datetime]:
    """``(period_key, as_of)`` for a run at ``now`` — the prior Friday's
    close and the ISO week that Friday belongs to."""
    day = close_day_before(now or _utcnow())
    as_of = datetime(day.year, day.month, day.day, AS_OF_HOUR_UTC)
    return industry_analytics.period_key_for(day), as_of


def as_of_for_period(period_key: str) -> datetime:
    """The as-of a ``period_key`` names. The jobs table stores the period,
    not the timestamp, so this is what makes a claimed job reproduce the
    exact inputs the loop intended — and it round-trips:
    ``period_key_for(as_of_for_period(k)) == k``."""
    key = str(period_key or "").strip().upper()
    try:
        year_s, week_s = key.split("-W", 1)
        day = date.fromisocalendar(int(year_s), int(week_s), _as_of_weekday() + 1)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unparseable period_key {period_key!r}: {exc}") from exc
    return datetime(day.year, day.month, day.day, AS_OF_HOUR_UTC)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _ensure_table(db) -> None:
    # Same belt-and-braces as `regen_worker`: `init_db` creates this table,
    # but a queue read that raises "no such table" on a half-initialised
    # database is a worse failure than an idempotent no-op DDL check.
    # `__table__` is typed as the generic `FromClause`, which has no
    # `.create` — the runtime object is always a `Table`.
    IndustryReportJob.__table__.create(bind=db.get_bind(), checkfirst=True)  # type: ignore[attr-defined]


def _job_dict(job: IndustryReportJob) -> dict[str, Any]:
    """Detached snapshot of a job row, safe to use after the session closes."""
    return {
        "id": job.id,
        "kind": job.kind,
        "taxonomy_version_id": job.taxonomy_version_id,
        "code": job.industry_group_code,
        "period_key": job.period_key,
        "run_id": job.run_id,
        "status": job.status,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "priority": job.priority,
        "not_before": job.not_before.isoformat() if job.not_before else None,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "heartbeat_at": job.heartbeat_at.isoformat() if job.heartbeat_at else None,
        "source": job.source,
        "force": bool(job.force),
        "report_id": job.report_id,
        "error_type": job.error_type or "",
        "error_message": job.error_message or "",
        "traceback_tail": job.traceback_tail or "",
        "progress": list(job.progress or []),
    }


def _append_progress(job_id: int, step: str, **extra: Any) -> None:
    """Append a waypoint to the job's trace, refreshing the heartbeat so a
    hung job is distinguishable from a killed one. Never raises."""
    try:
        with SessionLocal() as db:
            job = db.get(IndustryReportJob, job_id)
            if job is None:
                return
            steps = list(job.progress or [])
            steps.append({"step": step, "at": _utcnow().isoformat(), **extra})
            # Reassign (don't mutate) so the JSON column change is tracked.
            job.progress = steps[-_MAX_PROGRESS_ENTRIES:]
            job.heartbeat_at = _utcnow()
            db.commit()
    except Exception:  # pragma: no cover — telemetry must not break the job
        log.debug("progress append failed for industry job %d", job_id, exc_info=True)


# ---------------------------------------------------------------------------
# Queue API
# ---------------------------------------------------------------------------


def enqueue(
    code: str | None, period_key: str, *, kind: str = KIND_GROUP,
    source: str = "weekly_cron", force: bool = False,
    version: gics_registry.VersionInfo | str | int | None = None,
    max_attempts: int | None = None, run_id: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Queue one report job. Returns ``(job, created)``.

    Coalesces on ``(taxonomy version, code, period_key, kind)``: when an
    active job already exists it is returned with ``created=False``. That
    is what keeps an admin regenerate from running a second report
    concurrently with the Sunday cron's — and it holds across restarts,
    because the queue is a table rather than a dict.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind!r}; allowed: {', '.join(KINDS)}")
    info = gics_registry.resolve_version(version)
    group_code = str(code).strip() if code else None
    if kind == KIND_GROUP:
        if not group_code:
            raise ValueError("a group_report job needs an industry group code")
        gics_registry.group(group_code, version=info)  # UnknownNode before any insert
    else:
        group_code = None
    with _ENQUEUE_LOCK, SessionLocal() as db:
        _ensure_table(db)
        existing = db.execute(
            select(IndustryReportJob).where(
                IndustryReportJob.taxonomy_version_id == info.id,
                IndustryReportJob.industry_group_code.is_(None) if group_code is None
                else IndustryReportJob.industry_group_code == group_code,
                IndustryReportJob.period_key == period_key,
                IndustryReportJob.kind == kind,
                IndustryReportJob.status.in_(_ACTIVE_STATUSES),
            ).order_by(IndustryReportJob.id.desc())
        ).scalars().first()
        if existing is not None:
            log.info("industry job %d coalesced %s %s/%s (source=%s)",
                     existing.id, kind, group_code or "-", period_key, source)
            return _job_dict(existing), False
        now = _utcnow()
        job = IndustryReportJob(
            kind=kind,
            taxonomy_version_id=info.id,
            industry_group_code=group_code,
            period_key=period_key,
            run_id=run_id or str(uuid.uuid4()),
            status="queued",
            attempts=0,
            max_attempts=int(max_attempts or settings.industry_report_max_attempts),
            priority=PRIORITY[kind],
            enqueued_at=now,
            source=source,
            force=bool(force),
            progress=[{"step": "enqueued", "at": now.isoformat(), "source": source}],
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        log.info("industry job %d enqueued %s %s/%s (source=%s, force=%s)",
                 job.id, kind, group_code or "-", period_key, source, force)
        return _job_dict(job), True


def _editions_for_period(codes: list[str], period_key: str,
                         info: gics_registry.VersionInfo) -> set[str]:
    """The groups that already HAVE an edition for ``period_key``.

    One chunked query for the whole period, never one per group.

    Deliberately not ``report_store.latest_good_many``: that only ever
    returns rows flagged ``is_latest_good``, and ``save_report`` never
    flags an edition that landed as ``pending_review``. With
    ``INDUSTRY_REPORTS_REQUIRE_REVIEW`` on, every Sunday would therefore
    find no published edition for last Sunday's week, re-enqueue all of
    it, and pay the model again — forever, and invisibly, because the
    editions it is reproducing are sitting in the review queue. A week
    that has already been *generated* is a week not to generate again;
    whether a human has released it is a different question.
    """
    out: set[str] = set()
    with SessionLocal() as db:
        for i in range(0, len(codes), 200):
            rows = db.execute(
                select(IndustryReport.industry_group_code).where(
                    IndustryReport.taxonomy_version_id == info.id,
                    IndustryReport.industry_group_code.in_(codes[i:i + 200]),
                    IndustryReport.period_key == period_key,
                    IndustryReport.status.in_(
                        (industry_report_store.STATUS_SUCCEEDED,
                         industry_report_store.STATUS_PENDING_REVIEW)),
                )
            ).all()
            out.update(str(code) for (code,) in rows)
    return out


def enqueue_period(
    period_key: str, codes: list[str] | tuple[str, ...] | None = None, *,
    source: str = "weekly_cron", force: bool = False,
    version: gics_registry.VersionInfo | str | int | None = None,
    max_jobs: int | None = None, include_cross_snapshot: bool = True,
) -> dict[str, Any]:
    """Enqueue the whole period: one group report per active group (or per
    ``codes``) plus the cross-industry snapshot.

    A group that already has a published edition for ``period_key`` is
    skipped unless ``force`` — re-running a week that is already on the
    site spends LLM budget to reproduce it. Everything the call decided
    *not* to do is counted in the result: coalesced, skipped_published,
    unknown codes, and any group beyond ``INDUSTRY_REPORTS_MAX_JOBS_PER_RUN``.
    """
    info = gics_registry.resolve_version(version)
    if codes is None:
        wanted = [node.code for node in gics_registry.industry_groups(version=info)]
        unknown: list[str] = []
    else:
        known = {node.code for node in gics_registry.industry_groups(version=info)}
        wanted = [str(c).strip() for c in codes if str(c).strip() in known]
        unknown = sorted({str(c).strip() for c in codes if str(c).strip() not in known})

    already = set() if force or not wanted else _editions_for_period(wanted, period_key, info)
    skipped_published = [code for code in wanted if code in already]
    todo = [code for code in wanted if code not in already]

    cap = int(settings.industry_reports_max_jobs_per_run if max_jobs is None else max_jobs)
    over_budget = todo[cap:] if cap >= 0 else []
    todo = todo[:cap] if cap >= 0 else todo

    enqueued: list[dict[str, Any]] = []
    coalesced: list[dict[str, Any]] = []
    for code in todo:
        job, created = enqueue(code, period_key, kind=KIND_GROUP, source=source,
                               force=force, version=info)
        (enqueued if created else coalesced).append(job)

    cross: dict[str, Any] | None = None
    cross_created = False
    if include_cross_snapshot and (enqueued or coalesced or force):
        # Only worth a job when at least one group's stats will be written
        # this run: the snapshot is computed FROM those rows.
        cross, cross_created = enqueue(None, period_key, kind=KIND_CROSS, source=source,
                                       force=force, version=info)

    result = {
        "period_key": period_key,
        "as_of": as_of_for_period(period_key).isoformat(),
        "taxonomy_version": info.version_key,
        "taxonomy_version_id": info.id,
        "n_groups": len(wanted),
        "enqueued": len(enqueued),
        "coalesced": len(coalesced),
        "skipped_published": len(skipped_published),
        "skipped_published_codes": sorted(skipped_published),
        "unknown_codes": unknown,
        "over_budget": len(over_budget),
        "over_budget_codes": over_budget,
        "max_jobs_per_run": cap,
        "cross_snapshot": (
            {"job_id": cross["id"], "created": cross_created} if cross is not None
            else {"job_id": None, "created": False,
                  "reason": "no group job will write stats for this period"}
        ),
        "job_ids": [j["id"] for j in enqueued],
    }
    log.info("industry period %s: enqueued=%d coalesced=%d skipped_published=%d over_budget=%d",
             period_key, result["enqueued"], result["coalesced"],
             result["skipped_published"], result["over_budget"])
    return result


def recent_jobs(
    status: str | None = None, limit: int = 50, *, code: str | None = None,
    period_key: str | None = None,
) -> list[dict[str, Any]]:
    """Newest-first job rows for the admin telemetry endpoint."""
    with SessionLocal() as db:
        _ensure_table(db)
        q = select(IndustryReportJob).order_by(IndustryReportJob.id.desc()).limit(max(1, int(limit)))
        if status:
            q = q.where(IndustryReportJob.status == status)
        if code:
            q = q.where(IndustryReportJob.industry_group_code == str(code))
        if period_key:
            q = q.where(IndustryReportJob.period_key == str(period_key))
        return [_job_dict(j) for j in db.execute(q).scalars().all()]


def queue_counts(now: datetime | None = None) -> dict[str, int]:
    """Queue depth and today's outcomes — what the heartbeat note carries.

    `success=True written=0` is exactly the shape of report that hid a
    previous outage, so the heartbeat states counts rather than liveness
    alone.
    """
    at = now or _utcnow()
    midnight = datetime(at.year, at.month, at.day)
    out = {"queued": 0, "running": 0, "succeeded_today": 0, "failed_today": 0, "deferred": 0}
    with SessionLocal() as db:
        _ensure_table(db)
        for status, n in db.execute(
            select(IndustryReportJob.status, func.count(IndustryReportJob.id))
            .where(IndustryReportJob.status.in_(_ACTIVE_STATUSES))
            .group_by(IndustryReportJob.status)
        ).all():
            out[str(status)] = int(n)
        out["deferred"] = int(db.execute(
            select(func.count(IndustryReportJob.id)).where(
                IndustryReportJob.status == "queued",
                IndustryReportJob.not_before.is_not(None),
                IndustryReportJob.not_before > at,
            )
        ).scalar() or 0)
        for status, n in db.execute(
            select(IndustryReportJob.status, func.count(IndustryReportJob.id))
            .where(
                IndustryReportJob.status.in_(_FINISHED_STATUSES),
                IndustryReportJob.finished_at.is_not(None),
                IndustryReportJob.finished_at >= midnight,
            ).group_by(IndustryReportJob.status)
        ).all():
            out[f"{status}_today"] = int(n)
    return out


def heartbeat(now: datetime | None = None) -> str:
    """Persist the drainer's liveness + queue depth for cron-health.

    The drainer is scheduled by nothing — it is a thread that mostly sits
    idle — so without this row the only evidence it exists arrives on
    Sundays, and a drainer that died on Monday would look fine until the
    next week's reports silently failed to appear. ``success=False`` when
    a job failed today, because a failed report is exactly what an
    operator should be told about without opening the database.
    """
    counts = queue_counts(now)
    note = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    record_run(HEARTBEAT_NAME, success=counts["failed_today"] == 0, note=note)
    return note


# ---------------------------------------------------------------------------
# Claim / recovery
# ---------------------------------------------------------------------------


def _claim(db, job_id: int, attempts: int) -> bool:
    """The atomic half of the claim: queued→running, conditional on the row
    still being queued. A losing racer updates 0 rows and moves on."""
    res = db.execute(
        update(IndustryReportJob)
        .where(IndustryReportJob.id == job_id, IndustryReportJob.status == "queued")
        .values(status="running", started_at=_utcnow(), heartbeat_at=_utcnow(),
                attempts=attempts + 1)
    )
    db.commit()
    return bool(res.rowcount)


def _group_jobs_in_flight(db, version_id: int, period_key: str, *, exclude_id: int) -> int:
    return int(db.execute(
        select(func.count(IndustryReportJob.id)).where(
            IndustryReportJob.taxonomy_version_id == version_id,
            IndustryReportJob.period_key == period_key,
            IndustryReportJob.kind == KIND_GROUP,
            IndustryReportJob.status.in_(_ACTIVE_STATUSES),
            IndustryReportJob.id != exclude_id,
        )
    ).scalar() or 0)


def claim_next_job(now: datetime | None = None) -> int | None:
    """Claim the next eligible job, in (priority, id) order. Returns its id.

    Eligible means queued and past its ``not_before``. A cross_snapshot
    job whose period still has group jobs in flight is pushed out by
    ``CROSS_DEFER_MINUTES`` instead of being claimed — it is computed from
    the stats rows those jobs write, and a snapshot built halfway through
    the period would silently describe a partial week.
    """
    at = now or _utcnow()
    with SessionLocal() as db:
        _ensure_table(db)
        rows = db.execute(
            select(IndustryReportJob.id, IndustryReportJob.attempts, IndustryReportJob.kind,
                   IndustryReportJob.taxonomy_version_id, IndustryReportJob.period_key)
            .where(
                IndustryReportJob.status == "queued",
                (IndustryReportJob.not_before.is_(None)) | (IndustryReportJob.not_before <= at),
            )
            .order_by(IndustryReportJob.priority, IndustryReportJob.id)
        ).all()
        for job_id, attempts, kind, version_id, period_key in rows:
            if kind == KIND_CROSS and _group_jobs_in_flight(db, version_id, period_key, exclude_id=job_id):
                db.execute(
                    update(IndustryReportJob)
                    .where(IndustryReportJob.id == job_id, IndustryReportJob.status == "queued")
                    .values(not_before=at + timedelta(minutes=CROSS_DEFER_MINUTES))
                )
                db.commit()
                continue
            if _claim(db, job_id, int(attempts or 0)):
                return int(job_id)
        return None


def recover_orphans(now: datetime | None = None) -> dict[str, int]:
    """Startup pass over jobs the previous process left behind.

    - ``running`` rows mean the process died mid-report (OOM kill bypasses
      every except-block). Requeued with the retry backoff while attempts
      remain, then failed ``WorkerRestart`` so a group that reliably kills
      the process cannot crash-loop the service.
    - ``queued`` rows older than ``QUEUE_MAX_AGE_DAYS`` missed their week.
      Publishing them now would put a two-week-old "this week" on the
      site, so they are expired ``QueueExpired`` and the next cron
      enqueues the current period.
    """
    at = now or _utcnow()
    requeued = failed = expired = 0
    with SessionLocal() as db:
        _ensure_table(db)
        for job in db.execute(
            select(IndustryReportJob).where(IndustryReportJob.status == "running")
        ).scalars().all():
            if job.attempts < job.max_attempts:
                job.status = "queued"
                job.started_at = None
                job.not_before = at + timedelta(minutes=BACKOFF_MINUTES * max(1, job.attempts))
                job.progress = list(job.progress or []) + [
                    {"step": "requeued_after_process_restart", "at": at.isoformat()}
                ]
                requeued += 1
            else:
                job.status = "failed"
                job.finished_at = at
                job.error_type = "WorkerRestart"
                job.error_message = (
                    "Process died mid-report on the final attempt (likely OOM kill or "
                    "deploy). Not retrying automatically — the previous edition stays "
                    "the latest good one and is flagged stale."
                )
                failed += 1
        cutoff = at - timedelta(days=QUEUE_MAX_AGE_DAYS)
        for job in db.execute(
            select(IndustryReportJob).where(
                IndustryReportJob.status == "queued", IndustryReportJob.enqueued_at < cutoff,
            )
        ).scalars().all():
            job.status = "failed"
            job.finished_at = at
            job.error_type = "QueueExpired"
            job.error_message = (
                f"Queued for over {QUEUE_MAX_AGE_DAYS} days without being drained; "
                "expired instead of publishing a stale week."
            )
            expired += 1
        db.commit()
    if requeued or failed or expired:
        log.warning("industry report recovery: %d requeued, %d failed (repeat orphan), %d expired",
                    requeued, failed, expired)
    return {"requeued": requeued, "failed": failed, "expired": expired}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ContextBook:
    """Holds the ONE ``AnalyticsContext`` a period is drained with.

    Not a cache in the usual sense — it is a correctness device. The
    benchmark cohort is frozen when ``load_context`` returns, so two
    groups computed from two contexts are measured against two different
    cohorts and stop being comparable; and each context re-reads the whole
    universe's cached prices. One context per (period, taxonomy version),
    dropped as soon as the drainer moves to another period, so memory
    stays bounded on the worker.
    """

    def __init__(self) -> None:
        self._key: tuple[str, int] | None = None
        self._ctx: industry_analytics.AnalyticsContext | None = None
        self.loads = 0

    def context_for(
        self, period_key: str, as_of: datetime, info: gics_registry.VersionInfo,
    ) -> industry_analytics.AnalyticsContext:
        key = (period_key, info.id)
        if self._key != key or self._ctx is None:
            self._ctx = industry_analytics.load_context(as_of, version=info)
            self._key = key
            self.loads += 1
        return self._ctx

    def drop(self) -> None:
        self._key, self._ctx = None, None


def _validation_message(problems: list[str]) -> str:
    """Name the problems, and COUNT the ones the message could not carry."""
    shown = problems[:_MAX_VALIDATION_PROBLEMS]
    dropped = len(problems) - len(shown)
    tail = f" (+{dropped} more problem{'s' if dropped != 1 else ''} not shown)" if dropped else ""
    return f"{len(problems)} validation problem(s): " + "; ".join(shown) + tail


def _snapshot_for(period_key: str, info: gics_registry.VersionInfo) -> tuple[Any, list[str]]:
    """The cross-industry snapshot a group report should quote.

    The period's own snapshot is computed from the stats rows these very
    jobs write, so while the group reports are being generated it does not
    exist yet and the most recent prior snapshot is the honest stand-in.
    Returns ``(row_or_None, degraded_reasons)`` — the caller records them
    on the edition; the sections themselves carry the snapshot's own
    period, so a reader is never told last week's context is this week's.

    One snapshot feeds THREE sections, not one: ``cross_industry`` (the
    spillovers), ``companies`` (the event window) and ``outlook`` (the
    macro regime). A single ``cross_industry:…`` label would be read as
    scoping the degradation to the cross-industry section while the other
    two quietly carried last week's window under this week's as-of, so
    every section that reads the payload gets its own label.
    """
    current = industry_snapshot.snapshot_for_period(period_key, version=info)
    if current is not None:
        return current, []
    prior = industry_snapshot.latest_snapshot(version=info)
    if prior is not None:
        return prior, [f"{s}:prior_period:{prior.get('period_key')}" for s in _SNAPSHOT_FED_SECTIONS]
    return None, [f"{s}:none_yet" for s in _SNAPSHOT_FED_SECTIONS]


def _events_for(snapshot: Any, code: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The group's events out of the snapshot payload, WITH the provenance
    that says what a zero means — stored rows only, never a live provider
    call from inside a job.

    ``major_events`` is already truncated: ``compute_cross_snapshot`` keeps
    the top ``MAX_MAJOR_EVENTS`` by materiality across the WHOLE universe
    before storing them, so a group whose events ranked below the cut
    appears here with nothing at all. Three different situations therefore
    produce the same empty list — no snapshot, a genuinely quiet window, or
    a group squeezed out by the global cap — and publishing "Events in
    window: 0" for all three would state the last one as an observed fact.
    The group's *pre-cap* counts are on the snapshot's own group row
    (``events_14d`` / ``news_14d``), so what the cap dropped is countable
    exactly, and it is counted.
    """
    payload = (snapshot or {}).get("payload") or {} if isinstance(snapshot, dict) else {}
    events = [
        dict(e) for e in (payload.get("major_events") or [])
        if isinstance(e, dict) and e.get("industry_group_code") == code
    ]
    row = next(
        (r for r in (payload.get("groups") or [])
         if isinstance(r, dict) and r.get("code") == code),
        None,
    )
    in_window = (
        None if row is None
        else int(row.get("events_14d") or 0) + int(row.get("news_14d") or 0)
    )
    dropped = None if in_window is None else max(0, in_window - len(events))

    if not payload:
        reason = "no_snapshot"
    elif row is None:
        reason = "group_absent_from_snapshot"
    elif in_window == 0:
        reason = "none_stored_in_window"
    elif dropped and not events:
        reason = "all_dropped_by_snapshot_global_cap"
    elif dropped:
        reason = "partly_dropped_by_snapshot_global_cap"
    else:
        reason = None

    provenance = {
        "snapshot_period_key": (snapshot or {}).get("period_key") if isinstance(snapshot, dict) else None,
        "snapshot_as_of": payload.get("as_of"),
        "window_days": payload.get("events_window_days"),
        "n_in_report": len(events),
        "n_in_window_for_group": in_window,
        "n_dropped_by_snapshot_global_cap": dropped,
        "snapshot_global_cap": getattr(industry_snapshot, "MAX_MAJOR_EVENTS", None),
        "snapshot_kept_universe_wide": len(payload.get("major_events") or []),
        "snapshot_catalysts_universe_wide": payload.get("n_events"),
        "snapshot_news_universe_wide": payload.get("n_news"),
        # Each channel's own window and why it is empty, carried through
        # verbatim so "no events" can always name its channel.
        "channels": payload.get("events_sources"),
        "reason": reason,
    }
    return events, provenance


def _coverage_of(stats: Any, events: dict[str, Any] | None = None) -> dict[str, Any]:
    """The report's coverage block: the stats sample, the provider cache's
    stale-serve ledger and the event-window provenance, so "how good was
    the data" is answerable from the edition alone."""
    coverage = dict(getattr(stats, "sample", None) or {})
    if events is not None:
        coverage["events"] = events
    try:
        from . import provider_cache
        coverage["provider_stale"] = provider_cache.stale_stats()
    except Exception as exc:  # telemetry, not evidence — a failure is named
        coverage["provider_stale"] = {"reason": f"unavailable: {type(exc).__name__}"}
    return coverage


def _freshness_of(stats: Any, payload: dict[str, Any]) -> dict[str, Any]:
    sample = getattr(stats, "sample", None) or {}
    method = getattr(stats, "method", None) or {}
    metadata = ((payload.get("sections") or {}).get("metadata") or {}).get("facts") or {}
    as_of = getattr(stats, "as_of", None)
    return {
        "data_as_of": as_of.isoformat() if isinstance(as_of, datetime) else None,
        "prices_max_date": sample.get("prices_max_date"),
        "prices_min_date": sample.get("prices_min_date"),
        "stats_method_version": method.get("version"),
        "knowledge_version": metadata.get("knowledge_version"),
        "taxonomy_version": method.get("taxonomy_version") or metadata.get("taxonomy_version"),
        "atlas_snapshot": (
            ((payload.get("sections") or {}).get("cross_industry") or {}).get("facts") or {}
        ).get("atlas_snapshot_date"),
    }


def _server_facts(analyst: Any, stats: Any, snapshot: Any, prior: Any,
                  events: list[dict[str, Any]], *, job: dict[str, Any],
                  payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """An INDEPENDENT copy of the server-computed facts, for the validator.

    The validator's first job is to prove that ``facts`` in the published
    payload are exactly what the server computed — an LLM never writes into
    facts, and every number the interpretation quotes must be present in
    them. Both checks need a second copy built from the inputs. Reading the
    facts back out of the payload under review instead makes the mutation
    guard compare an object with itself (``x != x``, never true) and
    derives the allowed-numbers whitelist from the very document it is
    supposed to constrain, so an LLM section merged wholesale — facts key
    and all — would validate clean and publish invented numbers.

    ``build_facts`` is public for exactly this ("Exposed so a job can hand
    the same dict to the validator after the writer returns"). Two inputs
    have to be matched deliberately:

    * ``mode`` — ``write_report`` downgrades ``llm`` to ``llm_unavailable``
      when the model returns nothing, and stamps the final mode into the
      metadata facts. The payload's ``analyst_narrative`` IS that final
      mode, so passing it reproduces the same value.
    * ``metadata.generated_at`` — a clock read, different by construction on
      a second call. It is carried across by name rather than compared;
      it is an identifier, not a measurement (the validator scrubs dates
      out of the number whitelist), and it is the only field this skips.

    ``degraded`` / ``errors`` go to throwaway lists: the edition's own were
    recorded by the writer and must not be duplicated by the rebuild.
    """
    mode = str(payload.get("analyst_narrative") or "")
    facts = writer.build_facts(
        analyst, stats, snapshot, prior, list(events or []),
        run_id=job["run_id"], mode=mode, degraded=[], errors=[],
    )
    published_metadata = ((payload.get("sections") or {}).get("metadata") or {}).get("facts")
    if isinstance(published_metadata, dict) and isinstance(facts.get("metadata"), dict):
        facts["metadata"]["generated_at"] = published_metadata.get("generated_at")
    return facts


def _run_group_report(job: dict[str, Any], info: gics_registry.VersionInfo,
                      ctx: industry_analytics.AnalyticsContext, as_of: datetime) -> dict[str, Any]:
    """Stats → edition → validation → save, for one claimed group job."""
    code, period_key, job_id = job["code"], job["period_key"], job["id"]
    # The final attempt runs the deterministic writer: a week that failed
    # twice on the model must still end with an edition on the site,
    # labelled for what it is rather than passed off as the analyst's.
    deterministic = int(job["attempts"]) >= int(job["max_attempts"])

    _append_progress(job_id, "computing_stats")
    stats = industry_analytics.compute_group_stats(
        code, as_of=as_of, version=info, period_key=period_key, context=ctx, persist=True,
    )
    _append_progress(job_id, "stats_ready", stats_id=stats.id,
                     n_priced=(stats.sample or {}).get("n_with_prices"))

    analyst = get_industry_analyst(code, version=info)
    snapshot, snapshot_reasons = _snapshot_for(period_key, info)
    prior = industry_report_store.latest_good(code, version=info)
    events, events_provenance = _events_for(snapshot, code)

    _append_progress(job_id, "writing_report",
                     mode="deterministic" if deterministic else "default")
    with llm_call_context(agent_name=analyst.display_name, run_id=job["run_id"],
                          feature="industry_report", route="cheap"):
        result = writer.write_report(analyst, stats, snapshot, prior, events,
                                     run_id=job["run_id"], deterministic=deterministic)
    result.degraded.extend(snapshot_reasons)
    dropped = events_provenance.get("n_dropped_by_snapshot_global_cap") or 0
    if dropped:
        # Count what was dropped. The section will print "Events in
        # window: <kept>"; without this the difference is invisible.
        result.degraded.append(f"companies:events:snapshot_global_cap_dropped:{dropped}")
    if deterministic:
        result.degraded.append(f"generation:deterministic_final_attempt:{job['attempts']}")

    facts = _server_facts(analyst, stats, snapshot, prior, events, job=job,
                          payload=result.payload)
    problems = validator.validate(result.payload, facts)
    if problems:
        _append_progress(job_id, "validation_failed", n_problems=len(problems))
        raise ReportRejected(_validation_message(problems))

    _append_progress(job_id, "validated")
    report = industry_report_store.save_report(
        code=code, period_key=period_key, as_of=as_of, payload=result.payload, version=info,
        stats_id=stats.id,
        source_manifest=((result.payload.get("sections") or {}).get("sources") or {})
        .get("facts", {}).get("manifest", []),
        coverage=_coverage_of(stats, events_provenance),
        freshness=_freshness_of(stats, result.payload),
        generation=result.generation,
        degraded=result.degraded,
        errors=result.errors,
        job_id=job_id,
    )
    return {
        "report_id": report.id,
        "note": (
            f"version={report.version} status={report.status} "
            f"mode={result.payload.get('analyst_narrative')} degraded={len(result.degraded)}"
        ),
    }


def _run_cross_snapshot(job: dict[str, Any], info: gics_registry.VersionInfo,
                        as_of: datetime) -> dict[str, Any]:
    _append_progress(job["id"], "computing_cross_snapshot")
    row = industry_snapshot.compute_cross_snapshot(
        job["period_key"], as_of, version=info, persist=True,
    )
    coverage = (row.payload or {}).get("coverage") or {}
    return {
        "report_id": None,
        "note": (
            f"snapshot_id={row.id} groups={coverage.get('n_groups')} "
            f"with_stats={coverage.get('n_with_stats')} "
            f"insufficient={coverage.get('n_insufficient_sample')}"
        ),
    }


def execute_job(job_id: int, *, book: ContextBook | None = None) -> dict[str, Any]:
    """Run one claimed job to completion and record the outcome.

    A failure never writes a report row: the prior edition keeps
    ``is_latest_good`` and this row becomes the ``last_attempt`` the read
    API shows beside it. Attempts below ``max_attempts`` go back to
    ``queued`` with a 15-minutes-per-attempt backoff; the last one is
    final. Catches BaseException for the same reason the memo worker does
    — a cancellation that vanished would leave a job ``running`` forever.
    """
    with SessionLocal() as db:
        row = db.get(IndustryReportJob, job_id)
        if row is None:
            return {}
        job = _job_dict(row)
    info = gics_registry.resolve_version(job["taxonomy_version_id"])
    as_of = as_of_for_period(job["period_key"])
    started = _utcnow()
    _append_progress(job_id, "worker_claimed")
    log.info("industry job %d STARTING %s %s/%s (attempt %d/%d, run_id=%s)",
             job_id, job["kind"], job["code"] or "-", job["period_key"],
             job["attempts"], job["max_attempts"], job["run_id"])
    try:
        if job["kind"] == KIND_CROSS:
            outcome = _run_cross_snapshot(job, info, as_of)
        else:
            ctx = (book or ContextBook()).context_for(job["period_key"], as_of, info)
            outcome = _run_group_report(job, info, ctx, as_of)
        with SessionLocal() as db:
            r = db.get(IndustryReportJob, job_id)
            if r is not None:
                r.status = "succeeded"
                r.finished_at = _utcnow()
                r.heartbeat_at = r.finished_at
                r.report_id = outcome["report_id"]
                r.error_type = ""
                r.error_message = ""
                db.commit()
        _append_progress(job_id, f"job_succeeded {outcome['note']}")
        log.info("industry job %d SUCCEEDED in %.1fs (%s)", job_id,
                 (_utcnow() - started).total_seconds(), outcome["note"])
    except (SystemExit, KeyboardInterrupt):  # pragma: no cover — process shutdown
        _append_progress(job_id, "process_exit_signal")
        raise
    except BaseException as exc:
        tb = traceback.format_exc()
        _append_progress(job_id, f"exception_caught {type(exc).__name__}")
        # `safe_exc` because a provider URL (with its key) can reach an
        # exception string, and this text is stored and served to admins.
        log.error("industry job %d FAILED %s %s/%s after %.1fs: %s: %s", job_id, job["kind"],
                  job["code"] or "-", job["period_key"], (_utcnow() - started).total_seconds(),
                  type(exc).__name__, safe_exc(exc))
        now = _utcnow()
        retry = job["attempts"] < job["max_attempts"]
        with SessionLocal() as db:
            r = db.get(IndustryReportJob, job_id)
            if r is not None:
                r.error_type = type(exc).__name__
                r.error_message = redact(exc)[:_MAX_ERROR_CHARS]
                r.traceback_tail = tb[-1500:]
                r.heartbeat_at = now
                if retry:
                    r.status = "queued"
                    r.started_at = None
                    r.not_before = now + timedelta(minutes=BACKOFF_MINUTES * max(1, r.attempts))
                else:
                    r.status = "failed"
                    r.finished_at = now
                db.commit()
    # A report run holds a whole period's price series plus a rendered
    # edition; the worker is RSS-capped, so the pages go back to the OS
    # rather than into CPython's freelists. Both paths — a job that died
    # partway through is where the most garbage is left.
    try:
        from . import memory_probe
        memory_probe.trim_memory(f"industry_report_job_{job_id}")
    except Exception:  # pragma: no cover — housekeeping must not fail a job
        log.debug("trim_memory failed after industry job %d", job_id, exc_info=True)
    with SessionLocal() as db:
        r = db.get(IndustryReportJob, job_id)
        return _job_dict(r) if r is not None else {}


def process_next_job(*, book: ContextBook | None = None,
                     now: datetime | None = None) -> dict[str, Any] | None:
    """Claim + execute one job synchronously. Returns the finished job
    dict, or None when nothing is eligible. This is the drainer loop's
    body, exposed so tests (and the failure-path integration test) drain
    the queue deterministically instead of racing a polling thread."""
    job_id = claim_next_job(now)
    if job_id is None:
        return None
    return execute_job(job_id, book=book)


def drain(limit: int = 200, *, now: datetime | None = None) -> list[dict[str, Any]]:
    """Drain up to ``limit`` eligible jobs, sharing one analytics context
    across each period. Used by tests and by an operator draining by hand;
    the worker thread runs the same body one job at a time."""
    book = ContextBook()
    done: list[dict[str, Any]] = []
    for _ in range(max(0, int(limit))):
        job = process_next_job(book=book, now=now)
        if job is None:
            break
        done.append(job)
    return done


# ---------------------------------------------------------------------------
# Worker thread lifecycle
# ---------------------------------------------------------------------------

_worker_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _worker_loop() -> None:
    recover_orphans()
    log.info("industry report drainer started (poll=%.1fs, heartbeat=%ds)",
             POLL_SECONDS, HEARTBEAT_SECONDS)
    book = ContextBook()
    last_beat = 0.0
    while not _stop_event.is_set():
        try:
            # The heartbeat is what keeps cron-health honest between
            # Sundays: this thread is idle six days a week, and an idle
            # thread that says nothing is indistinguishable from a dead one.
            now = _utcnow()
            if last_beat == 0.0 or (now.timestamp() - last_beat) >= HEARTBEAT_SECONDS:
                heartbeat(now)
                last_beat = now.timestamp()
            if process_next_job(book=book) is None:
                book.drop()  # nothing queued: release the period's prices
                _stop_event.wait(POLL_SECONDS)
        except (SystemExit, KeyboardInterrupt):  # pragma: no cover
            raise
        except BaseException:  # pragma: no cover — the loop must survive anything
            log.exception("industry report drainer error; continuing")
            _stop_event.wait(POLL_SECONDS)


def start_worker(*, force: bool = False) -> bool:
    """Start the singleton drainer thread. Returns True when running.

    No-ops (returning False) when ``ENABLE_INDUSTRY_REPORTS=false`` — which
    is the web service, where a page view must never generate — or under
    pytest unless ``force=True``, because tests drive the queue through
    ``process_next_job`` and a background thread writing reports during a
    test session is exactly the accidental-spend mode conftest prevents.
    """
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return True
    if not settings.enable_industry_reports:
        log.info("industry report drainer disabled via ENABLE_INDUSTRY_REPORTS")
        return False
    if "pytest" in sys.modules and not force:
        log.info("industry report drainer not started under pytest (use force=True)")
        return False
    _stop_event.clear()
    _worker_thread = threading.Thread(
        target=_worker_loop, name="industry-report-worker", daemon=True,
    )
    _worker_thread.start()
    return True


def is_running() -> bool:
    """Whether this process's drainer thread is alive. Read by the worker
    entrypoint's heartbeat: the thread writes its own cron-health row, and
    the main loop speaks for it only when it is missing — a dead drainer
    reported as healthy by a living process is the failure this avoids."""
    return _worker_thread is not None and _worker_thread.is_alive()


def stop_worker(timeout: float = 5.0) -> None:
    """Signal the drainer to exit. An in-flight job finishes on its own
    (the thread is a daemon and dies with the process either way)."""
    global _worker_thread
    _stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=timeout)
        _worker_thread = None
