"""FEAT-003 — Industry Analysis ops endpoints (slice 5).

Under `/api/admin/*`, so `admin_auth`'s prefix middleware guards them the
moment they are mounted (bearer `ADMIN_API_TOKEN`), and `auth/policy.py`
classifies them as `admin`: a customer JWT never opens one, and none of
them is browser-called, so none is exempt.

Operations and their boundaries:

* **import** — reads the bundled knowledge JSON (the one taxonomy source)
  and writes the node rows. Idempotent by checksum; the same key with a
  different structure is a 409, never a silent rewrite of a version other
  rows already point at.
* **classify** — one bulk pass over `companies` (two reads, one insert).
  No per-ticker loop, so it is safe on the web process.
* **regenerate** — ENQUEUES. It never generates a report in the request:
  the worker's drainer owns that, and a web process that generated one
  would be the "expensive work in a page request" failure this repo has
  already paid for. Until the drainer module lands, this answers 503 and
  names what is missing rather than pretending a job was queued.
* **jobs** — reads the queue. Counts by status come from SQL, not from
  the returned page, so a capped list never understates the backlog.
* **recover-legacy** — reconciles explicitly identified legacy jobs after
  an operator verifies predecessor retirement. It never executes work;
  ownership and every expected row field must still match.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from sqlalchemy import func, select

from ..config import settings
from ..database import SessionLocal
from ..models import IndustryReportJob
from ..rate_limit import LIMITS, limiter
from ..schemas.industry import (
    ClassifyOut,
    ClassifyRequest,
    IndustryJobsOut,
    LegacyIndustryRecoveryRequest,
    RegenerateOut,
    RegenerateRequest,
    TaxonomyImportOut,
    TaxonomyImportRequest,
)
from ..services import gics_registry, industry_analytics, industry_classification

log = logging.getLogger(__name__)
router = APIRouter()

# Lists in a response are capped so one call cannot return the whole
# universe; every cap reports what it dropped.
CHANGED_LIMIT = 200
UNMAPPED_LIMIT = 100
JOBS_LIMIT = 50


def _utcnow() -> datetime:
    """Clock seam — tests monkeypatch this instead of freezing time."""
    return datetime.utcnow()


def _error(status: int, error_code: str, message: str, **extra: Any) -> HTTPException:
    """A structured refusal: `code` + `message` plus whatever names the
    state (the group, the remedy, the failed attempt). The first argument
    is `error_code`, not `code`, so a caller can pass the industry-group
    `code=` in `extra` without colliding with it."""
    return HTTPException(status_code=status, detail={"code": error_code, "message": message, **extra})


def _active_or_503() -> gics_registry.VersionInfo:
    try:
        return gics_registry.require_active_version()
    except gics_registry.TaxonomyNotImported:
        raise _error(
            503, "taxonomy_not_imported",
            "no active GICS taxonomy version; import one first",
            remedy="POST /api/admin/industries/taxonomy/import with {\"activate\": true}",
        ) from None


def default_period_key(now: datetime | None = None) -> str:
    """The ISO week of the most recent as-of weekday (Friday by default).

    The weekly loop owns the canonical derivation; this mirrors it from
    the same settings so an admin regenerate and the cron land on the
    same `period_key` rather than creating a second edition series.
    """
    now = now or _utcnow()
    target = int(settings.industry_reports_as_of_weekday)
    back = (now.weekday() - target) % 7
    return industry_analytics.period_key_for(now - timedelta(days=back))


# ---------------------------------------------------------------------------
# Taxonomy import
# ---------------------------------------------------------------------------


@router.post("/api/admin/industries/taxonomy/import", response_model=TaxonomyImportOut)
@limiter.limit(LIMITS["industry_admin"])
def import_taxonomy_endpoint(
    response: Response,
    request: Request, payload: TaxonomyImportRequest | None = None,
) -> TaxonomyImportOut:
    """Import the bundled knowledge JSON as a taxonomy version.

    409 when the key is already imported with a different node checksum:
    a changed structure is a new version, because old reports and
    classifications point at the old one by id.
    """
    payload = payload or TaxonomyImportRequest()
    try:
        result = gics_registry.import_from_knowledge_json(
            version_key=payload.version_key, activate=payload.activate, notes=payload.notes,
        )
    except gics_registry.TaxonomyChecksumMismatch as exc:
        raise _error(409, "taxonomy_checksum_mismatch", str(exc)) from None
    except FileNotFoundError as exc:
        # The loader raises by design: a missing knowledge base is a build
        # defect, and an empty taxonomy would surface weeks later.
        raise _error(503, "knowledge_base_missing", str(exc)) from None
    except ValueError as exc:
        raise _error(422, "bad_taxonomy_payload", str(exc)) from None
    return TaxonomyImportOut(drift=gics_registry.bundled_drift(), **result)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@router.post("/api/admin/industries/classify", response_model=ClassifyOut)
@limiter.limit(LIMITS["industry_admin"])
def classify_endpoint(
    request: Request, response: Response, payload: ClassifyRequest | None = None,
) -> ClassifyOut:
    """Re-classify the universe (or the named tickers) against the active
    taxonomy. Bulk: two reads and one insert, whatever the size."""
    payload = payload or ClassifyRequest()
    info = _active_or_503()
    summary = industry_classification.classify_all(
        version=info, tickers=payload.tickers, reclassify=payload.reclassify, force=payload.force,
    )
    changed = list(summary.get("changed") or [])
    unmapped = list(summary.get("unmapped_labels") or [])
    return ClassifyOut(
        taxonomy_version=summary["taxonomy_version"],
        classified=summary["classified"],
        inserted=summary["inserted"],
        reclassified=summary["reclassified"],
        restamped=summary["restamped"],
        unchanged=summary["unchanged"],
        stale_detected=summary["stale_detected"],
        stale_fixed=summary["stale_fixed"],
        counts=summary["counts"],
        sources=summary["sources"],
        changed=changed[:CHANGED_LIMIT],
        changed_total=len(changed),
        changed_truncated=max(0, len(changed) - CHANGED_LIMIT),
        unmapped_labels=unmapped[:UNMAPPED_LIMIT],
        unmapped_labels_total=len(unmapped),
        unmapped_labels_truncated=max(0, len(unmapped) - UNMAPPED_LIMIT),
        mapping_caveat=gics_registry.MAPPING_CAVEAT,
    )


# ---------------------------------------------------------------------------
# Report regeneration
# ---------------------------------------------------------------------------


def _worker():
    """The drainer module, or `None` when it has not landed yet.

    Imported lazily and by name so this file does not depend on slice 4's
    import graph; the route says "queue unavailable" rather than 500ing
    on an ImportError.
    """
    try:
        from ..services import industry_report_worker  # type: ignore[attr-defined]
    except ImportError:
        return None
    return industry_report_worker


@router.post("/api/admin/industries/reports/regenerate", status_code=202, response_model=RegenerateOut)
@limiter.limit(LIMITS["industry_admin"])
def regenerate_reports_endpoint(
    response: Response,
    request: Request, payload: RegenerateRequest | None = None,
) -> RegenerateOut:
    """Queue report generation for one period — the same job path the
    Sunday cron uses, so an admin refresh and a scheduled one cannot
    diverge. 202: the worker does the work.

    Codes are validated against the registry first, so a typo is a 404
    here rather than a job that fails three times in the worker.
    """
    payload = payload or RegenerateRequest()
    info = _active_or_503()

    # A drifted taxonomy means the bundled knowledge JSON no longer
    # describes the node set the registry is serving, so a run queued now
    # would score groups the deploy has already changed and publish the
    # result as this week's edition. The daily classification loop already
    # reports this (`taxonomy_drift=1` in cron-health); refusing here keeps
    # an operator from queueing a week against the old structure without
    # meaning to. Explicitly overridable, because re-running the current
    # structure deliberately is legitimate.
    drift = gics_registry.bundled_drift(info)
    if drift is not None and not payload.accept_stale_taxonomy:
        raise _error(
            409, "taxonomy_drift",
            "the bundled knowledge base no longer matches the active taxonomy version; "
            "import the new structure under a new version key, or pass "
            "accept_stale_taxonomy=true to queue against the structure now active",
            drift=drift, taxonomy_version=info.version_key,
        )

    codes: list[str] | None = None
    if payload.codes is not None:
        codes = []
        for raw in payload.codes:
            try:
                codes.append(gics_registry.group(str(raw).strip(), version=info).code)
            except gics_registry.UnknownNode as exc:
                raise _error(404, "unknown_industry_group", str(exc), industry_group_code=str(raw)) from None
    period_key = payload.period_key or default_period_key()

    worker = _worker()
    if worker is None or not hasattr(worker, "enqueue_period"):
        raise _error(
            503, "report_queue_unavailable",
            "the industry report drainer (app/services/industry_report_worker.py, FEAT-003 slice 4) "
            "is not on this build, so nothing can be queued",
            period_key=period_key, taxonomy_version=info.version_key,
            requested=len(codes) if codes is not None else len(gics_registry.industry_groups(version=info)),
        )
    try:
        result = worker.enqueue_period(period_key, codes=codes, source="admin", force=payload.force)
    except TypeError as exc:
        # The frozen 4→5 contract is enqueue_period(period_key, codes=None,
        # *, source, force). A mismatch is a wiring bug; say which, rather
        # than returning a 500 with a stack trace.
        raise _error(
            503, "report_queue_contract_mismatch",
            f"industry_report_worker.enqueue_period did not accept the agreed arguments: {exc}",
            period_key=period_key,
        ) from None
    withheld = set(result.get("skipped_withheld_codes") or [])
    return RegenerateOut(
        period_key=period_key,
        taxonomy_version=info.version_key,
        requested=len(codes) if codes is not None else len(gics_registry.industry_groups(version=info)),
        # `enqueued`/`coalesced`/`skipped_published` are COUNTS on the
        # queue's contract; the code lists live under the `_codes` keys.
        # Reading the counts as lists is what made this a 500.
        enqueued=[{"code": c} for c in (result.get("enqueued_codes") or [])],
        coalesced=[{"code": c} for c in (result.get("coalesced_codes") or [])],
        skipped=(
            # A week whose only product is an audit-only template was
            # generated but is NOT on the site (owner decision 1); calling
            # it "published" would tell the operator retrying it that
            # there is nothing to retry.
            [{"code": c,
              "reason": ("generated this period as an audit-only template (not published); "
                         "pass force=true to retry")
              if c in withheld else "already generated this period"}
             for c in (result.get("skipped_published_codes") or [])]
            + [{"code": c, "reason": "over the per-run job cap"}
               for c in (result.get("over_budget_codes") or [])]
            + [{"code": c, "reason": "unknown to this taxonomy version"}
               for c in (result.get("unknown_codes") or [])]
        ),
        note=str(result.get("note") or "queued; the worker's drainer picks these up on its next tick"),
    )


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


def _job_dict(row: IndustryReportJob) -> dict[str, Any]:
    return {
        "id": row.id,
        "kind": row.kind,
        "code": row.industry_group_code,
        "period_key": row.period_key,
        "run_id": row.run_id,
        "status": row.status,
        "attempts": row.attempts,
        "max_attempts": row.max_attempts,
        "priority": row.priority,
        "not_before": row.not_before.isoformat() if row.not_before else None,
        "enqueued_at": row.enqueued_at.isoformat() if row.enqueued_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "heartbeat_at": row.heartbeat_at.isoformat() if row.heartbeat_at else None,
        "source": row.source,
        "force": bool(row.force),
        "report_id": row.report_id,
        "snapshot_id": row.snapshot_id,
        "ownership_tracked": bool(row.owner_token),
        "lease_expires_at": row.lease_expires_at.isoformat() if row.lease_expires_at else None,
        "error_type": row.error_type or "",
        "error_message": (row.error_message or "")[:500],
        "progress_waypoints": len(row.progress or []),
    }


@router.post("/api/admin/industries/jobs/recover-legacy")
@limiter.limit(LIMITS["industry_admin"])
def recover_legacy_industry_jobs_endpoint(
    request: Request, response: Response, payload: LegacyIndustryRecoveryRequest,
) -> dict[str, Any]:
    """Reconcile explicitly identified jobs after verified predecessor retirement.

    This operator-only action never runs a report or calls a provider. The
    service compares every expected field and refuses changed or owned jobs;
    ordinary worker startup continues to defer unknown legacy ownership.
    """
    from ..services.industry_legacy_recovery import recover_legacy_jobs

    try:
        return recover_legacy_jobs(
            [job.model_dump(mode="python") for job in payload.expected_jobs],
            payload.retirement_evidence,
        )
    except ValueError as exc:
        raise _error(400, "invalid_legacy_recovery", str(exc)) from None


@router.get("/api/admin/industries/jobs", response_model=IndustryJobsOut)
def industry_jobs_endpoint(
    status: str | None = Query(None, description="queued | running | succeeded | failed | skipped"),
    code: str | None = Query(None, description="Industry group code."),
    period_key: str | None = Query(None),
    limit: int = Query(JOBS_LIMIT, ge=1, le=500),
) -> IndustryJobsOut:
    """The report queue, newest first, with status counts computed in SQL
    over the whole (filtered) queue — so a capped page never makes a
    backlog look smaller than it is."""
    info = gics_registry.active_version()
    filters = []
    if info is not None:
        filters.append(IndustryReportJob.taxonomy_version_id == info.id)
    if status:
        filters.append(IndustryReportJob.status == status)
    if code:
        filters.append(IndustryReportJob.industry_group_code == str(code))
    if period_key:
        filters.append(IndustryReportJob.period_key == str(period_key))

    with SessionLocal() as db:
        rows = db.execute(
            select(IndustryReportJob).where(*filters)
            .order_by(IndustryReportJob.id.desc()).limit(limit)
        ).scalars().all()
        counts = {
            str(s): int(n) for s, n in db.execute(
                select(IndustryReportJob.status, func.count(IndustryReportJob.id))
                .where(*filters).group_by(IndustryReportJob.status)
            ).all()
        }
    total = sum(counts.values())
    worker = _worker()
    return IndustryJobsOut(
        count=len(rows),
        limit=limit,
        truncated=max(0, total - len(rows)),
        status_counts=counts,
        taxonomy_version=info.version_key if info is not None else "",
        drainer={
            "module_present": worker is not None,
            "enabled": bool(settings.enable_industry_reports),
            "note": (
                "the drainer runs on the worker service only (ENABLE_INDUSTRY_REPORTS); "
                "this endpoint reads the queue, it does not drain it"
                if worker is not None else
                "app/services/industry_report_worker.py is not on this build (FEAT-003 slice 4)"
            ),
        },
        jobs=[_job_dict(r) for r in rows],
    )
