"""Explicit operator cutover after every unowned predecessor is confirmed dead.

No claim or generation occurs here. The ordinary worker never invokes this;
NULL ownership remains deferred until an operator provides exact expectations.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from ..agents.log_safety import redact_unbounded
from ..database import SessionLocal
from ..models import CrossIndustrySnapshot, IndustryReport, IndustryReportJob
from . import industry_report_store
from . import industry_report_worker as worker

log = logging.getLogger(__name__)

_FIELDS = {"id", "run_id", "attempts", "started_at", "heartbeat_at"}


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("Expected an ISO timestamp or null")
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


def recover_legacy_jobs(expected_jobs: list[dict], retirement_evidence: str) -> dict:
    """Recover at most 50 exact unowned attempts after operator retirement proof.

    The authenticated route supplies the attestation. It is retained redacted
    in every successful disposition; this service cannot independently verify
    the host's process retirement. Changed rows and ambiguous receipts reject.
    """
    if not isinstance(expected_jobs, list) or not 1 <= len(expected_jobs) <= 50:
        raise ValueError("Expected 1 to 50 explicit legacy job identities")
    if not isinstance(retirement_evidence, str) or not retirement_evidence.strip() or len(retirement_evidence) > 4000:
        raise ValueError("Nonempty retirement evidence is required (maximum 4000 characters)")
    expected = []
    for raw in expected_jobs:
        if (not isinstance(raw, dict) or set(raw) != _FIELDS
                or type(raw["id"]) is not int or raw["id"] < 1
                or type(raw["attempts"]) is not int or raw["attempts"] < 0
                or not isinstance(raw["run_id"], str) or not raw["run_id"]):
            raise ValueError("Invalid expected legacy job identity")
        expected.append({**raw, "started_at": _timestamp(raw["started_at"]),
                         "heartbeat_at": _timestamp(raw["heartbeat_at"])})
    if len({row["id"] for row in expected}) != len(expected):
        raise ValueError("Legacy job identities must be unique")
    evidence = redact_unbounded(retirement_evidence.strip())
    results = []
    with SessionLocal() as db:
        for item in expected:
            result = {"id": item["id"], "action": "rejected"}
            predicates = [IndustryReportJob.status == "running", IndustryReportJob.owner_token.is_(None),
                          IndustryReportJob.lease_expires_at.is_(None)]
            predicates += [getattr(IndustryReportJob, key) == value for key, value in item.items()]
            changed = db.execute(update(IndustryReportJob).where(*predicates).values(
                owner_token=None).execution_options(synchronize_session=False)).rowcount
            if not changed:
                result["reason"] = "expected_legacy_attempt_did_not_match"
            else:
                job = db.get(IndustryReportJob, item["id"])
                reports = list(db.execute(select(
                    IndustryReport.id, IndustryReport.taxonomy_version_id, IndustryReport.industry_group_code,
                    IndustryReport.period_key, IndustryReport.as_of, IndustryReport.status,
                ).where(IndustryReport.job_id == job.id)))
                report = reports[0] if len(reports) == 1 else None
                try:
                    expected_as_of = worker.as_of_for_period(job.period_key)
                except (ValueError, TypeError):
                    expected_as_of = None
                snapshot_ids = list(db.scalars(select(CrossIndustrySnapshot.id).where(
                    CrossIndustrySnapshot.taxonomy_version_id == job.taxonomy_version_id,
                    CrossIndustrySnapshot.period_key == job.period_key,
                    CrossIndustrySnapshot.as_of == expected_as_of,
                ))) if job.kind == worker.KIND_CROSS and expected_as_of is not None else []
                consistent = report is not None and (
                    report.taxonomy_version_id == job.taxonomy_version_id
                    and report.industry_group_code == job.industry_group_code
                    and report.period_key == job.period_key
                    and report.as_of == expected_as_of
                    # Every stored status is a receipt, audit_only included: a final
                    # attempt that wrote an audit-only template DID publish its output.
                    and report.status in industry_report_store.REPORT_STATUSES
                    and (job.report_id is None or job.report_id == report.id))
                if expected_as_of is None:
                    result["reason"] = "invalid_stored_period"
                elif len(reports) > 1 or (reports and (job.kind != worker.KIND_GROUP or not consistent)):
                    result["reason"] = "ambiguous_or_inconsistent_publication"
                    result["report_ids"] = [r.id for r in reports]
                elif snapshot_ids:
                    result["reason"] = "ambiguous_legacy_snapshot"
                    result["snapshot_ids"] = snapshot_ids
                elif job.snapshot_id is not None or (job.report_id is not None and not consistent):
                    result["reason"] = "unverified_publication_receipt"
                elif job.kind not in worker.KINDS:
                    result["reason"] = "unknown_job_kind"
                else:
                    now = worker._utcnow()
                    if consistent:
                        job.report_id = report.id
                        job.status, job.finished_at = "succeeded", now
                        job.error_type = job.error_message = job.traceback_tail = ""
                        result.update(action="published_recovered", report_id=report.id)
                    elif job.attempts < job.max_attempts:
                        job.status, job.started_at = "queued", None
                        job.not_before = now + timedelta(minutes=worker.BACKOFF_MINUTES * max(1, job.attempts))
                        result["action"] = "requeued"
                    else:
                        job.status, job.finished_at = "failed", now
                        job.error_type = "WorkerRestart"
                        job.error_message = "Operator confirmed predecessor retirement; legacy attempt cap exhausted without a verified publication."
                        result["action"] = "failed"
                    job.progress = list(job.progress or []) + [{
                        "step": "operator_legacy_recovery", "at": now.isoformat(),
                        "retirement_evidence": evidence, "action": result["action"],
                        "expected": {key: value.isoformat() if isinstance(value, datetime) else value
                                     for key, value in item.items()},
                        "report_id": job.report_id,
                    }]
            results.append(result)
        # One bounded transaction: an unexpected later failure rolls back all
        # dispositions, and rejected no-op locks do not alter stored values.
        db.commit()
    log.warning("operator industry legacy recovery dispositions: %s", results)
    return {"requested": len(expected), "recovered": sum(r["action"] != "rejected" for r in results),
            "results": results, "retirement_evidence": evidence}
