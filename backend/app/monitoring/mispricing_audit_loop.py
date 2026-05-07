"""Wave 10 — daily mispricing-audit cron.

Scheduled at 04:30 UTC (between outcome_loop@02:30 and history_backfill
@03:15) so the most recent memos are already evaluated when the audit
samples them.

Audits up to 20 of the most-recent memos with non-empty
mispricing_thesis fields, persists the run via
`mispricing_audit.persist_audit`, and the result is automatically
picked up by the PM's self-improvement context block on the next
synthesis. Closes the loop without manual intervention.

Cost: ~$0.01-0.02/day (one cheap-tier LLM call). Skipped silently
when no API key is configured.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..services.mispricing_audit import (
    aggregate_scores,
    persist_audit,
    run_audit,
)
from . import record_run

log = logging.getLogger(__name__)


def run_once(*, limit: int = 20) -> Dict[str, Any]:
    audit = run_audit(limit=limit)
    aggregate = aggregate_scores(audit)
    audit_id = persist_audit(audit, aggregate)
    note = (
        f"audited={audit.get('audited')} "
        f"weak={aggregate.get('weak_memo_count')} "
        f"id={audit_id}"
    )
    record_run(
        "mispricing_audit_loop",
        success=audit_id is not None or audit.get("audited") == 0,
        note=note,
    )
    return {
        "audited": audit.get("audited"),
        "audit_id": audit_id,
        "aggregate": aggregate,
    }


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron", hour=4, minute=30,
        id="mispricing_audit_loop", replace_existing=True,
    )
