"""FEAT-003 — daily company → GICS industry-group classification audit.

Runs 03:40 UTC on the worker, after ``history_backfill`` (03:15) has
refreshed company profiles and before ``scorecard_loop`` (03:45). Every
run: make sure a taxonomy is active (bootstrapping the bundled version on
a fresh database), classify every ``companies`` row in bulk, and record
the state counts.

``success=False`` whenever any company is ``missing`` — an unmapped
provider label is exactly the kind of drift that would otherwise route a
company to the wrong (or no) Industry Group Analyst without anyone
noticing. The note carries every count and the top unmapped labels so
the operator reading ``/api/admin/cron-health`` can extend the alias map
without opening a database.
"""
from __future__ import annotations

import logging

from ..agents.log_safety import safe_exc
from ..services import gics_registry, industry_classification
from . import record_run

log = logging.getLogger(__name__)

LOOP_NAME = "industry_classification_loop"
HOUR_UTC = 3
MINUTE_UTC = 40
_NOTE_LABEL_CAP = 5


def _note(summary: dict) -> str:
    counts = summary.get("counts") or {}
    parts = [
        f"taxonomy={summary.get('taxonomy_version', '?')}",
        f"classified={summary.get('classified', 0)}",
        f"mapped={counts.get('mapped', 0)}",
        f"fallback={counts.get('fallback', 0)}",
        f"missing={counts.get('missing', 0)}",
        f"conflict={counts.get('conflict', 0)}",
        f"stale={counts.get('stale', 0)}",
        f"inserted={summary.get('inserted', 0)}",
        f"reclassified={summary.get('reclassified', 0)}",
        f"stale_fixed={summary.get('stale_fixed', 0)}",
    ]
    labels = summary.get("unmapped_labels") or []
    if labels:
        shown = "; ".join(
            f"{lab.get('sector') or '?'}/{lab.get('industry') or '?'}×{lab.get('count', 0)}"
            for lab in labels[:_NOTE_LABEL_CAP]
        )
        more = f" (+{len(labels) - _NOTE_LABEL_CAP} more)" if len(labels) > _NOTE_LABEL_CAP else ""
        parts.append(f"unmapped_labels={shown}{more}")
    return " ".join(parts)


def run_once() -> dict:
    """Bootstrap the taxonomy if needed, classify everything, record counts.

    An exception is recorded as ``success=False`` with the exception type
    before it propagates — the loop must reach both health surfaces even
    when it dies, or the endpoint reads "never run" forever.
    """
    try:
        active = gics_registry.ensure_taxonomy(activate=True)
        if active is None:
            raise gics_registry.TaxonomyNotImported("no taxonomy could be activated")
        summary = industry_classification.classify_all(version=active)
    except Exception as exc:
        record_run(LOOP_NAME, success=False, note=f"error={type(exc).__name__}: {safe_exc(exc)}")
        raise
    missing = int((summary.get("counts") or {}).get("missing", 0))
    record_run(LOOP_NAME, success=missing == 0, note=_note(summary))
    log.info("industry classification: %s", _note(summary))
    return summary


def register(scheduler) -> None:
    scheduler.add_job(
        run_once, "cron", hour=HOUR_UTC, minute=MINUTE_UTC,
        id=LOOP_NAME, replace_existing=True, max_instances=1, coalesce=True,
    )
