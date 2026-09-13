"""Read-only triage of every persisted outcome, including orphaned rows.

Age alone cannot prove a baseline is correct: historical provider responses
and their coverage were not persisted. No row is certified clean by this audit.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import MemoOutcome, MemoSnapshot
from .outcome_service import DEFAULT_HORIZONS, PRICE_DATE_TOLERANCE_DAYS

_BASELINE = re.compile(r"(?:^|,\s*)baseline=([^,]+)")


def _utc(value: datetime | None) -> datetime | None:
    # ORM columns use naive datetime.utcnow(); offsets supplied by other
    # callers must be converted, not stripped or double-suffixed with Z.
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    normalized = _utc(value)
    return normalized.isoformat().replace("+00:00", "Z") if normalized else None


def _classify(row: dict[str, Any]) -> dict[str, Any]:
    generated = _utc(row["generated_at"])
    evaluated = _utc(row["evaluated_at"])
    horizon = row["horizon_days"]
    valid_horizon = isinstance(horizon, int) and horizon > 0
    threshold = (horizon + 30) * 7 / 5 if valid_horizon else None
    elapsed = evaluated - generated if generated and evaluated else None
    reasons = []
    if row["snapshot_id"] is None:
        reasons.append("missing_snapshot")
    if not generated or not evaluated:
        reasons.append("missing_or_invalid_timestamp")
    if not valid_horizon:
        reasons.append("invalid_horizon")
    if elapsed is not None and elapsed < timedelta(0):
        reasons.append("evaluated_before_generated")
    if row["snapshot_ticker"] and row["snapshot_ticker"] != row["ticker"]:
        reasons.append("snapshot_ticker_mismatch")
    if row["as_of_date"] is not None:
        reasons.append("backtest_outcome")

    late = None
    if elapsed is not None and elapsed >= timedelta(0) and valid_horizon:
        # Cross-multiply to retain exact microsecond boundaries. Do not
        # truncate elapsed time with .days or use PostgreSQL integer division.
        late = elapsed * 5 > timedelta(days=(horizon + 30) * 7)

    match = _BASELINE.search(row.get("note") or "")
    baseline_date = None
    baseline_offset = None
    baseline_outside_tolerance = False
    if match:
        try:
            baseline_date = date.fromisoformat(match[1].strip())
        except ValueError:
            reasons.append("invalid_recorded_baseline")
        if baseline_date and generated:
            baseline_offset = (baseline_date - generated.date()).days
            baseline_outside_tolerance = abs(baseline_offset) > PRICE_DATE_TOLERANCE_DAYS
    indeterminate = bool(reasons)
    if late:
        reasons.append("late_evaluation_threshold")
    if baseline_outside_tolerance:
        reasons.append("recorded_baseline_outside_tolerance")
    status = (
        "candidate" if late or baseline_outside_tolerance
        else "indeterminate" if indeterminate else "not_flagged"
    )
    return {
        **row,
        "generated_at": _iso(row["generated_at"]),
        "evaluated_at": _iso(row["evaluated_at"]),
        "as_of_date": _iso(row["as_of_date"]),
        "evaluation_age_days": elapsed.total_seconds() / 86400 if elapsed is not None else None,
        "threshold_days": threshold,
        "late_evaluation_candidate": late,
        "recorded_baseline_date": baseline_date.isoformat() if baseline_date else None,
        "recorded_baseline_offset_days": baseline_offset,
        "baseline_metadata_present": bool(match),
        "status": status,
        "reasons": reasons,
    }


def audit_outcomes(db: Session) -> dict[str, Any]:
    """SELECT only: no table creation, provider fetch, evaluation, or write.

    One outer-joined statement observes all rows in a consistent statement
    snapshot on Postgres, without loading memo_json blobs. No limit or date
    filter is applied. Totals are computed from the same rows we return.
    """
    statement = select(
        *MemoOutcome.__table__.c,
        MemoSnapshot.id.label("snapshot_id"),
        MemoSnapshot.ticker.label("snapshot_ticker"),
        MemoSnapshot.version.label("snapshot_version"),
        MemoSnapshot.generated_at,
        MemoSnapshot.as_of_date,
    ).outerjoin(
        MemoSnapshot, MemoSnapshot.id == MemoOutcome.memo_snapshot_id,
    ).order_by(MemoOutcome.id)
    with db.no_autoflush:
        rows = [_classify(dict(row)) for row in db.execute(statement).mappings()]
    counts = Counter(row["status"] for row in rows)
    reasons = Counter(reason for row in rows for reason in row["reasons"])
    by_horizon: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_horizon.setdefault(str(row["horizon_days"]), {
            "total": 0, "candidate": 0, "indeterminate": 0, "not_flagged": 0,
            "late_evaluation_candidates": 0,
        })
        bucket["total"] += 1
        bucket[row["status"]] += 1
        bucket["late_evaluation_candidates"] += int(row["late_evaluation_candidate"] is True)
    return {
        "audit_version": 1,
        "audited_at": _iso(datetime.now(UTC)),
        "read_only": True,
        "total_rows": len(rows),
        "returned_rows": len(rows),
        "truncated": False,
        "excluded_rows": 0,
        "counts": {
            "candidate": counts["candidate"],
            "indeterminate": counts["indeterminate"],
            "not_flagged": counts["not_flagged"],
            "late_evaluation_candidates": reasons["late_evaluation_threshold"],
            "missing_snapshot": reasons["missing_snapshot"],
            "recorded_baseline_outside_tolerance": reasons["recorded_baseline_outside_tolerance"],
            "without_baseline_metadata": sum(not row["baseline_metadata_present"] for row in rows),
        },
        "reason_counts": dict(sorted(reasons.items())),
        "by_horizon": by_horizon,
        "predicate": {
            "expression": "evaluated_at - generated_at > (horizon_days + 30) * 7.0 / 5.0 days",
            "comparison": "strictly greater; exact elapsed time, not rounded calendar dates",
            "timezone": "UTC; naive stored timestamps are interpreted as UTC",
            "threshold_days": {str(h): (h + 30) * 7 / 5 for h in DEFAULT_HORIZONS},
            "baseline_tolerance_days": PRICE_DATE_TOLERANCE_DAYS,
        },
        "limitations": [
            "Candidates require investigation; the threshold is triage, not proof of contamination.",
            "Not-flagged rows are unverified, not certified clean. Truncated provider responses can "
            "shift a baseline even below the threshold.",
            "The 7/5 approximation assumes complete weekday bars; holidays, endpoint semantics, "
            "stale caches, and missing observations can change coverage.",
            "Legacy rows lack baseline dates and historical provider/window provenance. "
            "Recorded baseline dates alone do not verify prices, benchmark alignment, or total returns.",
            "All stored rows are included, including new evaluations, backtests, and missing snapshots. "
            "No outcome or headline KPI is changed.",
        ],
        "rows": rows,
    }
