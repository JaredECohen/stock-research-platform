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

from ..models import MemoOutcome, MemoOutcomeEligibility, MemoSnapshot
from .outcome_eligibility import identity_matches
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


def _dev_copy_set_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Directly classified dev-copy rows versus the enumerated evidence set."""
    from .outcome_eligibility import REASON_DEV_COPY
    from .outcome_eligibility_evidence import DEV_COPY_SNAPSHOTS

    expected = {sid for sid, _, _ in DEV_COPY_SNAPSHOTS}
    observed_rows = [
        row for row in rows
        if row.get("eligibility_reason") == REASON_DEV_COPY and row.get("eligibility_inherited_from") is None
    ]
    observed = {row["memo_snapshot_id"] for row in observed_rows}
    return {
        "expected_snapshots": len(expected),
        "observed_snapshots": len(observed),
        "observed_rows": len(observed_rows),
        "matches": observed == expected,
        "missing_snapshot_ids": sorted(expected - observed),
        "unexpected_snapshot_ids": sorted(observed - expected),
    }


def audit_outcomes(db: Session) -> dict[str, Any]:
    """SELECT only: no table creation, provider fetch, evaluation, or write.

    One outer-joined statement observes all rows in a consistent statement
    snapshot on Postgres, without loading memo_json blobs. No limit or date
    filter is applied. Totals are computed from the same rows we return.

    W6: each row is ANNOTATED with its track-record eligibility (identity-
    valid ledger row, else null = unclassified). The audit never filters on
    it: every stored row is listed whatever its eligibility, which is what
    lets a before/after receipt prove exclusion changed no outcome.
    """
    led = MemoOutcomeEligibility
    statement = select(
        *MemoOutcome.__table__.c,
        MemoSnapshot.id.label("snapshot_id"),
        MemoSnapshot.ticker.label("snapshot_ticker"),
        MemoSnapshot.version.label("snapshot_version"),
        MemoSnapshot.generated_at,
        MemoSnapshot.as_of_date,
        led.eligible.label("track_record_eligible"),
        led.reason.label("eligibility_reason"),
        led.generation_mode.label("generation_mode_observed"),
        led.rating_source.label("rating_source"),
        led.inherited_from_snapshot_id.label("eligibility_inherited_from"),
    ).outerjoin(
        MemoSnapshot, MemoSnapshot.id == MemoOutcome.memo_snapshot_id,
    ).outerjoin(
        led, identity_matches(led, MemoSnapshot),
    ).order_by(MemoOutcome.id)
    with db.no_autoflush:
        rows = [_classify(dict(row)) for row in db.execute(statement).mappings()]
    counts = Counter(row["status"] for row in rows)
    # Kept OUTSIDE `counts`: consumers iterate `counts` as the triage statuses.
    eligibility_reasons = Counter(
        row.get("eligibility_reason") or "unclassified" for row in rows
    )
    track_record_eligibility = {
        "eligible": sum(1 for row in rows if row.get("track_record_eligible") is True),
        "excluded": sum(1 for row in rows if row.get("track_record_eligible") is False),
        "unclassified": sum(1 for row in rows if row.get("track_record_eligible") is None),
        "by_reason": dict(sorted(eligibility_reasons.items())),
        # The W6 receipt check: the dev-copy exclusion compared as an
        # enumerated snapshot-id set, not as a single row count.
        "dev_copy_set": _dev_copy_set_check(rows),
    }
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
        "track_record_eligibility": track_record_eligibility,
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
            "Every stored row is listed whatever its eligibility; the track record counts only "
            "eligible rows (W6). No outcome is changed.",
        ],
        "rows": rows,
    }
