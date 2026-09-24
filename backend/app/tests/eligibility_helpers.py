"""Test helpers for the W6 outcome-eligibility ledger.

Eligibility is fail-closed: a snapshot with no ledger row is excluded from the
track record, calibration, attribution, reliability and postmortem selection.
Seeds that reach those readers WITHOUT going through a sweep
(`evaluate_all_due` / `run_postmortems` classify on their own) either call
`classify_all` or `mark` the snapshot explicitly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import MemoOutcomeEligibility, MemoSnapshot
from app.services import outcome_eligibility


def mark(
    db: Session, snapshot_id: int, *, eligible: bool = True,
    reason: str = outcome_eligibility.REASON_LIVE, sector: str | None = None,
    rating_source: str = outcome_eligibility.SOURCE_UNKNOWN,
) -> MemoOutcomeEligibility:
    """Upsert a current-version, identity-valid ledger row for one snapshot.

    An upsert, not an insert: sqlite reuses a deleted max rowid, so a test
    that re-seeds a ticker can meet the previous occupant's row.
    """
    outcome_eligibility.ensure_table(db)
    snap = db.execute(
        select(MemoSnapshot.ticker, MemoSnapshot.generated_at, MemoSnapshot.trigger)
        .where(MemoSnapshot.id == snapshot_id)
    ).one()
    row = db.get(MemoOutcomeEligibility, snapshot_id)
    if row is None:
        row = MemoOutcomeEligibility(memo_snapshot_id=snapshot_id)
        db.add(row)
    values: dict[str, Any] = {
        "ticker": snap.ticker,
        "snapshot_generated_at": snap.generated_at,
        "analysis_generated_at": snap.generated_at,
        "trigger": snap.trigger,
        "inherited_from_snapshot_id": None,
        "eligible": eligible,
        "reason": reason,
        "generation_mode": "live" if eligible else None,
        "rating_source": rating_source,
        "sector": sector,
        "rule_version": outcome_eligibility.RULE_VERSION,
        "classified_at": datetime.utcnow(),
    }
    for key, value in values.items():
        setattr(row, key, value)
    db.commit()
    return row


def classify_all(db: Session) -> dict[str, Any]:
    """Run the real sweep on this session's database."""
    return outcome_eligibility.classify_pending(db=db)


# ---------------------------------------------------------------------------
# Isolated engines for exact-count tests
# ---------------------------------------------------------------------------
#
# The suite shares one database, so a count asserted there is a statement
# about test ordering. Exact-count eligibility / track-record tests build a
# private sqlite file and point every module that opens its own session at it.

def isolated_sessions(tmp_path: Any, monkeypatch: Any, *modules: Any) -> Any:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'w6-isolated.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    for module in modules:
        monkeypatch.setattr(module, "SessionLocal", sessions)
    return sessions, engine


def add_snapshot(
    db: Session, *, id: int | None = None, ticker: str, generated_at: datetime,
    mode: str | None = "live", version: int = 1, trigger: str = "full_reanalysis",
    parent_version: int | None = None, as_of_date: datetime | None = None,
    rating: str = "Bullish", sector: str | None = "Technology",
    memo_generated_at: str | None = None, **memo: Any,
) -> MemoSnapshot:
    body: dict[str, Any] = {"ticker": ticker, "rating_label": rating}
    if sector is not None:
        body["sector"] = sector
    if mode is not None:
        body["generation_mode"] = mode
    if memo_generated_at is not None:
        body["generated_at"] = memo_generated_at
    body.update(memo)
    snap = MemoSnapshot(
        id=id, ticker=ticker, version=version, parent_version=parent_version,
        trigger=trigger, memo_json=body, revision_log=[],
        generated_at=generated_at, as_of_date=as_of_date,
    )
    db.add(snap)
    db.flush()
    return snap


def add_outcome(
    db: Session, snap: MemoSnapshot, *, horizon: int, forward_return: float | None,
    alpha: float | None, rating: str | None = None,
    evaluated_at: datetime | None = None,
) -> Any:
    from datetime import timedelta

    from app.models import MemoOutcome
    from app.services.outcome_service import _thesis_held

    label = rating if rating is not None else str((snap.memo_json or {}).get("rating_label") or "")
    benchmark = None if alpha is None or forward_return is None else forward_return - alpha
    row = MemoOutcome(
        memo_snapshot_id=snap.id, ticker=snap.ticker, rating_at_memo=label,
        confidence_at_memo=60.0, price_at_memo=100.0, horizon_days=horizon,
        evaluated_at=evaluated_at or (snap.generated_at + timedelta(days=horizon + 1)),
        forward_return=forward_return, benchmark_return=benchmark, alpha=alpha,
        thesis_held=_thesis_held(label, forward_return) if forward_return is not None else None,
        note="seeded",
    )
    db.add(row)
    db.flush()
    return row
