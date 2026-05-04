"""Versioned memo persistence (Phase F).

Each `run_stock_memo` invocation creates a new `MemoSnapshot` row tagged with
a `trigger` (`full_reanalysis`, `incremental_patch`, `first_run`, …) and an
optional `parent_version` so an incremental patch chains off the previous
version. This decouples memo *history* from the legacy single-row-per-ticker
`StockMemo` table, which we keep in place for back-compat.

Why a separate table:
- `StockMemo` (legacy) is upsert-ish; readers see only the latest. We need
  the full timeline so the UI can show "memo updated 2 days ago because of
  Q1 2026 earnings" and the reflection layer can compare across versions.
- A patch contract requires lineage (`parent_version`) — `StockMemo` has no
  notion of that.

The functions here are intentionally thin so callers (graph, future update
orchestrator, news-impact-agent) can compose them.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import MemoSnapshot
from ..schemas import StockMemoOut

log = logging.getLogger(__name__)

# Allowed triggers — kept here as the source of truth so callers don't pass
# free-form strings the UI can't reason about.
TRIGGERS = {
    "first_run",            # ticker analyzed for the first time
    "full_reanalysis",      # filing / earnings refresh
    "incremental_patch",    # news_impact_agent decided news is material
    "force_refresh",        # explicit user-driven refresh
    "scheduled",            # background job, e.g. quarterly refresh
}


def _ensure_table(db: Session) -> None:
    """Create the memo_snapshots table if missing.

    Mirrors the lazy-create pattern used by `app.cache.snapshots`. Lets
    direct-import callers (tests, scripts) hit memo_store without first
    going through the FastAPI startup hook that calls `init_db()`.
    """
    bind = db.get_bind()
    MemoSnapshot.__table__.create(bind=bind, checkfirst=True)


def _next_version(db: Session, ticker: str) -> int:
    _ensure_table(db)
    row = db.execute(
        select(MemoSnapshot.version)
        .where(MemoSnapshot.ticker == ticker)
        .order_by(MemoSnapshot.version.desc())
        .limit(1)
    ).first()
    return (row[0] + 1) if row and row[0] is not None else 1


def save_memo(
    memo: StockMemoOut,
    *,
    trigger: str = "full_reanalysis",
    parent_version: Optional[int] = None,
    revision_log: Optional[List[Dict[str, Any]]] = None,
    as_of_date: Optional[Any] = None,
    db: Optional[Session] = None,
) -> MemoSnapshot:
    """Persist a memo as a new version. Returns the inserted snapshot row.

    `as_of_date` (Wave 1C) marks a memo as a backtest reproduction —
    distinct from `generated_at`. When set, the memo is excluded from
    the default `latest_memo` lookup (callers explicitly opt in via
    `latest_memo(..., as_of=...)`).

    `revision_log` lets callers attach structured context about *what changed*
    in this version (e.g., for a patch: which fields the news_impact_agent
    edited and why). When a new full reanalysis lands, the log is reset to
    a single "full_reanalysis" entry so the chain stays interpretable.
    """
    if trigger not in TRIGGERS:
        raise ValueError(f"unknown trigger: {trigger!r}; allowed: {sorted(TRIGGERS)}")
    own = db is None
    if own:
        db = SessionLocal()
    try:
        # Round-trip through json to make the payload safe for SQLite's JSON
        # column even when fields contain non-serializable types like datetime.
        memo_payload: Dict[str, Any] = json.loads(memo.model_dump_json())
        version = _next_version(db, memo.ticker)
        # Coerce date → datetime for SQLite (DateTime column).
        as_of_dt = None
        if as_of_date is not None:
            from datetime import date as _date
            as_of_dt = (
                datetime.combine(as_of_date, datetime.min.time())
                if isinstance(as_of_date, _date) and not isinstance(as_of_date, datetime)
                else as_of_date
            )
        snap = MemoSnapshot(
            ticker=memo.ticker,
            version=version,
            parent_version=parent_version,
            trigger=trigger,
            memo_json=memo_payload,
            revision_log=list(revision_log or [
                {
                    "version": version,
                    "trigger": trigger,
                    "at": datetime.utcnow().isoformat(),
                    "as_of_date": as_of_dt.isoformat() if as_of_dt else None,
                }
            ]),
            as_of_date=as_of_dt,
        )
        db.add(snap)
        db.commit()
        db.refresh(snap)
        db.expunge(snap)
        return snap
    finally:
        if own:
            db.close()


def latest_memo(
    ticker: str, *,
    include_backtests: bool = False,
    db: Optional[Session] = None,
) -> Optional[MemoSnapshot]:
    """Return the highest-version snapshot for `ticker`, or None.

    By default, backtest snapshots (those with `as_of_date` set) are
    excluded — callers asking for "the latest memo" want the live one.
    Pass `include_backtests=True` to consider every version regardless
    of mode.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = (
            select(MemoSnapshot)
            .where(MemoSnapshot.ticker == ticker.upper())
            .order_by(MemoSnapshot.version.desc())
            .limit(1)
        )
        if not include_backtests:
            stmt = stmt.where(MemoSnapshot.as_of_date.is_(None))
        snap = db.execute(stmt).scalars().first()
        if snap is None:
            return None
        db.expunge(snap)
        return snap
    finally:
        if own:
            db.close()


def memo_freshness(
    memo: MemoSnapshot, *, db: Optional[Session] = None,
) -> Dict[str, Any]:
    """Return staleness verdict for `memo` (Wave 9b Phase 2d).

    A memo is considered stale when a 10-Q / 10-K / 8-K filing has been
    posted (`filing_date`) or a quarterly earnings call held
    (`call_date`) after the memo was generated. The user-facing
    "Re-run research" button bypasses this check; this function exists
    for the auto-refresh path on memo reads.

    Output:
        {
          "stale": bool,
          "reason": str,           # human-readable trigger label
          "trigger": Optional[str] # "new_filing" | "new_transcript" | None
        }
    """
    from ..models import EarningsTranscript, FilingDoc
    own = db is None
    if own:
        db = SessionLocal()
    try:
        cutoff = memo.generated_at
        cutoff_date = cutoff.date() if hasattr(cutoff, "date") else cutoff
        latest_filing = db.execute(
            select(FilingDoc.filing_date, FilingDoc.filing_type, FilingDoc.accession_number)
            .where(
                FilingDoc.ticker == memo.ticker,
                FilingDoc.filing_date.is_not(None),
                FilingDoc.filing_date > cutoff_date,
            )
            .order_by(FilingDoc.filing_date.desc())
            .limit(1)
        ).first()
        if latest_filing:
            d, ftype, acc = latest_filing
            return {
                "stale": True,
                "reason": f"new {ftype} on {d.isoformat()}",
                "trigger": "new_filing",
                "trigger_id": acc,
            }
        latest_transcript = db.execute(
            select(EarningsTranscript.call_date, EarningsTranscript.period)
            .where(
                EarningsTranscript.ticker == memo.ticker,
                EarningsTranscript.call_date.is_not(None),
                EarningsTranscript.call_date > cutoff_date,
            )
            .order_by(EarningsTranscript.call_date.desc())
            .limit(1)
        ).first()
        if latest_transcript:
            d, period = latest_transcript
            return {
                "stale": True,
                "reason": f"new transcript for {period} on {d.isoformat()}",
                "trigger": "new_transcript",
                "trigger_id": period,
            }
        return {"stale": False, "reason": "", "trigger": None, "trigger_id": None}
    finally:
        if own:
            db.close()


def memo_history(
    ticker: str, *, limit: int = 50, db: Optional[Session] = None,
) -> List[MemoSnapshot]:
    """Return the timeline of memo versions for `ticker`, newest first."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        rows = list(
            db.execute(
                select(MemoSnapshot)
                .where(MemoSnapshot.ticker == ticker.upper())
                .order_by(MemoSnapshot.version.desc())
                .limit(limit)
            ).scalars().all()
        )
        for r in rows:
            db.expunge(r)
        return rows
    finally:
        if own:
            db.close()


def memo_to_pydantic(snap: MemoSnapshot) -> StockMemoOut:
    """Re-hydrate a stored snapshot back into the pydantic model."""
    return StockMemoOut.model_validate(snap.memo_json)
