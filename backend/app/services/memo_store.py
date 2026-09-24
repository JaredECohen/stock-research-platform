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
from copy import deepcopy
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import MemoSnapshot
from ..schemas import StockMemoOut

log = logging.getLogger(__name__)


class StoredMemoUnreadable(ValueError):
    """A stored snapshot that no longer validates as `StockMemoOut`.

    Carries the row's identity and the failing field paths only, never the
    memo body, because the message reaches API errors, chat-tool output and
    loop notes. A ValueError (as pydantic's ValidationError is) so any
    existing `except ValueError` keeps catching it. The snapshot is left
    exactly as stored: ambiguous legacy shapes are refused, not coerced.
    """

    def __init__(self, *, ticker: str, version: int, snapshot_id: int | None,
                 fields: tuple[str, ...]) -> None:
        self.ticker = ticker
        self.version = version
        self.snapshot_id = snapshot_id
        self.fields = fields
        super().__init__(
            f"stored memo {ticker} v{version} (snapshot {snapshot_id}) does not validate at "
            + ", ".join(fields)
        )

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
    parent_version: int | None = None,
    revision_log: list[dict[str, Any]] | None = None,
    as_of_date: Any | None = None,
    db: Session | None = None,
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
    # Assignment/model_copy can bypass pydantic validation. A serializable
    # object is not necessarily a readable memo; validate before any DB work.
    memo = StockMemoOut.model_validate(memo.model_dump(mode="python", warnings=False))
    own = db is None
    if own:
        db = SessionLocal()
    try:
        # Round-trip through json to make the payload safe for SQLite's JSON
        # column even when fields contain non-serializable types like datetime.
        memo_payload: dict[str, Any] = json.loads(memo.model_dump_json())
        from .regen_lease import LeaseLost, assert_current
        job = assert_current(db=db, lock=True)
        if job is not None:
            if job.ticker != memo.ticker.upper():
                raise LeaseLost(f"Regeneration job {job.id} cannot publish another ticker")
            if job.memo_version is not None:
                # The job's publication receipt makes retries idempotent even
                # when graph return/worker completion was interrupted.
                prior = db.execute(select(MemoSnapshot).where(
                    MemoSnapshot.ticker == job.ticker, MemoSnapshot.version == job.memo_version,
                )).scalar_one()
                db.expunge(prior)
                return prior
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
        db.flush()
        if job is not None:
            job.memo_version = snap.version
            db.flush()
        if own:
            db.commit()
            db.refresh(snap)
        db.expunge(snap)
        return snap
    finally:
        if own:
            db.close()


def memo_version(ticker: str, version: int) -> MemoSnapshot | None:
    """Read an exact live snapshot; never substitute the latest or generate."""
    with SessionLocal() as db:
        _ensure_table(db)
        return db.execute(select(MemoSnapshot).where(
            MemoSnapshot.ticker == ticker.upper(),
            MemoSnapshot.version == version,
            MemoSnapshot.as_of_date.is_(None),
        )).scalar_one_or_none()


def latest_memo(
    ticker: str, *,
    include_backtests: bool = False,
    db: Session | None = None,
) -> MemoSnapshot | None:
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
    memo: MemoSnapshot, *, db: Session | None = None,
) -> dict[str, Any]:
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
    ticker: str, *, limit: int = 50, db: Session | None = None,
) -> list[MemoSnapshot]:
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


def _validate_stored(snap: MemoSnapshot, payload: Any) -> StockMemoOut:
    try:
        return StockMemoOut.model_validate(payload)
    except ValidationError as exc:
        # Paths only: `include_input=False` keeps stored memo text out of the error.
        fields = tuple(dict.fromkeys(
            ".".join(str(part) for part in err["loc"]) or "<root>"
            for err in exc.errors(include_url=False, include_input=False, include_context=False)
        ))
        raise StoredMemoUnreadable(
            ticker=snap.ticker, version=snap.version, snapshot_id=snap.id, fields=fields,
        ) from exc


def memo_to_pydantic(snap: MemoSnapshot) -> StockMemoOut:
    """Read a snapshot, projecting only unambiguous legacy case lists.

    This adapter never writes to the snapshot and is deliberately absent from
    new-publication validation. An absent legacy headline remains empty; no
    research narrative is inferred. The complete original case is retained in
    visible degradation metadata alongside its source snapshot identity.
    A snapshot that still does not validate raises StoredMemoUnreadable
    (identity and field paths, no body).
    """
    payload = deepcopy(snap.memo_json)
    if not isinstance(payload, dict):
        return _validate_stored(snap, payload)
    for field in ("bull_case", "bear_case"):
        legacy = payload.get(field)
        if not isinstance(legacy, list):
            continue
        if all(isinstance(point, str) for point in legacy):
            points = list(legacy)
        elif all(isinstance(point, dict) and isinstance(point.get("key_point"), str)
                 for point in legacy):
            points = [point["key_point"] for point in legacy]
        else:
            # Ambiguous shapes are refused, never coerced: validation below
            # raises StoredMemoUnreadable and the row stays as stored.
            continue
        payload[field] = {"headline": "", "key_points": points}
        agent = "Stored memo compatibility"
        degraded = payload.setdefault("degraded_agents", [])
        if agent not in degraded:
            degraded.append(agent)
        payload.setdefault("degradation_events", []).append({
            "agent": agent,
            "error_type": "LegacyCaseShape",
            "message": f"{field} was a legacy list; exact points retained, headline unavailable.",
            "field": field,
            "source_snapshot_id": snap.id,
            "source_snapshot_version": snap.version,
            "source_snapshot_ticker": snap.ticker,
            "original_value": legacy,
        })
    return _validate_stored(snap, payload)
