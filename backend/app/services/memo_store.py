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
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
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
    # W2a: `section_availability` is computed on the way OUT, from the stored
    # payload, and the presenter replaces hidden prose with a placeholder. A
    # memo carrying it is therefore a presented memo; saving one would make
    # the placeholder text the stored truth and hide those sections from
    # every later rule change. Refuse loudly rather than strip silently, so
    # the caller that round-tripped a presented memo is found, not masked.
    if memo.section_availability:
        raise ValueError(
            f"refusing to save a presented memo for {memo.ticker}: section_availability is "
            "read-time only; save the raw memo, not the output of the presenter"
        )
    own = db is None
    if own:
        db = SessionLocal()
    try:
        # Round-trip through json to make the payload safe for SQLite's JSON
        # column even when fields contain non-serializable types like datetime.
        # The exclude is a backstop behind the refusal above: the stored row
        # never carries the read-time map, even an empty one, so there is
        # no stored value for a reader to mistake for the current verdict.
        # (public_samples._build_memo stores the PRESENTED public copy, map
        # included, on purpose: that row is a rendering, not the memo.)
        memo_payload: dict[str, Any] = json.loads(
            memo.model_dump_json(exclude={"section_availability"})
        )
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


def legacy_case_points(legacy: list[Any]) -> list[str] | None:
    """The exact points of an unambiguous legacy bull/bear list, else None.

    Only two list shapes are unambiguous: all strings, or all
    `{"key_point": <str>}`. Anything else (mixed, null or unknown keys) is
    None so every reader refuses the same shapes `memo_to_pydantic` does.
    """
    if all(isinstance(point, str) for point in legacy):
        return list(legacy)
    if all(isinstance(point, dict) and isinstance(point.get("key_point"), str)
           for point in legacy):
        return [point["key_point"] for point in legacy]
    return None


@dataclass(frozen=True)
class PatchChain:
    """What a news-patch snapshot's lineage says about its fields (W2a).

    `fields`: every field a patch between this snapshot and its base
    replaced (`revision_log[*].fields_patched`). `base`: the first non-patch
    ancestor, read raw, which the PM-view and mispricing rules are evaluated
    on — a patch never re-runs the PM synthesis. `complete=False` means a hop
    was missing, unreadable or unlogged, so the base is unknown and the
    presenter errs toward hiding.
    """
    fields: frozenset[str] = frozenset()
    base: StockMemoOut | None = None
    complete: bool = True


NOT_A_PATCH = PatchChain()
_INCOMPLETE = PatchChain(complete=False)

# (ticker, version) -> projected (trigger, parent_version, revision_log) or
# None, plus ("memo", ticker, version) -> the base memo. One dict per
# request, so a history listing walks each ancestor once.
ChainCache = dict[tuple[Any, ...], Any]


def patch_chain_for(
    snap: MemoSnapshot, *, db: Session | None = None, max_hops: int = 50,
    cache: ChainCache | None = None,
) -> PatchChain:
    """Walk an `incremental_patch` snapshot back to its base.

    Cost: one indexed SELECT per hop, projected to
    `(trigger, parent_version, revision_log)` — never the memo body — plus
    one read of the base body. Only patch snapshots walk at all (pins only,
    at most `MAX_PATCHES_PER_DAY` a day). A database error returns the
    conservative incomplete chain rather than failing the read.

    On a caller's session the walk runs inside a SAVEPOINT: on Postgres a
    failed statement aborts the whole transaction, so swallowing the error
    without rolling back would fail the caller's next statement (the
    commentary cache lookup, the sample upsert, the next history row).
    """
    if getattr(snap, "trigger", None) != "incremental_patch":
        return NOT_A_PATCH
    cache = {} if cache is None else cache
    own = db is None
    session = SessionLocal() if own else db
    assert session is not None
    try:
        if own:
            return _walk_chain(session, snap, max_hops, cache)
        with session.begin_nested():
            return _walk_chain(session, snap, max_hops, cache)
    except SQLAlchemyError as exc:
        log.warning("memo patch chain walk failed for %s v%s: %s",
                    snap.ticker, snap.version, type(exc).__name__)
        return _INCOMPLETE
    finally:
        if own:
            session.close()


def _walk_chain(session: Session, snap: MemoSnapshot, max_hops: int, cache: ChainCache) -> PatchChain:
    fields: set[str] = set()
    ticker = snap.ticker
    trigger, parent, log_entries = snap.trigger, snap.parent_version, snap.revision_log
    for _ in range(max_hops):
        patched = [
            entry.get("fields_patched") for entry in (log_entries or [])
            if isinstance(entry, dict) and isinstance(entry.get("fields_patched"), list)
        ]
        if not patched:
            # Patches written before fields were logged (a59ff56): which
            # fields this hop changed is unknown, so it credits none.
            return PatchChain(frozenset(fields), None, False)
        for names in patched:
            fields.update(str(n) for n in names or [])
        if parent is None:
            return PatchChain(frozenset(fields), None, False)
        key = (ticker, parent)
        if key not in cache:
            row = session.execute(
                select(MemoSnapshot.trigger, MemoSnapshot.parent_version,
                       MemoSnapshot.revision_log)
                .where(MemoSnapshot.ticker == ticker, MemoSnapshot.version == parent)
            ).first()
            cache[key] = tuple(row) if row is not None else None
        hop = cache[key]
        if hop is None:
            return PatchChain(frozenset(fields), None, False)
        trigger, next_parent, log_entries = hop
        if trigger != "incremental_patch":
            base = _base_memo(session, ticker, parent, cache)
            if base is None:
                return PatchChain(frozenset(fields), None, False)
            return PatchChain(frozenset(fields), base, True)
        parent = next_parent
    return PatchChain(frozenset(fields), None, False)


def _base_memo(session: Session, ticker: str, version: int, cache: ChainCache) -> StockMemoOut | None:
    key = ("memo", ticker, version)
    if key not in cache:
        row = session.execute(
            select(MemoSnapshot).where(MemoSnapshot.ticker == ticker, MemoSnapshot.version == version)
        ).scalar_one_or_none()
        memo: StockMemoOut | None = None
        if row is not None:
            try:
                memo = memo_to_pydantic(row)
            except StoredMemoUnreadable:
                memo = None
        cache[key] = memo
    return cache[key]


def present_snapshot(
    snap: MemoSnapshot, *, db: Session | None = None, cache: ChainCache | None = None,
) -> StockMemoOut:
    """The customer-facing memo for a stored snapshot (W2a).

    `memo_to_pydantic` then the presenter, with the patch chain resolved.
    Raises `StoredMemoUnreadable` exactly as `memo_to_pydantic` does. The
    snapshot row is only read, never written.
    """
    from . import memo_sections
    memo = memo_to_pydantic(snap)
    chain = patch_chain_for(snap, db=db, cache=cache)
    return memo_sections.present_memo(
        memo, patched_fields=chain.fields, base=chain.base, chain_complete=chain.complete,
    )


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
        points = legacy_case_points(legacy)
        if points is None:
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
