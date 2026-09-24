"""Generic fenced quarantine, in-place audits and restore for FinancialPeriod rows.

Owner decision 2026-09-24 (FIX-006): FMP is the primary fundamentals provider,
and rows that contradict it are quarantined, never deleted. A quarantine is
the same move the September BK repair made: only `ticker` changes, to a
reserved 16-character `~Q` namespace that no ticker-scoped reader can load
(`scorecard_pit.backfill_available_at`, the one unscoped reader, excludes
`~Q` explicitly). Row id and every other field are preserved, and a
`financial_data_repairs` row holds the complete before-image under a digest.

In-place changes the primary provider makes to a row it did not own
(`primary_adoption`) and same-provider value changes (`restatement`) get the
same durable audit, with before- and after-images, so `restore_repair` can
reverse any of the three. Nothing here calls a provider or touches memos,
outcomes or scores.
"""
from __future__ import annotations

import hashlib
import logging
import math
import uuid
from collections import Counter
from datetime import date, datetime
from typing import Any, cast

from sqlalchemy import Date, DateTime, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import FinancialDataRepair, FinancialPeriod
from .bk_fundamental_repair import QUARANTINE_PREFIX, _digest, _snapshot

log = logging.getLogger(__name__)

PRIMARY_PROVIDER = "fmp"
QUARANTINE_KIND = "fmp_primary_takeover"
RESTORE_DISPLACED_KIND = "restore_displaced"
ADOPTION_KIND = "primary_adoption"
RESTATEMENT_KIND = "restatement"
QUARANTINE_KINDS = frozenset({QUARANTINE_KIND, RESTORE_DISPLACED_KIND})
IN_PLACE_KINDS = frozenset({ADOPTION_KIND, RESTATEMENT_KIND})
# The fields a quarantine decision was based on. The move re-checks them in
# the UPDATE's WHERE clause, so a row another process changed after it was
# read cannot be moved on stale evidence (Postgres re-evaluates the WHERE
# against the committed version under READ COMMITTED).
_FENCE_COLUMNS = ("period", "statement", "line_item", "period_end", "fiscal_year", "fiscal_quarter",
                  "value", "currency", "source", "available_at", "available_at_source")


class QuarantineFenceError(RuntimeError):
    """A row changed between being read and being moved; the caller rolls back."""


def _now() -> datetime:
    return datetime.utcnow()


def repair_id_for(audit_key: str | None, kind: str, ticker: str) -> str:
    """Deterministic id under a one-shot ledger, so a resumed pass can tell a
    committed ticker from an interrupted one; random otherwise."""
    if audit_key:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"marketmosaic:{audit_key}:{kind}:{ticker.upper()}"))
    return str(uuid.uuid4())


def namespace_for(repair_id: str, ticker: str) -> str:
    # One namespace per ticker-run: the original (period, statement, line)
    # keys were unique under the ticker, so they stay unique inside it.
    return QUARANTINE_PREFIX + hashlib.sha256(f"{repair_id}:{ticker}".encode()).hexdigest()[:14]


def _fence(row: FinancialPeriod, ticker: str) -> list:
    clauses = [FinancialPeriod.id == row.id, FinancialPeriod.ticker == ticker]
    for name in _FENCE_COLUMNS:
        column = getattr(FinancialPeriod, name)
        value = getattr(row, name)
        if isinstance(value, float) and not math.isfinite(value):
            continue  # NaN never compares equal; the id + other fields still fence it.
        clauses.append(column.is_(None) if value is None else column == value)
    return clauses


def _move(db: Session, row: FinancialPeriod, *, from_ticker: str, to_ticker: str) -> None:
    moved = cast(CursorResult, db.execute(update(FinancialPeriod).where(*_fence(row, from_ticker))
                                          .values(ticker=to_ticker).execution_options(synchronize_session=False)))
    if moved.rowcount != 1:
        raise QuarantineFenceError("financial quarantine fence: row changed concurrently")
    db.expire(row)


def _unique_id(db: Session, repair_id: str) -> str:
    return repair_id if db.get(FinancialDataRepair, repair_id) is None else str(uuid.uuid4())


def quarantine_rows(db: Session, *, ticker: str, actions: list[dict[str, Any]], kind: str = QUARANTINE_KIND,
                    status: str = "applied", mode: str = "supervised", repair_id: str | None = None,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Move rows aside under a fresh reserved namespace with a durable audit. Never deletes.

    `actions`: `{"row": FinancialPeriod, "reason": str, "primary_fact": dict | None,
    "pre_run": snapshot | None}`. `status="planned"` records the plan and moves nothing
    (the unattended guard); `apply_planned` moves it later after review.
    """
    if status not in {"applied", "planned"} or not actions:
        raise ValueError("quarantine needs actions and status applied or planned")
    ticker = ticker.strip().upper()
    repair_id = repair_id or str(uuid.uuid4())
    namespace = namespace_for(repair_id, ticker)
    now = _now()
    plan = {"kind": kind, "ticker": ticker, "provider": PRIMARY_PROVIDER, "namespace": namespace,
            "mode": mode, "created_at": now.isoformat(),
            "actions": [{"id": a["row"].id, "action": "quarantine", "reason": a["reason"],
                         "before": _snapshot(a["row"]), "pre_run": a.get("pre_run"),
                         "primary_fact": a.get("primary_fact")} for a in actions],
            "counts": dict(Counter(a["reason"] for a in actions)),
            "memo_rows_modified": 0, "outcome_rows_modified": 0, **(extra or {})}
    digest = _digest(plan)
    db.add(FinancialDataRepair(id=repair_id, digest=digest, status=status, plan=plan, result={},
                               created_at=now, applied_at=now if status == "applied" else None))
    db.flush()
    if status == "applied":
        for action in actions:
            _move(db, action["row"], from_ticker=ticker, to_ticker=namespace)
        db.flush()
    log.info("financial quarantine %s ticker=%s repair=%s status=%s rows=%d counts=%s",
             kind, ticker, repair_id, status, len(actions), plan["counts"])
    return {"repair_id": repair_id, "namespace": namespace, "digest": digest, "status": status,
            "rows": len(actions), "counts": plan["counts"]}


def record_in_place(db: Session, *, ticker: str, kind: str, entries: list[tuple[FinancialPeriod, dict]],
                    repair_id: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Durable before/after audit for adoptions and restatements already flushed."""
    if kind not in IN_PLACE_KINDS:
        raise ValueError("in-place audit kind must be primary_adoption or restatement")
    if not entries:
        return None
    ticker = ticker.strip().upper()
    repair_id = _unique_id(db, repair_id or str(uuid.uuid4()))
    now = _now()
    plan = {"kind": kind, "ticker": ticker, "provider": PRIMARY_PROVIDER, "created_at": now.isoformat(),
            "actions": [{"id": row.id, "action": kind, "before": before, "after": _snapshot(row)}
                        for row, before in entries],
            "memo_rows_modified": 0, "outcome_rows_modified": 0, **(extra or {})}
    plan["counts"] = {kind: len(entries)}
    digest = _digest(plan)
    db.add(FinancialDataRepair(id=repair_id, digest=digest, status="applied", plan=plan,
                               result={"rows": len(entries)}, created_at=now, applied_at=now))
    db.flush()
    return {"repair_id": repair_id, "digest": digest, "rows": len(entries)}


def plan_kind(repair_id: str, *, db: Session | None = None) -> str | None:
    """The saved plan's kind, or None (unknown id, BK plan, or no table yet)."""
    own = db is None
    db = db or SessionLocal()
    try:
        repair = db.get(FinancialDataRepair, repair_id)
        return (repair.plan or {}).get("kind") if repair is not None else None
    except Exception:
        return None  # The apply route falls through to the BK path and its own errors.
    finally:
        if own:
            db.close()


def _load(db: Session, repair_id: str) -> FinancialDataRepair:
    repair = db.execute(select(FinancialDataRepair).where(FinancialDataRepair.id == repair_id)
                        .with_for_update()).scalar_one_or_none()
    if repair is None:
        raise LookupError("Unknown financial repair")
    if _digest(repair.plan) != repair.digest:
        raise RuntimeError("Repair digest mismatch")
    return repair


def _key(snapshot: dict) -> tuple:
    return snapshot["period"], snapshot["statement"], snapshot["line_item"]


def _without(snapshot: dict, *keys: str) -> dict:
    return {k: v for k, v in snapshot.items() if k not in keys}


def _sqlite_begin(db: Session) -> None:
    # SQLite defers BEGIN until the first write; begin explicitly so the
    # whole restore/apply is one transaction even after only SELECTs.
    connection = db.connection()
    raw = connection.connection.dbapi_connection
    if connection.dialect.name == "sqlite" and raw is not None and not getattr(raw, "in_transaction", True):
        connection.exec_driver_sql("BEGIN")


def apply_planned(repair_id: str, digest: str, *, db: Session | None = None) -> dict[str, Any]:
    """Apply an unattended-mode planned quarantine after review; never calls a provider.

    Every row must still equal its recorded before-image, or nothing moves.
    The primary provider's replacement rows are written by the next refresh.
    """
    own = db is None
    db = db or SessionLocal()
    try:
        _sqlite_begin(db)
        repair = _load(db, repair_id)
        if digest != repair.digest:
            raise RuntimeError("Repair digest mismatch")
        plan = repair.plan
        if plan.get("kind") not in QUARANTINE_KINDS:
            raise ValueError("Not a quarantine plan")
        if repair.status == "applied":
            return {**(repair.result or {}), "repair_id": repair_id, "already_applied": True, "committed": own}
        if repair.status != "planned":
            raise RuntimeError("Repair is not in planned state")
        ticker = plan["ticker"]
        rows = {r.id: r for r in db.execute(select(FinancialPeriod).where(
            FinancialPeriod.id.in_([a["id"] for a in plan["actions"]])).with_for_update()).scalars()}
        for action in plan["actions"]:
            row = rows.get(action["id"])
            if row is None or _snapshot(row) != action["before"]:
                raise QuarantineFenceError("financial quarantine fence: planned row changed since review")
        for action in plan["actions"]:
            _move(db, rows[action["id"]], from_ticker=ticker, to_ticker=plan["namespace"])
        now = _now()
        result = {"repair_id": repair_id, "status": "applied", "rows_quarantined": len(plan["actions"]),
                  "namespace": plan["namespace"], "applied_after_review": True, "committed": own}
        repair.status, repair.applied_at, repair.result = "applied", now, result
        db.flush()
        if own:
            db.commit()
        return result
    except Exception:
        if own:
            db.rollback()
        raise
    finally:
        if own:
            db.close()


def _decode(name: str, value: Any) -> Any:
    column_type = FinancialPeriod.__table__.columns[name].type
    if value is None:
        return None
    if isinstance(column_type, DateTime):
        return datetime.fromisoformat(value)
    if isinstance(column_type, Date):
        return date.fromisoformat(value)
    if name == "value" and isinstance(value, str):
        return float(value)
    return value


def restore_repair(repair_id: str, *, db: Session | None = None) -> dict[str, Any]:
    """Reverse one applied quarantine, adoption or restatement audit.

    Quarantine: every quarantined row must still equal its before-image except
    `ticker`; the rows now occupying the restored keys (the provider
    replacements) are moved into a new `restore_displaced` quarantine with its
    own audit, then the originals return. In-place audits: every row must
    still equal its after-image (a later fetch timestamp is allowed) and is
    set back to its before-image, including a pre-run label (a relabel the
    same run made), after moving that key's occupant aside the same way.
    Any mismatch aborts the whole restore.
    """
    own = db is None
    db = db or SessionLocal()
    try:
        _sqlite_begin(db)
        repair = _load(db, repair_id)
        plan = repair.plan
        kind = plan.get("kind")
        if repair.status == "restored":
            return {**(repair.result or {}), "repair_id": repair_id, "already_restored": True, "committed": own}
        if repair.status != "applied":
            raise RuntimeError("Only an applied repair can be restored")
        ticker = plan["ticker"]
        ids = [a["id"] for a in plan["actions"]]
        rows = {r.id: r for r in db.execute(select(FinancialPeriod).where(FinancialPeriod.id.in_(ids))
                                            .with_for_update()).scalars()}
        restored_by: dict[str, Any] = {"restored_at": _now().isoformat()}
        if kind in QUARANTINE_KINDS:
            namespace = plan["namespace"]
            for action in plan["actions"]:
                row = rows.get(action["id"])
                if (row is None or row.ticker != namespace
                        or _without(_snapshot(row), "ticker") != _without(action["before"], "ticker")):
                    raise QuarantineFenceError("financial quarantine fence: quarantined row changed")
            keys = {(a["before"]["period"], a["before"]["statement"], a["before"]["line_item"]) for a in plan["actions"]}
            occupants = [r for r in db.execute(select(FinancialPeriod).where(FinancialPeriod.ticker == ticker)
                                               .with_for_update()).scalars()
                         if (r.period, r.statement, r.line_item) in keys]
            if occupants:
                displaced = quarantine_rows(db, ticker=ticker, kind=RESTORE_DISPLACED_KIND,
                                            actions=[{"row": r, "reason": "displaced_by_restore"} for r in occupants],
                                            extra={"restores": repair_id})
                restored_by["displaced_repair_id"] = displaced["repair_id"]
                restored_by["displaced_ids"] = sorted(r.id for r in occupants)
            for action in plan["actions"]:
                _move(db, rows[action["id"]], from_ticker=namespace, to_ticker=ticker)
        elif kind in IN_PLACE_KINDS:
            for action in plan["actions"]:
                row = rows.get(action["id"])
                if (row is None or _without(_snapshot(row), "fetched_at") != _without(action["after"], "fetched_at")):
                    raise QuarantineFenceError("financial audit fence: row changed since the audited write")
            # An adoption's before-image predates a relabel the same run made,
            # so restoring it moves the row back to its old label. Whatever
            # now occupies that key (FMP's own row for the old label) is moved
            # aside with its own audit, as a quarantine restore does.
            relabelled = [a for a in plan["actions"] if _key(a["before"]) != _key(_snapshot(rows[a["id"]]))]
            if relabelled:
                keys = {_key(a["before"]) for a in relabelled}
                occupants = [r for r in db.execute(select(FinancialPeriod).where(FinancialPeriod.ticker == ticker)
                                                   .with_for_update()).scalars()
                             if r.id not in rows and (r.period, r.statement, r.line_item) in keys]
                if occupants:
                    displaced = quarantine_rows(db, ticker=ticker, kind=RESTORE_DISPLACED_KIND,
                                                actions=[{"row": r, "reason": "displaced_by_restore"} for r in occupants],
                                                extra={"restores": repair_id})
                    restored_by["displaced_repair_id"] = displaced["repair_id"]
                    restored_by["displaced_ids"] = sorted(r.id for r in occupants)
                # Temporary keys first: restored labels may swap or chain.
                for action in relabelled:
                    rows[action["id"]].period = "~" + uuid.uuid4().hex[:15]
                db.flush()
            for action in plan["actions"]:
                row = rows[action["id"]]
                for name, value in action["before"].items():
                    if name != "id":
                        setattr(row, name, _decode(name, value))
        else:
            raise ValueError("Repair kind cannot be restored here")
        restored_by["rows_restored"] = len(ids)
        repair.status = "restored"
        repair.result = {**(repair.result or {}), "restored_by": restored_by}
        db.flush()
        if own:
            db.commit()
        log.info("financial repair restored kind=%s ticker=%s repair=%s rows=%d", kind, ticker, repair_id, len(ids))
        return {"repair_id": repair_id, "kind": kind, "ticker": ticker, "status": "restored", **restored_by,
                "committed": own}
    except Exception:
        if own:
            db.rollback()
        raise
    finally:
        if own:
            db.close()


# The design names the entry point `restore_quarantine`; it also reverses
# adoptions and restatements (integration critique item 4).
restore_quarantine = restore_repair
