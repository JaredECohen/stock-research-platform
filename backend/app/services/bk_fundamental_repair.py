"""Fenced BK repair from a saved canonical-provider plan, without outcome writes.

Quarantine preserves complete FinancialPeriod rows under a reserved noncompany
namespace. Every original identity and every proposed replacement is durable.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import FinancialDataRepair, FinancialPeriod
from . import fundamental_history_service as history

QUARANTINE_PREFIX = "~Q"
_FIELDS = tuple(column.name for column in FinancialPeriod.__table__.columns)


def _snapshot(row: FinancialPeriod) -> dict:
    result = {}
    for key in _FIELDS:
        value = getattr(row, key)
        if isinstance(value, (date, datetime)):
            value = value.isoformat()
        elif isinstance(value, float) and not math.isfinite(value):
            value = str(value)
        result[key] = value
    return result


def _digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _naive_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("bad_fetched_at must be an exact datetime")
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


def _key(row: dict) -> tuple:
    return row["period"], row["statement"], row["line_item"]


def _identity(row: dict) -> tuple:
    return row["period_end"], row["statement"], row["line_item"], row["fiscal_quarter"] is not None


def _build_plan(rows: list[dict], *, bad_fetched_at: datetime, payload: dict, provider_issues: list[dict], plan_id: str, fetched_at: datetime) -> dict:
    affected = [r for r in rows if r["fetched_at"] == bad_fetched_at.isoformat()]
    if not affected:
        raise ValueError("No BK rows match the exact bad-run timestamp")
    namespace = QUARANTINE_PREFIX + hashlib.sha256(plan_id.encode()).hexdigest()[:14]
    facts = defaultdict(list)
    for statement, lines in history.LINES.items():
        for row in payload[statement]:
            if row.get("source") != "fmp" or row.get("fiscal_label_source") != "provider":
                continue
            fy, fq = history.history._parse_period(row["period"])
            for line in lines:
                if line in row:
                    fact = {"period": row["period"], "period_end": row["period_end"], "fiscal_year": fy,
                            "fiscal_quarter": fq, "statement": statement, "line_item": line,
                            "value": row[line], "currency": row["currency"], "source": "fmp"}
                    facts[_identity(fact)].append(fact)
    affected_ids = {r["id"] for r in affected}
    existing = {_key(r): r for r in rows}
    candidates = {}
    reasons = {}
    for row in affected:
        options = facts.get(_identity(row), [])
        if len(options) != 1:
            reasons[row["id"]] = "no_unique_canonical_fact"
        else:
            candidates[row["id"]] = options[0]
    destinations = defaultdict(list)
    for row in affected:
        if row["id"] in candidates:
            destinations[_key(candidates[row["id"]])].append(row)
    for key, group in destinations.items():
        occupant = existing.get(key)
        if occupant and occupant["id"] not in affected_ids:
            for row in group:
                reasons[row["id"]] = "destination_owned_by_unaffected_row"
            continue
        # An affected exact-key occupant can be retained when its end agrees.
        # Otherwise multiple candidates cannot be ranked from the receipt.
        exact = [r for r in group if _key(r) == key]
        winner = exact[0]["id"] if len(exact) == 1 else (group[0]["id"] if len(group) == 1 else None)
        for row in group:
            if row["id"] != winner:
                reasons[row["id"]] = "duplicate_canonical_destination"
    actions = []
    for row in affected:
        after = dict(row)
        if row["id"] in reasons:
            after["ticker"] = namespace
            action, reason = "quarantine", reasons[row["id"]]
        else:
            after.update(candidates[row["id"]])
            after["fetched_at"] = fetched_at.isoformat()
            action, reason = "restore", "canonical_BK_exact_end_statement_line_cadence"
        actions.append({"id": row["id"], "action": action, "reason": reason, "before": row, "after": after})
    return {"plan_id": plan_id, "ticker": "BK", "bad_fetched_at": bad_fetched_at.isoformat(),
            "provider": "fmp", "provider_symbol": "BK", "canonical_fetched_at": fetched_at.isoformat(),
            "quarantine_namespace": namespace, "affected_ids": sorted(affected_ids),
            "counts": dict(Counter(a["action"] for a in actions)), "actions": actions,
            "destination_fence": [r for r in rows if r["id"] not in affected_ids],
            "provider_issues": provider_issues, "canonical_payload": payload,
            "outcome_rows_modified": 0, "memo_rows_modified": 0}


def prepare_bk_repair(bad_fetched_at: datetime, *, db: Session | None = None) -> dict:
    """Fetch canonical BK once and save a reviewable plan; no financial writes."""
    bad = _naive_utc(bad_fetched_at)
    own = db is None
    db = db or SessionLocal()
    try:
        FinancialDataRepair.__table__.create(db.get_bind(), checkfirst=True)
        rows = [_snapshot(r) for r in db.execute(select(FinancialPeriod).where(FinancialPeriod.ticker == "BK")).scalars()]
        affected = [r for r in rows if r["fetched_at"] == bad.isoformat()]
        if not affected:
            raise ValueError("No BK rows match the exact bad-run timestamp")
        ends = [date.fromisoformat(r["period_end"]) for r in affected if r["period_end"]]
        start = min(ends) if ends else date.today().replace(year=date.today().year - 2)
        if own:
            db.rollback()
        provider = next((p for p in history.get_data_service()._live_chain("financials") if getattr(p, "name", None) == "fmp"), None)
        if provider is None or not callable(getattr(provider, "get_financial_history", None)):
            raise ValueError("Canonical FMP financial history adapter unavailable")
        try:
            raw = provider.get_financial_history("BK", start)
        except Exception as exc:
            raise RuntimeError(f"Canonical provider fetch failed: {type(exc).__name__}") from None
        if not isinstance(raw, dict):
            raise ValueError("Canonical BK provider returned no financial payload")
        issues = []
        payload = history._clean_payload(raw, "fmp", "BK", issues)
        if not all(payload[s] for s in history.LINES):
            raise ValueError("Canonical BK statements incomplete; no repair plan saved")
        plan_id = str(uuid4())
        plan = _build_plan(rows, bad_fetched_at=bad, payload=payload, provider_issues=issues,
                           plan_id=plan_id, fetched_at=datetime.utcnow())
        digest = _digest(plan)
        db.add(FinancialDataRepair(id=plan_id, digest=digest, plan=plan, result={}))
        db.flush()
        if own:
            db.commit()
        return {"plan_id": plan_id, "digest": digest, "status": "planned", "financial_rows_modified": 0,
                "committed": own, "plan": plan}
    except Exception:
        if own:
            db.rollback()
        raise
    finally:
        if own:
            db.close()


def read_bk_repair_plan(plan_id: str, *, db: Session | None = None) -> dict:
    own = db is None
    db = db or SessionLocal()
    try:
        row = db.get(FinancialDataRepair, plan_id)
        if row is None:
            raise LookupError("Unknown financial repair plan")
        return {"plan_id": row.id, "digest": row.digest, "status": row.status, "plan": row.plan, "result": row.result}
    finally:
        if own:
            db.close()


def apply_bk_repair(plan_id: str, digest: str, *, db: Session | None = None) -> dict:
    """Apply the saved exact before-images atomically; never calls a provider."""
    own = db is None
    db = db or SessionLocal()
    try:
        repair = db.execute(select(FinancialDataRepair).where(FinancialDataRepair.id == plan_id).with_for_update()).scalar_one_or_none()
        if repair is None:
            raise LookupError("Unknown financial repair plan")
        if digest != repair.digest or _digest(repair.plan) != repair.digest:
            raise RuntimeError("Repair digest mismatch")
        if repair.status == "applied":
            return {**repair.result, "already_applied": True, "committed": own}
        if repair.status != "planned":
            raise RuntimeError("Repair is not in planned state")
        plan = repair.plan
        current = list(db.execute(select(FinancialPeriod).where(
            FinancialPeriod.ticker.in_(["BK", plan["quarantine_namespace"]])).with_for_update()).scalars())
        by_id = {r.id: r for r in current}
        expected = {r["id"]: r for r in plan["destination_fence"]}
        expected.update({a["id"]: a["before"] for a in plan["actions"]})
        if {r.id: _snapshot(r) for r in current} != expected:
            raise RuntimeError("Financial repair version fence changed; prepare a new plan")
        # Move the entire affected key set aside before any relabel, so unique
        # keys cannot collide transiently. Quarantined rows retain their keys.
        connection = db.connection()
        if connection.dialect.name == "sqlite" and not connection.connection.dbapi_connection.in_transaction:
            connection.exec_driver_sql("BEGIN")
        with db.begin_nested():
            for action in plan["actions"]:
                by_id[action["id"]].ticker = plan["quarantine_namespace"]
            db.flush()
            for action in plan["actions"]:
                row = by_id[action["id"]]
                after = action["after"]
                if action["action"] == "quarantine":
                    continue  # Preserve all original fields except the namespace.
                for key in ("ticker", "period", "fiscal_year", "fiscal_quarter", "value", "currency", "source"):
                    setattr(row, key, after[key])
                row.period_end = date.fromisoformat(after["period_end"]) if after["period_end"] else None
                row.fetched_at = datetime.fromisoformat(after["fetched_at"]) if after["fetched_at"] else None
            db.flush()
            result = {"plan_id": plan_id, "digest": digest, "status": "applied", "success": True,
                      "affected_ids": plan["affected_ids"], "counts": plan["counts"],
                      "quarantine_namespace": plan["quarantine_namespace"], "financial_rows_modified": len(plan["actions"]),
                      "outcome_rows_modified": 0, "memo_rows_modified": 0, "committed": own,
                      "actions": [{**action, "actual_after": _snapshot(by_id[action["id"]])} for action in plan["actions"]]}
            repair.status = "applied"
            repair.applied_at = datetime.utcnow()
            repair.result = result
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
