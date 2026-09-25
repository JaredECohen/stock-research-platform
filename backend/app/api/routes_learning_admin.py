"""W7 learning ledger — the operator's control surface (seven routes).

Everything here sits under `/api/admin/*`, so `admin_auth`'s prefix
middleware requires the bearer `ADMIN_API_TOKEN` the moment it is mounted,
and `auth/policy.py` classifies it `admin` by the same rule. None of these is
browser-called, so none is in `admin_auth.EXEMPT_PREFIXES`.

* **status** — ceiling, DB mode, effective mode, the promotion gates, counts
  and the last ten control events.
* **mode** — append a mode event. Inject is refused (409) until the gates
  are green unless `force` is set, and a forced promotion is recorded.
* **items / items/{id}/status** — the ledger with posteriors, and the
  per-lesson kill switch (suppress / retire / reactivate); history is
  appended, never overwritten.
* **renders** — the audit trail: "why did the agent see this?".
* **preview** — what the ledger holds for a ticker, ranked, with posteriors.
  Writes nothing, so priors can be evaluated without generating a memo.
* **corpus-inventory** — the S17 census (aggregate SQL; never selects
  embedding values).

All reads are bounded (`limit <= 200`); nothing here calls a provider or a
model.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ..database import SessionLocal
from ..learning import control, ledger
from ..models import LearningEvidence, LearningItem, LearningRender

router = APIRouter()

MAX_LIMIT = 200
EVIDENCE_PER_ITEM = 10


class LearningModeRequest(BaseModel):
    mode: Literal["off", "shadow", "inject"]
    reason: str = Field(min_length=1, max_length=2000)
    force: bool = False


class LearningItemStatusRequest(BaseModel):
    status: Literal["active", "suppressed", "retired"]
    reason: str = Field(min_length=1, max_length=500)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@router.get("/api/admin/learning/status")
def learning_status() -> dict[str, Any]:
    now = datetime.utcnow()
    since = now - timedelta(days=30)
    with SessionLocal() as db:
        try:
            db_mode: str | None = control.db_mode(db)
        except ValueError:
            db_mode = None  # an unknown stored mode; effective_mode fails closed
        items: dict[str, dict[str, int]] = {"kind": {}, "status": {}, "scope_type": {}}
        for kind, status, scope_type, n in db.execute(
            select(LearningItem.kind, LearningItem.status, LearningItem.scope_type, func.count())
            .group_by(LearningItem.kind, LearningItem.status, LearningItem.scope_type)
        ).all():
            for key, value in (("kind", kind), ("status", status), ("scope_type", scope_type)):
                items[key][value] = items[key].get(value, 0) + int(n)
        evidence = {v: int(n) for v, n in db.execute(
            select(LearningEvidence.verdict, func.count()).group_by(LearningEvidence.verdict)
        ).all()}
        renders: dict[str, dict[str, int]] = {"mode": {}, "consumer": {}}
        considered: Counter[str] = Counter()
        for mode, consumer, n in db.execute(
            select(LearningRender.mode, LearningRender.consumer, func.count())
            .where(LearningRender.created_at >= since)
            .group_by(LearningRender.mode, LearningRender.consumer)
        ).all():
            renders["mode"][mode] = renders["mode"].get(mode, 0) + int(n)
            renders["consumer"][consumer] = renders["consumer"].get(consumer, 0) + int(n)
        for (rows,) in db.execute(
            select(LearningRender.considered).where(
                LearningRender.created_at >= since, LearningRender.considered.is_not(None),
            ).limit(5000)
        ).all():
            for entry in rows or []:
                if isinstance(entry, dict) and entry.get("use") in ("applied", "contradicted"):
                    considered[str(entry["use"])] += 1
        gates = control.gates(now=now, db=db)
        events = control.recent_events(db, limit=10)
        epoch = control.epoch(db)
    return {
        "ceiling": control.ceiling(),
        "db_mode": db_mode,
        "effective_mode": control.effective_mode(),
        "ledger_epoch": _iso(epoch),
        "gates": gates,
        "counts": {
            "items": items,
            "evidence": evidence,
            "renders_30d": renders,
            "considered_30d": {"applied": considered["applied"], "contradicted": considered["contradicted"]},
        },
        "recent_events": events,
    }


@router.post("/api/admin/learning/mode")
def learning_set_mode(body: LearningModeRequest) -> Any:
    try:
        return control.set_mode(body.mode, reason=body.reason, actor="admin", force=body.force)
    except control.GatesNotMet as exc:
        return JSONResponse(status_code=409, content={"error": "gates_not_met", "gates": exc.gates})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.get("/api/admin/learning/items")
def learning_items(
    kind: Literal["lesson", "observation"] | None = None,
    scope_type: Literal["company", "industry_group", "sector"] | None = None,
    scope_key: str | None = Query(None, max_length=64),
    status: Literal["active", "suppressed", "retired", "superseded"] | None = None,
    limit: int = Query(50, ge=1, le=MAX_LIMIT),
) -> dict[str, Any]:
    now = datetime.utcnow()
    stmt = select(LearningItem)
    if kind:
        stmt = stmt.where(LearningItem.kind == kind)
    if scope_type:
        stmt = stmt.where(LearningItem.scope_type == scope_type)
    if scope_key:
        # Company keys are tickers (stored upper-case); group codes and
        # sector slugs are matched exactly.
        key = scope_key.strip().upper() if scope_type == "company" else scope_key.strip()
        stmt = stmt.where(LearningItem.scope_key == key)
    if status:
        stmt = stmt.where(LearningItem.status == status)
    with SessionLocal() as db:
        rows = db.execute(stmt.order_by(LearningItem.id.desc()).limit(limit)).scalars().all()
        post = ledger.posteriors(db, [r.id for r in rows if r.kind == "lesson"], now=now)
        evidence: dict[int, list[dict[str, Any]]] = {r.id: [] for r in rows}
        if rows:
            for ev in db.execute(
                select(LearningEvidence).where(LearningEvidence.item_id.in_(list(evidence)))
                .order_by(LearningEvidence.id.desc())
            ).scalars().all():
                bucket = evidence[ev.item_id]
                if len(bucket) < EVIDENCE_PER_ITEM:
                    bucket.append({
                        "verdict": ev.verdict, "applies": ev.applies, "ticker": ev.ticker,
                        "memo_snapshot_id": ev.memo_snapshot_id, "horizon_days": ev.horizon_days,
                        "alpha": ev.alpha, "rationale": ev.rationale, "observed_at": _iso(ev.observed_at),
                    })
        items = [{**ledger.item_dict(r, post.get(r.id)), "evidence": evidence[r.id]} for r in rows]
    return {"count": len(items), "limit": limit, "items": items}


@router.post("/api/admin/learning/items/{item_id}/status")
def learning_item_status(item_id: int, body: LearningItemStatusRequest) -> Any:
    try:
        return ledger.set_status(item_id, body.status, reason=body.reason, actor="admin")
    except LookupError:
        raise HTTPException(status_code=404, detail="learning item not found") from None
    except ledger.ItemSuperseded:
        return JSONResponse(status_code=409, content={
            "error": "item_superseded",
            "detail": "a superseded item was replaced by a newer one; change that one instead",
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.get("/api/admin/learning/renders")
def learning_renders(
    ticker: str | None = Query(None, max_length=16),
    run_id: str | None = Query(None, max_length=64),
    memo_snapshot_id: int | None = None,
    mode: Literal["shadow", "inject"] | None = None,
    limit: int = Query(50, ge=1, le=MAX_LIMIT),
) -> dict[str, Any]:
    stmt = select(LearningRender)
    if ticker:
        stmt = stmt.where(LearningRender.ticker == ticker.strip().upper())
    if run_id:
        stmt = stmt.where(LearningRender.run_id == run_id)
    if memo_snapshot_id is not None:
        stmt = stmt.where(LearningRender.memo_snapshot_id == memo_snapshot_id)
    if mode:
        stmt = stmt.where(LearningRender.mode == mode)
    with SessionLocal() as db:
        rows = db.execute(stmt.order_by(LearningRender.id.desc()).limit(limit)).scalars().all()
        renders = [{
            "id": r.id, "run_id": r.run_id, "consumer": r.consumer, "ticker": r.ticker, "mode": r.mode,
            "memo_snapshot_id": r.memo_snapshot_id, "chars": r.chars, "items": r.items or [],
            "dropped": r.dropped or [], "considered": r.considered, "error_type": r.error_type,
            "created_at": _iso(r.created_at),
        } for r in rows]
    return {"count": len(renders), "limit": limit, "renders": renders}


@router.get("/api/admin/learning/preview")
def learning_preview(
    ticker: str = Query(..., min_length=1, max_length=16),
    consumer: Literal["pm_memo", "sector", "industry_group", "critic"] = "pm_memo",
) -> dict[str, Any]:
    try:
        out = ledger.preview(ticker, consumer=consumer)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    out["effective_mode"] = control.effective_mode()
    return out


@router.get("/api/admin/corpus-inventory")
def corpus_inventory() -> dict[str, Any]:
    from ..services import corpus_inventory as census
    return census.build_inventory()
