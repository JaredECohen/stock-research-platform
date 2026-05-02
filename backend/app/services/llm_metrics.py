"""Aggregations over the LLMCallLog audit table (Wave 1A).

Used by the admin endpoint and the CLI cost report. Functions return
plain dicts so they're trivially JSON-serializable.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import LLMCallLog


def _ensure_table(db: Session) -> None:
    """Mirror the cache.snapshots pattern — direct callers don't need init_db()."""
    LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)


def cost_per_run(run_id: str, *, db: Optional[Session] = None) -> Dict[str, Any]:
    """Per-call detail + totals for one memo run."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        rows = list(db.execute(
            select(LLMCallLog)
            .where(LLMCallLog.run_id == run_id)
            .order_by(LLMCallLog.generated_at.asc())
        ).scalars().all())
        calls = [{
            "agent_name": r.agent_name,
            "provider": r.provider,
            "model": r.model,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "duration_ms": r.duration_ms,
            "success": r.success,
            "generated_at": r.generated_at.isoformat() if r.generated_at else None,
        } for r in rows]
        return {
            "run_id": run_id,
            "n_calls": len(rows),
            "tokens_in": sum(r.tokens_in for r in rows),
            "tokens_out": sum(r.tokens_out for r in rows),
            "tokens_total": sum(r.tokens_in + r.tokens_out for r in rows),
            "duration_ms_total": sum(r.duration_ms for r in rows),
            "n_failures": sum(1 for r in rows if not r.success),
            "calls": calls,
        }
    finally:
        if own:
            db.close()


def cost_per_agent(*, since: Optional[datetime] = None,
                   db: Optional[Session] = None) -> Dict[str, Dict[str, int]]:
    """Aggregate by agent_name. Returns {agent: {n_calls, tokens, duration_ms_total, n_failures}}."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = select(LLMCallLog)
        if since:
            stmt = stmt.where(LLMCallLog.generated_at >= since)
        rows = list(db.execute(stmt).scalars().all())
        agg: Dict[str, Dict[str, int]] = {}
        for r in rows:
            a = agg.setdefault(r.agent_name, {
                "n_calls": 0, "tokens_in": 0, "tokens_out": 0,
                "duration_ms_total": 0, "n_failures": 0,
            })
            a["n_calls"] += 1
            a["tokens_in"] += r.tokens_in
            a["tokens_out"] += r.tokens_out
            a["duration_ms_total"] += r.duration_ms
            if not r.success:
                a["n_failures"] += 1
        return agg
    finally:
        if own:
            db.close()


def cost_per_provider(*, since: Optional[datetime] = None,
                      db: Optional[Session] = None) -> Dict[str, Dict[str, int]]:
    """Aggregate by provider name."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = select(LLMCallLog)
        if since:
            stmt = stmt.where(LLMCallLog.generated_at >= since)
        rows = list(db.execute(stmt).scalars().all())
        agg: Dict[str, Dict[str, int]] = {}
        for r in rows:
            a = agg.setdefault(r.provider, {
                "n_calls": 0, "tokens_in": 0, "tokens_out": 0, "n_failures": 0,
            })
            a["n_calls"] += 1
            a["tokens_in"] += r.tokens_in
            a["tokens_out"] += r.tokens_out
            if not r.success:
                a["n_failures"] += 1
        return agg
    finally:
        if own:
            db.close()


def slowest_calls(*, since: Optional[datetime] = None, n: int = 20,
                  db: Optional[Session] = None) -> List[Dict[str, Any]]:
    """Top-N slowest calls in the window. Useful for finding pathological prompts."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = select(LLMCallLog).order_by(LLMCallLog.duration_ms.desc()).limit(n)
        if since:
            stmt = stmt.where(LLMCallLog.generated_at >= since)
        rows = list(db.execute(stmt).scalars().all())
        return [{
            "agent_name": r.agent_name,
            "provider": r.provider,
            "model": r.model,
            "duration_ms": r.duration_ms,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "run_id": r.run_id,
            "generated_at": r.generated_at.isoformat() if r.generated_at else None,
        } for r in rows]
    finally:
        if own:
            db.close()


def gc_old(*, max_age_days: int = 90, db: Optional[Session] = None) -> int:
    """Delete rows older than `max_age_days`. Returns count deleted.

    Default 90 days per locked decision in MASTER_PLAN §5. Idempotent.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        rows = db.execute(
            select(LLMCallLog).where(LLMCallLog.generated_at < cutoff)
        ).scalars().all()
        n = len(rows)
        for row in rows:
            db.delete(row)
        db.commit()
        return n
    finally:
        if own:
            db.close()
