"""Aggregations over the LLMCallLog audit table (Wave 1A).

Used by the admin endpoint and the CLI cost report. Functions return
plain dicts so they're trivially JSON-serializable.

Wave 8D: USD cost estimation. Token counts are stored at the call
site; this module multiplies them by best-effort price-per-MTok rates
to produce dollar figures. The price table is conservative — when a
specific model isn't found, fall through to a per-provider default,
then to zero. Update `MODEL_PRICES_PER_MTOK` when a provider's
pricing changes.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import LLMCallLog


# Best-effort prices per million tokens (USD), input / output.
# Edit when a provider's official pricing changes; not load-bearing for
# any logic — only the cost estimate output uses these.
MODEL_PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-5":           (3.50, 14.00),
    "gpt-5.4":         (5.00, 20.00),
    "gpt-5.5":         (8.00, 32.00),
    "gpt-5.5-pro":     (15.00, 60.00),
    "gpt-4o-mini":     (0.15, 0.60),
    "gpt-5-mini":      (0.50, 2.00),
    # Anthropic
    "claude-haiku-4-5":  (0.50, 2.50),
    "claude-opus-4-7":   (15.00, 75.00),
    "claude-opus-4-8":   (15.00, 75.00),  # placeholder — update when pricing publishes
    "claude-sonnet-4-6": (3.00, 15.00),
    # Google / Vertex
    "gemini-2.5-flash":  (0.30, 1.25),
    "gemini-2.5-pro":    (3.50, 14.00),
}

# Provider-level fallback (when the specific model isn't tabulated).
PROVIDER_PRICE_FALLBACK: Dict[str, Tuple[float, float]] = {
    "openai":    (3.00, 12.00),
    "anthropic": (3.00, 15.00),
    "gemini":    (3.00, 12.00),
}


def estimate_cost_usd(provider: str, model: str,
                      tokens_in: int, tokens_out: int) -> float:
    """Multiply tokens by per-MTok rates. Best-effort: missing prices
    fall to provider default, then zero. Never raises."""
    p_in, p_out = MODEL_PRICES_PER_MTOK.get(
        (model or "").lower(),
        PROVIDER_PRICE_FALLBACK.get((provider or "").lower(), (0.0, 0.0)),
    )
    cost_in = (max(0, int(tokens_in or 0)) / 1_000_000.0) * p_in
    cost_out = (max(0, int(tokens_out or 0)) / 1_000_000.0) * p_out
    return round(cost_in + cost_out, 6)


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
            "cost_usd": estimate_cost_usd(
                r.provider, r.model, r.tokens_in, r.tokens_out,
            ),
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
            "cost_usd_total": round(sum(c["cost_usd"] for c in calls), 6),
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
        agg: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            a = agg.setdefault(r.agent_name, {
                "n_calls": 0, "tokens_in": 0, "tokens_out": 0,
                "duration_ms_total": 0, "n_failures": 0, "cost_usd": 0.0,
            })
            a["n_calls"] += 1
            a["tokens_in"] += r.tokens_in
            a["tokens_out"] += r.tokens_out
            a["duration_ms_total"] += r.duration_ms
            a["cost_usd"] += estimate_cost_usd(
                r.provider, r.model, r.tokens_in, r.tokens_out,
            )
            if not r.success:
                a["n_failures"] += 1
        for v in agg.values():
            v["cost_usd"] = round(v["cost_usd"], 6)
        return agg
    finally:
        if own:
            db.close()


def cost_per_provider(*, since: Optional[datetime] = None,
                      db: Optional[Session] = None) -> Dict[str, Dict[str, Any]]:
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
        agg: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            a = agg.setdefault(r.provider, {
                "n_calls": 0, "tokens_in": 0, "tokens_out": 0,
                "n_failures": 0, "cost_usd": 0.0,
            })
            a["n_calls"] += 1
            a["tokens_in"] += r.tokens_in
            a["tokens_out"] += r.tokens_out
            a["cost_usd"] += estimate_cost_usd(
                r.provider, r.model, r.tokens_in, r.tokens_out,
            )
            if not r.success:
                a["n_failures"] += 1
        for v in agg.values():
            v["cost_usd"] = round(v["cost_usd"], 6)
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
