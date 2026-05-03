"""Wave 6A — checkpoint store + decorator for resumable memo runs.

Each `run_stock_memo` call carries a `run_id` (Wave 1A). When a major
step (fundamentals fetch, sector research, valuation, …) completes,
its result is JSON-serialized and stored at `(run_id, step_name)`. If
that same run_id is reused (e.g., after a transient failure), the
decorator transparently returns the stored result instead of
recomputing.

TTL: 24 hours by default. A daily monitoring job
(`monitoring/checkpoint_gc.py`) deletes expired rows.

Design points:
- The decorator is *opt-in*. Callers wrap a function with
  `@checkpointed("step_name")` and the runtime takes care of
  serialize/deserialize. Without the decorator the function runs
  normally — no behavior change.
- `run_id` flows via `llm_call_context` (Wave 1A) so the decorator
  can pick it up without changing every step's signature.
- Pydantic models are JSON-friendly via `model_dump_json()`. Plain
  Python types (dict / list / scalar) round-trip via the JSON column.
- Anything that can't be JSON-serialized (e.g. SQLAlchemy rows, custom
  classes without `model_dump`) silently bypasses the checkpoint —
  the decorator falls through to a normal call. This keeps the
  decorator safe to drop on any function, not just JSON-pure ones.
"""
from __future__ import annotations

import functools
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import MemoRunCheckpoint

log = logging.getLogger(__name__)

# Default TTL for checkpoint rows.
DEFAULT_TTL_HOURS = 24

T = TypeVar("T")


def _ensure_table(db: Session) -> None:
    bind = db.get_bind()
    MemoRunCheckpoint.__table__.create(bind=bind, checkfirst=True)


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Public store API
# ---------------------------------------------------------------------------

def save_step(
    run_id: str, step_name: str, *,
    payload: Any, ticker: Optional[str] = None,
    ttl_hours: int = DEFAULT_TTL_HOURS,
    db: Optional[Session] = None,
) -> bool:
    """Persist `payload` for `(run_id, step_name)`. Returns True on success.

    Idempotent on `(run_id, step_name)` — re-saving overwrites the prior
    payload (useful when a step's output evolves mid-run).
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        try:
            blob = _to_json_safe(payload)
        except (TypeError, ValueError) as exc:
            log.debug("checkpoint serialize skipped for %s/%s: %s",
                      run_id, step_name, exc)
            return False
        existing = db.execute(
            select(MemoRunCheckpoint).where(
                MemoRunCheckpoint.run_id == run_id,
                MemoRunCheckpoint.step_name == step_name,
            )
        ).scalar_one_or_none()
        expires = _now() + timedelta(hours=ttl_hours)
        if existing is not None:
            existing.payload = blob
            existing.ticker = ticker or existing.ticker
            existing.generated_at = _now()
            existing.expires_at = expires
        else:
            db.add(MemoRunCheckpoint(
                run_id=run_id, step_name=step_name, ticker=ticker,
                payload=blob, generated_at=_now(), expires_at=expires,
            ))
        db.commit()
        return True
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("checkpoint save failed for %s/%s: %s", run_id, step_name, exc)
        return False
    finally:
        if own:
            db.close()


def load_step(
    run_id: str, step_name: str, *, db: Optional[Session] = None,
) -> Optional[Dict[str, Any]]:
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        row = db.execute(
            select(MemoRunCheckpoint).where(
                MemoRunCheckpoint.run_id == run_id,
                MemoRunCheckpoint.step_name == step_name,
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        if row.expires_at and _now() >= row.expires_at:
            return None
        return row.payload
    finally:
        if own:
            db.close()


def gc_expired(*, db: Optional[Session] = None) -> int:
    """Delete checkpoint rows past their TTL. Returns count removed."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        rows = db.execute(
            select(MemoRunCheckpoint).where(
                MemoRunCheckpoint.expires_at < _now(),
            )
        ).scalars().all()
        for r in rows:
            db.delete(r)
        db.commit()
        return len(rows)
    finally:
        if own:
            db.close()


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def _to_json_safe(value: Any) -> Any:
    """Best-effort JSON-serialize. Pydantic models go through `model_dump`.

    Raises if the value isn't JSON-serializable; the caller falls through
    to a non-checkpointed execution.
    """
    if hasattr(value, "model_dump"):
        out = value.model_dump(mode="json")
        # Round-trip to verify it's actually JSON-clean.
        json.dumps(out)
        return out
    json.dumps(value, default=str)
    return value


def _from_json_safe(value: Any, return_type: Optional[Type] = None) -> Any:
    """Re-hydrate a stored payload into the caller's expected type when
    a Pydantic class is provided. Otherwise returns the raw dict / scalar."""
    if value is None:
        return None
    if return_type is not None and hasattr(return_type, "model_validate"):
        try:
            return return_type.model_validate(value)
        except Exception:
            return value
    return value


def checkpointed(
    step_name: str, *, return_type: Optional[Type] = None,
    ttl_hours: int = DEFAULT_TTL_HOURS,
):
    """Decorator: cache the function's return value under `(run_id, step_name)`.

    `run_id` is read from `agents.llm.llm_call_context` (Wave 1A). When
    no run_id is in scope, the function runs un-checkpointed. The same
    fall-through happens when the return value can't be JSON-serialized,
    or when the underlying store fails — never block a memo on a
    checkpoint hiccup.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            run_id = _current_run_id()
            if not run_id:
                return fn(*args, **kwargs)
            cached = load_step(run_id, step_name)
            if cached is not None:
                hydrated = _from_json_safe(cached, return_type=return_type)
                return hydrated
            result = fn(*args, **kwargs)
            try:
                save_step(run_id, step_name, payload=result, ttl_hours=ttl_hours)
            except Exception as exc:  # pragma: no cover — defensive
                log.debug("checkpoint save failed for %s/%s: %s",
                          run_id, step_name, exc)
            return result
        return wrapper
    return decorator


def _current_run_id() -> Optional[str]:
    """Pull `run_id` out of the active llm_call_context (Wave 1A)."""
    try:
        from ..agents.llm import current_call_context
    except ImportError:
        return None
    try:
        ctx = current_call_context() or {}
    except Exception:
        return None
    rid = ctx.get("run_id") if isinstance(ctx, dict) else None
    return rid or None
