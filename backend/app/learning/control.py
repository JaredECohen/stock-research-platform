"""W7 learning mode control: off | shadow | inject.

The effective mode is ``min(settings.learning_mode_max, DB mode)``:

* the env ceiling (`LEARNING_MODE_MAX`) is the deploy-time upper bound and
  the emergency stop — "off" means no DB read at all, so prompts are
  byte-identical to the pre-ledger code;
* the DB mode is the latest ``learning_control_events`` mode row, or
  "shadow" when there is none. It is the only thing a promotion changes, and
  it lives in Postgres because web and worker are separate processes: a
  module-level flag set on one would be invisible to the other (CLAUDE.md,
  "Cross-process state belongs in the database").

The DB read is cached for 60 s per process. That cache is a cache OF the
database, never a source of truth: a demotion reaches both services within a
minute, and ``set_mode`` clears the local copy at once. Any failure reading
the DB fails CLOSED to "off" (prompts revert to today's), never open.

Promotion to inject is gated (``gates``) and manual; ``force`` exists for the
owner and is recorded on the event. Demotion is automatic on contamination
(``ledger.integrity_check``, K1).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal
from ..models import LearningControlEvent, LearningItem, LearningRender
from ..services import outcome_eligibility

log = logging.getLogger(__name__)

MODES = ("off", "shadow", "inject")
_RANK = {m: i for i, m in enumerate(MODES)}
DEFAULT_DB_MODE = "shadow"
CACHE_SECONDS = 60
EPOCH_REASON = "ledger_epoch"

# Promotion gates (design §6.1 with the S18 critique: G2 counts only renders
# linked to a memo snapshot, G3 is defined on chars and error_type).
MIN_SHADOW_RUNS = 10
MIN_SHADOW_TICKERS = 3
MIN_SOAK_DAYS = 7
# Character budgets per consumer (header and footer included). The renderer
# (S19, `learning/context.py`) packs whole lines under these; G3 checks that
# no audited render exceeded its budget.
RENDER_BUDGETS: dict[str, int] = {"pm_memo": 1500, "sector": 900, "industry_group": 700, "critic": 700}
POSTMORTEM_ORIGINS = ("postmortem", "postmortem_sector", "postmortem_backfill")


class GatesNotMet(Exception):
    """Promotion to inject refused; ``gates`` says which gate is red."""

    def __init__(self, gates: dict[str, Any]) -> None:
        super().__init__("learning promotion gates not met")
        self.gates = gates


# Monotonic clock, injectable so the TTL is tested deterministically.
_clock: Callable[[], float] = time.monotonic
_cache: dict[str, Any] = {}


def _cache_clear() -> None:
    _cache.clear()


def mode_events():
    """Control events that set a mode (the epoch marker is not one)."""
    return select(LearningControlEvent).where(LearningControlEvent.reason != EPOCH_REASON)


def db_mode(db: Session) -> str:
    """The latest mode event's mode, or "shadow" when none exists."""
    row = db.execute(
        mode_events().order_by(LearningControlEvent.created_at.desc(), LearningControlEvent.id.desc()).limit(1)
    ).scalars().first()
    if row is None:
        return DEFAULT_DB_MODE
    if row.mode not in _RANK:
        # An unknown stored value is not guessed at: the caller fails closed.
        raise ValueError(f"unknown learning mode in control event {row.id}")
    return row.mode


def _cached_db_mode() -> str:
    now = _clock()
    hit = _cache.get("db_mode")
    if hit is not None and now - hit[1] < CACHE_SECONDS:
        return str(hit[0])
    with SessionLocal() as db:
        mode = db_mode(db)
    _cache["db_mode"] = (mode, now)
    return mode


def ceiling() -> str:
    value = str(settings.learning_mode_max or "off").strip().lower()
    return value if value in _RANK else "off"


def effective_mode() -> str:
    top = ceiling()
    if top == "off":
        return "off"  # no DB read: the CI/dev default and the emergency stop
    try:
        db = _cached_db_mode()
    except Exception as exc:
        # Fail closed: prompts revert to today's. Type only — never the URL.
        log.warning("learning control: DB mode unreadable (%s); effective mode is off", type(exc).__name__)
        return "off"
    return min(top, db, key=_RANK.__getitem__)


def _latest_mode_event(db: Session) -> LearningControlEvent | None:
    return db.execute(
        mode_events().order_by(LearningControlEvent.created_at.desc(), LearningControlEvent.id.desc()).limit(1)
    ).scalars().first()


def gates(*, now: datetime | None = None, db: Session | None = None) -> dict[str, Any]:
    """The four promotion gates, evaluated live from the database."""
    now = now or datetime.utcnow()
    if db is None:
        with SessionLocal() as own:
            return gates(now=now, db=own)
    # G1 integrity: K1 scoping — only outcome-derived items have an origin
    # snapshot to be contaminated. Filing observations never block it.
    contaminated = int(db.execute(
        select(func.count(LearningItem.id)).where(
            LearningItem.status == "active",
            LearningItem.origin_kind.in_(POSTMORTEM_ORIGINS),
            LearningItem.origin_snapshot_id.is_not(None),
            ~outcome_eligibility.eligible_exists(LearningItem.origin_snapshot_id),
        )
    ).scalar_one())
    g1 = {"ok": contaminated == 0, "contaminated_active_items": contaminated}

    since = soak_window_start(db)
    soak_days = (now - since).total_seconds() / 86400 if since is not None else None
    # Projected columns only: the audit's item lists are not needed here.
    renders = select(
        LearningRender.id, LearningRender.consumer, LearningRender.mode, LearningRender.memo_snapshot_id,
        LearningRender.run_id, LearningRender.ticker, LearningRender.chars, LearningRender.error_type,
    )
    if since is not None:
        renders = renders.where(LearningRender.created_at >= since)
    rows = db.execute(renders.where(LearningRender.created_at <= now)).all()
    linked = [
        r for r in rows
        if r.consumer == "pm_memo" and r.mode == "shadow" and r.memo_snapshot_id is not None
    ]
    runs = {r.run_id for r in linked if r.run_id}
    tickers = {r.ticker for r in linked}
    # The soak starts at the latest mode event, or at the ledger epoch (the
    # first night the deployed ledger ran) when nobody has switched modes.
    g2 = {
        "ok": soak_days is not None and soak_days >= MIN_SOAK_DAYS
        and len(runs) >= MIN_SHADOW_RUNS and len(tickers) >= MIN_SHADOW_TICKERS,
        "since": since.isoformat() if since is not None else None,
        "soak_days": round(soak_days, 2) if soak_days is not None else None,
        "linked_runs": len(runs), "tickers": len(tickers),
        "required": {"days": MIN_SOAK_DAYS, "runs": MIN_SHADOW_RUNS, "tickers": MIN_SHADOW_TICKERS},
    }
    over_budget = [r.id for r in rows if r.chars > RENDER_BUDGETS.get(r.consumer, 0)]
    errored = [r.id for r in rows if r.error_type]
    g3 = {"ok": bool(rows) and not over_budget and not errored, "renders": len(rows),
          "over_budget": over_budget[:20], "errored": errored[:20]}
    active = int(db.execute(
        select(func.count(LearningItem.id)).where(LearningItem.status == "active")
    ).scalar_one())
    g4 = {"ok": active >= 1, "active_items": active}
    out = {"G1": g1, "G2": g2, "G3": g3, "G4": g4}
    out["promotable"] = all(g["ok"] for g in (g1, g2, g3, g4))
    return out


def set_mode(
    mode: str, *, reason: str, actor: str = "admin", force: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append a mode event. Inject is refused unless the gates are green or
    ``force`` is set (and then the override is recorded on the row)."""
    if mode not in _RANK:
        raise ValueError(f"unknown learning mode {mode!r}; expected one of {MODES}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a reason is required for every learning mode change")
    if actor not in ("admin", "auto"):
        raise ValueError(f"unknown learning control actor {actor!r}")
    now = now or datetime.utcnow()
    with SessionLocal() as db:
        state = gates(now=now, db=db)
        if mode == "inject" and not state["promotable"] and not force:
            raise GatesNotMet(state)
        event = LearningControlEvent(
            mode=mode, actor=actor, reason=reason.strip()[:2000], gates=state,
            forced=bool(force and mode == "inject" and not state["promotable"]), created_at=now,
        )
        db.add(event)
        db.commit()
        event_id = event.id
    _cache_clear()
    log.info("learning control: mode=%s actor=%s forced=%s event=%s", mode, actor, bool(force), event_id)
    return {"event_id": event_id, "effective_mode": effective_mode(), "gates": state}


def ensure_epoch(db: Session, *, now: datetime | None = None) -> datetime:
    """The one ``ledger_epoch`` marker; created on first use, never moved.

    Its mode is the DB mode at the time, so even a reader that forgot to
    skip epoch rows would read the same mode (it never promotes)."""
    row = db.execute(
        select(LearningControlEvent).where(LearningControlEvent.reason == EPOCH_REASON)
        .order_by(LearningControlEvent.id).limit(1)
    ).scalars().first()
    if row is not None:
        return row.created_at
    at = now or datetime.utcnow()
    db.add(LearningControlEvent(mode=db_mode(db), actor="system", reason=EPOCH_REASON,
                                gates={}, forced=False, created_at=at))
    db.flush()
    return at


def recent_events(db: Session, limit: int = 10) -> list[dict[str, Any]]:
    rows = db.execute(
        select(LearningControlEvent).order_by(LearningControlEvent.id.desc()).limit(limit)
    ).scalars().all()
    return [
        {"id": r.id, "mode": r.mode, "actor": r.actor, "reason": r.reason, "forced": bool(r.forced),
         "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in rows
    ]


def epoch(db: Session) -> datetime | None:
    row = db.execute(
        select(LearningControlEvent.created_at).where(LearningControlEvent.reason == EPOCH_REASON)
        .order_by(LearningControlEvent.id).limit(1)
    ).first()
    return row[0] if row is not None else None


def soak_window_start(db: Session) -> datetime | None:
    latest = _latest_mode_event(db)
    return latest.created_at if latest is not None else epoch(db)


def demote_if_injecting(db: Session, *, reason: str, now: datetime) -> bool:
    """Append an automatic shadow event only when the DB mode is inject — a
    demotion must never RAISE an "off" mode to shadow."""
    if db_mode(db) != "inject":
        return False
    db.add(LearningControlEvent(mode="shadow", actor="auto", reason=reason[:2000], gates={},
                                forced=False, created_at=now))
    return True
