"""Wave 5A — versioned DCF persistence.

Mirrors the shape of `memo_store.py`: thin save/load helpers + a lineage
chain via `parent_version`. Each call to `save_version` creates a new
DB row; the updater computes the diff between consecutive versions and
records it as `assumption_changes` on the new row so reviewers can audit
drift over time.

The model itself (DCFAssumptions / DCFResult) remains the single source
of truth — this module just snapshots it. `latest_version` returns the
freshest snapshot for a ticker, suitable for the agent path that needs
"the live DCF" without rebuilding.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import DCFModel
from ..schemas import DCFAssumptions, DCFResult

log = logging.getLogger(__name__)


# Allowed triggers — kept narrow so the audit log stays readable.
TRIGGERS = {
    "initial",          # first save for this ticker
    "earnings_update",  # roll-forward at quarter close
    "force_refresh",    # explicit user/admin trigger
    "memo_rebuild",     # rebuilt from scratch (assumption set replaced)
}


def _ensure_table(db: Session) -> None:
    bind = db.get_bind()
    DCFModel.__table__.create(bind=bind, checkfirst=True)


def _next_version(db: Session, ticker: str) -> int:
    _ensure_table(db)
    row = db.execute(
        select(DCFModel.version)
        .where(DCFModel.ticker == ticker)
        .order_by(DCFModel.version.desc())
        .limit(1)
    ).first()
    return (row[0] + 1) if row and row[0] is not None else 1


def save_version(
    ticker: str, *, assumptions: DCFAssumptions,
    dcf_result: Optional[DCFResult] = None, trigger: str = "initial",
    parent_version: Optional[int] = None,
    assumption_changes: Optional[List[Dict[str, Any]]] = None,
    db: Optional[Session] = None,
) -> DCFModel:
    """Persist a new DCF version. Returns the inserted row.

    `assumption_changes` should be a list of `{field, from, to, rationale}`
    dicts describing what shifted from `parent_version` to this version.
    Empty for initial seeds.
    """
    if trigger not in TRIGGERS:
        raise ValueError(f"unknown DCF trigger: {trigger!r}; allowed: {sorted(TRIGGERS)}")
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        version = _next_version(db, ticker.upper())
        row = DCFModel(
            ticker=ticker.upper(),
            version=version,
            parent_version=parent_version,
            trigger=trigger,
            assumptions=assumptions.model_dump(mode="json"),
            dcf_result=(dcf_result.model_dump(mode="json") if dcf_result else {}),
            assumption_changes=list(assumption_changes or []),
            change_log=[
                {
                    "version": version,
                    "trigger": trigger,
                    "at": datetime.utcnow().isoformat(),
                    "parent_version": parent_version,
                }
            ],
            generated_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        db.expunge(row)
        return row
    finally:
        if own:
            db.close()


def latest_version(
    ticker: str, *, db: Optional[Session] = None,
) -> Optional[DCFModel]:
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        row = db.execute(
            select(DCFModel)
            .where(DCFModel.ticker == ticker.upper())
            .order_by(DCFModel.version.desc())
            .limit(1)
        ).scalars().first()
        if row is None:
            return None
        db.expunge(row)
        return row
    finally:
        if own:
            db.close()


def version_history(
    ticker: str, *, limit: int = 25, db: Optional[Session] = None,
) -> List[DCFModel]:
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        rows = list(
            db.execute(
                select(DCFModel)
                .where(DCFModel.ticker == ticker.upper())
                .order_by(DCFModel.version.desc())
                .limit(limit)
            ).scalars().all()
        )
        for r in rows:
            db.expunge(r)
        return rows
    finally:
        if own:
            db.close()


def assumptions_to_pydantic(row: DCFModel) -> DCFAssumptions:
    return DCFAssumptions.model_validate(row.assumptions)


def result_to_pydantic(row: DCFModel) -> Optional[DCFResult]:
    if not row.dcf_result:
        return None
    try:
        return DCFResult.model_validate(row.dcf_result)
    except Exception:
        return None


def update_on_earnings_close(ticker: str) -> Optional[DCFModel]:
    """High-level entry point for the post-earnings DCF roll-forward.

    Orchestrates: load latest version → ask LLM updater for adjustments →
    re-run the engine with the updated assumptions → persist as v(N+1).
    Returns the new row, or None if no prior version exists (caller
    should kick off `build_dcf(...)` to seed initial first).
    """
    prior = latest_version(ticker)
    if prior is None:
        return None
    prior_assumptions = assumptions_to_pydantic(prior)

    from ..agents.dcf_updater import actuals_from_history, update_for_new_period
    from ..finance import dcf as dcf_engine

    actuals = actuals_from_history(ticker)
    new_assumptions, change_rows = update_for_new_period(
        ticker, prior_assumptions, actuals,
    )

    new_result: Optional[DCFResult] = None
    try:
        new_result = dcf_engine.build_full_dcf(ticker, new_assumptions)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("DCF rebuild failed for %s post-update: %s", ticker, exc)

    return save_version(
        ticker, assumptions=new_assumptions, dcf_result=new_result,
        trigger="earnings_update", parent_version=prior.version,
        assumption_changes=change_rows,
    )
