"""Shared harness for the S2 route-gating tests (FEAT-002).

On top of `auth_helpers.ClerkStub`: plan-shaped tokens (Free = verified
email, never bootstrapped; Pro = bootstrapped, so the card-less trial is
running), stored-memo fixtures for synthetic tickers that no other test
generates, and the structured-error check every 4xx body must pass.

Tests import these by name and define the `clerk` / `auth_on` / `client`
fixtures themselves (see `auth_helpers` for why).
"""
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.models import MemoSnapshot, RegenJob, UsageEvent
from app.schemas.accounts import StructuredError
from app.services import memo_store
from app.tests.auth_helpers import ClerkStub, bearer, new_email, new_sub
from app.tests.factories import make_memo

_SEEDED = False


def seed_demo_universe() -> None:
    """Companies + screener rows from the demo fixture, once per process.
    Cheap after the first call and idempotent, so tests that need a
    populated universe (chat ticker extraction, the worker's "ticker
    already known" path) can call it freely."""
    global _SEEDED
    if _SEEDED:
        return
    from app.tests.fixtures.seed_demo_data import run_full_seed
    run_full_seed()
    _SEEDED = True


def free_user(clerk: ClerkStub) -> tuple[str, str]:
    """(sub, token) for a verified user who never bootstrapped: no trial,
    so `resolve_plan` says Free."""
    sub = new_sub()
    return sub, clerk.token(sub=sub, email=new_email(), verified=True)


def pro_user(client: TestClient, clerk: ClerkStub) -> tuple[str, str]:
    """(sub, token) for a user on the 7-day trial — Pro until it ends."""
    sub = new_sub()
    tok = clerk.token(sub=sub, email=new_email(), verified=True)
    resp = client.post("/api/me/bootstrap", headers=bearer(tok))
    assert resp.status_code == 200, resp.text
    assert resp.json()["plan"]["plan"] == "pro", resp.text
    return sub, tok


def user_id_for(client: TestClient, token: str) -> int:
    resp = client.get("/api/me", headers=bearer(token))
    assert resp.status_code == 200, resp.text
    return int(resp.json()["user"]["id"])


def assert_structured(resp, *, code: str, status: int | None = None) -> dict[str, Any]:
    """The body must be `{"detail": StructuredError}` with this `code`."""
    if status is not None:
        assert resp.status_code == status, f"{resp.status_code}: {resp.text}"
    detail = resp.json()["detail"]
    err = StructuredError.model_validate(detail)
    assert err.code == code, detail
    return detail


def store_memo(ticker: str) -> MemoSnapshot:
    """A valid stored memo for `ticker` (factory-built, no pipeline)."""
    return memo_store.save_memo(make_memo(ticker=ticker, company_name=f"{ticker} Corp"))


def purge_memos(*tickers: str) -> None:
    with SessionLocal() as db:
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker.in_(tickers)).delete(synchronize_session=False)
        db.commit()


def purge_jobs() -> None:
    with SessionLocal() as db:
        db.query(RegenJob).delete()
        db.commit()


def usage_events(user_id: int, feature: str) -> list[UsageEvent]:
    with SessionLocal() as db:
        rows = db.query(UsageEvent).filter(
            UsageEvent.user_id == user_id, UsageEvent.feature == feature,
        ).order_by(UsageEvent.id).all()
        db.expunge_all()
        return rows
