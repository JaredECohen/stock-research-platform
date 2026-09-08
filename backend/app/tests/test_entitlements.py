"""`auth/entitlements.authorize` — the one call site (FEAT-002, S1).

Exercised at function level with a real DB session and a synthetic
request carrying a `Principal`, so each check in the order documented
in the module is pinned independently of any route.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from app.auth import entitlements, features, usage
from app.auth.entitlements import EntitlementError, authorize, require_feature
from app.auth.middleware import customer_auth_middleware
from app.auth.principal import Principal
from app.config import settings
from app.database import SessionLocal
from app.models import AdminOverride, UsageEvent, User
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth, new_sub

NOW = datetime(2026, 9, 8, 12, 0, 0)


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def limits_on(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "usage_limits_enabled", True)
    monkeypatch.setattr(settings, "entitlement_overrides_json", "{}")
    yield


def make_user(*, verified: bool = True, trial_days: int | None = None, state: str = "active") -> User:
    with SessionLocal() as db:
        u = User(
            external_id=new_sub(), email_hash=uuid.uuid4().hex, account_state=state,
            email_verified_at=NOW if verified else None, created_at=NOW,
            trial_started_at=NOW if trial_days else None,
            trial_ends_at=NOW + timedelta(days=trial_days) if trial_days else None,
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        return u


def principal_for(user: User, *, verified: bool | None = None) -> Principal:
    return Principal(
        kind="user", user_id=user.id, external_id=user.external_id,
        email_verified=(user.email_verified_at is not None) if verified is None else verified,
        email_hash=user.email_hash, account_state=user.account_state,
    )


def request_for(principal: Principal | None) -> StarletteRequest:
    req = StarletteRequest({"type": "http", "method": "GET", "path": "/api/x", "headers": [],
                            "query_string": b"", "client": ("127.0.0.1", 1), "state": {}})
    if principal is not None:
        req.state.principal = principal
    return req


# ---------------------------------------------------------------------------
# Order of checks
# ---------------------------------------------------------------------------

def test_wall_off_returns_an_uncharged_grant(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    g = authorize(request_for(None), "research_run")
    assert not g.charged and g.usage_event_id is None


def test_anonymous_is_401(limits_on):
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(Principal.anonymous()), "memo_view", resource="NVDA")
    assert ei.value.status_code == 401 and ei.value.detail["code"] == "auth_required"


def test_unknown_feature_is_a_programming_error(limits_on):
    with pytest.raises(ValueError):
        authorize(request_for(Principal.anonymous()), "not_a_feature")


def test_suspended_is_403(limits_on):
    u = make_user(state="suspended")
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(principal_for(u)), "memo_view", resource="NVDA", now=NOW)
    assert ei.value.status_code == 403 and ei.value.detail["code"] == "account_suspended"


def test_suspend_override_is_403_even_for_pro(limits_on):
    u = make_user(trial_days=5)
    with SessionLocal() as db:
        db.add(AdminOverride(user_id=u.id, kind="suspend", reason="test", starts_at=NOW - timedelta(days=1)))
        db.commit()
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(principal_for(u)), "memo_view", resource="NVDA", now=NOW)
    assert ei.value.detail["code"] == "account_suspended"


def test_unverified_email_blocked_on_cost_bearing_only(limits_on):
    u = make_user(verified=False)
    p = principal_for(u)
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(p), "research_run", now=NOW)
    assert ei.value.status_code == 403 and ei.value.detail["code"] == "email_unverified"
    g = authorize(request_for(p), "memo_view", resource="NVDA", now=NOW)
    assert g.charged


def test_plan_override_grants_pro(limits_on):
    u = make_user()
    with SessionLocal() as db:
        db.add(AdminOverride(user_id=u.id, kind="plan", value="pro", reason="test",
                             starts_at=NOW - timedelta(days=1), expires_at=NOW + timedelta(days=30)))
        db.commit()
    g = authorize(request_for(principal_for(u)), "portfolio", now=NOW)
    assert g.plan == "pro"
    g.release()


# ---------------------------------------------------------------------------
# plan × feature matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plan", ["free", "pro"])
@pytest.mark.parametrize("feature", sorted(features.FEATURES))
def test_plan_feature_matrix_matches_the_registry(limits_on, plan, feature):
    u = make_user(trial_days=5 if plan == "pro" else None)
    p = principal_for(u)
    a = features.allowance(feature, plan)
    resource = "NVDA"
    try:
        g = authorize(request_for(p), feature, resource=resource, now=NOW)
    except EntitlementError as exc:
        assert exc.status_code == 402 and exc.detail["code"] == "plan_required", (feature, plan, exc.detail)
        assert not a.allowed or a.follows_memo, (feature, plan)
        assert exc.detail["plan"] == plan and exc.detail["upgrade_url"] == "/pricing"
        return
    assert a.allowed and not a.follows_memo, (feature, plan)
    assert g.plan == plan
    assert g.charged == a.metered, (feature, plan)
    g.commit()


# ---------------------------------------------------------------------------
# memo_view: 3 distinct tickers on Free
# ---------------------------------------------------------------------------

def test_free_memo_view_counts_distinct_tickers(limits_on):
    u = make_user()
    p = principal_for(u)
    for t in ("NVDA", "COST", "NVDA", "JPM", "COST"):
        g = authorize(request_for(p), "memo_view", resource=t, now=NOW)
        g.commit()
    with SessionLocal() as db:
        assert usage.used(db, u.id, "memo_view", "2026-09") == 3
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(p), "memo_view", resource="LLY", now=NOW)
    d = ei.value.detail
    assert ei.value.status_code == 402 and d["code"] == "quota_exceeded"
    assert d["used"] == 3 and d["limit"] == 3 and d["remaining"] == 0
    assert d["resets_at"].startswith("2026-10-01T00:00:00")
    assert d["upgrade_url"] == "/pricing" and "distinct tickers" in d["message"]
    # the already-opened ticker still works after the quota is hit
    assert not authorize(request_for(p), "memo_view", resource="nvda", now=NOW).charged
    with SessionLocal() as db:
        hits = db.query(UsageEvent).filter(UsageEvent.user_id == u.id).count()
        assert hits == 3


def test_pro_memo_view_is_unlimited_but_measured(limits_on):
    u = make_user(trial_days=5)
    p = principal_for(u)
    for i in range(5):
        authorize(request_for(p), "memo_view", resource=f"T{i}", now=NOW).commit()
    with SessionLocal() as db:
        assert usage.used(db, u.id, "memo_view", "2026-09") == 5


# ---------------------------------------------------------------------------
# follows-memo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("feature", ["dcf", "comps"])
def test_free_dcf_and_comps_follow_the_memo(limits_on, feature):
    u = make_user()
    p = principal_for(u)
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(p), feature, resource="NVDA", now=NOW)
    assert ei.value.status_code == 402 and ei.value.detail["code"] == "plan_required"
    assert ei.value.detail["extra"] == {"ticker": "NVDA"}
    authorize(request_for(p), "memo_view", resource="NVDA", now=NOW).commit()
    g = authorize(request_for(p), feature, resource="NVDA", now=NOW)
    assert not g.charged
    with pytest.raises(EntitlementError):
        authorize(request_for(p), feature, resource="COST", now=NOW)


def test_pro_dcf_does_not_need_a_memo(limits_on):
    u = make_user(trial_days=5)
    assert authorize(request_for(principal_for(u)), "dcf", resource="COST", now=NOW).plan == "pro"


# ---------------------------------------------------------------------------
# Meters, leases, flags
# ---------------------------------------------------------------------------

def test_research_run_free_second_run_is_402(limits_on):
    u = make_user()
    p = principal_for(u)
    g = authorize(request_for(p), "research_run", resource="NVDA", idempotency_key="rr-" + uuid.uuid4().hex, now=NOW)
    assert g.charged and g.remaining == 0
    g.commit()
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(p), "research_run", resource="COST", idempotency_key="rr-" + uuid.uuid4().hex, now=NOW)
    assert ei.value.detail["code"] == "quota_exceeded" and ei.value.detail["limit"] == 1


def test_released_grant_gives_the_run_back(limits_on):
    u = make_user()
    p = principal_for(u)
    g = authorize(request_for(p), "research_run", resource="NVDA", idempotency_key="rr-" + uuid.uuid4().hex, now=NOW)
    g.release()
    g.release()  # idempotent
    g2 = authorize(request_for(p), "research_run", resource="COST", idempotency_key="rr-" + uuid.uuid4().hex, now=NOW)
    assert g2.charged
    g2.commit()


def test_same_idempotency_key_replays_without_a_second_charge(limits_on):
    u = make_user()
    p = principal_for(u)
    key = "rr-" + uuid.uuid4().hex
    g1 = authorize(request_for(p), "research_run", resource="NVDA", idempotency_key=key, now=NOW)
    g2 = authorize(request_for(p), "research_run", resource="NVDA", idempotency_key=key, now=NOW)
    assert g1.usage_event_id == g2.usage_event_id and g2.replayed
    with SessionLocal() as db:
        assert usage.used(db, u.id, "research_run", "2026-09") == 1


def test_pm_chat_concurrency_lease(limits_on):
    u = make_user()
    p = principal_for(u)
    g1 = authorize(request_for(p), "pm_chat", now=NOW)
    g2 = authorize(request_for(p), "pm_chat", now=NOW)
    assert g1.lease_token and g2.lease_token
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(p), "pm_chat", now=NOW)
    assert ei.value.status_code == 429 and ei.value.detail["code"] == "concurrency_limited"
    assert ei.value.headers["Retry-After"]
    g1.commit()
    g3 = authorize(request_for(p), "pm_chat", now=NOW)
    g3.release()
    g2.release()
    with SessionLocal() as db:
        # 2 committed/reserved… g1 committed (1), g2 released, g3 released → 1 used
        assert usage.used(db, u.id, "pm_chat", "2026-09") == 1


def test_usage_limits_off_keeps_plan_gating_but_skips_meters(limits_on, monkeypatch):
    monkeypatch.setattr(settings, "usage_limits_enabled", False)
    u = make_user()
    p = principal_for(u)
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(p), "portfolio", now=NOW)
    assert ei.value.detail["code"] == "plan_required"
    for t in ("A", "B", "C", "D", "E"):
        g = authorize(request_for(p), "memo_view", resource=t, now=NOW)
        assert not g.charged
    assert not authorize(request_for(p), "dcf", resource="ZZZ", now=NOW).charged


def test_overrides_json_changes_the_allowance(limits_on, monkeypatch):
    monkeypatch.setattr(settings, "entitlement_overrides_json", '{"memo_view": {"free": 1}}')
    features._parse_overrides.cache_clear()
    u = make_user()
    p = principal_for(u)
    authorize(request_for(p), "memo_view", resource="NVDA", now=NOW).commit()
    with pytest.raises(EntitlementError) as ei:
        authorize(request_for(p), "memo_view", resource="COST", now=NOW)
    assert ei.value.detail["limit"] == 1


def test_first_value_is_marked_once(limits_on):
    u = make_user()
    p = principal_for(u)
    authorize(request_for(p), "memo_view", resource="NVDA", now=NOW).commit()
    authorize(request_for(p), "memo_view", resource="COST", now=NOW).commit()
    with SessionLocal() as db:
        from app.models import AnalyticsEvent
        row = db.get(User, u.id)
        assert row.first_value_at is not None
        n = db.query(AnalyticsEvent).filter(
            AnalyticsEvent.user_id == u.id, AnalyticsEvent.event_name == "first_value").count()
        assert n == 1


# ---------------------------------------------------------------------------
# require_feature dependency
# ---------------------------------------------------------------------------

def _dep_app() -> FastAPI:
    mini = FastAPI()
    mini.middleware("http")(customer_auth_middleware)

    @mini.get("/api/ok/{ticker}")
    def ok(ticker: str, grant=Depends(require_feature("memo_view"))):
        return {"charged": grant.charged, "event": grant.usage_event_id}

    @mini.get("/api/boom/{ticker}")
    def boom(ticker: str, grant=Depends(require_feature("research_run"))):
        raise HTTPException(status_code=500, detail="generation failed")

    @mini.get("/api/probe")
    def probe(request: Request):
        return {"kind": request.state.principal.kind}

    return mini


def test_require_feature_commits_on_success_and_releases_on_failure(auth_on):
    c = TestClient(_dep_app(), raise_server_exceptions=False)
    tok = auth_on.token()
    resp = c.get("/api/ok/nvda", headers=bearer(tok))
    assert resp.status_code == 200, resp.text
    event_id = resp.json()["event"]
    assert resp.json()["charged"] and event_id
    with SessionLocal() as db:
        assert db.get(UsageEvent, event_id).status == "committed"
        uid = db.get(UsageEvent, event_id).user_id

    resp = c.get("/api/boom/cost", headers=bearer(tok))
    assert resp.status_code == 500
    with SessionLocal() as db:
        ev = db.query(UsageEvent).filter(UsageEvent.user_id == uid, UsageEvent.feature == "research_run").one()
        assert ev.status == "released"
        assert usage.used(db, uid, "research_run", usage.period_key()) == 0


def test_entitlement_snapshot_shape(limits_on):
    u = make_user()
    with SessionLocal() as db:
        state = entitlements.resolve_for_user(db, db.get(User, u.id), NOW)
        snap = entitlements.entitlement_snapshot(db, db.get(User, u.id), state, now=NOW)
    assert set(snap) == set(features.FEATURES)
    mv = snap["memo_view"]
    assert mv.allowed and mv.limit == 3 and mv.used == 0 and mv.remaining == 3 and mv.metered
    assert not snap["portfolio"].allowed
    assert snap["dcf"].follows_memo
