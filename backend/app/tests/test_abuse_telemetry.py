"""`GET /api/admin/abuse-telemetry` (FEAT-002 phase 6, S4).

The report aggregates rows that other tests also write, so every
assertion here keys on markers unique to this module (scope, route,
IP hash) and compares counts for those, never totals.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.auth import analytics
from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import AnalyticsEvent, UILog, User
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth, new_email, new_sub
from app.tests.billing_helpers import make_user, uid

ADMIN_TOKEN = "admin-token-telemetry-tests"


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def admin(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", ADMIN_TOKEN)
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


@pytest.fixture()
def seeded():
    """Rate-limit and quota hits, trial creations from one address, and
    http rows — all with this test's markers; removed afterwards."""
    tag = f"zz_{uid()}"
    now = datetime.utcnow()
    scope, route, ip_hash = f"{tag}_scope", f"/api/{tag}/route", f"{tag}_iphash"
    with SessionLocal() as db:
        for plan, n in (("free", 3), ("pro", 1), (None, 1)):
            for _ in range(n):
                db.add(AnalyticsEvent(ts=now - timedelta(hours=1), event_name="rate_limit_hit", plan=plan,
                                      anon_id=tag, props={"scope": scope, "route": route, "method": "POST"}))
        db.add(AnalyticsEvent(ts=now - timedelta(hours=25), event_name="rate_limit_hit", plan="free",
                              anon_id=tag, props={"scope": scope, "route": route, "method": "POST"}))
        for _ in range(2):
            db.add(AnalyticsEvent(ts=now - timedelta(minutes=10), event_name="quota_hit", plan="free",
                                  anon_id=tag, props={"feature": f"{tag}_feature"}))
        db.add(UILog(ts=now - timedelta(minutes=5), source="backend", kind="http", path=f"/api/stocks/{tag}/memo",
                     method="GET", status_code=429, session_id=tag))
        db.add(UILog(ts=now - timedelta(minutes=5), source="backend", kind="http", path=f"/api/stocks/{tag}/memo",
                     method="GET", status_code=200, session_id=tag))
        db.add(UILog(ts=now - timedelta(minutes=5), source="backend", kind="http", path="/api/public/config",
                     method="GET", status_code=429, session_id=tag))
        db.commit()
    # Trial started 12h ago (7-day trial, 6.5 days left): inside the 24h window.
    users = [make_user(trial_ends_in=timedelta(days=6, hours=12), now=now) for _ in range(2)]
    with SessionLocal() as db:
        for u in users:
            db.get(User, u.id).bootstrap_ip_hash = ip_hash
        db.commit()
    yield {"tag": tag, "scope": scope, "route": route, "ip_hash": ip_hash, "users": users}
    with SessionLocal() as db:
        db.query(AnalyticsEvent).filter(AnalyticsEvent.anon_id == tag).delete(synchronize_session=False)
        db.query(UILog).filter(UILog.session_id == tag).delete(synchronize_session=False)
        db.commit()


def test_requires_the_admin_token_and_never_a_customer_jwt(client, auth_on, admin):
    assert client.get("/api/admin/abuse-telemetry").status_code == 401
    tok = auth_on.token(sub=new_sub(), email=new_email())
    assert client.get("/api/admin/abuse-telemetry", headers=bearer(tok)).status_code == 401
    assert client.get("/api/admin/abuse-telemetry", headers=admin).status_code == 200


def test_report_aggregates_the_seeded_events(client, admin, seeded):
    resp = client.get("/api/admin/abuse-telemetry", headers=admin)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["window_hours"] == 24
    hits = body["rate_limit_hits"]
    assert hits["by_scope"][seeded["scope"]] == 5, "the 25h-old hit is outside the window"
    assert hits["by_route"][f"POST {seeded['route']}"] == 5
    assert hits["by_plan"]["free"] >= 3 and hits["by_plan"]["pro"] >= 1 and hits["by_plan"]["anon"] >= 1
    assert hits["total"] >= 5
    assert body["quota_hits"]["by_feature"][f"{seeded['tag']}_feature"] == 2
    per_ip = {r["ip_hash"]: r["trials"] for r in body["trials_per_ip_hash"]}
    assert per_ip[seeded["ip_hash"]] == 2
    assert body["trials_started"] >= 2
    # ui_logs: the /api/stocks rows count (one 429 of two); the public
    # config 429 is excluded from the non-public denominator.
    assert body["api_requests_non_public"] >= 2 and body["api_requests_429"] >= 1
    assert 0 < body["share_429"] <= 1


def test_report_is_computed_from_rows_not_process_state(seeded):
    """Every input is a table, so the same numbers come out of any
    process — the cross-process rule this repo learned the hard way."""
    with SessionLocal() as db:
        a = analytics.abuse_report(db, hours=24)
        b = analytics.abuse_report(db, hours=24)
    assert a["rate_limit_hits"]["by_scope"][seeded["scope"]] == b["rate_limit_hits"]["by_scope"][seeded["scope"]] == 5


def test_report_never_carries_an_address_or_email(client, admin, seeded):
    text = client.get("/api/admin/abuse-telemetry", headers=admin).text
    assert "testclient" not in text and "@" not in text
    assert seeded["ip_hash"] in text, "hashes are the identifier, and only hashes"


def test_route_is_in_the_audit_table_as_admin():
    from app.auth import policy
    pol, explicit = policy.lookup("GET", "/api/admin/abuse-telemetry")
    assert explicit and pol.level == policy.ADMIN
