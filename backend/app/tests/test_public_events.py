"""`POST /api/public/events` (FEAT-002, S3).

The one write endpoint anonymous traffic can reach. It must be bounded
(names, prop keys, string lengths, batch size, body size), it must never
let a client assert who it is (user attribution only from a verified
bearer), and it must always answer 200 so a probe learns nothing from
the status code.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.auth.analytics import ANALYTICS_EVENT_ALLOWLIST, MAX_PROP_CHARS
from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import AnalyticsEvent, User
from app.services import analytics_service
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth, new_sub


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture(params=["wall_off", "wall_on"])
def wall(request, monkeypatch, clerk):
    if request.param == "wall_on":
        yield from enable_auth(monkeypatch, clerk)
    else:
        monkeypatch.setattr(settings, "auth_enabled", False)
        yield None


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def anon_id():
    """A per-test anon id so the assertions read back only their own rows."""
    value = f"test-{uuid.uuid4().hex[:12]}"
    yield value
    with SessionLocal() as db:
        db.query(AnalyticsEvent).filter(AnalyticsEvent.anon_id == value).delete(synchronize_session=False)
        db.commit()


def _rows(anon_id: str) -> list[AnalyticsEvent]:
    with SessionLocal() as db:
        rows = db.query(AnalyticsEvent).filter(AnalyticsEvent.anon_id == anon_id).order_by(AnalyticsEvent.id).all()
        db.expunge_all()
        return rows


def _post(client, payload, anon_id: str, **headers):
    return client.post("/api/public/events", json=payload, headers={"X-Anon-Id": anon_id, **headers})


# ---------------------------------------------------------------------------
# Allowlists and caps
# ---------------------------------------------------------------------------

def test_allowlisted_events_are_stored_and_unknown_names_rejected(wall, client, anon_id):
    resp = _post(client, {"events": [
        {"name": "landing_view", "props": {"page": "/"}},
        {"name": "sample_view", "props": {"ticker": "NVDA"}},
        {"name": "drop_table", "props": {}},
        {"name": 42},
        "not-an-object",
    ]}, anon_id)
    assert resp.status_code == 200
    assert resp.json() == {"accepted": 2, "rejected": 3}
    rows = _rows(anon_id)
    assert [r.event_name for r in rows] == ["landing_view", "sample_view"]
    assert all(r.source == "fe" for r in rows)
    assert rows[1].props == {"ticker": "NVDA"}


def test_prop_keys_are_filtered_and_strings_bounded(wall, client, anon_id):
    long = "x" * (MAX_PROP_CHARS + 50)
    resp = _post(client, {"events": [{
        "name": "sample_interact",
        "props": {"kind": long, "ticker": "COST", "cookie": "secret", "nested": {"a": 1}, "list": [1, 2], "limit": 3},
    }]}, anon_id)
    assert resp.json()["accepted"] == 1
    props = _rows(anon_id)[0].props
    assert set(props) == {"kind", "ticker", "limit"}
    assert len(props["kind"]) == MAX_PROP_CHARS
    assert props["limit"] == 3


def test_batch_is_capped_at_fifty_events(wall, client, anon_id):
    events = [{"name": "landing_view"} for _ in range(analytics_service.MAX_EVENTS_PER_BATCH + 7)]
    resp = _post(client, {"events": events}, anon_id)
    assert resp.json() == {"accepted": 50, "rejected": 7}
    assert len(_rows(anon_id)) == 50


def test_oversized_body_is_dropped_but_still_200(wall, client, anon_id):
    big = {"events": [{"name": "landing_view", "props": {"page": "p" * 4000}} for _ in range(40)]}
    assert len(json.dumps(big)) > analytics_service.MAX_BODY_BYTES
    resp = _post(client, big, anon_id)
    assert resp.status_code == 200 and resp.json() == {"accepted": 0, "rejected": 0}
    assert _rows(anon_id) == []


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'"string"', b'{"events": "nope"}', b'{"events": null}', b"{}"])
def test_malformed_bodies_are_always_200(wall, client, anon_id, raw):
    resp = client.post(
        "/api/public/events", content=raw,
        headers={"content-type": "application/json", "X-Anon-Id": anon_id},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": 0, "rejected": 0}


def test_every_allowlisted_name_is_accepted(wall, client, anon_id):
    resp = _post(client, {"events": [{"name": n} for n in sorted(ANALYTICS_EVENT_ALLOWLIST)]}, anon_id)
    assert resp.json()["accepted"] == len(ANALYTICS_EVENT_ALLOWLIST)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def test_anonymous_events_carry_anon_and_session_ids_but_no_user(wall, client, anon_id):
    resp = _post(client, {"events": [{"name": "pricing_view"}]}, anon_id, **{"X-Session-Id": "sess-" + "s" * 200})
    assert resp.json()["accepted"] == 1
    row = _rows(anon_id)[0]
    assert row.anon_id == anon_id
    assert row.session_id is not None and len(row.session_id) == analytics_service.MAX_ID_CHARS
    assert row.user_id is None and row.plan is None


def test_a_client_cannot_assert_a_user_id(wall, client, anon_id):
    resp = _post(client, {"events": [{"name": "landing_view", "user_id": 1, "plan": "pro"}]}, anon_id)
    assert resp.json()["accepted"] == 1
    row = _rows(anon_id)[0]
    assert row.user_id is None and row.plan is None


def test_valid_bearer_attributes_events_to_the_user_only_when_the_wall_is_on(wall, client, anon_id, clerk):
    sub = new_sub()
    resp = _post(client, {"events": [{"name": "first_value", "props": {"feature": "memo_view"}}]}, anon_id,
                 **bearer(clerk.token(sub, email=f"{sub}@example.com")))
    assert resp.status_code == 200 and resp.json()["accepted"] == 1
    row = _rows(anon_id)[0]
    if settings.auth_enabled:
        with SessionLocal() as db:
            user = db.query(User).filter(User.external_id == sub).one()
        assert row.user_id == user.id
        assert row.plan in ("free", "pro")
    else:
        # No login wall: a bearer is meaningless and must not be trusted.
        assert row.user_id is None


def test_invalid_bearer_is_anonymous_not_an_error(wall, client, anon_id, clerk):
    forged = clerk.token(alg="HS256")
    resp = _post(client, {"events": [{"name": "landing_view"}]}, anon_id, **bearer(forged))
    assert resp.status_code == 200 and resp.json()["accepted"] == 1
    assert _rows(anon_id)[0].user_id is None
    resp = _post(client, {"events": [{"name": "landing_view"}]}, anon_id, Authorization="Bearer garbage")
    assert resp.status_code == 200 and resp.json()["accepted"] == 1


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def test_client_timestamps_are_used_when_sane_and_replaced_otherwise(wall, client, anon_id):
    now = datetime.utcnow()
    sane = (now - timedelta(minutes=3)).replace(microsecond=0)
    resp = _post(client, {"events": [
        {"name": "landing_view", "ts": sane.isoformat() + "Z"},
        {"name": "landing_view", "ts": int((sane - datetime(1970, 1, 1)).total_seconds() * 1000)},
        {"name": "landing_view", "ts": "1999-01-01T00:00:00Z"},
        {"name": "landing_view", "ts": "garbage"},
        {"name": "landing_view", "ts": True},
    ]}, anon_id)
    assert resp.json()["accepted"] == 5
    rows = _rows(anon_id)
    assert rows[0].ts == sane
    assert abs((rows[1].ts - sane).total_seconds()) < 1
    for r in rows[2:]:
        assert abs((r.ts - now).total_seconds()) < 60, "bad client clocks fall back to server time"


def test_write_failure_is_reported_as_rejected_not_raised(anon_id):
    class BrokenSession:
        def add_all(self, rows):
            raise RuntimeError("db down")

        def rollback(self):
            pass

    out = analytics_service.ingest_batch({"events": [{"name": "landing_view"}]}, db=BrokenSession(), anon_id=anon_id)
    assert out == {"accepted": 0, "rejected": 1}
