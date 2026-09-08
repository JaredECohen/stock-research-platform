"""`auth/ratelimit.py` — DB-backed fixed windows, leases, structured 429s.

slowapi is disabled for the suite (`RATE_LIMIT_ENABLED=false`, see
backend/conftest.py), so the slowapi-bridge test builds its own tiny app
with an enabled in-memory limiter rather than relying on the real one.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.auth import ratelimit
from app.database import SessionLocal
from app.rate_limit import rate_limit_exceeded_handler

NOW = datetime(2026, 9, 8, 12, 0, 30)


def _ident() -> str:
    return "id-" + uuid.uuid4().hex[:12]


@pytest.mark.parametrize("spec,expected", [
    ("10/minute", (10, 60)),
    ("1/5minute", (1, 300)),
    ("3/hour", (3, 3600)),
    ("120/minute", (120, 60)),
    ("2/day", (2, 86400)),
    ("5 / second", (5, 1)),
])
def test_parse_limit(spec, expected):
    assert ratelimit.parse_limit(spec) == expected


def test_parse_limit_rejects_garbage():
    with pytest.raises(ValueError):
        ratelimit.parse_limit("lots")


def test_every_default_scope_parses():
    for scope in ratelimit.DEFAULT_SCOPES:
        assert ratelimit.scope_limit(scope)[0] > 0


def test_overrides_json_changes_a_scope(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "rate_limit_overrides_json", '{"research": "5/hour", "bogus": "x"}')
    assert ratelimit.scope_limit("research") == (5, 3600)
    monkeypatch.setattr(settings, "rate_limit_overrides_json", "not json")
    assert ratelimit.scope_limit("research") == (3, 3600)


def test_window_arithmetic_and_retry_after():
    ident = _ident()
    with SessionLocal() as db:
        results = [ratelimit.check(db, "test", ident, limit=2, window_seconds=60, now=NOW) for _ in range(3)]
    assert [r.allowed for r in results] == [True, True, False]
    assert [r.count for r in results] == [1, 2, 3]
    assert results[0].retry_after == 0
    # NOW is 30s into the minute window → 30s until it turns.
    assert results[2].retry_after == 30
    assert results[2].window_seconds == 60 and results[2].limit == 2


def test_next_window_starts_clean():
    ident = _ident()
    with SessionLocal() as db:
        for _ in range(3):
            ratelimit.check(db, "test", ident, limit=2, window_seconds=60, now=NOW)
        later = ratelimit.check(db, "test", ident, limit=2, window_seconds=60, now=NOW + timedelta(seconds=31))
    assert later.allowed and later.count == 1


def test_sqlite_fallback_path_matches(monkeypatch):
    """SQLite < 3.35 has no RETURNING; the insert/update/select fallback
    must count identically."""
    monkeypatch.setattr(ratelimit, "SQLITE_RETURNING", False)
    ident = _ident()
    with SessionLocal() as db:
        counts = [ratelimit.check(db, "test", ident, limit=2, window_seconds=60, now=NOW).count for _ in range(3)]
    assert counts == [1, 2, 3]


def _request(ip: str, principal=None) -> Request:
    scope = {
        "type": "http", "method": "GET", "path": "/api/x", "headers": [], "query_string": b"",
        "client": (ip, 1234), "state": {},
    }
    req = Request(scope)
    if principal is not None:
        req.state.principal = principal
    return req


def test_user_first_ip_second_keying():
    """Two users behind one IP are independent; one user on two IPs shares
    a bucket; the IP ceiling still bounds an address running many accounts."""
    u1, u2 = int(uuid.uuid4().int % 10**9), int(uuid.uuid4().int % 10**9)
    ip = f"10.0.0.{uuid.uuid4().int % 250 + 1}"
    limit, window = 2, 60
    with SessionLocal() as db:
        # explicit check() calls so the limit is controlled without config
        for _ in range(limit):
            assert ratelimit.check(db, "user:s", str(u1), limit=limit, window_seconds=window, now=NOW).allowed
        assert not ratelimit.check(db, "user:s", str(u1), limit=limit, window_seconds=window, now=NOW).allowed
        # u2 on the same IP is untouched
        assert ratelimit.check(db, "user:s", str(u2), limit=limit, window_seconds=window, now=NOW).allowed
        # u1 from a second IP is still refused (user key)
        assert not ratelimit.check(db, "user:s", str(u1), limit=limit, window_seconds=window, now=NOW).allowed
        # the IP bucket is IP_MULTIPLIER x wider, shared by everyone behind it
        ip_limit = limit * ratelimit.IP_MULTIPLIER
        for _ in range(ip_limit):
            assert ratelimit.check(db, "ip:s", ip, limit=ip_limit, window_seconds=window, now=NOW).allowed
        assert not ratelimit.check(db, "ip:s", ip, limit=ip_limit, window_seconds=window, now=NOW).allowed


def test_enforce_raises_a_structured_429_with_retry_after(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "rate_limit_overrides_json", '{"research": "1/hour"}')
    uid = int(uuid.uuid4().int % 10**9)
    ip = f"10.1.0.{uuid.uuid4().int % 250 + 1}"
    with SessionLocal() as db:
        ratelimit.enforce(db, _request(ip), "research", user_id=uid, now=NOW)
        with pytest.raises(ratelimit.RateLimited) as ei:
            ratelimit.enforce(db, _request(ip), "research", user_id=uid, now=NOW)
    exc = ei.value
    assert exc.status_code == 429
    assert exc.headers["Retry-After"] == str(exc.detail["retry_after"])
    for field in ("code", "scope", "retry_after", "window_seconds", "message"):
        assert field in exc.detail, field
    assert exc.detail["code"] == "rate_limited"
    assert exc.detail["scope"] == "user:research"
    assert exc.detail["window_seconds"] == 3600
    assert 0 < exc.detail["retry_after"] <= 3600


def test_rate_limited_renders_through_fastapi():
    mini = FastAPI()

    @mini.get("/x")
    def x():
        raise ratelimit.RateLimited(ratelimit.RateResult(False, "user:test", 3, 2, 60, 17))

    resp = TestClient(mini).get("/x")
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "17"
    body = resp.json()["detail"]
    assert body["code"] == "rate_limited" and body["scope"] == "user:test"
    assert body["retry_after"] == 17 and body["window_seconds"] == 60
    assert "message" in body


def test_slowapi_handler_emits_the_same_shape():
    limiter = Limiter(key_func=get_remote_address, storage_uri="memory://", enabled=True,
                      default_limits=["2/minute"], headers_enabled=True)
    mini = FastAPI()
    mini.state.limiter = limiter
    mini.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    mini.add_middleware(SlowAPIMiddleware)

    @mini.get("/y")
    def y():
        return {"ok": True}

    c = TestClient(mini)
    assert c.get("/y").status_code == 200
    assert c.get("/y").status_code == 200
    resp = c.get("/y")
    assert resp.status_code == 429
    body = resp.json()["detail"]
    for field in ("code", "scope", "retry_after", "window_seconds", "message"):
        assert field in body, field
    assert body["code"] == "rate_limited" and body["scope"] == "ip"
    assert body["window_seconds"] == 60 and body["limit"] == 2
    assert 1 <= body["retry_after"] <= 61
    assert resp.headers["retry-after"] == str(body["retry_after"])
    assert resp.headers.get("x-ratelimit-limit") == "2"


def test_leases_cap_concurrency_and_expire():
    uid = int(uuid.uuid4().int % 10**9)
    with SessionLocal() as db:
        t1 = ratelimit.lease(db, user_id=uid, feature="pm_chat", max_concurrent=2, now=NOW)
        t2 = ratelimit.lease(db, user_id=uid, feature="pm_chat", max_concurrent=2, now=NOW)
        assert t1 and t2
        assert ratelimit.lease(db, user_id=uid, feature="pm_chat", max_concurrent=2, now=NOW) is None
        assert ratelimit.release_lease(db, t1) is True
        assert ratelimit.release_lease(db, t1) is False
        assert ratelimit.lease(db, user_id=uid, feature="pm_chat", max_concurrent=2, now=NOW)
        # an expired lease no longer counts
        later = NOW + timedelta(seconds=121)
        assert ratelimit.lease(db, user_id=uid, feature="pm_chat", max_concurrent=2, now=later)


def test_gc_drops_expired_windows_and_leases():
    ident = _ident()
    uid = int(uuid.uuid4().int % 10**9)
    with SessionLocal() as db:
        ratelimit.check(db, "gc", ident, limit=5, window_seconds=60, now=NOW)
        ratelimit.lease(db, user_id=uid, feature="pm_chat", max_concurrent=2, ttl_seconds=10, now=NOW)
        assert ratelimit.gc_expired(db, now=NOW) == 0
        assert ratelimit.gc_expired(db, now=NOW + timedelta(seconds=61)) >= 2


# ---------------------------------------------------------------------------
# The caller's address, through the proxy
# ---------------------------------------------------------------------------

def _peer_request(headers: list[tuple[bytes, bytes]], client: tuple[str, int] = ("10.0.0.1", 1)) -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "query_string": b"",
                    "headers": headers, "client": client})


def test_client_ip_is_one_definition_shared_with_slowapi():
    """One notion of "the caller" for the per-IP DB buckets, slowapi's
    ceilings and the bootstrap IP hash — they must never disagree."""
    import app.rate_limit as slow

    assert ratelimit.client_ip is slow.client_ip
    assert slow._key_func(_peer_request([(b"x-forwarded-for", b"203.0.113.9")])) == "203.0.113.9"


def test_client_ip_takes_the_entry_the_trusted_proxy_appended(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    assert ratelimit.client_ip(_peer_request([(b"x-forwarded-for", b"203.0.113.9")])) == "203.0.113.9"
    # Render appends the peer it saw AFTER whatever the client sent.
    assert ratelimit.client_ip(_peer_request([(b"x-forwarded-for", b"1.2.3.4, 203.0.113.9")])) == "203.0.113.9"
    assert ratelimit.client_ip(_peer_request([(b"x-forwarded-for", b" 1.2.3.4 ,203.0.113.9 ,")])) == "203.0.113.9"
    assert ratelimit.client_ip(_peer_request([])) == "10.0.0.1"
    assert ratelimit.client_ip(_peer_request([], client=None)) == "unknown"


def test_client_ip_cannot_be_chosen_by_the_client(monkeypatch):
    """Regression: the first version read the FIRST entry, so
    `X-Forwarded-For: <random>` gave each request its own per-IP bucket
    and each trial a unique bootstrap_ip_hash."""
    from app.config import settings
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    seen = {
        ratelimit.client_ip(_peer_request([(b"x-forwarded-for", f"10.{i}.0.1, 203.0.113.9".encode())]))
        for i in range(5)
    }
    assert seen == {"203.0.113.9"}


def test_client_ip_hops_zero_ignores_the_header(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "trusted_proxy_hops", 0)
    assert ratelimit.client_ip(_peer_request([(b"x-forwarded-for", b"203.0.113.9")])) == "10.0.0.1"


def test_client_ip_counts_hops_from_the_right(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "trusted_proxy_hops", 2)
    hdr = b"spoofed, 203.0.113.9, 198.51.100.7"  # client, CDN-observed peer, Render-observed CDN
    assert ratelimit.client_ip(_peer_request([(b"x-forwarded-for", hdr)])) == "203.0.113.9"
    # Fewer entries than trusted hops: our proxies did not write this header.
    assert ratelimit.client_ip(_peer_request([(b"x-forwarded-for", b"203.0.113.9")])) == "10.0.0.1"


def test_ip_buckets_follow_the_forwarded_client(monkeypatch):
    """`enforce()`'s ip: bucket keys on the proxy-derived address, so two
    callers behind Render are independent and a spoof prefix changes nothing."""
    from app.config import settings
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    monkeypatch.setattr(settings, "rate_limit_overrides_json", '{"bootstrap": "1/hour"}')
    a, b = f"203.0.113.{uuid.uuid4().int % 250 + 1}", f"198.51.100.{uuid.uuid4().int % 250 + 1}"

    def req(xff: str) -> Request:
        r = _peer_request([(b"x-forwarded-for", xff.encode())], client=("10.0.0.1", 1))  # same socket peer
        r.scope["state"] = {}
        return r

    with SessionLocal() as db:
        ratelimit.enforce(db, req(a), "bootstrap", user_id=None, now=NOW)
        ratelimit.enforce(db, req(b), "bootstrap", user_id=None, now=NOW)  # other caller, same proxy: allowed
        with pytest.raises(ratelimit.RateLimited):
            ratelimit.enforce(db, req(f"10.9.9.9, {a}"), "bootstrap", user_id=None, now=NOW)


def test_slowapi_limiter_keys_on_the_forwarded_client(monkeypatch):
    """The REAL limiter's key function. Behind Render every socket peer is
    the proxy, so keying on `request.client` made the 3/hour bootstrap
    cap a site-wide one: the fourth signup from anyone got a 429."""
    from app.config import settings
    from app.rate_limit import _key_func
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    limiter = Limiter(key_func=_key_func, storage_uri="memory://", enabled=True,
                      default_limits=["1/minute"], headers_enabled=True)
    mini = FastAPI()
    mini.state.limiter = limiter
    mini.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    mini.add_middleware(SlowAPIMiddleware)

    @mini.get("/z")
    def z():
        return {"ok": True}

    c = TestClient(mini)  # every request shares one socket peer, like behind a proxy
    assert c.get("/z", headers={"X-Forwarded-For": "203.0.113.1"}).status_code == 200
    assert c.get("/z", headers={"X-Forwarded-For": "203.0.113.2"}).status_code == 200, \
        "a second caller behind the same proxy has its own bucket"
    assert c.get("/z", headers={"X-Forwarded-For": "203.0.113.1"}).status_code == 429
    assert c.get("/z", headers={"X-Forwarded-For": "10.9.9.9, 203.0.113.1"}).status_code == 429, \
        "a client-supplied prefix does not escape the bucket"
