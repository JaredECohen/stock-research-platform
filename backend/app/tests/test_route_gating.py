"""Route gating under the login wall (FEAT-002, slice S2).

Every route this slice owns, crossed with anonymous / Free / Pro, plus
the `GET /memo` contract that replaces inline generation: 409 `no_memo`,
stale-but-served, `ondemand=true` → 202 via the analyze path, `as_of` →
404, `sync=true` → 403. The agent graph is monkeypatched to fail loudly
so a test cannot pass by accidentally generating a memo. With the wall
off (the default) the legacy paths are pinned unchanged.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import routes_data_catalog, routes_stocks
from app.auth import policy
from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import RegenJob, UsageEvent
from app.services import company_geography, memo_store
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth
from app.tests.gating_helpers import (
    assert_structured,
    free_user,
    pro_user,
    purge_jobs,
    purge_memos,
    purge_rate_windows,
    seed_demo_universe,
    store_memo,
    user_id_for,
)

# Synthetic tickers: no other test stores a memo for these, so "no
# snapshot" is a fact rather than an assumption about collection order.
NO_MEMO = "ZZNM"
ONDEMAND = "ZZOD"
STORED = ("GATEA", "GATEB", "GATEC", "GATED")


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture()
def auth_on(monkeypatch, clerk):
    yield from enable_auth(monkeypatch, clerk)


@pytest.fixture()
def client():
    seed_demo_universe()
    return TestClient(app)


@pytest.fixture(autouse=True)
def _no_inline_generation(monkeypatch):
    """The wall-on paths must never reach the graph or the provider
    backfill; both fail the test if touched. Cleared per test."""
    def boom(*_a, **_kw):
        raise AssertionError("run_stock_memo was called inside a request")

    def boom_universe(*_a, **_kw):
        raise AssertionError("_ensure_lazy_universe ran inside a request under the login wall")

    monkeypatch.setattr(routes_stocks, "run_stock_memo", boom)
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe", boom_universe)
    purge_memos(NO_MEMO, ONDEMAND, *STORED)
    purge_jobs()
    yield
    purge_memos(NO_MEMO, ONDEMAND, *STORED)
    purge_jobs()


# ---------------------------------------------------------------------------
# Plan × route matrix
# ---------------------------------------------------------------------------

# (method, path, json, expected for Free). Anonymous is always 401; Pro
# always gets past the gate (asserted as "not a gate refusal" so a thin
# demo dataset cannot turn a policy test into a data test).
MATRIX = [
    ("GET", "/api/stocks", None, 200),
    ("GET", "/api/stocks/NVDA/analyze/status", None, 200),
    ("GET", "/api/screener?limit=3", None, 200),
    ("POST", "/api/screener/run", {"theme": None, "limit": 3}, 200),
    ("POST", "/api/screener/custom", {"rules": [], "sort_by": "market_cap", "limit": 3}, 200),
    ("GET", "/api/stocks/NVDA/memos", None, 402),
    ("GET", "/api/stocks/NVDA/memory", None, 402),
    ("GET", "/api/macro/series", None, 402),
    ("POST", "/api/macro/analyze", {"scenario": "soft landing"}, 402),
    ("POST", "/api/screener/nl", {"query": "profitable large caps"}, 402),
    ("GET", "/api/data-catalog/meta", None, 402),
    ("GET", "/api/data-catalog/series", None, 402),
    ("GET", "/api/data-catalog/ticker/NVDA/geography", None, 402),
    ("POST", "/api/portfolio/build", {"market_view": "soft landing", "num_holdings": 5}, 402),
    # Free follows the memo: no memo opened this month → plan_required.
    ("GET", "/api/dcf/NVDA/saved", None, 402),
    ("GET", "/api/dcf/NVDA/default-assumptions", None, 402),
    ("GET", "/api/dcf/NVDA/consensus", None, 402),
    ("POST", "/api/dcf/NVDA", None, 402),
    ("GET", "/api/comps/NVDA", None, 402),
]
GATE_STATUSES = {401, 402, 403, 429}


def _call(client, method, path, body, token=None):
    headers = bearer(token) if token else {}
    if method == "GET":
        return client.get(path, headers=headers)
    return client.post(path, json=body, headers=headers)


@pytest.mark.parametrize("method,path,body,free_status", MATRIX)
def test_anonymous_is_refused_with_401(auth_on, client, method, path, body, free_status):
    resp = _call(client, method, path, body)
    detail = assert_structured(resp, code="auth_required", status=401)
    assert resp.headers.get("www-authenticate") == "Bearer"
    assert detail["message"]


@pytest.mark.parametrize("method,path,body,free_status", MATRIX)
def test_free_plan_matrix(auth_on, client, method, path, body, free_status):
    _sub, tok = free_user(auth_on)
    resp = _call(client, method, path, body, tok)
    if free_status == 402:
        detail = assert_structured(resp, code="plan_required", status=402)
        assert detail["plan"] == "free"
        assert detail["upgrade_url"] == "/pricing"
        assert detail["feature"] == policy.classify(method, path.split("?")[0]).feature
    else:
        assert resp.status_code == free_status, resp.text


@pytest.mark.parametrize("method,path,body,free_status", MATRIX)
def test_pro_plan_passes_the_gate(auth_on, client, method, path, body, free_status):
    _sub, tok = pro_user(client, auth_on)
    resp = _call(client, method, path, body, tok)
    assert resp.status_code not in GATE_STATUSES, resp.text
    assert resp.status_code < 500, resp.text


def test_customer_jwt_never_opens_an_admin_route(auth_on, client, monkeypatch):
    """The new admin telemetry route is admin-token only, like the rest of
    the prefix: a Pro customer JWT is not consulted for it."""
    monkeypatch.setattr(settings, "admin_api_token", "admin-secret-for-gating-tests")
    _sub, tok = pro_user(client, auth_on)
    assert client.get("/api/admin/abuse-telemetry", headers=bearer(tok)).status_code == 401
    ok = client.get("/api/admin/abuse-telemetry", headers=bearer("admin-secret-for-gating-tests"))
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["window_hours"] == 24
    assert {"rate_limit_hits", "quota_hits", "trials_per_ip_hash", "share_429"} <= set(body)


# ---------------------------------------------------------------------------
# GET /memo without inline generation
# ---------------------------------------------------------------------------

def test_no_snapshot_is_409_no_memo_with_analyze_path(auth_on, client):
    _sub, tok = free_user(auth_on)
    resp = client.get(f"/api/stocks/{NO_MEMO}/memo", headers=bearer(tok))
    detail = assert_structured(resp, code="no_memo", status=409)
    assert detail["extra"]["analyze_path"] == f"/api/stocks/{NO_MEMO}/analyze"
    assert detail["extra"]["ticker"] == NO_MEMO
    # Nothing was metered: the customer saw no memo.
    uid = user_id_for(client, tok)
    with SessionLocal() as db:
        assert db.query(UsageEvent).filter(UsageEvent.user_id == uid).count() == 0


def test_stale_snapshot_is_served_and_flagged(auth_on, client, monkeypatch):
    store_memo(STORED[0])
    monkeypatch.setattr(
        memo_store, "memo_freshness",
        lambda _snap, **_kw: {"stale": True, "reason": "10-Q filed after memo", "trigger": "new_filing"},
    )
    _sub, tok = free_user(auth_on)
    resp = client.get(f"/api/stocks/{STORED[0]}/memo", headers=bearer(tok))
    assert resp.status_code == 200, resp.text
    assert resp.headers["X-Memo-Stale"] == "true"
    assert resp.headers["X-Memo-Stale-Reason"] == "10-Q filed after memo"
    assert resp.headers["X-Memo-Stale-Trigger"] == "new_filing"
    assert resp.headers["X-Memo-Source"] == "cache"
    assert resp.headers["X-Memo-Version"] == "1"
    assert resp.json()["ticker"] == STORED[0]


def test_fresh_snapshot_is_served_and_metered_once(auth_on, client):
    store_memo(STORED[0])
    _sub, tok = free_user(auth_on)
    first = client.get(f"/api/stocks/{STORED[0]}/memo", headers=bearer(tok))
    assert first.status_code == 200, first.text
    assert "X-Memo-Stale" not in first.headers
    second = client.get(f"/api/stocks/{STORED[0]}/memo", headers=bearer(tok))
    assert second.status_code == 200
    uid = user_id_for(client, tok)
    with SessionLocal() as db:
        events = db.query(UsageEvent).filter(UsageEvent.user_id == uid, UsageEvent.feature == "memo_view").all()
        assert [(e.resource_ref, e.status) for e in events] == [(STORED[0], "committed")]


def test_ondemand_without_snapshot_routes_through_the_analyze_path(auth_on, client):
    _sub, tok = pro_user(client, auth_on)
    uid = user_id_for(client, tok)
    resp = client.get(f"/api/stocks/{ONDEMAND}/memo?ondemand=true", headers=bearer(tok))
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "started" and body["charged"] is True and body["job_id"]
    with SessionLocal() as db:
        job = db.get(RegenJob, body["job_id"])
        assert job.requested_by_user_id == uid
        event = db.get(UsageEvent, job.usage_event_id)
        assert event.feature == "research_run" and event.status == "reserved"
        assert event.resource_ref == ONDEMAND
    # No memo_view charge rode along with the 202.
    with SessionLocal() as db:
        assert db.query(UsageEvent).filter(UsageEvent.user_id == uid, UsageEvent.feature == "memo_view").count() == 0


def test_ondemand_draws_on_the_research_rate_window(auth_on, client, monkeypatch):
    """Regression: `GET /memo?ondemand=true` starts a charged research run
    but sat behind the memo route's `data` scope (120/min), so the plan's
    3/hour ceiling on runs applied only to `POST /analyze`. The ondemand
    branch must draw on `research` exactly as the POST does."""
    purge_rate_windows()
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_overrides_json", '{"research": "1/hour"}')
    _sub, tok = pro_user(client, auth_on)
    first = client.get(f"/api/stocks/{ONDEMAND}/memo?ondemand=true", headers=bearer(tok))
    assert first.status_code == 202, first.text
    second = client.get(f"/api/stocks/{NO_MEMO}/memo?ondemand=true", headers=bearer(tok))
    detail = assert_structured(second, code="rate_limited", status=429)
    assert detail["scope"] == "user:research" and detail["limit"] == 1
    assert int(second.headers["Retry-After"]) >= 1
    # The refused call queued nothing and reserved nothing.
    uid = user_id_for(client, tok)
    with SessionLocal() as db:
        assert [j.ticker for j in db.query(RegenJob).all()] == [ONDEMAND]
        runs = db.query(UsageEvent).filter(UsageEvent.user_id == uid, UsageEvent.feature == "research_run").all()
        assert [e.resource_ref for e in runs] == [ONDEMAND]
    # The POST shares the window: it is the same ceiling, not a second one.
    posted = client.post(f"/api/stocks/{NO_MEMO}/analyze", headers=bearer(tok))
    assert_structured(posted, code="rate_limited", status=429)


def test_inline_override_cannot_skip_metering_under_the_wall(auth_on, client, monkeypatch):
    """Regression: the store-only branch was selected on
    `memo_inline_generation_effective`, so `MEMO_INLINE_GENERATION=true`
    with the wall on sent customers down the legacy path — no
    `memo_view` charge, and the graph plus the provider backfill run
    in-request for free. Under the wall the override is not honoured.
    (The autouse fixture fails the test if the graph or the backfill is
    reached.)"""
    monkeypatch.setattr(settings, "memo_inline_generation", True)
    assert settings.memo_inline_generation_effective is True
    store_memo(STORED[0])
    _sub, tok = free_user(auth_on)
    served = client.get(f"/api/stocks/{STORED[0]}/memo", headers=bearer(tok))
    assert served.status_code == 200, served.text
    assert served.headers["X-Memo-Source"] == "cache"
    uid = user_id_for(client, tok)
    with SessionLocal() as db:
        events = db.query(UsageEvent).filter(UsageEvent.user_id == uid, UsageEvent.feature == "memo_view").all()
        assert [(e.resource_ref, e.status) for e in events] == [(STORED[0], "committed")]
    missing = client.get(f"/api/stocks/{NO_MEMO}/memo", headers=bearer(tok))
    assert_structured(missing, code="no_memo", status=409)
    queued = client.get(f"/api/stocks/{NO_MEMO}/memo?ondemand=true", headers=bearer(tok))
    assert queued.status_code == 202, queued.text  # the worker path, charged


def test_inline_off_with_the_wall_off_is_store_only_and_uncharged(client, monkeypatch):
    """The override's remaining meaning: `MEMO_INLINE_GENERATION=false`
    without accounts serves from the store (worker-only generation) and
    charges nobody, because `authorize` is a no-op with auth off."""
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "memo_inline_generation", False)
    store_memo(STORED[0])
    with SessionLocal() as db:
        events_before = db.query(UsageEvent).count()
    assert client.get(f"/api/stocks/{STORED[0]}/memo").status_code == 200
    detail = assert_structured(client.get(f"/api/stocks/{NO_MEMO}/memo"), code="no_memo", status=409)
    assert detail["extra"]["analyze_path"] == f"/api/stocks/{NO_MEMO}/analyze"
    with SessionLocal() as db:
        assert db.query(UsageEvent).count() == events_before


def test_memo_view_release_does_not_lock_the_ticker_out(auth_on, client, monkeypatch):
    """Regression for the meter (S1 `usage.reserve`): the memo_view key is
    user:memo_view:month:TICKER, and a release under it — the handler
    raising after `authorize` — used to make every later open of that
    ticker a `quota_exceeded` replay with `used=0`. The failed open must
    cost nothing and the next open must succeed and be charged once."""
    store_memo(STORED[0])
    _sub, tok = free_user(auth_on)
    uid = user_id_for(client, tok)
    real = memo_store.memo_to_pydantic
    state = {"failed": False}

    def flaky(snap):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("snapshot schema drift")
        return real(snap)

    monkeypatch.setattr(memo_store, "memo_to_pydantic", flaky)
    broken = TestClient(app, raise_server_exceptions=False).get(f"/api/stocks/{STORED[0]}/memo", headers=bearer(tok))
    assert broken.status_code == 500
    with SessionLocal() as db:
        events = db.query(UsageEvent).filter(UsageEvent.user_id == uid, UsageEvent.feature == "memo_view").all()
        assert [(e.resource_ref, e.status) for e in events] == [(STORED[0], "released")]
    usage = client.get("/api/me/usage", headers=bearer(tok))
    assert usage.status_code == 200, usage.text

    again = client.get(f"/api/stocks/{STORED[0]}/memo", headers=bearer(tok))
    assert again.status_code == 200, again.text
    with SessionLocal() as db:
        events = db.query(UsageEvent).filter(UsageEvent.user_id == uid, UsageEvent.feature == "memo_view").all()
        assert [(e.resource_ref, e.status) for e in events] == [(STORED[0], "committed")]
    # Still one of the three Free tickers, not two.
    for t in STORED[1:3]:
        store_memo(t)
        assert client.get(f"/api/stocks/{t}/memo", headers=bearer(tok)).status_code == 200
    store_memo(STORED[3])
    detail = assert_structured(client.get(f"/api/stocks/{STORED[3]}/memo", headers=bearer(tok)),
                               code="quota_exceeded", status=402)
    assert detail["used"] == 3


def test_unreadable_memo_is_structured_422_and_uncharged(auth_on, client):
    """FIX-004 residual: a stored snapshot whose legacy bull_case mixes
    shapes is refused as stored (never coerced), as 422 `memo_unreadable`
    rather than a 500, and — like any failed open — costs nothing."""
    snap = store_memo(STORED[0])
    with SessionLocal() as db:
        row = db.get(type(snap), snap.id)
        payload = dict(row.memo_json)
        payload["bull_case"] = ["x", {"key_point": "y"}]
        row.memo_json = payload
        db.commit()
    _sub, tok = free_user(auth_on)
    uid = user_id_for(client, tok)
    detail = assert_structured(client.get(f"/api/stocks/{STORED[0]}/memo", headers=bearer(tok)),
                               code="memo_unreadable", status=422)
    assert detail["extra"] == {"ticker": STORED[0], "version": snap.version, "fields": ["bull_case"]}
    with SessionLocal() as db:
        events = db.query(UsageEvent).filter(UsageEvent.user_id == uid, UsageEvent.feature == "memo_view").all()
        assert [(e.resource_ref, e.status) for e in events] == [(STORED[0], "released")]
        assert db.get(type(snap), snap.id).memo_json == payload
    usage = client.get("/api/me/usage", headers=bearer(tok))
    assert usage.status_code == 200, usage.text


def test_sync_analyze_is_refused_under_the_wall(auth_on, client):
    _sub, tok = pro_user(client, auth_on)
    resp = client.post(f"/api/stocks/{NO_MEMO}/analyze?sync=true", headers=bearer(tok))
    assert_structured(resp, code="feature_disabled", status=403)
    with SessionLocal() as db:
        assert db.query(RegenJob).count() == 0


def test_as_of_is_feature_disabled_for_customers(auth_on, client):
    store_memo(STORED[0])
    _sub, tok = pro_user(client, auth_on)
    resp = client.get(f"/api/stocks/{STORED[0]}/memo?as_of=2025-01-02", headers=bearer(tok))
    assert_structured(resp, code="feature_disabled", status=404)
    # A malformed date is still the validation error it always was.
    assert client.get(f"/api/stocks/{STORED[0]}/memo?as_of=nope", headers=bearer(tok)).status_code == 422


# ---------------------------------------------------------------------------
# Meters and follows-memo
# ---------------------------------------------------------------------------

def test_free_memo_view_counts_three_distinct_tickers(auth_on, client):
    for t in STORED:
        store_memo(t)
    _sub, tok = free_user(auth_on)
    for t in STORED[:3]:
        assert client.get(f"/api/stocks/{t}/memo", headers=bearer(tok)).status_code == 200
    fourth = client.get(f"/api/stocks/{STORED[3]}/memo", headers=bearer(tok))
    detail = assert_structured(fourth, code="quota_exceeded", status=402)
    assert detail["used"] == 3 and detail["limit"] == 3 and detail["remaining"] == 0
    assert detail["resets_at"] and detail["upgrade_url"] == "/pricing"
    # Re-opening a counted ticker is free.
    assert client.get(f"/api/stocks/{STORED[0]}/memo", headers=bearer(tok)).status_code == 200


def test_free_dcf_and_comps_follow_the_opened_memo(auth_on, client):
    store_memo(STORED[0])
    _sub, tok = free_user(auth_on)
    refused = client.get(f"/api/dcf/{STORED[0]}/saved", headers=bearer(tok))
    detail = assert_structured(refused, code="plan_required", status=402)
    assert detail["extra"]["ticker"] == STORED[0]
    assert client.get(f"/api/stocks/{STORED[0]}/memo", headers=bearer(tok)).status_code == 200
    opened = client.get(f"/api/dcf/{STORED[0]}/saved", headers=bearer(tok))
    assert opened.status_code == 200, opened.text
    assert opened.json() == {"has_saved": False, "ticker": STORED[0]}
    # Comps passes the gate too (the handler may still 404 on a synthetic
    # ticker with no financials — that is data, not policy).
    comps = client.get(f"/api/comps/{STORED[0]}", headers=bearer(tok))
    assert comps.status_code not in GATE_STATUSES, comps.text
    other = client.get(f"/api/comps/{STORED[1]}", headers=bearer(tok))
    assert_structured(other, code="plan_required", status=402)


def test_pro_memo_view_is_unlimited(auth_on, client):
    for t in STORED:
        store_memo(t)
    _sub, tok = pro_user(client, auth_on)
    for t in STORED:
        assert client.get(f"/api/stocks/{t}/memo", headers=bearer(tok)).status_code == 200


# ---------------------------------------------------------------------------
# Data catalog knobs
# ---------------------------------------------------------------------------

def test_geography_llm_fallback_is_forced_off_for_customers(auth_on, client, monkeypatch):
    seen = {}

    def fake_geo(ticker, *, allow_llm_fallback=True):
        seen["allow"] = allow_llm_fallback
        return None

    monkeypatch.setattr(company_geography, "get_geography", fake_geo)
    _sub, tok = pro_user(client, auth_on)
    resp = client.get("/api/data-catalog/ticker/NVDA/geography?allow_llm=true", headers=bearer(tok))
    assert resp.status_code == 200, resp.text
    assert seen["allow"] is False


def test_geography_llm_flag_passes_through_with_the_wall_off(client, monkeypatch):
    seen = {}

    def fake_geo(ticker, *, allow_llm_fallback=True):
        seen["allow"] = allow_llm_fallback
        return None

    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(company_geography, "get_geography", fake_geo)
    assert client.get("/api/data-catalog/ticker/NVDA/geography?allow_llm=true").status_code == 200
    assert seen["allow"] is True


def test_series_force_refresh_is_ignored_for_customers(auth_on, client, monkeypatch):
    from app.services import data_catalog_service
    seen = {}

    def fake_fetch(series_id, *, force_refresh=False, **_kw):
        seen["force"] = force_refresh
        return None

    monkeypatch.setattr(data_catalog_service, "fetch_series", fake_fetch)
    series_id = routes_data_catalog.SERIES_REGISTRY[0].series_id
    _sub, tok = pro_user(client, auth_on)
    resp = client.get(f"/api/data-catalog/series/{series_id}?force_refresh=true", headers=bearer(tok))
    assert resp.status_code == 502  # the stub returned nothing; the knob is what matters
    assert seen["force"] is False


# ---------------------------------------------------------------------------
# Per-user rate limit
# ---------------------------------------------------------------------------

def test_per_user_rate_limit_is_a_structured_429(auth_on, client, monkeypatch):
    purge_rate_windows()
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_overrides_json", '{"data": "2/minute"}')
    _sub, tok = free_user(auth_on)
    assert client.get("/api/stocks", headers=bearer(tok)).status_code == 200
    assert client.get("/api/stocks", headers=bearer(tok)).status_code == 200
    third = client.get("/api/stocks", headers=bearer(tok))
    detail = assert_structured(third, code="rate_limited", status=429)
    assert detail["scope"] == "user:data" and detail["limit"] == 2
    assert int(third.headers["Retry-After"]) >= 1
    assert detail["retry_after"] == int(third.headers["Retry-After"])
    # Another user on the same address has their own bucket.
    _sub2, tok2 = free_user(auth_on)
    assert client.get("/api/stocks", headers=bearer(tok2)).status_code == 200


def test_evaluate_outcomes_is_one_global_window_under_the_wall(auth_on, client, monkeypatch):
    """Plan §3.3: the outcome loop is platform-wide work, so behind the
    wall it is limited once for everyone (`evaluate_outcomes`, keyed
    "global"), not per user — otherwise N accounts run it N times. The
    route stays browser-exempt from the admin token and Pro-only."""
    from app.api import routes_admin

    calls: list[int] = []
    monkeypatch.setattr(routes_admin.outcome_service, "evaluate_all_due", lambda: calls.append(1) or {"evaluated": 0})
    purge_rate_windows()
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    _sub, tok = pro_user(client, auth_on)
    _sub2, tok2 = pro_user(client, auth_on)
    first = client.post("/api/admin/evaluate-outcomes", headers=bearer(tok))
    assert first.status_code == 200, first.text
    second = client.post("/api/admin/evaluate-outcomes", headers=bearer(tok2))
    detail = assert_structured(second, code="rate_limited", status=429)
    assert detail["scope"] == "evaluate_outcomes" and detail["limit"] == 1 and detail["window_seconds"] == 600
    assert int(second.headers["Retry-After"]) >= 1
    assert calls == [1], "the refused call never ran the loop"
    # Free is refused by the policy before the window is touched.
    _sub3, tok3 = free_user(auth_on)
    assert_structured(client.post("/api/admin/evaluate-outcomes", headers=bearer(tok3)),
                      code="plan_required", status=402)


def test_evaluate_outcomes_is_unlimited_with_the_wall_off(client, monkeypatch):
    from app.api import routes_admin

    calls: list[int] = []
    monkeypatch.setattr(routes_admin.outcome_service, "evaluate_all_due", lambda: calls.append(1) or {"evaluated": 0})
    purge_rate_windows()
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    for _ in range(3):
        assert client.post("/api/admin/evaluate-outcomes").status_code == 200
    assert calls == [1, 1, 1]


# ---------------------------------------------------------------------------
# Wall off: today's paths, untouched
# ---------------------------------------------------------------------------

def test_wall_off_memo_route_keeps_the_legacy_409(client, monkeypatch):
    """With the wall off the route still resolves the universe in-request
    and answers the data-only case with the legacy string detail."""
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(routes_stocks, "_ensure_lazy_universe", lambda _t: "data_only")
    monkeypatch.setattr(memo_store, "latest_memo", lambda _t, **_kw: None)
    resp = client.get(f"/api/stocks/{NO_MEMO}/memo")
    assert resp.status_code == 409
    assert isinstance(resp.json()["detail"], str)
    assert "ondemand=true" in resp.json()["detail"]


def test_wall_off_routes_need_no_token(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    for path in ("/api/stocks", "/api/stocks/NVDA/memos", "/api/macro/series", "/api/data-catalog/meta",
                 "/api/dcf/NVDA/saved"):
        assert client.get(path).status_code == 200, path
