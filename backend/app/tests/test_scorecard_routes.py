"""Phase 6 (slice C) — `/api/scorecard/*` reads, the frozen v1 export
and the three admin enqueue endpoints.

Bare `TestClient(app)` (no lifespan — the seed is a live-provider sweep;
conftest already created the tables). The routes-test universe is scored
at an as-of no other suite uses (2019-12-31), so "latest run on or before"
resolves to this file's own run regardless of collection order. The
export header is compared byte for byte against a frozen string: that IS
the contract test.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient

from app.api import admin_auth
from app.auth import policy
from app.config import settings
from app.database import SessionLocal
from app.finance import scorecard_spec
from app.finance.scorecard_evaluation_math import EVALUATION_CAVEATS
from app.main import app
from app.services import scorecard_queue
from app.services import scorecard_service as svc
from app.tests.scorecard_helpers import REQUESTED_BY, insert_run, purge, seed_universe

UNIVERSE = {
    "Technology": ["ZRT0", "ZRT1", "ZRT2", "ZRT3"],
    "Financial Services": ["ZRT4", "ZRT5", "ZRT6"],
    "Healthcare": ["ZRT7", "ZRT8", "ZRT9"],
}
TICKERS = sorted(t for ts in UNIVERSE.values() for t in ts)
AS_OF = date(2019, 12, 31)
EMPTY_AS_OF = date(2018, 6, 30)
NOW = datetime(2020, 1, 15, 12, 0, 0)
ADMIN_TOKEN = "scorecard-admin-token-xyz"

FROZEN_HEADER = (
    "contract_version,version_key,spec_hash,run_id,as_of,price_date,data_available_at,ticker,sector,coverage,"
    "overall_z,overall_score,universe_percentile,sector_percentile,z_valuation,z_quality,z_growth,"
    "z_profitability,z_efficiency,z_leverage,z_capital_allocation,z_earnings_quality,pct_valuation,pct_quality,"
    "pct_growth,pct_profitability,pct_efficiency,pct_leverage,pct_capital_allocation,pct_earnings_quality,"
    "top_positive_1,top_positive_2,top_positive_3,top_negative_1,top_negative_2,top_negative_3,generated_at\n"
)


@pytest.fixture(scope="module", autouse=True)
def _scored_universe():
    purge(TICKERS, requested_by=(REQUESTED_BY, "admin"))
    seed_universe(UNIVERSE, fiscal_years=(2015, 2016, 2017, 2018), price_through=AS_OF, price_months=20, seed=3)
    original = (svc._utcnow, scorecard_queue._utcnow, svc._price_series)
    svc._utcnow, scorecard_queue._utcnow, svc._price_series = (lambda: NOW), (lambda: NOW), (lambda ticker: None)
    try:
        run = svc.run_scorecard(None, AS_OF, tickers=TICKERS, requested_by=REQUESTED_BY)
        assert run["status"] == scorecard_queue.STATUS_SUCCEEDED, run
        empty_id = insert_run(version_key=svc.VERSION_KEY, as_of=EMPTY_AS_OF, kind="manual")
        yield {"run": run, "empty_id": empty_id}
    finally:
        svc._utcnow, scorecard_queue._utcnow, svc._price_series = original
        purge(TICKERS, requested_by=(REQUESTED_BY, "admin"))


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(svc, "_utcnow", lambda: NOW)


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def test_universe_404_before_any_run_and_for_unknown_versions(client):
    assert client.get("/api/scorecard", params={"as_of": "1990-01-01"}).status_code == 404
    resp = client.get("/api/scorecard", params={"version": "fs-nope"})
    assert resp.status_code == 404 and "fs-nope" in resp.json()["detail"]
    assert client.get("/api/scorecard/ZRT0", params={"version": "fs-nope"}).status_code == 404
    assert client.get("/api/scorecard/ZZZZ").status_code == 404


def test_universe_shape_sort_and_filter(client, _scored_universe):
    resp = client.get("/api/scorecard", params={"as_of": AS_OF.isoformat(), "limit": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["version_key"] == svc.VERSION_KEY and body["run_id"] == _scored_universe["run"]["run_id"]
    assert body["as_of"] == "2019-12-31" and body["is_month_end"] is True and body["stale"] is False
    assert body["universe_size"] == 10 and body["scored"] == 10 and body["insufficient"] == 0
    assert len(body["rows"]) == 5 and [r["rank"] for r in body["rows"]] == [1, 2, 3, 4, 5]
    scores = [r["overall_score"] for r in body["rows"]]
    assert scores == sorted(scores, reverse=True)
    row = body["rows"][0]
    assert set(row["category_score"]) == set(scorecard_spec.FAMILY_NAMES)
    assert row["company_name"].endswith("Test Co") and row["top_positive"] and row["top_negative"]

    fin = client.get("/api/scorecard", params={"as_of": AS_OF.isoformat(), "sector": "financials"}).json()
    assert {r["ticker"] for r in fin["rows"]} == {"ZRT4", "ZRT5", "ZRT6"}
    asc = client.get("/api/scorecard", params={"as_of": AS_OF.isoformat(), "sort_by": "quality", "order": "asc"}).json()
    q = [r["category_z"]["quality"] for r in asc["rows"]]
    assert q == sorted(q)


def test_universe_rejects_sort_columns_outside_the_whitelist(client):
    resp = client.get("/api/scorecard", params={"as_of": AS_OF.isoformat(), "sort_by": "feature_raw"})
    assert resp.status_code == 422 and "sort_by" in resp.json()["detail"]
    assert client.get("/api/scorecard", params={"order": "sideways"}).status_code == 422


def test_ticker_detail_separates_observed_from_model_read(client, _scored_universe):
    resp = client.get("/api/scorecard/zrt4", params={"months": 12})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ticker"] == "ZRT4" and body["sector"] == "Financials" and body["sector_raw"] == "Financial Services"
    assert body["as_of"] == "2019-12-31" and body["latest_period"] == "FY2018"
    assert body["data_available_at"] == "2019-03-16" and body["price_date"] == "2019-12-31"   # FY2018 end + 75d
    assert body["spec_hash"] == scorecard_spec.spec_hash() and body["run_id"] == _scored_universe["run"]["run_id"]
    features = {f["name"]: f for f in body["features"]}
    assert set(features) == set(scorecard_spec.FEATURE_NAMES)
    ev = features["ebitda_ev_yield"]
    assert ev["applicable"] is False and ev["raw"] is None and ev["z"] is None and ev["reason"].startswith("excluded:")
    fcf = features["fcf_yield"]
    assert fcf["raw"] is not None and fcf["z"] is not None and fcf["basis"].startswith("universe")
    assert set(body["categories"]) == set(scorecard_spec.FAMILY_NAMES)
    assert body["categories"]["leverage"]["n_features"] == 0 and body["categories"]["leverage"]["z"] is None
    assert [p["as_of"] for p in body["history"]] == ["2019-12-31"]
    assert body["disagreement"] is None and body["reconciliation"] == ""
    assert client.get("/api/scorecard/ZRT4", params={"as_of": "2019-01-01"}).status_code == 404


def test_spec_and_evaluation_endpoints(client):
    spec = client.get("/api/scorecard/spec").json()
    assert spec["version_key"] == svc.VERSION_KEY and spec["spec_hash"] == scorecard_spec.spec_hash()
    assert [f["name"] for f in spec["families"]] == list(scorecard_spec.FAMILY_NAMES)
    assert spec["rules"]["missing_inputs"]["partner_line_zero_fill"] is False
    assert "50 = z of 0" in spec["score_scale"]
    assert client.get("/api/scorecard/spec", params={"version": "fs-nope"}).status_code == 404

    ev = client.get("/api/scorecard/evaluation").json()
    assert ev["version_key"] == svc.VERSION_KEY and ev["caveats"] == list(EVALUATION_CAVEATS)
    assert client.get("/api/scorecard/evaluation", params={"kind": "double_lasso"}).status_code == 200


def test_first_read_registers_the_in_code_spec_lazily_and_falls_back_on_failure(client, monkeypatch):
    """The decision: the registry is ensured by the loop AND by the routes
    lazily. A web process that answers before the worker has ticked must
    still report `source: registry`, and a failed attempt must degrade to
    the in-code spec rather than break the read."""
    from app.api import routes_scorecard
    from app.models import ScorecardVersion
    with SessionLocal() as db:
        db.query(ScorecardVersion).filter(ScorecardVersion.version_key == svc.VERSION_KEY).delete()
        db.commit()
    monkeypatch.setattr(routes_scorecard, "_registry_attempted", False)
    spec = client.get("/api/scorecard/spec").json()
    assert spec["source"] == "registry" and spec["spec_hash"] == scorecard_spec.spec_hash()
    with SessionLocal() as db:
        rows = db.query(ScorecardVersion).filter(ScorecardVersion.version_key == svc.VERSION_KEY).all()
    assert len(rows) == 1 and rows[0].is_active

    # Attempted once per process, not once per request.
    calls: list[int] = []
    monkeypatch.setattr(svc, "ensure_version_registered", lambda **kw: calls.append(1))
    client.get("/api/scorecard/spec")
    assert calls == []

    # A failing registration is logged and the in-code spec is served.
    with SessionLocal() as db:
        db.query(ScorecardVersion).filter(ScorecardVersion.version_key == svc.VERSION_KEY).delete()
        db.commit()
    monkeypatch.setattr(routes_scorecard, "_registry_attempted", False)

    def broken(**kw):
        raise RuntimeError("db is read-only")

    monkeypatch.setattr(svc, "ensure_version_registered", broken)
    resp = client.get("/api/scorecard/spec")
    assert resp.status_code == 200 and resp.json()["source"] == "code"
    # Kill switch short-circuits before the registry attempt.
    monkeypatch.setattr(routes_scorecard, "_registry_attempted", False)
    monkeypatch.setattr(settings, "enable_scorecard", False)
    assert client.get("/api/scorecard/spec").status_code == 404
    assert routes_scorecard._registry_attempted is False


def test_kill_switch_turns_every_read_into_a_feature_disabled_404(client, monkeypatch):
    monkeypatch.setattr(settings, "enable_scorecard", False)
    for path in ("/api/scorecard", "/api/scorecard/spec", "/api/scorecard/evaluation", "/api/scorecard/export",
                 "/api/scorecard/ZRT0"):
        resp = client.get(path, params={"as_of": AS_OF.isoformat()})
        assert resp.status_code == 404, path
        assert resp.json()["detail"]["code"] == "feature_disabled", path


# ---------------------------------------------------------------------------
# Export — the contract test
# ---------------------------------------------------------------------------

def test_export_csv_header_is_byte_exact_and_carries_the_contract_headers(client, _scored_universe):
    resp = client.get("/api/scorecard/export", params={"format": "csv", "contract": "v1", "as_of": AS_OF.isoformat()})
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-scorecard-contract"] == "v1"
    assert resp.headers["x-scorecard-version"] == svc.VERSION_KEY
    assert resp.headers["x-scorecard-as-of"] == "2019-12-31"
    assert resp.headers["x-scorecard-run-id"] == _scored_universe["run"]["run_id"]
    assert resp.headers["etag"] == f'"{_scored_universe["run"]["run_id"]}"'
    assert resp.headers["content-type"].startswith("text/csv")
    assert resp.headers["content-disposition"] == f'attachment; filename="scorecard_{svc.VERSION_KEY}_2019-12-31.csv"'
    text = resp.text
    first_line = text.split("\n", 1)[0] + "\n"
    assert first_line == FROZEN_HEADER
    assert first_line.encode("utf-8") == FROZEN_HEADER.encode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["ticker"] for r in rows] == TICKERS
    assert rows[0]["contract_version"] == "v1" and rows[0]["spec_hash"] == scorecard_spec.spec_hash()
    assert rows[0]["generated_at"] == "2020-01-15T12:00:00"


def test_export_streams_zero_rows_as_header_only(client, _scored_universe):
    resp = client.get("/api/scorecard/export", params={"as_of": EMPTY_AS_OF.isoformat()})
    assert resp.status_code == 200 and resp.text == FROZEN_HEADER
    assert resp.headers["x-scorecard-contract"] == "v1" and resp.headers["x-scorecard-as-of"] == "2018-06-30"
    as_json = client.get("/api/scorecard/export", params={"as_of": EMPTY_AS_OF.isoformat(), "format": "json"})
    assert as_json.status_code == 200 and as_json.json()["rows"] == []


def test_export_json_and_bad_parameters(client, _scored_universe):
    resp = client.get("/api/scorecard/export", params={"format": "json", "as_of": AS_OF.isoformat(),
                                                        "include_features": "true"})
    assert resp.status_code == 200
    body = json.loads(resp.text)
    assert body["contract"] == "v1" and body["columns"] == list(svc.EXPORT_COLUMNS_V1)
    assert len(body["rows"]) == 10 and "feature_raw" in body["rows"][0] and "feature_z" in body["rows"][0]
    assert client.get("/api/scorecard/export", params={"contract": "v2"}).status_code == 422
    assert client.get("/api/scorecard/export", params={"format": "parquet"}).status_code == 422
    assert client.get("/api/scorecard/export", params={"as_of": "1990-01-01"}).status_code == 404


def test_export_token_rule(client, monkeypatch):
    # Empty token: the export follows the reads' policy (Pro when the wall is on).
    assert policy.classify("GET", "/api/scorecard/export").level == policy.PRO
    assert client.get("/api/scorecard/export", params={"as_of": AS_OF.isoformat()}).status_code == 200

    monkeypatch.setattr(settings, "scorecard_export_token", "export-token-123")
    pol = policy.classify("GET", "/api/scorecard/export")
    assert pol.is_public, "with a token the customer wall must let the route enforce its own bearer"
    assert client.get("/api/scorecard/export", params={"as_of": AS_OF.isoformat()}).status_code == 401
    bad = client.get("/api/scorecard/export", params={"as_of": AS_OF.isoformat()},
                     headers={"Authorization": "Bearer export-token-124"})
    assert bad.status_code == 401
    good = client.get("/api/scorecard/export", params={"as_of": AS_OF.isoformat()},
                      headers={"Authorization": "Bearer export-token-123"})
    assert good.status_code == 200 and good.text.startswith(FROZEN_HEADER)
    # The export token opens ONLY the export.
    assert not admin_auth.is_protected("GET", "/api/scorecard/export")
    assert client.get("/api/scorecard/spec", headers={"Authorization": "Bearer export-token-123"}).status_code == 200


# ---------------------------------------------------------------------------
# Auth classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/api/scorecard", "/api/scorecard/spec", "/api/scorecard/evaluation", "/api/scorecard/NVDA",
])
def test_reads_are_pro_under_the_login_wall(path):
    pol, explicit = policy.lookup("GET", path)
    assert explicit and pol.level == policy.PRO and pol.feature == "scorecard", path
    assert not admin_auth.is_protected("GET", path)


def test_spec_and_evaluation_are_listed_before_the_ticker_template():
    """`lookup` returns the first matching row, so the literal paths must
    precede `/api/scorecard/{ticker}` in the table or they inherit its
    classification by accident (same today, not necessarily forever)."""
    templates = [t for _m, t, _p in policy.ROUTES if t.startswith("/api/scorecard")]
    assert templates.index("/api/scorecard/{ticker}") == len(templates) - 1
    assert {"/api/scorecard/spec", "/api/scorecard/evaluation", "/api/scorecard/export"} <= set(templates)


# ---------------------------------------------------------------------------
# Admin enqueue endpoints
# ---------------------------------------------------------------------------

@pytest.fixture()
def with_token(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", ADMIN_TOKEN)
    yield {"Authorization": f"Bearer {ADMIN_TOKEN}"}


@pytest.mark.parametrize("path", [
    "/api/admin/scorecard/refresh", "/api/admin/scorecard/evaluate", "/api/admin/scorecard/backfill",
])
def test_admin_routes_are_protected(path, with_token, client):
    assert admin_auth.is_protected("POST", path) and policy.classify("POST", path).level == policy.ADMIN
    assert client.post(path, json={}).status_code == 401
    assert client.post(path, json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_admin_refresh_enqueues_and_coalesces(with_token, client):
    body = {"as_of": "2019-12-31", "tickers": ["zrt0", "ZRT1"], "kind": "manual"}
    first = client.post("/api/admin/scorecard/refresh", json=body, headers=with_token)
    assert first.status_code == 202, first.text
    run = first.json()["run"]
    assert run["status"] == "queued" and run["run_kind"] == "manual" and run["as_of"] == "2019-12-31"
    assert run["params"] == {"tickers": ["ZRT0", "ZRT1"]} and first.json()["created"] is True
    again = client.post("/api/admin/scorecard/refresh", json=body, headers=with_token)
    assert again.status_code == 202 and again.json()["created"] is False and again.json()["run"]["id"] == run["id"]
    assert client.post("/api/admin/scorecard/refresh", json={"version_key": "fs-nope"}, headers=with_token).status_code == 404
    assert scorecard_queue.get_run(run["id"])["requested_by"] == "admin"


def test_admin_evaluate_enqueues_an_evaluate_run(with_token, client):
    resp = client.post("/api/admin/scorecard/evaluate", json={"as_of": "2019-12-31"}, headers=with_token)
    assert resp.status_code == 202, resp.text
    assert resp.json()["run"]["run_kind"] == "evaluate" and resp.json()["run"]["status"] == "queued"


def test_admin_backfill_skips_scored_month_ends_and_queues_pit_prepare_first(with_token, client, _scored_universe):
    resp = client.post("/api/admin/scorecard/backfill", json={"months": 3, "end": "2019-12-15"}, headers=with_token)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["pit_prepare"]["run_kind"] == "pit_prepare" and body["pit_prepare"]["as_of"] == "2019-12-31"
    assert body["skipped_existing"] == ["2019-12-31"], "the routes universe already scored that month end"
    queued = [(r["as_of"], r["run_kind"]) for r in body["enqueued"]]
    assert queued == [("2019-10-31", "backfill"), ("2019-11-30", "backfill")]
    assert all(r["id"] > body["pit_prepare"]["id"] for r in body["enqueued"]), "pit_prepare drains first (FIFO)"
    assert "valuation n/a" in body["note"]
