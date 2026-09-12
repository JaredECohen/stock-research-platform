"""`GET /api/public/samples*` (FEAT-002, S3).

The logged-out sample pages must be unable to cost money: the request
path reads `public_samples` rows and nothing else. Three guards here —
an AST check on what the request-side modules import, a monkeypatch that
raises on any LLM / provider call during a request, and the allowlist
(a memo existing for a ticker is not permission to show it).

Every test runs with the login wall both off (default) and on (stub
Clerk tenant, no network) — the routes are public either way.
"""
from __future__ import annotations

import ast
import json
import pathlib
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import SessionLocal
from app.main import app
from app.models import MemoSnapshot, PublicSample
from app.schemas import AgentFinding, MispricingThesis, ValuationVerdict
from app.services import memo_store, public_samples
from app.tests.auth_helpers import ClerkStub, bearer, enable_auth
from app.tests.factories import make_finding, make_memo

APP_DIR = pathlib.Path(__file__).resolve().parents[1]
ROUTE_MODULE = APP_DIR / "api" / "routes_public.py"
SERVICE_MODULE = APP_DIR / "services" / "public_samples.py"

FORBIDDEN_PREFIXES = ("app.agents", "app.providers", "app.services.data_service")

# Invented symbols so nothing here collides with memos other tests store.
LISTED = ["ZZS1", "ZZS2"]
UNLISTED = "ZZS9"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture(params=["wall_off", "wall_on"])
def wall(request, monkeypatch, clerk):
    """Run each test under both deployments. The stub tenant's JWKS is
    served from memory; no network."""
    if request.param == "wall_on":
        yield from enable_auth(monkeypatch, clerk)
    else:
        monkeypatch.setattr(settings, "auth_enabled", False)
        yield None


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _allowlist_and_cleanup(monkeypatch):
    monkeypatch.setattr(settings, "sample_tickers", ",".join(LISTED))
    _purge()
    yield
    _purge()


def _purge() -> None:
    with SessionLocal() as db:
        db.query(PublicSample).filter(
            PublicSample.ticker.in_(LISTED + [UNLISTED, public_samples.CONTROL_TICKER])
        ).delete(synchronize_session=False)
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker.in_(LISTED + [UNLISTED])).delete(synchronize_session=False)
        db.commit()


def _memo(ticker: str, **overrides):
    """A memo with every ledger leg populated and a long-form report to strip."""
    earnings = make_finding(
        "Earnings Analyst",
        long_form_report="EIGHT PARAGRAPHS OF EARNINGS PROSE " * 20,
        data={"structured": {
            "period": "Q2 FY26", "overall_tone": "constructive",
            "guidance_changes": [
                {"metric": "FY revenue", "prior": "$40B", "current": "$42B", "direction": "raised", "rationale": "demand"},
            ],
        }},
    )
    base = dict(
        ticker=ticker, company_name=f"{ticker} Holdings", sector="Technology",
        earnings_agent_view=earnings,
        sector_agent_view=make_finding("Sector Analyst", long_form_report="SECTOR PROSE " * 50),
        mispricing_thesis=MispricingThesis(
            consensus_view="Street models 8% growth", our_view="We see 12%",
            gap="Share gains under-modelled", falsifiers=["Two quarters below 8%"],
        ),
        valuation_verdict=ValuationVerdict(verdict="undervalued", dcf_base_upside=0.18, summary="Cheap on our DCF"),
        price_at_memo=101.5, price_at_memo_at=datetime(2026, 9, 1, 12, 0, 0),
        dcf_summary={"base_implied_price": 119.8, "base_upside": 0.18},
    )
    base.update(overrides)
    return make_memo(**base)


def _build(ticker: str, monkeypatch, *, now: datetime | None = None) -> dict:
    """Run the worker-side builder with the provider-touching kinds stubbed:
    what this file tests is the request side; the loop tests cover those."""
    monkeypatch.setattr(public_samples, "_build_comps", lambda t: (None, None, ["comps: stubbed out"]))
    monkeypatch.setattr(public_samples, "_build_prices", lambda t: ({"points": [{"date": "2026-09-05", "close": 100.0}]}, "stub", []))
    return public_samples.build_for_ticker(ticker, now=now)


def _walk(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_walk(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_walk(v, key) for v in value)
    return False


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------

def _resolve(node: ast.AST, package: str) -> list[str]:
    """Absolute module names an import node refers to; relative imports are
    resolved against `package` (the importing module's package)."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    assert isinstance(node, ast.ImportFrom)
    if node.level == 0:
        return [node.module or ""]
    parts = package.split(".")
    base = ".".join(parts[: len(parts) - (node.level - 1)])
    module = f"{base}.{node.module}" if node.module else base
    return [module] + [f"{module}.{alias.name}" for alias in node.names]


def _imports_with_owner(tree: ast.Module) -> list[tuple[ast.AST, str | None]]:
    """(import node, enclosing top-level function name or None)."""
    out: list[tuple[ast.AST, str | None]] = []

    def visit(node: ast.AST, owner: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, owner or child.name)
            elif isinstance(child, ast.Import | ast.ImportFrom):
                out.append((child, owner))
            else:
                visit(child, owner)

    visit(tree, None)
    return out


def _forbidden(names: list[str]) -> bool:
    return any(n == p or n.startswith(p + ".") for n in names for p in FORBIDDEN_PREFIXES)


def test_route_module_never_imports_agents_providers_or_data_service():
    tree = ast.parse(ROUTE_MODULE.read_text())
    bad = [
        ast.unparse(node) for node, _owner in _imports_with_owner(tree)
        if _forbidden(_resolve(node, "app.api"))
    ]
    assert not bad, f"routes_public.py imports cost-bearing modules: {bad}"


def test_service_module_confines_provider_imports_to_worker_functions():
    """The request-side half of `public_samples.py` (module level and every
    function not named in `WORKER_ONLY_FUNCTIONS`) must not reach the
    LLM or provider layers."""
    tree = ast.parse(SERVICE_MODULE.read_text())
    bad = []
    for node, owner in _imports_with_owner(tree):
        if _forbidden(_resolve(node, "app.services")) and owner not in public_samples.WORKER_ONLY_FUNCTIONS:
            bad.append(f"{ast.unparse(node)} (in {owner or 'module scope'})")
    assert not bad, f"request-side code imports cost-bearing modules: {bad}"


def test_route_module_never_calls_the_worker_side_builders():
    tree = ast.parse(ROUTE_MODULE.read_text())
    referenced = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    leaked = sorted(referenced & set(public_samples.WORKER_ONLY_FUNCTIONS))
    assert not leaked, f"routes_public.py references worker-only builders: {leaked}"


def test_requests_never_reach_the_llm_or_the_provider_chain(wall, client, monkeypatch):
    memo_store.save_memo(_memo(LISTED[0]))
    _build(LISTED[0], monkeypatch)

    def boom(*a, **k):
        raise AssertionError("a public request reached a cost-bearing call")

    from app.agents import llm
    from app.services import data_service
    for name in ("chat_json", "chat_text"):
        monkeypatch.setattr(llm, name, boom)
    monkeypatch.setattr(data_service, "get_data_service", boom)

    assert client.get("/api/public/samples").status_code == 200
    for ticker in LISTED + [UNLISTED]:
        resp = client.get(f"/api/public/samples/{ticker}")
        assert resp.status_code in (200, 404), resp.text


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

def test_unlisted_ticker_is_404_even_when_a_memo_exists(wall, client, monkeypatch):
    memo_store.save_memo(_memo(UNLISTED))
    resp = client.get(f"/api/public/samples/{UNLISTED}")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "not_found" and detail["sample_tickers"] == LISTED
    # ...and the worker refuses to build it too.
    with pytest.raises(ValueError):
        public_samples.build_for_ticker(UNLISTED)


def test_listed_but_unbuilt_is_200_with_nulls_and_degraded(wall, client):
    resp = client.get(f"/api/public/samples/{LISTED[1]}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ticker"] == LISTED[1] and body["built_at"] is None
    for kind in public_samples.KINDS:
        assert body[kind] is None
        assert f"{kind}: not built" in body["degraded"]
    assert body["kinds_built"] == []
    assert body["expectations_ledger"]["price_implied"]["status"] == "n/a"
    assert any("not been built" in d for d in body["disclosures"])
    assert resp.headers["cache-control"] == "public, max-age=3600"
    assert resp.headers["etag"].startswith('"')


def test_list_reports_build_state_for_every_listed_ticker(wall, client, monkeypatch):
    memo_store.save_memo(_memo(LISTED[0]))
    _build(LISTED[0], monkeypatch)
    resp = client.get("/api/public/samples")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=300"
    rows = {r["ticker"]: r for r in resp.json()}
    assert list(rows) == LISTED
    assert rows[LISTED[0]]["company_name"] == f"{LISTED[0]} Holdings"
    assert "memo" in rows[LISTED[0]]["kinds"] and rows[LISTED[0]]["built_at"]
    assert rows[LISTED[1]]["kinds"] == [] and rows[LISTED[1]]["built_at"] is None
    assert public_samples.CONTROL_TICKER not in rows


# ---------------------------------------------------------------------------
# Built sample: shape, stripping, expectations ledger, disclosures
# ---------------------------------------------------------------------------

def test_built_sample_strips_long_form_reports_and_carries_the_ledger(wall, client, monkeypatch):
    memo_store.save_memo(_memo(LISTED[0]))
    result = _build(LISTED[0], monkeypatch)
    assert "memo" in result["built"]

    body = client.get(f"/api/public/samples/{LISTED[0]}").json()
    memo = body["memo"]
    assert memo["company_name"] == f"{LISTED[0]} Holdings"
    assert memo["mispricing_thesis"]["our_view"] == "We see 12%"
    assert memo["bull_case"] and memo["bear_case"]
    assert not _walk(memo, "long_form_report"), "long-form agent reports leaked into the public copy"
    assert memo["round_findings"] == []
    # The guidance leg needs the structured earnings block, so that stays.
    assert memo["earnings_agent_view"]["data"]["structured"]["guidance_changes"][0]["metric"] == "FY revenue"

    ledger = body["expectations_ledger"]
    assert ledger["columns"] == ["reported_consensus", "management_guidance", "price_implied", "our_forecast"]
    consensus = ledger["reported_consensus"]
    assert consensus["status"] == "available"
    assert consensus["items"][0]["basis"] == "interpretation"
    assert consensus["items"][0]["value"] == "Street models 8% growth"

    guidance = ledger["management_guidance"]
    assert guidance["status"] == "available"
    assert guidance["items"][0]["basis"] == "observed"
    assert guidance["items"][0]["value"]["direction"] == "raised"

    price = ledger["price_implied"]
    assert price["status"] == "available"
    by_source = {i["source"]: i for i in price["items"]}
    assert by_source["price_at_memo"]["basis"] == "observed" and by_source["price_at_memo"]["value"] == 101.5
    assert by_source["valuation_verdict.dcf_base_upside"]["basis"] == "interpretation"
    assert by_source["dcf_summary.base_implied_price"]["value"] == 119.8

    ours = ledger["our_forecast"]
    assert ours["status"] == "available"
    assert {i["source"] for i in ours["items"]} == {
        "mispricing_thesis.our_view", "mispricing_thesis.gap", "mispricing_thesis.falsifiers",
    }
    assert all(i["basis"] == "interpretation" for i in ours["items"])

    assert body["built_at"] and body["kinds_built"][0] == "memo"
    assert any("research and education only" in d for d in body["disclosures"])
    assert any("not obtained, not that it is zero" in d for d in body["disclosures"])
    assert any(body["built_at"][:19] in d for d in body["disclosures"]), "built_at must be disclosed"
    # Kinds that produced no row are reported as not built; the reasons
    # (stubbed comps, no LLM) travel with the loop's `record_run` note.
    assert "comps: not built" in body["degraded"]
    assert "commentary: not built" in body["degraded"]
    assert "comps: stubbed out" in result["degraded"]
    assert "commentary: skipped (no LLM configured)" in result["degraded"]


def test_ledger_says_not_captured_and_n_a_with_reasons_never_zero():
    memo = make_memo(ticker="ZZS0").model_dump(mode="json")
    ledger = public_samples.build_expectations_ledger(memo)
    assert ledger["reported_consensus"]["status"] == "not_captured"
    assert ledger["management_guidance"] == {"status": "not_captured", "items": [], "reason": "not captured"}
    assert ledger["price_implied"]["status"] == "n/a" and "price snapshot" in ledger["price_implied"]["reason"]
    assert ledger["our_forecast"]["status"] == "not_captured"
    assert "0" not in json.dumps({k: v for k, v in ledger.items() if k != "note"})

    with_price = dict(memo, price_at_memo=50.0)
    ledger = public_samples.build_expectations_ledger(with_price)
    assert ledger["price_implied"]["status"] == "n/a" and "DCF" in ledger["price_implied"]["reason"]

    assert public_samples.build_expectations_ledger(None)["price_implied"]["reason"] == "no stored memo"


def test_public_copy_of_a_stored_memo_round_trips_through_the_schema(monkeypatch):
    """Stripping must not break the memo shape the frontend's MemoCard expects."""
    from app.schemas import StockMemoOut
    memo = _memo(LISTED[0]).model_dump(mode="json")
    stripped = public_samples.strip_for_public(memo)
    out = StockMemoOut.model_validate(stripped)
    assert out.earnings_agent_view.long_form_report is None
    assert out.sector_agent_view.long_form_report is None
    assert out.mispricing_thesis.consensus_view == "Street models 8% growth"
    assert isinstance(out.earnings_agent_view, AgentFinding)


# ---------------------------------------------------------------------------
# ETag / 304
# ---------------------------------------------------------------------------

def test_prices_are_served_as_the_contract_list_not_the_stored_row(wall, client, monkeypatch):
    """Plan §3.1: `prices: [{date, close}]`. The row is stored as
    `{points: [...]}` (the size-cap stripper and the JSON column want an
    object) but the frontend maps over the list directly, so serving the
    stored shape would be a TypeError on the sample page."""
    memo_store.save_memo(_memo(LISTED[0]))
    _build(LISTED[0], monkeypatch)
    with SessionLocal() as db:
        stored = public_samples.rows_for(db, LISTED[0])["prices"].payload
    assert stored == {"points": [{"date": "2026-09-05", "close": 100.0}]}

    body = client.get(f"/api/public/samples/{LISTED[0]}").json()
    assert body["prices"] == [{"date": "2026-09-05", "close": 100.0}]
    assert all(set(p) == {"date", "close"} for p in body["prices"])
    # The other plan shapes are unchanged by the flattening.
    assert body["commentary"] is None and "commentary: not built" in body["degraded"]
    assert "prices: empty" not in body["degraded"]

    # An empty stored series is null + degraded, never `[]` masquerading
    # as "no price moves".
    with SessionLocal() as db:
        row = public_samples.rows_for(db, LISTED[0])["prices"]
        row.payload = {"points": []}
        db.commit()
    body = client.get(f"/api/public/samples/{LISTED[0]}").json()
    assert body["prices"] is None and "prices: empty" in body["degraded"]


def test_etag_304_round_trip_and_rollover(wall, client, monkeypatch):
    memo_store.save_memo(_memo(LISTED[0]))
    _build(LISTED[0], monkeypatch)
    first = client.get(f"/api/public/samples/{LISTED[0]}")
    etag = first.headers["etag"]
    assert etag.startswith('"') and etag.endswith('"')

    again = client.get(f"/api/public/samples/{LISTED[0]}", headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b""
    assert again.headers["etag"] == etag
    assert again.headers["cache-control"] == "public, max-age=3600"

    weak = client.get(f"/api/public/samples/{LISTED[0]}", headers={"If-None-Match": f'"stale", W/{etag}'})
    assert weak.status_code == 304

    miss = client.get(f"/api/public/samples/{LISTED[0]}", headers={"If-None-Match": '"something-else"'})
    assert miss.status_code == 200

    # A rebuild with different content rolls the tag over.
    memo_store.save_memo(_memo(LISTED[0], one_sentence_thesis="Revised thesis."))
    _build(LISTED[0], monkeypatch)
    rebuilt = client.get(f"/api/public/samples/{LISTED[0]}", headers={"If-None-Match": etag})
    assert rebuilt.status_code == 200 and rebuilt.headers["etag"] != etag

    # Unbuilt tickers get a tag of their own, distinct from any built one.
    assert client.get(f"/api/public/samples/{LISTED[1]}").headers["etag"] not in (etag, rebuilt.headers["etag"])


def test_etag_matching_helper_handles_lists_weak_tags_and_star():
    tag = '"abc"'
    assert public_samples.etag_matches('"abc"', tag)
    assert public_samples.etag_matches('W/"abc"', tag)
    assert public_samples.etag_matches('"x", "abc"', tag)
    assert public_samples.etag_matches("*", tag)
    assert not public_samples.etag_matches('"abd"', tag)
    assert not public_samples.etag_matches(None, tag)
    assert not public_samples.etag_matches("", tag)


# ---------------------------------------------------------------------------
# Payload cap
# ---------------------------------------------------------------------------

def test_payload_cap_strips_progressively_then_omits():
    memo = _memo(LISTED[0]).model_dump(mode="json")
    memo = public_samples.strip_for_public(memo)
    cap = 50_000
    assert len(public_samples.canonical_json(memo)) < cap

    # Bulk in a strippable place: citations.
    memo["sector_agent_view"]["evidence"] = [
        {"claim": "x" * 1000, "source": "s", "excerpt": "e"} for _ in range(80)
    ]
    fitted, notes = public_samples.fit_payload("memo", memo, cap=cap)
    assert fitted is not None and fitted["sector_agent_view"]["evidence"] == []
    assert notes and "evidence" in notes[0]
    assert len(public_samples.canonical_json(fitted)) <= cap

    # Bulk somewhere nothing can strip: the body text itself.
    memo["business_summary"] = "y" * (cap + 10)
    fitted, notes = public_samples.fit_payload("memo", memo, cap=cap)
    assert fitted is None and "omitted" in notes[-1]

    # Non-memo kinds have no reducers: fit or drop.
    fitted, notes = public_samples.fit_payload("prices", {"points": [{"date": "d", "close": 1.0}]}, cap=cap)
    assert fitted is not None and notes == []
    fitted, notes = public_samples.fit_payload("prices", {"blob": "z" * (cap + 1)}, cap=cap)
    assert fitted is None and notes[0].startswith("prices:")


def test_builder_drops_an_oversized_kind_and_records_why(monkeypatch):
    memo_store.save_memo(_memo(LISTED[0], business_summary="w" * 2_000_000))
    result = _build(LISTED[0], monkeypatch)
    assert "memo" not in result["built"]
    assert any(n.startswith("memo: payload exceeds") for n in result["degraded"])
    with SessionLocal() as db:
        assert db.get(PublicSample, (LISTED[0], "memo")) is None


# ---------------------------------------------------------------------------
# Kept rows from an earlier build are labelled
# ---------------------------------------------------------------------------

def test_a_kind_kept_from_an_earlier_build_is_flagged(wall, client, monkeypatch):
    memo_store.save_memo(_memo(LISTED[0]))
    old = datetime.utcnow() - timedelta(days=3)
    _build(LISTED[0], monkeypatch, now=old)

    def memo_fails(ticker, db):
        raise RuntimeError("memo store unavailable")

    monkeypatch.setattr(public_samples, "_build_memo", memo_fails)
    result = public_samples.build_for_ticker(LISTED[0])
    assert "memo" not in result["built"]
    assert any("previous row kept" in n for n in result["degraded"])

    body = client.get(f"/api/public/samples/{LISTED[0]}").json()
    assert body["memo"] is not None, "the last good memo row must survive a failed rebuild"
    assert any(d.startswith("memo: kept from an earlier build") for d in body["degraded"])


# ---------------------------------------------------------------------------
# Admin rebuild trigger
# ---------------------------------------------------------------------------

def test_admin_rebuild_writes_the_control_row_and_needs_the_admin_token(wall, client, monkeypatch, clerk):
    from app.tests.test_admin_auth import TOKEN
    monkeypatch.setattr(settings, "admin_api_token", TOKEN)

    assert client.post("/api/admin/samples/rebuild", json={"tickers": [LISTED[0]]}).status_code == 401
    # A customer JWT never opens an admin route.
    resp = client.post("/api/admin/samples/rebuild", json={}, headers=bearer(clerk.token()))
    assert resp.status_code == 401

    resp = client.post(
        "/api/admin/samples/rebuild", json={"tickers": [LISTED[0].lower()]},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["queued"] is True and resp.json()["tickers"] == [LISTED[0]]
    with SessionLocal() as db:
        assert public_samples.pending_request(db)["tickers"] == [LISTED[0]]

    # A second request merges rather than clobbers; unknown tickers are 422.
    resp = client.post("/api/admin/samples/rebuild", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200 and resp.json()["tickers"] == LISTED
    resp = client.post(
        "/api/admin/samples/rebuild", json={"tickers": [UNLISTED]}, headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "invalid_ticker"

    # The control row is invisible to the public routes.
    assert client.get(f"/api/public/samples/{public_samples.CONTROL_TICKER}").status_code == 404
    assert public_samples.CONTROL_TICKER not in {r["ticker"] for r in client.get("/api/public/samples").json()}


def test_public_routes_ignore_a_bad_bearer(wall, client, monkeypatch):
    """A stale token in the browser must not break the landing page."""
    resp = client.get(f"/api/public/samples/{LISTED[0]}", headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 200
