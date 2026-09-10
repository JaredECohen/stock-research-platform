"""FEAT-001 — the catalog and series routes with the login wall off.

The shipped default: no principal, so `authorize()` reports the
`unrestricted` plan and the route applies the Pro shape (5 companies ×
4 metrics × full history). Rows are seeded for throwaway tickers so no
demo backfill can change a value under us; nothing here touches a
provider or an LLM. `test_fundamentals_entitlements` covers the wall.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.api import admin_auth, routes_fundamentals
from app.auth import features, policy
from app.config import settings
from app.database import SessionLocal
from app.main import app, create_app
from app.models import ChartCommentary, CronLoopRun, FinancialPeriod
from app.monitoring import llm_log_gc
from app.rate_limit import LIMITS, limiter
from app.schemas import CatalogOut, SeriesResponse
from app.schemas.accounts import StructuredError
from app.services import fundamentals_catalog as C
from app.services import fundamentals_series_service as fss
from app.services import history_service
from app.tests.factories import seed_annual_periods

# The service's clock is pinned so staleness never depends on when the
# suite runs: FY2024 (ending 2024-12-31) is five months old at NOW.
NOW = datetime(2025, 6, 1, 12, 0, 0)
FRESH = datetime(2025, 5, 30)

SEEDED = ("FAPA", "FAPB")
UNKNOWN = ("ZZFA1", "ZZFA2", "ZZFA3")
LINES = {"revenue": 1000.0, "gross_profit": 600.0, "net_income": 100.0, "cash_from_operations": 200.0}

REQUIRED_METRICS = {
    "revenue", "revenue_growth_yoy", "gross_margin", "operating_margin", "net_margin", "net_income",
    "eps_diluted", "operating_cash_flow", "capex", "free_cash_flow", "fcf_margin", "fcf_after_sbc",
    "roic", "net_debt", "shares_diluted", "pe_ttm", "ev_ebitda", "ev_revenue", "p_fcf", "fcf_yield",
}


def _wipe(db) -> None:
    db.query(FinancialPeriod).filter(FinancialPeriod.ticker.in_(SEEDED)).delete(synchronize_session=False)
    db.commit()


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(fss, "_utcnow", lambda: NOW)


@pytest.fixture()
def seeded():
    """FAPA: FY2017–FY2024; FAPB: FY2023–FY2024. Fresh `fetched_at`."""
    with SessionLocal() as db:
        history_service._ensure_tables(db)
        _wipe(db)
        seed_annual_periods(
            db, "FAPA", {fy: {k: v * (1 + 0.05 * i) for k, v in LINES.items()}
                         for i, fy in enumerate(range(2017, 2025))},
            fetched_at=FRESH,
        )
        seed_annual_periods(db, "FAPB", {2023: LINES, 2024: LINES}, fetched_at=FRESH)
        db.commit()
    yield
    with SessionLocal() as db:
        _wipe(db)


def _series(client, **body):
    return client.post("/api/fundamentals/series", json=body)


def _structured(resp, *, status: int, code: str) -> dict:
    assert resp.status_code == status, f"{resp.status_code}: {resp.text}"
    detail = resp.json()["detail"]
    assert StructuredError.model_validate(detail).code == code, detail
    return detail


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def test_catalog_validates_and_lists_every_required_metric(client):
    resp = client.get("/api/fundamentals/catalog")
    assert resp.status_code == 200, resp.text
    cat = CatalogOut.model_validate(resp.json())
    assert cat.catalog_version == C.CATALOG_VERSION
    assert cat.frequency == "annual"
    assert cat.families == list(C.FAMILIES)
    assert cat.reserved_source_kinds == list(C.RESERVED_SOURCE_KINDS)
    ids = [m.id for m in cat.metrics]
    assert ids == list(C.metric_ids())
    assert REQUIRED_METRICS <= set(ids)
    for m in cat.metrics:
        assert m.frequency == "annual" and m.formula_text and m.provenance
        assert m.family in cat.families


# ---------------------------------------------------------------------------
# Series — the contract
# ---------------------------------------------------------------------------

def test_series_validates_through_the_contract(client, seeded):
    resp = _series(client, tickers=["fapa", "FAPB"], metrics=["revenue", "gross_margin"])
    assert resp.status_code == 200, resp.text
    out = SeriesResponse.model_validate(resp.json())
    assert out.catalog_version == C.CATALOG_VERSION
    assert out.periods == [f"FY{fy}" for fy in range(2017, 2025)]
    assert out.limits.applied.model_dump() == {"companies": 2, "metrics": 2, "years": None}
    assert out.limits.capped_by_plan is False
    assert out.unavailable == [] and len(out.fingerprint) == 64
    by = {(s.ticker, s.metric): s for s in out.series}
    assert set(by) == {("FAPA", "revenue"), ("FAPA", "gross_margin"), ("FAPB", "revenue"), ("FAPB", "gross_margin")}
    # Per-series currency: the ISO code on currency series, null otherwise.
    assert by[("FAPA", "revenue")].currency == "USD"
    assert by[("FAPA", "gross_margin")].currency is None
    # FAPB has no FY2017–FY2022: gap rows with a reason, never zero.
    fapb = by[("FAPB", "revenue")]
    assert fapb.coverage.model_dump() == {"first": "FY2023", "last": "FY2024", "n": 2, "expected": 8}
    assert all(p.value is None and p.reason == "missing_line" for p in fapb.points[:6])
    assert fapb.provenance.source == "test" and fapb.provenance.fetched_at == FRESH
    assert fapb.provenance.stale is False and fapb.provenance.stale_reason is None


def test_wall_off_applies_the_pro_shape(client, seeded):
    """5 × 4 × any range is accepted and nothing is reported as capped."""
    resp = _series(
        client, tickers=[*SEEDED, *UNKNOWN], metrics=["revenue", "net_margin", "fcf_after_sbc", "pe_ttm"],
        years=60,
    )
    assert resp.status_code == 200, resp.text
    out = SeriesResponse.model_validate(resp.json())
    assert out.limits.applied.model_dump() == {"companies": 5, "metrics": 4, "years": 60}
    assert out.limits.capped_by_plan is False
    assert len(out.series) == 20
    # The unknown tickers are per-ticker states, not an error.
    assert [(u.ticker, u.reason) for u in out.unavailable] == [(t, "not_backfilled") for t in UNKNOWN]
    assert all("Run research" in u.remedy for u in out.unavailable)
    zz = next(s for s in out.series if s.ticker == "ZZFA1" and s.metric == "revenue")
    assert all(p.value is None and p.reason == "not_backfilled" for p in zz.points)


def test_indexed_normalize_is_passed_through(client, seeded):
    resp = _series(client, tickers=["FAPA"], metrics=["revenue", "gross_margin"], normalize="indexed")
    out = SeriesResponse.model_validate(resp.json())
    assert out.normalize == "indexed"
    by = {s.metric: s for s in out.series}
    assert by["revenue"].indexed is True and by["revenue"].points[0].value == pytest.approx(100.0)
    assert by["gross_margin"].indexed is False


# ---------------------------------------------------------------------------
# Series — refusals
# ---------------------------------------------------------------------------

def test_unknown_metric_is_422_before_ticker_resolution(client, seeded):
    """A typo in the selection is the client's mistake whatever the tickers
    are — including tickers that would otherwise 404."""
    for tickers in (["FAPA"], list(UNKNOWN)):
        detail = _structured(_series(client, tickers=tickers, metrics=["revenue", "bogus"]),
                             status=422, code="invalid_request")
        assert "bogus" in detail["message"]
        assert detail["extra"]["unknown_metrics"] == ["bogus"]
        assert detail["extra"]["known_metrics"] == list(C.metric_ids())


def test_absolute_ceilings_are_pydantic_422s(client, seeded):
    six = [*SEEDED, *UNKNOWN, "ZZFA9"]
    assert _series(client, tickers=six, metrics=["revenue"]).status_code == 422
    assert _series(client, tickers=["FAPA"], metrics=["revenue", "net_income", "ebitda", "roic", "capex"]).status_code == 422
    assert _series(client, tickers=[], metrics=["revenue"]).status_code == 422
    assert _series(client, tickers=["FAPA"], metrics=[]).status_code == 422
    assert _series(client, tickers=["FAPA"], metrics=["revenue"], years=0).status_code == 422
    assert _series(client, tickers=["FAPA"], metrics=["revenue"], normalize="log").status_code == 422


def test_404_only_when_every_ticker_is_unknown(client, seeded):
    detail = _structured(_series(client, tickers=list(UNKNOWN), metrics=["revenue"]),
                         status=404, code="unknown_tickers")
    assert detail["extra"]["tickers"] == list(UNKNOWN)
    assert "Run research" in detail["message"]
    # One known ticker is enough to draw: the rest are `unavailable`.
    resp = _series(client, tickers=["FAPA", UNKNOWN[0]], metrics=["revenue"])
    assert resp.status_code == 200, resp.text
    out = SeriesResponse.model_validate(resp.json())
    assert [u.ticker for u in out.unavailable] == [UNKNOWN[0]]


def test_service_value_error_maps_to_422(client, seeded, monkeypatch):
    def boom(*_a, **_kw):
        raise ValueError("years must be >= 1")

    monkeypatch.setattr(routes_fundamentals, "build_series", boom)
    detail = _structured(_series(client, tickers=["FAPA"], metrics=["revenue"]), status=422, code="invalid_request")
    assert detail["message"] == "years must be >= 1"


# ---------------------------------------------------------------------------
# Wiring — limits, policy, admin separation, the rollback flag
# ---------------------------------------------------------------------------

FUNDAMENTALS_PATHS = {"/api/fundamentals/catalog", "/api/fundamentals/series"}


def test_routes_are_in_the_schema_with_a_rate_limit_decorator():
    paths = set(app.openapi()["paths"])
    assert FUNDAMENTALS_PATHS <= paths
    # slowapi registers decorated handlers by `module.function`; the
    # schema alone cannot show the decorator, this can.
    for name in ("get_catalog", "post_series"):
        limits = limiter._route_limits[f"app.api.routes_fundamentals.{name}"]
        assert [str(lim.limit) for lim in limits] == ["60 per 1 minute"]
    assert LIMITS["fundamentals_series"] == "60/minute"
    assert LIMITS["fundamentals_commentary"] == "10/minute"


def test_routes_are_customer_routes_not_admin_routes():
    live = [p for p in app.openapi()["paths"] if "fundamentals" in p]
    assert live and not any(p.startswith("/api/admin") for p in live)
    for method, path in (("GET", "/api/fundamentals/catalog"), ("POST", "/api/fundamentals/series")):
        assert not admin_auth.is_protected(method, path)
        pol, explicit = policy.lookup(method, path)
        assert explicit and pol.level == policy.AUTHENTICATED and pol.feature is None
    pol, explicit = policy.lookup("POST", "/api/fundamentals/commentary")
    assert explicit and pol.level == policy.AUTHENTICATED and pol.feature == "chart_commentary"


def test_flag_off_unmounts_both_routers(monkeypatch):
    monkeypatch.setattr(settings, "enable_fundamentals_explorer", False)
    off = create_app()
    assert not any("fundamentals" in p for p in off.openapi()["paths"])
    # The shipped app (flag default True) still has them.
    assert FUNDAMENTALS_PATHS <= set(app.openapi()["paths"])


def test_config_defaults():
    assert settings.enable_fundamentals_explorer is True
    assert settings.fundamentals_anon_commentary is False
    assert settings.fundamentals_commentary_model == ""


# ---------------------------------------------------------------------------
# Feature registry — the shape the route clamps to
# ---------------------------------------------------------------------------

def test_feature_shapes_and_commentary_meter():
    assert features.shape("fundamentals_explorer", "free").as_dict() == {
        "max_companies": 2, "max_metrics": 2, "max_years": 5,
    }
    assert features.shape("fundamentals_explorer", "pro").as_dict() == {
        "max_companies": 5, "max_metrics": 4, "max_years": None,
    }
    # The wall-off plan and anything else not Free gets the Pro shape.
    assert features.shape("fundamentals_explorer", "unrestricted") == features.shape("fundamentals_explorer", "pro")
    with pytest.raises(ValueError):
        features.shape("pm_chat", "free")

    cc = features.get("chart_commentary")
    assert cc.metered and cc.cost_bearing and cc.max_concurrent == 2
    assert features.allowance("chart_commentary", "free").limit == 5
    assert features.allowance("chart_commentary", "pro").limit == 100
    fe = features.get("fundamentals_explorer")
    assert not fe.metered and features.allowance("fundamentals_explorer", "free").allowed
    # The pricing page reads the same numbers the route enforces.
    reg = features.registry_for_config()
    assert reg["fundamentals_explorer"]["shape"]["free"]["max_years"] == 5
    assert reg["fundamentals_explorer"]["shape"]["pro"]["max_years"] is None
    assert "shape" not in reg["pm_chat"]


# ---------------------------------------------------------------------------
# Retention — llm_log_gc sweeps the commentary cache
# ---------------------------------------------------------------------------

def test_llm_log_gc_deletes_old_commentary_rows():
    now = datetime.utcnow()
    old_key, fresh_key = "gc-old-" + now.strftime("%H%M%S%f"), "gc-fresh-" + now.strftime("%H%M%S%f")
    with SessionLocal() as db:
        db.query(ChartCommentary).filter(ChartCommentary.cache_key.in_((old_key, fresh_key))).delete(
            synchronize_session=False)
        db.add(ChartCommentary(cache_key=old_key, fingerprint="f" * 64, created_at=now - timedelta(days=120)))
        db.add(ChartCommentary(cache_key=fresh_key, fingerprint="f" * 64, created_at=now - timedelta(days=3)))
        db.commit()

    result = llm_log_gc.run_once(max_age_days=90)
    assert result["deleted_commentaries"] >= 1 and result["max_age_days"] == 90
    with SessionLocal() as db:
        keys = {r[0] for r in db.query(ChartCommentary.cache_key).filter(
            ChartCommentary.cache_key.in_((old_key, fresh_key))).all()}
        assert keys == {fresh_key}
        note = db.query(CronLoopRun).filter(CronLoopRun.loop_name == "llm_log_gc").one().note
        assert "chart_commentaries" in note
        db.query(ChartCommentary).filter(ChartCommentary.cache_key == fresh_key).delete(synchronize_session=False)
        db.commit()
