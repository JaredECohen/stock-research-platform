"""Portfolio construction tests."""
import pytest
from fastapi.testclient import TestClient

from app.finance.portfolio_construction import industry_group_exposure
from app.schemas import PortfolioHolding, PortfolioRequest
from app.services.portfolio_service import build_model_portfolio


def test_basic_portfolio_respects_max_position_size():
    req = PortfolioRequest(market_view="soft landing with AI capex", num_holdings=10, max_position_size=0.15)
    p = build_model_portfolio(req)
    assert len(p.holdings) >= 5
    for h in p.holdings:
        assert 0 < h.weight <= 0.151  # tiny epsilon for rounding
    total = sum(h.weight for h in p.holdings)
    assert 0.99 <= total <= 1.01


def test_excluded_sectors_are_honored():
    req = PortfolioRequest(
        market_view="recession defense",
        num_holdings=8,
        max_position_size=0.20,
        excluded_sectors=["Energy", "Technology"],
    )
    p = build_model_portfolio(req)
    sectors = {h.sector for h in p.holdings}
    assert all("technology" not in s.lower() for s in sectors)
    assert all("energy" not in s.lower() for s in sectors)


def test_excluded_tickers_are_honored():
    req = PortfolioRequest(
        market_view="ai capex boom",
        num_holdings=8,
        max_position_size=0.20,
        excluded_tickers=["NVDA", "AMD"],
    )
    p = build_model_portfolio(req)
    tickers = {h.ticker for h in p.holdings}
    assert "NVDA" not in tickers
    assert "AMD" not in tickers


# --- FEAT-003: exposure by GICS industry group ---------------------------------


def _holding(ticker: str, weight: float) -> PortfolioHolding:
    return PortfolioHolding(ticker=ticker, company_name=ticker, sector="x", weight=weight, rationale="")


def test_industry_group_exposure_sums_weights_per_group_and_names_the_unmapped():
    holdings = [_holding("AAA", 0.4), _holding("BBB", 0.3), _holding("CCC", 0.2), _holding("DDD", 0.1)]
    classifications = {
        "AAA": {"industry_group_code": "4530", "state": "mapped"},
        "BBB": {"industry_group_code": "4530", "state": "conflict"},
        "CCC": {"industry_group_code": None, "state": "fallback"},   # sector only — not a group weight
    }
    out = industry_group_exposure(holdings, classifications, {"4530": "Semiconductors"})
    assert out["by_group"] == [
        {"code": "4530", "name": "Semiconductors", "sector_code": "45", "weight": 0.7, "tickers": ["AAA", "BBB"]},
    ]
    assert out["unmapped"] == [
        {"ticker": "CCC", "weight": 0.2, "state": "fallback"},
        {"ticker": "DDD", "weight": 0.1, "state": "unclassified"},
    ]
    assert out["mapped_weight"] == pytest.approx(0.7) and out["unmapped_weight"] == pytest.approx(0.3)
    assert out["largest"] == {"code": "4530", "name": "Semiconductors", "weight": 0.7} and out["n_groups"] == 1
    assert industry_group_exposure([], {}, {})["largest"] is None


def test_build_response_carries_an_industry_exposure_block():
    """The route adds the block from stored classifications; the base
    portfolio shape is untouched and the block never fails the build."""
    from app.main import app
    from app.services import gics_registry as reg
    from app.services import industry_classification as ic
    from app.tests.fixtures.demo_dataset import COMPANY_PROFILES

    info = reg.ensure_taxonomy(activate=True)
    ic.classify_all(tickers=sorted(COMPANY_PROFILES), version=info)
    with TestClient(app) as c:
        resp = c.post("/api/portfolio/build", json={"market_view": "soft landing with AI capex", "num_holdings": 8})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["holdings"] and body["sector_allocation"] and body["disclaimer"]
    block = body["industry_exposure"]
    assert block["status"] == "ok" and block["taxonomy_version"] == info.version_key
    assert block["mapping_caveat"] and "not advice" in block["note"]
    total = block["mapped_weight"] + block["unmapped_weight"]
    assert total == pytest.approx(sum(h["weight"] for h in body["holdings"]), abs=1e-3)
    held = {h["ticker"] for h in body["holdings"]}
    assert {t for g in block["by_group"] for t in g["tickers"]} | {u["ticker"] for u in block["unmapped"]} == held
    for g in block["by_group"]:
        assert reg.group(g["code"], version=info).name == g["name"]
    assert block["snapshot"]["status"] in ("ok", "no_snapshot")
    if block["snapshot"]["status"] == "ok":
        assert [r["code"] for r in block["snapshot"]["groups"]] == [g["code"] for g in block["by_group"]]


def test_industry_exposure_block_reports_a_missing_taxonomy_instead_of_failing(monkeypatch):
    from app.api.routes_portfolio import industry_exposure_block
    from app.schemas import ModelPortfolio
    from app.services import gics_registry as reg

    portfolio = ModelPortfolio(
        name="n", market_view="v", risk_level="balanced", holdings=[_holding("NVDA", 1.0)],
        sector_allocation={"x": 1.0}, concentration={}, risk_notes=[], top_thesis_drivers=[],
        what_could_invalidate=[], watch_items=[],
    )
    monkeypatch.setattr(reg, "active_version", lambda: None)
    block = industry_exposure_block(portfolio)
    assert block["status"] == "taxonomy_not_imported" and block["unmapped"][0]["ticker"] == "NVDA"

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(reg, "active_version", _boom)
    block = industry_exposure_block(portfolio)
    assert block["status"] == "unavailable" and block["reason"] == "RuntimeError"
