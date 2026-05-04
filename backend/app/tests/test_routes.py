"""HTTP smoke tests across the public API surface."""
from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_providers_status_lists_providers_and_mode():
    c = TestClient(app)
    r = c.get("/api/providers/status")
    assert r.status_code == 200
    data = r.json()
    # Wave 9b: runtime is live-only; demo has been retired from
    # `data_service.status()`. Live providers must always be reported.
    assert data["mode"] == "live"
    for provider in ("fmp", "alpha_vantage", "fred", "sec_edgar"):
        assert provider in data["providers"]


def test_list_stocks_includes_seed_universe():
    c = TestClient(app)
    r = c.get("/api/stocks")
    assert r.status_code == 200
    data = r.json()
    tickers = [d["ticker"] for d in data]
    for must_have in ["NVDA", "MSFT", "GOOGL", "JPM", "COST", "LLY"]:
        assert must_have in tickers


def test_chat_routes_to_stock_memo_for_ticker():
    c = TestClient(app)
    r = c.post("/api/chat", json={"message": "Analyze NVDA as a long-term investment.", "history": []})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "single_stock_analysis"
    assert data["memo"] is not None
    assert data["memo"]["ticker"] == "NVDA"
    assert data["memo"]["rating_label"] in ("Bullish", "Mixed Positive", "Neutral", "Mixed Negative", "Bearish")


def test_chat_routes_to_portfolio_for_build_request():
    c = TestClient(app)
    r = c.post("/api/chat", json={"message": "Build a 10-stock portfolio for a soft landing.", "history": []})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "portfolio_construction"
    assert data["portfolio"]
    assert len(data["portfolio"]["holdings"]) >= 5


def test_dcf_endpoint_returns_three_scenarios():
    c = TestClient(app)
    a = c.get("/api/dcf/MSFT/default-assumptions").json()
    r = c.post("/api/dcf/MSFT", json=a)
    assert r.status_code == 200
    data = r.json()
    for k in ("base", "bull", "bear"):
        assert k in data
        assert data[k]["implied_share_price"] > 0


def test_screener_returns_rows():
    c = TestClient(app)
    r = c.get("/api/screener?limit=10")
    assert r.status_code == 200
    data = r.json()
    assert len(data["rows"]) > 0
    assert data["rows"][0]["pm_score"] >= data["rows"][-1]["pm_score"]
