"""Portfolio construction tests."""
from app.schemas import PortfolioRequest
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
