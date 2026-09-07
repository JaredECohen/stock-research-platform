"""Offline FMP provider tests against synthetic `/stable/` fixtures.

Fixtures live in `fixtures/fmp/` — one small JSON body per endpoint the
provider calls (ticker `ACME`, invented numbers, no proprietary bulk
data) plus a handful of failure bodies. Every request goes through an
`httpx.MockTransport` handler keyed on the `/stable/<path>`, so real
`httpx` request / response objects flow through `_get` but no socket
is opened and no key is needed: the provider's `api_key` is set to a
throwaway string after construction.

What is pinned here is the *normalised* row contract downstream code
depends on (`history_service._INCOME_LINES` etc. read these keys), the
period labelling, `None` for absent fields, and the documented
"return None, never raise" behaviour on every failure class.
"""
from __future__ import annotations

import json
import pathlib
import socket
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest

from app.config import settings
from app.providers import fmp_provider as fmp
from app.providers.fmp_provider import FMPProvider, _to_float

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "fmp"
FAKE_KEY = "test-key-not-real"

INCOME_KEYS = {
    "period", "period_end", "currency", "revenue", "cost_of_revenue",
    "gross_profit", "r_and_d", "sga", "operating_income", "ebit", "ebitda",
    "net_income", "eps_diluted", "weighted_avg_shares_diluted",
    "interest_expense", "pretax_income", "tax_expense",
}
BALANCE_KEYS = {
    "period", "period_end", "currency", "total_assets", "total_liabilities",
    "shareholders_equity", "cash_and_equivalents", "short_term_investments",
    "short_term_debt", "long_term_debt", "total_debt", "goodwill",
    "current_assets", "current_liabilities",
}
CASH_KEYS = {
    "period", "period_end", "currency", "cash_from_operations", "capex",
    "free_cash_flow", "depreciation_and_amortization", "dividends_paid",
    "share_repurchases", "stock_based_compensation",
}
PROFILE_KEYS = {
    "ticker", "company_name", "exchange", "sector", "industry", "sub_industry",
    "country", "currency", "market_cap", "cik", "business_description",
    "fiscal_year_end", "is_active", "is_etf", "beta", "shares_outstanding",
    "last_price",
}
QUOTE_KEYS = {
    "ticker", "price", "previous_close", "change", "change_pct", "day_low",
    "day_high", "volume", "timestamp",
}
RATIO_KEYS = {
    "PE", "EV_Revenue", "EV_EBITDA", "PFCF", "FCF_yield", "ROIC", "ROE",
    "gross_margin", "operating_margin", "ebitda_margin", "fcf_margin",
    "net_margin", "debt_to_ebitda", "dividend_yield",
}


def _text(name: str) -> str:
    return (FIXTURES / name).read_text()


def _rows(name: str) -> List[Dict[str, Any]]:
    return json.loads(_text(name))


class _Route:
    def __init__(self, body: str, status: int, headers: Dict[str, str],
                 exc: Optional[Exception]) -> None:
        self.body, self.status, self.headers, self.exc = body, status, headers, exc


class _Router:
    """`/stable/<path>` → canned response; records every request."""

    def __init__(self) -> None:
        self.routes: Dict[str, _Route] = {}
        self.calls: List[Tuple[str, Dict[str, str]]] = []

    def add(self, path: str, body: str = "[]", *, status: int = 200,
            headers: Optional[Dict[str, str]] = None,
            exc: Optional[Exception] = None) -> None:
        self.routes[path] = _Route(body, status, headers or {}, exc)

    def add_fixture(self, path: str, name: str) -> None:
        self.add(path, _text(name))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(fmp.BASE_URL), str(request.url)
        path = request.url.path[len("/stable"):]
        self.calls.append((path, dict(request.url.params)))
        route = self.routes.get(path)
        if route is None:
            return httpx.Response(404, text='{"Error Message": "not routed by test"}')
        if route.exc is not None:
            raise route.exc
        return httpx.Response(route.status, text=route.body, headers=route.headers)


class _FakeHttpx:
    def __init__(self, router: _Router) -> None:
        self._router = router

    def Client(self, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._router), **kwargs)


def _happy_router(router: _Router) -> _Router:
    router.add_fixture("/profile", "profile.json")
    router.add_fixture("/shares-float", "shares_float.json")
    router.add_fixture("/quote", "quote.json")
    router.add_fixture("/historical-price-eod/full", "historical_price_eod_full.json")
    router.add_fixture("/income-statement", "income_statement_annual.json")
    router.add_fixture("/balance-sheet-statement", "balance_sheet_statement.json")
    router.add_fixture("/cash-flow-statement", "cash_flow_statement.json")
    router.add_fixture("/ratios", "ratios.json")
    router.add_fixture("/key-metrics", "key_metrics.json")
    router.add_fixture("/earnings", "earnings.json")
    router.add_fixture("/analyst-estimates", "analyst_estimates.json")
    router.add_fixture("/price-target-consensus", "price_target_consensus.json")
    router.add_fixture("/news/stock", "news_stock.json")
    router.add_fixture("/sp500-constituent", "sp500_constituent.json")
    return router


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def _refuse(*_a, **_k):
        raise RuntimeError("network access attempted during an offline fixture test")
    monkeypatch.setattr(socket.socket, "connect", _refuse)


@pytest.fixture
def router(monkeypatch) -> _Router:
    r = _Router()
    monkeypatch.setattr(fmp, "httpx", _FakeHttpx(r))
    return r


@pytest.fixture
def provider() -> FMPProvider:
    p = FMPProvider()
    p.api_key = FAKE_KEY
    return p


# ---------------------------------------------------------------------------
# Preconditions + transport
# ---------------------------------------------------------------------------

def test_no_real_key_is_configured_and_unconfigured_provider_never_calls_out(router):
    assert settings.fmp_api_key == ""
    p = FMPProvider()
    assert p.status().configured is False
    assert p.get_quote("ACME") is None
    assert router.calls == []


def test_requests_target_stable_namespace_with_key_and_upper_ticker(router, provider):
    _happy_router(router)
    assert provider.get_quote("acme") is not None
    path, params = router.calls[0]
    assert path == "/quote"
    assert params == {"symbol": "ACME", "apikey": FAKE_KEY}


def test_to_float_handles_provider_sentinels():
    assert _to_float(None) is None
    assert _to_float("") is None
    assert _to_float("-") is None
    assert _to_float("None") is None
    assert _to_float("abc") is None
    assert _to_float("12.5") == 12.5
    assert _to_float(0) == 0.0


# ---------------------------------------------------------------------------
# Happy-path row shapes
# ---------------------------------------------------------------------------

def test_sp500_constituents_keeps_symbols_only(router, provider):
    _happy_router(router)
    assert provider.get_sp500_constituents() == ["ACME", "BRKB", "ZETA", "PLAIN"]


def test_profile_normalises_fields(router, provider):
    _happy_router(router)
    prof = provider.get_company_profile("acme")
    assert prof is not None and set(prof) == PROFILE_KEYS
    assert prof["ticker"] == "ACME"
    assert prof["company_name"] == "Acme Industrial Corp"
    assert prof["sector"] == "Industrials"
    assert prof["sub_industry"] == prof["industry"] == "Industrial - Machinery"
    assert prof["market_cap"] == 48_200_000_000.0
    assert prof["cik"] == "0000123456"
    assert prof["beta"] == 1.12
    assert prof["last_price"] == 96.4
    assert prof["shares_outstanding"] == 500_000_000.0    # from /shares-float
    assert prof["is_active"] is True and prof["is_etf"] is False
    assert prof["fiscal_year_end"] is None
    assert [c[0] for c in router.calls] == ["/profile", "/shares-float"]


def test_profile_with_missing_fields_fills_documented_defaults(router, provider):
    router.add_fixture("/profile", "profile_missing_fields.json")
    router.add_fixture("/shares-float", "empty_list.json")
    prof = provider.get_company_profile("ACME")
    assert prof is not None and set(prof) == PROFILE_KEYS
    assert prof["ticker"] == "ACME"
    assert prof["company_name"] is None
    assert prof["exchange"] == "" and prof["sector"] == "" and prof["industry"] == ""
    assert prof["country"] == "US" and prof["currency"] == "USD"
    assert prof["market_cap"] is None and prof["beta"] is None
    assert prof["last_price"] is None and prof["shares_outstanding"] is None
    assert prof["business_description"] == ""


def test_profile_survives_shares_float_failure(router, provider):
    _happy_router(router)
    router.add("/shares-float", "boom", status=500)
    prof = provider.get_company_profile("ACME")
    assert prof is not None
    assert prof["shares_outstanding"] is None
    assert prof["market_cap"] == 48_200_000_000.0


def test_quote_shape(router, provider):
    _happy_router(router)
    q = provider.get_quote("ACME")
    assert q is not None and set(q) == QUOTE_KEYS
    assert q["price"] == 96.4 and q["previous_close"] == 95.1
    assert q["change_pct"] == 1.367
    assert q["volume"] == 3_120_000.0
    assert q["timestamp"] == 1757000000


def test_price_history_is_oldest_first(router, provider):
    _happy_router(router)
    bars = provider.get_price_history("ACME", days=5)
    assert bars is not None and len(bars) == 5
    dates = [b["date"] for b in bars]
    assert dates == sorted(dates)                      # FMP serves newest-first
    assert dates[0] == "2025-08-29" and dates[-1] == "2025-09-05"
    assert set(bars[0]) == {"date", "open", "high", "low", "close", "adjusted_close", "volume"}
    assert all(b["adjusted_close"] == b["close"] for b in bars)
    assert router.calls[0][1]["limit"] == "5"


def test_period_labels():
    label = FMPProvider._period_label
    assert label("2025-06-30", "FY") == "FY2025"
    assert label("2025-06-30", "fy") == "FY2025"
    assert label("2025-06-30", "Q4") == "2025Q4"
    assert label("2025-03-31", "q3") == "2025Q3"
    assert label("2024-12-31", None) == "2024Q4"       # derived from month
    assert label("2025-02-28", "") == "2025Q1"
    assert label("2025-06-30", "TTM") == "2025Q2"      # unknown token → month
    assert label("2024", None) == "2024"               # no month to derive
    assert label("", "FY") == ""


def test_financial_statements_normalise_all_three_statements(router, provider):
    _happy_router(router)
    fin = provider.get_financial_statements("ACME")
    assert fin is not None and set(fin) == {"income", "balance", "cash"}
    assert [c[0] for c in router.calls] == [
        "/income-statement", "/balance-sheet-statement", "/cash-flow-statement",
    ]
    assert all(c[1]["limit"] == "8" for c in router.calls)

    income, balance, cash = fin["income"], fin["balance"], fin["cash"]
    assert [r["period"] for r in income] == ["FY2025", "FY2024", "FY2023"]
    assert all(set(r) == INCOME_KEYS for r in income)
    fy25 = income[0]
    assert fy25["period_end"] == "2025-06-30" and fy25["currency"] == "USD"
    assert fy25["revenue"] == 6_200_000_000.0
    assert fy25["gross_profit"] == 6_200_000_000.0 - 3_720_000_000.0
    assert fy25["operating_income"] == 1_220_000_000.0
    assert fy25["pretax_income"] == 1_220_000_000.0 - 95_000_000.0
    assert fy25["tax_expense"] == 236_000_000.0
    assert fy25["net_income"] == 889_000_000.0
    assert fy25["eps_diluted"] == 1.78
    assert fy25["weighted_avg_shares_diluted"] == 500_000_000.0

    assert all(set(r) == BALANCE_KEYS for r in balance)
    b25 = balance[0]
    assert b25["short_term_debt"] == 210_000_000.0
    assert b25["long_term_debt"] == 1_650_000_000.0
    assert b25["total_debt"] == 1_860_000_000.0
    assert b25["cash_and_equivalents"] == 820_000_000.0
    assert b25["shareholders_equity"] == 3_500_000_000.0
    b23 = balance[2]                                   # no debt lines in payload
    assert b23["short_term_debt"] is None
    assert b23["long_term_debt"] is None
    assert b23["total_debt"] is None
    assert b23["goodwill"] is None

    assert all(set(r) == CASH_KEYS for r in cash)
    c25, c24 = cash[0], cash[1]
    assert c25["cash_from_operations"] == 1_150_000_000.0
    assert c25["capex"] == -310_000_000.0
    assert c25["free_cash_flow"] == 840_000_000.0
    assert c25["dividends_paid"] == -600_000_000.0
    assert c25["share_repurchases"] == -250_000_000.0
    assert c24["dividends_paid"] == -560_000_000.0     # `netDividendsPaid` fallback key


def test_quarterly_income_rows_tolerate_missing_fields():
    rows = [FMPProvider._income_row(r) for r in _rows("income_statement_quarter.json")]
    assert [r["period"] for r in rows] == ["2025Q4", "2025Q3", "2024Q4"]
    assert set(rows[1]) == INCOME_KEYS
    assert rows[1]["ebit"] is None and rows[1]["ebitda"] is None
    assert rows[1]["interest_expense"] is None
    assert rows[1]["revenue"] == 1_540_000_000.0      # the rest still parses


def test_financial_statements_require_income(router, provider):
    _happy_router(router)
    router.add_fixture("/income-statement", "empty_list.json")
    assert provider.get_financial_statements("ACME") is None
    router.add_fixture("/income-statement", "income_statement_annual.json")
    router.add("/balance-sheet-statement", "", status=500)
    fin = provider.get_financial_statements("ACME")
    assert fin is not None and fin["balance"] == [] and len(fin["cash"]) == 3


def test_ratios_merge_ratio_and_key_metric_endpoints(router, provider):
    _happy_router(router)
    r = provider.get_ratios("ACME")
    assert r is not None and set(r) == RATIO_KEYS
    assert r["PE"] == 54.2 and r["PFCF"] == 57.4
    assert r["gross_margin"] == 0.4 and r["net_margin"] == 0.1434
    assert r["dividend_yield"] == 0.0125
    assert r["EV_Revenue"] == 7.92 and r["EV_EBITDA"] == 33.6
    assert r["FCF_yield"] == 0.0174 and r["debt_to_ebitda"] == 0.71
    assert r["ROE"] == 0.254
    assert "ROIC" in r and isinstance(r["ROIC"], float)  # shape only; semantics owned elsewhere
    assert r["fcf_margin"] is None                     # documented "derive elsewhere"


def test_ratios_without_key_metrics_keep_ratio_side(router, provider):
    _happy_router(router)
    router.add("/key-metrics", "", status=500)
    r = provider.get_ratios("ACME")
    assert r is not None and r["PE"] == 54.2
    assert r["EV_EBITDA"] is None and r["ROIC"] is None and r["FCF_yield"] is None


def test_key_metrics_returns_raw_first_row(router, provider):
    _happy_router(router)
    km = provider.get_key_metrics("ACME")
    assert km is not None and km["returnOnInvestedCapital"] == 0.163
    router.add_fixture("/key-metrics", "empty_list.json")
    assert provider.get_key_metrics("ACME") is None


def test_earnings_keeps_rows_with_a_number_and_guards_zero_estimate(router, provider):
    _happy_router(router)
    e = provider.get_earnings("ACME")
    assert e is not None and set(e) == {"quarters"}
    q = e["quarters"]
    assert [r["report_date"] for r in q] == [
        "2025-10-28", "2025-07-29", "2025-04-29", "2025-01-28",
    ]                                                  # null/null row dropped
    assert set(q[0]) == {
        "period", "report_date", "eps_actual", "eps_estimate", "surprise_pct",
        "revenue_actual", "revenue_estimate",
    }
    future, beat, inline, zero_est = q
    assert future["eps_actual"] is None and future["surprise_pct"] is None
    assert abs(beat["surprise_pct"] - (0.5 - 0.46) / 0.46) < 1e-12
    assert inline["surprise_pct"] == 0.0
    assert zero_est["eps_estimate"] == 0.0 and zero_est["surprise_pct"] is None
    assert beat["revenue_actual"] == 1_650_000_000.0


def test_earnings_with_only_empty_rows_returns_none(router, provider):
    router.add("/earnings", json.dumps([{"symbol": "ACME", "date": "2025-10-28"}]))
    assert provider.get_earnings("ACME") is None
    router.add_fixture("/earnings", "empty_list.json")
    assert provider.get_earnings("ACME") is None


def test_estimates_are_chronological_with_legacy_growth_path(router, provider):
    _happy_router(router)
    est = provider.get_estimates("ACME")
    assert est is not None
    assert set(est) == {"annual", "revenue", "revenue_growth", "price_target"}
    assert [y["period"] for y in est["annual"]] == ["2025-06-30", "2026-06-30", "2027-06-30"]
    assert est["annual"][0]["revenue_avg"] is None and est["annual"][0]["eps_avg"] == 1.78
    assert est["annual"][1]["num_analysts_eps"] == 14
    # The row without revenueAvg is excluded from the legacy growth path.
    assert [r["value"] for r in est["revenue"]] == [6_700_000_000.0, 7_250_000_000.0]
    assert len(est["revenue_growth"]) == 1
    assert abs(est["revenue_growth"][0] - (7_250 - 6_700) / 6_700) < 1e-12
    assert est["price_target"] == {
        "target_high": 125.0, "target_low": 82.0,
        "target_consensus": 104.5, "target_median": 105.0,
    }
    assert router.calls[0][1]["period"] == "annual"


def test_estimates_partial_availability(router, provider):
    _happy_router(router)
    router.add_fixture("/analyst-estimates", "empty_list.json")
    est = provider.get_estimates("ACME")
    assert est is not None and est["annual"] == [] and est["price_target"] is not None
    router.add("/price-target-consensus", "", status=500)
    assert provider.get_estimates("ACME") is None
    router.add("/analyst-estimates", '{"symbol": "ACME"}')   # object instead of list
    assert provider.get_estimates("ACME") is None


def test_news_shape_and_fallback_keys(router, provider):
    _happy_router(router)
    news = provider.get_news("acme")
    assert news is not None and len(news) == 2
    assert set(news[0]) == {
        "title", "source", "published_at", "url", "summary", "tickers",
        "topics", "sentiment", "relevance_score",
    }
    assert news[0]["source"] == "Example Wire"
    assert news[0]["url"] == "https://example.invalid/news/1"
    assert news[1]["source"] is None                   # publisher absent
    assert news[1]["url"] == "https://example.invalid/news/2"   # `link` fallback
    assert all(n["tickers"] == ["ACME"] for n in news)
    assert router.calls[0][1]["symbols"] == "ACME"


def test_unsupported_capabilities_return_documented_empties(provider):
    assert provider.get_earnings_transcripts("ACME") is None
    assert provider.get_filings("ACME") is None
    assert provider.get_macro_series("DGS10") is None
    assert provider.list_tickers() == []


# ---------------------------------------------------------------------------
# Failure classes — every method returns None, never raises
# ---------------------------------------------------------------------------

_ALL_METHODS = [
    ("get_sp500_constituents", ()),
    ("get_company_profile", ("ACME",)),
    ("get_quote", ("ACME",)),
    ("get_price_history", ("ACME",)),
    ("get_financial_statements", ("ACME",)),
    ("get_ratios", ("ACME",)),
    ("get_key_metrics", ("ACME",)),
    ("get_earnings", ("ACME",)),
    ("get_estimates", ("ACME",)),
    ("get_news", ("ACME",)),
]

_FAILURES = {
    "429_retry_after": dict(body='{"Error Message": "Limit Reach"}', status=429,
                            headers={"Retry-After": "30"}),
    "401": dict(body='{"Error Message": "Invalid API KEY."}', status=401),
    "500": dict(body="Internal Server Error", status=500),
    "malformed_json": dict(body=_text("malformed_body.txt"), status=200),
    "empty_list": dict(body="[]", status=200),
    "object_not_list": dict(body='{"symbol": "ACME"}', status=200),
    "timeout": dict(exc=httpx.ReadTimeout("timed out")),
    "connect_error": dict(exc=httpx.ConnectError("dns")),
}


_ALL_PATHS = (
    "/sp500-constituent", "/profile", "/shares-float", "/quote",
    "/historical-price-eod/full", "/income-statement",
    "/balance-sheet-statement", "/cash-flow-statement", "/ratios",
    "/key-metrics", "/earnings", "/analyst-estimates",
    "/price-target-consensus", "/news/stock",
)


def _fail_everywhere(router: _Router, failure: str) -> None:
    for path in _ALL_PATHS:
        router.add(path, **_FAILURES[failure])


@pytest.mark.parametrize("failure", sorted(_FAILURES))
@pytest.mark.parametrize("method, args", _ALL_METHODS)
def test_every_method_returns_none_on_failure(router, provider, method, args, failure):
    if (method, failure) == ("get_financial_statements", "object_not_list"):
        pytest.skip("covered by the xfail below — the provider raises here today")
    _fail_everywhere(router, failure)
    assert getattr(provider, method)(*args) is None


@pytest.mark.xfail(
    strict=False,
    reason=(
        "get_financial_statements does `self._get(...) or []` and then "
        "iterates the result, so an HTTP-200 JSON *object* (FMP's "
        "`{\"Error Message\": ...}` shape) is iterated as a dict of keys "
        "and raises AttributeError instead of returning None"
    ),
)
def test_financial_statements_with_error_object_body_returns_none(router, provider):
    _fail_everywhere(router, "object_not_list")
    assert provider.get_financial_statements("ACME") is None


def test_rate_limit_is_not_retried_and_retry_after_is_ignored(router, provider):
    router.add("/quote", **_FAILURES["429_retry_after"])
    assert provider.get_quote("ACME") is None
    assert [c[0] for c in router.calls] == ["/quote"]  # exactly one attempt


def test_status_reflects_configuration_only(router, provider):
    """There is no health hook: a 429 / 401 leaves `status()` unchanged.
    Pinned so a future breaker shows up as an intentional test change."""
    before = provider.status()
    assert before.configured and before.healthy and before.notes == ""
    router.add("/quote", **_FAILURES["401"])
    assert provider.get_quote("ACME") is None
    after = provider.status()
    assert (after.configured, after.healthy, after.notes) == (True, True, "")
    assert "financials" in after.capabilities


def test_wrong_shaped_rows_propagate_as_an_exception(router, provider):
    """A list whose rows are not dicts is the one failure the provider
    does not guard — it raises out of the normaliser. That is tolerable
    only because `data_service._try_chain` catches per-provider
    exceptions; pinned so a change on either side is deliberate."""
    router.add("/quote", '["not-a-dict"]')
    with pytest.raises(AttributeError):
        provider.get_quote("ACME")
