"""Offline request contracts and complete historical daily-bar responses."""

from datetime import date, datetime, timedelta

import httpx
import pytest

from app.providers import alpha_vantage_provider as alpha
from app.providers import fmp_provider as fmp
from app.providers import polygon_provider as polygon
from app.providers import tiingo_provider as tiingo
from app.providers.price_history import normalize_history

TODAY = date(2026, 9, 13)


@pytest.fixture(autouse=True)
def fixed_date(monkeypatch):
    class Clock(date):
        @classmethod
        def today(cls):
            return TODAY

    for module in (alpha, fmp, polygon, tiingo):
        monkeypatch.setattr(module, "date", Clock)


def mock_http(monkeypatch, respond):
    real_client = httpx.Client
    calls = []

    def handle(req):
        calls.append(req)
        return respond(req)

    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    return calls


def provider(cls):
    p = cls()
    p.api_key = "fake-secret-key"
    return p


def bar(day="2026-09-11", close=100):
    return dict(date=day, open=99, high=101, low=98, close=close, volume=500)


def test_fmp_requests_explicit_two_year_dates_and_retains_all_rows(monkeypatch):
    # Deliberately more returned bars than days: no hidden final slicing.
    source = [bar((TODAY - timedelta(days=n)).isoformat()) for n in range(800)]
    calls = mock_http(monkeypatch, lambda req: httpx.Response(200, json=source))
    result = provider(fmp.FMPProvider).get_price_history("aapl", 740)
    assert len(calls) == 1
    assert calls[0].url.path == "/stable/historical-price-eod/full"
    assert dict(calls[0].url.params) == {
        "symbol": "AAPL",
        "from": (TODAY - timedelta(days=1184)).isoformat(),
        "to": TODAY.isoformat(),
        "apikey": "fake-secret-key",
    }
    assert len(result) == 800
    assert result[0]["date"] == source[-1]["date"]
    assert result[0]["close"] == result[0]["adjusted_close"] == 100


def test_fmp_long_history_windows_are_contiguous_and_failure_is_not_partial(monkeypatch):
    def respond(req):
        start = req.url.params["from"]
        return httpx.Response(200, json=[bar(start)])

    calls = mock_http(monkeypatch, respond)
    p = provider(fmp.FMPProvider)
    rows = p.get_price_history("SPY", 3000)
    assert len(calls) == 3 and len(rows) == 3
    spans = [(date.fromisoformat(c.url.params["from"]), date.fromisoformat(c.url.params["to"])) for c in calls]
    assert spans[0][0] == TODAY - timedelta(days=4800)
    assert spans[-1][1] == TODAY
    assert all((b - a).days <= 1825 for a, b in spans)
    assert all(spans[n][1] + timedelta(days=1) == spans[n + 1][0] for n in range(2))
    n = 0

    def fail_second(path, **params):
        nonlocal n
        n += 1
        return [bar(params["from"])] if n == 1 else None

    monkeypatch.setattr(p, "_get", fail_second)
    assert p.get_price_history("SPY", 3000) is None


@pytest.mark.parametrize("days,size", [(1, "compact"), (100, "compact"), (101, "full"), (740, "full")])
def test_alpha_raw_daily_history_parameters_and_adjustment_truth(monkeypatch, days, size):
    series = {
        "2024-01-02": {"1. open": "99", "2. high": "102", "3. low": "98", "4. close": "100", "5. volume": "500"},
        "2026-09-11": {"4. close": "200"},
    }
    calls = mock_http(monkeypatch, lambda req: httpx.Response(200, json={"Time Series (Daily)": series}))
    p = provider(alpha.AlphaVantageProvider)
    rows = p.get_price_history("spy", days)
    assert calls[0].url.params["function"] == "TIME_SERIES_DAILY"
    assert calls[0].url.params["outputsize"] == size
    assert calls[0].url.params["symbol"] == "SPY"
    assert len(rows) == 2  # full/compact source responses are never sliced
    assert rows[0]["close"] == 100 and rows[0]["adjusted_close"] is None
    assert p.price_history_provenance["close_basis"] == "raw_as_traded"


@pytest.mark.parametrize("marker", ["Note", "Information", "Error Message"])
def test_alpha_body_errors_are_unavailable_and_body_is_not_logged(monkeypatch, caplog, marker):
    mock_http(monkeypatch, lambda req: httpx.Response(200, json={marker: "fake-secret-key private provider detail"}))
    assert provider(alpha.AlphaVantageProvider).get_price_history("ACME", 740) is None
    assert marker in caplog.text
    assert "fake-secret-key" not in caplog.text and "private provider detail" not in caplog.text


def polygon_bar(utc="2026-09-11T04:00:00+00:00", close=100):
    return {"t": int(datetime.fromisoformat(utc).timestamp() * 1000), "c": close, "o": 99, "h": 101, "l": 98, "v": 500}


def test_polygon_paginated_history_keeps_all_bars_and_et_date(monkeypatch):
    def respond(req):
        if req.url.params.get("cursor") == "page2":
            return httpx.Response(
                200, json={"status": "OK", "adjusted": True, "results": [polygon_bar("2026-09-11T04:00:00+00:00", 101)]}
            )
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "adjusted": True,
                "results": [polygon_bar("2026-09-11T02:00:00+00:00")],
                "next_url": "https://api.massive.com/v2/aggs/ticker/ACME/range/1/day/2026-09-11/2026-09-13?cursor=page2&apiKey=do-not-forward",
            },
        )

    calls = mock_http(monkeypatch, respond)
    rows = provider(polygon.PolygonProvider).get_price_history("ACME", 1)
    assert len(calls) == 2 and len(rows) == 2
    assert [r["date"] for r in rows] == ["2026-09-10", "2026-09-11"]
    assert all(c.url.host == "api.polygon.io" for c in calls)
    assert all(c.url.params["apiKey"] == "fake-secret-key" for c in calls)
    assert all(
        c.url.params["adjusted"] == "true" and c.url.params["sort"] == "asc" and c.url.params["limit"] == "50000"
        for c in calls
    )
    assert rows[1]["close"] == rows[1]["adjusted_close"] == 101


@pytest.mark.parametrize("suffix", ["off_host", "wrong_ticker", "cycle", "second_page_failure", "unadjusted"])
def test_polygon_rejects_unreliable_pages_instead_of_partial_history(monkeypatch, suffix, caplog):
    good = "https://api.polygon.io/v2/aggs/ticker/ACME/range/1/day/2026-09-11/2026-09-13?cursor=one"
    n = 0

    def respond(req):
        nonlocal n
        n += 1
        if suffix == "second_page_failure" and n == 2:
            return httpx.Response(429)
        next_url = good
        if suffix == "off_host":
            next_url = good.replace("api.polygon.io", "evil.example")
        if suffix == "wrong_ticker":
            next_url = good.replace("ACME", "OTHER")
        return httpx.Response(
            200, json={"results": [polygon_bar()], "adjusted": suffix != "unadjusted", "next_url": next_url}
        )

    calls = mock_http(monkeypatch, respond)
    assert provider(polygon.PolygonProvider).get_price_history("ACME", 740) is None
    assert len(calls) <= 2
    assert "fake-secret-key" not in caplog.text


def test_tiingo_sorts_deduplicates_and_reports_bad_or_short_coverage(monkeypatch, caplog):
    mock_http(
        monkeypatch,
        lambda req: httpx.Response(
            200,
            json=[
                dict(bar(), adjClose=95),
                bar("2026-09-10", 99),
                dict(bar(), adjClose=95),
                bar("invalid"),
                bar("2026-09-09", -1),
            ],
        ),
    )
    rows = provider(tiingo.TiingoProvider).get_price_history("ACME", 740)
    assert [r["date"] for r in rows] == ["2026-09-10", "2026-09-11"]
    assert rows[1]["adjusted_close"] == 95
    assert "invalid=2" in caplog.text and "completeness=unverified" in caplog.text


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 0, -1, True, "nan"])
def test_invalid_closes_never_become_price_history(bad):
    assert (
        normalize_history(
            [bar(close=bad)], provider="test", ticker="ACME", start=TODAY - timedelta(days=3), end=TODAY, log=fmp.log
        )
        is None
    )


def test_conflicting_duplicate_dates_reject_history(caplog):
    assert (
        normalize_history(
            [bar(close=1), bar(close=2)],
            provider="test",
            ticker="ACME",
            start=TODAY - timedelta(days=3),
            end=TODAY,
            log=fmp.log,
        )
        is None
    )
    assert "conflicting_dates=1" in caplog.text


@pytest.mark.parametrize("days", [0, -1, True, 1.5, 10**20])
@pytest.mark.parametrize(
    "cls", [fmp.FMPProvider, tiingo.TiingoProvider, polygon.PolygonProvider, alpha.AlphaVantageProvider]
)
def test_invalid_horizons_do_not_make_requests(monkeypatch, cls, days):
    calls = mock_http(monkeypatch, lambda req: pytest.fail("unexpected request"))
    assert provider(cls).get_price_history("ACME", days) is None
    assert not calls
