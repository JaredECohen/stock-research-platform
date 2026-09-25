"""W5b — `GET /api/quotes`, the route behind the live-quote chip."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.auth import policy
from app.database import SessionLocal
from app.main import app
from app.models import Company
from app.schemas.quotes import QuotesOut
from app.services import quote_service

client = TestClient(app)


@pytest.fixture
def known(monkeypatch) -> list[str]:
    stamp = time.perf_counter_ns() % 10**7
    names = [f"QR{stamp}A", f"QR{stamp}B"]
    with SessionLocal() as db:
        for t in names:
            db.merge(Company(ticker=t, company_name=t, sector="Test", industry="Test", universe_tier="data_only"))
        db.commit()
    prices = {names[0]: 101.25}

    def fetch_one(ticker, _ds):
        return ({"ticker": ticker, "price": prices[ticker]}, "fmp") if ticker in prices else (None, None)

    monkeypatch.setattr(quote_service, "_fetch_one", fetch_one)
    return names


def test_quotes_shape_and_order(known):
    a, b = known
    resp = client.get(f"/api/quotes?tickers={b.lower()},ZZZZQ,{a}, {b}")
    assert resp.status_code == 200, resp.text
    body = QuotesOut.model_validate(resp.json())
    assert [q.ticker for q in body.quotes] == [b, "ZZZZQ", a]           # request order, deduped, upper-cased
    by = {q.ticker: q for q in body.quotes}
    assert by["ZZZZQ"].source == "unavailable" and by["ZZZZQ"].reason == "unknown_ticker"
    assert by[a].source == "live" and by[a].price == 101.25 and by[a].provider == "fmp"
    assert by[b].source == "unavailable" and by[b].reason == "no_stored_close"
    assert body.ttl_seconds_in_session == 900
    market = resp.json()["market"]
    assert set(market) == {"is_open", "reason", "session_date", "session_open", "session_close",
                           "early_close", "next_open"}
    # Aware UTC on the wire, so a browser never reads a time as local.
    assert market["session_open"].endswith("Z") and market["next_open"].endswith("Z")


def test_quotes_rejects_too_many_and_malformed():
    too_many = ",".join(f"T{i}" for i in range(51))
    for query in (f"tickers={too_many}", "tickers=A;B", "tickers=", "tickers=,,", ""):
        resp = client.get(f"/api/quotes?{query}")
        assert resp.status_code == 422, (query, resp.status_code)
    assert client.get("/api/quotes?tickers=" + "A" * 1001).status_code == 422


def test_quotes_cache_control_header(known):
    resp = client.get(f"/api/quotes?tickers={known[0]}")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "private, max-age=60"


def test_dcf_lab_saved_run_writes_no_dcf_model_row(monkeypatch):
    """The DCF Lab keeps POSTed saved assumptions verbatim (W5b critique).

    The engine defaults now read the live quote, so a lab that re-priced the
    saved assumptions to that quote before `runDCF` would, whenever the rest
    matched the defaults, hand `build_dcf` the default set, and `build_dcf`
    persists a default-equal run as a new `DCFModel` version. Pinned here:
    the defaults carry the live price, a verbatim saved run returns the
    saved price and writes nothing, and the re-priced set WOULD be the
    default set (which is why the upside against the live price is
    computed for display only, in the browser).
    """
    from sqlalchemy import func, select

    from app.models import DCFModel
    from app.schemas import DCFAssumptions

    ticker = "COST"
    created = False
    with SessionLocal() as db:
        if db.get(Company, ticker) is None:
            db.add(Company(ticker=ticker, company_name="Costco", sector="Consumer Staples",
                           industry="Retail", universe_tier="data_only"))
            db.commit()
            created = True
    live = 987.65
    monkeypatch.setattr(
        quote_service, "_fetch_one",
        lambda t, _ds: ({"ticker": t, "price": live}, "fmp") if t == ticker else (None, None),
    )

    def model_rows() -> int:
        with SessionLocal() as db:
            return db.execute(select(func.count()).select_from(DCFModel).where(DCFModel.ticker == ticker)).scalar_one()

    try:
        defaults = client.get(f"/api/dcf/{ticker}/default-assumptions")
        assert defaults.status_code == 200, defaults.text
        default_body = defaults.json()
        assert default_body["current_price"] == live
        saved = {**default_body, "current_price": 850.0}   # the price the version was saved at
        before = model_rows()
        resp = client.post(f"/api/dcf/{ticker}", json=saved)
        assert resp.status_code == 200, resp.text
        assert resp.json()["current_price"] == 850.0
        assert model_rows() == before
        repriced = DCFAssumptions.model_validate({**saved, "current_price": live})
        assert repriced.model_dump() == DCFAssumptions.model_validate(default_body).model_dump()
    finally:
        if created:
            with SessionLocal() as db:
                row = db.get(Company, ticker)
                if row is not None:
                    db.delete(row)
                    db.commit()


def test_quotes_route_has_a_customer_policy():
    """Signed-in (free) under the wall: anonymous visitors, like the public
    sample pages, never draw on the quote quota."""
    rows = {(m, p): pol for m, p, pol in policy.ROUTES}
    assert rows[("GET", "/api/quotes")].level == policy.AUTHENTICATED


def test_quotes_calendar_unavailable_is_503(known, monkeypatch):
    """tzdata missing from the image: a named 503 the chip can ignore, not a
    500 with a traceback, and no provider is called first."""
    from app.finance import market_calendar

    def unavailable(*_a, **_k):
        raise market_calendar.CalendarUnavailable("no tzdata")

    calls: list[str] = []
    monkeypatch.setattr(market_calendar, "market_state", unavailable)
    monkeypatch.setattr(quote_service, "_fetch_one", lambda t, _ds: calls.append(t) or (None, None))
    resp = client.get(f"/api/quotes?tickers={known[0]}")
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == "market calendar unavailable"
    assert calls == []
