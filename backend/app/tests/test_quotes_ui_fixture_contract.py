"""W5b — the LiveQuote UI fixture is the API's own output, and stays so.

`frontend/src/test/fixtures/quotes.wire.json` was produced by
`app/scripts/capture_quotes_ui_fixture.py`: the real `GET /api/quotes` route
through `TestClient`, with the clock frozen and the provider seam stubbed.
Every Vitest assertion about the chip is made against that file, which is
only worth anything while it is still what the API serves. The capture is
deterministic (fixed clock, fixed provider answers, its own rows cleared
before and after), so the comparison is EXACT, as for the Track Record
fixture; the field-set checks name the drifted key when it fails.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.schemas.quotes import MarketStateOut, QuoteOut, QuotesOut
from app.scripts import capture_quotes_ui_fixture as producer

FIXTURE = Path(__file__).resolve().parents[3] / "frontend" / "src" / "test" / "fixtures" / "quotes.wire.json"
RECAPTURE = (
    "re-capture it: cd backend && ENABLE_LIVE_DATA=false USE_DEMO_DATA=true "
    'OPENAI_API_KEY="" ANTHROPIC_API_KEY="" GEMINI_API_KEY="" '
    'DATABASE_URL="sqlite:////tmp/quotes-fixture.db" '
    "python -m app.scripts.capture_quotes_ui_fixture"
)
BODY_NAMES = [name for name, _now, _tickers in producer.BODIES]


def _committed() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_equals_producer():
    produced = json.loads(json.dumps(producer.build()))
    assert produced == _committed(), f"quotes.wire.json no longer matches the API; {RECAPTURE}"


def test_every_body_validates_and_field_sets_match_both_ways():
    body = _committed()
    assert set(body) == {"meta", *BODY_NAMES}
    for name in BODY_NAMES:
        raw = body[name]
        QuotesOut.model_validate(raw)
        assert set(raw) == set(QuotesOut.model_fields), (name, RECAPTURE)
        assert set(raw["market"]) == set(MarketStateOut.model_fields), (name, RECAPTURE)
        for quote in raw["quotes"]:
            assert set(quote) == set(QuoteOut.model_fields), (name, RECAPTURE)


def test_fixture_exercises_every_chip_state():
    body = _committed()
    sources = {name: body[name]["quotes"][0]["source"] for name in BODY_NAMES}
    assert sources == {"live_open": "live", "live_at_close": "live", "stale": "stale",
                       "eod_close": "eod_close", "unavailable": "unavailable"}
    assert body["live_open"]["market"]["is_open"] is True
    closed = body["live_at_close"]
    assert closed["market"]["is_open"] is False
    # "At close" is only honest when the provider's own time reaches the close.
    price_time = datetime.fromisoformat(closed["quotes"][0]["price_time"])
    assert price_time >= datetime.fromisoformat(closed["market"]["session_close"])
    assert body["unavailable"]["quotes"][0]["reason"] == "unknown_ticker"
    assert body["eod_close"]["quotes"][0]["provider"] == "daily_prices"
