"""Capture ``frontend/src/test/fixtures/quotes.wire.json`` — the wire fixture
every LiveQuote UI test reads (W5b).

Usage (from ``backend/``, against a THROWAWAY sqlite database)::

    ENABLE_LIVE_DATA=false USE_DEMO_DATA=true \\
    OPENAI_API_KEY="" ANTHROPIC_API_KEY="" GEMINI_API_KEY="" \\
    DATABASE_URL="sqlite:////tmp/quotes-fixture.db" \\
    python -m app.scripts.capture_quotes_ui_fixture

The fixture is NOT hand-written: every body is what the real
``GET /api/quotes`` route served through ``TestClient``, with the real
``quote_service`` policy, cache, stale ledger and durable-close fallback.
Only two things are pinned, and ``meta`` says so: the clock (frozen in
``quote_service._now`` and ``provider_cache._now``) and the provider seam
(``quote_service._fetch_one`` answers deterministic payloads; the batch
path is off, as it is in demo mode). ``app/tests/test_quotes_ui_fixture_contract.py``
re-runs ``build()`` and requires the committed file to equal it EXACTLY,
so a renamed key or a changed label rule fails with the re-capture command
before the chip can keep rendering a shape the API no longer sends.

The five chip states, one body each:

* ``live_open``: 10:45 ET on a Wednesday session, a fresh provider quote
  timed 10:44:30 ET;
* ``live_at_close``: 17:00 ET the same day, a provider quote timed exactly
  at the 16:00 ET close (the chip says "At close" only when the provider's
  time is at or after the session close);
* ``stale``: 10:45 ET, the cached row is 20 minutes old and the provider
  misses: served, labelled, and ledgered as a stale serve;
* ``eod_close``: 10:45 ET, no cached row and the provider misses: the last
  close from ``daily_prices`` (DB-only);
* ``unavailable``: a symbol that is not a company, never sent to a provider.

The capture's own rows (the ``QT*`` companies, their quote rows and the one
stored close) are deleted before and after, so the run is repeatable on the
test database and leaves nothing behind. It refuses to run with live data
enabled or against anything but a local sqlite file (``dbguard``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Set BEFORE any app module is imported (they are imported lazily below).
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("ENABLE_LIVE_DATA", "false")
os.environ.setdefault("USE_DEMO_DATA", "true")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "frontend" / "src" / "test" / "fixtures" / "quotes.wire.json"

LIVE, AT_CLOSE, STALE, EOD, UNKNOWN = "QTLIVE", "QTCLOSE", "QTSTALE", "QTEOD", "QTNONE"
COMPANIES = (LIVE, AT_CLOSE, STALE, EOD)
# Wednesday 2026-09-23: 10:45 ET (session open) and 17:00 ET (after the close).
OPEN_NOW = datetime(2026, 9, 23, 14, 45)
CLOSED_NOW = datetime(2026, 9, 23, 21, 0)
STALE_FETCHED_AT = datetime(2026, 9, 23, 14, 25)
STORED_CLOSE = {"date": "2026-09-22", "close": 45.67, "volume": 1_250_000}


def _epoch(naive_utc: datetime) -> int:
    return int(naive_utc.replace(tzinfo=UTC).timestamp())


# What the stubbed provider answers per symbol (None = the provider misses).
PROVIDER_ANSWERS: dict[str, dict[str, Any] | None] = {
    LIVE: {"ticker": LIVE, "price": 123.45, "previous_close": 121.0, "change": 2.45,
           "change_pct": 2.0248, "day_low": 120.8, "day_high": 124.1, "volume": 1_830_000.0,
           "timestamp": _epoch(datetime(2026, 9, 23, 14, 44, 30))},
    AT_CLOSE: {"ticker": AT_CLOSE, "price": 88.1, "previous_close": 89.0, "change": -0.9,
               "change_pct": -1.0112, "day_low": 87.6, "day_high": 89.4, "volume": 2_400_000.0,
               "timestamp": _epoch(datetime(2026, 9, 23, 20, 0))},
    STALE: None,
    EOD: None,
}
STALE_ROW = {"ticker": STALE, "price": 64.2, "previous_close": 63.5, "change": 0.7, "change_pct": 1.1024,
             "day_low": 63.4, "day_high": 64.5, "volume": 910_000.0,
             "timestamp": _epoch(datetime(2026, 9, 23, 14, 24, 40)), "provider": "fmp"}

BODIES: tuple[tuple[str, datetime, list[str]], ...] = (
    ("live_open", OPEN_NOW, [LIVE]),
    ("live_at_close", CLOSED_NOW, [AT_CLOSE]),
    ("stale", OPEN_NOW, [STALE]),
    ("eod_close", OPEN_NOW, [EOD]),
    ("unavailable", OPEN_NOW, [UNKNOWN]),
)
GENERATED_BY = (
    "python -m app.scripts.capture_quotes_ui_fixture — the real GET /api/quotes route through "
    "TestClient, with a frozen clock and a stubbed provider seam"
)


def _clear() -> None:
    from sqlalchemy import delete

    from app.database import SessionLocal
    from app.models import Company, DailyPrice, ProviderCache

    everything = (*COMPANIES, UNKNOWN)
    with SessionLocal() as db:
        db.execute(delete(ProviderCache).where(ProviderCache.capability == "quote", ProviderCache.key.in_(everything)))
        db.execute(delete(DailyPrice).where(DailyPrice.ticker.in_(everything)))
        db.execute(delete(Company).where(Company.ticker.in_(everything)))
        db.commit()


def _seed() -> None:
    from app.database import SessionLocal
    from app.models import Company
    from app.services import price_history_service

    with SessionLocal() as db:
        for ticker in COMPANIES:
            db.add(Company(ticker=ticker, company_name=f"{ticker} Fixture Co", sector="Industrials",
                           industry="Machinery", universe_tier="data_only"))
        db.commit()
    price_history_service.persist_prices(EOD, [STORED_CLOSE], source="fmp")


class _Clock:
    def __init__(self) -> None:
        self.now = OPEN_NOW

    def __call__(self) -> datetime:
        return self.now


@contextmanager
def _pinned(clock: _Clock) -> Iterator[None]:
    """Freeze the clocks, stub the provider seam, take any test provider
    off (so the cache path runs), and put everything back afterwards."""
    from app.services import provider_cache, quote_service
    from app.services.data_service import get_data_service

    ds = get_data_service()
    saved = (quote_service._now, provider_cache._now, quote_service._fetch_one,
             quote_service._batch_enabled, ds._test_provider)

    def fetch_one(ticker: str, _ds: Any) -> tuple[dict[str, Any] | None, str | None]:
        answer = PROVIDER_ANSWERS.get(ticker)
        return (dict(answer), "fmp") if answer else (None, None)

    quote_service._now = clock  # type: ignore[assignment]
    provider_cache._now = clock  # type: ignore[assignment]
    quote_service._fetch_one = fetch_one  # type: ignore[assignment]
    quote_service._batch_enabled = lambda _ds: False  # type: ignore[assignment]
    ds.register_test_provider(None)
    try:
        yield
    finally:
        (quote_service._now, provider_cache._now, quote_service._fetch_one,  # type: ignore[assignment]
         quote_service._batch_enabled) = saved[:4]
        ds.register_test_provider(saved[4])


def build() -> dict[str, Any]:
    """Seed, capture every body through the real route, clean up."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services import provider_cache

    client = TestClient(app)
    clock = _Clock()
    out: dict[str, Any] = {
        "meta": {
            "generated_by": GENERATED_BY,
            "endpoint": "GET /api/quotes",
            "bodies": {name: {"now_utc": now.isoformat() + "Z", "tickers": tickers} for name, now, tickers in BODIES},
            "pinned": ("the clock (quote_service._now, provider_cache._now) and the provider seam "
                       "(quote_service._fetch_one); everything else is the real route and service"),
            "note": "Nothing was edited after capture: every value is what the API served.",
        }
    }
    _clear()
    try:
        _seed()
        with _pinned(clock):
            clock.now = STALE_FETCHED_AT
            provider_cache.put("quote", STALE, dict(STALE_ROW))
            for name, now, tickers in BODIES:
                clock.now = now
                response = client.get("/api/quotes", params={"tickers": ",".join(tickers)})
                if response.status_code != 200:
                    raise RuntimeError(f"/api/quotes {name} answered {response.status_code}: {response.text[:200]}")
                out[name] = response.json()
    finally:
        _clear()
    return out


def _refuse_unsafe_environment() -> str | None:
    from app.config import settings
    from app.tests import dbguard

    if settings.enable_live_data or not settings.use_demo_data:
        return "refusing to capture with live data enabled — rerun with ENABLE_LIVE_DATA=false USE_DEMO_DATA=true"
    if not settings.database_url.startswith("sqlite"):
        return "refusing: point DATABASE_URL at a throwaway sqlite file, e.g. sqlite:////tmp/quotes-fixture.db"
    return dbguard.refusal(settings.database_url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    args = parser.parse_args(argv)
    refusal = _refuse_unsafe_environment()
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    from app.database import init_db

    init_db()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    print("now run: npx vitest run in frontend/, and pytest app/tests/test_quotes_ui_fixture_contract.py")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
