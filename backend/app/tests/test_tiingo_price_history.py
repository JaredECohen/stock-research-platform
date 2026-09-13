"""Tiingo history must request dates: an undated request is latest-only."""
from datetime import date, timedelta

import httpx
import pytest

from app.providers import tiingo_provider
from app.services.outcome_service import PRICE_WINDOW_RUNGS


@pytest.mark.parametrize("days", [1, 60, *PRICE_WINDOW_RUNGS])
def test_history_requests_dates_and_retains_complete_response(monkeypatch, days):
    today = date(2026, 9, 11)

    class Clock(date):
        @classmethod
        def today(cls):
            return today

    monkeypatch.setattr(tiingo_provider, "date", Clock)
    requests = []

    def respond(request):
        requests.append(request)
        params = request.url.params
        assert params["resampleFreq"] == "daily"
        # Simulate the actual endpoint distinction. The old implementation
        # gets one row, even though its final [-days:] slice looks correct.
        start = date.fromisoformat(params.get("startDate", today.isoformat()))
        end = date.fromisoformat(params.get("endDate", today.isoformat()))
        dates = [start + timedelta(days=n) for n in range((end - start).days + 1)]
        return httpx.Response(200, json=[
            {"date": f"{d.isoformat()}T00:00:00.000Z", "close": 100, "adjClose": 99}
            for d in dates if d.weekday() < 5
        ])

    real_client = httpx.Client
    monkeypatch.setattr(tiingo_provider.httpx, "Client", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(respond),
    ))
    provider = tiingo_provider.TiingoProvider()
    provider.api_key = "test-tiingo-key"
    rows = provider.get_price_history("TEST", days)

    assert len(requests) == 1
    assert requests[0].url.params["endDate"] == today.isoformat()
    assert "startDate" in requests[0].url.params
    expected_start = date.fromisoformat(requests[0].url.params["startDate"])
    expected = sum((expected_start + timedelta(days=n)).weekday() < 5 for n in range((today - expected_start).days + 1))
    assert len(rows) == expected
    assert len(rows) >= days
    assert rows[-1]["date"] == today.isoformat()
    assert rows == sorted(rows, key=lambda row: row["date"])
    assert rows[-1]["close"] == 100
    assert rows[-1]["adjusted_close"] == 99
