"""W5b — inside a memo run every quote read is at most 60 s old, end to end.

The quote cache moved from a fixed 60 s TTL to the calendar policy (15
minutes in session). `price_at_memo` and the memo DCF's `current_price` must
not start lagging by up to 15 minutes because of that, so `quote_service`
applies a 60 s floor whenever `safe_runner.in_memo_run()` (contract C6) is
true. That detector is only as good as its invariant: the memo run is the
one place a `DegradationLog` is activated. This test runs the real
`run_stock_memo` and spies at the `price_at_memo` capture in `graph.py`, so
a refactor that moves the activation, or reads the price outside it, fails
here rather than silently changing what a stored memo's price means.
"""
from __future__ import annotations

import sys

from app.agents import graph
from app.agents.safe_runner import in_memo_run
from app.services import market_data_service, quote_service


def test_memo_run_applies_60s_floor_end_to_end(monkeypatch):
    floors_seen: list[int | None] = []
    real_get_quotes = quote_service.get_quotes

    def spy_get_quotes(tickers, **kwargs):
        floors_seen.append(quote_service.effective_floor(kwargs.get("max_age_seconds")))
        return real_get_quotes(tickers, **kwargs)

    monkeypatch.setattr(quote_service, "get_quotes", spy_get_quotes)

    captures: list[dict] = []
    real_price = market_data_service.get_current_price

    def spy_price(ticker):
        caller = sys._getframe(1).f_code.co_filename
        start = len(floors_seen)
        price = real_price(ticker)
        captures.append({
            "from_graph": caller.endswith("graph.py"),
            "in_memo_run": in_memo_run(),
            "floors": floors_seen[start:],
            "price": price,
        })
        return price

    # graph.py imports get_current_price at the capture site, at call time.
    monkeypatch.setattr(market_data_service, "get_current_price", spy_price)

    assert quote_service.effective_floor() is None
    memo = graph.run_stock_memo("NVDA")

    at_memo = [c for c in captures if c["from_graph"]]
    assert at_memo, "the price_at_memo capture no longer goes through get_current_price"
    capture = at_memo[-1]
    assert capture["in_memo_run"] is True
    # The quote read behind price_at_memo ran with the 60 s memo floor.
    assert capture["floors"] == [quote_service.MEMO_QUOTE_MAX_AGE_SECONDS]
    assert memo.price_at_memo == capture["price"]
    # And the floor is gone once the run returns (the ContextVar is reset).
    assert in_memo_run() is False
    assert quote_service.effective_floor() is None
