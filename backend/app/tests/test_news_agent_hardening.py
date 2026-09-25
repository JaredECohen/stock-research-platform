"""N34 — the Gemini news path is governed before it is trusted.

Gemini items are model-written JSON from a search-grounded call: the title,
url and date are what the model typed. These pin the 2026-09-25 news-trace
fixes (plans/2026-09-24/news_trace_2026-09-25.json, N3(a) + critique):

- the domain allow-list runs BEFORE the provider-fallback decision, so a
  relevant-but-blocked Gemini answer falls back instead of leaving zero alerts;
- a Gemini item with no URL is dropped (it bypassed the allow-list);
- the prompt asks for 7 days, not 60 (assumption A10);
- published dates are parsed; an unreadable Gemini date is kept for display
  but flagged `date_unknown` (L8), and provider dates are normalised to ISO.

Network is blocked by the harness; every Gemini and provider call is faked.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app.agents import llm, news_agent
from app.cache import cache_get
from app.config import settings

TICKER = "SOCO"  # a ticker no other test writes news for


@pytest.fixture
def gemini_on(monkeypatch):
    """Gemini configured via the direct API; company name lookup and the
    provider feed faked; breaker clear."""
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(news_agent, "_company_name", lambda t: "Southern Company")
    news_agent.reload_domain_lists()
    llm.reset_circuit_breaker()
    yield
    llm.reset_circuit_breaker()


def _fake_gemini(monkeypatch, items):
    prompts: list[str] = []

    def fake(prompt, **kwargs):
        prompts.append(prompt)
        return {"items": items}

    monkeypatch.setattr(llm, "gemini_chat_json", fake)
    return prompts


def _fake_provider(monkeypatch, items):
    calls: list[str] = []

    def fake(ticker):
        calls.append(ticker)
        return [dict(it) for it in items]

    monkeypatch.setattr(news_agent.news_service, "get_news", fake)
    return calls


def _today_iso() -> str:
    return datetime.utcnow().date().isoformat()


def test_domain_blocked_gemini_items_fall_back_to_provider(gemini_on, monkeypatch):
    # Both Gemini items are relevant (ticker in the title) but on a domain the
    # allow-list does not carry. Before the fix the fallback decision ran
    # first, saw "relevant items", skipped the provider, then the allow-list
    # removed everything: zero alerts for the ticker.
    _fake_gemini(monkeypatch, [
        {"title": "SOCO signs nuclear deal", "summary": "s", "url": "https://benzinga.com/a",
         "published_at": _today_iso()},
        {"title": "SOCO raises capex", "summary": "s", "url": "https://benzinga.com/b",
         "published_at": _today_iso()},
    ])
    calls = _fake_provider(monkeypatch, [
        {"title": "Southern Company files 8-K", "summary": "p", "url": "https://www.reuters.com/x",
         "published_at": _today_iso()},
    ])

    alerts = news_agent.run(TICKER, force_refresh=True)

    assert calls == [TICKER]
    assert [a.source for a in alerts] == ["news_service"]
    assert alerts[0].title == "Southern Company files 8-K"


def test_urlless_gemini_item_dropped(gemini_on, monkeypatch):
    # A URL-less item used to pass `_filter_grounded_sources` untouched, so a
    # model-written claim with no checkable source reached news_hot.
    _fake_gemini(monkeypatch, [
        {"title": "SOCO invented a claim", "summary": "no source", "url": "",
         "published_at": _today_iso()},
        {"title": "SOCO signs nuclear deal", "summary": "sourced", "url": "https://www.reuters.com/n",
         "published_at": _today_iso()},
    ])
    _fake_provider(monkeypatch, [])

    report: dict = {}
    alerts = news_agent.run(TICKER, force_refresh=True, report=report)

    assert [a.title for a in alerts] == ["SOCO signs nuclear deal"]
    assert alerts[0].source == "gemini"
    assert report["origin"] == "gemini"
    stored = cache_get(f"news_hot:{TICKER}", "news_hot").payload["alerts"]
    assert [a["title"] for a in stored] == ["SOCO signs nuclear deal"]


def test_only_urlless_gemini_items_fall_back_to_provider(gemini_on, monkeypatch):
    _fake_gemini(monkeypatch, [
        {"title": "SOCO invented a claim", "summary": "no source", "url": "", "published_at": _today_iso()},
    ])
    _fake_provider(monkeypatch, [
        {"title": "Southern Company update", "summary": "p", "url": "https://reuters.com/y"},
    ])
    alerts = news_agent.run(TICKER, force_refresh=True)
    assert [(a.title, a.source) for a in alerts] == [("Southern Company update", "news_service")]


def test_gemini_window_7_days(gemini_on, monkeypatch):
    prompts = _fake_gemini(monkeypatch, [])
    _fake_provider(monkeypatch, [])

    news_agent.run(TICKER, force_refresh=True)

    (prompt,) = prompts
    seven = (date.today() - timedelta(days=7)).isoformat()
    sixty = (date.today() - timedelta(days=60)).isoformat()
    assert f"since {seven}" in prompt
    assert sixty not in prompt


def test_unparseable_gemini_date_is_kept_but_flagged(gemini_on, monkeypatch):
    _fake_gemini(monkeypatch, [
        {"title": "SOCO signs nuclear deal", "summary": "s", "url": "https://reuters.com/n",
         "published_at": "recently"},
        {"title": "SOCO raises capex", "summary": "s", "url": "https://reuters.com/c",
         "published_at": _today_iso()},
    ])
    _fake_provider(monkeypatch, [])

    alerts = news_agent.run(TICKER, force_refresh=True)

    # Kept for display, with the model's own text ...
    assert [a.published_at for a in alerts] == ["recently", _today_iso()]
    stored = cache_get(f"news_hot:{TICKER}", "news_hot").payload["alerts"]
    # ... and flagged; a dated item keeps its exact old shape (no key).
    assert stored[0]["date_unknown"] is True
    assert "date_unknown" not in stored[1]


def test_provider_compact_date_is_normalized_to_iso(gemini_on, monkeypatch):
    # Alpha Vantage writes `20260921T214039`; readers that parse ISO (the
    # weekly industry snapshot, the news panel) could not read it.
    monkeypatch.setattr(settings, "gemini_api_key", "")
    _fake_provider(monkeypatch, [
        {"title": "Southern Company update", "summary": "p", "url": "https://reuters.com/y",
         "published_at": "20260921T214039"},
        {"title": "Southern Company dividend", "summary": "p", "url": "https://reuters.com/z",
         "published_at": "2026-09-20"},
    ])
    alerts = news_agent.run(TICKER, force_refresh=True)
    assert [a.published_at for a in alerts] == ["2026-09-21T21:40:39Z", "2026-09-20"]


@pytest.mark.parametrize("value,expected", [
    ("2026-09-21T21:40:39Z", datetime(2026, 9, 21, 21, 40, 39)),
    ("2026-09-21T23:40:39+02:00", datetime(2026, 9, 21, 21, 40, 39)),
    ("20260921T214039", datetime(2026, 9, 21, 21, 40, 39)),
    ("Mon, 21 Sep 2026 21:40:39 GMT", datetime(2026, 9, 21, 21, 40, 39)),
    # A bare date is the LATEST instant it can mean, so a same-day story is
    # never judged older than something written earlier that day.
    ("2026-09-21", datetime(2026, 9, 21, 23, 59, 59, 999999)),
    ("September 21, 2026", datetime(2026, 9, 21, 23, 59, 59, 999999)),
    ("recently", None), ("", None), (None, None), ("2999-01-01", None),
])
def test_parse_published_at(value, expected):
    assert news_agent.parse_published_at(value) == expected


def test_breaker_open_is_reported_not_mistaken_for_no_news(gemini_on, monkeypatch):
    monkeypatch.setattr(llm, "_breaker_open", lambda provider: provider == "gemini")
    monkeypatch.setattr(llm, "gemini_chat_json", lambda *a, **k: None)
    _fake_provider(monkeypatch, [])
    report: dict = {}
    assert news_agent.run(TICKER, force_refresh=True, report=report) == []
    assert report == {"gemini_skipped": "breaker_open", "origin": "empty"}
