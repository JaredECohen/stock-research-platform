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


# --- relevance: BK / BNY (production: "all 5 Gemini items for BK failed
# relevance filter") -----------------------------------------------------

BK_NAME = "The Bank of New York Mellon Corporation"
BK_TOKENS = news_agent._name_tokens(BK_NAME)
_REUTERS = "https://www.reuters.com/business/finance/story"


@pytest.mark.parametrize("title", [
    "BNY beats third-quarter profit estimates on fee growth",
    "BNY to buy stake in fintech firm",
    "BNY names new CFO",
])
def test_bny_headlines_are_about_bk(title):
    # A 2-letter ticker is not matched and BNY is not in the legal name, so
    # every "BNY ..." headline used to fail relevance.
    assert news_agent._is_about_company({"title": title, "url": _REUTERS}, "BK", BK_TOKENS)


@pytest.mark.parametrize("title", [
    "Big bank stocks rally after Fed decision",
    "Regional bank earnings roundup: 9 lenders to watch",
    "New York Fed survey shows inflation expectations ease",
    "Bankruptcy filings rise in August",
])
def test_generic_bank_and_new_york_stories_are_not_about_bk(title):
    # "bank" and "york" matched as substrings, so these counted as BK news.
    assert not news_agent._is_about_company({"title": title, "url": _REUTERS}, "BK", BK_TOKENS)


@pytest.mark.parametrize("ticker,name,title,expected", [
    ("C", "Citigroup Inc.", "Citi to cut 2,000 more jobs", True),
    ("BAC", "Bank of America Corporation", "America's regional banks brace for rules", False),
    ("MS", "Morgan Stanley", "JPMorgan tops estimates", False),
])
def test_relevance_matches_whole_words_and_press_names(ticker, name, title, expected):
    tokens = news_agent._name_tokens(name)
    assert news_agent._is_about_company({"title": title, "url": _REUTERS}, ticker, tokens) is expected


@pytest.mark.parametrize("ticker,name,title,expected", [
    ("BK", BK_NAME, "BNY Mellon reports record custody assets", True),
    ("C", "Citigroup Inc.", "Citizens Financial raises outlook", False),
    ("GOOG", "Alphabet Inc.", "DOJ wins search remedy against Google", True),
    ("XOM", "Exxon Mobil Corporation", "ExxonMobil raises Permian output", True),
    ("BAC", "Bank of America Corporation", "Bank of America beats on trading", True),
    ("SOCO", "Southern Company", "Southern Company files 8-K", True),
])
def test_whole_word_matching_keeps_what_substrings_matched(ticker, name, title, expected):
    # Guards, not regressions: substring matching got these right, some by
    # accident ("goog" inside "google", "exxon" inside "exxonmobil"). Whole
    # words alone would lose them; the press alias, the run-together name
    # and the whole-name phrase are what keep them.
    tokens = news_agent._name_tokens(name)
    assert news_agent._is_about_company({"title": title, "url": _REUTERS}, ticker, tokens) is expected


def test_bny_items_reach_news_hot_for_bk(monkeypatch):
    # End to end on both paths: Gemini's BNY items survive; and when Gemini
    # has nothing, the provider fallback (same filter) keeps its BNY item
    # and drops the sector roundup.
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(news_agent, "_company_name", lambda t: BK_NAME)
    news_agent.reload_domain_lists()
    llm.reset_circuit_breaker()
    _fake_gemini(monkeypatch, [
        {"title": "BNY reports third-quarter results", "summary": "s",
         "url": "https://www.reuters.com/a", "published_at": _today_iso()},
        {"title": "BNY to buy wealth unit", "summary": "s",
         "url": "https://www.bloomberg.com/b", "published_at": _today_iso()},
    ])
    _fake_provider(monkeypatch, [])
    report: dict = {}
    alerts = news_agent.run("BK", force_refresh=True, report=report)
    assert [a.title for a in alerts] == ["BNY reports third-quarter results", "BNY to buy wealth unit"]
    assert report["origin"] == "gemini"

    _fake_gemini(monkeypatch, [])
    _fake_provider(monkeypatch, [
        {"title": "BNY names new CFO", "summary": "p", "url": "https://www.reuters.com/c"},
        {"title": "Big bank stocks rally after Fed decision", "summary": "p",
         "url": "https://www.reuters.com/d"},
    ])
    alerts = news_agent.run("BK", force_refresh=True)
    assert [(a.title, a.source) for a in alerts] == [("BNY names new CFO", "news_service")]


def test_source_url_only_gemini_item_is_governed(gemini_on, monkeypatch):
    # The Gemini URL check read `url or source_url`, the allow-list read only
    # `url`: a link under `source_url` passed as URL-less, past both lists.
    _fake_gemini(monkeypatch, [
        {"title": "SOCO signs nuclear deal", "summary": "s", "source_url": "https://msn.com/a",
         "published_at": _today_iso()},
        {"title": "SOCO raises capex", "summary": "s", "url": None,
         "source_url": "https://random-blog.example/b", "published_at": _today_iso()},
    ])
    _fake_provider(monkeypatch, [
        {"title": "Southern Company files 8-K", "summary": "p", "url": "https://www.reuters.com/x"},
    ])
    report: dict = {}
    alerts = news_agent.run(TICKER, force_refresh=True, report=report)
    assert [(a.title, a.source) for a in alerts] == [("Southern Company files 8-K", "news_service")]
    assert report["origin"] == "provider"


def test_filter_grounded_sources_reads_source_url():
    news_agent.reload_domain_lists()
    kept = news_agent._filter_grounded_sources([
        {"title": "a", "source_url": "https://msn.com/a"},
        {"title": "b", "source_url": "https://www.reuters.com/b"},
    ])
    assert [it["title"] for it in kept] == ["b"]


def test_grounding_cap_under_the_llm_core_name_is_reported(gemini_on, monkeypatch):
    # The llm core (M1) names the predicate `_grounding_cap_reached`; probing
    # only `grounding_cap_reached` counted every capped call as "no news".
    monkeypatch.setattr(llm, "_grounding_cap_reached", lambda: True, raising=False)
    monkeypatch.setattr(llm, "gemini_chat_json", lambda *a, **k: None)
    _fake_provider(monkeypatch, [])
    report: dict = {}
    assert news_agent.run(TICKER, force_refresh=True, report=report) == []
    assert report == {"gemini_skipped": "grounding_cap", "origin": "empty"}


def test_grounding_cap_predicate_exists_wherever_the_cap_setting_does():
    # Tripwire for the merge with the llm core: once the cap is configurable,
    # the news note can only count capped calls if one of the names it
    # probes exists. Skipped where the llm core has not landed yet.
    if not hasattr(settings, "gemini_grounded_max_per_day"):
        pytest.skip("llm core grounding cap not in this tree")
    assert callable(getattr(llm, "grounding_cap_reached", None)
                    or getattr(llm, "_grounding_cap_reached", None))
