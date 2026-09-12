"""Structural tests for the natural-language screener (`services/nl_screener.py`).

Without an LLM key `_llm_translate` returns None and `translate`
degrades to an empty rule chain over the default sort. The no-LLM tests
blank the key themselves (`no_llm_key`) rather than trusting the
environment — a developer `.env` carries a live key and `translate`
would otherwise issue a real completion per phrasing. The module has
no keyword heuristic of its own — that absence is recorded in the
no-LLM tests below as shape assertions rather than by pinning
`rules == []`, so a future deterministic fallback lands without
rewriting them.

The LLM path is exercised by monkeypatching `llm.chat_json` (the one
seam `_llm_translate` uses) and giving `settings` a throwaway key for
the duration of the test; nothing reaches a provider.
"""
from __future__ import annotations

import socket
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agents import llm
from app.config import settings
from app.database import SessionLocal
from app.models import ScreenerMetric, ThemeExposure
from app.schemas import CustomScreenRequest
from app.services import nl_screener
from app.tests.fixtures.seed_demo_data import run_full_seed

PHRASINGS = [
    "cheap profitable software",
    "high quality compounders with low debt",
    "AI exposure, beta under 1.2",
]
REQUEST_KEYS = {"rules", "sectors", "sort_by", "order", "limit"}
RUN_KEYS = {"query", "request", "themes", "rationale", "matched", "rows"}


@pytest.fixture(scope="module", autouse=True)
def _seeded():
    run_full_seed()


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def _refuse(*_a, **_k):
        raise RuntimeError("network access attempted during an offline structural test")
    monkeypatch.setattr(socket.socket, "connect", _refuse)


@pytest.fixture
def no_llm_key(monkeypatch):
    """Blank the key and make any LLM call a test failure, so the
    fallback path is exercised rather than assumed."""
    monkeypatch.setattr(settings, "openai_api_key", "")

    def _never(*_a, **_k):
        raise AssertionError("chat_json must not be called without a key")
    monkeypatch.setattr(llm, "chat_json", _never)


@pytest.fixture
def fake_llm(monkeypatch):
    """Route `_llm_translate` through a canned `chat_json` reply."""
    holder: dict = {}
    monkeypatch.setattr(settings, "openai_api_key", "test-key-not-real")

    def _chat_json(prompt: str, **kwargs: Any):
        holder["prompt"] = prompt
        holder["kwargs"] = kwargs
        return holder["reply"]
    monkeypatch.setattr(llm, "chat_json", _chat_json)

    def _set(reply: Any) -> dict:
        holder["reply"] = reply
        return holder
    return _set


def _assert_valid_request(req: CustomScreenRequest) -> None:
    assert isinstance(req, CustomScreenRequest)
    assert all(r.metric in nl_screener._ALLOWED_METRICS for r in req.rules)
    assert all(r.op in nl_screener._ALLOWED_OPS for r in req.rules)
    assert req.sort_by in nl_screener._ALLOWED_METRICS
    assert req.order in ("asc", "desc")
    assert req.limit == 50


# ---------------------------------------------------------------------------
# No LLM
# ---------------------------------------------------------------------------

def test_no_key_means_no_llm_call(no_llm_key):
    assert nl_screener._llm_translate("cheap software") is None


@pytest.mark.parametrize("query", PHRASINGS)
def test_translate_without_llm_returns_a_valid_request(no_llm_key, query):
    req, themes, rationale = nl_screener.translate(query)
    _assert_valid_request(req)
    assert all(t in nl_screener._SUPPORTED_THEMES for t in themes)
    assert isinstance(rationale, str)


# ---------------------------------------------------------------------------
# Malformed LLM output degrades to the same fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("junk", [None, "just prose, no json", ["a", "list"], 42, 3.5])
def test_non_dict_llm_output_degrades_to_fallback(fake_llm, junk):
    fake_llm(junk)
    assert nl_screener._llm_translate("cheap software") is None
    req, themes, rationale = nl_screener.translate("cheap software")
    _assert_valid_request(req)
    assert req.rules == [] and themes == [] and rationale == ""


def test_llm_prompt_carries_the_allowed_vocabulary(fake_llm):
    holder = fake_llm({"rules": []})
    nl_screener.translate("anything")
    assert holder["kwargs"]["route"] == "cheap"
    for metric in nl_screener._ALLOWED_METRICS:
        assert metric in holder["prompt"]
    for theme in nl_screener._SUPPORTED_THEMES:
        assert theme in holder["prompt"]


def test_junk_rule_entries_are_filtered_not_fatal(fake_llm):
    fake_llm({
        "rules": [
            "not a dict",
            {"metric": "pe_ttm", "op": "<", "value": "20"},          # string value coerces
            {"metric": "nope", "op": "<", "value": 1},                # unknown metric
            {"metric": "roic", "op": "~", "value": 1},                # unknown op
            {"metric": "beta", "op": "between", "value": 0.5},        # no value2
            {"metric": "beta", "op": "between", "value": 0.5, "value2": "1.5"},
            {"metric": "op_margin", "op": ">", "value": None},        # None → 0.0
            {"metric": "roe", "op": ">", "value": "lots"},            # unparsable → 0.0
        ],
        "themes": ["glp1", "bogus_theme", 7],
        "sectors": ["Technology", 42],
        "sort_by": "not_a_metric",
        "order": "sideways",
        "rationale": 123,
    })
    req, themes, rationale = nl_screener.translate("whatever")
    _assert_valid_request(req)
    assert [(r.metric, r.op, r.value, r.value2) for r in req.rules] == [
        ("pe_ttm", "<", 20.0, None),
        ("beta", "between", 0.5, 1.5),
        ("op_margin", ">", 0.0, None),
        ("roe", ">", 0.0, None),
    ]
    assert themes == ["glp1"]
    assert req.sectors == ["Technology", "42"]
    assert req.sort_by == "market_cap" and req.order == "desc"
    assert rationale == "123"


def test_valid_llm_output_round_trips(fake_llm):
    fake_llm({
        "rules": [{"metric": "gross_margin", "op": ">=", "value": 60}],
        "themes": ["ai_infrastructure", "ai_applications"],
        "sectors": [],
        "sort_by": "revenue_growth_yoy",
        "order": "asc",
        "rationale": "Read as margin-rich AI names.",
    })
    req, themes, rationale = nl_screener.translate("AI names with fat margins")
    assert [(r.metric, r.op, r.value) for r in req.rules] == [("gross_margin", ">=", 60.0)]
    assert themes == ["ai_infrastructure", "ai_applications"]
    assert req.sectors is None                      # empty list normalises to None
    assert req.sort_by == "revenue_growth_yoy" and req.order == "asc"
    assert rationale == "Read as margin-rich AI names."


def test_fcf_yield_rule_is_dropped_rather_than_raising(fake_llm):
    fake_llm({"rules": [{"metric": "fcf_yield", "op": ">", "value": 4}]})
    req, _, _ = nl_screener.translate("fcf yield above 4%")
    _assert_valid_request(req)


# ---------------------------------------------------------------------------
# run() — the payload routes_screener hands straight to the client
# ---------------------------------------------------------------------------

def test_run_payload_shape_without_llm(no_llm_key):
    out = nl_screener.run("cheap profitable software")
    assert set(out) == RUN_KEYS
    assert out["query"] == "cheap profitable software"
    assert set(out["request"]) == REQUEST_KEYS
    assert out["themes"] == [] and out["rationale"] == ""
    assert isinstance(out["rows"], list) and len(out["rows"]) <= 50
    assert out["matched"] >= len(out["rows"])
    for row in out["rows"]:
        assert {"ticker", "company_name", "sector", "pm_score", "rating_label", "metrics"} <= set(row)


def _seed_metric_and_theme(ticker: str, theme: str, score: float) -> None:
    with SessionLocal() as db:
        if db.get(ScreenerMetric, ticker) is None:
            db.add(ScreenerMetric(ticker=ticker, market_cap=1.0, last_updated=datetime.utcnow()))
        row = db.query(ThemeExposure).filter_by(ticker=ticker, theme=theme).first()
        if row is None:
            db.add(ThemeExposure(ticker=ticker, theme=theme, score=score, evidence=["seed"]))
        else:
            row.score = score
        db.commit()


def test_run_theme_overlay_intersects_rows(monkeypatch):
    _seed_metric_and_theme("MSFT", "glp1", 90.0)
    _seed_metric_and_theme("AAPL", "glp1", 5.0)        # below the 20-point overlay floor
    monkeypatch.setattr(nl_screener, "_llm_translate", lambda q: {"themes": ["glp1"], "rules": []})
    out = nl_screener.run("glp-1 exposure")
    assert out["themes"] == ["glp1"]
    tickers = {r["ticker"] for r in out["rows"]}
    assert "MSFT" in tickers and "AAPL" not in tickers
    assert out["matched"] == len(out["rows"])

    monkeypatch.setattr(nl_screener, "_llm_translate", lambda q: {"themes": ["consumer_credit"]})
    empty = nl_screener.run("consumer credit names")
    assert empty["rows"] == [] and empty["matched"] == 0


def test_endpoint_returns_the_run_payload(no_llm_key):
    from app.main import app
    client = TestClient(app)
    r = client.post("/api/screener/nl", json={"query": "cheap profitable software"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == RUN_KEYS and set(body["request"]) == REQUEST_KEYS
    assert client.post("/api/screener/nl", json={"query": "   "}).status_code == 400
    assert client.post("/api/screener/nl", json={"query": ""}).status_code == 422
