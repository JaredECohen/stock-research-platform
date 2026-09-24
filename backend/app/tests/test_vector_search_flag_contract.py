"""FIX-008b — `enable_vector_search` is reported, not consulted.

A production audit read `feature_flags.enable_vector_search=false` on
`/api/providers/status` as "semantic retrieval is disabled". No retrieval
path reads the flag. The status payload keeps the boolean (the Settings page
renders it) and adds a note saying what it does; this test holds the note
to the code: it toggles the flag and checks the filing and earnings
retrieval calls are identical, and that only the status route reads it.
Wiring a real switch is a retrieval-policy decision (owner), so if someone
does, this test fails and the note, config comment and docstrings must move
with it.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import PropertyMock

from fastapi.testclient import TestClient

import app as app_pkg
from app.agents import earnings_agent, filing_agent
from app.api.routes_health import FEATURE_FLAG_NOTES
from app.config import Settings, settings
from app.main import app
from app.services import retrieval_service, vector_store


def _retrieval_calls(monkeypatch, flag: bool) -> list[tuple]:
    monkeypatch.setattr(settings, "enable_vector_search", flag)
    # Retrieval runs before any LLM gate; keep the analysts on their
    # deterministic path so a run with live keys cannot spend on this test.
    monkeypatch.setattr(Settings, "has_llm", PropertyMock(return_value=False))
    calls: list[tuple] = []
    monkeypatch.setattr(vector_store, "search", lambda *a, **k: calls.append(
        ("vector", a, tuple(sorted((key, repr(v)) for key, v in k.items())))) or [])
    monkeypatch.setattr(retrieval_service, "search", lambda *a, **k: calls.append(
        ("bm25", a, tuple(sorted(k.items())))) or [])
    profile = {"ticker": "AAPL", "sector": "Technology"}
    filing_agent.run_filing_agent(
        profile, [{"type": "10-K", "url": "", "risk_factors": ["r1"], "mda": "m"}],
    )
    earnings_agent.run_earnings_agent(
        profile, {"period": "2025Q4", "prepared_remarks": "we grew", "qa": "q and a"}, {},
    )
    return calls


def test_status_notes_the_vector_flag_and_retrieval_really_ignores_it(monkeypatch):
    body = TestClient(app).get("/api/providers/status").json()
    # Backward compatible: the boolean stays where the Settings page reads it.
    assert isinstance(body["feature_flags"]["enable_vector_search"], bool)
    note = body["feature_flag_notes"]["enable_vector_search"]
    assert note.startswith("Not consulted by retrieval")

    # What the note claims: flipping the flag changes no retrieval call.
    off = _retrieval_calls(monkeypatch, False)
    on = _retrieval_calls(monkeypatch, True)
    assert off == on
    assert [c[0] for c in off] == ["vector", "bm25", "vector"]  # filing, its fallback, earnings

    # And nothing else reads it: the only attribute access is the status route.
    root = Path(app_pkg.__file__).parent
    readers = sorted(
        str(p.relative_to(root)) for p in root.rglob("*.py")
        if "tests" not in p.relative_to(root).parts
        and any(isinstance(n, ast.Attribute) and n.attr == "enable_vector_search"
                for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))))
    )
    assert readers == ["api/routes_health.py"]


def _filing_layers(monkeypatch, ticker: str, vector) -> list[str]:
    monkeypatch.setattr(Settings, "has_llm", PropertyMock(return_value=False))
    layers: list[str] = []

    def _vector(*a, **k):
        layers.append("vector")
        return vector()

    monkeypatch.setattr(vector_store, "search", _vector)
    monkeypatch.setattr(retrieval_service, "search",
                        lambda *a, **k: layers.append("bm25") or [])
    filing_agent.run_filing_agent(
        {"ticker": ticker, "sector": "Technology"},
        [{"type": "10-K", "url": "", "risk_factors": ["r1"], "mda": "m"}],
    )
    return layers


def test_note_names_every_route_to_the_bm25_fallback(monkeypatch):
    """The first note said BM25 ran "only when that search returns nothing",
    which is exactly the misreading the note exists to prevent: an index
    that raises, or a missing ticker, also lands on BM25. Each route the
    note names is driven here, so dropping one from the code or the text
    fails the test."""
    note = FEATURE_FLAG_NOTES["enable_vector_search"]

    def _raise():
        raise RuntimeError("index down")

    routes = {
        "returns nothing": _filing_layers(monkeypatch, "AAPL", lambda: []),
        "fails": _filing_layers(monkeypatch, "AAPL", _raise),
        "lack of a ticker": _filing_layers(monkeypatch, "", lambda: []),
    }
    assert routes == {
        "returns nothing": ["vector", "bm25"],
        "fails": ["vector", "bm25"],
        "lack of a ticker": ["bm25"],
    }
    for phrase in routes:
        assert phrase in note, phrase
    assert "only when" not in note and "always" not in note
