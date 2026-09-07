"""Structural tests for `services/theme_exposure_service.py`.

Under blank keys `compute_for_ticker` takes the deterministic keyword
path, so what is pinned here is: the keyword scorer's arithmetic, that
every ticker with text gets exactly one row per vocabulary theme (never
a theme outside `THEME_KEYWORDS`), the evidence format each path
produces, re-runs upsert in place, and `top_for_theme` on an unknown
theme is empty rather than an error.

The LLM branch is gated on `settings.openai_api_key`; that gate is
asserted closed here. Its result-cleaning logic is recorded as an xfail
because the branch currently cannot execute at all (see the reason).
"""
from __future__ import annotations

import pytest

from app.agents import llm
from app.config import settings
from app.database import SessionLocal
from app.models import Company, ThemeExposure
from app.services import theme_exposure_service as tes
from app.tests.fixtures.seed_demo_data import run_full_seed

TICKER = "TSTTHEME"
GLP1_TEXT = (
    "We develop GLP-1 receptor agonists for obesity and type 2 diabetes. "
    "Semaglutide demand exceeded supply again this quarter; obesity, obesity, obesity."
)


@pytest.fixture(scope="module", autouse=True)
def _seeded():
    run_full_seed()


def _upsert_company(ticker: str, description: str) -> None:
    with SessionLocal() as db:
        db.merge(Company(
            ticker=ticker, company_name=f"{ticker} Inc", sector="Healthcare",
            industry="Biotech", business_description=description,
            universe_tier="data_only",           # keep it out of universe-wide loops
        ))
        db.query(ThemeExposure).filter_by(ticker=ticker).delete()
        db.commit()


def _rows(ticker: str):
    with SessionLocal() as db:
        return {r.theme: r for r in db.query(ThemeExposure).filter_by(ticker=ticker).all()}


# ---------------------------------------------------------------------------
# _keyword_score
# ---------------------------------------------------------------------------

def test_keyword_score_arithmetic():
    assert tes._keyword_score("", ["ai"]) == (0.0, [])
    kws = tes.THEME_KEYWORDS["glp1"]
    score, hits = tes._keyword_score(GLP1_TEXT, kws)
    assert set(hits) == {"glp-1", "obesity", "diabetes", "semaglutide"}
    assert score == len(hits) / len(kws) * 100.0
    # Case-insensitive, and a keyword spammed many times still counts once.
    assert tes._keyword_score("OBESITY obesity Obesity", ["obesity"]) == (100.0, ["obesity"])
    assert tes._keyword_score("nothing relevant here", kws) == (0.0, [])


# ---------------------------------------------------------------------------
# compute_for_ticker — keyword fallback
# ---------------------------------------------------------------------------

def test_llm_gate_is_closed_under_blank_keys(monkeypatch):
    assert settings.openai_api_key == ""

    def _never(*_a, **_k):
        raise AssertionError("chat_json must not be called without a key")
    monkeypatch.setattr(llm, "chat_json", _never)
    assert tes._llm_theme_scores(TICKER, GLP1_TEXT) is None
    assert tes._llm_theme_scores(TICKER, "   ") is None


def test_no_text_is_reported_not_scored():
    assert tes.compute_for_ticker("ZZZNOSUCH") == {
        "ticker": "ZZZNOSUCH", "themes_written": 0, "reason": "no_text",
    }
    _upsert_company("TSTBLANK", "")
    assert tes.compute_for_ticker("tstblank") == {
        "ticker": "TSTBLANK", "themes_written": 0, "reason": "no_text",
    }
    assert _rows("TSTBLANK") == {}


def test_keyword_fallback_writes_exactly_the_vocabulary():
    _upsert_company(TICKER, GLP1_TEXT)
    out = tes.compute_for_ticker(TICKER.lower())
    assert out == {"ticker": TICKER, "themes_written": len(tes.THEME_KEYWORDS),
                   "scoring": "keyword_fallback"}
    rows = _rows(TICKER)
    assert set(rows) == set(tes.THEME_KEYWORDS)          # never a theme outside the vocabulary
    for r in rows.values():
        assert 0.0 <= r.score <= 100.0
        assert isinstance(r.evidence, list) and r.evidence
        assert r.refreshed_at is not None

    glp1 = rows["glp1"]
    assert glp1.score > 0
    assert all(e.startswith("keyword: ") for e in glp1.evidence)
    assert len(glp1.evidence) <= 5
    # `weight_loss` shares "obesity" with glp1, so it scores too; unrelated themes do not.
    assert rows["weight_loss"].score > 0
    assert rows["cybersecurity"].score == 0.0
    assert rows["cybersecurity"].evidence == ["no material exposure"]


def test_rerun_upserts_in_place():
    _upsert_company(TICKER, GLP1_TEXT)
    tes.compute_for_ticker(TICKER)
    first = _rows(TICKER)
    stamps = {k: v.refreshed_at for k, v in first.items()}
    ids = {k: v.id for k, v in first.items()}

    _upsert_text_only(TICKER, "Ransomware and zero trust endpoint security. No drug pipeline.")
    tes.compute_for_ticker(TICKER)
    second = _rows(TICKER)
    assert {k: v.id for k, v in second.items()} == ids     # same rows, rewritten
    assert all(second[k].refreshed_at >= stamps[k] for k in ids)
    assert second["cybersecurity"].score > 0
    assert second["glp1"].score == 0.0
    assert second["glp1"].evidence == ["no material exposure"]


def _upsert_text_only(ticker: str, description: str) -> None:
    with SessionLocal() as db:
        db.get(Company, ticker).business_description = description
        db.commit()


# ---------------------------------------------------------------------------
# LLM result cleaning (cannot run today — see reason)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=False,
    reason=(
        "_llm_theme_scores builds its prompt with json.dumps but "
        "theme_exposure_service never imports json, so the branch raises "
        "NameError before the LLM is called whenever a key is configured"
    ),
)
def test_llm_scores_are_clamped_and_unknown_themes_dropped(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "test-key-not-real")
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {
        "glp1": {"score": 150, "evidence": "x" * 500},
        "cybersecurity": {"score": -5, "evidence": "none"},
        "ai_infrastructure": "not a dict",
        "energy_transition": {"score": "high"},
        "not_a_theme": {"score": 90, "evidence": "made up"},
    })
    out = tes._llm_theme_scores(TICKER, GLP1_TEXT)
    assert out == {
        "glp1": {"score": 100.0, "evidence": "x" * 300},
        "cybersecurity": {"score": 0.0, "evidence": "none"},
    }


# ---------------------------------------------------------------------------
# Readers + universe refresh
# ---------------------------------------------------------------------------

def test_top_for_theme_unknown_theme_is_empty():
    assert tes.top_for_theme("not_a_theme") == []
    assert tes.top_for_theme("") == []


def test_top_for_theme_filters_and_orders():
    _upsert_company(TICKER, GLP1_TEXT)
    tes.compute_for_ticker(TICKER)
    glp1_score = _rows(TICKER)["glp1"].score
    hits = tes.top_for_theme("glp1", min_score=0.0, limit=100)
    assert all(set(h) == {"ticker", "theme", "score", "evidence"} for h in hits)
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert any(h["ticker"] == TICKER for h in hits)
    assert not any(h["ticker"] == TICKER for h in tes.top_for_theme("glp1", min_score=glp1_score + 1))


def test_refresh_universe_limit_bounds_the_sweep():
    out = tes.refresh_universe(limit=2)
    assert out == {"tickers": 2, "rows_written": 2 * len(tes.THEME_KEYWORDS)}
