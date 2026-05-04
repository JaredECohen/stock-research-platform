"""Wave 1C tests — as-of-date selector for historical / backtest memo runs.

Covers:
- `as_of_context` correctly pins `current_as_of_date()` and resets on exit.
- Cache keys are namespaced when an as_of date is active so live and backtest
  payloads don't shadow each other.
- `memo_store.save_memo` records `as_of_date`; `latest_memo` excludes backtest
  rows by default and surfaces them with `include_backtests=True`.
- API: `/api/stocks/{t}/memo?as_of=...` validates the date (422 on bad format,
  422 on future), and a successful backtest sets `X-Memo-As-Of`.
- Backtest path skips long-term memory writes (no append to companies/<T>.md).
- Backtest path does NOT promote a `data_only` ticker to `analyzed_on_demand`.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app.cache import snapshots as cache_snapshots
from app.database import SessionLocal
from app.main import app
from app.models import Company, MemoSnapshot
from app.schemas import (
    AgentFinding,
    BullBearCase,
    CriticReview,
    StockMemoOut,
)
from app.services import memo_store
from app.services.data_service import as_of_context, current_as_of_date


def _ensure_started() -> TestClient:
    from app.database import init_db
    from app.tests.fixtures.seed_demo_data import run_full_seed
    init_db()
    run_full_seed()
    return TestClient(app)


def _stub_memo(ticker: str = "ASOFT") -> StockMemoOut:
    finding = AgentFinding(agent="x", headline="h", summary="s", confidence=0.5)
    return StockMemoOut(
        ticker=ticker, company_name=ticker, sector="Technology",
        final_pm_view="pm view", rating_label="Neutral", confidence_score=50,
        one_sentence_thesis="thesis", business_summary="bd",
        sector_agent_view=finding, earnings_agent_view=finding,
        filing_agent_view=finding, valuation_agent_view=finding,
        comps_agent_view=finding, macro_sensitivity=finding,
        bull_case=BullBearCase(headline="bull", key_points=[]),
        bear_case=BullBearCase(headline="bear", key_points=[]),
        catalysts=[], key_risks=[], thesis_breakers=[],
        dcf_summary={}, portfolio_fit="",
        risk_committee_challenge=CriticReview(overall_assessment="ok"),
        final_verdict="verdict",
    )


# ---------------------------------------------------------------------------
# Context manager — basic plumbing
# ---------------------------------------------------------------------------

def test_as_of_context_pins_and_resets():
    assert current_as_of_date() is None
    target = date(2025, 12, 31)
    with as_of_context(target):
        assert current_as_of_date() == target
    assert current_as_of_date() is None


def test_as_of_context_none_is_noop():
    with as_of_context(None):
        assert current_as_of_date() is None


# ---------------------------------------------------------------------------
# Cache key segregation
# ---------------------------------------------------------------------------

def test_cache_key_segregation_live_vs_backtest():
    """Live cache_get must not see a payload written under an as-of context."""
    subject = "TESTKEY"
    kind = "asof_unit_test"

    # Clear any pre-existing rows for this subject.
    with SessionLocal() as db:
        cache_snapshots._ensure_table(db)
        db.query(cache_snapshots.ResearchSnapshot).filter(
            cache_snapshots.ResearchSnapshot.subject.like(f"{subject}%"),
        ).delete(synchronize_session=False)
        db.commit()

    # Write a payload under a backtest context.
    with as_of_context(date(2025, 1, 15)):
        cache_snapshots.cache_put(subject, kind, {"v": "backtest"})
        # Inside the same context, we see the backtest payload.
        hit = cache_snapshots.cache_get(subject, kind)
        assert hit is not None
        assert hit.payload.get("v") == "backtest"
        # The stored row uses the suffixed key.
        assert hit.subject == f"{subject}:asof:2025-01-15"

    # Outside the context, the live key is empty — no shadow.
    miss = cache_snapshots.cache_get(subject, kind)
    assert miss is None

    # Live write doesn't bleed into the backtest namespace either.
    cache_snapshots.cache_put(subject, kind, {"v": "live"})
    live = cache_snapshots.cache_get(subject, kind)
    assert live is not None and live.payload.get("v") == "live"

    with as_of_context(date(2025, 1, 15)):
        again = cache_snapshots.cache_get(subject, kind)
        assert again is not None
        assert again.payload.get("v") == "backtest"


# ---------------------------------------------------------------------------
# memo_store: as_of_date column + latest_memo filter
# ---------------------------------------------------------------------------

def test_save_memo_with_as_of_date_excluded_from_default_latest():
    ticker = "ASOFT1"
    # Reset any prior rows.
    with SessionLocal() as db:
        memo_store._ensure_table(db)
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == ticker).delete()
        db.commit()

    # Live memo (v1) — visible in default latest.
    memo_store.save_memo(_stub_memo(ticker), trigger="first_run")
    # Backtest memo (v2) — should NOT be returned by default.
    memo_store.save_memo(
        _stub_memo(ticker), trigger="full_reanalysis",
        as_of_date=date(2025, 6, 30), parent_version=1,
    )

    default = memo_store.latest_memo(ticker)
    assert default is not None
    assert default.version == 1
    assert default.as_of_date is None

    incl = memo_store.latest_memo(ticker, include_backtests=True)
    assert incl is not None
    assert incl.version == 2
    assert incl.as_of_date is not None


# ---------------------------------------------------------------------------
# API: validation
# ---------------------------------------------------------------------------

def test_memo_endpoint_rejects_future_as_of_date():
    c = _ensure_started()
    future = (date.today() + timedelta(days=365)).isoformat()
    r = c.get(f"/api/stocks/MSFT/memo?as_of={future}")
    assert r.status_code == 422
    assert "future" in r.json()["detail"].lower()


def test_memo_endpoint_rejects_malformed_as_of_date():
    c = _ensure_started()
    r = c.get("/api/stocks/MSFT/memo?as_of=not-a-date")
    assert r.status_code == 422
    detail = r.json()["detail"].lower()
    assert "yyyy-mm-dd" in detail or "as_of" in detail


# ---------------------------------------------------------------------------
# API: backtest path skips snapshot shortcut + tier promotion
# ---------------------------------------------------------------------------

def test_backtest_does_not_promote_data_only_tier():
    c = _ensure_started()
    # Pick a ticker that exists in the demo set but is not tier-1.
    ticker = "BAC"
    with SessionLocal() as db:
        row = db.get(Company, ticker)
        if row is None:
            return  # demo set didn't include BAC; skip
        row.universe_tier = "data_only"
        db.commit()
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == ticker).delete()
        db.commit()

    backtest = (date.today() - timedelta(days=180)).isoformat()
    r = c.get(f"/api/stocks/{ticker}/memo?as_of={backtest}")
    # Backtest path bypasses the data-only gate (it's diagnostic, not a charge).
    assert r.status_code == 200
    assert r.headers.get("X-Memo-As-Of", "").startswith(backtest)

    with SessionLocal() as db:
        row = db.get(Company, ticker)
        # Critical: tier must NOT have been promoted by a backtest run.
        assert row.universe_tier == "data_only"


def test_backtest_returns_fresh_memo_even_when_live_snapshot_exists():
    """Backtest path should always run fresh — never serve the cached live memo."""
    c = _ensure_started()
    # Force a live memo so a snapshot exists.
    r0 = c.post("/api/stocks/MSFT/analyze")
    assert r0.status_code == 200
    live_version = int(r0.headers.get("X-Memo-Version", "0"))

    backtest = (date.today() - timedelta(days=90)).isoformat()
    r = c.get(f"/api/stocks/MSFT/memo?as_of={backtest}")
    assert r.status_code == 200
    # The backtest produced a new snapshot row (with as_of_date set), and
    # the response carries the As-Of header.
    assert r.headers.get("X-Memo-As-Of", "").startswith(backtest)

    # Live latest_memo lookup should still return the live snapshot — the
    # backtest row must not shadow it.
    snap = memo_store.latest_memo("MSFT")
    assert snap is not None
    assert snap.as_of_date is None
    assert snap.version == live_version


# ---------------------------------------------------------------------------
# Memory write skip
# ---------------------------------------------------------------------------

def test_backtest_skips_long_term_memory_write(tmp_path, monkeypatch):
    """A backtest run must not append to the company memory notebook."""
    from app.config import settings
    from app.agents.graph import run_stock_memo
    from app.memory import longterm

    # Redirect the memory dir to a clean tmp path so we can assert on writes.
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))

    backtest = date.today() - timedelta(days=120)
    run_stock_memo("MSFT", as_of_date=backtest)

    company_file = longterm.company_memory_path("MSFT")
    # If a backtest accidentally wrote, the file would exist with the
    # standard frontmatter. Assert it never was created.
    assert not company_file.exists(), (
        f"Backtest run unexpectedly wrote {company_file}"
    )


def test_run_stock_memo_rejects_future_as_of_date():
    import pytest
    from app.agents.graph import run_stock_memo
    future = date.today() + timedelta(days=30)
    with pytest.raises(ValueError, match="future"):
        run_stock_memo("MSFT", as_of_date=future)
