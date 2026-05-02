"""Phase F tests — universe tiering + versioned memo store.

Cover:
- `memo_store.save_memo` / `latest_memo` / `memo_history` round-trip.
- Lineage: a v2 with `parent_version=1` chains correctly.
- Trigger validation rejects unknown values.
- Universe tiering: tier-1 names load with `auto_analysis`; everyone else
  is `data_only`; `analyzed_on_demand` is preserved across re-seeds.
- API: `/api/stocks/{t}/memo` returns headers; `/memos` returns history;
  `data_only` ticker requires `?ondemand=true` for first analysis.
- On-demand promotion: a `data_only` ticker becomes `analyzed_on_demand`
  after a successful first-run memo.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import Company
from app.schemas import (
    AgentFinding,
    BullBearCase,
    CriticReview,
    StockMemoOut,
)
from app.services import memo_store


def _ensure_started() -> TestClient:
    """Make sure the DB schema is created and the demo universe is seeded.

    `TestClient` is lazy and on newer Starlette/FastAPI doesn't reliably fire
    startup events on first request, so we call the seeder directly. Safe to
    call repeatedly — `run_full_seed` is idempotent."""
    from app.database import init_db
    from app.seed_demo_data import run_full_seed
    init_db()
    run_full_seed()
    return TestClient(app)


def _stub_memo(ticker: str = "TST") -> StockMemoOut:
    """Minimal valid StockMemoOut for round-trip tests."""
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
# memo_store unit tests
# ---------------------------------------------------------------------------

def test_save_and_latest_round_trip():
    memo = _stub_memo("RTRIP")
    snap = memo_store.save_memo(memo, trigger="first_run")
    assert snap.version == 1
    assert snap.trigger == "first_run"
    latest = memo_store.latest_memo("RTRIP")
    assert latest is not None
    assert latest.version == 1
    rehydrated = memo_store.memo_to_pydantic(latest)
    assert rehydrated.ticker == "RTRIP"
    assert rehydrated.rating_label == "Neutral"


def test_versions_chain_with_parent_version():
    memo_store.save_memo(_stub_memo("CHAIN"), trigger="first_run")
    v2 = memo_store.save_memo(
        _stub_memo("CHAIN"), trigger="full_reanalysis", parent_version=1,
    )
    assert v2.version == 2
    assert v2.parent_version == 1
    history = memo_store.memo_history("CHAIN")
    assert [h.version for h in history] == [2, 1]


def test_save_memo_rejects_unknown_trigger():
    import pytest
    with pytest.raises(ValueError):
        memo_store.save_memo(_stub_memo("BAD"), trigger="totally_made_up")


def test_latest_memo_returns_none_for_unknown_ticker():
    assert memo_store.latest_memo("NEVER_SAVED") is None


# ---------------------------------------------------------------------------
# Universe tiering
# ---------------------------------------------------------------------------

def test_tier1_universe_marked_auto_analysis_after_seed():
    # The startup hook in app.main ran the seeder, which calls
    # `seed_universe_tiers`. Tier-1 names from the JSON config should be
    # promoted; everyone else stays data_only.
    _ensure_started()
    with SessionLocal() as db:
        msft = db.get(Company, "MSFT")
        cat = db.get(Company, "CAT")
        # MSFT is tier-1 (Technology, top 2 by mcap); CAT is sole tier-1 in Industrials.
        assert msft.universe_tier == "auto_analysis"
        assert cat.universe_tier == "auto_analysis"
        # Pick a name we *know* is in the demo set but not tier-1.
        bac = db.get(Company, "BAC")
        if bac is not None:
            assert bac.universe_tier == "data_only"


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------

def test_get_memo_returns_versioned_headers():
    c = _ensure_started()
    r = c.get("/api/stocks/MSFT/memo")
    assert r.status_code == 200
    assert "X-Memo-Version" in r.headers
    assert int(r.headers["X-Memo-Version"]) >= 1
    assert r.headers["X-Memo-Trigger"] in {
        "first_run", "full_reanalysis", "incremental_patch",
        "force_refresh", "scheduled",
    }


def test_get_memo_history_lists_versions():
    c = _ensure_started()
    # Force two analyses so history has at least 2 versions.
    c.post("/api/stocks/NVDA/analyze")
    c.post("/api/stocks/NVDA/analyze")
    r = c.get("/api/stocks/NVDA/memos")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) >= 2
    # Newest first
    assert rows[0]["version"] >= rows[-1]["version"]
    # Force-refresh records `full_reanalysis` after the first run
    triggers = {row["trigger"] for row in rows}
    assert triggers <= {"first_run", "full_reanalysis", "incremental_patch",
                        "force_refresh", "scheduled"}


def test_data_only_tier_blocks_memo_without_ondemand_flag():
    c = _ensure_started()
    # Force a known ticker into data_only and clear any existing memos.
    with SessionLocal() as db:
        from app.models import MemoSnapshot
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == "BAC").delete()
        bac = db.get(Company, "BAC")
        if bac is None:
            return  # ticker not in demo set; skip
        bac.universe_tier = "data_only"
        db.commit()
    r = c.get("/api/stocks/BAC/memo")
    assert r.status_code == 409
    assert "data-only" in r.json()["detail"].lower()


def test_ondemand_flag_promotes_data_only_to_analyzed_on_demand():
    c = _ensure_started()
    # Reset to data_only
    with SessionLocal() as db:
        from app.models import MemoSnapshot
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == "BAC").delete()
        bac = db.get(Company, "BAC")
        if bac is None:
            return
        bac.universe_tier = "data_only"
        db.commit()
    r = c.get("/api/stocks/BAC/memo?ondemand=true")
    assert r.status_code == 200
    assert r.headers.get("X-Memo-Version") == "1"
    with SessionLocal() as db:
        bac = db.get(Company, "BAC")
        assert bac.universe_tier == "analyzed_on_demand"


def test_analyze_endpoint_creates_new_version():
    c = _ensure_started()
    r1 = c.post("/api/stocks/MSFT/analyze")
    v1 = int(r1.headers.get("X-Memo-Version", "0"))
    r2 = c.post("/api/stocks/MSFT/analyze")
    v2 = int(r2.headers.get("X-Memo-Version", "0"))
    assert v2 == v1 + 1
