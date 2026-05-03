"""Wave 8C tests — admin operational endpoints.

Each endpoint surfaces durable state from a service that already exists.
We assert the response shape and key numeric invariants, not LLM-driven
contents — those are tested in their respective wave's PR.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import DCFModel, MemoRunCheckpoint, MemoSnapshot
from app.schemas import DCFAssumptions
from app.services import dcf_store, update_orchestrator


def _stub_assumptions() -> DCFAssumptions:
    return DCFAssumptions(
        revenue_growth=[0.10, 0.09, 0.08, 0.07, 0.06],
        operating_margin=[0.25, 0.26, 0.27, 0.27, 0.27],
        tax_rate=0.21, da_pct_revenue=0.04, capex_pct_revenue=0.05,
        nwc_pct_revenue=0.02, terminal_growth=0.025,
        exit_ebitda_multiple=15.0, wacc=0.085,
        base_revenue=100.0, net_debt=10.0,
        diluted_shares=1.0, current_price=120.0,
    )


# ---------------------------------------------------------------------------
# DCF version history
# ---------------------------------------------------------------------------

def test_dcf_versions_returns_chain_newest_first():
    with SessionLocal() as db:
        dcf_store._ensure_table(db)
        db.query(DCFModel).filter(DCFModel.ticker == "ADMINX").delete()
        db.commit()
    a = _stub_assumptions()
    dcf_store.save_version("ADMINX", assumptions=a, trigger="initial")
    dcf_store.save_version(
        "ADMINX", assumptions=a, trigger="memo_rebuild",
        parent_version=1,
        assumption_changes=[
            {"field": "wacc", "from": 0.085, "to": 0.090,
             "rationale": "rates higher"},
        ],
    )
    c = TestClient(app)
    r = c.get("/api/admin/dcf-versions/ADMINX")
    assert r.status_code == 200
    body = r.json()
    versions = body["versions"]
    assert len(versions) == 2
    assert versions[0]["version"] == 2
    assert versions[1]["version"] == 1
    # Assumption changes round-trip on the v2 row.
    assert any(
        change["field"] == "wacc" for change in versions[0]["assumption_changes"]
    )


def test_dcf_versions_empty_for_unknown_ticker():
    c = TestClient(app)
    r = c.get("/api/admin/dcf-versions/NEVER_SEEDED_TICKER_X")
    assert r.status_code == 200
    assert r.json()["versions"] == []


# ---------------------------------------------------------------------------
# Update queue inspector
# ---------------------------------------------------------------------------

def test_update_queue_status_reports_in_flight_events():
    update_orchestrator._QUEUES.clear()
    update_orchestrator._QUEUES["WIP"].append({"kind": "full_reanalysis"})
    c = TestClient(app)
    r = c.get("/api/admin/update-queue")
    assert r.status_code == 200
    body = r.json()
    assert body["queue_depth_by_ticker"].get("WIP") == 1
    update_orchestrator._QUEUES.clear()


def test_update_queue_status_filters_by_ticker():
    update_orchestrator._QUEUES.clear()
    update_orchestrator._QUEUES["WIP"].append({"kind": "x"})
    update_orchestrator._QUEUES["OTHER"].append({"kind": "x"})
    c = TestClient(app)
    r = c.get("/api/admin/update-queue?ticker=WIP")
    assert r.status_code == 200
    body = r.json()
    assert body["queue_depth_by_ticker"] == {"WIP": 1}
    update_orchestrator._QUEUES.clear()


# ---------------------------------------------------------------------------
# News domain reload
# ---------------------------------------------------------------------------

def test_news_domain_reload_returns_counts():
    c = TestClient(app)
    r = c.post("/api/admin/news-domains/reload")
    assert r.status_code == 200
    body = r.json()
    # The shipped governance file has reuters in allowed and msn in blocked.
    assert body["allowed_count"] >= 1
    assert "reuters.com" in body["allowed_sample"] or body["allowed_count"] >= 5


# ---------------------------------------------------------------------------
# Lopsidedness audit
# ---------------------------------------------------------------------------

def _seed_memo_for_audit(ticker: str, *, bull_kp: int, bear_kp: int,
                         lean: str = "balanced") -> None:
    memo_json = {
        "ticker": ticker,
        "rating_label": "Neutral",
        "confidence_score": 50,
        "sector": "Technology",
        "bull_case": {"headline": "b", "key_points": [f"bp{i}" for i in range(bull_kp)]},
        "bear_case": {"headline": "x", "key_points": [f"xp{i}" for i in range(bear_kp)]},
        "sector_agent_view": {
            "agent": "Sector Analyst", "headline": "h", "summary": "s",
            "data": {
                "bull_bear_analysis": {
                    "sector_lean": lean,
                    "falsifiable_tests": [
                        {"statement": "x", "invalidates_side": "bull"},
                        {"statement": "y", "invalidates_side": "bear"},
                    ],
                },
            },
        },
    }
    with SessionLocal() as db:
        from app.services.memo_store import _ensure_table as _et
        _et(db)
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == ticker).delete()
        db.add(MemoSnapshot(
            ticker=ticker, version=1, parent_version=None,
            trigger="first_run", memo_json=memo_json,
            revision_log=[], generated_at=datetime.utcnow(),
        ))
        db.commit()


def test_lopsidedness_audit_reports_skew_and_lean_distribution():
    # Three balanced lean-distribution memos; bull/bear key-points equal so skew is 0.
    _seed_memo_for_audit("AUDA", bull_kp=3, bear_kp=3, lean="balanced")
    _seed_memo_for_audit("AUDB", bull_kp=3, bear_kp=3, lean="bull")
    _seed_memo_for_audit("AUDC", bull_kp=3, bear_kp=3, lean="bear")

    c = TestClient(app)
    r = c.get("/api/admin/lopsidedness-audit?n=20")
    assert r.status_code == 200
    body = r.json()
    assert body["inspected"] >= 3
    # We expect the seeded balanced rows; the audit may also pick up
    # other test fixtures, so we just check shape.
    assert "avg_bull_key_points" in body
    assert "avg_bear_key_points" in body
    assert "sector_lean_counts" in body
    assert set(body["sector_lean_counts"]) == {"bull", "bear", "balanced"}
    assert body["avg_falsifiable_tests_per_memo"] >= 0.0


def test_lopsidedness_audit_caps_at_n():
    """Audit should respect the n parameter."""
    for i in range(5):
        _seed_memo_for_audit(f"AUDN{i}", bull_kp=2, bear_kp=2)
    c = TestClient(app)
    r = c.get("/api/admin/lopsidedness-audit?n=2")
    assert r.status_code == 200
    body = r.json()
    assert body["inspected"] <= 2
    assert len(body["rows"]) <= 2
