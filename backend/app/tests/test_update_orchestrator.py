"""Wave 5B tests — update orchestrator + news-impact agent.

Covers:
- `news_impact_agent.assess` with no LLM → `material=False` (safe default).
- `_clamp_patch` enforces field allow-list, rating allow-list, and the
  ±15-pt confidence cap.
- `apply_patch` appends to bull/bear key_points and key_risks (additive,
  not replace) and replaces simple fields like `rating_label`.
- Discipline: a patch field without a rationale gets dropped.
- `on_news_alert` flow: when prior memo exists + LLM returns material,
  a new `incremental_patch` snapshot is created with `critic_skipped=True`
  in revision_log.
- No prior memo → `on_news_alert` returns `not_material`-ish reason.
- Daily patch cap: after 2 patches, further calls are gated.
- `on_filing_event` runs `run_stock_memo(force_refresh=True)`.
"""
from __future__ import annotations

from datetime import date as _date, datetime, timedelta
from unittest.mock import patch

from app.agents import news_impact_agent
from app.database import SessionLocal
from app.models import MemoSnapshot
from app.schemas import (
    AgentFinding,
    BullBearCase,
    CriticReview,
    NewsAlert,
    StockMemoOut,
)
from app.services import memo_store, update_orchestrator


def _stub_memo(ticker: str = "TSTU") -> StockMemoOut:
    f = AgentFinding(agent="x", headline="h", summary="s", confidence=0.6)
    return StockMemoOut(
        ticker=ticker, company_name=ticker, sector="Technology",
        final_pm_view="pm view", rating_label="Bullish", confidence_score=70.0,
        one_sentence_thesis="thesis", business_summary="bd",
        sector_agent_view=f, earnings_agent_view=f, filing_agent_view=f,
        valuation_agent_view=f, comps_agent_view=f, macro_sensitivity=f,
        bull_case=BullBearCase(headline="bull", key_points=["bp1"]),
        bear_case=BullBearCase(headline="bear", key_points=["bp2"]),
        catalysts=[], key_risks=[], thesis_breakers=[],
        dcf_summary={}, portfolio_fit="",
        risk_committee_challenge=CriticReview(overall_assessment="ok"),
        final_verdict="verdict",
    )


def _stub_alert(severity: str = "material") -> NewsAlert:
    return NewsAlert(
        ticker="TSTU", title="Guidance lowered for FY",
        summary="CFO cut FY guidance by 8% citing softer demand.",
        severity=severity, source="ap_newsroom",
        published_at=datetime.utcnow().isoformat(),
    )


def _reset_memos(ticker: str) -> None:
    with SessionLocal() as db:
        memo_store._ensure_table(db)
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == ticker).delete()
        db.commit()


# ---------------------------------------------------------------------------
# news_impact_agent
# ---------------------------------------------------------------------------

def test_assess_returns_not_material_without_llm(monkeypatch):
    """No LLM → safe default: don't push an unverified patch into a live memo."""
    from app.config import settings
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "")
    out = news_impact_agent.assess(_stub_memo(), _stub_alert())
    assert out["material"] is False
    assert out["patch"] == {}


def test_clamp_patch_caps_confidence_change_to_15():
    memo = _stub_memo()  # confidence_score = 70.0
    cleaned = news_impact_agent._clamp_patch(memo, {"confidence_score": 99.0})
    assert cleaned["confidence_score"] == 85.0  # 70 + 15
    cleaned2 = news_impact_agent._clamp_patch(memo, {"confidence_score": 10.0})
    assert cleaned2["confidence_score"] == 55.0  # 70 - 15


def test_clamp_patch_drops_unknown_fields_and_invalid_ratings():
    memo = _stub_memo()
    cleaned = news_impact_agent._clamp_patch(memo, {
        "rating_label": "Bullish",
        "rating_label_typo": "Bullish",  # unknown field key — drop
        "confidence_score": 65.0,
        "ticker": "DIFFERENT",  # not in allowed_fields — drop
    })
    assert "ticker" not in cleaned
    assert "rating_label_typo" not in cleaned
    assert cleaned["rating_label"] == "Bullish"
    assert cleaned["confidence_score"] == 65.0


def test_clamp_patch_drops_invalid_rating_value():
    memo = _stub_memo()
    cleaned = news_impact_agent._clamp_patch(memo, {"rating_label": "Moonshot"})
    assert "rating_label" not in cleaned


def test_apply_patch_appends_to_bull_bear_key_points():
    memo = _stub_memo()
    patched = news_impact_agent.apply_patch(memo, {
        "bull_case": {"key_points": ["new bull point from news"]},
        "bear_case": {"key_points": ["new bear point"]},
    })
    assert "new bull point from news" in patched.bull_case.key_points
    assert "bp1" in patched.bull_case.key_points  # original preserved
    assert "new bear point" in patched.bear_case.key_points


def test_apply_patch_replaces_rating_and_confidence():
    memo = _stub_memo()
    patched = news_impact_agent.apply_patch(memo, {
        "rating_label": "Mixed Negative",
        "confidence_score": 60.0,
        "one_sentence_thesis": "Thesis softened post-news.",
    })
    assert patched.rating_label == "Mixed Negative"
    assert patched.confidence_score == 60.0
    assert "softened" in patched.one_sentence_thesis


def test_apply_patch_appends_key_risks():
    memo = _stub_memo()
    patched = news_impact_agent.apply_patch(memo, {
        "key_risks": [{
            "title": "Guidance cut increases earnings risk",
            "detail": "FY revenue guide -8%",
            "severity": "high",
            "type": "company",
        }],
    })
    assert any("Guidance cut" in r.title for r in patched.key_risks)


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------

def test_on_news_alert_no_prior_memo_returns_reason():
    _reset_memos("TSTNOPRIOR")
    out = update_orchestrator.on_news_alert(
        "TSTNOPRIOR", _stub_alert(),
    )
    assert out["patched"] is False
    assert out["reason"] == "no_prior_memo"


def test_on_news_alert_writes_incremental_patch_when_material():
    _reset_memos("TSTPATCH")
    # Seed a v1 memo for TSTPATCH.
    memo_store.save_memo(_stub_memo("TSTPATCH"), trigger="first_run")

    fake_assessment = {
        "material": True,
        "patch": {
            "rating_label": "Mixed Positive",
            "confidence_score": 65.0,
            "one_sentence_thesis": "Thesis softened by guidance miss.",
        },
        "rationales": {
            "rating_label": "guidance miss + softer ratings flow",
            "confidence_score": "lower confidence on near-term execution",
            "one_sentence_thesis": "explicitly note guidance weakness",
        },
        "delta_summary": "Guidance miss; rating dialed back one notch.",
    }
    with patch.object(news_impact_agent, "assess", return_value=fake_assessment):
        out = update_orchestrator.on_news_alert("TSTPATCH", _stub_alert())
    assert out["patched"] is True
    assert out["version"] == 2

    snap = memo_store.latest_memo("TSTPATCH")
    assert snap.version == 2
    assert snap.trigger == "incremental_patch"
    assert snap.parent_version == 1
    # Patch revision log carries critic_skipped + rationales.
    log_entry = snap.revision_log[0]
    assert log_entry["critic_skipped"] is True
    assert "rating_label" in log_entry["fields_patched"]


def test_on_news_alert_drops_when_assessment_not_material():
    _reset_memos("TSTNOMA")
    memo_store.save_memo(_stub_memo("TSTNOMA"), trigger="first_run")
    fake_assessment = {"material": False, "patch": {}, "rationales": {}, "delta_summary": ""}
    with patch.object(news_impact_agent, "assess", return_value=fake_assessment):
        out = update_orchestrator.on_news_alert("TSTNOMA", _stub_alert())
    assert out["patched"] is False
    assert out["reason"] == "not_material"
    snap = memo_store.latest_memo("TSTNOMA")
    assert snap.version == 1  # no new version written


def test_daily_patch_cap_blocks_further_patches():
    _reset_memos("TSTCAP")
    # Seed memo + 2 incremental patches dated today.
    memo_store.save_memo(_stub_memo("TSTCAP"), trigger="first_run")
    memo_store.save_memo(_stub_memo("TSTCAP"), trigger="incremental_patch", parent_version=1)
    memo_store.save_memo(_stub_memo("TSTCAP"), trigger="incremental_patch", parent_version=2)

    fake_assessment = {
        "material": True,
        "patch": {"rating_label": "Mixed Negative"},
        "rationales": {"rating_label": "x"},
        "delta_summary": "x",
    }
    with patch.object(news_impact_agent, "assess", return_value=fake_assessment):
        out = update_orchestrator.on_news_alert("TSTCAP", _stub_alert())
    assert out["patched"] is False
    assert out["reason"] == "daily_cap_reached"


def test_on_filing_event_runs_full_reanalysis():
    """Just verify the entry point reaches `run_stock_memo(force_refresh=True)`."""
    with patch("app.agents.graph.run_stock_memo") as m:
        m.return_value = _stub_memo("TSTFE")
        out = update_orchestrator.on_filing_event("TSTFE")
    assert out["kind"] == "full_reanalysis"
    assert out["ticker"] == "TSTFE"
    m.assert_called_once_with("TSTFE", force_refresh=True)


def test_queue_depth_reports_in_flight_events():
    # Queue is process state; `queue_depth` exposes per-ticker depth.
    update_orchestrator._QUEUES.clear()
    update_orchestrator._QUEUES["TSTQ"].append({"kind": "full_reanalysis"})
    out = update_orchestrator.queue_depth("TSTQ")
    assert out == {"TSTQ": 1}
    update_orchestrator._QUEUES.clear()
