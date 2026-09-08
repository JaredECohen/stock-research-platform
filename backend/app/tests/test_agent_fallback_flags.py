"""RP-001 (S2) — every analyst and service fallback announces itself.

The rule under test: anything that changes what the reader sees in a memo
either raises or lands in `degraded_agents` (via `data["deterministic_fallback"]`,
which the graph promotes, or via `safe_runner.note_soft`). Enrichment that
fails (comps/risk narrative, earnings second pass) gets a `data` flag and a
log line but no banner entry, because the round-0 finding is deterministic
by design.

Every test runs with blank provider keys: `has_llm` is *patched* on and the
client factories are stubbed to raise, so a real client can never be
constructed and CI (no keys) stays deterministic and zero-cost.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import PropertyMock

import pytest

from app.agents import (
    comps_agent,
    earnings_agent,
    filing_agent,
    llm,
    macro_agent,
    news_impact_agent,
    orchestrator,
    risk_agent,
    sector_agents,
    technical_agent,
)
from app.agents.safe_runner import DegradationLog, active_log
from app.config import Settings, settings
from app.finance import dcf as dcf_engine
from app.schemas import (
    AgentFinding,
    BullBearCase,
    CriticReview,
    DCFAssumptions,
    StockMemoOut,
)
from app.services import update_orchestrator, valuation_service
from app.services.data_service import DataService
from app.services.fundamentals_service import get_full_financials


def _boom(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("simulated failure")


@pytest.fixture(autouse=True)
def _blank_keys_guard():
    assert not settings.has_llm, (
        "test_agent_fallback_flags must run with blank LLM keys "
        "(OPENAI_API_KEY='' ANTHROPIC_API_KEY='' GEMINI_API_KEY='')"
    )
    assert active_log() is None
    yield
    assert active_log() is None


@pytest.fixture
def llm_configured_but_empty(monkeypatch) -> List[str]:
    """`has_llm` on, every chat entry point returns nothing usable, every
    client factory forbidden. Returns the list of factories that were
    (wrongly) constructed so a test can assert it stayed empty."""
    constructed: List[str] = []

    def _forbid(name: str):
        def _factory(*args: Any, **kwargs: Any) -> None:
            constructed.append(name)
            raise AssertionError(f"{name} must not be constructed under blank keys")
        return _factory

    for key in ("openai_api_key", "anthropic_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, key, "")
    for factory in ("_openai_client", "_anthropic_client", "_gemini_client"):
        monkeypatch.setattr(llm, factory, _forbid(factory))
    for entry in ("chat_json", "chat_text", "gemini_chat_json", "gemini_chat_text"):
        monkeypatch.setattr(llm, entry, lambda *a, **k: None)
    monkeypatch.setattr(Settings, "has_llm", PropertyMock(return_value=True))
    assert settings.has_llm
    return constructed


def _nvda_inputs() -> Dict[str, Any]:
    fin = get_full_financials("NVDA")
    return {"profile": fin["profile"], "ratios": fin["ratios"], "earnings": fin["earnings"]}


_TRANSCRIPT = {
    "period": "2025Q4", "management_tone": "constructive",
    "prepared_remarks": "We grew revenue and expanded margins." * 40,
    "qa": "Analyst asked about guidance; management reaffirmed." * 40,
}

_FILINGS = [{
    "type": "10-K", "accession_number": "0001-TEST-10K", "period_end": "2025-12-31",
    "filing_date": "2026-02-01", "mda": "Revenue grew on data-center demand." * 20,
    "risk_factors": ["Customer concentration", "Export controls"],
    "business_description": "Accelerated computing.", "segments": ["Compute"],
}]


# ---------------------------------------------------------------------------
# 1. LLM-backed round-0 analysts flag the deterministic fallback
# ---------------------------------------------------------------------------

def _run_sector():
    i = _nvda_inputs()
    return sector_agents.run_sector_agent(i["profile"], i["ratios"])


def _run_earnings():
    i = _nvda_inputs()
    return earnings_agent.run_earnings_agent(i["profile"], _TRANSCRIPT, i["earnings"])


def _run_filing():
    return filing_agent.run_filing_agent(_nvda_inputs()["profile"], _FILINGS)


def _run_macro():
    return macro_agent.run_macro_agent(_nvda_inputs()["profile"], "soft_landing")


def _run_technical():
    return technical_agent.run_technical_agent({"ticker": "NVDA", "sector": "Technology"})


LLM_BACKED_RUNNERS = [
    pytest.param(_run_sector, "Sector Analyst", id="sector"),
    pytest.param(_run_earnings, "Earnings Analyst", id="earnings"),
    pytest.param(_run_filing, "Filing Analyst", id="filing"),
    pytest.param(_run_macro, "Macro Analyst", id="macro"),
    pytest.param(_run_technical, "Technical Analyst", id="technical"),
]


@pytest.mark.parametrize("runner,agent", LLM_BACKED_RUNNERS)
def test_llm_backed_analyst_flags_deterministic_fallback(llm_configured_but_empty, runner, agent):
    finding = runner()
    assert finding.agent == agent
    assert finding.data.get("deterministic_fallback"), finding.data
    assert llm_configured_but_empty == []


@pytest.mark.parametrize("runner,agent", LLM_BACKED_RUNNERS)
def test_no_flag_when_no_llm_is_configured(runner, agent):
    """Without keys the deterministic path IS the design — the flag would
    otherwise put every section of every demo memo on the banner."""
    finding = runner()
    assert finding.agent == agent
    assert "deterministic_fallback" not in finding.data


# ---------------------------------------------------------------------------
# 2. Comps / risk: deterministic at round 0, flagged only on re-fire
# ---------------------------------------------------------------------------

def _comps_inputs():
    profile = _nvda_inputs()["profile"]
    comps = valuation_service.build_comps("NVDA")
    assert comps is not None, "demo comps must build for NVDA"
    return profile, comps


def test_comps_round0_never_flags_even_with_llm_empty(llm_configured_but_empty):
    profile, comps = _comps_inputs()
    finding = comps_agent.run_comps_agent(profile, comps)
    assert "deterministic_fallback" not in finding.data
    assert "narrative_failed" not in finding.data  # None is "no narrative", not a crash


def test_risk_round0_never_flags_even_with_llm_empty(llm_configured_but_empty):
    i = _nvda_inputs()
    finding = risk_agent.run_risk_agent(i["profile"], i["ratios"], None)
    assert "deterministic_fallback" not in finding.data
    assert "narrative_failed" not in finding.data


def test_comps_and_risk_narrative_crash_is_flagged_but_not_degraded(
    llm_configured_but_empty, monkeypatch, caplog,
):
    monkeypatch.setattr(llm, "chat_json", _boom)
    profile, comps = _comps_inputs()
    i = _nvda_inputs()
    log = DegradationLog()
    with caplog.at_level(logging.WARNING), log.activate():
        c = comps_agent.run_comps_agent(profile, comps)
        r = risk_agent.run_risk_agent(i["profile"], i["ratios"], None)
    assert c.data["narrative_failed"].startswith("RuntimeError")
    assert r.data["narrative_failed"].startswith("RuntimeError")
    assert "deterministic_fallback" not in c.data
    assert "deterministic_fallback" not in r.data
    assert log.degraded_agents() == []
    assert "Comps Analyst narrative failed" in caplog.text
    assert "Risk Analyst narrative failed" in caplog.text


@pytest.mark.parametrize("failure", ["empty", "raise"])
def test_comps_and_risk_refire_fallthrough_is_flagged(llm_configured_but_empty, monkeypatch, failure):
    """On a deep-research follow-up an LLM answer was expected; shipping the
    round-0 text back is a degradation the graph must promote."""
    if failure == "raise":
        monkeypatch.setattr(llm, "chat_json", _boom)
    profile, comps = _comps_inputs()
    i = _nvda_inputs()
    c = comps_agent.run_comps_agent(profile, comps, prior_round_critique="Why the premium?")
    r = risk_agent.run_risk_agent(i["profile"], i["ratios"], None, prior_round_critique="Why?")
    assert "round-0 comps read kept" in c.data["deterministic_fallback"]
    assert "round-0 risk read kept" in r.data["deterministic_fallback"]
    if failure == "raise":
        assert "RuntimeError" in c.data["deterministic_fallback"]
        assert "RuntimeError" in r.data["deterministic_fallback"]


# ---------------------------------------------------------------------------
# 3. Retrieval and enrichment failures inside the analysts
# ---------------------------------------------------------------------------

def test_earnings_and_filing_retrieval_failure_lands_on_the_banner(monkeypatch):
    from app.services import vector_store
    monkeypatch.setattr(vector_store, "search", _boom)
    i = _nvda_inputs()
    log = DegradationLog()
    with log.activate():
        e = earnings_agent.run_earnings_agent(i["profile"], _TRANSCRIPT, i["earnings"])
        f = filing_agent.run_filing_agent(i["profile"], _FILINGS)
    assert e.data["retrieval_failed"].startswith("RuntimeError")
    assert f.data["retrieval_failed"].startswith("RuntimeError")
    assert log.degraded_agents() == ["Earnings Analyst", "Filing Analyst"]
    for ev in log.events():
        assert ev["error_type"] == "RuntimeError"
        assert ev["message"].startswith("retrieval unavailable")
    # Both analysts still produced a real finding, not the crash stub.
    assert e.confidence > 0.0 and f.confidence > 0.0


def test_filing_bm25_failure_is_soft_not_fatal(monkeypatch):
    """A dead news feed reaches the filing analyst through the BM25 layer;
    it used to propagate as a hard failure that replaced the whole
    section with the "unavailable" stub."""
    from app.services import retrieval_service
    monkeypatch.setattr(retrieval_service, "search", _boom)
    log = DegradationLog()
    with log.activate():
        f = filing_agent.run_filing_agent(_nvda_inputs()["profile"], _FILINGS)
    assert f.headline.endswith("highlights")
    assert f.data["retrieval_failed"].startswith("RuntimeError")
    assert log.degraded_agents() == ["Filing Analyst"]


def test_sector_bull_bear_parse_failure_is_flagged(monkeypatch, caplog):
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {
        "headline": "h", "summary": "s", "key_points": ["k"], "confidence": 0.7,
        "bull_bear_analysis": {"bull_case": "not a dict", "bear_case": {}},
    })
    i = _nvda_inputs()
    with caplog.at_level(logging.WARNING):
        finding = sector_agents.run_sector_agent(i["profile"], i["ratios"])
    assert finding.summary == "s"
    assert finding.data["bull_bear_parse_failed"]
    # The contract is still met by the deterministic builder.
    assert finding.data["bull_bear_analysis"]["bull_case"]["headline"]
    assert "bull/bear block" in caplog.text


def test_earnings_structured_parse_failure_is_flagged(monkeypatch, caplog):
    import app.schemas as schemas_pkg

    class _Broken:
        def __init__(self, **kwargs: Any) -> None:
            raise ValueError("simulated schema rejection")

    monkeypatch.setattr(schemas_pkg, "EarningsStructured", _Broken)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {
        "headline": "h", "summary": "s", "key_points": ["k"], "confidence": 0.7,
        "structured": {"period": "2025Q4"},
    })
    i = _nvda_inputs()
    with caplog.at_level(logging.WARNING):
        finding = earnings_agent.run_earnings_agent(i["profile"], _TRANSCRIPT, i["earnings"])
    assert finding.summary == "s"
    assert "structured" not in finding.data
    assert finding.data["structured_parse_failed"].startswith("ValueError")
    assert "structured block failed validation" in caplog.text


def test_earnings_second_pass_crash_is_flagged(llm_configured_but_empty, monkeypatch, caplog):
    calls: List[int] = []

    def _first_call_raises(*args: Any, **kwargs: Any):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("simulated failure")
        return None

    monkeypatch.setattr(llm, "chat_json", _first_call_raises)
    long_transcript = dict(_TRANSCRIPT, qa="Q: guidance? A: reaffirmed. " * 400)
    assert len(long_transcript["qa"]) > 8000
    i = _nvda_inputs()
    with caplog.at_level(logging.WARNING):
        finding = earnings_agent.run_earnings_agent(i["profile"], long_transcript, i["earnings"])
    assert finding.data["qa_pass_failed"].startswith("RuntimeError")
    assert finding.data["deterministic_fallback"]
    assert "Q&A second pass failed" in caplog.text


# ---------------------------------------------------------------------------
# 4. Technical analyst: a dead price feed is a degraded finding, not a stub
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("feed", ["empty", "raise"])
def test_technical_price_feed_failure_is_degraded(monkeypatch, feed):
    monkeypatch.setattr(
        technical_agent, "get_price_series",
        _boom if feed == "raise" else (lambda *a, **k: []),
    )
    log = DegradationLog()
    with log.activate():
        finding = technical_agent.run_technical_agent({"ticker": "NVDA"})
    assert finding.agent == "Technical Analyst"
    assert finding.data["degraded"] is True
    assert finding.confidence <= 0.3
    assert log.degraded_agents() == ["Technical Analyst"]
    event = log.events()[0]
    if feed == "raise":
        assert event["error_type"] == "RuntimeError"
        assert "simulated failure" in event["message"]
        assert finding.data["error"].startswith("RuntimeError")
    else:
        assert event["message"] == "price series empty"


def test_technical_short_series_is_honest_data_limit_not_degraded():
    """A few bars is a real answer ("not enough history"), not a failure."""
    log = DegradationLog()
    with log.activate():
        finding = technical_agent.run_technical_agent({"ticker": "NVDA"}, days=10)
    assert "insufficient" in finding.headline
    assert "degraded" not in finding.data
    assert log.degraded_agents() == []


# ---------------------------------------------------------------------------
# 5. DCF realism guardrail: a crashed check is a guardrail, not a pass
# ---------------------------------------------------------------------------

def _base_scenario():
    a = DCFAssumptions(
        revenue_growth=[0.1] * 5, operating_margin=[0.25] * 5, tax_rate=0.21,
        da_pct_revenue=0.04, capex_pct_revenue=0.05, nwc_pct_revenue=0.02,
        terminal_growth=0.025, exit_ebitda_multiple=15.0, wacc=0.085,
        base_revenue=1_000_000.0, net_debt=0.0, diluted_shares=1_000_000.0,
        current_price=10.0,
    )
    return dcf_engine.run_dcf(a)


def test_check_dcf_realism_reports_a_crashed_cohort_lookup(monkeypatch, caplog):
    monkeypatch.setattr(dcf_engine, "_cohort_p90_ev_ebitda", _boom)
    with caplog.at_level(logging.WARNING):
        rails = dcf_engine.check_dcf_realism(_base_scenario(), ticker="NVDA")
    errors = [g for g in rails if g.metric == "guardrail_error"]
    assert len(errors) == 1
    assert errors[0].severity == "warn"
    assert "RuntimeError" in errors[0].message
    assert "realism guardrail crashed" in caplog.text


def test_check_dcf_realism_has_no_error_entry_when_the_lookup_works(monkeypatch):
    monkeypatch.setattr(dcf_engine, "_cohort_p90_ev_ebitda", lambda t: None)
    rails = dcf_engine.check_dcf_realism(_base_scenario(), ticker="NVDA")
    assert not [g for g in rails if g.metric == "guardrail_error"]


# ---------------------------------------------------------------------------
# 6. Valuation service: engine-optional inputs are reader-visible
# ---------------------------------------------------------------------------

def test_peer_lookup_failure_records_comps_engine(monkeypatch):
    from app import database
    monkeypatch.setattr(valuation_service, "_peer_groups", lambda: {})
    monkeypatch.setattr(database, "SessionLocal", _boom)
    log = DegradationLog()
    with log.activate():
        assert valuation_service.get_peers("NVDA") == []
    assert log.degraded_agents() == ["Comps Engine"]
    event = log.events()[0]
    assert event["error_type"] == "RuntimeError"
    assert event["message"].startswith("peer lookup failed")


def test_consensus_estimates_failure_records_dcf_engine(monkeypatch):
    monkeypatch.setattr(DataService, "get_estimates", _boom)
    log = DegradationLog()
    with log.activate():
        assumptions = valuation_service.default_dcf_assumptions("NVDA")
    assert assumptions is not None  # the DCF still builds, without its anchor
    assert log.degraded_agents() == ["DCF Engine"]
    assert log.events()[0]["message"].startswith("consensus estimates unavailable")


def test_dcf_version_persistence_failure_is_error_and_on_the_banner(monkeypatch, caplog):
    """D4: the DCF still ships (unlike the memo store, this does not raise),
    but the DCF Versions page now lags with a trace instead of silently."""
    from app.services import dcf_store
    monkeypatch.setattr(dcf_store, "save_version", _boom)
    log = DegradationLog()
    with caplog.at_level(logging.ERROR, logger="app.services.valuation_service"), log.activate():
        result = valuation_service.build_dcf("NVDA", force_refresh=True)
    assert result is not None
    assert "DCF Store" in log.degraded_agents()
    record = next(r for r in caplog.records if "DCF version persistence failed" in r.getMessage())
    assert record.levelno == logging.ERROR


def test_valuation_service_fallbacks_are_noops_outside_a_run(monkeypatch):
    """Chat / screener / DCF Lab call the same service; for them a
    fallback is not a memo degradation."""
    monkeypatch.setattr(DataService, "get_estimates", _boom)
    assert active_log() is None
    assert valuation_service.default_dcf_assumptions("NVDA") is not None
    assert active_log() is None


# ---------------------------------------------------------------------------
# 7. News-impact agent and the update path
# ---------------------------------------------------------------------------

def _stub_memo(ticker: str) -> StockMemoOut:
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


def _alert(ticker: str):
    from app.schemas import NewsAlert
    return NewsAlert(
        ticker=ticker, title="Guidance lowered", summary="CFO cut FY guidance.",
        severity="material", source="test", published_at="2026-09-07T00:00:00",
    )


@pytest.mark.parametrize("failure,expected", [("raise", "RuntimeError"), ("empty", "LLMNoOutput")])
def test_news_impact_llm_failure_returns_error_and_not_material(
    llm_configured_but_empty, monkeypatch, failure, expected,
):
    if failure == "raise":
        monkeypatch.setattr(llm, "chat_json", _boom)
    out = news_impact_agent.assess(_stub_memo("TSTAF"), _alert("TSTAF"))
    assert out["material"] is False
    assert out["patch"] == {}
    assert out["error"] == expected
    assert llm_configured_but_empty == []


def test_news_impact_without_llm_carries_no_error():
    out = news_impact_agent.assess(_stub_memo("TSTAF"), _alert("TSTAF"))
    assert out["material"] is False
    assert "error" not in out


def test_on_news_alert_reports_a_failed_assessment_as_such(monkeypatch, caplog):
    from app.services import memo_store
    memo_store.save_memo(_stub_memo("TSTAERR"))
    monkeypatch.setattr(
        news_impact_agent, "assess",
        lambda memo, alert: {"material": False, "patch": {}, "rationales": {},
                             "delta_summary": "", "error": "RuntimeError"},
    )
    with caplog.at_level(logging.WARNING):
        out = update_orchestrator.on_news_alert("TSTAERR", _alert("TSTAERR"))
    assert out["patched"] is False
    assert out["reason"] == "assessment_error"
    assert out["error"] == "RuntimeError"
    assert "news impact assessment failed for TSTAERR" in caplog.text


# ---------------------------------------------------------------------------
# 8. Auto-regen gate: a crash is a gate error, not "nothing due"
# ---------------------------------------------------------------------------

def test_gate_db_error_is_reported_as_gate_error(monkeypatch):
    monkeypatch.setattr(update_orchestrator, "_persist_raw_data_only", lambda t: {})
    monkeypatch.setattr(
        update_orchestrator, "should_auto_regen",
        lambda t, **k: {"should": False, "reason": "db_error"},
    )
    assert update_orchestrator.on_filing_event("NVDA")["kind"] == "gate_error"
    assert update_orchestrator.on_transcript_event("NVDA", period="2025Q4")["kind"] == "gate_error"

    monkeypatch.setattr(
        update_orchestrator, "should_auto_regen",
        lambda t, **k: {"should": False, "reason": "stale_memo_90d_old"},
    )
    assert update_orchestrator.on_filing_event("NVDA")["kind"] == "skipped"


def test_regime_shift_counts_gate_errors(monkeypatch):
    monkeypatch.setattr(update_orchestrator, "_persist_raw_data_only", lambda t: {})
    monkeypatch.setattr(
        update_orchestrator, "_affected_tickers_for_regime_shift",
        lambda a, b: ["NVDA", "MSFT"],
    )
    monkeypatch.setattr(
        update_orchestrator, "should_auto_regen",
        lambda t, **k: {"should": False, "reason": "db_error" if t == "NVDA" else "no_memo_on_file"},
    )
    out = update_orchestrator.on_regime_shift("soft_landing", "recession")
    assert out["refreshed"] == ["NVDA", "MSFT"]
    assert out["gate_errors"] == ["NVDA"]


# ---------------------------------------------------------------------------
# 9. Comparison chat: a dropped ticker is named in the sources (D8)
# ---------------------------------------------------------------------------

def test_comparison_drop_is_named_in_sources(monkeypatch, caplog):
    monkeypatch.setattr(
        orchestrator, "classify_intent",
        lambda message: ("stock_comparison", ["NVDA", "ZZZZNOPE"], None),
    )

    def _fake_memo(ticker: str, **kwargs: Any) -> StockMemoOut:
        if ticker == "ZZZZNOPE":
            raise ValueError("Unknown ticker")
        m = _stub_memo(ticker)
        m.sources_used = [f"filing:{ticker}"]
        return m

    monkeypatch.setattr(orchestrator, "run_stock_memo", _fake_memo)
    with caplog.at_level(logging.WARNING):
        resp = orchestrator.Orchestrator().chat("Compare NVDA and ZZZZNOPE")
    assert resp.intent == "stock_comparison"
    assert resp.memo is not None and resp.memo.ticker == "NVDA"
    assert "memo unavailable: ZZZZNOPE" in resp.sources
    assert "filing:NVDA" in resp.sources
    assert "comparison memo unavailable for ZZZZNOPE" in caplog.text


def test_comparison_with_every_ticker_dropped_still_names_them(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "classify_intent",
        lambda message: ("stock_comparison", ["AAAANOPE", "ZZZZNOPE"], None),
    )
    monkeypatch.setattr(orchestrator, "run_stock_memo", _boom)
    resp = orchestrator.Orchestrator().chat("Compare AAAANOPE and ZZZZNOPE")
    assert resp.memo is None
    assert resp.sources == ["memo unavailable: AAAANOPE", "memo unavailable: ZZZZNOPE"]


# ---------------------------------------------------------------------------
# 10. Long-term memory: dropped structured facts leave a trace
# ---------------------------------------------------------------------------

def test_memory_entry_logs_dropped_structured_facts(caplog):
    from app.memory.longterm import MemoryEntry
    # Mixed key types cannot be sorted, so `sort_keys=True` raises.
    entry = MemoryEntry(
        date="2026-09-07", trigger="earnings", body="body",
        structured_facts={1: "a", "b": "c"},
    )
    with caplog.at_level(logging.WARNING, logger="app.memory.longterm"):
        rendered = entry.render()
    assert "structured-facts" not in rendered
    assert "structured facts dropped from memory entry 2026-09-07 (earnings)" in caplog.text


# ---------------------------------------------------------------------------
# 11. Fundamentals: a degraded earnings feed is announced, never cached
# ---------------------------------------------------------------------------

def _healthy_nvda_snapshot() -> None:
    """Make sure a healthy 90-day `company_cold` row exists before a test
    breaks the feed, so the assertions below distinguish "served an old
    healthy snapshot" from "served the poisoned one"."""
    full = get_full_financials("NVDA", force_refresh=True)
    assert full["earnings"], "demo NVDA earnings must be non-empty for this test"


def test_degraded_earnings_build_is_not_cached(monkeypatch):
    """Regression: the degraded build (earnings={}) used to be written as the
    quarter-long snapshot, so the banner fired once and every later memo
    hydrated `earnings={}` with no `note_soft` — a silent failure introduced
    by the change meant to remove one. A partial build must skip `cache_put`
    so the very next run retries the feed, exactly as the pre-degrade
    exception path did."""
    from app.cache import cache_get

    _healthy_nvda_snapshot()
    healthy = cache_get("NVDA", "company_cold")
    assert healthy is not None and healthy.payload["earnings"]

    original = DataService.get_earnings
    monkeypatch.setattr(DataService, "get_earnings", _boom)
    log = DegradationLog()
    with log.activate():
        degraded = get_full_financials("NVDA", force_refresh=True)
    assert degraded["earnings"] == {}
    assert "_partial" not in degraded, "the private marker must not leak to callers"
    assert log.degraded_agents() == ["Earnings Analyst"]
    assert log.events()[0]["error_type"] == "RuntimeError"

    # The newest live snapshot is still the healthy one — nothing was written.
    latest = cache_get("NVDA", "company_cold")
    assert latest is not None and latest.id == healthy.id
    assert latest.payload["earnings"]

    # Feed recovers: a plain (non-forced) read serves real earnings again and
    # records nothing — the reviewer's reproduction, inverted.
    monkeypatch.setattr(DataService, "get_earnings", original)
    log2 = DegradationLog()
    with log2.activate():
        recovered = get_full_financials("NVDA")
    assert recovered["earnings"], recovered["earnings"].keys()
    assert log2.degraded_agents() == []


def test_still_dead_earnings_feed_is_announced_on_every_run(monkeypatch):
    """Because the partial build is not cached, a feed that stays dead is
    re-announced by each run rather than hidden behind a hydrate."""
    monkeypatch.setattr(DataService, "get_earnings", _boom)
    for _ in range(2):
        log = DegradationLog()
        with log.activate():
            full = get_full_financials("NVDA", force_refresh=True)
        assert full["earnings"] == {}
        assert log.degraded_agents() == ["Earnings Analyst"]
