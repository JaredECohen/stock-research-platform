"""RP-001 — anything that changes what a reader sees lands on the banner.

The guard behind these tests: a failure on the memo path is either
(a) genuinely optional (telemetry, cache, enrichment) and only needs a log
line, or (b) changes the memo's content and must appear in
`StockMemoOut.degraded_agents` with its reason in `degradation_events`.
The mechanism that makes (b) a one-liner in code with no `DegradationLog`
in scope is `safe_runner.note_soft`, which writes to the log
`run_stock_memo` activates for the whole run via a ContextVar.

Every test here runs with blank provider keys: `has_llm` is *patched* on
for the LLM-path cases and the client factories are stubbed to raise, so a
real client can never be constructed and CI (no keys) stays deterministic.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import PropertyMock

import pytest

from app.agents import graph, llm
from app.agents.safe_runner import DegradationLog, active_log, note_soft
from app.config import Settings, settings
from app.schemas import (
    AgentFinding,
    BullBearCase,
    CriticReview,
    StockMemoOut,
    ValuationVerdict,
)
from app.services import memo_store
from app.services.data_service import DataService


def _boom(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("simulated capability failure")


@pytest.fixture(autouse=True)
def _blank_keys_guard():
    """These tests are only meaningful zero-cost. A developer `.env` with
    live keys would make the LLM-path cases spend money *and* stop
    proving determinism, so refuse to run rather than silently pass."""
    assert not settings.has_llm, (
        "test_degradation_surface must run with blank LLM keys "
        "(OPENAI_API_KEY='' ANTHROPIC_API_KEY='' GEMINI_API_KEY='')"
    )
    yield


# ---------------------------------------------------------------------------
# 1. A data capability that raises shows up under the service that owns it
# ---------------------------------------------------------------------------

# The DemoProvider is patched one level *above* the provider chain — on the
# `DataService` capability method — because `DataService._try_chain` absorbs
# a provider exception by design (class (a): the chain moves on to the next
# vendor). What the memo sees is the capability failing, and that is what a
# reader-visible degradation is measured against.
#
# `force_refresh=True` on every run so the 90-day fundamentals cache and the
# warm DCF/comps caches cannot hide the failure behind an earlier test's
# successful fetch.
CAPABILITY_CASES = [
    pytest.param("get_filings", "Filings Service", id="filings"),
    pytest.param("get_earnings_transcripts", "Transcript Service", id="transcript"),
    # `build_dcf` reads the live price through `get_close_series`, so a dead
    # price feed takes the DCF engine down with it.
    pytest.param("get_price_history", "DCF Engine", id="price-series"),
    # The memo path reaches news only through the filing analyst's BM25
    # retrieval, so today the failure is attributed to that analyst.
    pytest.param("get_news", "Filing Analyst", id="news"),
    # Consensus estimates are optional to the DCF engine but not to the
    # reader: `default_dcf_assumptions` records the lost anchor via
    # `note_soft("DCF Engine", ...)`.
    pytest.param("get_estimates", "DCF Engine", id="estimates"),
]


@pytest.mark.parametrize("method,expected_agent", CAPABILITY_CASES)
def test_capability_failure_lands_in_degraded_agents(monkeypatch, method, expected_agent):
    monkeypatch.setattr(DataService, method, _boom)
    memo = graph.run_stock_memo("NVDA", force_refresh=True)
    assert memo.ticker == "NVDA"
    assert expected_agent in memo.degraded_agents, memo.degraded_agents
    # The reason rides alongside the name, same order.
    assert [e["agent"] for e in memo.degradation_events] == memo.degraded_agents
    event = next(e for e in memo.degradation_events if e["agent"] == expected_agent)
    assert event["error_type"] == "RuntimeError"
    assert "simulated capability failure" in event["message"]


def test_earnings_feed_failure_degrades_the_earnings_analyst(monkeypatch):
    """The earnings feed is fetched inside the fundamentals stage, which is
    the one stage `run_stock_memo` does not wrap in a safe-runner: without
    a profile there is no memo. The earnings feed is not the profile,
    though — it only informs one section — so `_build_full_financials`
    degrades it (`earnings={}` + `note_soft("Earnings Analyst", ...)`)
    instead of failing the whole memo. Pinned here so a later change back
    to fatal is a deliberate choice, not drift."""
    original = DataService.get_earnings
    monkeypatch.setattr(DataService, "get_earnings", _boom)
    memo = graph.run_stock_memo("NVDA", force_refresh=True)
    assert memo.ticker == "NVDA"
    assert "Earnings Analyst" in memo.degraded_agents, memo.degraded_agents
    event = next(e for e in memo.degradation_events if e["agent"] == "Earnings Analyst")
    assert event["error_type"] == "RuntimeError"
    assert "simulated capability failure" in event["message"]
    # The section still ships — degraded, not replaced by the crash stub.
    assert memo.earnings_agent_view.confidence > 0.0
    # The degraded build must not have become NVDA's 90-day fundamentals
    # snapshot: with the feed healthy again, a plain read serves real
    # earnings (an older snapshot or a fresh fetch), so this test cannot
    # poison every later test in the session — or every later memo in prod.
    from app.services.fundamentals_service import get_full_financials
    monkeypatch.setattr(DataService, "get_earnings", original)
    assert get_full_financials("NVDA")["earnings"], "degraded build was cached"


def test_profile_failure_still_raises_value_error():
    """Unchanged contract: no profile means no memo, not a degraded one."""
    with pytest.raises(ValueError, match="Unknown ticker"):
        graph.run_stock_memo("ZZZZNOPE")


# ---------------------------------------------------------------------------
# 2. note_soft — a no-op outside a memo run, recorded once inside one
# ---------------------------------------------------------------------------

def test_note_soft_outside_a_run_returns_false_and_records_nothing():
    assert active_log() is None
    assert note_soft("Thesis Builder", "no run active") is False
    assert active_log() is None


def test_note_soft_inside_a_run_records_once():
    log = DegradationLog()
    with log.activate() as active:
        assert active is log
        assert active_log() is log
        assert note_soft("Thesis Builder", "first") is True
        # Dedupe per agent: the second call still reports "handled" but
        # does not double the banner entry.
        assert note_soft("Thesis Builder", "second") is True
    assert log.degraded_agents() == ["Thesis Builder"]
    assert log.events() == [{
        "agent": "Thesis Builder",
        "error_type": "DeterministicFallback",
        "message": "first",
    }]
    assert active_log() is None


def test_second_run_in_the_same_thread_starts_empty():
    """The regen worker runs memos back to back in one long-lived thread.
    `activate()` resets the ContextVar token in `finally`, so neither a
    clean run nor one that raised can leak its log into the next."""
    first = DegradationLog()
    with first.activate():
        note_soft("Thesis Builder", "from run one")
    second = DegradationLog()
    with second.activate():
        assert active_log() is second
        assert second.degraded_agents() == []
        note_soft("PM Synthesis", "from run two")
    assert first.degraded_agents() == ["Thesis Builder"]
    assert second.degraded_agents() == ["PM Synthesis"]

    third = DegradationLog()
    with pytest.raises(RuntimeError):
        with third.activate():
            raise RuntimeError("run three blew up")
    assert active_log() is None
    assert note_soft("Anything", "after the failed run") is False


def test_events_is_a_copy_not_an_alias():
    log = DegradationLog()
    log.record_soft("PM Synthesis", "fallback")
    snapshot = log.events()
    log.record_soft("Thesis Builder", "another")
    assert [e["agent"] for e in snapshot] == ["PM Synthesis"]
    snapshot[0]["agent"] = "mutated"
    assert log.failures[0]["agent"] == "PM Synthesis"


def test_run_stock_memo_activates_the_log_for_helpers_without_a_handle(monkeypatch):
    """`_build_thesis_from_findings` has no DegradationLog parameter; it
    reports through `note_soft`. A failing gap clause must therefore reach
    the memo's banner only because `run_stock_memo` activated the log."""
    monkeypatch.setattr(graph, "_market_gap_clause", _boom)
    memo = graph.run_stock_memo("NVDA")
    assert memo.degraded_agents.count("Thesis Builder") == 1
    event = next(e for e in memo.degradation_events if e["agent"] == "Thesis Builder")
    assert event["error_type"] == "RuntimeError"
    # And the run left nothing behind for the next memo in this thread.
    assert active_log() is None
    monkeypatch.undo()
    clean = graph.run_stock_memo("NVDA")
    assert "Thesis Builder" not in clean.degraded_agents


def test_first_stage_of_the_inner_body_already_sees_the_memo_log(monkeypatch):
    """Activation lives in `run_stock_memo`'s outer `with` (alongside
    `as_of_context` / `llm_call_context`) rather than inside
    `_run_stock_memo_inner` — the spec's wording. The guarantee that wording
    asked for is what matters: every line of the inner body, from its very
    first stage, runs under the *same* log whose events land on the memo.
    Pin that end to end so a later move of the wrapper (S3 splits the inner
    into stages) cannot open a gap at the top of the run."""
    seen: List[Any] = []
    real_fundamentals = graph._checkpointed_fundamentals

    def spy(*args: Any, **kwargs: Any):
        seen.append(active_log())
        return real_fundamentals(*args, **kwargs)

    monkeypatch.setattr(graph, "_checkpointed_fundamentals", spy)
    # A late-run soft failure gives the log a known entry to compare on.
    monkeypatch.setattr(graph, "_market_gap_clause", _boom)

    memo = graph.run_stock_memo("NVDA")

    assert len(seen) == 1, "fundamentals is the first stage and runs once"
    assert isinstance(seen[0], DegradationLog)
    # Same accumulator object, start to finish: what the first stage could
    # have written to is exactly what the memo reports.
    assert seen[0].events() == memo.degradation_events
    assert "Thesis Builder" in memo.degraded_agents
    assert active_log() is None


# ---------------------------------------------------------------------------
# 3. LLM configured but returning nothing: the PM paths say so — with no
#    real client ever constructed
# ---------------------------------------------------------------------------

def test_pm_paths_flag_deterministic_fallback_when_llm_returns_nothing(monkeypatch):
    constructed: List[str] = []

    def _forbid(name: str):
        def _factory(*args: Any, **kwargs: Any) -> None:
            constructed.append(name)
            raise AssertionError(f"{name} must not be constructed under blank keys")
        return _factory

    # Belt and braces: keys blank (so `embeddings.embed` takes its hash
    # path), every client factory forbidden, every chat entry point patched
    # to the "nothing usable" outcome. `has_llm` is what gates the PM
    # adjuster and the fallback promotion, so it is patched on.
    for key in ("openai_api_key", "anthropic_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, key, "")
    for factory in ("_openai_client", "_anthropic_client", "_gemini_client"):
        monkeypatch.setattr(llm, factory, _forbid(factory))
    for entry in ("chat_json", "chat_text", "gemini_chat_json", "gemini_chat_text"):
        monkeypatch.setattr(llm, entry, lambda *a, **k: None)
    monkeypatch.setattr(Settings, "has_llm", PropertyMock(return_value=True))
    assert settings.has_llm

    memo = graph.run_stock_memo("NVDA")

    assert constructed == [], constructed
    assert "PM Synthesis" in memo.degraded_agents
    assert "PM DCF Adjuster" in memo.degraded_agents
    by_agent: Dict[str, Dict[str, Any]] = {e["agent"]: e for e in memo.degradation_events}
    assert by_agent["PM Synthesis"]["error_type"] == "DeterministicFallback"
    assert by_agent["PM DCF Adjuster"]["error_type"] == "DeterministicFallback"
    assert "no usable proposal" in by_agent["PM DCF Adjuster"]["message"]
    # No provider was reached, so nothing failed over either.
    assert "LLM provider" not in memo.degraded_agents


def test_pm_dcf_adjuster_llm_exception_is_distinguishable_from_consensus(monkeypatch):
    """Before RP-001 a crashed adjuster and 'PM agreed with consensus' both
    returned (None, [], '') with nothing on the banner."""
    from app.agents import dcf_pm_adjuster as adj

    monkeypatch.setattr(type(adj.settings), "has_llm", PropertyMock(return_value=True))
    monkeypatch.setattr(adj.llm, "chat_json", _boom)
    log = DegradationLog()
    dcf = _minimal_dcf()
    with log.activate():
        out = adj.adjust_dcf_for_pm_view(
            ticker="NVDA", initial_dcf=dcf, findings={}, run_id="t",
        )
    assert out == (None, [], "")
    assert log.degraded_agents() == ["PM DCF Adjuster"]
    assert log.failures[0]["error_type"] == "RuntimeError"

    # Outside a run the same failure is still a no-op for the caller.
    monkeypatch.setattr(adj.llm, "chat_json", lambda *a, **k: None)
    assert adj.adjust_dcf_for_pm_view(
        ticker="NVDA", initial_dcf=dcf, findings={}, run_id="t",
    ) == (None, [], "")


def _minimal_dcf():
    from app.schemas import DCFAssumptions, DCFResult, DCFScenario
    a = DCFAssumptions(
        revenue_growth=[0.1] * 5, operating_margin=[0.25] * 5, tax_rate=0.21,
        da_pct_revenue=0.04, capex_pct_revenue=0.05, nwc_pct_revenue=0.02,
        terminal_growth=0.025, exit_ebitda_multiple=15.0, wacc=0.085,
        base_revenue=1_000_000.0, net_debt=0.0, diluted_shares=1_000_000.0,
        current_price=100.0,
    )
    scen = DCFScenario(
        name="base", label="base", assumptions=a, projections=[],
        pv_explicit=0.0, terminal_value_gordon=0.0, terminal_value_exit_multiple=0.0,
        pv_terminal_gordon=0.0, pv_terminal_exit=0.0,
        enterprise_value_gordon=1.0, enterprise_value_exit=1.0,
        enterprise_value_blended=1.0, equity_value=1.0,
        implied_share_price=100.0, upside_pct=0.0,
    )
    return DCFResult(
        ticker="NVDA", current_price=100.0,
        base=scen,
        bull=scen.model_copy(update={"name": "bull"}),
        bear=scen.model_copy(update={"name": "bear"}),
    )


# ---------------------------------------------------------------------------
# 4. The two verdict-stage safe_calls that used to log nowhere
# ---------------------------------------------------------------------------

def test_valuation_verdict_crash_is_on_the_banner(monkeypatch):
    monkeypatch.setattr(graph, "_build_valuation_verdict", _boom)
    memo = graph.run_stock_memo("NVDA")
    assert "Valuation Verdict" in memo.degraded_agents
    # The fallback is the empty card the reader would otherwise get silently.
    assert memo.valuation_verdict == ValuationVerdict()


def test_mispricing_fallback_crash_is_on_the_banner(monkeypatch):
    monkeypatch.setattr(graph, "_build_mispricing_fallback", _boom)
    memo = graph.run_stock_memo("NVDA")
    assert "Mispricing Fallback" in memo.degraded_agents
    assert memo.degraded_agents == [e["agent"] for e in memo.degradation_events]


# ---------------------------------------------------------------------------
# 5. Additive schema: old rows validate, new fields default
# ---------------------------------------------------------------------------

def _stub_memo(ticker: str = "TDS") -> StockMemoOut:
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
        degraded_agents=["Sector Analyst"],
        degradation_events=[{
            "agent": "Sector Analyst", "error_type": "RuntimeError", "message": "m",
        }],
    )


def test_stored_snapshot_without_the_new_fields_still_validates():
    """Every memo_snapshots row written before RP-001 lacks both fields."""
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import MemoSnapshot

    snap = memo_store.save_memo(_stub_memo())
    with SessionLocal() as db:
        row = db.execute(
            select(MemoSnapshot).where(MemoSnapshot.id == snap.id)
        ).scalar_one()
        legacy = dict(row.memo_json)
        legacy.pop("degradation_events")
        legacy.pop("extra_agent_views")
        row.memo_json = legacy
        db.commit()
    history = memo_store.memo_history("TDS", limit=1)
    assert history and history[0].id == snap.id
    assert "degradation_events" not in history[0].memo_json
    memo = memo_store.memo_to_pydantic(history[0])
    assert memo.degraded_agents == ["Sector Analyst"]
    assert memo.degradation_events == []
    assert memo.extra_agent_views == {}


def test_new_fields_round_trip_through_the_store():
    snap = memo_store.save_memo(_stub_memo("TDR"))
    memo = memo_store.memo_to_pydantic(snap)
    assert memo.degradation_events == [{
        "agent": "Sector Analyst", "error_type": "RuntimeError", "message": "m",
    }]
    assert memo.extra_agent_views == {}


# ---------------------------------------------------------------------------
# 6. Class (c) — ops/telemetry endpoints keep their contract but stop
#    failing silently
# ---------------------------------------------------------------------------

def test_ui_log_ingest_failure_is_logged_but_still_returns_200(monkeypatch, caplog):
    """`POST /api/admin/ui-log` is a browser-called telemetry sink whose
    contract is "always 200". RP-001's (c) rule ("make it raise") is
    deliberately not applied here — a logging hiccup must not 500 the UI —
    but the failure now reaches the server log with a traceback instead of
    vanishing into `ok=False`."""
    import logging

    from fastapi.testclient import TestClient

    from app import database
    from app.main import app

    def _no_db(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated database outage")

    monkeypatch.setattr(database, "SessionLocal", _no_db)
    client = TestClient(app)
    with caplog.at_level(logging.ERROR, logger="app.api.routes_admin"):
        resp = client.post("/api/admin/ui-log", json={"events": [{"kind": "click"}]})
    assert resp.status_code == 200
    assert resp.json() == {"written": 0, "ok": False}
    record = next(r for r in caplog.records if "ui-log ingest failed" in r.getMessage())
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert "simulated database outage" in caplog.text
