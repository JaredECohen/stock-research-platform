"""No model-written or provider-quoted text in the application log
(slice B8-A2a; attribution design §4.9, critique #15).

Each case plants a sentinel where these sites used to leak it — the text
of an exception raised around a model call, a value the model returned, a
traceback — drives the site, and greps every record the INFO-level
production log would carry. The operational line still names the site
and the exception TYPE; the redacted detail stays at DEBUG
(`log_safety.log_safely`).

The last case runs the same call sites through the real LLM layer with a
fake provider client whose response and error text carry the sentinel,
which is what a live memo run does.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import pytest

from app.agents import llm
from app.config import settings
from app.schemas import AgentFinding, CritiqueQuestion, DCFAssumptions
from app.tests import llm_fakes
from app.tests.llm_fakes import FakeClient, anthropic_response, openai_response

SENTINEL = "SENTINEL-privacy-7f3a9"


def _raise(*_a: Any, **_k: Any) -> Any:
    raise RuntimeError(f"provider said: {SENTINEL} (quoted request text)")


def _finding(agent: str = "Sector Analyst") -> AgentFinding:
    return AgentFinding(agent=agent, headline="h", summary="s", key_points=["k"], confidence=0.6)


def _assumptions() -> DCFAssumptions:
    return DCFAssumptions(
        revenue_growth=[0.10, 0.09, 0.08, 0.07, 0.06],
        operating_margin=[0.25, 0.26, 0.27, 0.27, 0.27],
        tax_rate=0.21, da_pct_revenue=0.04, capex_pct_revenue=0.05, nwc_pct_revenue=0.02,
        terminal_growth=0.025, exit_ebitda_multiple=15.0, wacc=0.085,
        base_revenue=1_000_000.0, net_debt=0.0, diluted_shares=1_000_000.0, current_price=100.0,
    )


def _with_llm(monkeypatch) -> None:
    """Sites gated on a configured key (never reached: every call is faked)."""
    monkeypatch.setattr(settings, "openai_api_key", "stub-openai")
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-anthropic")


# --- one driver per cleaned-up site -----------------------------------------

def _dcf_pm_adjuster_call(monkeypatch):
    from app.agents import dcf_pm_adjuster
    monkeypatch.setattr(dcf_pm_adjuster.llm, "chat_json", _raise)
    assert dcf_pm_adjuster._propose_adjustments(
        ticker="PRIV", prior=_assumptions(), findings={"sector": _finding()}, run_id="priv-1",
    ) is None


def _dcf_pm_adjuster_rebuild(monkeypatch):
    from app.agents import dcf_pm_adjuster
    from app.services import valuation_service
    monkeypatch.setattr(dcf_pm_adjuster, "_propose_adjustments", lambda **_k: {
        "updates": {"wacc": 0.09}, "rationales": {"wacc": "higher rates"}, "headline": "raise wacc",
    })
    monkeypatch.setattr(valuation_service, "build_dcf", _raise)
    _with_llm(monkeypatch)
    from app.tests.test_dcf_pm_adjuster import _stub_dcf
    out = dcf_pm_adjuster.adjust_dcf_for_pm_view(
        ticker="PRIV", initial_dcf=_stub_dcf(), findings={"sector": _finding()}, run_id="priv-2")
    assert out[0] is None


def _dcf_updater(monkeypatch):
    from app.agents import dcf_updater
    monkeypatch.setattr(dcf_updater.llm, "chat_json", _raise)
    _with_llm(monkeypatch)
    assert dcf_updater._llm_propose_updates("PRIV", _assumptions(), {}, _assumptions()) is None


def _pm_critique(monkeypatch):
    from app.agents import deep_research
    monkeypatch.setattr(deep_research.llm, "chat_json", _raise)
    deep_research.pm_critique(round_num=0, current_findings={"sector": _finding()},
                              rounds_so_far=[], run_id="priv-3")


def _deep_research_refire(monkeypatch):
    from app.agents import deep_research
    monkeypatch.setattr(deep_research.llm, "chat_json", lambda *a, **k: None)
    deep_research.run_dialog_loop(
        run_id="priv-4", initial_findings={"sector": _finding()},
        re_fire={"sector": _raise}, max_rounds=1,
        seed_questions=[CritiqueQuestion(target_agent="sector", question="why?")],
    )


def _fact_extraction(monkeypatch):
    from app.agents import fact_extraction
    _with_llm(monkeypatch)
    monkeypatch.setattr(fact_extraction.llm, "chat_json", _raise)
    assert fact_extraction._llm_enrich("some filing text", ticker="PRIV", kind="10-K",
                                       source_id="acc-1") is None


def _long_form_enrich(monkeypatch):
    from app.agents import long_form
    monkeypatch.setattr(long_form.llm, "chat_text", _raise)
    assert long_form._enriched_long_form(_finding(), ticker="PRIV", agent_name="Sector Analyst") is None


def _long_form_build(monkeypatch):
    from app.agents import long_form
    monkeypatch.setattr(long_form, "build_long_form_report", _raise)
    long_form.attach_long_form(_finding(), ticker="PRIV", agent_name="Sector Analyst")


def _theme_exposure(monkeypatch):
    from app.services import theme_exposure_service
    _with_llm(monkeypatch)
    monkeypatch.setattr(llm, "chat_json", _raise)
    assert theme_exposure_service._llm_theme_scores("PRIV", "transcript text") is None


def _portfolio_brief(monkeypatch):
    from app.schemas import PortfolioRequest
    from app.services import portfolio_brief
    _with_llm(monkeypatch)
    # int("<model text>") raises a ValueError that QUOTES the model's text,
    # which in turn echoes the user's market view.
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"horizon_years": SENTINEL})
    brief = portfolio_brief.extract_brief(PortfolioRequest(market_view="AI exposure"))
    assert brief is not None


def _scenario_drivers(monkeypatch):
    from app.services import scenario_assumptions
    _with_llm(monkeypatch)
    side = {"growth_bp": 50, "drivers": [{"name": f"{SENTINEL} driver", "rationale": "r"}]}
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"bull": side, "bear": side})
    scenario_assumptions.build_bull_bear({"ticker": "PRIV", "sector": "Technology"}, _assumptions())


def _public_sample_builder(monkeypatch):
    from app.services import public_samples
    monkeypatch.setattr(public_samples, "is_listed", lambda _t: True)
    for name in ("_build_memo", "_build_dcf", "_build_fundamentals", "_build_screener_row"):
        monkeypatch.setattr(public_samples, name, lambda *a, **k: (None, None, []))
    monkeypatch.setattr(public_samples, "_build_prices", lambda *a, **k: (None, None, []))
    monkeypatch.setattr(public_samples, "_build_commentary", lambda *a, **k: (None, None, []))
    monkeypatch.setattr(public_samples, "_build_comps", _raise)
    out = public_samples.build_for_ticker("PRIV")
    assert any("comps: build failed (RuntimeError)" in d for d in out["degraded"])


def _index_research_notes(monkeypatch):
    import scripts.index_research_notes as notes
    _with_llm(monkeypatch)
    monkeypatch.setattr(llm, "chat_text", _raise)
    notes._llm_summarize("A research note body.")


def _reflection_facts(monkeypatch):
    from app.agents import fact_extraction, reflection_agent
    from app.services import history_service
    from app.tests.test_update_orchestrator import _stub_memo
    monkeypatch.setattr(settings, "enable_long_term_memory", True)
    monkeypatch.setattr(history_service, "backfill_ticker", lambda *_a, **_k: None)
    monkeypatch.setattr(fact_extraction, "collect_structured_facts", _raise)
    memo = _stub_memo("PRIV").model_copy(update={"sources_used": ["filing:0000000001-26-000001"]})
    triggers, _written = reflection_agent.run(memo)
    assert triggers, "no delta trigger: the fact-extraction site was not reached"


SITES = {
    "dcf_pm_adjuster.call": _dcf_pm_adjuster_call,
    "dcf_pm_adjuster.rebuild": _dcf_pm_adjuster_rebuild,
    "dcf_updater": _dcf_updater,
    "deep_research.pm_critique": _pm_critique,
    "deep_research.refire": _deep_research_refire,
    "fact_extraction": _fact_extraction,
    "long_form.enrich": _long_form_enrich,
    "long_form.build": _long_form_build,
    "theme_exposure_service": _theme_exposure,
    "portfolio_brief.parse": _portfolio_brief,
    "scenario_assumptions.drivers": _scenario_drivers,
    "public_samples.builder": _public_sample_builder,
    "index_research_notes": _index_research_notes,
    "reflection_agent.facts": _reflection_facts,
}


@pytest.mark.parametrize("site", sorted(SITES))
def test_privacy_sentinel_never_logged(site, monkeypatch, caplog, tmp_path):
    # Sites that write long-term memory write it here, never to the repo's.
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    caplog.set_level(logging.INFO)
    SITES[site](monkeypatch)
    leaked = [f"{r.name} {r.levelname}: {r.getMessage()[:160]}"
              for r in caplog.records if SENTINEL in r.getMessage()]
    assert not leaked, f"{site} logged model or exception text: {leaked}"
    # Rendered text too: a traceback (exc_info) is formatted, not a message.
    assert SENTINEL not in caplog.text, f"{site} logged a traceback carrying the text"


def test_privacy_sentinel_never_logged_through_the_llm_layer(monkeypatch, caplog):
    """A fake-client run: the provider's response and its error text both
    carry the sentinel, and the sites above run through the real
    `chat_json`/`chat_text` (with their `action=`), so the call line, the
    failover line and the agents' own failure logs are all exercised."""
    from app.agents import dcf_updater, long_form
    from app.services import scenario_assumptions

    side = {"growth_bp": 50, "drivers": [{"name": f"{SENTINEL} driver", "rationale": SENTINEL}]}
    good = openai_response(json.dumps({"bull": side, "bear": side}))
    llm_fakes.live(
        monkeypatch,
        openai=FakeClient(good),
        anthropic=FakeClient(RuntimeError(f"upstream 500: {SENTINEL}")),
        active="anthropic",
    )
    caplog.set_level(logging.INFO)
    with llm.llm_call_context(run_id="priv-e2e", origin="worker:regen"):
        # anthropic raises (sentinel in the message) -> failover to openai.
        scenario_assumptions.build_bull_bear({"ticker": "PRIV"}, _assumptions())
        dcf_updater._llm_propose_updates("PRIV", _assumptions(), {"note": SENTINEL}, _assumptions())
        long_form._enriched_long_form(_finding(), ticker="PRIV", agent_name="Sector Analyst")
    assert SENTINEL not in caplog.text
    lines = [r.getMessage() for r in caplog.records if r.name == "app.llm.calls"]
    assert lines, "the fake-client calls reached the LLM layer"
    rows = llm_fakes.rows_for("priv-e2e")
    assert {r.action for r in rows} >= {"dcf.scenarios", "dcf.update", "memo.long_form"}
    # A long-form row names the specialist it expands, not the registry's
    # generic "Long-form Writer" (long_form.py is an AGENT_CONTEXT_SITE);
    # both attempts (the failed one and its failover) say so.
    assert {r.agent_name for r in rows if r.action == "memo.long_form"} == {"Sector Analyst"}
    for row in rows:
        for column in row.__table__.columns.keys():
            assert SENTINEL not in str(getattr(row, column)), column


def test_public_sample_commentary_records_the_served_model(monkeypatch):
    """G20: the stored commentary row says which model actually wrote it
    (sent and provider-reported), in the never-served `basis`."""
    from app.services import public_samples
    client = FakeClient(anthropic_response("A neutral paragraph.", model="claude-haiku-4-5-20251001"))
    llm_fakes.live(monkeypatch, anthropic=client, active="anthropic")
    memo = {"ticker": "PRIV", "rating_label": "Neutral", "section_availability": {}}
    payload, _ref, notes = public_samples._build_commentary(
        "PRIV", memo, now=datetime(2026, 9, 26), memo_source_ref="snapshot:1")
    assert notes == [] and payload is not None
    basis = payload["basis"]
    assert basis["served_model"] == "claude-haiku-4-5-20251001"
    assert basis["provider"] == "anthropic" and basis["model_sent"]
    # Provenance only: the page still gets {text, generated_at, model}.
    assert set(public_samples._public_shape("commentary", payload)) == {"text", "generated_at", "model"}


def _intake_still_logs_its_rationale() -> bool:
    """G1 (FIX-020) owns `intake.py` and replaces its INFO line, which
    prints the PM's model-written rationale, with a character count. Until
    that lands the memo-run test tolerates exactly that one line; the
    allowance switches itself off when the source no longer prints it, so
    after G1 merges the test is strict with no edit."""
    import inspect

    from app.agents import intake
    return "rationale=%s" in inspect.getsource(intake)


def test_privacy_sentinel_never_logged_in_a_memo_run(monkeypatch, caplog, tmp_path):
    """The contract's shape (critique #15): a whole fake-client memo run,
    every provider answering with the sentinel in every field, and not one
    INFO-or-above record, traceback included, carries it."""
    from app.agents.graph import run_stock_memo
    from app.tests.llm_fakes import gemini_response

    body = json.dumps({
        "rating_label": "Neutral", "confidence": 0.6, "confidence_score": 60,
        "headline": SENTINEL, "summary": SENTINEL, "key_points": [SENTINEL],
        "rationale": SENTINEL, "narrative": SENTINEL, "one_sentence_thesis": SENTINEL,
        "final_pm_view": SENTINEL, "final_verdict": SENTINEL, "skip": [],
        "items": [], "questions": [], "updates": {}, "rationales": {},
    })
    llm_fakes.live(
        monkeypatch,
        openai=FakeClient(openai_response(body)),
        anthropic=FakeClient(anthropic_response(body)),
        gemini=FakeClient(gemini_response(body)),
        active="anthropic",
    )
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    caplog.set_level(logging.INFO)
    run_stock_memo("AAPL", run_id="priv-memo-run")
    assert llm_fakes.rows_for("priv-memo-run"), "the memo run never reached the LLM layer"

    tolerate_intake = _intake_still_logs_its_rationale()
    leaked = [
        f"{r.name}:{r.lineno} {r.levelname}"
        for r in caplog.records
        if r.levelno >= logging.INFO and SENTINEL in caplog.handler.format(r)
        and not (tolerate_intake and r.name == "app.agents.intake"
                 and r.getMessage().startswith("PM intake for "))
    ]
    assert not leaked, f"model or exception text reached the log: {leaked}"
