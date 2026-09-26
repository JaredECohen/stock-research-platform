"""One `llm_call_logs` row and one `app.llm.calls` line per provider attempt
(slice B7-M1; design `design-llm-attribution-logging.md` §4.4-4.6 and §7C,
cases C1-C7, with the critique's parametrised failover matrix).

Offline fake clients (`llm_fakes`) answer with a served model that differs
from the one sent, so every assertion about `model_served` would fail if
the code merely copied the model it sent (critique #3). Run this file in
the CI-version venv: served-model attributes differ across SDK versions.
"""
from __future__ import annotations

import logging
import time
import uuid

import pytest

from app.agents import llm, llm_attribution
from app.config import settings
from app.models import LLMCallLog
from app.tests import llm_fakes
from app.tests.llm_fakes import FakeClient, anthropic_response, gemini_response, openai_response


def _run_id() -> str:
    return f"line-{uuid.uuid4().hex[:10]}"


def _lines(caplog) -> list[dict[str, str]]:
    return [llm_fakes.parse_line(r.getMessage()) for r in caplog.records if r.name == "app.llm.calls"]


def _records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "app.llm.calls"]


@pytest.fixture(autouse=True)
def _reset():
    llm.reset_circuit_breaker()
    llm.reset_failover_state()
    yield
    llm.reset_circuit_breaker()
    llm.reset_failover_state()


# ---------------------------------------------------------------------------
# C1: success -> exactly one INFO line whose keys are the §4.6 list, and a row
# ---------------------------------------------------------------------------

def test_success_writes_one_line_and_one_linked_row(monkeypatch, caplog):
    client = FakeClient(openai_response(model="gpt-5.4-2026-03-01", prompt_tokens=1000,
                                        completion_tokens=200))
    llm_fakes.live(monkeypatch, openai=client)
    run_id = _run_id()
    with caplog.at_level(logging.INFO, logger="app.llm.calls"), \
            llm.llm_call_context(run_id=run_id, origin="worker:regen", job_id="regen:25",
                                 feature="research_run"):
        out = llm.chat_json("p", route="cheap", model="gpt-5.4", action="analyst.sector",
                            ticker="NVDA")
    assert out == {"ok": True}
    records = _records(caplog)
    assert len(records) == 1 and records[0].levelno == logging.INFO
    line = llm_fakes.parse_line(records[0].getMessage())
    assert list(line) == ["v", *llm_attribution.LINE_FIELDS]
    assert line["outcome"] == "ok" and line["attempt"] == "1"
    assert line["agent"] == "Sector Analyst" and line["action"] == "analyst.sector"
    assert line["model_requested"] == "gpt-5.4" and line["model"] == "gpt-5.4"
    assert line["model_served"] == "gpt-5.4-2026-03-01"
    assert line["resolution"] == "explicit"
    assert (line["origin"], line["job"], line["ticker"], line["route"]) == (
        "worker:regen", "regen:25", "NVDA", "cheap")
    (row,) = llm_fakes.rows_for(run_id)
    assert row.call_id == line["call"] and len(row.call_id) == 32
    assert row.attempt == 1 and row.served_model == "gpt-5.4-2026-03-01"
    assert row.ticker == "NVDA" and row.route == "cheap" and row.job_id == "regen:25"
    assert row.origin == "worker:regen" and row.process_role in ("web", "worker")
    assert row.requested_provider == "openai" and row.requested_model == "gpt-5.4"
    assert row.cost_usd == pytest.approx(float(line["cost_usd"])) and row.cost_usd > 0
    assert row.error_type is None and row.finish_reason == "stop" == line["finish"]
    usage = llm.last_usage()
    assert usage["call_id"] == row.call_id and usage["served_model"] == "gpt-5.4-2026-03-01"
    assert usage["cost_usd"] == pytest.approx(row.cost_usd)


# ---------------------------------------------------------------------------
# C2: failover, parametrised over direction x entry x reason (critique #3)
# ---------------------------------------------------------------------------

_BAD = {"chat_json": "no json here", "chat_text": ""}


@pytest.mark.parametrize("primary,partner", [("openai", "anthropic"), ("anthropic", "openai")])
@pytest.mark.parametrize("entry", ["chat_json", "chat_text"])
@pytest.mark.parametrize("reason", ["call_failed", "breaker_open", "client_unavailable"])
def test_failover_links_two_attempts(monkeypatch, caplog, primary, partner, entry, reason):
    served = {"openai": "gpt-served-2026-02-02", "anthropic": "claude-served-2026-02-02"}
    good = '{"ok": true}'

    def _resp(provider: str, text: str):
        if provider == "openai":
            return openai_response(text or None, model=served["openai"])
        return anthropic_response(text, model=served["anthropic"],
                                  stop_reason="max_tokens" if not text or "json" in text else "end_turn")

    clients = {
        primary: FakeClient(_resp(primary, _BAD[entry])),
        partner: FakeClient(_resp(partner, good)),
    }
    if reason == "client_unavailable":
        clients[primary] = None
    llm_fakes.live(monkeypatch, active=primary, **clients)
    if reason == "breaker_open":
        llm._FAILURE_COUNTERS[primary] = llm._BREAKER_THRESHOLD
        llm._FAILURE_LAST_AT[primary] = time.time()
    run_id = _run_id()
    fn = getattr(llm, entry)
    with caplog.at_level(logging.INFO), llm.llm_call_context(run_id=run_id):
        out = fn("prompt", route="strong", action="analyst.comps", ticker="MSFT")
    assert out == ({"ok": True} if entry == "chat_json" else good)

    first, second = llm_fakes.rows_for(run_id)
    assert first.call_id == second.call_id
    assert (first.attempt, second.attempt) == (1, 2)
    assert (first.provider, second.provider) == (primary, partner)
    primary_requested = llm._model_for(primary, "strong")
    assert first.requested_model == second.requested_model == primary_requested
    assert second.model == llm._model_for(partner, "strong")
    assert second.served_model == served[partner]
    assert second.failover_reason == reason and second.model_resolution == "failover_default"
    assert second.success is True
    if reason == "call_failed":
        # A response arrived (unusable), so the provider's own name is kept.
        assert first.served_model == served[primary]
        assert first.error_type in ("invalid_json_response", "empty_response")
    else:
        assert first.served_model is None and first.tokens_in == 0
        assert first.error_type == f"skipped:{reason}"

    lines = _lines(caplog)
    assert [ln["attempt"] for ln in lines] == ["1", "2"]
    assert {ln["call"] for ln in lines} == {first.call_id}
    assert lines[0]["outcome"] == ("error" if reason == "call_failed" else "skipped")
    assert lines[1]["failover_from"] == primary and lines[1]["failover_reason"] == reason
    failover = [r.getMessage() for r in caplog.records
                if r.name == "app.agents.llm" and r.getMessage().startswith("LLM failover")]
    assert len(failover) == 1 and f"call={first.call_id}" in failover[0]
    assert f"from {primary} to {partner} ({reason})" in failover[0]


def test_gemini_served_model_differs_from_sent_and_vertex(monkeypatch):
    """Attempt rows keep the provider's own `model_version`, even when the
    model sent came from VERTEX_MODEL (the resolution says so)."""
    client = FakeClient(gemini_response(model_version="gemini-served-xyz"))
    llm_fakes.live(monkeypatch, gemini=client)
    monkeypatch.setattr(settings, "vertex_project_id", "proj")
    monkeypatch.setattr(settings, "vertex_model", "gemini-2.5-pro")
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        llm.gemini_chat_text("p", action="news.search", ticker="NVDA")
    (row,) = llm_fakes.rows_for(run_id)
    assert row.model == "gemini-2.5-pro" and row.served_model == "gemini-served-xyz"
    assert row.model_resolution == "vertex_override"
    assert row.requested_model == settings.gemini_news_model


# ---------------------------------------------------------------------------
# C3: skips get a row and a WARNING line
# ---------------------------------------------------------------------------

def test_primary_breaker_open_with_no_partner_writes_one_skip_row(monkeypatch, caplog):
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response()))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    llm._FAILURE_COUNTERS["openai"] = llm._BREAKER_THRESHOLD
    llm._FAILURE_LAST_AT["openai"] = time.time()
    run_id = _run_id()
    with caplog.at_level(logging.INFO), llm.llm_call_context(run_id=run_id):
        assert llm.chat_json("p", action="mispricing.audit") is None
    (row,) = llm_fakes.rows_for(run_id)
    assert row.error_type == "skipped:breaker_open" and row.success is False
    assert row.tokens_in == row.tokens_out == 0 and row.duration_ms == 0
    (line,) = _lines(caplog)
    assert line["outcome"] == "skipped" and line["error"] == "skipped:breaker_open"
    assert _records(caplog)[0].levelno == logging.WARNING


def test_partner_breaker_open_writes_the_partner_skip(monkeypatch):
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response("bad")),
                   anthropic=FakeClient(anthropic_response()))
    llm._FAILURE_COUNTERS["anthropic"] = llm._BREAKER_THRESHOLD
    llm._FAILURE_LAST_AT["anthropic"] = time.time()
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        assert llm.chat_json("p", action="analyst.sector") is None
    first, second = llm_fakes.rows_for(run_id)
    assert second.attempt == 2 and second.error_type == "skipped:partner_breaker_open"
    assert llm.get_failover_state()["count"] == 0


def test_gemini_breaker_skip_is_written(monkeypatch):
    llm_fakes.live(monkeypatch, gemini=FakeClient(gemini_response()))
    llm._FAILURE_COUNTERS["gemini"] = llm._BREAKER_THRESHOLD
    llm._FAILURE_LAST_AT["gemini"] = time.time()
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        assert llm.gemini_chat_json("p", action="news.search") is None
    (row,) = llm_fakes.rows_for(run_id)
    assert row.provider == "gemini" and row.error_type == "skipped:breaker_open"


def test_demo_only_writes_no_row_and_does_not_hop(monkeypatch):
    """No client in demo-only mode is configuration, not an outage: no row
    (every CI run would write one) and no failover to a partner that has
    no client either (critique #2)."""
    llm_fakes.live(monkeypatch, openai=None, anthropic=None)
    monkeypatch.setattr(llm, "_demo_only", lambda: True)
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        assert llm.chat_json("p", action="analyst.sector") is None
    assert llm_fakes.rows_for(run_id) == []
    assert llm.get_failover_state()["count"] == 0
    assert llm.failover_partner("openai") is None


# ---------------------------------------------------------------------------
# C4: a provider-foreign override is recorded, not silently dropped
# ---------------------------------------------------------------------------

def test_foreign_override_is_recorded(monkeypatch):
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response()))
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        llm.chat_json("p", route="cheap", model="claude-haiku-4-5", action="analyst.sector")
    (row,) = llm_fakes.rows_for(run_id)
    assert row.model_resolution == "foreign_override_dropped"
    assert row.requested_model == "claude-haiku-4-5"
    assert row.model == settings.openai_cheap_model


# ---------------------------------------------------------------------------
# C5: privacy
# ---------------------------------------------------------------------------

def test_no_prompt_response_exception_or_key_reaches_a_line_or_row(monkeypatch, caplog):
    secret = "sk-ant-SENTINELKEY0123456789abcdefghijklmnopqrstuvwxyz"
    boom = RuntimeError("EXCEPTION-SENTINEL with key " + secret)
    llm_fakes.live(monkeypatch, openai=FakeClient(boom),
                   anthropic=FakeClient(anthropic_response("RESPONSE-SENTINEL not json")))
    monkeypatch.setattr(settings, "anthropic_api_key", secret)
    run_id = _run_id()
    with caplog.at_level(logging.DEBUG), llm.llm_call_context(run_id=run_id):
        llm.chat_json("PROMPT-SENTINEL", system="SYSTEM-SENTINEL", action="analyst.sector")
    rows = llm_fakes.rows_for(run_id)
    assert len(rows) == 2
    for sentinel in ("PROMPT-SENTINEL", "SYSTEM-SENTINEL", "RESPONSE-SENTINEL",
                     "EXCEPTION-SENTINEL", "SENTINELKEY"):
        assert sentinel not in caplog.text
        for row in rows:
            for column in LLMCallLog.__table__.columns:
                assert sentinel not in str(getattr(row, column.name)), column.name
    for record in _records(caplog):
        message = record.getMessage()
        assert "\n" not in message and len(message) < 1000
        assert llm.redact_unbounded(message) == message


# ---------------------------------------------------------------------------
# C6: the kill switch silences only successful INFO lines
# ---------------------------------------------------------------------------

def test_kill_switch_keeps_warnings(monkeypatch, caplog):
    monkeypatch.setattr(settings, "llm_call_log_enabled", False)
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response("not json"), openai_response()))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    run_id = _run_id()
    with caplog.at_level(logging.INFO), llm.llm_call_context(run_id=run_id):
        assert llm.chat_json("p", action="analyst.sector") is None     # error
        assert llm.chat_json("p", action="analyst.sector") == {"ok": True}  # ok
    lines = _lines(caplog)
    assert [ln["outcome"] for ln in lines] == ["error"]
    assert len(llm_fakes.rows_for(run_id)) == 2, "rows are always written"


# ---------------------------------------------------------------------------
# C7: breaker transitions log once each
# ---------------------------------------------------------------------------

def test_breaker_transition_lines_once(monkeypatch, caplog):
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response("bad")))
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with caplog.at_level(logging.WARNING, logger="app.agents.llm"):
        for _ in range(5):
            llm.chat_json("p", action="analyst.sector")
        # Cooldown elapses: the next check resets, once.
        llm._FAILURE_LAST_AT["openai"] = time.time() - llm._BREAKER_COOLDOWN_SECONDS - 1
        assert not llm.breaker_open("openai")
        assert not llm.breaker_open("openai")
    breaker = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm_breaker")]
    assert len(breaker) == 2
    assert breaker[0].startswith("llm_breaker state=open provider=openai failures=3 proc=")
    assert breaker[1].startswith("llm_breaker state=reset provider=openai reason=cooldown proc=")


# ---------------------------------------------------------------------------
# Row hygiene
# ---------------------------------------------------------------------------

def test_strings_truncated_to_column_length(monkeypatch):
    """Postgres rejects an over-long value and the swallowed INSERT would
    lose the whole row (critique #6); every string column is cut to its
    declared length."""
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response(model="m" * 300)))
    run_id = "r" * 40
    with llm.llm_call_context(run_id=run_id, origin="o" * 300, job_id="j" * 300,
                              feature="f" * 300, agent_name="A" * 300):
        llm.chat_json("p", ticker="T" * 300, action="analyst.sector", model="gpt-" + "x" * 300)
    (row,) = llm_fakes.rows_for(run_id)
    for column in LLMCallLog.__table__.columns:
        length = getattr(column.type, "length", None)
        value = getattr(row, column.name)
        if length and isinstance(value, str):
            assert len(value) <= length, column.name
    assert row.origin == "o" * 96 and row.served_model == "m" * 96 and row.ticker == "T" * 16


def test_reasoning_tokens_not_double_billed(monkeypatch):
    """OpenAI completion_tokens already include reasoning tokens: stored
    for information, never added to billed output (critique #13)."""
    from app.services import llm_metrics
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response(
        prompt_tokens=1000, completion_tokens=500, reasoning_tokens=400)))
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        llm.chat_json("p", model="gpt-5.5", action="analyst.sector")
    (row,) = llm_fakes.rows_for(run_id)
    assert (row.tokens_out, row.reasoning_tokens) == (500, 400)
    assert row.cost_usd == pytest.approx(
        llm_metrics.estimate_cost_usd("openai", "gpt-5.5", 1000, 500))


def test_default_origin_derivation(monkeypatch):
    import sys
    from types import SimpleNamespace

    from app import runtime_role
    monkeypatch.delenv(runtime_role.PROCESS_ROLE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "__main__",
                        SimpleNamespace(__spec__=SimpleNamespace(name="app.worker")))
    assert runtime_role.default_origin() == "worker:other"
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(__spec__=None,
                                                                 __file__="/usr/bin/uvicorn"))
    monkeypatch.setattr(sys, "argv", ["/usr/bin/uvicorn", "app.main:app"])
    assert runtime_role.default_origin() == "web:other"
    monkeypatch.setitem(sys.modules, "__main__",
                        SimpleNamespace(__spec__=SimpleNamespace(name="scripts.postmortem_backfill")))
    monkeypatch.setattr(sys, "argv", ["/app/backend/scripts/postmortem_backfill.py"])
    assert runtime_role.default_origin() == "script:postmortem_backfill"
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(__spec__=None,
                                                                 __file__="/x/corpus_repair.py"))
    monkeypatch.setattr(sys, "argv", ["/x/corpus_repair.py"])
    assert runtime_role.default_origin() == "script:corpus_repair"


def test_a_row_with_no_context_origin_gets_the_process_default(monkeypatch):
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response()))
    monkeypatch.setattr(llm, "_DEFAULT_ORIGIN", "script:unit")
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        llm.chat_json("p", action="analyst.sector")
    (row,) = llm_fakes.rows_for(run_id)
    assert row.origin == "script:unit"


def test_gemini_via_chat_json_one_row_one_line_one_guard(monkeypatch, caplog):
    """chat_json -> gemini_chat_json -> gemini_chat_text is ONE call: one
    guard evaluation, one row, one line (critique #17)."""
    llm_fakes.live(monkeypatch, gemini=FakeClient(gemini_response()))
    checks: list[str] = []
    real = llm_attribution.check

    def _spy(entry, action, *, mode):
        checks.append(entry)
        return real(entry, action, mode=mode)

    monkeypatch.setattr(llm_attribution, "check", _spy)
    run_id = _run_id()
    with caplog.at_level(logging.INFO, logger="app.llm.calls"), llm.llm_call_context(run_id=run_id):
        assert llm.chat_json("p", provider_override="gemini", action="news.search") == {"ok": True}
    assert checks == ["chat_json"]
    (row,) = llm_fakes.rows_for(run_id)
    assert row.action == "news.search" and row.agent_name == "News Agent"
    assert len(_lines(caplog)) == 1


def test_call_cost_accumulates_across_attempts(monkeypatch):
    """`last_usage()` used to describe only the final attempt, so a failed
    first attempt's spend vanished from the judge's budget (G19)."""
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response("bad", prompt_tokens=1000)),
                   anthropic=FakeClient(anthropic_response(input_tokens=1000)))
    run_id = _run_id()
    with llm.llm_call_context(run_id=run_id):
        llm.chat_json("p", action="learning.judge")
    first, second = llm_fakes.rows_for(run_id)
    usage = llm.last_usage()
    assert usage["call_cost_usd"] == pytest.approx(first.cost_usd + second.cost_usd)
    assert usage["cost_usd"] == pytest.approx(second.cost_usd)


def test_the_row_and_line_carry_the_real_process_role(monkeypatch, caplog):
    """`process_role` must be the process that made the call. A constant
    "web" is exactly the cron-health mislabel (every worker loop reported
    as web), so the worker role is forced and checked by value."""
    from app import runtime_role
    monkeypatch.setenv(runtime_role.PROCESS_ROLE_ENV, "worker")
    monkeypatch.setattr(llm, "_PROC", None)      # computed once per process
    llm_fakes.live(monkeypatch, openai=FakeClient(openai_response()))
    run_id = _run_id()
    with caplog.at_level(logging.INFO, logger="app.llm.calls"), \
            llm.llm_call_context(run_id=run_id):
        llm.chat_json("p", action="analyst.sector")
    (row,) = llm_fakes.rows_for(run_id)
    assert row.process_role == "worker"
    (line,) = _lines(caplog)
    assert line["proc"] == "worker"


def test_failover_events_are_capped_but_every_failover_is_counted(monkeypatch):
    """Long-lived loop threads never drain their context's event list, so
    it is bounded (design gap G18); the process counter still sees all."""
    monkeypatch.setattr(llm, "_FAILOVER_EVENTS_MAX", 2)
    llm.reset_failover_state()
    for _ in range(5):
        llm._record_failover("openai", "anthropic", "call_failed")
    assert len(llm.consume_failover_events()) == 2
    assert llm.get_failover_state()["count"] == 5
