"""The attribution guard, the action registry, and the context precedence
(slice B7-M1; design `design-llm-attribution-logging.md` §4.2-4.3, §7B).

A production call site is simulated by compiling a probe with a filename
under `backend/app/agents/`, so the guard's frame walk sees a production
path; calls made directly from this file are exempt (they test the LLM
layer itself). Nothing here builds a real client.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import pytest

from app.agents import llm, llm_attribution
from app.config import settings
from app.tests import llm_fakes

_APP = Path(llm.__file__).resolve().parents[1]
PROBE_FILE = str(_APP / "agents" / "_probe_site.py")


def _probe(src: str) -> dict:
    """Run `src` as if it lived in app/agents/_probe_site.py."""
    ns: dict = {"llm": llm}
    exec(compile(src, PROBE_FILE, "exec"), ns)
    return ns


@pytest.fixture(autouse=True)
def _clean_guard():
    llm_attribution.reset_violations()
    yield
    llm_attribution.reset_violations()


# ---------------------------------------------------------------------------
# G0: a nested context no longer resets the agent
# ---------------------------------------------------------------------------

def test_g0_nested_context_keeps_agent():
    """`_pm_synthesis` opens `llm_call_context(static_prefix_chars=…)` inside
    the "PM Synthesis" context; the old default agent_name="unknown" was
    truthy and survived the merge, so every PM Synthesis row read "unknown"."""
    with llm.llm_call_context(agent_name="PM Synthesis", run_id="g0"):
        with llm.llm_call_context(static_prefix_chars=10):
            ctx = llm.current_call_context()
    assert ctx["agent_name"] == "PM Synthesis"
    assert ctx["static_prefix_chars"] == 10
    assert ctx["run_id"] == "g0"


def test_context_carries_the_new_attribution_fields():
    with llm.llm_call_context(origin="loop:news_loop", job_id="regen:25"):
        with llm.llm_call_context(ticker="NVDA", action="news.search", role="news"):
            ctx = llm.current_call_context()
    assert (ctx["origin"], ctx["job_id"], ctx["ticker"], ctx["action"], ctx["role"]) == (
        "loop:news_loop", "regen:25", "NVDA", "news.search", "news")


# ---------------------------------------------------------------------------
# The guard: strict / warn / off / test-exempt
# ---------------------------------------------------------------------------

def test_guard_strict_raises_and_names_the_site(monkeypatch):
    monkeypatch.setattr(settings, "llm_attribution_mode", "strict")
    ns = _probe("def probe():\n    return llm.chat_json('x')\n")
    with pytest.raises(llm_attribution.LLMAttributionError) as err:
        ns["probe"]()
    assert "app/agents/_probe_site.py:probe" in str(err.value)
    assert llm_attribution.VIOLATIONS == {"app/agents/_probe_site.py:probe": 1}


def test_guard_strict_rejects_an_unregistered_action(monkeypatch):
    monkeypatch.setattr(settings, "llm_attribution_mode", "strict")
    ns = _probe("def probe():\n    return llm.chat_text('x', action='made.up')\n")
    with pytest.raises(llm_attribution.LLMAttributionError, match="action=made.up"):
        ns["probe"]()


def test_guard_runs_before_the_no_provider_short_circuit(monkeypatch):
    """CI has no keys, so `provider == "none"` returns first; the guard is
    the first statement of every entry, so it still fires (critique #9)."""
    monkeypatch.setattr(settings, "llm_attribution_mode", "strict")
    monkeypatch.setattr(type(settings), "active_llm_provider", property(lambda self: "none"))
    for entry in ("chat_json", "chat_text", "gemini_chat_json", "gemini_chat_text"):
        ns = _probe(f"def probe():\n    return llm.{entry}('x')\n")
        with pytest.raises(llm_attribution.LLMAttributionError, match=f"entry={entry}"):
            ns["probe"]()


def test_guard_warn_logs_once_per_site_and_the_row_is_unattributed(monkeypatch, caplog):
    llm_fakes.live(monkeypatch, openai=llm_fakes.FakeClient(llm_fakes.openai_response()))
    monkeypatch.setattr(settings, "llm_attribution_mode", "warn")
    run_id = f"guard-{uuid.uuid4().hex[:8]}"
    ns = _probe("def probe():\n    return llm.chat_json('x')\n")
    with caplog.at_level(logging.WARNING, logger="app.agents.llm_attribution"), \
            llm.llm_call_context(run_id=run_id):
        assert ns["probe"]() == {"ok": True}
        assert ns["probe"]() == {"ok": True}
    warnings = [r.getMessage() for r in caplog.records if "llm_attribution_missing" in r.getMessage()]
    assert warnings == [
        "llm_attribution_missing entry=chat_json site=app/agents/_probe_site.py:probe action=-"
    ]
    assert llm_attribution.VIOLATIONS == {"app/agents/_probe_site.py:probe": 2}
    rows = llm_fakes.rows_for(run_id)
    assert len(rows) == 2 and {r.agent_name for r in rows} == {"unattributed"}


def test_guard_off_records_nothing(monkeypatch):
    monkeypatch.setattr(settings, "llm_attribution_mode", "off")
    ns = _probe("def probe():\n    return llm.chat_json('x')\n")
    ns["probe"]()
    assert llm_attribution.VIOLATIONS == {}


def test_a_test_caller_is_exempt(monkeypatch):
    monkeypatch.setattr(settings, "llm_attribution_mode", "strict")
    llm.chat_json("direct unit test of the LLM layer")
    assert llm_attribution.VIOLATIONS == {}


def test_a_registered_action_passes_strict(monkeypatch):
    monkeypatch.setattr(settings, "llm_attribution_mode", "strict")
    ns = _probe("def probe():\n    return llm.chat_json('x', action='analyst.sector')\n")
    ns["probe"]()
    assert llm_attribution.VIOLATIONS == {}


def test_an_action_on_the_context_also_attributes(monkeypatch):
    monkeypatch.setattr(settings, "llm_attribution_mode", "strict")
    ns = _probe(
        "def probe():\n"
        "    with llm.llm_call_context(action='news.search'):\n"
        "        return llm.gemini_chat_text('x')\n"
    )
    ns["probe"]()
    assert llm_attribution.VIOLATIONS == {}


def test_prod_forces_warn(monkeypatch, caplog):
    """A strict raise inside the memo pipeline is swallowed by safe_call
    and would silently stub every memo, so production never runs strict
    (critique #10), and says so once."""
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "llm_attribution_mode", "strict")
    monkeypatch.setattr(llm, "_PROD_WARN_NOTED", False)
    with caplog.at_level(logging.WARNING, logger="app.agents.llm"):
        assert llm.attribution_mode() == "warn"
        assert llm.attribution_mode() == "warn"
    notes = [r for r in caplog.records if "LLM_ATTRIBUTION_MODE=strict ignored" in r.getMessage()]
    assert len(notes) == 1
    ns = _probe("def probe():\n    return llm.chat_json('x')\n")
    ns["probe"]()  # no raise
    assert llm_attribution.VIOLATIONS == {"app/agents/_probe_site.py:probe": 1}


def test_violations_are_bounded_by_site(monkeypatch):
    """Warn mode on a days-long worker: a bounded site -> count dict, never
    a per-call list (critique #11)."""
    monkeypatch.setattr(settings, "llm_attribution_mode", "warn")
    monkeypatch.setattr(llm_attribution, "VIOLATIONS_MAX_SITES", 3)
    for i in range(6):
        ns = _probe(f"def probe_{i}():\n    return llm.chat_json('x')\n")
        ns[f"probe_{i}"]()
        ns[f"probe_{i}"]()
    assert len(llm_attribution.VIOLATIONS) == 3
    assert set(llm_attribution.VIOLATIONS.values()) == {2}


# ---------------------------------------------------------------------------
# Precedence: which agent a row names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctx_agent,expected", [
    ("PM Synthesis", "PM Synthesis"),          # a specific context agent wins
    ("run_stock_memo", "Sector Analyst"),      # an umbrella yields to the registry
    ("unknown", "Sector Analyst"),
    (None, "Sector Analyst"),
])
def test_agent_precedence(monkeypatch, ctx_agent, expected):
    llm_fakes.live(monkeypatch, openai=llm_fakes.FakeClient(llm_fakes.openai_response()))
    run_id = f"prec-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(agent_name=ctx_agent, run_id=run_id):
        llm.chat_json("x", action="analyst.sector")
    (row,) = llm_fakes.rows_for(run_id)
    assert row.agent_name == expected
    assert row.action == "analyst.sector" and row.role == "analyst"


def test_an_umbrella_name_is_kept_when_no_action_resolves(monkeypatch):
    """Until slice A2a adds action= everywhere, a memo-run call with no
    action keeps the run-level name rather than degrading to nothing."""
    llm_fakes.live(monkeypatch, openai=llm_fakes.FakeClient(llm_fakes.openai_response()))
    run_id = f"umb-{uuid.uuid4().hex[:8]}"
    with llm.llm_call_context(agent_name="run_stock_memo", run_id=run_id):
        llm.chat_json("x")
    (row,) = llm_fakes.rows_for(run_id)
    assert row.agent_name == "run_stock_memo"


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

def test_action_registry_is_well_formed():
    for name, spec in llm_attribution.ACTIONS.items():
        assert llm_attribution.ACTION_RE.match(name), name
        assert len(name) <= llm_attribution.ACTION_MAX_LEN, name
        assert spec.role in llm_attribution.ROLES, name
        assert spec.tier in llm_attribution.TIERS, name
        assert spec.kind in ("chat", "embed", "sdk"), name
        assert spec.agent and spec.agent not in llm_attribution.UMBRELLA_AGENTS, name
        for key in (spec.effort_key, spec.failover_effort_key):
            assert key is None or key in llm_attribution.EFFORT_KEYS, name


def test_registry_covers_the_program_actions():
    """Later slices call these; none of them may need to edit the registry."""
    needed = {
        "pm.synthesis", "pm.revision", "pm.counterfactual", "pm.intake", "risk.review",
        "review.recheck", "news.memo_fetch", "ops.validate_model_access", "chat.sdk_turn",
        "embed.index", "embed.query", "embed.repair",
    } | {f"debate.{side}_{phase}" for side in ("bull", "bear")
         for phase in ("research", "open", "rebut")}
    assert needed <= set(llm_attribution.ACTIONS)


def test_line_values_are_sanitised_and_never_masked():
    """Every value is a whitelisted scalar under 40 unbroken chars, so the
    40-char secret mask never eats a legitimate field."""
    line = llm_attribution.format_call_line({
        "call": uuid.uuid4().hex, "agent": 'PM "Synthesis"\nx', "action": "pm.synthesis",
        "origin": "worker:regen", "run_id": str(uuid.uuid4()), "job": "regen:25",
        "model_served": "claude-opus-5-5-20261001",
    })
    assert "\n" not in line and '"PM _Synthesis__x"' in line
    assert llm.redact_unbounded(line) == line
