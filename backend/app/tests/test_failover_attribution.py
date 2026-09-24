"""A failover lands on the memo whose LLM calls caused it — and only that one.

`llm` records failover events in a context-local list that `run_stock_memo`
drains into its DegradationLog. Two things make the drain point matter:
PM synthesis, the critic, reflection and the long-form/DCF enrichment all
call the LLM *after* the specialist round, and the regen worker runs memos
back to back in one long-lived thread without `copy_context`, so an event
that is not drained by the end of a run is still there when the next one
starts. Runs deterministically (blank keys) — the events are injected.
"""
from __future__ import annotations

import pytest

from app.agents import graph, llm
from app.agents.safe_runner import DegradationLog
from app.config import settings


@pytest.fixture(autouse=True)
def _clean():
    llm.reset_failover_state()
    yield
    llm.reset_failover_state()


def test_run_stock_memo_discards_stale_events_and_keeps_late_ones(monkeypatch):
    # Not an assert: a failing one renders its operand, the Settings object.
    if settings.has_llm:
        pytest.fail("must stay zero-cost: no provider is ever called", pytrace=False)

    logs: list = []

    class _SpyLog(DegradationLog):
        def __init__(self):
            super().__init__()
            logs.append(self)

    monkeypatch.setattr(graph, "DegradationLog", _SpyLog)

    # Left behind by the previous memo in this thread.
    llm._record_failover("openai", "anthropic", "stale_previous_run")

    # Reflection is the last LLM-touching stage of the run — well after
    # the post-specialist drain point.
    def _reflection_that_failed_over(memo):
        llm._record_failover("openai", "anthropic", "call_failed")
        return ([], [])

    monkeypatch.setattr(graph, "_run_reflection_step", _reflection_that_failed_over)

    memo = graph.run_stock_memo("NVDA")

    assert memo.degraded_agents.count("LLM provider") == 1
    assert len(logs) == 1
    messages = [f["message"] for f in logs[0].failures if f["agent"] == "LLM provider"]
    assert messages == ["failed over from openai to anthropic: call_failed"], (
        "the stale event must not be pinned on this memo; the late one must be"
    )
    assert logs[0].failures[[f["agent"] for f in logs[0].failures].index("LLM provider")]["error_type"] == "ProviderFailover"
    assert llm.consume_failover_events() == [], "nothing may leak into the next run"


def test_run_stock_memo_leaves_no_events_for_the_next_run(monkeypatch):
    """Even a failover in the specialist round is fully drained by the end."""
    def _specialist_failover(*a, **kw):
        llm._record_failover("anthropic", "openai", "breaker_open")
        return None

    monkeypatch.setattr(graph, "latest_transcript", _specialist_failover)
    memo = graph.run_stock_memo("NVDA")
    assert "LLM provider" in memo.degraded_agents
    assert llm.consume_failover_events() == []
