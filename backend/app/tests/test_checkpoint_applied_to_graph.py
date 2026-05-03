"""Wave 8A tests — apply @checkpointed to graph steps.

Wave 6A shipped the decorator + table; this PR wires it into every major
graph step. These tests verify that the wiring actually skips re-execution
on a retried run with the same `run_id`.

Two modes of evidence:
1. Direct: call `_checkpointed_*` twice within the same `llm_call_context`
   (run_id) and confirm the underlying specialist runs only once. We patch
   the specialist function to count invocations.
2. End-to-end: run `run_stock_memo(ticker, run_id=X)` once, then call it
   again with the same `run_id` and verify a checkpointed wrapper hits
   the cache instead of the agent.

The wrappers fall through to a normal call when no run_id is in scope,
so unit tests of specialist agents in isolation are unaffected.
"""
from __future__ import annotations

from unittest.mock import patch

from app.agents import graph as graph_module
from app.agents.llm import llm_call_context
from app.database import SessionLocal
from app.models import MemoRunCheckpoint
from app.schemas import AgentFinding


def _reset_checkpoints() -> None:
    from app.services import checkpoint_store
    with SessionLocal() as db:
        checkpoint_store._ensure_table(db)
        db.query(MemoRunCheckpoint).delete()
        db.commit()


def _stub_finding(name: str = "Test") -> AgentFinding:
    return AgentFinding(agent=name, headline="h", summary="s", confidence=0.6)


# ---------------------------------------------------------------------------
# Direct wrapper tests — every specialist stub fires once per run_id
# ---------------------------------------------------------------------------

def test_checkpointed_sector_caches_within_run():
    _reset_checkpoints()
    calls = []

    def stub(profile, ratios):
        calls.append(1)
        return _stub_finding("Sector Analyst")

    with patch.object(graph_module, "run_sector_agent", side_effect=stub):
        # First call within run_id="r-A" hits the underlying agent.
        with llm_call_context(run_id="r-A"):
            graph_module._checkpointed_sector({"ticker": "X"}, {})
            graph_module._checkpointed_sector({"ticker": "X"}, {})
    assert len(calls) == 1, "second call within same run_id should hit cache"


def test_checkpointed_sector_runs_per_distinct_run_id():
    _reset_checkpoints()
    calls = []

    def stub(profile, ratios):
        calls.append(1)
        return _stub_finding("Sector Analyst")

    with patch.object(graph_module, "run_sector_agent", side_effect=stub):
        with llm_call_context(run_id="r-A"):
            graph_module._checkpointed_sector({"ticker": "X"}, {})
        with llm_call_context(run_id="r-B"):
            graph_module._checkpointed_sector({"ticker": "X"}, {})
    assert len(calls) == 2


def test_checkpointed_falls_through_without_run_id():
    """No run_id in scope → no checkpointing → every call hits the agent.

    Important so tests + ad-hoc scripts that call specialists without a
    run_id don't get spurious cache hits."""
    _reset_checkpoints()
    calls = []

    def stub(profile, ratios):
        calls.append(1)
        return _stub_finding("Sector Analyst")

    with patch.object(graph_module, "run_sector_agent", side_effect=stub):
        graph_module._checkpointed_sector({"ticker": "X"}, {})
        graph_module._checkpointed_sector({"ticker": "X"}, {})
    assert len(calls) == 2


def test_checkpointed_valuation_caches_within_run():
    _reset_checkpoints()
    calls = []

    def stub(profile, ratios, dcf):
        calls.append(1)
        return _stub_finding("Valuation Analyst")

    with patch.object(graph_module, "run_valuation_agent", side_effect=stub):
        with llm_call_context(run_id="r-V"):
            graph_module._checkpointed_valuation({"ticker": "X"}, {}, None)
            graph_module._checkpointed_valuation({"ticker": "X"}, {}, None)
    assert len(calls) == 1


def test_checkpointed_critic_caches_within_run():
    _reset_checkpoints()
    calls = []
    from app.schemas import CriticReview

    def stub(memo_dict):
        calls.append(1)
        return CriticReview(overall_assessment="ok")

    with patch.object(graph_module, "run_critic", side_effect=stub):
        with llm_call_context(run_id="r-C"):
            graph_module._checkpointed_critic({"x": 1})
            graph_module._checkpointed_critic({"x": 1})
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# End-to-end: full memo run + retry with same run_id
# ---------------------------------------------------------------------------

def test_run_stock_memo_retry_with_same_run_id_hits_checkpoints():
    """First call computes everything; second with same run_id should
    hit the cache for at least the sector finding (and not re-execute
    `run_sector_agent`)."""
    _reset_checkpoints()
    sector_calls = []

    real_sector = graph_module.run_sector_agent

    def counting_sector(profile, ratios):
        sector_calls.append(1)
        return real_sector(profile, ratios)

    with patch.object(graph_module, "run_sector_agent", side_effect=counting_sector):
        run_id = "deterministic-run-id-for-test"
        memo1 = graph_module.run_stock_memo("MSFT", run_id=run_id)
        memo2 = graph_module.run_stock_memo("MSFT", run_id=run_id)
    assert memo1 and memo2
    assert len(sector_calls) == 1, (
        f"expected sector agent to fire once across two same-run-id calls, "
        f"got {len(sector_calls)}"
    )


def test_run_stock_memo_distinct_run_ids_re_execute_specialists():
    """Two separate runs (different run_ids) both fire the underlying
    specialists. Sanity check: caching isn't leaking across runs."""
    _reset_checkpoints()
    sector_calls = []
    real_sector = graph_module.run_sector_agent

    def counting_sector(profile, ratios):
        sector_calls.append(1)
        return real_sector(profile, ratios)

    with patch.object(graph_module, "run_sector_agent", side_effect=counting_sector):
        graph_module.run_stock_memo("MSFT", run_id="run-1")
        graph_module.run_stock_memo("MSFT", run_id="run-2")
    assert len(sector_calls) == 2


# ---------------------------------------------------------------------------
# Long-form default
# ---------------------------------------------------------------------------

def test_long_form_reports_default_on():
    """Wave 8A flips the master-plan-recommended default. Verify the
    settings object reads True out of the box."""
    from app.config import settings
    # Tests use an isolated settings object; we just check the class default.
    from app.config import Settings
    fresh = Settings()
    assert fresh.enable_long_form_reports is True
