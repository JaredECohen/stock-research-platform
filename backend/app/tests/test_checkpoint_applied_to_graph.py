"""Wave 8A tests — @checkpointed on the graph steps, now via the roster.

Wave 6A shipped the decorator + table; Wave 8A wired it into every major
graph step; RP-003 moved the eight analyst wrappers onto
`roster.AGENTS` (built once at import from each spec's frozen
`checkpoint` name). These tests verify the wiring still skips
re-execution on a retried run with the same `run_id`.

Two modes of evidence:
1. Direct: call the roster's checkpointed runner twice within the same
   `llm_call_context` (run_id) and confirm the underlying specialist runs
   only once. We patch the runner on `roster` — graph.py no longer
   imports it, so a patch there would be a silent no-op (D5).
2. End-to-end: run `run_stock_memo(ticker, run_id=X)` once, then call it
   again with the same `run_id` and verify a checkpointed wrapper hits
   the cache instead of the agent.

The wrappers fall through to a normal call when no run_id is in scope,
so unit tests of specialist agents in isolation are unaffected.
"""
from __future__ import annotations

from unittest.mock import patch

from app.agents import graph as graph_module
from app.agents import roster
from app.agents.llm import llm_call_context
from app.database import SessionLocal
from app.models import MemoRunCheckpoint
from app.schemas import AgentFinding
from app.tests.factories import make_inputs


def _reset_checkpoints() -> None:
    from app.services import checkpoint_store
    with SessionLocal() as db:
        checkpoint_store._ensure_table(db)
        db.query(MemoRunCheckpoint).delete()
        db.commit()


def _stub_finding(name: str = "Test") -> AgentFinding:
    return AgentFinding(agent=name, headline="h", summary="s", confidence=0.6)


def _sector_runner():
    return roster.checkpointed_runner(roster.AGENTS_BY_KEY["sector"])


# ---------------------------------------------------------------------------
# Direct wrapper tests — every specialist stub fires once per run_id
# ---------------------------------------------------------------------------

def test_checkpointed_sector_caches_within_run():
    _reset_checkpoints()
    calls = []

    def stub(profile, ratios, **kw):
        calls.append(1)
        return _stub_finding("Sector Analyst")

    inputs = make_inputs("X")
    with patch.object(roster, "run_sector_agent", side_effect=stub):
        # First call within run_id="r-A" hits the underlying agent.
        with llm_call_context(run_id="r-A"):
            _sector_runner()(inputs)
            _sector_runner()(inputs)
    assert len(calls) == 1, "second call within same run_id should hit cache"


def test_checkpointed_sector_runs_per_distinct_run_id():
    _reset_checkpoints()
    calls = []

    def stub(profile, ratios, **kw):
        calls.append(1)
        return _stub_finding("Sector Analyst")

    inputs = make_inputs("X")
    with patch.object(roster, "run_sector_agent", side_effect=stub):
        with llm_call_context(run_id="r-A"):
            _sector_runner()(inputs)
        with llm_call_context(run_id="r-B"):
            _sector_runner()(inputs)
    assert len(calls) == 2


def test_checkpointed_falls_through_without_run_id():
    """No run_id in scope → no checkpointing → every call hits the agent.

    Important so tests + ad-hoc scripts that call specialists without a
    run_id don't get spurious cache hits."""
    _reset_checkpoints()
    calls = []

    def stub(profile, ratios, **kw):
        calls.append(1)
        return _stub_finding("Sector Analyst")

    inputs = make_inputs("X")
    with patch.object(roster, "run_sector_agent", side_effect=stub):
        _sector_runner()(inputs)
        _sector_runner()(inputs)
    assert len(calls) == 2


def test_checkpointed_valuation_caches_within_run():
    _reset_checkpoints()
    calls = []

    def stub(profile, ratios, dcf, **kw):
        calls.append(1)
        return _stub_finding("Valuation Analyst")

    runner = roster.checkpointed_runner(roster.AGENTS_BY_KEY["valuation"])
    inputs = make_inputs("X")
    with patch.object(roster, "run_valuation_agent", side_effect=stub):
        with llm_call_context(run_id="r-V"):
            runner(inputs)
            runner(inputs)
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


def test_roster_wrappers_use_the_frozen_step_names():
    """A retried run resumes only if the step names match what the
    interrupted run saved, so the roster must checkpoint under exactly the
    names `KNOWN_STEPS` freezes."""
    _reset_checkpoints()
    inputs = make_inputs("X")
    with patch.object(roster, "run_sector_agent",
                      side_effect=lambda *a, **k: _stub_finding("Sector Analyst")):
        with llm_call_context(run_id="r-names"):
            _sector_runner()(inputs)
    with SessionLocal() as db:
        saved = {r.step_name for r in db.query(MemoRunCheckpoint).filter_by(run_id="r-names")}
    assert saved == {"graph.sector_finding"}
    assert "graph.sector_finding" in roster.KNOWN_STEPS


# ---------------------------------------------------------------------------
# End-to-end: full memo run + retry with same run_id
# ---------------------------------------------------------------------------

def test_run_stock_memo_retry_with_same_run_id_hits_checkpoints():
    """First call computes everything; second with same run_id should
    hit the cache for at least the sector finding (and not re-execute
    `run_sector_agent`)."""
    _reset_checkpoints()
    sector_calls = []

    real_sector = roster.run_sector_agent

    def counting_sector(profile, ratios, **kw):
        sector_calls.append(1)
        return real_sector(profile, ratios, **kw)

    with patch.object(roster, "run_sector_agent", side_effect=counting_sector):
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
    real_sector = roster.run_sector_agent

    def counting_sector(profile, ratios, **kw):
        sector_calls.append(1)
        return real_sector(profile, ratios, **kw)

    with patch.object(roster, "run_sector_agent", side_effect=counting_sector):
        graph_module.run_stock_memo("MSFT", run_id="run-1")
        graph_module.run_stock_memo("MSFT", run_id="run-2")
    assert len(sector_calls) == 2


def test_run_stock_memo_saves_only_known_steps():
    """Every checkpoint a memo run writes is a `KNOWN_STEPS` name — the
    set the worker's progress merge and the status endpoint understand."""
    _reset_checkpoints()
    run_id = "known-steps-run"
    graph_module.run_stock_memo("MSFT", run_id=run_id)
    with SessionLocal() as db:
        saved = {r.step_name for r in db.query(MemoRunCheckpoint).filter_by(run_id=run_id)}
    assert saved, "a memo run must checkpoint at least its gather steps"
    assert saved <= set(roster.KNOWN_STEPS), sorted(saved - set(roster.KNOWN_STEPS))
    assert set(roster.GATHER_STEPS) <= saved


# ---------------------------------------------------------------------------
# Long-form default
# ---------------------------------------------------------------------------

def test_long_form_reports_default_on():
    """Wave 8A flips the master-plan-recommended default. Verify the
    settings object reads True out of the box."""
    # Tests use an isolated settings object; we just check the class default.
    from app.config import Settings, settings
    fresh = Settings()
    assert fresh.enable_long_form_reports is True
