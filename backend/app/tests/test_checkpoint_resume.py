"""Wave 6A tests — checkpoint resume.

Covers:
- `save_step` / `load_step` round-trip with JSON-friendly payloads.
- TTL: rows past their `expires_at` are not returned and are removed by `gc_expired`.
- `@checkpointed` returns the cached payload on second invocation with
  the same `run_id`, skipping the wrapped function entirely.
- `@checkpointed` falls through (no caching) when no `run_id` is in
  scope — preserves existing behavior for callers that don't set one.
- Decorator handles Pydantic models via `model_dump`/`model_validate`
  round-trip when `return_type` is supplied.
- Non-JSON-serializable returns silently skip caching but still execute
  normally.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.agents.llm import llm_call_context
from app.database import SessionLocal
from app.models import MemoRunCheckpoint
from app.schemas import AgentFinding
from app.services import checkpoint_store


def _reset_table() -> None:
    with SessionLocal() as db:
        checkpoint_store._ensure_table(db)
        db.query(MemoRunCheckpoint).delete()
        db.commit()


# ---------------------------------------------------------------------------
# Store API
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip():
    _reset_table()
    ok = checkpoint_store.save_step(
        "run-1", "fundamentals", payload={"x": 1, "y": [2, 3]},
    )
    assert ok is True
    out = checkpoint_store.load_step("run-1", "fundamentals")
    assert out == {"x": 1, "y": [2, 3]}


def test_save_step_overwrites_existing():
    _reset_table()
    checkpoint_store.save_step("run-2", "step", payload={"v": 1})
    checkpoint_store.save_step("run-2", "step", payload={"v": 2})
    out = checkpoint_store.load_step("run-2", "step")
    assert out == {"v": 2}


def test_load_returns_none_for_unknown_keys():
    _reset_table()
    assert checkpoint_store.load_step("never-saved", "x") is None


def test_expired_rows_are_not_returned():
    _reset_table()
    # Save with TTL=0 so it's expired immediately.
    checkpoint_store.save_step("run-exp", "step", payload={"v": 1}, ttl_hours=0)
    # Manually backdate the row to make sure expires_at < now.
    with SessionLocal() as db:
        row = db.query(MemoRunCheckpoint).filter_by(run_id="run-exp").first()
        row.expires_at = datetime.utcnow() - timedelta(seconds=1)
        db.commit()
    assert checkpoint_store.load_step("run-exp", "step") is None


def test_gc_expired_removes_only_expired_rows():
    _reset_table()
    checkpoint_store.save_step("fresh", "x", payload={"v": 1}, ttl_hours=24)
    checkpoint_store.save_step("stale", "x", payload={"v": 1})
    with SessionLocal() as db:
        stale = db.query(MemoRunCheckpoint).filter_by(run_id="stale").first()
        stale.expires_at = datetime.utcnow() - timedelta(seconds=1)
        db.commit()
    n = checkpoint_store.gc_expired()
    assert n == 1
    assert checkpoint_store.load_step("fresh", "x") == {"v": 1}
    assert checkpoint_store.load_step("stale", "x") is None


def test_save_step_skips_unserializable_payload(tmp_path):
    _reset_table()

    class NotJSON:
        def __repr__(self):
            return "<custom>"

    # Custom type with no model_dump and not JSON-serializable beyond `default=str`.
    # `default=str` is permissive — most things will round-trip via __str__.
    # Use something genuinely unserializable: a set inside a tuple isn't supported by the JSON column path even via default=str.
    # Instead, force the failure by patching json.dumps to raise.
    from unittest.mock import patch
    with patch("app.services.checkpoint_store.json.dumps", side_effect=TypeError("nope")):
        ok = checkpoint_store.save_step("run-bad", "step", payload={"a": 1})
    assert ok is False


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def test_checkpointed_caches_dict_result_within_run_id():
    _reset_table()
    calls = []

    @checkpoint_store.checkpointed("step_a")
    def expensive() -> dict:
        calls.append(1)
        return {"value": 42}

    with llm_call_context(run_id="run-cached"):
        first = expensive()
        second = expensive()
    assert first == {"value": 42}
    assert second == {"value": 42}
    # Wrapped function only ran once.
    assert len(calls) == 1


def test_checkpointed_runs_normally_without_run_id():
    _reset_table()
    calls = []

    @checkpoint_store.checkpointed("step_b")
    def expensive() -> dict:
        calls.append(1)
        return {"value": 99}

    # No llm_call_context → no run_id → decorator falls through.
    assert expensive() == {"value": 99}
    assert expensive() == {"value": 99}
    assert len(calls) == 2


def test_checkpointed_independent_per_run_id():
    _reset_table()
    calls = []

    @checkpoint_store.checkpointed("step_c")
    def expensive() -> dict:
        calls.append(1)
        return {"runs": len(calls)}

    with llm_call_context(run_id="A"):
        a1 = expensive()
    with llm_call_context(run_id="B"):
        b1 = expensive()
    with llm_call_context(run_id="A"):
        a2 = expensive()  # Should hit A's cache, not re-run.

    assert a1["runs"] == 1
    assert b1["runs"] == 2  # B fired the function a 2nd time.
    assert a2 == a1  # cached.
    assert len(calls) == 2  # only two underlying calls.


def test_checkpointed_pydantic_round_trip_via_return_type():
    _reset_table()
    calls = []

    @checkpoint_store.checkpointed("step_pyd", return_type=AgentFinding)
    def make_finding() -> AgentFinding:
        calls.append(1)
        return AgentFinding(
            agent="t", headline="h", summary="s", confidence=0.9,
        )

    with llm_call_context(run_id="run-pyd"):
        first = make_finding()
        second = make_finding()
    assert isinstance(first, AgentFinding)
    assert isinstance(second, AgentFinding)
    assert first.headline == second.headline
    assert len(calls) == 1


def test_checkpointed_falls_through_when_save_fails(monkeypatch):
    """If the store can't serialize, the function still returns its
    result; the next call re-runs (no false cache hit)."""
    _reset_table()
    calls = []

    @checkpoint_store.checkpointed("step_unsafe")
    def returns_strange() -> dict:
        calls.append(1)
        return {"v": 1}

    monkeypatch.setattr(
        checkpoint_store, "save_step", lambda *a, **kw: False,
    )
    with llm_call_context(run_id="run-unsafe"):
        first = returns_strange()
        second = returns_strange()
    assert first == {"v": 1}
    assert second == {"v": 1}
    assert len(calls) == 2  # save kept failing → re-runs.


# ---------------------------------------------------------------------------
# W2b 7(a) — source-ledger facts survive a resume (opt-in, roster steps only)
# ---------------------------------------------------------------------------

def _analyst_step(calls: list[int]):
    from app.agents.source_ledger import register_source

    @checkpoint_store.checkpointed("graph.test_finding", return_type=AgentFinding, capture_sources=True)
    def run() -> AgentFinding:
        calls.append(1)
        # What an analyst does inside its step: register its prompt payload.
        register_source("technical", "technical:T", {"rsi_14": 68.2, "sma_50": 109.79})
        return AgentFinding(agent="Technical Analyst", headline="h", summary="RSI at 68.", confidence=0.6)

    return run


def test_checkpoint_capture_and_replay():
    """A resumed run loads the stored finding instead of re-running the
    analyst, so the facts it registered inside the step are replayed from
    the row; without them the registry would be silently incomplete."""
    from app.agents.source_ledger import SourceLedger
    _reset_table()
    calls: list[int] = []
    step = _analyst_step(calls)
    first = SourceLedger()
    with first.activate(), llm_call_context(run_id="run-src"):
        step()
    stored = checkpoint_store.load_step_sources("run-src", "graph.test_finding")
    assert stored is not None and stored["v"] == 1 and len(stored["facts"]) == 2

    resumed = SourceLedger()
    with resumed.activate(), llm_call_context(run_id="run-src"):
        step()
    assert calls == [1]                                   # the analyst did not re-run ...
    snap = resumed.snapshot()
    assert snap.complete                                  # ... and nothing is missing
    assert {round(f.value, 2) for f in snap.facts} == {68.2, 109.79}
    assert snap.resolves("technical:T")


def test_checkpoint_row_without_sources_marks_registry_incomplete():
    """A row written before the `sources` column (or by a failed capture):
    the number check must report "not checked", not flag real figures."""
    from app.agents.source_ledger import SourceLedger
    _reset_table()
    checkpoint_store.save_step("run-old", "graph.test_finding", payload=AgentFinding(
        agent="Technical Analyst", headline="h", summary="s", confidence=0.6).model_dump(mode="json"))
    assert checkpoint_store.load_step_sources("run-old", "graph.test_finding") is None
    calls: list[int] = []
    ledger = SourceLedger()
    with ledger.activate(), llm_call_context(run_id="run-old"):
        finding = _analyst_step(calls)()
    assert calls == [] and finding.headline == "h"
    assert ledger.snapshot().incomplete_steps == ("graph.test_finding",)


def test_capture_is_opt_in():
    """Non-roster steps never write `sources` (and never mark anything)."""
    from app.agents.source_ledger import SourceLedger, register_source
    _reset_table()

    @checkpoint_store.checkpointed("graph.plain")
    def plain() -> dict:
        register_source("technical", "technical:T", {"rsi_14": 50.0})
        return {"v": 1}

    ledger = SourceLedger()
    with ledger.activate(), llm_call_context(run_id="run-plain"):
        plain()
        plain()
    assert checkpoint_store.load_step_sources("run-plain", "graph.plain") is None
    assert ledger.snapshot().complete


def test_roster_steps_capture_sources():
    """Every roster step is built by `_build_checkpointed`, and a step it
    builds persists the facts its analyst registered (behaviour, not the
    source text: a roster spec with a fake runner, run under a ledger and a
    run_id)."""
    import dataclasses

    from app.agents import roster
    from app.agents.source_ledger import SourceLedger, register_source
    _reset_table()
    assert all(roster.checkpointed_runner(spec) is roster._CHECKPOINTED[spec] for spec in roster.AGENTS)

    calls: list[int] = []

    def run(inputs, critique):
        calls.append(1)
        register_source("technical", "technical:T", {"rsi_14": 61.7})
        return AgentFinding(agent="Technical Analyst", headline="h", summary="s", confidence=0.6)

    spec = dataclasses.replace(roster.AGENTS[0], run=run)
    runner = roster._build_checkpointed(spec)
    ledger = SourceLedger()
    with ledger.activate(), llm_call_context(run_id="run-roster"):
        runner(None)  # type: ignore[arg-type]
    stored = checkpoint_store.load_step_sources("run-roster", spec.checkpoint)
    assert stored is not None and len(stored["facts"]) == 1

    # ... and replays them on resume without re-running the analyst.
    resumed = SourceLedger()
    with resumed.activate(), llm_call_context(run_id="run-roster"):
        runner(None)  # type: ignore[arg-type]
    assert calls == [1]
    assert {round(f.value, 2) for f in resumed.snapshot().facts} == {61.7}
