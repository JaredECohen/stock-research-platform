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
