"""LLM call trace logging (Wave 1A) — DB-only audit trail.

Verifies:
- Every successful provider call writes a row.
- Failed calls write a row with success=False + error.
- run_id + agent_name from llm_call_context propagates to the row.
- cost_per_run / cost_per_agent / cost_per_provider aggregations work.
- gc_old deletes rows older than threshold.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.agents import llm as llm_mod
from app.agents.llm import llm_call_context, _record_usage
from app.config import settings
from app.database import SessionLocal
from app.models import LLMCallLog
from app.services import llm_metrics
from sqlalchemy import select


def _count_rows() -> int:
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        return len(list(db.execute(select(LLMCallLog)).scalars().all()))


def _all_rows():
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        return list(db.execute(select(LLMCallLog)).scalars().all())


def setup_function(_fn):
    """Wipe the LLMCallLog between tests for clean assertions."""
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        db.query(LLMCallLog).delete()
        db.commit()
    llm_mod.reset_circuit_breaker()


# ---------------------------------------------------------------------------
# Direct _record_usage writes
# ---------------------------------------------------------------------------

def test_record_usage_writes_a_row_with_default_context():
    _record_usage("openai", "gpt-5.5", 100, 50, duration_ms=200, success=True)
    rows = _all_rows()
    assert len(rows) == 1
    r = rows[0]
    assert r.provider == "openai"
    assert r.model == "gpt-5.5"
    assert r.tokens_in == 100
    assert r.tokens_out == 50
    assert r.duration_ms == 200
    assert r.success is True
    assert r.agent_name == "unknown"      # default context
    assert r.run_id is None


def test_llm_call_context_tags_subsequent_calls():
    with llm_call_context(agent_name="Sector Analyst", run_id="abc-123", route="cheap"):
        _record_usage("openai", "gpt-5.4", 50, 25)
    rows = _all_rows()
    assert len(rows) == 1
    r = rows[0]
    assert r.agent_name == "Sector Analyst"
    assert r.run_id == "abc-123"
    assert r.route == "cheap"


def test_failed_call_records_success_false_and_error():
    _record_usage("anthropic", "claude-opus-4-7", 0, 0,
                  duration_ms=1500, success=False, error="rate_limit_exceeded")
    rows = _all_rows()
    assert len(rows) == 1
    assert rows[0].success is False
    assert "rate_limit" in rows[0].error


def test_context_unset_after_with_block():
    with llm_call_context(agent_name="PM", run_id="r1"):
        _record_usage("openai", "gpt-5.5", 10, 5)
    # Outside the with: defaults
    _record_usage("openai", "gpt-5.5", 10, 5)
    rows = sorted(_all_rows(), key=lambda r: r.id)
    assert rows[0].agent_name == "PM"
    assert rows[0].run_id == "r1"
    assert rows[1].agent_name == "unknown"
    assert rows[1].run_id is None


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def test_cost_per_run_aggregates_by_run_id():
    with llm_call_context(agent_name="A", run_id="run-x"):
        _record_usage("openai", "gpt-5.4", 100, 50, duration_ms=100, success=True)
    with llm_call_context(agent_name="B", run_id="run-x"):
        _record_usage("openai", "gpt-5.4", 200, 100, duration_ms=200, success=True)
    with llm_call_context(agent_name="C", run_id="run-y"):
        _record_usage("openai", "gpt-5.4", 50, 25, duration_ms=50, success=False, error="x")

    info = llm_metrics.cost_per_run("run-x")
    assert info["n_calls"] == 2
    assert info["tokens_in"] == 300
    assert info["tokens_out"] == 150
    assert info["tokens_total"] == 450
    assert info["duration_ms_total"] == 300
    assert info["n_failures"] == 0


def test_cost_per_agent_aggregates_correctly():
    with llm_call_context(agent_name="Sector", run_id="r"):
        _record_usage("openai", "gpt-5.4", 100, 50, duration_ms=100)
    with llm_call_context(agent_name="Sector", run_id="r"):
        _record_usage("openai", "gpt-5.4", 100, 50, duration_ms=100)
    with llm_call_context(agent_name="Critic", run_id="r"):
        _record_usage("anthropic", "claude-opus-4-7", 200, 100, duration_ms=300)

    agg = llm_metrics.cost_per_agent()
    assert agg["Sector"]["n_calls"] == 2
    assert agg["Sector"]["tokens_in"] == 200
    assert agg["Sector"]["tokens_out"] == 100
    assert agg["Critic"]["n_calls"] == 1
    assert agg["Critic"]["tokens_in"] == 200


def test_cost_per_provider_aggregates_correctly():
    _record_usage("openai", "gpt-5.4", 100, 50)
    _record_usage("openai", "gpt-5.5", 200, 100)
    _record_usage("anthropic", "claude-opus-4-7", 50, 25)
    _record_usage("gemini", "gemini-2.5-pro", 0, 0, success=False, error="x")

    agg = llm_metrics.cost_per_provider()
    assert agg["openai"]["n_calls"] == 2
    assert agg["anthropic"]["n_calls"] == 1
    assert agg["gemini"]["n_failures"] == 1


def test_slowest_calls_returns_top_n_in_order():
    durations = [50, 200, 1500, 500, 75]
    for d in durations:
        _record_usage("openai", "gpt-5.4", 10, 5, duration_ms=d)
    top3 = llm_metrics.slowest_calls(n=3)
    got = [c["duration_ms"] for c in top3]
    assert got == sorted(durations, reverse=True)[:3]


def test_gc_deletes_rows_older_than_window():
    _record_usage("openai", "gpt-5.4", 10, 5)  # fresh row
    # Forge an old row by manipulating generated_at.
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        old = LLMCallLog(
            agent_name="old", provider="openai", model="gpt-5.4",
            tokens_in=1, tokens_out=1, duration_ms=10, success=True,
            generated_at=datetime.utcnow() - timedelta(days=120),
        )
        db.add(old)
        db.commit()

    deleted = llm_metrics.gc_old(max_age_days=90)
    assert deleted == 1
    # Fresh row survives
    remaining = _all_rows()
    assert len(remaining) == 1
    assert remaining[0].agent_name == "unknown"


# ---------------------------------------------------------------------------
# Provider call paths actually log rows
# ---------------------------------------------------------------------------

def test_openai_call_path_logs_a_row(monkeypatch):
    """Mock the OpenAI client to simulate a successful response and assert
    a row is persisted with token attribution."""
    monkeypatch.setattr(settings, "openai_api_key", "stub")

    class _Usage:
        prompt_tokens = 42
        completion_tokens = 17

    class _Choice:
        class message:
            content = '{"ok": true}'

    class _Resp:
        usage = _Usage()
        choices = [_Choice()]

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**_kw):
                    return _Resp()

    with patch.object(llm_mod, "_openai_client", return_value=_Client()), \
         patch.object(llm_mod, "_breaker_open", return_value=False):
        with llm_call_context(agent_name="Test Agent", run_id="t-1"):
            out = llm_mod.chat_json("hi", provider_override="openai", model="gpt-5.4")

    assert out == {"ok": True}
    rows = _all_rows()
    assert len(rows) == 1
    r = rows[0]
    assert r.agent_name == "Test Agent"
    assert r.run_id == "t-1"
    assert r.provider == "openai"
    assert r.model == "gpt-5.4"
    assert r.tokens_in == 42
    assert r.tokens_out == 17
    assert r.success is True
