"""Wave 8D — token-cost logging end-to-end check.

Wave 1A shipped `LLMCallLog` + the `llm_metrics` aggregations. This
file proves the data stays comprehensive and now carries a USD cost
estimate so future analyses can answer "how much did this run cost"
without external pricing lookups.

Covers:
- `estimate_cost_usd` returns 0 for unknown providers (never raises).
- Tabulated models produce non-zero estimates that scale with tokens.
- `cost_per_run` includes `cost_usd_total` + per-call `cost_usd`.
- `cost_per_agent` and `cost_per_provider` include `cost_usd`.
- Token logging fires for every provider helper that recorded usage:
  inserting a fake row via `_record_usage` shows up in the next query.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.database import SessionLocal
from app.models import LLMCallLog
from app.services import llm_metrics


def _reset_logs() -> None:
    with SessionLocal() as db:
        llm_metrics._ensure_table(db)
        db.query(LLMCallLog).delete()
        db.commit()


def _seed_call(
    *, run_id: str = "test-run", agent_name: str = "Sector Analyst",
    provider: str = "openai", model: str = "gpt-5",
    tokens_in: int = 1000, tokens_out: int = 500,
    duration_ms: int = 850, success: bool = True,
) -> None:
    with SessionLocal() as db:
        llm_metrics._ensure_table(db)
        db.add(LLMCallLog(
            run_id=run_id, agent_name=agent_name, provider=provider,
            model=model, route="strong", tokens_in=tokens_in,
            tokens_out=tokens_out, duration_ms=duration_ms,
            success=success, error="" if success else "test_error",
            generated_at=datetime.utcnow(),
        ))
        db.commit()


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------

def test_estimate_cost_returns_zero_for_unknown_provider():
    assert llm_metrics.estimate_cost_usd("unknown", "unknown", 1000, 500) == 0.0


def test_estimate_cost_for_known_model_scales_with_tokens():
    a = llm_metrics.estimate_cost_usd("openai", "gpt-5", 1_000_000, 0)
    b = llm_metrics.estimate_cost_usd("openai", "gpt-5", 2_000_000, 0)
    assert b > a > 0
    assert abs(b - 2 * a) < 1e-6


def test_estimate_cost_uses_provider_fallback_when_model_unknown():
    """An openai call with an unfamiliar model name should still produce
    a non-zero estimate via the provider fallback price."""
    out = llm_metrics.estimate_cost_usd("openai", "gpt-7-experimental", 1_000_000, 0)
    assert out > 0


def test_estimate_cost_handles_negative_or_none_inputs():
    assert llm_metrics.estimate_cost_usd("openai", "gpt-5", -100, None) == 0.0
    assert llm_metrics.estimate_cost_usd("openai", "gpt-5", None, None) == 0.0


# ---------------------------------------------------------------------------
# Aggregation includes USD figures
# ---------------------------------------------------------------------------

def test_cost_per_run_includes_cost_usd_total():
    _reset_logs()
    _seed_call(run_id="run-A", tokens_in=1_000_000, tokens_out=500_000,
               provider="openai", model="gpt-5")
    out = llm_metrics.cost_per_run("run-A")
    assert out["n_calls"] == 1
    assert out["tokens_total"] == 1_500_000
    # gpt-5: $3.50/MTok input + $14.00/MTok output → 3.50 + 7.00 = 10.50
    assert abs(out["cost_usd_total"] - 10.50) < 1e-3
    assert out["calls"][0]["cost_usd"] > 0


def test_cost_per_agent_includes_cost_usd():
    _reset_logs()
    _seed_call(agent_name="Sector Analyst", run_id="r1",
               tokens_in=500_000, tokens_out=200_000,
               provider="openai", model="gpt-5")
    _seed_call(agent_name="Risk Committee", run_id="r1",
               tokens_in=100_000, tokens_out=50_000,
               provider="anthropic", model="claude-opus-4-7")
    agg = llm_metrics.cost_per_agent()
    assert "Sector Analyst" in agg
    assert "Risk Committee" in agg
    assert agg["Sector Analyst"]["cost_usd"] > 0
    assert agg["Risk Committee"]["cost_usd"] > 0
    # Opus 4.7 is much more expensive per token than gpt-5 → check ordering.
    rate_sector = agg["Sector Analyst"]["cost_usd"] / agg["Sector Analyst"]["tokens_in"]
    rate_critic = agg["Risk Committee"]["cost_usd"] / agg["Risk Committee"]["tokens_in"]
    assert rate_critic > rate_sector


def test_cost_per_provider_aggregates_correctly():
    _reset_logs()
    _seed_call(provider="openai", model="gpt-5",
               tokens_in=100_000, tokens_out=50_000)
    _seed_call(provider="openai", model="gpt-5",
               tokens_in=200_000, tokens_out=100_000)
    _seed_call(provider="anthropic", model="claude-opus-4-7",
               tokens_in=50_000, tokens_out=10_000)
    agg = llm_metrics.cost_per_provider()
    assert agg["openai"]["n_calls"] == 2
    assert agg["openai"]["tokens_in"] == 300_000
    assert agg["openai"]["cost_usd"] > 0
    assert agg["anthropic"]["cost_usd"] > 0


# ---------------------------------------------------------------------------
# End-to-end: every provider call goes through _record_usage
# ---------------------------------------------------------------------------

def test_record_usage_writes_a_row_with_full_context():
    _reset_logs()
    from app.agents.llm import _record_usage, llm_call_context
    with llm_call_context(agent_name="UnitTest Agent", run_id="record-run-1"):
        _record_usage(
            "openai", "gpt-5", input_tokens=123, output_tokens=456,
            duration_ms=42, success=True,
        )
    out = llm_metrics.cost_per_run("record-run-1")
    assert out["n_calls"] == 1
    call = out["calls"][0]
    assert call["agent_name"] == "UnitTest Agent"
    assert call["provider"] == "openai"
    assert call["model"] == "gpt-5"
    assert call["tokens_in"] == 123
    assert call["tokens_out"] == 456
    assert call["duration_ms"] == 42
    assert call["success"] is True
    assert call["cost_usd"] > 0
