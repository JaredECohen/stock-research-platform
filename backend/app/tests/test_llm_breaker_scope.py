"""The breaker endpoints must declare which process they describe.

The LLM circuit breaker lives in module-level dicts in `agents.llm`, so
it is per-process. After the 2026-08-12 worker split, memo regen — and
therefore most LLM calls — runs in `marketmosaic-worker`, while
`/api/admin/llm-breakers` is served by the web service. The endpoint
would otherwise return clean-looking numbers about the wrong process,
which is the same failure mode as `cron-health` reporting
`stale_count: 0` when it could see nothing at all.

This is a labelling fix, not a state fix, and that is deliberate:
  - cross-process *diagnosis* already works, via
    `/api/admin/llm-recent-failures` (reads `LLMCallLog` from the DB) and
    the `provider_failure` row a tripped breaker writes to CacheCostLog;
  - the breaker self-heals after 120s idle, so the 2026-05-31 "stuck
    open until restart" incident cannot recur, and a cross-process reset
    would only ever skip the tail of a cooldown.
Persisting breaker state to the DB would add writes for a condition that
lives ~2 minutes and is already diagnosable. Saying what we can see is
the proportionate fix.
"""
from __future__ import annotations

from app.api.routes_admin import get_llm_breakers, reset_llm_breakers

_PROVIDERS = {"openai", "anthropic", "gemini"}


def test_breaker_state_declares_the_reporting_process():
    out = get_llm_breakers()
    assert out["reported_by"] in ("web", "worker")
    assert "marketmosaic-worker" in out["scope_note"], (
        "the scope note must name the process whose breakers are NOT visible"
    )
    assert "llm-recent-failures" in out["scope_note"], (
        "the scope note must point at the cross-process alternative"
    )


def test_breaker_state_still_reports_every_provider():
    """The labelling must not cost the actual payload."""
    providers = get_llm_breakers()["providers"]
    assert set(providers) == _PROVIDERS
    for name, state in providers.items():
        assert set(state) >= {
            "failure_count", "is_open", "cooldown_seconds",
        }, f"{name} lost fields"


def test_reset_reports_scope_rather_than_implying_a_fleet_wide_reset():
    """A bare `{"reset": "all"}` would imply it cleared the worker too."""
    out = reset_llm_breakers(None)
    assert out["reset"] == "all"
    assert out["reported_by"] in ("web", "worker")
    assert "this process only" in out["scope_note"].lower()
    assert set(out["providers"]) == _PROVIDERS


def test_reset_actually_resets_this_process():
    """The caveat must not be mistaken for the reset having become a no-op."""
    from app.agents import llm

    llm._FAILURE_COUNTERS["openai"] = llm._BREAKER_THRESHOLD
    try:
        assert get_llm_breakers()["providers"]["openai"]["failure_count"] > 0
        reset_llm_breakers("openai")
        assert get_llm_breakers()["providers"]["openai"]["failure_count"] == 0
    finally:
        llm._FAILURE_COUNTERS["openai"] = 0
        llm._FAILURE_LAST_AT.pop("openai", None)
