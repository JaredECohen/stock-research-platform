"""Health and provider-status endpoints."""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from ..agents import llm
from ..config import settings
from ..services import provider_cache
from ..services.data_service import get_data_service

router = APIRouter()


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _ago(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 90:
        return f"{s} seconds ago"
    if s < 90 * 60:
        return f"{max(1, round(s / 60))} minutes ago"
    return f"{max(1, round(s / 3600))} hours ago"


def _llm_degradation(
    mode: str, active: str, breakers: dict[str, dict[str, Any]], failover: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Why the LLM layer is not running as configured, in plain sentences.

    Three conditions count: nothing configured while data is live (memos
    are shipping deterministic stubs labelled as research), the active
    provider's breaker is open (every call is short-circuiting to None),
    or a failover inside the cooldown window (memos are coming from the
    backup vendor). Anything the ops page can act on has to be readable
    without the breaker internals, hence sentences rather than codes.
    """
    reasons: list[str] = []
    if not settings.has_llm and mode == "live":
        reasons.append("No LLM provider configured")
    if active in breakers and breakers[active]["is_open"]:
        n = breakers[active]["failure_count"]
        reasons.append(
            f"{active.capitalize()} circuit breaker is open after {n} consecutive failures"
        )
    last_at = failover.get("last_at")
    if last_at is not None:
        since = time.time() - float(last_at)
        if since < settings.llm_failover_cooldown_seconds:
            reasons.append(
                f"Failed over from {failover['last_from']} to {failover['last_to']} "
                f"{_ago(since)}"
            )
    return bool(reasons), reasons


def _llm_status(mode: str) -> dict[str, Any]:
    """The `llm` block of `/api/providers/status`.

    Shape is a contract with the frontend status page — keep every key,
    add rather than rename. Model names and booleans only; never key
    material (the key set is asserted secret-free in
    `test_providers_status_contract`).
    """
    # Per-process, like the admin breaker endpoint: this describes the
    # web service. The worker's breakers are only visible via the DB.
    # The contract's `breakers` is one row per provider; failover has its
    # own top-level object below.
    breakers = llm.get_breaker_state(include_failover=False)
    failover = llm.get_failover_state()
    summary = llm.model_summary()
    active = settings.active_llm_provider
    degraded, reasons = _llm_degradation(mode, active, breakers, failover)
    return {
        "configured": settings.has_llm,
        "provider_choice": settings.llm_provider,
        "active_provider": active,
        "openai_configured": settings.has_openai,
        "anthropic_configured": settings.has_anthropic,
        "gemini_configured": settings.has_gemini,
        "openai_strong_model": settings.openai_strong_model,
        "openai_cheap_model": settings.openai_cheap_model,
        "anthropic_strong_model": settings.anthropic_strong_model,
        "anthropic_cheap_model": settings.anthropic_cheap_model,
        "role_models": summary["role_models"],
        "breakers": breakers,
        "failover": {
            "enabled": settings.llm_failover_enabled,
            "count": failover["count"],
            "last_from": failover["last_from"],
            "last_to": failover["last_to"],
            "last_at": _iso(failover["last_at"]),
            "last_reason": failover["last_reason"],
        },
        "degraded": degraded,
        "degradation_reasons": reasons,
    }


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app_env": settings.app_env,
        "mode": get_data_service().mode(),
        "llm_configured": settings.has_llm,
        "llm_provider": settings.active_llm_provider,
        # The deploy-visible canary: Render's RENDER_GIT_COMMIT, null outside Render.
        "build": {"git_commit": settings.render_git_commit or None},
        # FEAT-003: the routing value THIS process loaded, not what
        # render.yaml says — a Blueprint sync may not apply a newly added key,
        # and routing would then stay off silently. A boolean only. The worker
        # (no HTTP port) reports its own in the `worker_heartbeat` cron-health
        # note as `industry_routing=on|off`.
        "industry_analyst_routing": bool(settings.enable_industry_analyst_routing),
        # The 2026-09-25 program's wave switches, as THIS process loaded
        # them, so a post-deploy check can confirm waves G-I without the
        # admin token. Mode names and a model name only — never a key.
        # `llm_research_model` is null while the research tier is blank
        # (legacy routing); `attribution_mode` is the EFFECTIVE mode
        # (production always runs "warn").
        "debate_mode": settings.debate_mode,
        "reviewer_mode": settings.reviewer_mode,
        "llm_research_model": settings.llm_research_model or None,
        "chat_sdk": bool(settings.chat_agents_sdk),
        "attribution_mode": llm.attribution_mode(),
    }


# `enable_vector_search=false` was read in a production audit as "semantic
# retrieval disabled". It is not consulted anywhere (see config.py); the
# note travels with the flag so neither the Settings page nor an audit of
# this payload can draw that conclusion again.
FEATURE_FLAG_NOTES: dict[str, str] = {
    "enable_vector_search": (
        "Not consulted by retrieval: the filing and earnings analysts search "
        "the vector index first whenever the ticker is known; the filing "
        "analyst falls back to BM25 keyword search when that search returns "
        "nothing, fails, or is skipped for lack of a ticker."
    ),
}


@router.get("/api/providers/status")
def providers_status() -> dict:
    ds = get_data_service()
    statuses = {name: asdict(s) for name, s in ds.status().items()}
    missing_keys = [
        name for name, s in statuses.items()
        if not s["configured"] and name != "demo" and name != "sec_edgar"
    ]
    mode = ds.mode()
    return {
        "mode": mode,
        "providers": statuses,
        "missing_api_keys": missing_keys,
        "llm_configured": settings.has_llm,
        "llm": _llm_status(mode),
        # Stale-serve ledger from provider_cache: how often the last 24h fell
        # back to an expired row (and how old), or refused one as too stale.
        # DB-backed, so web and worker report the same picture. Never raises.
        "stale_cache": provider_cache.stale_stats(window_hours=24),
        "feature_flags": {
            "use_demo_data": settings.use_demo_data,
            "enable_live_data": settings.enable_live_data,
            "enable_agent_critic": settings.enable_agent_critic,
            "enable_vector_search": settings.enable_vector_search,
        },
        # Additive: what a reported flag actually does, where its name
        # misleads. Keys above stay booleans (the Settings page renders them).
        "feature_flag_notes": dict(FEATURE_FLAG_NOTES),
    }
