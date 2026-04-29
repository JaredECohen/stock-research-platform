"""Health and provider-status endpoints."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from fastapi import APIRouter

from ..config import settings
from ..services.data_service import get_data_service

router = APIRouter()


def _llm_status() -> Dict:
    return {
        "configured": settings.has_llm,
        "provider_choice": settings.llm_provider,
        "active_provider": settings.active_llm_provider,
        "openai_configured": settings.has_openai,
        "anthropic_configured": settings.has_anthropic,
        "openai_strong_model": settings.openai_strong_model,
        "openai_cheap_model": settings.openai_cheap_model,
        "anthropic_strong_model": settings.anthropic_strong_model,
        "anthropic_cheap_model": settings.anthropic_cheap_model,
    }


@router.get("/health")
def health() -> Dict:
    return {
        "status": "ok",
        "app_env": settings.app_env,
        "mode": get_data_service().mode(),
        "llm_configured": settings.has_llm,
        "llm_provider": settings.active_llm_provider,
    }


@router.get("/api/providers/status")
def providers_status() -> Dict:
    ds = get_data_service()
    statuses = {name: asdict(s) for name, s in ds.status().items()}
    missing_keys = [
        name for name, s in statuses.items()
        if not s["configured"] and name != "demo" and name != "sec_edgar"
    ]
    return {
        "mode": ds.mode(),
        "providers": statuses,
        "missing_api_keys": missing_keys,
        "llm_configured": settings.has_llm,
        "llm": _llm_status(),
        "feature_flags": {
            "use_demo_data": settings.use_demo_data,
            "enable_live_data": settings.enable_live_data,
            "enable_agent_critic": settings.enable_agent_critic,
            "enable_vector_search": settings.enable_vector_search,
        },
    }
