"""Application configuration loaded from environment variables.

A single Settings object is constructed at import time. Modules that need
runtime config import `settings` from here. All values are safe defaults so
the application boots even with a completely empty environment.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_env_files() -> list[str]:
    """Resolve env files in load order.

    Order matters — pydantic-settings applies LATER files on top of earlier
    ones. The intended precedence is:
        1. config.env   ← committed defaults (model assignments, feature
                          flags, runtime tuning). In git.
        2. .env         ← gitignored secrets + per-deployment overrides
                          (API keys, DATABASE_URL, …).
        3. process env  ← OS environment wins over both (handled by
                          pydantic-settings automatically).

    Both files are searched at the repo root and inside `backend/` so the
    same code works regardless of where the process is launched from.
    Missing files are silently skipped by pydantic-settings.
    """
    here = Path(__file__).resolve()
    backend_dir = here.parent.parent  # backend/
    repo_root = here.parent.parent.parent  # repo root
    return [
        # Defaults / committed config — loaded first.
        str(repo_root / "config.env"),
        str(backend_dir / "config.env"),
        # Secrets / per-developer overrides — loaded second so they win.
        str(repo_root / ".env"),
        str(backend_dir / ".env"),
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=tuple(_project_env_files()),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM — provider selection + per-provider config
    llm_provider: str = "auto"  # "auto" | "openai" | "anthropic"
    openai_api_key: str = ""
    openai_strong_model: str = "gpt-5.5"
    openai_cheap_model: str = "gpt-4.1-mini"
    # Multi-agent role assignment (Phase 3+). Defaults reflect the architecture
    # spec: PM=GPT-5.5 Pro, sector/tool agents=GPT-5.4, critic=Opus 4.7,
    # news/social/long-doc analysts=Gemini.
    openai_pm_model: str = "gpt-5.5-pro"
    openai_sector_model: str = "gpt-5.4"
    openai_tool_model: str = "gpt-5.4"
    # Macro agent: GPT-5.4 default per the architecture spec; flip to Gemini
    # by setting OPENAI_MACRO_MODEL="" + GEMINI_API_KEY in the agent code path.
    openai_macro_model: str = "gpt-5.4"
    anthropic_api_key: str = ""
    anthropic_strong_model: str = "claude-opus-4-7"
    anthropic_cheap_model: str = "claude-haiku-4-5"
    anthropic_critic_model: str = "claude-opus-4-7"
    # Gemini (Google GenAI) — used for news/social/long-doc analysts.
    # Two access paths:
    #   - Direct API: set GEMINI_API_KEY. Quick setup; generous free tier.
    #   - Vertex AI: set VERTEX_PROJECT_ID (+ optionally VERTEX_LOCATION,
    #     VERTEX_MODEL). Auth via Google Application Default Credentials —
    #     run `gcloud auth application-default login` locally, or set
    #     GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json in prod.
    # When both are configured, Vertex wins so production deployments
    # don't accidentally fall back to API-key auth.
    gemini_api_key: str = ""
    gemini_news_model: str = "gemini-2.5-flash"
    gemini_social_model: str = "gemini-2.5-flash"
    gemini_longdoc_model: str = "gemini-3.1-pro"
    vertex_project_id: str = ""
    vertex_location: str = "us-central1"
    # When set, overrides the per-agent Gemini model envs across all Gemini
    # calls (news / social / longdoc) on the Vertex backend. Leave empty to
    # let each agent use its own GEMINI_*_MODEL.
    vertex_model: str = ""

    # Database
    database_url: str = "sqlite:///./marketmosaic.db"

    # Feature flags
    use_demo_data: bool = True
    enable_live_data: bool = False
    enable_agent_critic: bool = True
    enable_vector_search: bool = False
    # Phase 3: route single_stock_analysis through the OpenAI Agents SDK
    # instead of the legacy hand-rolled graph. Default off so existing tests
    # keep using the deterministic legacy path.
    use_agents_sdk: bool = False
    # Phase 5: always-on monitoring loops (EDGAR, news, social, macro). Default
    # off in dev/test; flip on in prod via env.
    enable_monitoring: bool = False
    # Long-term agent memory (filesystem markdown, delta-triggered).
    # `memory_dir` is the root; companies live at <root>/companies/<TICKER>.md
    # and sectors at <root>/sectors/<sector_slug>.md. Set absolute or
    # relative-to-CWD; default keeps state inside the backend dir for dev.
    enable_long_term_memory: bool = True
    memory_dir: str = "./memory"
    # When entry count crosses this cap, the oldest entries are condensed
    # into a "Historical context" block rather than discarded outright.
    memory_max_entries: int = 50
    memory_condense_batch: int = 10

    # Providers
    fmp_api_key: str = ""
    alpha_vantage_api_key: str = ""
    fred_api_key: str = ""
    polygon_api_key: str = ""
    tiingo_api_key: str = ""
    finnhub_api_key: str = ""
    intrinio_api_key: str = ""
    nasdaq_data_link_api_key: str = ""
    sec_user_agent: str = "MarketMosaic contact@example.com"

    # App / server
    app_env: str = "development"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:5173"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Runtime
    cache_ttl_seconds: int = 3600
    max_agent_context_chars: int = 60000
    max_stocks_in_portfolio: int = 25
    default_stock_universe: str = "large_cap_demo"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_gemini(self) -> bool:
        """True when ANY Gemini access path is configured (direct API or Vertex)."""
        return bool(self.gemini_api_key) or self.has_vertex

    @property
    def has_vertex(self) -> bool:
        """True when Vertex AI backend is configured. Auth handled by ADC —
        set GOOGLE_APPLICATION_CREDENTIALS or run `gcloud auth application-default
        login`. Vertex wins over direct API when both are set."""
        return bool(self.vertex_project_id)

    @property
    def has_llm(self) -> bool:
        return self.has_openai or self.has_anthropic

    @property
    def active_llm_provider(self) -> str:
        """Resolve the effective provider, honoring LLM_PROVIDER + key presence."""
        choice = (self.llm_provider or "auto").lower()
        if choice == "anthropic" and self.has_anthropic:
            return "anthropic"
        if choice == "openai" and self.has_openai:
            return "openai"
        # auto: prefer Anthropic if configured, else OpenAI
        if self.has_anthropic:
            return "anthropic"
        if self.has_openai:
            return "openai"
        return "none"

    @property
    def llm_enabled(self) -> bool:
        return self.has_llm and not self.use_demo_data_only

    @property
    def use_demo_data_only(self) -> bool:
        # If demo data flag is explicit and live data disabled, force demo
        return self.use_demo_data and not self.enable_live_data


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
