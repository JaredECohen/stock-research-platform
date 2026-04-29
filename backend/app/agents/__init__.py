"""Agent framework for MarketMosaic.

The orchestrator routes user intents to specialist agents and synthesizes a
final PM view. Every agent emits Pydantic-validated structured output so the
frontend can render rich, deterministic memos even when the LLM is offline.
"""
from .orchestrator import Orchestrator  # noqa: F401
