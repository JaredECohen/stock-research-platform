"""Offline provider clients for the LLM-layer tests (slice B7-M1).

Each fake records the exact request kwargs and answers with a response
shaped like the pinned SDKs' objects (anthropic 0.125, openai 3.8,
google-genai 2.22): only the attributes `agents/llm.py` reads. The served
model a fake reports is deliberately NOT the one sent (a dated alias), so a
test can tell "the provider said" from "we asked for" (attribution
critique #3).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.agents import llm
from app.config import settings


def anthropic_response(text: str = '{"ok": true}', *, model: str = "claude-served-2026-01-01",
                       input_tokens: int = 100, output_tokens: int = 20,
                       stop_reason: str = "end_turn", refusal_category: str | None = None) -> Any:
    stop_details = (SimpleNamespace(type="refusal", category=refusal_category)
                    if stop_reason == "refusal" else None)
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0),
        content=[SimpleNamespace(text=text)] if text else [],
        stop_reason=stop_reason,
        stop_details=stop_details,
    )


def openai_response(text: str | None = '{"ok": true}', *, model: str = "gpt-served-2026-01-01",
                    prompt_tokens: int = 100, completion_tokens: int = 20,
                    reasoning_tokens: int | None = None, finish_reason: str = "stop") -> Any:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
    )


def gemini_response(text: str = '{"ok": true}', *, model_version: str = "gemini-served-001",
                    prompt: int = 100, candidates: int = 20, thoughts: int | None = None,
                    tool_use: int | None = None, cached: int | None = None,
                    finish_reason: str = "STOP") -> Any:
    return SimpleNamespace(
        model_version=model_version,
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt, candidates_token_count=candidates,
            thoughts_token_count=thoughts, tool_use_prompt_token_count=tool_use,
            cached_content_token_count=cached,
        ),
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
    )


class FakeClient:
    """One object that answers as any of the three SDKs."""

    def __init__(self, *responses: Any):
        self.requests: list[dict[str, Any]] = []
        self._responses = list(responses)

    def _next(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def messages(self) -> Any:
        return SimpleNamespace(create=self._next)

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=SimpleNamespace(create=self._next))

    @property
    def models(self) -> Any:
        return SimpleNamespace(generate_content=self._next)


def live(monkeypatch, *, openai: Any = None, anthropic: Any = None, gemini: Any = None,
         active: str = "openai") -> None:
    """Configure both chat providers (stub keys) and install fake clients,
    running as a live deployment would (not demo-only). A provider passed
    as None has a key but NO client (the "client unavailable" case)."""
    monkeypatch.setattr(settings, "openai_api_key", "stub-openai")
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-anthropic")
    monkeypatch.setattr(settings, "gemini_api_key", "stub-gemini")
    monkeypatch.setattr(settings, "vertex_project_id", "")
    monkeypatch.setattr(settings, "llm_failover_enabled", True)
    monkeypatch.setattr(type(settings), "active_llm_provider", property(lambda self: active))
    monkeypatch.setattr(llm, "_demo_only", lambda: False)
    monkeypatch.setattr(llm, "_openai_client", lambda: openai)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: anthropic)
    monkeypatch.setattr(llm, "_gemini_client", lambda: gemini)
    llm.reset_circuit_breaker()
    llm.reset_failover_state()


def rows_for(run_id: str) -> list[Any]:
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import LLMCallLog
    with SessionLocal() as db:
        LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
        return list(db.execute(
            select(LLMCallLog).where(LLMCallLog.run_id == run_id).order_by(LLMCallLog.id)
        ).scalars())


def parse_line(message: str) -> dict[str, str]:
    """logfmt → dict (values may be double-quoted)."""
    import shlex
    parts = shlex.split(message)
    assert parts[0] == "llm_call", message
    out: dict[str, str] = {}
    for part in parts[1:]:
        key, _, value = part.partition("=")
        out[key] = value
    return out
