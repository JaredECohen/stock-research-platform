"""LLM helper with provider selection (OpenAI + Anthropic) and model routing.

`route` selects between a strong (PM/critic synthesis) and cheap (extraction)
model. The provider is resolved per call via `settings.active_llm_provider`,
which honors `LLM_PROVIDER` (auto/openai/anthropic) and key presence.

When no LLM is configured, helpers return None and callers fall back to
deterministic stub findings.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from ..config import settings

log = logging.getLogger(__name__)

try:  # OpenAI SDK is optional at runtime
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:  # Anthropic SDK is optional at runtime
    from anthropic import Anthropic  # type: ignore
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore

try:  # Gemini SDK is optional at runtime — graceful skip if missing.
    from google import genai as _genai  # type: ignore
except Exception:  # pragma: no cover
    _genai = None  # type: ignore


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
# Per-provider failure counter. After three consecutive failures, calls into
# that provider short-circuit to a typed empty/None response and we log a
# `provider_failure` row to CacheCostLog so the issue is visible.

_FAILURE_COUNTERS: Dict[str, int] = {"openai": 0, "anthropic": 0, "gemini": 0}
_BREAKER_THRESHOLD = 3


def _record_failure(provider: str) -> None:
    _FAILURE_COUNTERS[provider] = _FAILURE_COUNTERS.get(provider, 0) + 1
    if _FAILURE_COUNTERS[provider] >= _BREAKER_THRESHOLD:
        try:
            from ..cache import log_cost
            log_cost(provider, "provider_failure", 0,
                     note=f"{provider} circuit breaker tripped at {_FAILURE_COUNTERS[provider]} failures")
        except Exception:  # pragma: no cover
            pass


def _record_success(provider: str) -> None:
    _FAILURE_COUNTERS[provider] = 0


def _breaker_open(provider: str) -> bool:
    return _FAILURE_COUNTERS.get(provider, 0) >= _BREAKER_THRESHOLD


def reset_circuit_breaker(provider: Optional[str] = None) -> None:
    """Test helper / ops surface — clear the breaker for a provider (or all)."""
    if provider is None:
        for k in list(_FAILURE_COUNTERS.keys()):
            _FAILURE_COUNTERS[k] = 0
    else:
        _FAILURE_COUNTERS[provider] = 0


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

def _openai_client() -> Optional[Any]:
    if not settings.openai_api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=settings.openai_api_key)
    except Exception as exc:  # pragma: no cover
        log.warning("OpenAI client init failed: %s", exc)
        return None


def _anthropic_client() -> Optional[Any]:
    if not settings.anthropic_api_key or Anthropic is None:
        return None
    try:
        return Anthropic(api_key=settings.anthropic_api_key)
    except Exception as exc:  # pragma: no cover
        log.warning("Anthropic client init failed: %s", exc)
        return None


def _gemini_client() -> Optional[Any]:
    if not settings.gemini_api_key or _genai is None:
        return None
    try:
        return _genai.Client(api_key=settings.gemini_api_key)
    except Exception as exc:  # pragma: no cover
        log.warning("Gemini client init failed: %s", exc)
        return None


def gemini_chat_text(
    prompt: str,
    *,
    system: str = "",
    model: Optional[str] = None,
    enable_search_grounding: bool = False,
    max_tokens: int = 800,
) -> Optional[str]:
    """Lightweight Gemini text-completion wrapper.

    Search grounding is enabled by passing the `google_search` tool to the
    Generate Content API. The caller is responsible for filtering grounded
    sources against any allow/block list.
    """
    if _breaker_open("gemini"):
        return None
    client = _gemini_client()
    if client is None:
        return None
    chosen_model = model or settings.gemini_news_model
    full_prompt = (system + "\n\n" + prompt).strip() if system else prompt
    try:
        # Build config dynamically — different google-genai versions accept
        # slightly different shapes. We err on the side of being permissive.
        config: Dict[str, Any] = {"temperature": 0.3, "max_output_tokens": max_tokens}
        if enable_search_grounding:
            try:
                from google.genai import types  # type: ignore
                config["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            except Exception:
                # Older versions: tools accept a dict
                config["tools"] = [{"google_search": {}}]
        resp = client.models.generate_content(
            model=chosen_model,
            contents=full_prompt,
            config=config,
        )
        text = getattr(resp, "text", None)
        if text:
            _record_success("gemini")
            return text
        _record_failure("gemini")
        return None
    except Exception as exc:  # pragma: no cover
        log.warning("Gemini call failed: %s", exc)
        _record_failure("gemini")
        return None


def gemini_chat_json(
    prompt: str,
    *,
    system: str = "",
    model: Optional[str] = None,
    enable_search_grounding: bool = False,
    max_tokens: int = 800,
) -> Optional[Dict[str, Any]]:
    """JSON-mode wrapper around `gemini_chat_text` — appends a 'JSON only'
    instruction and parses the result with the same `_extract_json` helper as
    the Anthropic branch.
    """
    sys_with_json = (system + "\n\nReturn ONLY valid JSON, no prose.").strip()
    text = gemini_chat_text(
        prompt, system=sys_with_json, model=model,
        enable_search_grounding=enable_search_grounding, max_tokens=max_tokens,
    )
    return _extract_json(text or "")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _model_for(provider: str, route: str) -> str:
    if provider == "anthropic":
        return settings.anthropic_strong_model if route == "strong" else settings.anthropic_cheap_model
    return settings.openai_strong_model if route == "strong" else settings.openai_cheap_model


# ---------------------------------------------------------------------------
# Anthropic helpers
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from a model response."""
    if not text:
        return None
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fenced block
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # First {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _anthropic_chat(client: Any, *, model: str, system: str, user: str, max_tokens: int) -> Optional[str]:
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate text blocks
        parts = []
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts).strip() or None
    except Exception as exc:  # pragma: no cover
        log.warning("Anthropic call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def _openai_chat_json(client: Any, *, model: str, system: str, user: str, max_tokens: int) -> Optional[Dict[str, Any]]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user + "\n\nReturn ONLY valid JSON."})
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as exc:  # pragma: no cover
        log.warning("OpenAI JSON call failed: %s", exc)
        return None


def _openai_chat_text(client: Any, *, model: str, system: str, user: str, max_tokens: int) -> Optional[str]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as exc:  # pragma: no cover
        log.warning("OpenAI text call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def chat_json(
    prompt: str,
    *,
    system: str = "",
    route: str = "cheap",
    max_tokens: int = 800,
    provider_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Single-shot JSON-mode chat call. Returns parsed dict or None.

    `provider_override` lets a caller force a specific provider regardless of
    `settings.active_llm_provider`. Used by the Phase 4 critic agent which
    intentionally crosses the provider family boundary.
    """
    provider = (provider_override or settings.active_llm_provider).lower()
    if provider == "none":
        return None
    if _breaker_open(provider):
        return None

    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return None
        model = _model_for("anthropic", route)
        sys_with_json = (system + "\n\nReturn ONLY valid JSON, no prose.").strip()
        text = _anthropic_chat(client, model=model, system=sys_with_json, user=prompt, max_tokens=max_tokens)
        if text is None:
            _record_failure("anthropic")
            return None
        _record_success("anthropic")
        return _extract_json(text)

    if provider == "gemini":
        return gemini_chat_json(prompt, system=system, max_tokens=max_tokens)

    # OpenAI
    client = _openai_client()
    if client is None:
        return None
    model = _model_for("openai", route)
    out = _openai_chat_json(client, model=model, system=system, user=prompt, max_tokens=max_tokens)
    if out is None:
        _record_failure("openai")
    else:
        _record_success("openai")
    return out


def chat_text(
    prompt: str,
    *,
    system: str = "",
    route: str = "cheap",
    max_tokens: int = 600,
    provider_override: Optional[str] = None,
) -> Optional[str]:
    provider = (provider_override or settings.active_llm_provider).lower()
    if provider == "none":
        return None
    if _breaker_open(provider):
        return None

    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return None
        model = _model_for("anthropic", route)
        text = _anthropic_chat(client, model=model, system=system, user=prompt, max_tokens=max_tokens)
        if text is None:
            _record_failure("anthropic")
        else:
            _record_success("anthropic")
        return text

    if provider == "gemini":
        return gemini_chat_text(prompt, system=system, max_tokens=max_tokens)

    client = _openai_client()
    if client is None:
        return None
    model = _model_for("openai", route)
    text = _openai_chat_text(client, model=model, system=system, user=prompt, max_tokens=max_tokens)
    if text is None:
        _record_failure("openai")
    else:
        _record_success("openai")
    return text
