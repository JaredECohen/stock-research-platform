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
from typing import Any

from ..config import settings
from .log_safety import log_safely, redact

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

_FAILURE_COUNTERS: dict[str, int] = {"openai": 0, "anthropic": 0, "gemini": 0}
# Wall-clock timestamp of last failure per provider — used by the
# self-healing breaker to auto-reset after _BREAKER_COOLDOWN_SECONDS
# without a fresh failure. Without auto-reset the breaker pins open
# until process restart; transient blips from one bad call (e.g.,
# the PM DCF Adjuster's forced-OpenAI override before that was fixed)
# would cascade-block every other call's provider lookup forever.
_FAILURE_LAST_AT: dict[str, float] = {}
_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_SECONDS = 120.0  # 2 min idle = self-heal


def _record_failure(provider: str) -> None:
    import time as _time
    _FAILURE_COUNTERS[provider] = _FAILURE_COUNTERS.get(provider, 0) + 1
    _FAILURE_LAST_AT[provider] = _time.time()
    if _FAILURE_COUNTERS[provider] >= _BREAKER_THRESHOLD:
        try:
            from ..cache import log_cost
            log_cost(provider, "provider_failure", 0,
                     note=f"{provider} circuit breaker tripped at {_FAILURE_COUNTERS[provider]} failures")
        except Exception:  # pragma: no cover
            pass


def _record_success(provider: str) -> None:
    _FAILURE_COUNTERS[provider] = 0
    _FAILURE_LAST_AT.pop(provider, None)


def _breaker_open(provider: str) -> bool:
    """True iff the breaker is tripped AND not yet eligible to retry.

    Auto-resets after `_BREAKER_COOLDOWN_SECONDS` of no fresh failure.
    The cooldown gives the provider time to recover from a transient
    blip without blocking otherwise-healthy callers. Resetting on time
    is intentionally permissive — a real outage will re-trip on the
    first retry; a transient was real-but-brief and worth retrying.
    """
    import time as _time
    if _FAILURE_COUNTERS.get(provider, 0) < _BREAKER_THRESHOLD:
        return False
    last_at = _FAILURE_LAST_AT.get(provider)
    if last_at is None:
        return True
    if _time.time() - last_at >= _BREAKER_COOLDOWN_SECONDS:
        # Cooldown elapsed — reset the counter and let one call through.
        _FAILURE_COUNTERS[provider] = 0
        _FAILURE_LAST_AT.pop(provider, None)
        return False
    return True


def reset_circuit_breaker(provider: str | None = None) -> None:
    """Test helper / ops surface — clear the breaker for a provider (or all)."""
    if provider is None:
        for k in list(_FAILURE_COUNTERS.keys()):
            _FAILURE_COUNTERS[k] = 0
        _FAILURE_LAST_AT.clear()
    else:
        _FAILURE_COUNTERS[provider] = 0
        _FAILURE_LAST_AT.pop(provider, None)


def get_breaker_state(include_failover: bool = True) -> dict[str, dict[str, Any]]:
    """Snapshot of circuit-breaker state for the admin endpoint.

    Carries a `failover` key (see `get_failover_state`) beside the three
    provider rows so the admin breaker view shows a hop when one happened;
    a breaker that never opened tells only half the story once failover
    exists. `include_failover=False` is for consumers that already expose
    failover elsewhere and need strictly one row per provider (the
    `/api/providers/status` contract).
    """
    import time as _time
    now = _time.time()
    out: dict[str, dict[str, Any]] = {}
    for provider in ("openai", "anthropic", "gemini"):
        last_at = _FAILURE_LAST_AT.get(provider)
        count = _FAILURE_COUNTERS.get(provider, 0)
        out[provider] = {
            "failure_count": count,
            "is_open": count >= _BREAKER_THRESHOLD and (
                last_at is not None
                and now - last_at < _BREAKER_COOLDOWN_SECONDS
            ),
            "seconds_since_last_failure": (
                round(now - last_at, 1) if last_at is not None else None
            ),
            "cooldown_seconds": _BREAKER_COOLDOWN_SECONDS,
        }
    if include_failover:
        out["failover"] = get_failover_state()
    return out


# ---------------------------------------------------------------------------
# Bounded provider failover (openai <-> anthropic)
# ---------------------------------------------------------------------------
# One hop, once per call, never a loop. The module-level dict is the ops
# view (per-process, same scope caveat as the breaker); the contextvar is
# the per-run view the memo pipeline drains into its DegradationLog so a
# memo produced on the backup vendor says so.

_FAILOVER_STATE: dict[str, Any] = {
    "count": 0,
    "last_from": None,
    "last_to": None,
    "last_at": None,
    "last_reason": None,
}
_FAILOVER_PARTNER = {"openai": "anthropic", "anthropic": "openai"}


def _failover_partner(provider: str) -> str | None:
    """The provider we may fail over to from `provider`, or None.

    None when failover is disabled, when `provider` has no partner (gemini
    stays a specialist path), or when the partner has no key — a failover
    to an unconfigured provider would just be a second failure.
    """
    if not settings.llm_failover_enabled:
        return None
    partner = _FAILOVER_PARTNER.get(provider)
    if partner == "openai" and settings.has_openai:
        return partner
    if partner == "anthropic" and settings.has_anthropic:
        return partner
    return None


def _record_failover(src: str, dst: str, reason: str) -> None:
    import time as _time
    _FAILOVER_STATE["count"] += 1
    _FAILOVER_STATE["last_from"] = src
    _FAILOVER_STATE["last_to"] = dst
    _FAILOVER_STATE["last_at"] = _time.time()
    _FAILOVER_STATE["last_reason"] = reason
    events = _FAILOVER_EVENTS.get()
    if events is None:
        events = []
        _FAILOVER_EVENTS.set(events)
    events.append({"from": src, "to": dst, "reason": reason})
    # No exception to hand over — the wrappers already turned it into
    # None — but the line still goes through the redacting path so a
    # future reason string can never carry key material.
    log_safely(log, f"LLM failover from {src} to {dst} ({reason})", None)


def get_failover_state() -> dict[str, Any]:
    """Copy of the per-process failover counters. No secrets: `last_reason`
    is a short category ("breaker_open" / "call_failed"), never an
    exception body."""
    return dict(_FAILOVER_STATE)


def consume_failover_events() -> list[dict[str, str]]:
    """Drain the failover events recorded in the current context.

    Context-local (like `llm_call_context`) so a regen-worker memo run
    and a concurrent web chat call do not read each other's events.
    """
    events = _FAILOVER_EVENTS.get()
    if not events:
        return []
    out = list(events)
    events.clear()
    return out


def reset_failover_state() -> None:
    """Test helper — clear the ops counters and the current context's events."""
    _FAILOVER_STATE.update(
        count=0, last_from=None, last_to=None, last_at=None, last_reason=None,
    )
    events = _FAILOVER_EVENTS.get()
    if events:
        events.clear()


# ---------------------------------------------------------------------------
# Per-role model resolution
# ---------------------------------------------------------------------------
# Each agent role has an env knob (OPENAI_PM_MODEL, OPENAI_SECTOR_MODEL, …).
# An unset env resolves to "" — and "" reached the Agents SDK verbatim,
# which the SDK rejects. The route default is what an empty knob means.

_ROLE_SETTINGS = {
    # role: (settings attribute, route the role runs on)
    "pm": ("openai_pm_model", "strong"),
    "sector": ("openai_sector_model", "cheap"),
    "tool": ("openai_tool_model", "cheap"),
    "macro": ("openai_macro_model", "cheap"),
    "critic": ("anthropic_critic_model", "strong"),
    "strong": (None, "strong"),
    "cheap": (None, "cheap"),
}


def _provider_for_role(role: str) -> str:
    """The provider `role` actually runs on when no explicit one is given.

    Every role follows the active provider except the critic, which
    `critic_agent` force-routes to Anthropic whenever a key is present
    (Phase 4: the reviewer should not share the author's vendor). The
    role table has to say the same thing, or the ops page answers
    "which model reviewed this memo" with a model that never ran.
    """
    if role == "critic" and settings.has_anthropic:
        return "anthropic"
    return settings.active_llm_provider


def resolve_role_model(role: str, provider: str | None = None) -> str:
    """Model name to run `role` on, never blank and never provider-foreign.

    Returns the configured per-role model when it is non-blank AND named
    for `provider`'s family; otherwise that provider's route default. The
    provider defaults to the one the role really runs on (see
    `_provider_for_role`); the Agents SDK runtime passes `provider="openai"`
    explicitly because that SDK only speaks OpenAI regardless of which
    provider `chat_json` would pick.
    """
    try:
        attr, route = _ROLE_SETTINGS[role]
    except KeyError:
        raise ValueError(f"unknown LLM role: {role!r}") from None
    prov = (provider or _provider_for_role(role)).lower()
    if prov not in _FAILOVER_PARTNER:
        # "none" (no keys) still has to yield a usable name for the SDK
        # shim's Agent objects; OpenAI is the shape every default carries.
        prov = "openai"
    configured = (getattr(settings, attr, "") or "").strip() if attr else ""
    if configured and _model_matches_provider(configured, prov):
        return configured
    return _model_for(prov, route)


def model_summary() -> dict[str, Any]:
    """Routing snapshot for the startup log and the status endpoint.

    Contains model names and booleans only — never key material.
    """
    return {
        "active_provider": settings.active_llm_provider,
        "provider_choice": settings.llm_provider,
        "role_models": {role: resolve_role_model(role) for role in _ROLE_SETTINGS},
        "configured": {
            "openai": settings.has_openai,
            "anthropic": settings.has_anthropic,
            "gemini": settings.has_gemini,
        },
    }


# ---------------------------------------------------------------------------
# Per-call usage tracking (Phase C)
# ---------------------------------------------------------------------------
# After every provider call we stash `{provider, input_tokens, output_tokens,
# total_tokens, model}` into a thread-local. Call-site wrappers (the cache
# layer in particular) read this with `last_usage()` and pass `total_tokens`
# into `cache_put(cost_tokens=...)` so warm vs cold accounting reflects real
# spend, not the rough constants we used in demo mode.
import contextvars  # noqa: E402
import threading  # noqa: E402  (keep local — only needed here)

_USAGE_STATE = threading.local()


# ---------------------------------------------------------------------------
# Wave 1A — LLM call trace logging context
# ---------------------------------------------------------------------------
# `LLMCallContext` (set via `llm_call_context()`) tags every provider call
# with the agent name + run_id so the persisted LLMCallLog row can be
# attributed to a specific memo run. Default values when no context is set
# keep older callers working unchanged.

_CALL_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "llm_call_context",
    default={"agent_name": "unknown", "run_id": None, "route": "", "user_id": None, "feature": None},
)
# Failover events for the current context; `None` default (not `[]`) so a
# shared mutable default can't leak events across contexts.
_FAILOVER_EVENTS: contextvars.ContextVar[list[dict[str, str]] | None] = (
    contextvars.ContextVar("llm_failover_events", default=None)
)


class llm_call_context:
    """Context manager that tags subsequent llm.* calls with agent + run_id.

    Usage:
        with llm_call_context(agent_name="Sector Analyst", run_id=run_id):
            llm.chat_json(...)
    """

    def __init__(self, *, agent_name: str = "unknown", run_id: str | None = None,
                 route: str = "", user_id: int | None = None,
                 feature: str | None = None) -> None:
        # `user_id` / `feature` (FEAT-002) attribute spend to the customer
        # and product feature that caused it; the worker sets them from
        # the RegenJob row, the chat route from the request principal.
        self._values = {
            "agent_name": agent_name, "run_id": run_id, "route": route,
            "user_id": user_id, "feature": feature,
        }
        self._token: contextvars.Token | None = None

    def __enter__(self) -> llm_call_context:
        # Layer on top of any existing context — fields the caller didn't set
        # carry over. Lets nested calls override only what they need.
        prev = _CALL_CONTEXT.get()
        merged = {**prev, **{k: v for k, v in self._values.items() if v}}
        self._token = _CALL_CONTEXT.set(merged)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _CALL_CONTEXT.reset(self._token)


def current_call_context() -> dict[str, Any]:
    """Return the active llm_call_context dict (agent_name / run_id / route).

    Used by Wave 6A's checkpoint decorator to read the active `run_id`
    without threading it through every step's signature.
    """
    return dict(_CALL_CONTEXT.get())


def _record_usage(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    duration_ms: int = 0,
    success: bool = True,
    error: str = "",
) -> None:
    total = max(0, int(input_tokens or 0)) + max(0, int(output_tokens or 0))
    _USAGE_STATE.last = {
        "provider": provider,
        "model": model,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": total,
    }
    # Persist to the LLMCallLog audit table (Wave 1A). Lazy import to avoid
    # an import-time cycle (models → cache → ... ). DB failures must NEVER
    # break the LLM call path — wrap and swallow.
    try:
        from ..database import SessionLocal
        from ..models import LLMCallLog
        ctx = _CALL_CONTEXT.get()
        with SessionLocal() as db:
            # Lazy create so direct-import callers don't need init_db().
            LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
            db.add(LLMCallLog(
                run_id=ctx.get("run_id"),
                agent_name=ctx.get("agent_name") or "unknown",
                provider=provider,
                model=model,
                route=ctx.get("route") or "",
                tokens_in=int(input_tokens or 0),
                tokens_out=int(output_tokens or 0),
                duration_ms=int(duration_ms or 0),
                success=bool(success),
                error=str(error or "")[:500],
                user_id=ctx.get("user_id"),
                feature=ctx.get("feature"),
            ))
            db.commit()
    except Exception as exc:  # pragma: no cover - defense in depth
        log_safely(log, "LLMCallLog persist failed", exc)


def last_usage() -> dict[str, Any] | None:
    """Return the usage dict from the most recent provider call on this thread.

    Calling this *consumes* the value: subsequent calls return None until the
    next provider call records new usage. This prevents the same usage being
    accidentally double-counted across two cache_put sites.
    """
    val = getattr(_USAGE_STATE, "last", None)
    if val is not None:
        _USAGE_STATE.last = None
    return val


def _usage_from_openai(resp: Any) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)


def _usage_from_anthropic(msg: Any) -> tuple[int, int]:
    usage = getattr(msg, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def _usage_from_gemini(resp: Any) -> tuple[int, int]:
    # google-genai exposes `usage_metadata.{prompt_token_count, candidates_token_count}`.
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return 0, 0
    return (
        int(getattr(meta, "prompt_token_count", 0) or 0),
        int(getattr(meta, "candidates_token_count", 0) or 0),
    )


def _safe_finish_diagnostic(provider: str, response: Any) -> str:
    """Read existing stop metadata without persisting arbitrary response text."""
    try:
        if provider == "anthropic":
            field = "stop_reason"
            value = getattr(response, field, None)
            allowed = {"end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal", "model_context_window_exceeded"}
        elif provider == "openai":
            field = "finish_reason"
            value = getattr(response.choices[0], field, None)
            allowed = {"stop", "length", "tool_calls", "content_filter", "function_call"}
        else:
            field = "finish_reason"
            candidates = getattr(response, "candidates", None) or []
            value = getattr(candidates[0], field, None) if candidates else None
            allowed = {
                "stop", "max_tokens", "safety", "recitation", "other", "blocklist",
                "prohibited_content", "spii", "malformed_function_call", "language",
                "unexpected_tool_call", "finish_reason_unspecified",
            }
        if value is None:
            return ""
        name = value if isinstance(value, str) else getattr(value, "name", None)
        reason = name.lower() if isinstance(name, str) and name.lower() in allowed else "unrecognized"
        return f";{field}={reason}"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

_demo_only_noted = False


def _demo_only() -> bool:
    """USE_DEMO_DATA=true with ENABLE_LIVE_DATA=false (every test run) means
    no LLM traffic either: a developer .env carrying live keys must not turn
    the suite into paid calls. Callers see "no client", the same path as a
    missing key, so breaker and failover bookkeeping are untouched."""
    global _demo_only_noted
    if not settings.use_demo_data_only:
        return False
    if not _demo_only_noted:
        _demo_only_noted = True
        log.debug("demo-only mode: LLM clients are not constructed")
    return True


def _openai_client() -> Any | None:
    if _demo_only() or not settings.openai_api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=settings.openai_api_key)
    except Exception as exc:  # pragma: no cover
        log_safely(log, "OpenAI client init failed", exc)
        return None


def _anthropic_client() -> Any | None:
    if _demo_only() or not settings.anthropic_api_key or Anthropic is None:
        return None
    try:
        return Anthropic(api_key=settings.anthropic_api_key)
    except Exception as exc:  # pragma: no cover
        log_safely(log, "Anthropic client init failed", exc)
        return None


def _gemini_client() -> Any | None:
    """Construct a Gemini client.

    Backend selection precedence (Vertex wins when both are set):
      1. Vertex AI:    `VERTEX_PROJECT_ID` set → `Client(vertexai=True, project=…)`.
                       Auth via Google Application Default Credentials.
      2. Direct API:   `GEMINI_API_KEY` set → `Client(api_key=…)`.
      3. Otherwise:    None (caller falls back to deterministic stub).
    """
    if _genai is None or _demo_only():
        return None
    try:
        if settings.has_vertex:
            return _genai.Client(
                vertexai=True,
                project=settings.vertex_project_id,
                location=settings.vertex_location or "us-central1",
            )
        if settings.gemini_api_key:
            return _genai.Client(api_key=settings.gemini_api_key)
        return None
    except Exception as exc:  # pragma: no cover
        log_safely(log, "Gemini client init failed", exc)
        return None


def _resolve_gemini_model(caller_model: str | None, default: str) -> str:
    """Pick the Gemini model for a call.

    Order of precedence:
      1. `caller_model` (explicit override at the call site)
      2. `settings.vertex_model` when Vertex is configured (global override)
      3. `default` (per-agent env, e.g. settings.gemini_news_model)
    """
    if caller_model and caller_model.strip():
        return caller_model.strip()
    if settings.has_vertex and settings.vertex_model:
        return settings.vertex_model
    return default


def gemini_chat_text(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    enable_search_grounding: bool = False,
    max_tokens: int = 800,
    _json_mode: bool = False,
) -> Any:
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
    chosen_model = _resolve_gemini_model(model, settings.gemini_news_model)
    full_prompt = (system + "\n\n" + prompt).strip() if system else prompt
    import time as _time
    t0 = _time.perf_counter()
    in_tok = out_tok = 0
    received_response = False
    out = None
    error = ""
    finish_diagnostic = ""
    try:
        # Build config dynamically — different google-genai versions accept
        # slightly different shapes. We err on the side of being permissive.
        config: dict[str, Any] = {"temperature": 0.3, "max_output_tokens": max_tokens}
        if enable_search_grounding:
            try:
                from google.genai import types  # type: ignore
                config["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            except Exception:
                # Older versions: tools accept a dict
                log.debug("google-genai types.Tool unavailable; using the dict tool config")
                config["tools"] = [{"google_search": {}}]
        resp = client.models.generate_content(
            model=chosen_model,
            contents=full_prompt,
            config=config,
        )
        received_response = True
        in_tok, out_tok = _usage_from_gemini(resp)
        finish_diagnostic = _safe_finish_diagnostic("gemini", resp)
        text = getattr(resp, "text", None)
        out = _extract_json(text) if _json_mode and text else (text or None)
        if out is None:
            error = "invalid_json_response" if text else "empty_response"
    except Exception as exc:  # pragma: no cover
        error = f"{'response_error' if received_response else 'provider_error'}:{type(exc).__name__}"
        log.warning("Gemini call failed (%s)", error)
        out = None
    # JSON success is decided only after the existing recovery parser runs.
    # A failed parse still consumed the response's real tokens: one call, one row.
    _record_usage(
        "gemini", chosen_model, in_tok, out_tok,
        duration_ms=int((_time.perf_counter() - t0) * 1000),
        success=out is not None, error=error + finish_diagnostic if error else "",
    )
    if out is None:
        _record_failure("gemini")
    else:
        _record_success("gemini")
    return out


def gemini_chat_json(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    enable_search_grounding: bool = False,
    max_tokens: int = 800,
) -> dict[str, Any] | None:
    """JSON-mode wrapper around `gemini_chat_text` — appends a 'JSON only'
    instruction and parses the result with the same `_extract_json` helper as
    the Anthropic branch.
    """
    sys_with_json = (system + "\n\nReturn ONLY valid JSON, no prose.").strip()
    return gemini_chat_text(
        prompt, system=sys_with_json, model=model,
        enable_search_grounding=enable_search_grounding, max_tokens=max_tokens,
        _json_mode=True,
    )


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


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a model response.

    Handles three failure modes seen in the wild:
      1. Plain JSON wrapped in a ```json fence.
      2. Gemini's grounded responses occasionally emit two copies of the
         JSON prefix back-to-back (a tool-use retry artifact). Walk every
         '{' offset and accept the first one that parses.
      3. Truncated JSON arrays — the model stopped mid-string. Trim the
         trailing partial item and close the array/object so we recover
         what was emitted.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Truncated-or-duplicated `{"items": [ ... ]}` recovery. Walk every
    # `"items": [` occurrence and salvage the complete objects inside —
    # this handles both Gemini's grounded duplicate-prefix artifact
    # (later copy is more complete) and outright truncation. We pick
    # whichever pass yields the most items.
    best: list[dict[str, Any]] | None = None
    for items_match in re.finditer(r'"items"\s*:\s*\[', text):
        complete = _walk_array_objects(text, items_match.end())
        if not complete:
            continue
        try:
            parsed = [json.loads(o) for o in complete]
        except Exception:
            continue
        if best is None or len(parsed) > len(best):
            best = parsed
    if best is not None:
        return {"items": best}
    # Last resort: greedy first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            # Length only — the body may carry prompt or provider text.
            log.debug("LLM output was not parseable JSON after every recovery (len=%d)", len(text))
            return None
    log.debug("LLM output held no JSON object to recover (len=%d)", len(text))
    return None


def _walk_array_objects(text: str, start_idx: int) -> list[str]:
    """Return raw text of every top-level `{...}` inside the array beginning
    at `start_idx` (which should point just past the opening `[`). Stops at
    the array's closing `]` or end-of-string."""
    out: list[str] = []
    depth = 0
    in_str = False
    esc = False
    obj_start = -1
    i = start_idx
    while i < len(text):
        ch = text[i]
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    out.append(text[obj_start : i + 1])
                    obj_start = -1
            elif ch == "]" and depth == 0:
                break
        i += 1
    return out


def _anthropic_supports_custom_temp(model: str) -> bool:
    """Newer Anthropic models (Opus 4.7+, Sonnet 4.6+, Haiku 4.5+) reject
    `temperature` other than the default and return HTTP 400 with
    `temperature is deprecated for this model`. Empirically observed
    2026-05-30 against claude-opus-4-7 and claude-haiku-4-5. Older
    Claude 3.x models accept it. We gate the parameter rather than
    pinning a fixed default so the call works across the supported
    model fleet."""
    m = (model or "").lower().strip()
    if not m.startswith("claude-"):
        return True
    # The "4-x" generation deprecates the parameter. Pattern: "claude-
    # {opus,sonnet,haiku}-4-N" or "claude-{family}-4-N-YYYYMMDD".
    if "-4-" in m or m.endswith("-4"):
        return False
    return True


def _anthropic_chat(
    client: Any, *, model: str, system: str, user: str, max_tokens: int,
    json_mode: bool = False,
) -> Any:
    import time as _time
    t0 = _time.perf_counter()
    in_tok = out_tok = 0
    received_response = False
    out = None
    error = ""
    finish_diagnostic = ""
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system or "You are a helpful assistant.",
            "messages": [{"role": "user", "content": user}],
        }
        if _anthropic_supports_custom_temp(model):
            kwargs["temperature"] = 0.3
        msg = client.messages.create(**kwargs)
        received_response = True
        # Capture real token usage for cost accounting (Phase C) + log row (Wave 1A).
        in_tok, out_tok = _usage_from_anthropic(msg)
        finish_diagnostic = _safe_finish_diagnostic("anthropic", msg)
        # Concatenate text blocks
        parts = []
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        text = "".join(parts).strip() or None
        out = _extract_json(text) if json_mode and text is not None else text
        if out is None:
            error = "invalid_json_response" if text else "empty_response"
    except Exception as exc:  # pragma: no cover
        error = f"{'response_error' if received_response else 'provider_error'}:{type(exc).__name__}"
        log.warning("Anthropic call failed (%s)", error)
        out = None
    _record_usage(
        "anthropic", model, in_tok, out_tok,
        duration_ms=int((_time.perf_counter() - t0) * 1000),
        success=out is not None, error=error + finish_diagnostic if error else "",
    )
    return out


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def _openai_token_kwarg(model: str, n: int) -> dict[str, int]:
    """Return the correct max-output-tokens kwarg for the OpenAI model family.

    GPT-5.x and the o-series reasoning models (o1, o3, o4, …) reject
    `max_tokens` and require `max_completion_tokens`. Older / non-reasoning
    chat models (gpt-4.1, gpt-4o, gpt-3.5, …) still take `max_tokens`.
    Verified empirically against gpt-5.4, gpt-5.5, and gpt-4.1-mini on
    2026-05-02; the new-name convention also covers o1 / o3 reasoning
    models which use the same API contract.
    """
    m = (model or "").lower().strip()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return {"max_completion_tokens": int(n)}
    return {"max_tokens": int(n)}


def _openai_supports_custom_temp(model: str) -> bool:
    """GPT-5.x and o-series reasoning models reject `temperature` other
    than the default (1). Older chat models accept it. Verified
    empirically 2026-05-03 — gpt-5.5 returns 400 on temperature=0.3."""
    m = (model or "").lower().strip()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return False
    return True


def _openai_chat_json(client: Any, *, model: str, system: str, user: str, max_tokens: int) -> dict[str, Any] | None:
    import time as _time
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user + "\n\nReturn ONLY valid JSON."})
    t0 = _time.perf_counter()
    in_tok = out_tok = 0
    received_response = False
    out = None
    error = ""
    finish_diagnostic = ""
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            **_openai_token_kwarg(model, max_tokens),
        }
        if _openai_supports_custom_temp(model):
            kwargs["temperature"] = 0.3
        resp = client.chat.completions.create(**kwargs)
        received_response = True
        in_tok, out_tok = _usage_from_openai(resp)
        finish_diagnostic = _safe_finish_diagnostic("openai", resp)
        content = resp.choices[0].message.content
        try:
            out = json.loads(content)
        except (ValueError, TypeError):
            error = "invalid_json_response" if content else "empty_response"
        if out is None and not error:
            error = "invalid_json_response"
    except Exception as exc:  # pragma: no cover
        error = f"{'response_error' if received_response else 'provider_error'}:{type(exc).__name__}"
        log.warning("OpenAI JSON call failed (%s)", error)
        out = None
    _record_usage(
        "openai", model, in_tok, out_tok,
        duration_ms=int((_time.perf_counter() - t0) * 1000),
        success=out is not None, error=error + finish_diagnostic if error else "",
    )
    return out


def _openai_chat_text(client: Any, *, model: str, system: str, user: str, max_tokens: int) -> str | None:
    import time as _time
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    t0 = _time.perf_counter()
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            **_openai_token_kwarg(model, max_tokens),
        }
        if _openai_supports_custom_temp(model):
            kwargs["temperature"] = 0.3
        resp = client.chat.completions.create(**kwargs)
        dur = int((_time.perf_counter() - t0) * 1000)
        in_tok, out_tok = _usage_from_openai(resp)
        out = resp.choices[0].message.content
        _record_usage("openai", model, in_tok, out_tok,
                      duration_ms=dur, success=bool(out))
        return out
    except Exception as exc:  # pragma: no cover
        dur = int((_time.perf_counter() - t0) * 1000)
        log_safely(log, "OpenAI text call failed", exc)
        _record_usage("openai", model, 0, 0,
                      duration_ms=dur, success=False, error=redact(exc))
        return None


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def _model_matches_provider(model: str | None, provider: str) -> bool:
    """True iff `model` is named for the given provider's family.

    Wave 9b — agents pass per-role model envs that were originally
    OpenAI-shaped (`OPENAI_TOOL_MODEL`, `OPENAI_SECTOR_MODEL`, …).
    When the active provider is Anthropic or Gemini, those names 404
    and every LLM call silently fails through to the deterministic
    stub. We treat a mismatch as "no override" so the route default
    for the active provider kicks in.
    """
    if not model:
        return True
    m = model.lower().strip()
    if not m:
        return True
    if provider == "openai":
        return m.startswith(("gpt-", "o1-", "o3-", "o4-", "openai/"))
    if provider == "anthropic":
        return m.startswith(("claude-", "anthropic/"))
    if provider in ("gemini", "vertex"):
        return m.startswith(("gemini-", "models/gemini", "publishers/google"))
    return True  # unknown provider — pass through unchanged


def _call_json(
    provider: str, *, prompt: str, system: str, route: str,
    max_tokens: int, model: str | None,
) -> dict[str, Any] | None:
    """One JSON-mode call against `provider`, feeding its breaker counters.

    Returns None when the provider is unconfigured (no counter change) or
    the call failed (failure recorded). `model` must already be vetted by
    `_model_matches_provider` for this provider — the failover hop passes
    None so it lands on the partner's own route default.
    """
    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return None
        chosen = (model or "").strip() or _model_for("anthropic", route)
        sys_with_json = (system + "\n\nReturn ONLY valid JSON, no prose.").strip()
        out = _anthropic_chat(
            client, model=chosen, system=sys_with_json, user=prompt,
            max_tokens=max_tokens, json_mode=True,
        )
        if out is None:
            _record_failure("anthropic")
        else:
            _record_success("anthropic")
        return out

    client = _openai_client()
    if client is None:
        return None
    chosen = (model or "").strip() or _model_for("openai", route)
    out = _openai_chat_json(client, model=chosen, system=system, user=prompt, max_tokens=max_tokens)
    if out is None:
        _record_failure("openai")
    else:
        _record_success("openai")
    return out


def _call_text(
    provider: str, *, prompt: str, system: str, route: str,
    max_tokens: int, model: str | None,
) -> str | None:
    """Text twin of `_call_json`; same contract."""
    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return None
        chosen = (model or "").strip() or _model_for("anthropic", route)
        text = _anthropic_chat(client, model=chosen, system=system, user=prompt, max_tokens=max_tokens)
        if text is None:
            _record_failure("anthropic")
        else:
            _record_success("anthropic")
        return text

    client = _openai_client()
    if client is None:
        return None
    chosen = (model or "").strip() or _model_for("openai", route)
    text = _openai_chat_text(client, model=chosen, system=system, user=prompt, max_tokens=max_tokens)
    if text is None:
        _record_failure("openai")
    else:
        _record_success("openai")
    return text


def _with_failover(provider: str, call: Any, **kwargs: Any) -> Any:
    """Run `call(provider, **kwargs)`, hopping to the partner provider once.

    The hop happens when `provider`'s breaker is already open or the call
    returns None (the wrappers convert exceptions to None). It is skipped
    — returning None exactly as before failover existed — when failover
    is disabled, no partner is configured, or the partner's breaker is
    open too. The partner runs on its *own* route default (`model=None`)
    because the caller's model name belongs to the failed provider.
    """
    if _breaker_open(provider):
        reason = "breaker_open"
    else:
        out = call(provider, **kwargs)
        if out is not None:
            return out
        reason = "call_failed"

    partner = _failover_partner(provider)
    if partner is None:
        return None
    if _breaker_open(partner):
        log.debug("LLM failover from %s to %s skipped: partner breaker open", provider, partner)
        return None
    _record_failover(provider, partner, reason)
    return call(partner, **{**kwargs, "model": None})


def chat_json(
    prompt: str,
    *,
    system: str = "",
    route: str = "cheap",
    # Wave 9b — bumped from 800 → 1600. Specialist agents routinely
    # emit 4-6KB of JSON (headline + multi-paragraph summary + 8-12
    # key_points). 800 tokens was truncating responses mid-string,
    # which `_extract_json` couldn't parse → caller fell through to
    # deterministic stub. Per-call overrides still apply.
    max_tokens: int = 1600,
    provider_override: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Single-shot JSON-mode chat call. Returns parsed dict or None.

    `provider_override` lets a caller force a specific provider regardless of
    `settings.active_llm_provider`. Used by the Phase 4 critic agent which
    intentionally crosses the provider family boundary.

    `model` is an explicit per-call model override. When set (and non-empty),
    it bypasses the `route="strong"|"cheap"` default. Per-agent envs
    (`OPENAI_PM_MODEL`, `OPENAI_SECTOR_MODEL`, `ANTHROPIC_CRITIC_MODEL`, …)
    flow through this knob: the call site reads `settings.openai_pm_model`
    (or whichever role applies) and passes it here, so changing the env
    reroutes that one agent without code changes. Empty string is treated
    as "use the route default" for ergonomic env handling.

    OpenAI and Anthropic fail over to each other once per call when the
    other is configured — see `_with_failover`. Gemini does not.
    """
    provider = (provider_override or settings.active_llm_provider).lower()
    if provider == "none":
        return None

    if provider == "gemini":
        if _breaker_open("gemini"):
            return None
        return gemini_chat_json(prompt, system=system, model=model, max_tokens=max_tokens)

    # Drop any provider-foreign model override so the route default
    # for the active provider is used.
    if not _model_matches_provider(model, provider):
        model = None

    return _with_failover(
        provider, _call_json,
        prompt=prompt, system=system, route=route, max_tokens=max_tokens, model=model,
    )


def chat_text(
    prompt: str,
    *,
    system: str = "",
    route: str = "cheap",
    max_tokens: int = 600,
    provider_override: str | None = None,
    model: str | None = None,
) -> str | None:
    """Same `model` and failover semantics as `chat_json`. Returns plain text or None."""
    provider = (provider_override or settings.active_llm_provider).lower()
    if provider == "none":
        return None

    if provider == "gemini":
        if _breaker_open("gemini"):
            return None
        return gemini_chat_text(prompt, system=system, model=model, max_tokens=max_tokens)

    # Drop any provider-foreign model override (see chat_json comment).
    if not _model_matches_provider(model, provider):
        model = None

    return _with_failover(
        provider, _call_text,
        prompt=prompt, system=system, route=route, max_tokens=max_tokens, model=model,
    )
