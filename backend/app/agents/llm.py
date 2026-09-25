"""LLM helper with provider selection (OpenAI + Anthropic) and model routing.

`route` selects between a strong (PM/critic synthesis) and cheap (extraction)
model. The provider is resolved per call via `settings.active_llm_provider`,
which honors `LLM_PROVIDER` (auto/openai/anthropic) and key presence.

When no LLM is configured, helpers return None and callers fall back to
deterministic stub findings.

Attribution (owner, 2026-09-25: "log which agent/llm model does what
action"). Every public entry (`chat_json`, `chat_text`, `gemini_chat_json`,
`gemini_chat_text`) takes `action=` from the registry in
`llm_attribution`, runs the attribution guard as its first statement, and
opens one *call* scope; each provider *attempt* inside it (a failover hop
is a second attempt, a skipped attempt still counts) writes exactly one
`llm_call_logs` row and one `app.llm.calls` line through `_record_usage`,
the single writer.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..config import settings
from . import llm_attribution as attribution
from .log_safety import log_safely, redact_unbounded

log = logging.getLogger(__name__)
# Dedicated logger for the per-attempt line so its level can be tuned
# without touching the rest of the LLM layer's logging.
call_log = logging.getLogger("app.llm.calls")

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
# Process label (web / worker) for rows and lines
# ---------------------------------------------------------------------------

_PROC: str | None = None
_DEFAULT_ORIGIN: str | None = None


def _proc() -> str:
    """This process's role, computed once: it cannot change while running."""
    global _PROC
    if _PROC is None:
        from ..runtime_role import process_role
        _PROC = process_role()
    return _PROC


def _default_origin() -> str:
    global _DEFAULT_ORIGIN
    if _DEFAULT_ORIGIN is None:
        from ..runtime_role import default_origin
        _DEFAULT_ORIGIN = default_origin()
    return _DEFAULT_ORIGIN


def _emit(logger: logging.Logger, level: int, line: str) -> None:
    """Log an operational line through the secret masks WITHOUT the 300-char
    truncation `log_safely` applies: these lines are built from whitelisted
    scalars and are useless cut in half."""
    logger.log(level, "%s", redact_unbounded(line))


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


def _breaker_line(state: str, provider: str, **extra: Any) -> None:
    """One WARNING per breaker TRANSITION (attribution critique #19): the
    trip used to be recorded only in CacheCostLog and the reset not at all.
    Breaker state is per process, so the line says which one."""
    pairs: list[tuple[str, Any]] = [("state", state), ("provider", provider)]
    pairs += list(extra.items())
    pairs.append(("proc", _proc()))
    _emit(log, logging.WARNING, "llm_breaker " + attribution.format_kv(pairs))


def _record_failure(provider: str) -> None:
    import time as _time
    _FAILURE_COUNTERS[provider] = _FAILURE_COUNTERS.get(provider, 0) + 1
    _FAILURE_LAST_AT[provider] = _time.time()
    count = _FAILURE_COUNTERS[provider]
    if count >= _BREAKER_THRESHOLD:
        try:
            from ..cache import log_cost
            log_cost(provider, "provider_failure", 0,
                     note=f"{provider} circuit breaker tripped at {count} failures")
        except Exception:  # pragma: no cover
            pass
    if count == _BREAKER_THRESHOLD:
        _breaker_line("open", provider, failures=count)


def _record_success(provider: str) -> None:
    was_open = _FAILURE_COUNTERS.get(provider, 0) >= _BREAKER_THRESHOLD
    _FAILURE_COUNTERS[provider] = 0
    _FAILURE_LAST_AT.pop(provider, None)
    if was_open:
        _breaker_line("reset", provider, reason="success")


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
        _breaker_line("reset", provider, reason="cooldown")
        return False
    return True


def breaker_open(provider: str) -> bool:
    """Public: is `provider`'s breaker open in this process? (The debate
    harness checks it before a pair failover.)"""
    return _breaker_open(provider)


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
# Long-lived loop threads never drain their context's event list (only the
# memo pipeline consumes it), so it is capped (design gap G18).
_FAILOVER_EVENTS_MAX = 256


def _failover_partner(provider: str) -> str | None:
    """The provider we may fail over to from `provider`, or None.

    None when failover is disabled, when `provider` has no partner (gemini
    stays a specialist path), when the partner has no key — a failover
    to an unconfigured provider would just be a second failure — and in
    demo-only mode, where no client is ever built (attribution critique
    #2: the hop used to go ahead there and log a failover to nothing).
    """
    if not settings.llm_failover_enabled:
        return None
    if _demo_only():
        return None
    partner = _FAILOVER_PARTNER.get(provider)
    if partner == "openai" and settings.has_openai:
        return partner
    if partner == "anthropic" and settings.has_anthropic:
        return partner
    return None


def failover_partner(provider: str) -> str | None:
    """Public form of `_failover_partner` for the debate harness."""
    return _failover_partner(provider)


def _record_failover(src: str, dst: str, reason: str, *,
                     from_model: str | None = None, to_model: str | None = None) -> None:
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
    if len(events) < _FAILOVER_EVENTS_MAX:
        events.append({"from": src, "to": dst, "reason": reason})
    # The legacy prefix is kept byte-for-byte so existing log searches still
    # match; the key=value suffix names the call, agent and both models.
    # No exception text: the wrappers already turned it into a category.
    att = _ATTEMPT.get() or {}
    ctx = _CALL_CONTEXT.get()
    suffix = attribution.format_kv([
        ("call", att.get("call_id")),
        ("agent", _resolve_agent(ctx, att.get("action") or ctx.get("action"))),
        ("action", att.get("action") or ctx.get("action")),
        ("from_model", from_model),
        ("to_model", to_model),
        ("run_id", ctx.get("run_id")),
        ("ticker", att.get("ticker") or ctx.get("ticker")),
    ])
    _emit(log, logging.WARNING, f"LLM failover from {src} to {dst} ({reason}) {suffix}")


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


_PROD_WARN_NOTED = False


def attribution_mode() -> str:
    """The attribution guard's EFFECTIVE mode.

    APP_ENV=production always runs "warn", whatever LLM_ATTRIBUTION_MODE
    says: strict raises inside chat_json, the memo pipeline's safe_call
    swallows the raise, and every memo would silently fall back to stub
    findings (attribution critique #10). Said once, at WARNING.
    """
    global _PROD_WARN_NOTED
    mode = settings.llm_attribution_mode
    if (settings.app_env or "").strip().lower() == "production" and mode != "warn":
        if not _PROD_WARN_NOTED:
            _PROD_WARN_NOTED = True
            log.warning(
                "LLM_ATTRIBUTION_MODE=%s ignored: production always runs the "
                "attribution guard in warn mode", mode,
            )
        return "warn"
    return mode


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
    default={
        "agent_name": "unknown", "run_id": None, "route": "", "user_id": None, "feature": None,
        "action": None, "role": None, "ticker": None, "job_id": None, "origin": None,
    },
)
# Failover events for the current context; `None` default (not `[]`) so a
# shared mutable default can't leak events across contexts.
_FAILOVER_EVENTS: contextvars.ContextVar[list[dict[str, str]] | None] = (
    contextvars.ContextVar("llm_failover_events", default=None)
)
# The current CALL (one public entry invocation): call_id, attempt number,
# action, ticker, the requested/sent models and why they differ, the
# failover link. Opened once by the outermost public entry; nested entries
# (chat_json -> gemini_chat_json -> gemini_chat_text) reuse it, so one call
# is guarded, logged and written once (attribution critique #17).
_ATTEMPT: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("llm_attempt", default=None)
)


class llm_call_context:
    """Context manager that tags subsequent llm.* calls with agent + run_id.

    Usage:
        with llm_call_context(agent_name="Sector Analyst", run_id=run_id):
            llm.chat_json(...)

    Only the fields a caller passes are layered on; everything else carries
    over from the enclosing context. `agent_name` defaults to None for that
    reason: its old default "unknown" was truthy, so a nested context that
    set only `static_prefix_chars` reset the agent, and every PM Synthesis
    row since 2026-09-19 was recorded as "unknown" (design gap G0).

    An UMBRELLA context (a route, a loop, a worker, a chat turn, a script)
    sets origin / job_id / run_id only, never agent_name: an agent named
    there would be credited with every specialist call nested under it
    (attribution critique #1; `llm_attribution.AGENT_CONTEXT_SITES`).
    """

    def __init__(self, *, agent_name: str | None = None, run_id: str | None = None,
                 route: str = "", user_id: int | None = None,
                 feature: str | None = None,
                 static_prefix_chars: int = 0,
                 action: str | None = None, role: str | None = None,
                 ticker: str | None = None, job_id: str | None = None,
                 origin: str | None = None) -> None:
        # `user_id` / `feature` (FEAT-002) attribute spend to the customer
        # and product feature that caused it; the worker sets them from
        # the RegenJob row, the chat route from the request principal.
        # `static_prefix_chars` tells `_anthropic_chat` that the first N
        # characters of the user prompt are byte-stable across calls (a
        # template) and may be cached; N must land right after a newline.
        self._values = {
            "agent_name": agent_name, "run_id": run_id, "route": route,
            "user_id": user_id, "feature": feature,
            "static_prefix_chars": int(static_prefix_chars or 0),
            "action": action, "role": role, "ticker": ticker,
            "job_id": job_id, "origin": origin,
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


@contextmanager
def _call_scope(entry: str, *, action: str | None, ticker: str | None,
                route: str, max_tokens: int) -> Iterator[dict[str, Any]]:
    """Open the call scope for a public entry, running the attribution
    guard first. A nested entry reuses the outer scope (one call = one
    guard evaluation, one row per attempt, one line per attempt)."""
    outer = _ATTEMPT.get()
    if outer is not None:
        if action and not outer.get("action"):
            outer["action"] = action
        if ticker and not outer.get("ticker"):
            outer["ticker"] = ticker
        yield outer
        return
    ctx_action = _CALL_CONTEXT.get().get("action")
    attributed = attribution.check(entry, action or ctx_action, mode=attribution_mode())
    att: dict[str, Any] = {
        "call_id": uuid.uuid4().hex,
        "attempt": 1,
        "entry": entry,
        "action": action,
        "ticker": ticker,
        "route": route,
        "max_tokens": max_tokens,
        "unattributed": not attributed,
        "call_cost_usd": 0.0,
    }
    token = _ATTEMPT.set(att)
    try:
        yield att
    finally:
        _ATTEMPT.reset(token)


def _resolve_agent(ctx: dict[str, Any], action: str | None) -> str:
    """Design §4.3 precedence: a specific context agent, else the action's
    registry agent, else a non-"unknown" umbrella name, else
    "unattributed"."""
    named = ctx.get("agent_name") or ""
    if named and named not in attribution.UMBRELLA_AGENTS:
        return str(named)
    spec = attribution.spec_for(action)
    if spec is not None:
        return spec.agent
    if named and named not in ("unknown", "unattributed"):
        return str(named)
    return attribution.UNATTRIBUTED


_FINISH_RE = re.compile(r";(?:stop_reason|finish_reason)=([a-z_]+)")


def _fit(column: str, value: Any) -> Any:
    """Truncate a string to its `llm_call_logs` column length. Postgres
    rejects an over-long value, the INSERT error is swallowed, and the whole
    row would be lost (attribution critique #6); SQLite would not notice."""
    if value is None or not isinstance(value, str):
        return value
    from ..models import LLMCallLog
    length = getattr(LLMCallLog.__table__.c[column].type, "length", None)
    return value[:length] if length else value


def _record_usage(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    duration_ms: int = 0,
    success: bool = True,
    error: str = "",
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    served_model: str | None = None,
    reasoning_tokens: int | None = None,
    finish_reason: str | None = None,
    refused: bool = False,
    grounded: bool | None = None,
    update_last_usage: bool = True,
) -> None:
    """The single writer: one `llm_call_logs` row and one `app.llm.calls`
    line per provider attempt (design §4.5).

    `model` is the name SENT (cost is priced by it; the price table is
    keyed by configured names); `served_model` is what the provider said
    served the request. A direct call with no call scope (unit tests, old
    callers) gets a fresh call_id and attempt 1.
    """
    att = _ATTEMPT.get()
    if att is None:
        att = {"call_id": uuid.uuid4().hex, "attempt": 1, "call_cost_usd": 0.0}
    ctx = _CALL_CONTEXT.get()
    action = att.get("action") or ctx.get("action")
    spec = attribution.spec_for(action)
    agent = _resolve_agent(ctx, action)
    role = ctx.get("role") or (spec.role if spec is not None else None)
    ticker = att.get("ticker") or ctx.get("ticker")
    route = att.get("route") or ctx.get("route") or ""
    origin = ctx.get("origin") or _default_origin()
    error = str(error or "")
    error_type = error.split(";", 1)[0][:64] if error else None
    if finish_reason is None and error:
        m = _FINISH_RE.search(error)
        finish_reason = m.group(1) if m else None
    skipped = bool(error_type and error_type.startswith("skipped:"))
    outcome = "skipped" if skipped else ("ok" if success else "error")
    n_in = int(input_tokens or 0)
    n_out = int(output_tokens or 0)
    n_read = int(cache_read_tokens or 0)
    n_write = int(cache_write_tokens or 0)
    cost: float | None
    try:
        from ..services.llm_metrics import estimate_cost_usd
        cost = float(estimate_cost_usd(
            provider, model, n_in, n_out,
            cache_read_tokens=n_read, cache_write_tokens=n_write,
        ))
    except Exception:  # pragma: no cover - pricing must never break a call
        cost = None
    att["error_type"] = error_type
    att["refused"] = bool(refused)
    att["call_cost_usd"] = float(att.get("call_cost_usd") or 0.0) + float(cost or 0.0)

    if update_last_usage and not skipped:
        # `total_tokens` stays input + output: for Anthropic `input_tokens` is
        # the uncached remainder, so cached tokens are priced separately
        # rather than summed into the figure the snapshot cache treats as
        # "spend". The keys after cache_write_tokens are additive.
        _USAGE_STATE.last = {
            "provider": provider,
            "model": model,
            "input_tokens": n_in,
            "output_tokens": n_out,
            "total_tokens": max(0, n_in) + max(0, n_out),
            "cache_read_tokens": n_read,
            "cache_write_tokens": n_write,
            "call_id": att.get("call_id"),
            "attempt": att.get("attempt"),
            "served_model": served_model,
            "cost_usd": cost,
            # Every attempt of this call so far, so a failed first attempt's
            # spend is not dropped (design gap G19; the learning judge).
            "call_cost_usd": att["call_cost_usd"],
            "refused": bool(refused),
        }

    fields = {
        "call": att.get("call_id"), "attempt": att.get("attempt"), "outcome": outcome,
        "agent": agent, "role": role, "action": action, "provider": provider,
        "model_requested": att.get("requested_model"), "model": model,
        "model_served": served_model, "resolution": att.get("model_resolution"),
        "failover_from": att.get("failover_from"), "failover_reason": att.get("failover_reason"),
        "effort": att.get("effort"), "run_id": ctx.get("run_id"), "ticker": ticker,
        "job": ctx.get("job_id"), "origin": origin, "feature": ctx.get("feature"),
        "route": route, "tokens_in": n_in, "tokens_out": n_out, "cache_read": n_read,
        "cache_write": n_write, "reasoning_tokens": reasoning_tokens,
        "max_tokens": att.get("max_tokens"), "finish": finish_reason,
        "cost_usd": f"{cost:.6f}" if cost is not None else None,
        "ms": int(duration_ms or 0), "error": error_type, "proc": _proc(),
    }
    # The line goes out BEFORE the DB write so a DB failure never loses it.
    level = logging.INFO if outcome == "ok" else logging.WARNING
    if level == logging.WARNING or settings.llm_call_log_enabled:
        _emit(call_log, level, attribution.format_call_line(fields))

    # Persist to the LLMCallLog audit table (Wave 1A). Lazy import to avoid
    # an import-time cycle (models → cache → ... ). DB failures must NEVER
    # break the LLM call path — wrap and swallow.
    try:
        from ..database import SessionLocal
        from ..models import LLMCallLog
        with SessionLocal() as db:
            # Lazy create so direct-import callers don't need init_db().
            LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
            db.add(LLMCallLog(
                run_id=_fit("run_id", ctx.get("run_id")),
                ticker=_fit("ticker", ticker),
                agent_name=_fit("agent_name", agent),
                provider=_fit("provider", provider),
                model=_fit("model", model),
                route=_fit("route", route),
                tokens_in=n_in,
                tokens_out=n_out,
                cache_read_tokens=n_read,
                cache_write_tokens=n_write,
                duration_ms=int(duration_ms or 0),
                success=bool(success),
                error=error[:500],
                user_id=ctx.get("user_id"),
                feature=_fit("feature", ctx.get("feature")),
                call_id=_fit("call_id", att.get("call_id")),
                attempt=att.get("attempt"),
                action=_fit("action", action),
                role=_fit("role", role),
                origin=_fit("origin", origin),
                job_id=_fit("job_id", ctx.get("job_id")),
                process_role=_fit("process_role", _proc()),
                requested_provider=_fit("requested_provider", att.get("requested_provider")),
                requested_model=_fit("requested_model", att.get("requested_model")),
                served_model=_fit("served_model", served_model),
                model_resolution=_fit("model_resolution", att.get("model_resolution")),
                failover_reason=_fit("failover_reason", att.get("failover_reason")),
                effort=_fit("effort", att.get("effort")),
                max_tokens=att.get("max_tokens"),
                reasoning_tokens=reasoning_tokens,
                finish_reason=_fit("finish_reason", finish_reason),
                error_type=_fit("error_type", error_type),
                cost_usd=cost,
                grounded=grounded,
            ))
            db.commit()
    except Exception as exc:  # pragma: no cover - defense in depth
        log_safely(log, "LLMCallLog persist failed", exc)


def _record_skip(provider: str, model: str, reason: str, *, grounded: bool | None = None) -> None:
    """A row and a WARNING line for an attempt that made no provider
    request (design gap G11): zero tokens, zero cost, and it never touches
    `last_usage()`, which describes the last request actually made."""
    _record_usage(provider, model, 0, 0, success=False, error=reason,
                  grounded=grounded, update_last_usage=False)


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


def _served(value: Any) -> str | None:
    """A provider-reported model id, or None (fakes and older SDKs)."""
    return value if isinstance(value, str) and value else None


def _usage_from_openai(resp: Any) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)


def _reasoning_from_openai(resp: Any) -> int | None:
    """OpenAI reasoning tokens — informational only: they are already
    inside completion_tokens, so they are never added to billed output
    (attribution critique #13)."""
    details = getattr(getattr(resp, "usage", None), "completion_tokens_details", None)
    value = getattr(details, "reasoning_tokens", None)
    return int(value) if isinstance(value, int) else None


def _usage_from_anthropic(msg: Any) -> tuple[int, int]:
    usage = getattr(msg, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def _cache_usage_from_anthropic(msg: Any) -> tuple[int, int]:
    """(cache_write_tokens, cache_read_tokens); both 0 when the response has none."""
    usage = getattr(msg, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    )


def _cache_usage_from_openai(resp: Any) -> int:
    """Cached prompt tokens (already counted inside `prompt_tokens`)."""
    details = getattr(getattr(resp, "usage", None), "prompt_tokens_details", None)
    return int(getattr(details, "cached_tokens", 0) or 0)


def _cacheable_system(text: str) -> str | list[dict[str, Any]]:
    """The string form is the API's shorthand for one text block, so the
    block form with a cache marker renders byte-identically."""
    if not settings.llm_prompt_caching_enabled:
        return text
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _user_content(user: str) -> str | list[dict[str, Any]]:
    """Split the user turn at the static prefix a call site declared via
    `llm_call_context(static_prefix_chars=…)`, marking the stable head as
    cacheable. Only splits right after a newline so the two blocks re-join
    to the original text; otherwise the prompt goes through unchanged."""
    if not settings.llm_prompt_caching_enabled:
        return user
    n = int(_CALL_CONTEXT.get().get("static_prefix_chars") or 0)
    if 0 < n < len(user) and user[n - 1] == "\n":
        return [
            {"type": "text", "text": user[:n], "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user[n:]},
        ]
    return user


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


def _finish_from_diagnostic(diagnostic: str) -> str | None:
    m = _FINISH_RE.search(diagnostic or "")
    return m.group(1) if m else None


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


def _gemini_resolution(caller_model: str | None) -> str:
    """Why the Gemini model sent differs (or not) from the requested one."""
    if caller_model and caller_model.strip():
        return "explicit"
    if settings.has_vertex and settings.vertex_model:
        return "vertex_override"
    return "route_default"


def gemini_chat_text(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    enable_search_grounding: bool = False,
    max_tokens: int = 800,
    _json_mode: bool = False,
    action: str | None = None,
    ticker: str | None = None,
) -> Any:
    """Lightweight Gemini text-completion wrapper.

    Search grounding is enabled by passing the `google_search` tool to the
    Generate Content API. The caller is responsible for filtering grounded
    sources against any allow/block list.
    """
    with _call_scope("gemini_chat_text", action=action, ticker=ticker,
                     route="", max_tokens=max_tokens) as att:
        from ..services.regen_lease import assert_current
        assert_current()
        from ..services.industry_lease import assert_current as assert_industry_current
        assert_industry_current()
        chosen_model = _resolve_gemini_model(model, settings.gemini_news_model)
        if not att.get("requested_provider"):
            att.update(
                requested_provider="gemini",
                requested_model=(model or "").strip() or settings.gemini_news_model,
                model_resolution=_gemini_resolution(model),
            )
        att["max_tokens"] = max_tokens
        grounded = bool(enable_search_grounding) or None
        if _breaker_open("gemini"):
            _record_skip("gemini", chosen_model, "skipped:breaker_open", grounded=grounded)
            return None
        client = _gemini_client()
        if client is None:
            return None
        return _gemini_generate(client, chosen_model, prompt, system=system,
                                enable_search_grounding=enable_search_grounding,
                                max_tokens=max_tokens, json_mode=_json_mode)


def _gemini_generate(client: Any, chosen_model: str, prompt: str, *, system: str,
                     enable_search_grounding: bool, max_tokens: int, json_mode: bool) -> Any:
    full_prompt = (system + "\n\n" + prompt).strip() if system else prompt
    import time as _time
    t0 = _time.perf_counter()
    in_tok = out_tok = 0
    received_response = False
    out = None
    error = ""
    finish_diagnostic = ""
    served = None
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
        served = _served(getattr(resp, "model_version", None))
        finish_diagnostic = _safe_finish_diagnostic("gemini", resp)
        text = getattr(resp, "text", None)
        out = _extract_json(text) if json_mode and text else (text or None)
        if out is None:
            error = "invalid_json_response" if text else "empty_response"
    except Exception as exc:  # pragma: no cover
        error = f"{'response_error' if received_response else 'provider_error'}:{type(exc).__name__}"
        out = None
    # JSON success is decided only after the existing recovery parser runs.
    # A failed parse still consumed the response's real tokens: one call, one row.
    _record_usage(
        "gemini", chosen_model, in_tok, out_tok,
        duration_ms=int((_time.perf_counter() - t0) * 1000),
        success=out is not None, error=error + finish_diagnostic if error else "",
        served_model=served, finish_reason=_finish_from_diagnostic(finish_diagnostic),
        grounded=bool(enable_search_grounding) or None,
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
    action: str | None = None,
    ticker: str | None = None,
) -> dict[str, Any] | None:
    """JSON-mode wrapper around `gemini_chat_text` — appends a 'JSON only'
    instruction and parses the result with the same `_extract_json` helper as
    the Anthropic branch.
    """
    with _call_scope("gemini_chat_json", action=action, ticker=ticker,
                     route="", max_tokens=max_tokens):
        from ..services.regen_lease import assert_current
        assert_current()
        from ..services.industry_lease import assert_current as assert_industry_current
        assert_industry_current()
        sys_with_json = (system + "\n\nReturn ONLY valid JSON, no prose.").strip()
        return gemini_chat_text(
            prompt, system=sys_with_json, model=model,
            enable_search_grounding=enable_search_grounding, max_tokens=max_tokens,
            _json_mode=True, action=action, ticker=ticker,
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
    in_tok = out_tok = cache_w = cache_r = 0
    received_response = False
    out = None
    error = ""
    finish_diagnostic = ""
    served = None
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": _cacheable_system(system or "You are a helpful assistant."),
            "messages": [{"role": "user", "content": _user_content(user)}],
        }
        if _anthropic_supports_custom_temp(model):
            kwargs["temperature"] = 0.3
        msg = client.messages.create(**kwargs)
        received_response = True
        # Capture real token usage for cost accounting (Phase C) + log row (Wave 1A).
        in_tok, out_tok = _usage_from_anthropic(msg)
        cache_w, cache_r = _cache_usage_from_anthropic(msg)
        served = _served(getattr(msg, "model", None))
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
        out = None
    # The per-attempt `app.llm.calls` line carries the failure category (and
    # replaces the old per-wrapper WARNING), so one attempt = one line.
    _record_usage(
        "anthropic", model, in_tok, out_tok,
        duration_ms=int((_time.perf_counter() - t0) * 1000),
        success=out is not None, error=error + finish_diagnostic if error else "",
        cache_read_tokens=cache_r, cache_write_tokens=cache_w,
        served_model=served, finish_reason=_finish_from_diagnostic(finish_diagnostic),
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
    in_tok = out_tok = cache_r = 0
    received_response = False
    out = None
    error = ""
    finish_diagnostic = ""
    served = None
    reasoning = None
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
        cache_r = _cache_usage_from_openai(resp)
        served = _served(getattr(resp, "model", None))
        reasoning = _reasoning_from_openai(resp)
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
        out = None
    _record_usage(
        "openai", model, in_tok, out_tok,
        duration_ms=int((_time.perf_counter() - t0) * 1000),
        success=out is not None, error=error + finish_diagnostic if error else "",
        cache_read_tokens=cache_r, served_model=served, reasoning_tokens=reasoning,
        finish_reason=_finish_from_diagnostic(finish_diagnostic),
    )
    return out


def _openai_chat_text(client: Any, *, model: str, system: str, user: str, max_tokens: int) -> str | None:
    import time as _time
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    t0 = _time.perf_counter()
    in_tok = out_tok = cache_r = 0
    received_response = False
    out = None
    error = ""
    finish_diagnostic = ""
    served = None
    reasoning = None
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            **_openai_token_kwarg(model, max_tokens),
        }
        if _openai_supports_custom_temp(model):
            kwargs["temperature"] = 0.3
        resp = client.chat.completions.create(**kwargs)
        received_response = True
        in_tok, out_tok = _usage_from_openai(resp)
        cache_r = _cache_usage_from_openai(resp)
        served = _served(getattr(resp, "model", None))
        reasoning = _reasoning_from_openai(resp)
        finish_diagnostic = _safe_finish_diagnostic("openai", resp)
        out = resp.choices[0].message.content or None
        if out is None:
            error = "empty_response"
    except Exception as exc:  # pragma: no cover
        # A category, like the other wrappers — never the provider's message,
        # which used to be stored here redacted but still up to 300 chars
        # (design gap G16).
        error = f"{'response_error' if received_response else 'provider_error'}:{type(exc).__name__}"
        out = None
    _record_usage(
        "openai", model, in_tok, out_tok,
        duration_ms=int((_time.perf_counter() - t0) * 1000),
        success=out is not None, error=error + finish_diagnostic if error else "",
        cache_read_tokens=cache_r, served_model=served, reasoning_tokens=reasoning,
        finish_reason=_finish_from_diagnostic(finish_diagnostic),
    )
    return out


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


class _NoClient:
    """Sentinel: the provider had no client, so no request was made. Kept
    distinct from None (a request that failed) so failover can tell
    "client unavailable" from "call failed" (attribution critique #2)."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<no client>"


_NO_CLIENT: Any = _NoClient()


def _has_key(provider: str) -> bool:
    return (provider == "openai" and settings.has_openai) or (
        provider == "anthropic" and settings.has_anthropic)


def _begin_request(att: dict[str, Any] | None, *, max_tokens: int) -> None:
    """Per-attempt facts `_record_usage` reads, reset before each request."""
    if att is None:
        return
    att["max_tokens"] = int(max_tokens)
    att["error_type"] = None
    att["refused"] = False


def _call_json(
    provider: str, *, prompt: str, system: str, route: str,
    max_tokens: int, model: str | None,
) -> Any:
    """One JSON-mode call against `provider`, feeding its breaker counters.

    Returns `_NO_CLIENT` when the provider has no client (no counter
    change), None when the call failed (failure recorded). `model` must
    already be vetted by `_model_matches_provider` for this provider — the
    failover hop passes None so it lands on the partner's own route default.
    """
    from ..services.regen_lease import assert_current
    assert_current()
    from ..services.industry_lease import assert_current as assert_industry_current
    assert_industry_current()
    att = _ATTEMPT.get()
    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return _NO_CLIENT
        chosen = (model or "").strip() or _model_for("anthropic", route)
        sys_with_json = (system + "\n\nReturn ONLY valid JSON, no prose.").strip()
        _begin_request(att, max_tokens=max_tokens)
        out = _anthropic_chat(
            client, model=chosen, system=sys_with_json, user=prompt,
            max_tokens=max_tokens, json_mode=True,
        )
        _breaker_after("anthropic", out)
        return out

    client = _openai_client()
    if client is None:
        return _NO_CLIENT
    chosen = (model or "").strip() or _model_for("openai", route)
    _begin_request(att, max_tokens=max_tokens)
    out = _openai_chat_json(client, model=chosen, system=system, user=prompt, max_tokens=max_tokens)
    _breaker_after("openai", out)
    return out


def _call_text(
    provider: str, *, prompt: str, system: str, route: str,
    max_tokens: int, model: str | None,
) -> Any:
    """Text twin of `_call_json`; same contract."""
    from ..services.regen_lease import assert_current
    assert_current()
    from ..services.industry_lease import assert_current as assert_industry_current
    assert_industry_current()
    att = _ATTEMPT.get()
    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return _NO_CLIENT
        chosen = (model or "").strip() or _model_for("anthropic", route)
        _begin_request(att, max_tokens=max_tokens)
        text = _anthropic_chat(client, model=chosen, system=system, user=prompt, max_tokens=max_tokens)
        _breaker_after("anthropic", text)
        return text

    client = _openai_client()
    if client is None:
        return _NO_CLIENT
    chosen = (model or "").strip() or _model_for("openai", route)
    _begin_request(att, max_tokens=max_tokens)
    text = _openai_chat_text(client, model=chosen, system=system, user=prompt, max_tokens=max_tokens)
    _breaker_after("openai", text)
    return text


def _breaker_after(provider: str, out: Any) -> None:
    if out is None:
        _record_failure(provider)
    else:
        _record_success(provider)


def _with_failover(provider: str, call: Any, **kwargs: Any) -> Any:
    """Run `call(provider, **kwargs)`, hopping to the partner provider once.

    The hop happens when `provider`'s breaker is already open, its client
    is unavailable, or the call returns None (the wrappers convert
    exceptions to None). It is skipped — returning None exactly as before
    failover existed — when failover is disabled, no partner is
    configured, or the partner's breaker is open too. The partner runs on
    its *own* route default (`model=None`) because the caller's model name
    belongs to the failed provider.

    Every attempt leaves a row: a skipped primary (open breaker, client
    unavailable) and a skipped partner (open breaker) are written as
    `skipped:<why>`, so attempt 2 is never orphaned (critique #2, G11).
    """
    att = _ATTEMPT.get()
    route = kwargs.get("route", "cheap")
    sent = (kwargs.get("model") or "").strip() or _model_for(provider, route)
    if _breaker_open(provider):
        reason = "breaker_open"
        _record_skip(provider, sent, "skipped:breaker_open")
    else:
        out = call(provider, **kwargs)
        if out is _NO_CLIENT:
            if _demo_only():
                # Configuration, not an outage (and every CI run): no row,
                # no hop — the partner has no client either.
                return _NO_CLIENT
            reason = "client_unavailable"
            if _has_key(provider):
                # A key but no client: the SDK failed to import or construct.
                # Written, so the partner's attempt 2 is never orphaned. With
                # no key at all it is configuration the routing line reports.
                _record_skip(provider, sent, "skipped:client_unavailable")
        elif out is not None:
            return out
        else:
            reason = "call_failed"

    partner = _failover_partner(provider)
    if partner is None:
        return None
    partner_model = _model_for(partner, route)
    if att is not None:
        att.update(attempt=2, failover_from=provider, failover_reason=reason,
                   model_resolution="failover_default")
    if _breaker_open(partner):
        log.debug("LLM failover from %s to %s skipped: partner breaker open", provider, partner)
        _record_skip(partner, partner_model, "skipped:partner_breaker_open")
        return None
    _record_failover(provider, partner, reason, from_model=sent, to_model=partner_model)
    return call(partner, **{**kwargs, "model": None})


def _prepare(provider: str, model: str | None, route: str) -> str | None:
    """Record the requested provider/model on the call scope and drop a
    provider-foreign model override (the route default is used instead)."""
    att = _ATTEMPT.get()
    literal = (model or "").strip()
    requested = literal or _model_for(provider, route)
    resolution = "explicit" if literal else "route_default"
    if not _model_matches_provider(model, provider):
        # Wave 9b's silent drop, now visible: the row says which name was
        # asked for and why another one was sent.
        log.debug("dropping a provider-foreign model override for %s", provider)
        resolution = "foreign_override_dropped"
        model = None
    if att is not None:
        att.update(requested_provider=provider, requested_model=requested,
                   model_resolution=resolution)
    return model


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
    action: str | None = None,
    ticker: str | None = None,
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

    `action` names what the call does (a key of
    `llm_attribution.ACTIONS`); `ticker` the company it is about.

    OpenAI and Anthropic fail over to each other once per call when the
    other is configured — see `_with_failover`. Gemini does not.
    """
    with _call_scope("chat_json", action=action, ticker=ticker, route=route,
                     max_tokens=max_tokens):
        provider = (provider_override or settings.active_llm_provider).lower()
        if provider == "none":
            return None
        if provider == "gemini":
            return gemini_chat_json(prompt, system=system, model=model, max_tokens=max_tokens,
                                    action=action, ticker=ticker)
        model = _prepare(provider, model, route)
        out = _with_failover(
            provider, _call_json,
            prompt=prompt, system=system, route=route, max_tokens=max_tokens, model=model,
        )
        return None if out is _NO_CLIENT else out


def chat_text(
    prompt: str,
    *,
    system: str = "",
    route: str = "cheap",
    max_tokens: int = 600,
    provider_override: str | None = None,
    model: str | None = None,
    action: str | None = None,
    ticker: str | None = None,
) -> str | None:
    """Same `model`, `action` and failover semantics as `chat_json`. Returns plain text or None."""
    with _call_scope("chat_text", action=action, ticker=ticker, route=route,
                     max_tokens=max_tokens):
        provider = (provider_override or settings.active_llm_provider).lower()
        if provider == "none":
            return None
        if provider == "gemini":
            return gemini_chat_text(prompt, system=system, model=model, max_tokens=max_tokens,
                                    action=action, ticker=ticker)
        model = _prepare(provider, model, route)
        out = _with_failover(
            provider, _call_text,
            prompt=prompt, system=system, route=route, max_tokens=max_tokens, model=model,
        )
        return None if out is _NO_CLIENT else out
