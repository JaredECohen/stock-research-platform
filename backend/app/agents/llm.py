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
from dataclasses import dataclass
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


def provider_of_model(model: str | None) -> str | None:
    """Which provider a model name belongs to, from its family prefix."""
    m = (model or "").lower().strip()
    if m.startswith(("claude-", "anthropic/")):
        return "anthropic"
    if m.startswith(("gpt-", "openai/", "text-embedding-")) or re.match(r"^o\d", m):
        return "openai"
    if m.startswith(("gemini-", "models/gemini", "publishers/google")):
        return "gemini"
    return None


_UNPRICED_WARNED: set[str] = set()


def unpriced_configured_models() -> list[str]:
    """Every model the LIVE settings name (defaults, config.env, Render env)
    that has no exact price row, each logged once at WARNING. The test on
    the price table covers code defaults only; a model set by environment
    at wave H would otherwise price at a provider-default guess silently."""
    from ..services import llm_metrics
    missing: list[str] = []
    for name in type(settings).model_fields:
        if not name.endswith("_model"):
            continue
        value = str(getattr(settings, name, "") or "").strip()
        if not value:
            continue
        provider = provider_of_model(value) or ""
        if llm_metrics.price_source(provider, value) == "model":
            continue
        missing.append(value)
        if value not in _UNPRICED_WARNED:
            _UNPRICED_WARNED.add(value)
            log.warning(
                "configured model %s (%s) has no price row in llm_metrics; its calls "
                "cost at a provider-default guess", attribution.sanitize(value), name.upper(),
            )
    return sorted(set(missing))


# ---------------------------------------------------------------------------
# Routing by action tier (integration plan P1, §2)
# ---------------------------------------------------------------------------
# Every call names an action; the action belongs to a tier; a CONFIGURED
# tier resolves to (provider, model, effort, failover). A blank tier is
# today's route/model logic exactly, so the code defaults change nothing and
# the owner's model migration (wave H) is a render.yaml change that can be
# rolled back per agent (LLM_ACTION_TIER_OVERRIDES) or wholesale.

# Model-based failover when a tier is configured (DEVPLAN 2026-09-25 items
# 2, 7): Opus 5.5 -> gpt-6-sol, Haiku 4.5 -> gpt-4.1-mini, and the GPT-6
# models back to Opus 5.5. A blank tier keeps today's partner-route default.
FAILOVER_MODEL_MAP: dict[str, tuple[str, str]] = {
    "claude-opus-5-5": ("openai", "gpt-6-sol"),
    "claude-haiku-4-5": ("openai", "gpt-4.1-mini"),
    "gpt-6-astra": ("anthropic", "claude-opus-5-5"),
    "gpt-6-sol": ("anthropic", "claude-opus-5-5"),
}

# Tiers whose resolved route `chat_json`/`chat_text` actually dispatch. The
# news tier is reported (it IS the Gemini news model) but its callers use
# the gemini entries directly; utility and embed are today's routes.
_DISPATCH_TIERS = frozenset({"research", "debate", "reviewer", "chat"})


@dataclass(frozen=True)
class ActionRoute:
    """Where one action runs. `configured=False` means today's legacy
    route: the call site's own provider/model/route decide, and the other
    fields only describe that legacy default where it is knowable."""

    action: str
    tier: str
    configured: bool
    provider: str | None = None
    model: str | None = None
    effort: str | None = None
    floor: int | None = None
    failover_provider: str | None = None
    failover_model: str | None = None
    failover_effort: str | None = None
    reason: str = ""

    @property
    def failover(self) -> tuple[str, str, str | None] | None:
        if self.failover_provider and self.failover_model:
            return (self.failover_provider, self.failover_model, self.failover_effort)
        return None

    def describe(self) -> str:
        if not self.configured:
            return f"legacy({self.reason})" if self.reason else "legacy"
        text = f"{self.provider}:{self.model}@{self.effort or '-'}"
        if self.failover:
            text += f" failover={self.failover_provider}:{self.failover_model}@{self.failover_effort or '-'}"
        return text


def tier_overrides() -> dict[str, str]:
    """LLM_ACTION_TIER_OVERRIDES as {action: tier}. An unknown action is an
    unresolvable request and raises (config.py checks only the shape)."""
    out: dict[str, str] = {}
    text = settings.llm_action_tier_overrides or ""
    for item in (p.strip() for p in text.split(",") if p.strip()):
        action, _, tier = item.partition(":")
        action, tier = action.strip(), tier.strip().lower()
        if action not in attribution.ACTIONS:
            raise ValueError(f"LLM_ACTION_TIER_OVERRIDES names an unknown action: {action!r}")
        out[action] = tier
    return out


def _effort_for(provider: str | None, model: str | None, effort: str | None) -> str | None:
    """The effort that would actually be sent to (provider, model)."""
    if not model:
        return None
    if provider == "anthropic":
        return _anthropic_effort(model, effort)
    if provider == "openai":
        return _openai_effort(model, effort)
    return None


def _route_floor(provider: str | None, model: str | None) -> int | None:
    if provider not in ("anthropic", "openai") or not model:
        return None
    floor = _effective_max_tokens(provider, model, 0, hop=False)
    return floor or None


def _legacy_route(action: str, tier: str, reason: str) -> ActionRoute:
    """Describe today's route for a non-configured tier: the active
    provider on the cheap route for utility actions (the only tier whose
    legacy route is fixed), with its partner's cheap default as failover."""
    provider = settings.active_llm_provider
    if tier == "embed":
        return ActionRoute(action, tier, False, "openai", "text-embedding-3-small", reason=reason)
    if tier != "utility" or provider not in _FAILOVER_PARTNER:
        return ActionRoute(action, tier, False, reason=reason)
    partner = _FAILOVER_PARTNER[provider]
    has_partner = _has_key(partner)
    return ActionRoute(
        action, tier, False, provider, _model_for(provider, "cheap"),
        failover_provider=partner if has_partner else None,
        failover_model=_model_for(partner, "cheap") if has_partner else None,
        reason=reason,
    )


def resolve_action_route(action: str) -> ActionRoute:
    """(provider, model, effort, floor, failover) for `action`.

    Raises ValueError for an unregistered action. A tier with no model
    configured, a model whose provider has no key, an override to
    "legacy", or an action that owns its route (`routed=False`) all
    resolve to `configured=False`: today's route/model logic.
    """
    spec = attribution.ACTIONS.get(action)
    if spec is None:
        raise ValueError(f"unregistered LLM action: {action!r}")
    tier = tier_overrides().get(action, spec.tier)
    if tier == "legacy":
        return _legacy_route(action, spec.tier, "override:legacy")
    if not spec.routed:
        # Its route is the call site's own (the legacy critic, the SDK memo
        # exchange), so nothing about it is describable from the tier.
        return ActionRoute(action, tier, False, reason="call site owns its route")
    effort_key = spec.effort_key or "default"
    fo_key = spec.failover_effort_key or effort_key

    provider: str | None
    model: str
    effort: str
    fo: tuple[str, str] | None
    fo_effort: str
    if tier == "research":
        model = settings.llm_research_model
        provider = provider_of_model(model)
        effort = settings.llm_pm_effort if effort_key == "pm" else settings.llm_default_effort
        fo_model = settings.llm_research_failover_model
        fo = ((provider_of_model(fo_model) or "", fo_model) if fo_model
              else FAILOVER_MODEL_MAP.get(model))
        fo_effort = (settings.llm_failover_pm_effort if fo_key == "pm"
                     else settings.llm_failover_default_effort)
    elif tier == "debate":
        provider = settings.debate_provider or provider_of_model(settings.debate_model)
        model = settings.debate_model
        if provider and not _model_matches_provider(model, provider):
            model = ""
        if provider and not model:
            # The design's blank-model rule: the provider's strong default.
            model = _model_for(provider, "strong")
        effort = (settings.debate_research_effort if effort_key == "debate_research"
                  else settings.debate_effort)
        fo = FAILOVER_MODEL_MAP.get(model)
        fo_effort = effort
    elif tier == "reviewer":
        provider = settings.risk_reviewer_provider or None
        model = settings.risk_reviewer_model
        if provider and not (model and _model_matches_provider(model, provider)):
            model = (settings.anthropic_critic_model if provider == "anthropic"
                     else settings.openai_strong_model)
        effort = (settings.review_recheck_effort if effort_key == "recheck"
                  else settings.risk_reviewer_effort)
        fo = FAILOVER_MODEL_MAP.get(model)
        fo_effort = effort
    elif tier == "chat":
        model = settings.chat_model
        provider = provider_of_model(model)
        effort = settings.chat_effort
        fo = FAILOVER_MODEL_MAP.get(model)
        fo_effort = effort
    elif tier == "news":
        model = _resolve_gemini_model(None, settings.gemini_news_model)
        return ActionRoute(action, tier, True, "gemini", model,
                           effort=_gemini_thinking(model)[0])
    else:  # utility, embed: today's routes
        return _legacy_route(action, tier, f"{tier} tier")

    if not model or provider not in ("anthropic", "openai"):
        return _legacy_route(action, tier, f"{tier} tier blank")
    if not _has_key(provider):
        return _legacy_route(action, tier, f"no {provider} key")
    fo_provider = fo_model_name = fo_sent_effort = None
    if fo is not None and fo[0] in ("anthropic", "openai") and _has_key(fo[0]):
        fo_provider, fo_model_name = fo
        # Effort is re-resolved for the failover (provider, model): the
        # debate's "high" stays "high" on gpt-6-sol, and a level the
        # partner does not accept is dropped (bull/bear critique #12).
        fo_sent_effort = _effort_for(fo_provider, fo_model_name, fo_effort or None)
    return ActionRoute(
        action, tier, True, provider, model,
        effort=_effort_for(provider, model, effort or None),
        floor=_route_floor(provider, model),
        failover_provider=fo_provider, failover_model=fo_model_name,
        failover_effort=fo_sent_effort,
    )


def _dispatch_route(action: str | None) -> ActionRoute | None:
    """The configured route a chat entry should use for `action`, or None
    for today's behaviour (no action, a legacy tier, or a Gemini tier)."""
    if not action or action not in attribution.ACTIONS:
        return None
    route = resolve_action_route(action)
    if not route.configured or route.tier not in _DISPATCH_TIERS:
        return None
    return route


def model_summary() -> dict[str, Any]:
    """Routing snapshot for the startup log and the status endpoint.

    Contains model names and booleans only — never key material. Beyond
    the per-role table: the tier routes (one representative action per
    tier, so the worker's and web's routing lines can be compared at a
    glance after a render.yaml change), the Gemini models, the chat
    surface, the failover map, and the attribution mode (design §4.10).
    """
    try:
        unpriced_configured_models()
    except Exception as exc:  # pragma: no cover - a summary must not fail on it
        log_safely(log, "price-row check failed", exc)
    tiers: dict[str, str] = {}
    for label, action in (("research", "analyst.sector"), ("research_pm", "pm.synthesis"),
                          ("debate", "debate.bull_open"), ("reviewer", "risk.review"),
                          ("chat", "chat.sdk_turn"), ("news", "news.search"),
                          ("utility", "chat.classify")):
        try:
            tiers[label] = resolve_action_route(action).describe()
        except Exception as exc:  # pragma: no cover - e.g. a bad override
            tiers[label] = f"error:{type(exc).__name__}"
    gemini = {
        "news": _resolve_gemini_model(None, settings.gemini_news_model),
        "social": _resolve_gemini_model(None, settings.gemini_social_model),
        "longdoc": _resolve_gemini_model(None, settings.gemini_longdoc_model),
        "backend": "vertex" if settings.has_vertex else ("api" if settings.gemini_api_key else "off"),
    }
    chat_route = resolve_action_route("chat.sdk_turn")
    chat = (f"sdk:{chat_route.model if chat_route.configured else resolve_role_model('pm', 'openai')}"
            if settings.chat_agents_sdk else f"legacy:{_model_for(settings.active_llm_provider, 'strong')}")
    return {
        "active_provider": settings.active_llm_provider,
        "provider_choice": settings.llm_provider,
        "role_models": {role: resolve_role_model(role) for role in _ROLE_SETTINGS},
        "configured": {
            "openai": settings.has_openai,
            "anthropic": settings.has_anthropic,
            "gemini": settings.has_gemini,
        },
        "tiers": tiers,
        "gemini": gemini,
        "chat": chat,
        "failover_map": {k: f"{p}:{m}" for k, (p, m) in FAILOVER_MODEL_MAP.items()},
        "attribution_mode": attribution_mode(),
        "reviewer_mode": settings.reviewer_mode,
        "debate_mode": settings.debate_mode,
    }


def _configured_model_names() -> dict[str, set[str]]:
    """Every model the live settings can send, by provider."""
    names: dict[str, set[str]] = {"anthropic": set(), "openai": set(), "gemini": set()}
    for field in type(settings).model_fields:
        if not field.endswith("_model"):
            continue
        value = str(getattr(settings, field, "") or "").strip()
        provider = provider_of_model(value)
        if value and provider in names:
            names[provider].add(value)
    # The failover targets a configured model can reach are sent too.
    for configured in [m for group in names.values() for m in group]:
        target = FAILOVER_MODEL_MAP.get(configured)
        if target is not None:
            names[target[0]].add(target[1])
    return names


def _listed_ids(client: Any) -> set[str]:
    ids: set[str] = set()
    for item in client.models.list():
        raw = getattr(item, "id", None) or getattr(item, "name", None) or ""
        if isinstance(raw, str) and raw:
            ids.add(raw.split("/", 1)[1] if raw.startswith("models/") else raw)
    return ids


def model_access_report() -> dict[str, str]:
    """Can this deployment's keys reach every configured model?

    Uses each provider's `models.list` ONLY — never a generation, so it
    costs nothing and can run at startup (plan §8.1: the owner confirms
    model access before wave H from this line). A dated listing
    (`claude-haiku-4-5-20251001`) satisfies its alias. Logs one line:
    `model_access claude-opus-5-5=ok gpt-6-sol=missing ...`.
    """
    factories = {"anthropic": _anthropic_client, "openai": _openai_client, "gemini": _gemini_client}
    report: dict[str, str] = {}
    for provider, models in _configured_model_names().items():
        if not models:
            continue
        try:
            client = factories[provider]()
        except Exception as exc:  # pragma: no cover - factories already swallow
            client = None
            log_safely(log, f"model_access: {provider} client failed", exc)
        if client is None:
            for model in models:
                report[model] = "unchecked:no_client"
            continue
        try:
            listed = _listed_ids(client)
        except Exception as exc:
            for model in models:
                report[model] = f"unchecked:{type(exc).__name__}"
            continue
        for model in models:
            hit = model in listed or any(i.startswith(model + "-") for i in listed)
            report[model] = "ok" if hit else "missing"
    line = "model_access " + attribution.format_kv(sorted(report.items()))
    level = logging.WARNING if any(v == "missing" for v in report.values()) else logging.INFO
    _emit(log, level, line)
    return report


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
    if provider == "openai" and (model or "").lower().startswith("gpt-6"):
        from ..services.llm_metrics import GPT6_LONG_CONTEXT_TOKENS
        if n_in > GPT6_LONG_CONTEXT_TOKENS:
            # GPT-6 bills the WHOLE request at 2x input / 1.5x output above
            # 272K input tokens; the estimate above does not, so say so.
            log.warning(
                "GPT-6 long-context request: %d input tokens > %d on %s (billed at the "
                "long-context rate; cost_usd understates it)",
                n_in, GPT6_LONG_CONTEXT_TOKENS, attribution.sanitize(model),
            )
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
    """(billed input, billed output) for a Gemini response.

    google-genai 2.22 `usage_metadata`: `prompt_token_count` +
    `tool_use_prompt_token_count` (the grounding tool's own prompt) is
    input; `candidates_token_count` + `thoughts_token_count` is output,
    because thoughts bill as output and candidates leave them out (design
    gap G13; news trace N3(c)). Without this, every Gemini row understated
    its cost once thinking was on.
    """
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return 0, 0
    return (
        int(getattr(meta, "prompt_token_count", 0) or 0)
        + int(getattr(meta, "tool_use_prompt_token_count", 0) or 0),
        int(getattr(meta, "candidates_token_count", 0) or 0)
        + int(getattr(meta, "thoughts_token_count", 0) or 0),
    )


def _gemini_extra_usage(resp: Any) -> tuple[int | None, int]:
    """(thoughts, cached prompt tokens): thoughts for the row's
    `reasoning_tokens` (already inside billed output), cached tokens priced
    at the model's cache rate (they are inside `prompt_token_count`)."""
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return None, 0
    thoughts = getattr(meta, "thoughts_token_count", None)
    cached = getattr(meta, "cached_content_token_count", None)
    return (int(thoughts) if isinstance(thoughts, int) else None,
            int(cached) if isinstance(cached, int) else 0)


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
            return _genai.Client(api_key=settings.gemini_api_key,
                                 http_options=_gemini_http_options())
        return None
    except Exception as exc:  # pragma: no cover
        log_safely(log, "Gemini client init failed", exc)
        return None


# google-genai 2.22's `HttpOptions.timeout` is in MILLISECONDS (the news
# trace critique caught a spec that said 30 = seconds). Unbounded, one hung
# grounded call held a monitoring thread indefinitely.
GEMINI_TIMEOUT_MS = 30_000


def _gemini_http_options() -> Any:
    try:
        from google.genai import types  # type: ignore
        return types.HttpOptions(timeout=GEMINI_TIMEOUT_MS)
    except Exception:  # pragma: no cover - older SDKs accept the dict form
        return {"timeout": GEMINI_TIMEOUT_MS}


_GEMINI_3_FLASH_RE = re.compile(r"^gemini-3\.[7-9]-flash(?!-lite)")


def _gemini_thinking(model: str) -> tuple[str | None, dict[str, Any] | None]:
    """(effort label for the row, `thinking_config`) per Gemini model.

    Explicit per model (model research 2026-09-25; owner item 7):
      - 3.5-flash-lite: thinkingLevel "minimal" (the news model);
      - 3.7/3.8-flash: "low" — never "minimal", which they reject;
      - 3.1-pro-preview: "low";
      - 2.5-flash: thinking_budget=0, kept only for an explicit override
        (new keys cannot call it any more);
      - anything else (2.5-pro via VERTEX_MODEL): nothing sent — Pro cannot
        disable thinking.
    """
    m = (model or "").lower().strip()
    if m.startswith("gemini-3.5-flash-lite"):
        return "minimal", {"thinking_level": "MINIMAL"}
    if _GEMINI_3_FLASH_RE.match(m):
        return "low", {"thinking_level": "LOW"}
    if m.startswith("gemini-3.1-pro"):
        return "low", {"thinking_level": "LOW"}
    if m.startswith("gemini-2.5-flash"):
        return "budget0", {"thinking_budget": 0}
    return None, None


def _grounding_cap_reached() -> bool:
    """True when today's (UTC) successful grounded Gemini calls reached
    GEMINI_GROUNDED_MAX_PER_DAY. Counted from llm_call_logs, not a
    module-level counter, so the web and worker processes share one cap.
    A DB error fails OPEN (logged): losing news is worse than a few
    over-cap grounded calls, which cost cents."""
    cap = int(settings.gemini_grounded_max_per_day)
    if cap <= 0:
        return True
    try:
        from datetime import datetime

        from sqlalchemy import func, select

        from ..database import SessionLocal
        from ..models import LLMCallLog
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with SessionLocal() as db:
            LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)
            used = db.execute(
                select(func.count()).select_from(LLMCallLog).where(
                    LLMCallLog.grounded.is_(True),
                    LLMCallLog.success.is_(True),
                    LLMCallLog.generated_at >= today,
                )
            ).scalar() or 0
        return int(used) >= cap
    except Exception as exc:  # pragma: no cover - defense in depth
        log_safely(log, "Gemini grounding-cap count failed; allowing the call", exc)
        return False


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
        att["effort"] = _gemini_thinking(chosen_model)[0]
        grounded = bool(enable_search_grounding) or None
        if _breaker_open("gemini"):
            _record_skip("gemini", chosen_model, "skipped:breaker_open", grounded=grounded)
            return None
        client = _gemini_client()
        if client is None:
            return None
        if enable_search_grounding and _grounding_cap_reached():
            # Over the daily grounded budget: a skip row the news loop can
            # count, and the caller falls back to the provider feed.
            _record_skip("gemini", chosen_model, "skipped:grounding_cap", grounded=True)
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
    thoughts = None
    cached = 0
    try:
        # Build config dynamically — different google-genai versions accept
        # slightly different shapes. We err on the side of being permissive.
        config: dict[str, Any] = {"temperature": 0.3, "max_output_tokens": max_tokens}
        thinking = _gemini_thinking(chosen_model)[1]
        if thinking is not None:
            config["thinking_config"] = thinking
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
        thoughts, cached = _gemini_extra_usage(resp)
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
        reasoning_tokens=thoughts, cache_read_tokens=cached,
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
    """Only the legacy Claude families still take `temperature`.

    Newer Anthropic models reject sampling parameters with HTTP 400
    (observed 2026-05-30 on claude-opus-4-7 and claude-haiku-4-5; the
    claude-api reference lists them as removed on Opus 4.7+, Sonnet 5,
    Fable and Opus 5/5.5). The old check was a DENYlist on "-4-", so every
    newer family (`claude-opus-5-5`, `claude-sonnet-5`, `claude-fable-5-1`)
    was sent `temperature=0.3` and 400'd. An allowlist fails safe for the
    next family instead. Non-Claude names are not this function's business.
    """
    m = (model or "").lower().strip()
    if not m.startswith("claude-"):
        return True
    return m.startswith(("claude-3", "claude-2", "claude-instant"))


# Anthropic models that accept `output_config.effort` (the claude-api
# reference: Opus 4.5+, Sonnet 4.6+, Fable/Mythos; NOT Haiku 4.5 or Sonnet
# 4.5, where it is an error).
_ANTHROPIC_EFFORT_RE = re.compile(
    r"^claude-(opus-4-[5-9]|opus-[5-9]|sonnet-4-6|sonnet-[5-9]|fable-|mythos-)"
)
_ANTHROPIC_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
# Always-thinking Anthropic families: thinking cannot be disabled (or is on
# by default) and counts against max_tokens, so they get the thinking floor
# (bull/bear critique #12 added sonnet-5).
_ALWAYS_THINKING = ("claude-opus-5", "claude-fable-", "claude-sonnet-5", "claude-mythos-")
# Opus 5.5's API default effort is "medium" (one below Opus 5's "high");
# sent explicitly so the row says what ran rather than relying on a default.
_EXPLICIT_DEFAULT_EFFORT = {"claude-opus-5-5": "medium"}


def _anthropic_effort(model: str, effort: str | None) -> str | None:
    """The effort to send in `output_config`, or None to send nothing."""
    m = (model or "").lower().strip()
    if not _ANTHROPIC_EFFORT_RE.match(m):
        if effort:
            log.debug("effort %s not sent: %s does not accept it", effort, m)
        return None
    if not effort:
        for prefix, default in _EXPLICIT_DEFAULT_EFFORT.items():
            if m.startswith(prefix):
                return default
        return None
    e = effort.strip().lower()
    if e not in _ANTHROPIC_EFFORTS:
        # none/minimal are OpenAI levels; Anthropic's lowest is "low".
        log.debug("effort %s not sent: not an Anthropic effort level", e)
        return None
    return e


# Refusal categories Opus 5.5's safety classifiers report in
# `stop_details.category` (anthropic 0.125 `RefusalStopDetails`); anything
# else is recorded as "other" so no free text reaches a row.
_REFUSAL_CATEGORIES = frozenset({"cyber", "bio", "frontier_llm", "reasoning_extraction", "general_harms"})


def _refusal_category(msg: Any) -> str | None:
    """`refusal:<category>` when the response is a refusal, else None.

    A refusal is HTTP 200 with `stop_reason == "refusal"`: not an outage,
    and retrying the same route refuses again (bull/bear critique #11).
    """
    if getattr(msg, "stop_reason", None) != "refusal":
        return None
    category = getattr(getattr(msg, "stop_details", None), "category", None)
    if not isinstance(category, str) or not category:
        return "refusal:unspecified"
    return f"refusal:{category if category in _REFUSAL_CATEGORIES else 'other'}"


def _anthropic_chat(
    client: Any, *, model: str, system: str, user: str, max_tokens: int,
    json_mode: bool = False, effort: str | None = None,
) -> Any:
    import time as _time
    t0 = _time.perf_counter()
    in_tok = out_tok = cache_w = cache_r = 0
    received_response = False
    out = None
    error = ""
    finish_diagnostic = ""
    served = None
    refused = False
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": _cacheable_system(system or "You are a helpful assistant."),
            "messages": [{"role": "user", "content": _user_content(user)}],
        }
        if _anthropic_supports_custom_temp(model):
            kwargs["temperature"] = 0.3
        # No `thinking` param is ever sent: Opus 5.5 cannot disable thinking
        # (a 400 at every effort level) and runs adaptive when it is omitted.
        # Effort is the only depth control, and only where accepted.
        if effort:
            kwargs["output_config"] = {"effort": effort}
        msg = client.messages.create(**kwargs)
        received_response = True
        # Capture real token usage for cost accounting (Phase C) + log row (Wave 1A).
        in_tok, out_tok = _usage_from_anthropic(msg)
        cache_w, cache_r = _cache_usage_from_anthropic(msg)
        served = _served(getattr(msg, "model", None))
        finish_diagnostic = _safe_finish_diagnostic("anthropic", msg)
        refusal = _refusal_category(msg)
        if refusal is not None:
            refused = True
            error = refusal
        else:
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
        refused=refused,
    )
    return out


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

_OPENAI_GPT_RE = re.compile(r"^gpt-(\d+)")
_OPENAI_O_SERIES_RE = re.compile(r"^o\d")


def _openai_is_reasoning(model: str) -> bool:
    """gpt-N with N >= 5 and the o-series are reasoning models.

    The old checks were a prefix list (`gpt-5`, `o1`, `o3`, `o4`), so
    `gpt-6-sol` / `gpt-6-astra` got `max_tokens` and `temperature=0.3`,
    both of which GPT-6 rejects (model research 2026-09-25; DEVPLAN item 7).
    """
    m = (model or "").lower().strip()
    g = _OPENAI_GPT_RE.match(m)
    if g:
        return int(g.group(1)) >= 5
    return bool(_OPENAI_O_SERIES_RE.match(m))


def _openai_token_kwarg(model: str, n: int) -> dict[str, int]:
    """Return the correct max-output-tokens kwarg for the OpenAI model family.

    GPT-5+ (GPT-6 included) and the o-series reasoning models (o1, o3, o4, …)
    reject `max_tokens` and require `max_completion_tokens`. Older /
    non-reasoning chat models (gpt-4.1, gpt-4o, gpt-3.5, …) still take
    `max_tokens`. Verified empirically against gpt-5.4, gpt-5.5, and
    gpt-4.1-mini on 2026-05-02; the new-name convention also covers o1 / o3
    reasoning models which use the same API contract.
    """
    if _openai_is_reasoning(model):
        return {"max_completion_tokens": int(n)}
    return {"max_tokens": int(n)}


def _openai_supports_custom_temp(model: str) -> bool:
    """GPT-5+ and o-series reasoning models reject `temperature` other
    than the default (1). Older chat models accept it. Verified
    empirically 2026-05-03 — gpt-5.5 returns 400 on temperature=0.3."""
    return not _openai_is_reasoning(model)


def _openai_effort(model: str, effort: str | None) -> str | None:
    """`reasoning_effort` to send, or None. Reasoning models only (gpt-4.1
    would 400). gpt-6-astra rejects "none" — an unresolvable request, so it
    raises rather than silently sending something else."""
    if not effort:
        return None
    m = (model or "").lower().strip()
    if not _openai_is_reasoning(m):
        log.debug("effort %s not sent: %s is not a reasoning model", effort, m)
        return None
    e = effort.strip().lower()
    if m.startswith("gpt-6-astra") and e == "none":
        raise ValueError("gpt-6-astra rejects reasoning_effort='none'")
    return e


def _openai_response_format(schema: dict[str, Any] | None) -> dict[str, Any]:
    """`json_object` by default; strict `json_schema` when the caller
    passes a schema (the research run recommended strict json_schema over
    json_object for GPT-6, and the reviewer relies on it). `schema` is a
    JSON Schema, optionally wrapped as {"name": ..., "schema": {...}}."""
    if not schema:
        return {"type": "json_object"}
    if "schema" in schema and isinstance(schema.get("schema"), dict):
        name = str(schema.get("name") or "response")
        body = schema["schema"]
    else:
        name = str(schema.get("title") or "response")
        body = schema
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "response"
    return {"type": "json_schema", "json_schema": {"name": name, "schema": body, "strict": True}}


def _openai_chat_json(client: Any, *, model: str, system: str, user: str, max_tokens: int,
                      effort: str | None = None, schema: dict[str, Any] | None = None,
                      ) -> dict[str, Any] | None:
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
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": _openai_response_format(schema),
            **_openai_token_kwarg(model, max_tokens),
        }
        if _openai_supports_custom_temp(model):
            kwargs["temperature"] = 0.3
        if effort:
            kwargs["reasoning_effort"] = effort
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


def _openai_chat_text(client: Any, *, model: str, system: str, user: str, max_tokens: int,
                      effort: str | None = None) -> str | None:
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
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **_openai_token_kwarg(model, max_tokens),
        }
        if _openai_supports_custom_temp(model):
            kwargs["temperature"] = 0.3
        if effort:
            kwargs["reasoning_effort"] = effort
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


def _begin_request(att: dict[str, Any] | None, *, max_tokens: int,
                   effort: str | None = None) -> None:
    """Per-attempt facts `_record_usage` reads, reset before each request."""
    if att is None:
        return
    att["max_tokens"] = int(max_tokens)
    att["effort"] = effort
    att["error_type"] = None
    att["refused"] = False


def _is_failover_hop() -> bool:
    att = _ATTEMPT.get()
    return bool(att and int(att.get("attempt") or 1) > 1)


# GPT-6 needs room for reasoning before any visible output ("reserve >=25k
# tokens", model research 2026-09-25). The same floor covers gpt-5.x on a
# failover hop: that is where the critic's 1,600-token default ran out
# inside reasoning and returned nothing (bullish-skew F5; the bull/bear
# critique's missing item).
_OPENAI_REASONING_FLOOR = 25_000


def _effective_max_tokens(provider: str, model: str, requested: int, *, hop: bool) -> int:
    """The max_tokens actually sent for (provider, model).

    - Always-thinking Anthropic models: at least LLM_THINKING_MAX_TOKENS_FLOOR,
      because thinking counts against max_tokens.
    - Every Anthropic request, failover hops included: at most
      LLM_ANTHROPIC_NONSTREAM_MAX_TOKENS. The SDK raises before sending a
      non-streaming request whose max_tokens implies >10 minutes (~21.3k),
      and `_with_failover` used to forward a 25k reviewer budget verbatim
      (bull/bear critique #5).
    - GPT-6 always, gpt-5.x on a failover hop: at least 25,000.

    Today's models on their primary route (claude-opus-4-8, Haiku 4.5,
    gpt-5.5, gpt-4.1-mini) are unchanged.
    """
    n = int(requested)
    m = (model or "").lower().strip()
    if provider == "anthropic":
        if m.startswith(_ALWAYS_THINKING):
            n = max(n, int(settings.llm_thinking_max_tokens_floor))
        return min(n, int(settings.llm_anthropic_nonstream_max_tokens))
    if provider == "openai":
        if m.startswith("gpt-6") or (hop and m.startswith("gpt-5")):
            return max(n, _OPENAI_REASONING_FLOOR)
    return n


# Content outcomes: the provider answered, the answer was unusable. Not a
# health signal when the caller owns its retry policy (failover=False).
_CONTENT_FAILURES = frozenset({"invalid_json_response", "empty_response"})


def _breaker_after(provider: str, out: Any, *, failover: bool = True) -> None:
    """Feed the process-wide breaker from one attempt's outcome.

    Breaker isolation (bull/bear critique #4): a `failover=False` caller
    (the debate harness) retries and fails over on its own, and a debate
    advocate's truncated or unparseable output must not open the breaker
    the PM relies on moments later. Those calls count transport errors
    only. A refusal is a content decision, never provider health, so it
    counts for nobody.
    """
    if out is not None:
        _record_success(provider)
        return
    att = _ATTEMPT.get() or {}
    error_type = att.get("error_type") or ""
    if error_type.startswith("refusal"):
        return
    if not failover and error_type in _CONTENT_FAILURES:
        return
    _record_failure(provider)


def _call_json(
    provider: str, *, prompt: str, system: str, route: str,
    max_tokens: int, model: str | None,
    effort: str | None = None, schema: dict[str, Any] | None = None,
    failover: bool = True,
) -> Any:
    """One JSON-mode call against `provider`, feeding its breaker counters.

    Returns `_NO_CLIENT` when the provider has no client (no counter
    change), None when the call failed (failure recorded). `model` must
    already be vetted by `_model_matches_provider` for this provider — the
    failover hop passes None so it lands on the partner's own route default.
    `effort` is re-resolved for the model actually sent (dropped where the
    model does not accept it); `effort`/`schema` reach the wrappers only
    when set, so today's requests are byte-identical.
    """
    from ..services.regen_lease import assert_current
    assert_current()
    from ..services.industry_lease import assert_current as assert_industry_current
    assert_industry_current()
    att = _ATTEMPT.get()
    hop = _is_failover_hop()
    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return _NO_CLIENT
        chosen = (model or "").strip() or _model_for("anthropic", route)
        sys_with_json = (system + "\n\nReturn ONLY valid JSON, no prose.").strip()
        sent_effort = _anthropic_effort(chosen, effort)
        n = _effective_max_tokens("anthropic", chosen, max_tokens, hop=hop)
        _begin_request(att, max_tokens=n, effort=sent_effort)
        extra: dict[str, Any] = {"effort": sent_effort} if sent_effort else {}
        out = _anthropic_chat(
            client, model=chosen, system=sys_with_json, user=prompt,
            max_tokens=n, json_mode=True, **extra,
        )
        _breaker_after("anthropic", out, failover=failover)
        return out

    client = _openai_client()
    if client is None:
        return _NO_CLIENT
    chosen = (model or "").strip() or _model_for("openai", route)
    sent_effort = _openai_effort(chosen, effort)
    n = _effective_max_tokens("openai", chosen, max_tokens, hop=hop)
    _begin_request(att, max_tokens=n, effort=sent_effort)
    extra = {}
    if sent_effort:
        extra["effort"] = sent_effort
    if schema:
        extra["schema"] = schema
    out = _openai_chat_json(client, model=chosen, system=system, user=prompt, max_tokens=n, **extra)
    _breaker_after("openai", out, failover=failover)
    return out


def _call_text(
    provider: str, *, prompt: str, system: str, route: str,
    max_tokens: int, model: str | None,
    effort: str | None = None, schema: dict[str, Any] | None = None,
    failover: bool = True,
) -> Any:
    """Text twin of `_call_json`; same contract (`schema` is ignored)."""
    from ..services.regen_lease import assert_current
    assert_current()
    from ..services.industry_lease import assert_current as assert_industry_current
    assert_industry_current()
    att = _ATTEMPT.get()
    hop = _is_failover_hop()
    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return _NO_CLIENT
        chosen = (model or "").strip() or _model_for("anthropic", route)
        sent_effort = _anthropic_effort(chosen, effort)
        n = _effective_max_tokens("anthropic", chosen, max_tokens, hop=hop)
        _begin_request(att, max_tokens=n, effort=sent_effort)
        extra: dict[str, Any] = {"effort": sent_effort} if sent_effort else {}
        text = _anthropic_chat(client, model=chosen, system=system, user=prompt,
                               max_tokens=n, **extra)
        _breaker_after("anthropic", text, failover=failover)
        return text

    client = _openai_client()
    if client is None:
        return _NO_CLIENT
    chosen = (model or "").strip() or _model_for("openai", route)
    sent_effort = _openai_effort(chosen, effort)
    n = _effective_max_tokens("openai", chosen, max_tokens, hop=hop)
    _begin_request(att, max_tokens=n, effort=sent_effort)
    extra = {"effort": sent_effort} if sent_effort else {}
    text = _openai_chat_text(client, model=chosen, system=system, user=prompt,
                             max_tokens=n, **extra)
    _breaker_after("openai", text, failover=failover)
    return text


def _failover_hop(provider: str, failover_route: tuple[str, str, str | None] | None,
                  effort: str | None) -> tuple[str, str | None, str | None, str] | None:
    """(partner, partner model or None, effort, resolution) or None.

    A configured tier fails over by MODEL (Opus 5.5 -> gpt-6-sol, …) with
    its own failover effort; otherwise today's rule: the partner provider
    on its own route default, carrying the caller's effort (re-resolved for
    whatever model that is).
    """
    if failover_route is not None:
        partner, model, fo_effort = failover_route
        if not settings.llm_failover_enabled or _demo_only() or not _has_key(partner):
            return None
        return partner, model, fo_effort, "failover_mapped"
    legacy = _failover_partner(provider)
    if legacy is None:
        return None
    return legacy, None, effort, "failover_default"


def _with_failover(provider: str, call: Any, *, failover: bool = True,
                   failover_route: tuple[str, str, str | None] | None = None,
                   **kwargs: Any) -> Any:
    """Run `call(provider, **kwargs)`, hopping to the partner provider once.

    The hop happens when `provider`'s breaker is already open, its client
    is unavailable, or the call returns None (the wrappers convert
    exceptions to None). It is skipped — returning None exactly as before
    failover existed — when failover is disabled, no partner is
    configured, or the partner's breaker is open too. The partner runs on
    its *own* route default (`model=None`) because the caller's model name
    belongs to the failed provider.

    `failover=False` (the debate pair, which fails over as a PAIR on its
    own) returns None after the primary attempt: no hop, and content
    failures do not feed the shared breaker (`_breaker_after`).

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
        out = call(provider, failover=failover, **kwargs)
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
            reason = "refused" if (att or {}).get("refused") else "call_failed"

    if not failover:
        return None
    hop = _failover_hop(provider, failover_route, kwargs.get("effort"))
    if hop is None:
        return None
    partner, mapped_model, partner_effort, resolution = hop
    partner_model = mapped_model or _model_for(partner, route)
    if att is not None:
        att.update(attempt=2, failover_from=provider, failover_reason=reason,
                   model_resolution=resolution)
    if _breaker_open(partner):
        log.debug("LLM failover from %s to %s skipped: partner breaker open", provider, partner)
        _record_skip(partner, partner_model, "skipped:partner_breaker_open")
        return None
    _record_failover(provider, partner, reason, from_model=sent, to_model=partner_model)
    return call(partner, failover=failover,
                **{**kwargs, "model": mapped_model, "effort": partner_effort})


def _tier_request(action: str | None, provider: str, model: str | None, effort: str | None
                  ) -> tuple[str, str | None, str | None, tuple[str, str, str | None] | None, bool]:
    """Apply a configured tier route: (provider, model, effort, failover
    route, tiered). The tier replaces the call site's provider and model
    (the per-role env knobs are today's routing; the tier IS the owner's
    migration) but an explicit `effort=` from the caller still wins."""
    route = _dispatch_route(action or _CALL_CONTEXT.get().get("action"))
    if route is None or route.provider is None:
        return provider, model, effort, None, False
    return route.provider, route.model, (effort or route.effort), route.failover, True


def _prepare(provider: str, model: str | None, route: str, *, tiered: bool = False) -> str | None:
    """Record the requested provider/model on the call scope and drop a
    provider-foreign model override (the route default is used instead)."""
    att = _ATTEMPT.get()
    literal = (model or "").strip()
    requested = literal or _model_for(provider, route)
    resolution = "tier" if tiered else ("explicit" if literal else "route_default")
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
    failover: bool = True,
    effort: str | None = None,
    schema: dict[str, Any] | None = None,
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

    Keyword-only, each defaulting to today's behaviour:
      - `failover`: False = no provider hop, and content failures do not
        feed the shared breaker (the debate pair fails over on its own).
      - `effort`: sent as `output_config.effort` / `reasoning_effort` only
        to models that accept it (never Haiku 4.5 or gpt-4.1).
      - `schema`: strict `json_schema` on OpenAI; ignored elsewhere in v1.
      - `action`: what the call does (a key of `llm_attribution.ACTIONS`);
        `ticker`: the company it is about.

    OpenAI and Anthropic fail over to each other once per call when the
    other is configured — see `_with_failover`. Gemini does not.
    """
    with _call_scope("chat_json", action=action, ticker=ticker, route=route,
                     max_tokens=max_tokens):
        provider = (provider_override or settings.active_llm_provider).lower()
        if provider == "none":
            return None
        provider, model, effort, fo_route, tiered = _tier_request(action, provider, model, effort)
        if provider == "gemini":
            return gemini_chat_json(prompt, system=system, model=model, max_tokens=max_tokens,
                                    action=action, ticker=ticker)
        model = _prepare(provider, model, route, tiered=tiered)
        out = _with_failover(
            provider, _call_json, failover=failover, failover_route=fo_route,
            prompt=prompt, system=system, route=route, max_tokens=max_tokens, model=model,
            effort=effort, schema=schema,
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
    failover: bool = True,
    effort: str | None = None,
    schema: dict[str, Any] | None = None,
    action: str | None = None,
    ticker: str | None = None,
) -> str | None:
    """Same `model`, `failover`, `effort`, `action` semantics as `chat_json`
    (`schema` is accepted and ignored). Returns plain text or None."""
    with _call_scope("chat_text", action=action, ticker=ticker, route=route,
                     max_tokens=max_tokens):
        provider = (provider_override or settings.active_llm_provider).lower()
        if provider == "none":
            return None
        provider, model, effort, fo_route, tiered = _tier_request(action, provider, model, effort)
        if provider == "gemini":
            return gemini_chat_text(prompt, system=system, model=model, max_tokens=max_tokens,
                                    action=action, ticker=ticker)
        model = _prepare(provider, model, route, tiered=tiered)
        out = _with_failover(
            provider, _call_text, failover=failover, failover_route=fo_route,
            prompt=prompt, system=system, route=route, max_tokens=max_tokens, model=model,
            effort=effort, schema=schema,
        )
        return None if out is _NO_CLIENT else out
