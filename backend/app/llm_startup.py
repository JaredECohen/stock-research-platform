"""The two LLM lines each service logs at startup, web and worker alike.

- `LLM routing: ...` — `llm.model_summary()` in one line: the provider, the
  per-role models, the tier routes, the Gemini models, the chat surface, the
  failover map and the attribution / reviewer / debate modes. Before the
  2026-09-25 program only the web service printed a routing line, and only
  its role table; the worker — which runs the monitoring loops, the memo
  regen queue and every Gemini call — printed nothing, so a `render.yaml`
  model change (wave H) could not be checked on the process that spends
  most of the money. Building the summary also runs the price-row check, so
  a configured model with no price row now WARNs on the worker too.
- `model_access ...` — `llm.model_access_report()`: each provider's
  `models.list`, never a generation, so it costs nothing (plan §8.1: the
  owner confirms model access before wave H from this line). It makes one
  network round trip per configured provider, so it runs on a daemon thread
  and never delays boot or a deploy's health check.

Names and booleans only: nothing here reads or prints key material.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)


def _kv(key: str, value: Any) -> str:
    text = str(value if value not in (None, "") else "-")
    return f'{key}="{text}"' if " " in text else f"{key}={text}"


def routing_line(summary: dict[str, Any] | None = None) -> str:
    """`LLM routing: k=v ...` from `llm.model_summary()` (fixed field order)."""
    if summary is None:
        from .agents import llm
        summary = llm.model_summary()
    from .config import settings

    configured = ",".join(k for k, v in (summary.get("configured") or {}).items() if v) or "none"
    parts = [
        _kv("provider", summary.get("active_provider")),
        _kv("choice", summary.get("provider_choice")),
        _kv("configured", configured),
        _kv("failover", "on" if settings.llm_failover_enabled else "off"),
        _kv("attribution", summary.get("attribution_mode")),
        _kv("reviewer", summary.get("reviewer_mode")),
        _kv("debate", summary.get("debate_mode")),
        _kv("chat", summary.get("chat")),
    ]
    parts += [_kv(f"gemini.{k}", v) for k, v in (summary.get("gemini") or {}).items()]
    parts += [_kv(f"tier.{k}", v) for k, v in (summary.get("tiers") or {}).items()]
    fmap = summary.get("failover_map") or {}
    parts.append(_kv("failover_map", ",".join(f"{k}>{v}" for k, v in fmap.items()) or "-"))
    parts += [_kv(f"role.{k}", v) for k, v in (summary.get("role_models") or {}).items()]
    return "LLM routing: " + " ".join(parts)


def log_routing(logger: logging.Logger | None = None) -> str | None:
    """Log the routing line; never raises (a startup log must not stop a
    boot). Returns the line, or None when the summary failed."""
    out = logger or log
    try:
        line = routing_line()
    except Exception as exc:  # pragma: no cover - startup hardening
        out.warning("LLM routing summary failed: %s", type(exc).__name__)
        return None
    out.info("%s", line)
    return line


def _model_access() -> None:
    try:
        from .agents import llm
        llm.model_access_report()
    except Exception as exc:  # pragma: no cover - a diagnostic must not kill a thread noisily
        log.warning("model_access check failed: %s", type(exc).__name__)


def start_model_access_check() -> threading.Thread:
    """Run `llm.model_access_report()` once, on a daemon thread."""
    thread = threading.Thread(target=_model_access, name="llm-model-access", daemon=True)
    thread.start()
    return thread
