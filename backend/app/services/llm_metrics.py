"""Aggregations over the LLMCallLog audit table (Wave 1A).

Used by the admin endpoint and the CLI cost report. Functions return
plain dicts so they're trivially JSON-serializable.

Wave 8D: USD cost estimation. Token counts are stored at the call
site; this module multiplies them by list price-per-MTok rates to
produce dollar figures. `MODEL_PRICES_PER_MTOK` carries every model the
app routes to (`test_llm_prices.py` pins that against `Settings`); a
model outside it prices at the provider default and is logged once,
and `unit_economics` names it, because a silent default is how the
table drifted 3x in both directions before 2026-09-19. Update the
table — and `PRICES_VERIFIED_ON` — when a provider's pricing changes.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import LLMCallLog

log = logging.getLogger(__name__)

# List prices per million tokens (USD), input / output: standard tier, and
# the short-context (<= 200K-token prompt) rate where a provider tiers by
# prompt length. Verified against the providers' own pricing pages on
# PRICES_VERIFIED_ON:
#   https://platform.claude.com/docs/en/about-claude/pricing
#   https://developers.openai.com/api/docs/pricing   ("All models" table)
#   https://ai.google.dev/gemini-api/docs/pricing    (paid tier)
# Not load-bearing for any logic — only the cost estimate output uses
# these — but `test_llm_prices.py` fails when a model named in `Settings`
# has no row here, so a routing change cannot silently price at a default.
PRICES_VERIFIED_ON = "2026-09-19"
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5":           (1.25, 10.00),
    "gpt-5-mini":      (0.25, 2.00),
    "gpt-5.4":         (2.50, 15.00),
    "gpt-5.5":         (5.00, 30.00),
    "gpt-5.5-pro":     (30.00, 180.00),
    "gpt-4.1-mini":    (0.40, 1.60),
    "gpt-4o-mini":     (0.15, 0.60),
    # Anthropic
    "claude-haiku-4-5":  (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5":   (2.00, 10.00),
    "claude-opus-4-7":   (5.00, 25.00),
    "claude-opus-4-8":   (5.00, 25.00),
    "claude-opus-5":     (5.00, 25.00),
    # Google / Vertex
    "gemini-2.5-flash":       (0.30, 2.50),
    "gemini-2.5-pro":         (1.25, 10.00),
    "gemini-3.1-pro":         (2.00, 12.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

# Provider-level fallback (when the specific model isn't tabulated). It keeps
# a total from reading as $0, but it is a guess, not a price: every use is
# logged once per model and `unit_economics` names the model in its report.
PROVIDER_PRICE_FALLBACK: dict[str, tuple[float, float]] = {
    "openai":    (3.00, 12.00),
    "anthropic": (3.00, 15.00),
    "gemini":    (3.00, 12.00),
}

# Dated snapshot IDs — Anthropic `claude-haiku-4-5-20251001`, OpenAI
# `gpt-5.5-2026-04-23` — are priced as their alias.
_SNAPSHOT_SUFFIX_RE = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")


def price_key(model: str) -> str:
    """Normalise a logged model ID to its row in `MODEL_PRICES_PER_MTOK`."""
    return _SNAPSHOT_SUFFIX_RE.sub("", (model or "").strip().lower())


def price_source(provider: str, model: str) -> str:
    """Where a (provider, model) pair's rate comes from: ``"model"`` (an
    exact row), ``"provider_default"`` (the provider fallback — a guess),
    or ``"unpriced"`` (neither; the estimate is $0)."""
    if price_key(model) in MODEL_PRICES_PER_MTOK:
        return "model"
    if (provider or "").lower() in PROVIDER_PRICE_FALLBACK:
        return "provider_default"
    return "unpriced"


# (provider, model) pairs already warned about, so a busy loop on one
# unlisted model logs once, not once per call.
_warned_models: set[tuple[str, str]] = set()


def _rates(provider: str, model: str) -> tuple[float, float]:
    prov = (provider or "").lower()
    key = price_key(model)
    source = price_source(prov, key)
    if source == "model":
        return MODEL_PRICES_PER_MTOK[key]
    if (prov, key) not in _warned_models:
        _warned_models.add((prov, key))
        if source == "provider_default":
            log.warning(
                "llm_metrics: no price row for %s/%s — costing it at the %s provider "
                "default; add the model to MODEL_PRICES_PER_MTOK",
                prov, key or "?", prov,
            )
        else:
            log.warning(
                "llm_metrics: no price row or provider default for %s/%s — costing it at $0",
                prov or "?", key or "?",
            )
    return PROVIDER_PRICE_FALLBACK.get(prov, (0.0, 0.0))


def estimate_cost_usd(provider: str, model: str,
                      tokens_in: int, tokens_out: int, *,
                      cache_read_tokens: int = 0,
                      cache_write_tokens: int = 0) -> float:
    """Multiply tokens by per-MTok rates. Best-effort: missing prices
    fall to the provider default (logged once per model), then zero.
    Never raises.

    Prompt-cache tokens are priced per provider convention: Anthropic's
    `input_tokens` excludes them (writes bill 1.25x, reads 0.1x of the
    input rate); OpenAI's `prompt_tokens` includes them (reads bill 0.5x).
    """
    prov = (provider or "").lower()
    p_in, p_out = _rates(prov, model)
    n_in = max(0, int(tokens_in or 0))
    n_out = max(0, int(tokens_out or 0))
    n_read = max(0, int(cache_read_tokens or 0))
    n_write = max(0, int(cache_write_tokens or 0))
    if prov == "anthropic":
        billed_in = n_in + 1.25 * n_write + 0.1 * n_read
    elif prov == "openai":
        billed_in = max(0, n_in - n_read) + 0.5 * n_read
    else:
        billed_in = n_in
    cost_in = (billed_in / 1_000_000.0) * p_in
    cost_out = (n_out / 1_000_000.0) * p_out
    return round(cost_in + cost_out, 6)


def _ensure_table(db: Session) -> None:
    """Mirror the cache.snapshots pattern — direct callers don't need init_db()."""
    LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)


def cost_per_run(run_id: str, *, db: Session | None = None) -> dict[str, Any]:
    """Per-call detail + totals for one memo run."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        rows = list(db.execute(
            select(LLMCallLog)
            .where(LLMCallLog.run_id == run_id)
            .order_by(LLMCallLog.generated_at.asc())
        ).scalars().all())
        calls = [{
            "agent_name": r.agent_name,
            "provider": r.provider,
            "model": r.model,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "cache_read_tokens": int(r.cache_read_tokens or 0),
            "cache_write_tokens": int(r.cache_write_tokens or 0),
            "cost_usd": estimate_cost_usd(
                r.provider, r.model, r.tokens_in, r.tokens_out,
                cache_read_tokens=r.cache_read_tokens,
                cache_write_tokens=r.cache_write_tokens,
            ),
            "duration_ms": r.duration_ms,
            "success": r.success,
            "generated_at": r.generated_at.isoformat() if r.generated_at else None,
        } for r in rows]
        return {
            "run_id": run_id,
            "n_calls": len(rows),
            "tokens_in": sum(r.tokens_in for r in rows),
            "tokens_out": sum(r.tokens_out for r in rows),
            "tokens_total": sum(r.tokens_in + r.tokens_out for r in rows),
            "cache_read_tokens": sum(int(r.cache_read_tokens or 0) for r in rows),
            "cache_write_tokens": sum(int(r.cache_write_tokens or 0) for r in rows),
            "cost_usd_total": round(sum(c["cost_usd"] for c in calls), 6),
            "duration_ms_total": sum(r.duration_ms for r in rows),
            "n_failures": sum(1 for r in rows if not r.success),
            "calls": calls,
        }
    finally:
        if own:
            db.close()


def cost_per_agent(*, since: datetime | None = None,
                   db: Session | None = None) -> dict[str, dict[str, int]]:
    """Aggregate by agent_name. Returns {agent: {n_calls, tokens, duration_ms_total, n_failures}}."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = select(LLMCallLog)
        if since:
            stmt = stmt.where(LLMCallLog.generated_at >= since)
        rows = list(db.execute(stmt).scalars().all())
        agg: dict[str, dict[str, Any]] = {}
        for r in rows:
            a = agg.setdefault(r.agent_name, {
                "n_calls": 0, "tokens_in": 0, "tokens_out": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "duration_ms_total": 0, "n_failures": 0, "cost_usd": 0.0,
            })
            a["n_calls"] += 1
            a["tokens_in"] += r.tokens_in
            a["tokens_out"] += r.tokens_out
            a["cache_read_tokens"] += int(r.cache_read_tokens or 0)
            a["cache_write_tokens"] += int(r.cache_write_tokens or 0)
            a["duration_ms_total"] += r.duration_ms
            a["cost_usd"] += estimate_cost_usd(
                r.provider, r.model, r.tokens_in, r.tokens_out,
                cache_read_tokens=r.cache_read_tokens,
                cache_write_tokens=r.cache_write_tokens,
            )
            if not r.success:
                a["n_failures"] += 1
        for v in agg.values():
            v["cost_usd"] = round(v["cost_usd"], 6)
        return agg
    finally:
        if own:
            db.close()


def cost_per_provider(*, since: datetime | None = None,
                      db: Session | None = None) -> dict[str, dict[str, Any]]:
    """Aggregate by provider name."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = select(LLMCallLog)
        if since:
            stmt = stmt.where(LLMCallLog.generated_at >= since)
        rows = list(db.execute(stmt).scalars().all())
        agg: dict[str, dict[str, Any]] = {}
        for r in rows:
            a = agg.setdefault(r.provider, {
                "n_calls": 0, "tokens_in": 0, "tokens_out": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "n_failures": 0, "cost_usd": 0.0,
            })
            a["n_calls"] += 1
            a["tokens_in"] += r.tokens_in
            a["tokens_out"] += r.tokens_out
            a["cache_read_tokens"] += int(r.cache_read_tokens or 0)
            a["cache_write_tokens"] += int(r.cache_write_tokens or 0)
            a["cost_usd"] += estimate_cost_usd(
                r.provider, r.model, r.tokens_in, r.tokens_out,
                cache_read_tokens=r.cache_read_tokens,
                cache_write_tokens=r.cache_write_tokens,
            )
            if not r.success:
                a["n_failures"] += 1
        for v in agg.values():
            v["cost_usd"] = round(v["cost_usd"], 6)
        return agg
    finally:
        if own:
            db.close()


def slowest_calls(*, since: datetime | None = None, n: int = 20,
                  db: Session | None = None) -> list[dict[str, Any]]:
    """Top-N slowest calls in the window. Useful for finding pathological prompts."""
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = select(LLMCallLog).order_by(LLMCallLog.duration_ms.desc()).limit(n)
        if since:
            stmt = stmt.where(LLMCallLog.generated_at >= since)
        rows = list(db.execute(stmt).scalars().all())
        return [{
            "agent_name": r.agent_name,
            "provider": r.provider,
            "model": r.model,
            "duration_ms": r.duration_ms,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "run_id": r.run_id,
            "generated_at": r.generated_at.isoformat() if r.generated_at else None,
        } for r in rows]
    finally:
        if own:
            db.close()


def gc_old(*, max_age_days: int = 90, db: Session | None = None) -> int:
    """Delete rows older than `max_age_days`. Returns count deleted.

    Default 90 days per locked decision in MASTER_PLAN §5. Idempotent.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        rows = db.execute(
            select(LLMCallLog).where(LLMCallLog.generated_at < cutoff)
        ).scalars().all()
        n = len(rows)
        for row in rows:
            db.delete(row)
        db.commit()
        return n
    finally:
        if own:
            db.close()
