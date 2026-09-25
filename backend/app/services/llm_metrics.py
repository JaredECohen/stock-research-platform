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
from datetime import date, datetime, timedelta
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
#
# 2026-09-25: the program's models (DEVPLAN owner decision 7, verified
# against official sources in plans/2026-09-24/model_research_2026-09-25.json)
# and the embedding model were added, with per-model cache rates below.
PRICES_VERIFIED_ON = "2026-09-25"
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-6-astra":     (10.00, 50.00),
    "gpt-6-sol":       (2.00, 10.00),
    "gpt-6-luna":      (0.10, 0.50),
    "gpt-5":           (1.25, 10.00),
    "gpt-5-mini":      (0.25, 2.00),
    "gpt-5.4":         (2.50, 15.00),
    "gpt-5.5":         (5.00, 30.00),
    "gpt-5.5-pro":     (30.00, 180.00),
    "gpt-4.1-mini":    (0.40, 1.60),
    "gpt-4o-mini":     (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.00),
    # Anthropic
    "claude-haiku-4-5":  (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5":   (2.00, 10.00),
    "claude-opus-4-7":   (5.00, 25.00),
    "claude-opus-4-8":   (5.00, 25.00),
    "claude-opus-5":     (5.00, 25.00),
    "claude-opus-5-5":   (4.00, 20.00),
    # Google / Vertex
    "gemini-2.5-flash":       (0.30, 2.50),
    "gemini-2.5-pro":         (1.25, 10.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.5-flash-lite":  (0.30, 2.50),
    # Launch price through 2026-12-31; `DATED_PRICES` carries the rise.
    "gemini-3.8-flash":       (0.75, 3.75),
}

# Per-model prompt-cache rates, $/MTok: (cache READ, cache WRITE or None).
# These replace the hard-coded 0.1x / 0.5x / 1.25x multipliers, which were
# wrong once the fleet spread: Opus 5.5 reads at 0.05x, GPT-6 at 0.10x,
# gpt-4.1-mini at 0.25x. Provider conventions still decide WHICH tokens
# these apply to (see `estimate_cost_usd`). Every row in
# MODEL_PRICES_PER_MTOK has one (`test_llm_prices` pins it). Rows marked
# "carried" keep the pre-2026-09-25 multiplier's value because the
# program's research did not re-verify them; changing them would move
# historical dashboards without new evidence.
MODEL_CACHE_PRICES_PER_MTOK: dict[str, tuple[float, float | None]] = {
    # OpenAI: reads are inside prompt_tokens; OpenAI bills no cache write,
    # the 1.25x write rate is recorded for GPT-6 as the research listed it.
    "gpt-6-astra":     (1.00, 12.50),
    "gpt-6-sol":       (0.20, 2.50),
    "gpt-6-luna":      (0.01, 0.125),
    "gpt-5":           (0.625, None),   # carried (0.5x)
    "gpt-5-mini":      (0.125, None),   # carried (0.5x)
    "gpt-5.4":         (1.25, None),    # carried (0.5x)
    "gpt-5.5":         (0.50, None),
    "gpt-5.5-pro":     (15.00, None),   # carried (0.5x)
    "gpt-4.1-mini":    (0.10, None),
    "gpt-4o-mini":     (0.075, None),   # carried (0.5x)
    "text-embedding-3-small": (0.02, None),
    # Anthropic: reads/writes are reported OUTSIDE input_tokens.
    "claude-haiku-4-5":  (0.10, 1.25),
    "claude-sonnet-4-6": (0.30, 3.75),
    "claude-sonnet-5":   (0.20, 2.50),
    "claude-opus-4-7":   (0.50, 6.25),
    "claude-opus-4-8":   (0.50, 6.25),
    "claude-opus-5":     (0.50, 6.25),
    "claude-opus-5-5":   (0.20, 5.00),
    # Gemini: cached_content_token_count is inside prompt_token_count.
    "gemini-2.5-flash":       (0.03, None),
    "gemini-2.5-pro":         (0.125, None),  # carried (0.1x)
    "gemini-3.1-pro-preview": (0.20, None),
    "gemini-3.5-flash-lite":  (0.03, None),
    "gemini-3.8-flash":       (0.075, None),
}

# Scheduled price changes: model -> [(first day the new rate applies,
# (in, out), (cache read, cache write))]. A row's cost is priced at the
# rate in force on the day it was written.
DATED_PRICES: dict[str, list[tuple[date, tuple[float, float], tuple[float, float | None]]]] = {
    "gemini-3.8-flash": [(date(2027, 1, 1), (1.50, 7.50), (0.15, None))],
}

# GPT-6 bills a whole request at 2x input / 1.5x output above this many
# input tokens. Not priced in (no call is expected near it); `llm.py`
# logs a WARNING so one is never silent.
GPT6_LONG_CONTEXT_TOKENS = 272_000

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


def _as_day(on: date | datetime | None) -> date:
    if on is None:
        return datetime.utcnow().date()
    return on.date() if isinstance(on, datetime) else on


def _dated(key: str, on: date | datetime | None
           ) -> tuple[tuple[float, float], tuple[float, float | None]] | None:
    """The scheduled rate in force for `key` on `on`, or None (base row)."""
    changes = DATED_PRICES.get(key)
    if not changes:
        return None
    day = _as_day(on)
    current = None
    for starts, io, cache in sorted(changes, key=lambda c: c[0]):
        if day >= starts:
            current = (io, cache)
    return current


def cache_rates(provider: str, model: str, *, on: date | datetime | None = None
                ) -> tuple[float, float]:
    """(cache read, cache write) $/MTok for a model.

    A tabulated model uses its own row (a missing write rate means the
    provider bills none: 0). An untabulated model is already priced at the
    provider-default guess, so its cache tokens take that provider's old
    convention (Anthropic 0.1x/1.25x, OpenAI 0.5x, Gemini full price) —
    still a guess, and `_rates` has already logged it as one.
    """
    prov = (provider or "").lower()
    key = price_key(model)
    dated = _dated(key, on)
    if dated is not None:
        read, write = dated[1]
        return read, float(write or 0.0)
    if key in MODEL_CACHE_PRICES_PER_MTOK:
        read, write = MODEL_CACHE_PRICES_PER_MTOK[key]
        return read, float(write or 0.0)
    p_in, _ = PROVIDER_PRICE_FALLBACK.get(prov, (0.0, 0.0))
    if prov == "anthropic":
        return 0.1 * p_in, 1.25 * p_in
    if prov == "openai":
        return 0.5 * p_in, 0.0
    return p_in, 0.0


def estimate_cost_usd(provider: str, model: str,
                      tokens_in: int, tokens_out: int, *,
                      cache_read_tokens: int = 0,
                      cache_write_tokens: int = 0,
                      on: date | datetime | None = None) -> float:
    """Multiply tokens by per-MTok rates. Best-effort: missing prices
    fall to the provider default (logged once per model), then zero.
    Never raises.

    Prompt-cache tokens are priced at the model's own cache rates
    (`MODEL_CACHE_PRICES_PER_MTOK`); the provider convention decides which
    tokens they apply to. Anthropic's `input_tokens` EXCLUDES cache reads
    and writes, so those are added; OpenAI's `prompt_tokens` and Gemini's
    `prompt_token_count` INCLUDE cached reads, so those are carved out of
    the input and re-priced. `on` selects a scheduled (dated) price; the
    default is today.

    Output is priced as reported: OpenAI completion_tokens and Anthropic
    output_tokens already include reasoning/thinking, and `llm.py` adds
    Gemini thoughts into tokens_out before it gets here, so reasoning is
    never billed twice (attribution critique #13).
    """
    prov = (provider or "").lower()
    key = price_key(model)
    dated = _dated(key, on)
    p_in, p_out = dated[0] if dated is not None else _rates(prov, model)
    read_rate, write_rate = cache_rates(prov, model, on=on)
    n_in = max(0, int(tokens_in or 0))
    n_out = max(0, int(tokens_out or 0))
    n_read = max(0, int(cache_read_tokens or 0))
    n_write = max(0, int(cache_write_tokens or 0))
    if prov == "anthropic":
        cost_in = n_in * p_in + n_write * write_rate + n_read * read_rate
    elif prov in ("openai", "gemini"):
        cost_in = max(0, n_in - n_read) * p_in + min(n_read, n_in) * read_rate
    else:
        cost_in = n_in * p_in
    cost_out = n_out * p_out
    return round((cost_in + cost_out) / 1_000_000.0, 6)


def not_skipped_filter():
    """SQL predicate for rows that are NOT skip rows.

    A skipped attempt (open breaker, no client, grounding cap) made no
    provider request: it has zero tokens and costs nothing, but counting it
    as a call or a failure would inflate every existing aggregate
    (attribution critique #14). Legacy rows have NULL error_type.
    """
    from sqlalchemy import or_
    return or_(LLMCallLog.error_type.is_(None), ~LLMCallLog.error_type.like("skipped:%"))


def _ensure_table(db: Session) -> None:
    """Mirror the cache.snapshots pattern — direct callers don't need init_db()."""
    LLMCallLog.__table__.create(bind=db.get_bind(), checkfirst=True)


def _row_cost(r: Any) -> float:
    return estimate_cost_usd(
        r.provider, r.model, r.tokens_in, r.tokens_out,
        cache_read_tokens=r.cache_read_tokens,
        cache_write_tokens=r.cache_write_tokens,
        on=r.generated_at,
    )


def cost_per_run(run_id: str, *, agents: Any = None,
                 db: Session | None = None) -> dict[str, Any]:
    """Per-call detail + totals for one memo run.

    `agents` (an iterable of agent names) restricts the rows, e.g. the
    debate budget reads only the Bull/Bear Advocate spend of the run.
    Skip rows (no provider request) are excluded; see
    `not_skipped_filter`.
    """
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = (
            select(LLMCallLog)
            .where(LLMCallLog.run_id == run_id, not_skipped_filter())
            .order_by(LLMCallLog.generated_at.asc())
        )
        if agents is not None:
            stmt = stmt.where(LLMCallLog.agent_name.in_(sorted(set(agents))))
        rows = list(db.execute(stmt).scalars().all())
        calls = [{
            "agent_name": r.agent_name,
            "provider": r.provider,
            "model": r.model,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "cache_read_tokens": int(r.cache_read_tokens or 0),
            "cache_write_tokens": int(r.cache_write_tokens or 0),
            "cost_usd": _row_cost(r),
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
        stmt = select(LLMCallLog).where(not_skipped_filter())
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
            a["cost_usd"] += _row_cost(r)
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
        stmt = select(LLMCallLog).where(not_skipped_filter())
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
            a["cost_usd"] += _row_cost(r)
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
        stmt = (select(LLMCallLog).where(not_skipped_filter())
                .order_by(LLMCallLog.duration_ms.desc()).limit(n))
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


def skipped_attempts(*, since: datetime | None = None,
                     db: Session | None = None) -> dict[str, int]:
    """Skip rows by reason (`skipped:breaker_open` → n), which every other
    aggregate here excludes. A skip made no provider request, so it has no
    cost, but a breaker that keeps skipping is exactly what a reviewer of
    the cost report needs to see."""
    from sqlalchemy import func
    own = db is None
    if own:
        db = SessionLocal()
    try:
        _ensure_table(db)
        stmt = (select(LLMCallLog.error_type, func.count())
                .where(LLMCallLog.error_type.like("skipped:%"))
                .group_by(LLMCallLog.error_type))
        if since:
            stmt = stmt.where(LLMCallLog.generated_at >= since)
        return {str(k): int(n) for k, n in db.execute(stmt).all()}
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
