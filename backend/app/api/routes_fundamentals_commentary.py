"""FEAT-001 — `POST /api/fundamentals/commentary` (slice S3 fills this).

S2 mounts this router (behind `ENABLE_FUNDAMENTALS_EXPLORER`) so the
app imports cleanly whichever slice merges first; the route itself is
S3's. What is already wired for it, so S3 only adds the handler:

  - `auth/policy.py` classifies the path as `metered:chart_commentary`
    (signed-in when the wall is on; the plan meters it).
  - `auth/features.py` `chart_commentary`: Free 5 / Pro 100 a month,
    `max_concurrent=2` — `require_feature("chart_commentary",
    resource_param=None)` takes the DB-backed lease and reserves the
    meter *before* the LLM call; `grant.release()` gives it back when
    the call returns nothing (a degraded body is free).
  - `rate_limit.LIMITS["fundamentals_commentary"]` (10/minute per IP,
    always) and `gating.rate_scope("llm_light")` (per user, wall on).
  - `models.ChartCommentary` — the cross-process cache keyed by
    `cache_key`, swept after 90 days by `monitoring/llm_log_gc`.
  - `settings.fundamentals_anon_commentary` (default False): with the
    wall OFF, an anonymous caller gets the deterministic degraded shape
    (`degraded_reason="commentary requires an account"`), no LLM call,
    nothing charged. `settings.fundamentals_commentary_model` pins the
    model; empty = the cheap route's default.

Contract: `schemas.fundamentals.CommentaryRequest` → `CommentaryOut`.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
