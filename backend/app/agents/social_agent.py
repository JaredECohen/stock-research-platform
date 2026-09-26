"""Social media agent — honest "no data" outside demo mode (FIX-016 / L7).

We have no social data source. The old implementation asked Gemini, with no
search grounding, to "aggregate" seven days of X / Reddit / StockTwits
sentiment — a model with no access to those feeds, so the number was
invented — and fell back to `tools.get_social_sentiment`, a hash of the
ticker's characters, labelled as sentiment. Both are gone from live mode:

- live mode (`not settings.use_demo_data_only`) returns
  `{"source": "none", "status": "unavailable"}` with no LLM call, no cache
  read and no numbers, so nothing downstream can mistake an absence of data
  for a reading;
- demo mode keeps the deterministic stub (labelled `source: "stub"`) so the
  offline demo and the test suite still exercise the scalar's shape.

A grounded or API-backed source (StockTwits / Reddit / X) is a later owner
decision; wire it here, behind real fetched data with cited sources, and
never route an ungrounded model call back in.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from ..cache import cache_get, cache_put
from ..config import settings
from . import tools as _tools


def run(ticker: str, *, force_refresh: bool = False) -> dict[str, Any]:
    """Social sentiment for `ticker`: "unavailable" live, the stub in demo."""
    if not settings.use_demo_data_only:
        # Checked before the cache on purpose: a `social_hot` row written
        # today by the pre-fix code (Gemini guess or hash stub) must not be
        # served back as data after deploy.
        return _tools.social_unavailable(ticker)

    today_key = f"social_hot:{ticker}:{date.today().isoformat()}"
    if not force_refresh:
        cached = cache_get(today_key, "social_hot", max_age_seconds=24 * 3600)
        if cached and isinstance(cached.payload, dict):
            return cached.payload

    payload = {**_tools.get_social_sentiment(ticker), "source": "stub"}
    cache_put(today_key, "social_hot", payload=payload,
              sources_used=[f"social:{ticker}"],
              generated_by="social_agent", cost_tokens=0,
              ttl_seconds=24 * 3600)
    return payload
