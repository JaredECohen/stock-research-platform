"""Read-through TTL cache for raw provider responses (Wave 9b).

Sits between `data_service` and the live provider chain. Every per-
capability call routes through `cached_call(capability, key, fetcher,
…)` which (a) returns a fresh row if one exists, (b) falls through to
the provider on miss/expiry, (c) writes the new response back, and
(d) serves a stale row when the provider also misses — but only up to
a per-capability age cap (`MAX_STALE_BY_CAPABILITY`); beyond that the
caller gets None and the memo degrades honestly instead of quietly
building on data from months ago.

Per-capability TTLs are tuned to how frequently the underlying data
actually changes:

    profile     7 days   (description, sector, FY-end, CIK — stable;
                          market cap drifts but doesn't justify daily
                          refetch on every research call)
    prices      1 day    (full daily history; refresh after each close)
    quote       60 s     (intraday last-trade price for valuation
                          comparison; profile's last_price is stale
                          for fast movers like NVDA)
    ratios      1 day    (price-dependent metrics)
    estimates   1 day    (sell-side updates frequently but not
                          intra-day for most names)
    earnings    1 day    (calendar / surprises)
    news        1 hour   (time-sensitive)
    macro       1 day    (mostly weekly / monthly publication)

Every stale serve and every too-stale refusal is written to the
`cache_cost_logs` ledger (subject `provider_cache`) so the web and
worker processes see one shared picture via `stale_stats()`; an
in-process counter would only describe whichever process served the
request.
"""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..cache import CacheCostLog, log_cost
from ..config import settings
from ..database import SessionLocal
from ..models import ProviderCache

log = logging.getLogger(__name__)


# In-seconds. None = never expires (always reuse cache).
TTL_BY_CAPABILITY: Dict[str, int] = {
    "profile":   7 * 86400,
    "prices":         86400,
    "quote":             60,
    "ratios":         86400,
    "estimates":      86400,
    "earnings":       86400,
    "news":            3600,
    "macro":          86400,
}

# Oldest row `cached_call` will still serve when the provider misses.
# Roughly "how long before this data is worse than no data": a month-old
# profile is still the right company, a week-old price series is a
# usable backdrop, but an hour-old quote is not intraday anymore.
MAX_STALE_BY_CAPABILITY: Dict[str, int] = {
    "profile":   30 * 86400,
    "prices":     7 * 86400,
    "quote":           3600,
    "ratios":     7 * 86400,
    "estimates": 14 * 86400,
    "earnings":  14 * 86400,
    "news":           86400,
    "macro":     14 * 86400,
}
DEFAULT_MAX_STALE_SECONDS = 7 * 86400

# Ledger identity for stale-serve monitoring rows (see `stale_stats`).
STALE_LOG_SUBJECT = "provider_cache"
STALE_SERVED_KIND = "stale_served"
STALE_REFUSED_KIND = "stale_refused"

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _now() -> datetime:
    return datetime.utcnow()


def _parse_duration(text: str) -> int:
    """'3600', '900s', '15m', '12h', '7d' → seconds. Raises ValueError."""
    text = text.strip().lower()
    unit = 1
    if text and text[-1] in _DURATION_UNITS:
        unit = _DURATION_UNITS[text[-1]]
        text = text[:-1]
    seconds = int(text) * unit
    if seconds < 0:
        raise ValueError("negative duration")
    return seconds


@lru_cache(maxsize=8)
def _parse_max_stale_overrides(raw: str) -> Dict[str, int]:
    """Parse `PROVIDER_CACHE_MAX_STALE` leniently: one bad entry must not
    take the others down with it, and must not fail startup. Memoised on
    the raw string so a malformed entry is warned about once, not on
    every cache lookup."""
    overrides: Dict[str, int] = {}
    for entry in raw.replace(";", ",").split(","):
        entry = entry.strip()
        if not entry:
            continue
        cap, sep, value = entry.partition("=")
        if not sep:
            cap, sep, value = entry.partition(":")
        cap = cap.strip().lower()
        try:
            if not sep or not cap:
                raise ValueError("expected capability=seconds")
            overrides[cap] = _parse_duration(value)
        except ValueError as exc:
            log.warning(
                "ignoring invalid provider_cache_max_stale entry %r: %s",
                entry, exc,
            )
    return overrides


def max_stale_seconds(capability: str) -> int:
    """Age cap for stale fallback: env override > table default > 7d."""
    overrides = _parse_max_stale_overrides(settings.provider_cache_max_stale or "")
    if capability in overrides:
        return overrides[capability]
    return MAX_STALE_BY_CAPABILITY.get(capability, DEFAULT_MAX_STALE_SECONDS)


def _is_fresh(fetched_at: datetime, ttl_seconds: Optional[int]) -> bool:
    if ttl_seconds is None:
        return True  # never-expire mode
    return _now() - fetched_at < timedelta(seconds=ttl_seconds)


def _read_row(capability: str, key: str) -> Optional[Tuple[Any, datetime]]:
    """(payload, fetched_at) for the row, or None when absent."""
    with SessionLocal() as db:
        row = db.execute(
            select(ProviderCache).where(
                ProviderCache.capability == capability,
                ProviderCache.key == key,
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return row.payload_json, row.fetched_at


def get(
    capability: str, key: str,
    *, ttl_seconds: Optional[int] = None,
    serve_stale: bool = False,
    max_age_seconds: Optional[int] = None,
) -> Optional[Any]:
    """Read the cached payload for `(capability, key)`.

    Returns None when no row exists. When a row exists but is past
    `ttl_seconds`, returns None *unless* `serve_stale=True` (used as
    the last-resort fallback when the provider also missed). With
    `serve_stale=True`, `max_age_seconds` bounds how old that fallback
    may be; rows older than it are refused (None) as well.
    """
    found = _read_row(capability, key)
    if found is None:
        return None
    payload, fetched_at = found
    if _is_fresh(fetched_at, ttl_seconds):
        return payload
    if not serve_stale:
        return None
    if max_age_seconds is not None and not _is_fresh(fetched_at, max_age_seconds):
        return None
    return payload


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _backoff(attempt: int) -> None:
    """Short jittered pause between `put` retries so two callers that
    collided don't immediately re-collide. Worst case across all retries
    is well under a second; tests stub `_sleep` to keep it free."""
    _sleep(random.uniform(0, 0.05) * attempt)


def put(capability: str, key: str, payload: Any) -> None:
    """Upsert a cache row. No-op when payload is empty (None / [] / {}).

    Race-safe: when two callers miss the cache simultaneously and both
    try to INSERT, the second one would otherwise blow up the unique
    index on `(capability, key)`. We retry as an UPDATE on
    IntegrityError so concurrent screener / memo runs don't 500.
    """
    if payload in (None, [], {}):
        return
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            with SessionLocal() as db:
                existing = db.execute(
                    select(ProviderCache).where(
                        ProviderCache.capability == capability,
                        ProviderCache.key == key,
                    )
                ).scalar_one_or_none()
                if existing is None:
                    db.add(ProviderCache(
                        capability=capability, key=key,
                        payload_json=payload, fetched_at=_now(),
                    ))
                else:
                    existing.payload_json = payload
                    existing.fetched_at = _now()
                db.commit()
            return
        except IntegrityError:
            # Lost the insert race — the next attempt sees the winner's
            # row and updates it. Three strikes means something other
            # than a race is wrong, so surface it (payload deliberately
            # not logged: provider responses can be large).
            if attempt == attempts:
                log.warning(
                    "provider_cache.put gave up after %d IntegrityErrors "
                    "capability=%s key=%s", attempts, capability, key,
                )
                raise
            _backoff(attempt)


def invalidate(capability: str, key: Optional[str] = None) -> int:
    """Drop rows. Pass `key=None` to clear every row for `capability`.

    Returns the number of rows deleted.
    """
    with SessionLocal() as db:
        stmt = select(ProviderCache).where(ProviderCache.capability == capability)
        if key is not None:
            stmt = stmt.where(ProviderCache.key == key)
        rows = db.execute(stmt).scalars().all()
        n = len(rows)
        for row in rows:
            db.delete(row)
        db.commit()
        return n


def _record_stale(kind: str, capability: str, key: str, age_seconds: int) -> None:
    """Write the monitoring row. A ledger hiccup must not take the data
    path down with it, so failures are logged rather than raised."""
    note = json.dumps(
        {"capability": capability, "key": key, "age_seconds": age_seconds},
        separators=(",", ":"),
    )
    try:
        log_cost(STALE_LOG_SUBJECT, kind, 0, note=note)
    except Exception:
        log.warning(
            "provider_cache: could not record %s for capability=%s key=%s",
            kind, capability, key, exc_info=True,
        )


def cached_call(
    capability: str, key: str, fetcher: Callable[[], Any],
    *, ttl_seconds: Optional[int] = None,
    force_refresh: bool = False,
) -> Optional[Any]:
    """Read-through: cache hit → return; miss → call `fetcher`, write,
    return; provider miss → fall back to a stale cached row no older
    than `max_stale_seconds(capability)`, else None.

    `ttl_seconds` defaults to `TTL_BY_CAPABILITY[capability]`. Pass an
    explicit value (or `None` to never expire) to override.
    """
    if ttl_seconds is None and not force_refresh:
        ttl_seconds = TTL_BY_CAPABILITY.get(capability)

    if not force_refresh:
        cached = get(capability, key, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached

    fresh = fetcher()
    if fresh is not None and fresh != [] and fresh != {}:
        put(capability, key, fresh)
        return fresh

    # Provider also missed — better stale than empty, up to a point.
    found = _read_row(capability, key)
    if found is None:
        return None
    payload, fetched_at = found
    age = int((_now() - fetched_at).total_seconds())
    cap = max_stale_seconds(capability)
    if age >= cap:
        log.warning(
            "provider miss and cached row too stale capability=%s key=%s "
            "age_seconds=%d max_stale_seconds=%d", capability, key, age, cap,
        )
        _record_stale(STALE_REFUSED_KIND, capability, key, age)
        return None
    log.warning(
        "provider miss, serving stale cached row capability=%s key=%s "
        "age_seconds=%d max_stale_seconds=%d", capability, key, age, cap,
    )
    _record_stale(STALE_SERVED_KIND, capability, key, age)
    return payload


# Upper bound on ledger rows one `stale_stats` call will scan. Stale
# serves are rare in healthy operation; hitting this means a provider
# has been down for a while and the counts are a floor, flagged via
# `truncated`.
STALE_STATS_ROW_LIMIT = 5000


def stale_stats(window_hours: int = 24) -> Dict[str, Any]:
    """Aggregate the stale-serve ledger over the trailing window.

    One bounded query; never raises — a failure returns zeros plus an
    `error` key so the status endpoint stays up when the DB doesn't.
    """
    result: Dict[str, Any] = {
        "window_hours": window_hours,
        "stale_served": 0,
        "stale_refused": 0,
        "by_capability": {},
        "oldest_served_age_seconds": None,
        "truncated": False,
    }
    try:
        since = _now() - timedelta(hours=window_hours)
        with SessionLocal() as db:
            rows = db.execute(
                select(CacheCostLog.kind, CacheCostLog.note)
                .where(
                    CacheCostLog.subject == STALE_LOG_SUBJECT,
                    CacheCostLog.kind.in_((STALE_SERVED_KIND, STALE_REFUSED_KIND)),
                    CacheCostLog.generated_at >= since,
                )
                .order_by(CacheCostLog.generated_at.desc())
                .limit(STALE_STATS_ROW_LIMIT + 1)
            ).all()
        if len(rows) > STALE_STATS_ROW_LIMIT:
            result["truncated"] = True
            rows = rows[:STALE_STATS_ROW_LIMIT]

        by_cap: Dict[str, Dict[str, Any]] = result["by_capability"]
        oldest_served: Optional[int] = None
        for kind, note in rows:
            try:
                info = json.loads(note or "{}")
            except ValueError:
                info = {}
            cap = str(info.get("capability") or "unknown")
            age = info.get("age_seconds")
            age = int(age) if isinstance(age, (int, float)) else None
            bucket = by_cap.setdefault(
                cap, {"served": 0, "refused": 0, "max_age_seconds": None},
            )
            if kind == STALE_SERVED_KIND:
                result["stale_served"] += 1
                bucket["served"] += 1
                if age is not None and (oldest_served is None or age > oldest_served):
                    oldest_served = age
            else:
                result["stale_refused"] += 1
                bucket["refused"] += 1
            if age is not None and (
                bucket["max_age_seconds"] is None or age > bucket["max_age_seconds"]
            ):
                bucket["max_age_seconds"] = age
        result["oldest_served_age_seconds"] = oldest_served
    except Exception as exc:
        log.warning("provider_cache.stale_stats failed: %s", exc, exc_info=True)
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
