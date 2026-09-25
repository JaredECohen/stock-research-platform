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

    profile        7 days   (description, sector, FY-end, CIK — stable;
                             market cap drifts but doesn't justify daily
                             refetch on every research call)
    prices         1 day    (full daily history; refresh after each close)
    quote          15 min   fixed backstop only. `quote_service` owns the
                             real policy: 15 min while the NYSE session is
                             open, until the next open once it has closed
                             (calendar-aware; `finance/market_calendar`)
    entitlement    1 day    (a provider endpoint the plan refuses, e.g. FMP
                             /batch-quote answering 402: remembered in the
                             DB so both processes skip it for a day)
    ratios         1 day    (price-dependent metrics)
    key_metrics    1 day    (same derivation as ratios)
    estimates      1 day    (sell-side updates frequently but not
                             intra-day for most names)
    earnings       1 day    (calendar / surprises)
    financials     7 days   (passed as a `ttl_override` at the call site)
    filings        7 days   (document bodies; immutable once filed)
    filings_index  15 min   (accession list only; drives the 30-min poll)
    transcripts    12 hours (half the transcript poller's daily cadence)
    news           1 hour   (time-sensitive)
    macro          1 day    (mostly weekly / monthly publication)

A capability that is NOT in the table gets `DEFAULT_TTL_SECONDS`, not
"cache forever". It used to get the latter, because `TTL_BY_CAPABILITY
.get(capability)` returned None and `_is_fresh(fetched_at, None)` reads
None as never-expire — so every capability someone forgot to list was
cached permanently. `filings`, `transcripts` and `key_metrics` were all
in that state, and the consequence was not a stale number: SEC EDGAR was
read exactly once per ticker for the life of the deployment, the filing
poller diffed against a frozen accession list, and no filing event ever
fired. Never-expire is now something a caller has to *ask* for, by
passing the `NEVER_EXPIRES` sentinel where the reader can see it.

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
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..cache import CacheCostLog, log_cost
from ..config import settings
from ..database import SessionLocal
from ..models import ProviderCache

log = logging.getLogger(__name__)


# Explicit "this row never goes stale". Pass it as `ttl_seconds` at a
# call site that genuinely means it; an *absent* table entry does not
# mean this and must never be read as if it did. Negative because every
# real TTL is a non-negative number of seconds, so no capability can
# collide with it by accident.
NEVER_EXPIRES = -1

# TTL used when a capability is not in the table below. One hour: short
# enough that a capability someone forgot to list re-reads its provider
# within the hour instead of freezing until the next deploy, long enough
# that a single memo run (which touches a capability several times) still
# costs one provider call. It is a backstop, not a recommendation — every
# capability `DataService` actually uses is listed, and
# `test_cached_capabilities_have_explicit_ttls` fails the build when a new
# one is added without an entry here or a `ttl_override` at the call site.
DEFAULT_TTL_SECONDS = 3600

# In seconds.
TTL_BY_CAPABILITY: dict[str, int] = {
    "profile":       7 * 86400,
    "prices":             86400,
    # The backstop for a direct `cached_call("quote", ...)`. Quote reads go
    # through `quote_service`, which applies the calendar-aware policy on
    # top of this row (15 min in session, until the next open after the
    # close) and a 60 s floor inside a memo run.
    "quote":                900,
    "ratios":             86400,
    "key_metrics":        86400,
    "estimates":          86400,
    "earnings":           86400,
    # Filing *bodies*. The text of a filing never changes once it is on
    # EDGAR, so the only reason to refetch is that a new accession has
    # appeared — which `filings_index` detects, and `edgar_poller`
    # responds to by invalidating this row for that one ticker. A short
    # TTL here would buy nothing and cost ~10 multi-megabyte document
    # fetches per ticker per pass.
    "filings":       7 * 86400,
    # Accession numbers only (`fetch_text=False`): one submissions.json
    # read, no document bodies. Deliberately half the poller's 30-minute
    # interval so a row is always expired by the time the next pass asks
    # — a TTL equal to the cadence leaves the poll racing its own cache
    # and skipping every other cycle.
    "filings_index":       900,
    # Half the transcript poller's daily cadence, for the same reason.
    "transcripts":       43200,
    "news":                3600,
    "macro":              86400,
    # "This plan refuses that endpoint" (e.g. FMP /batch-quote -> 402).
    # A day, so a plan upgrade is noticed by the next day without paying a
    # refused call on every request in between.
    "entitlement":        86400,
}

# Oldest row `cached_call` will still serve when the provider misses.
# Roughly "how long before this data is worse than no data": a month-old
# profile is still the right company, a week-old price series is a
# usable backdrop, but an hour-old quote is not intraday anymore.
MAX_STALE_BY_CAPABILITY: dict[str, int] = {
    "profile":       30 * 86400,
    "prices":         7 * 86400,
    "quote":               3600,
    "ratios":         7 * 86400,
    "key_metrics":    7 * 86400,
    "estimates":     14 * 86400,
    "earnings":      14 * 86400,
    # A filing body is immutable, so an old one is still the right
    # document — the risk is only that a newer filing exists, which is
    # the index's job to notice.
    "filings":       30 * 86400,
    # A stale index cannot detect anything new, but serving it keeps the
    # poller's diff stable instead of re-firing the whole window.
    "filings_index":  7 * 86400,
    "transcripts":   30 * 86400,
    "news":               86400,
    "macro":         14 * 86400,
    # An expired entitlement memo is not evidence of anything: retry.
    "entitlement":        86400,
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
def _parse_max_stale_overrides(raw: str) -> dict[str, int]:
    """Parse `PROVIDER_CACHE_MAX_STALE` leniently: one bad entry must not
    take the others down with it, and must not fail startup. Memoised on
    the raw string so a malformed entry is warned about once, not on
    every cache lookup."""
    overrides: dict[str, int] = {}
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


def ttl_seconds_for(capability: str) -> int:
    """The TTL `cached_call` applies when the caller passes none.

    Unlisted capabilities get `DEFAULT_TTL_SECONDS` rather than "no
    expiry". This function is the whole fix for the never-expire class
    of bug: a `dict.get` with no default returns None, and None means
    "always fresh" one layer down.
    """
    return TTL_BY_CAPABILITY.get(capability, DEFAULT_TTL_SECONDS)


def _is_fresh(fetched_at: datetime, ttl_seconds: int | None) -> bool:
    """Age check. `NEVER_EXPIRES` (or None, `get`'s "no TTL asked for"
    default) means the row is fresh whatever its age."""
    if ttl_seconds is None or ttl_seconds == NEVER_EXPIRES:
        return True  # never-expire mode, asked for explicitly
    return _now() - fetched_at < timedelta(seconds=ttl_seconds)


def _read_row(capability: str, key: str) -> tuple[Any, datetime] | None:
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


def read_rows(capability: str, keys: Sequence[str]) -> dict[str, tuple[Any, datetime]]:
    """`{key: (payload, fetched_at)}` for every row that exists, in ONE query.

    The batch read behind a list of quotes: a per-key `_read_row` would be
    one SELECT per ticker on every page render. Absent keys are simply
    missing from the result. No TTL is applied; the caller owns freshness.
    """
    wanted = list(dict.fromkeys(keys))
    if not wanted:
        return {}
    with SessionLocal() as db:
        rows = db.execute(
            select(ProviderCache.key, ProviderCache.payload_json, ProviderCache.fetched_at).where(
                ProviderCache.capability == capability,
                ProviderCache.key.in_(wanted),
            )
        ).all()
    return {key: (payload, fetched_at) for key, payload, fetched_at in rows}


def _lookup(
    capability: str, key: str,
    *, ttl_seconds: int | None,
    serve_stale: bool,
    max_age_seconds: int | None,
) -> tuple[Any | None, int | None]:
    """The one place the fresh / stale / too-stale decision is made, so
    `get` and `cached_call` cannot drift apart on the age cap.

    Returns `(payload, age_seconds)`. `payload` is None when there is no
    row *or* the row was refused; `age_seconds` is None only when there
    is no row, which lets `cached_call` tell the two apart and log the
    age without a second read.
    """
    found = _read_row(capability, key)
    if found is None:
        return None, None
    payload, fetched_at = found
    age = int((_now() - fetched_at).total_seconds())
    if _is_fresh(fetched_at, ttl_seconds):
        return payload, age
    if not serve_stale:
        return None, age
    if max_age_seconds is not None and not _is_fresh(fetched_at, max_age_seconds):
        return None, age
    return payload, age


def get(
    capability: str, key: str,
    *, ttl_seconds: int | None = None,
    serve_stale: bool = False,
    max_age_seconds: int | None = None,
) -> Any | None:
    """Read the cached payload for `(capability, key)`.

    Unlike `cached_call`, this is a raw read: `ttl_seconds=None` here
    means "no age constraint asked for", so any row is returned. It does
    not consult `TTL_BY_CAPABILITY` — callers that want the capability's
    policy applied should go through `cached_call`, or pass
    `ttl_seconds=ttl_seconds_for(capability)`.

    Returns None when no row exists. When a row exists but is past
    `ttl_seconds`, returns None *unless* `serve_stale=True` (used as
    the last-resort fallback when the provider also missed). With
    `serve_stale=True`, `max_age_seconds` bounds how old that fallback
    may be; rows older than it are refused (None) as well.
    """
    payload, _age = _lookup(
        capability, key, ttl_seconds=ttl_seconds,
        serve_stale=serve_stale, max_age_seconds=max_age_seconds,
    )
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


def put_many(capability: str, payloads: Mapping[str, Any]) -> None:
    """Upsert several rows in one session and one commit.

    Empty payloads are skipped, as in `put`. The batch write behind a list
    of quotes; on `IntegrityError` (another process inserted one of these
    keys between our read and our insert) the batch is rolled back and each
    key goes through `put`, which already retries that race as an UPDATE.
    """
    items = {k: v for k, v in payloads.items() if v not in (None, [], {})}
    if not items:
        return
    try:
        with SessionLocal() as db:
            existing = {
                row.key: row for row in db.execute(
                    select(ProviderCache).where(
                        ProviderCache.capability == capability,
                        ProviderCache.key.in_(list(items)),
                    )
                ).scalars()
            }
            now = _now()
            for key, payload in items.items():
                row = existing.get(key)
                if row is None:
                    db.add(ProviderCache(
                        capability=capability, key=key, payload_json=payload, fetched_at=now,
                    ))
                else:
                    row.payload_json = payload
                    row.fetched_at = now
            db.commit()
    except IntegrityError:
        log.info(
            "provider_cache.put_many lost an insert race capability=%s keys=%d; "
            "falling back to per-key put", capability, len(items),
        )
        for key, payload in items.items():
            put(capability, key, payload)


def invalidate(capability: str, key: str | None = None) -> int:
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


def record_stale(kind: str, capability: str, key: str, age_seconds: int) -> None:
    """Write the monitoring row. A ledger hiccup must not take the data
    path down with it, so failures are logged rather than raised.

    Public because `quote_service` makes its own stale decision (calendar-
    aware) and must still show up in `stale_stats` for both processes."""
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


# Internal call sites and tests predate the public name.
_record_stale = record_stale


def cached_call(
    capability: str, key: str, fetcher: Callable[[], Any],
    *, ttl_seconds: int | None = None,
    force_refresh: bool = False,
) -> Any | None:
    """Read-through: cache hit → return; miss → call `fetcher`, write,
    return; provider miss → fall back to a stale cached row no older
    than `max_stale_seconds(capability)`, else None.

    `ttl_seconds` defaults to `ttl_seconds_for(capability)` — the table
    entry, or `DEFAULT_TTL_SECONDS` when the capability is unlisted.
    Pass an explicit number of seconds to override it, or the
    `NEVER_EXPIRES` sentinel for a payload that genuinely never goes
    stale. `None` means "decide for me", which is why it cannot also
    mean "never expire": that overload is what made every unlisted
    capability immortal.
    """
    if ttl_seconds is None and not force_refresh:
        ttl_seconds = ttl_seconds_for(capability)

    if not force_refresh:
        cached = get(capability, key, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached

    fresh = fetcher()
    if fresh is not None and fresh != [] and fresh != {}:
        put(capability, key, fresh)
        return fresh

    # Provider also missed — better stale than empty, up to a point.
    # Same decision path as `get(serve_stale=True, max_age_seconds=…)`.
    # ttl_seconds=0 because whatever TTL the caller asked for, a row we
    # consult after a provider miss is by definition the stale fallback
    # (the fresh check already failed or was skipped by force_refresh),
    # and only the age cap should decide.
    cap = max_stale_seconds(capability)
    payload, age = _lookup(
        capability, key, ttl_seconds=0, serve_stale=True, max_age_seconds=cap,
    )
    if age is None:
        return None
    if payload is None:
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


def stale_stats(window_hours: int = 24) -> dict[str, Any]:
    """Aggregate the stale-serve ledger over the trailing window.

    One bounded query; never raises — a failure returns zeros plus an
    `error` key so the status endpoint stays up when the DB doesn't.
    """
    result: dict[str, Any] = {
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

        by_cap: dict[str, dict[str, Any]] = result["by_capability"]
        oldest_served: int | None = None
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
