"""NYSE regular-session calendar (W5b). Pure: no DB, no network.

Live quotes are cached for ~15 minutes *while the market is open* and until
the next open once it has closed (`services/quote_service.py`). That needs
to know when the market is open, and nothing in the codebase did:
`price_history_service._last_weekday` is explicitly "not an exchange
calendar". A calendar library would be a dependency for ten dates a year,
so the rules are written out here, with the unscheduled closures in an
exception table beside them.

Conventions, because mixed ones are how DST bugs start:

* Every function that takes an instant takes **naive UTC**, the persisted
  convention (`provider_cache.fetched_at`, every `DateTime` column). An
  aware datetime raises `ValueError` rather than being guessed at.
* Every instant returned is **aware UTC**, so an API that serialises it
  emits an explicit offset and a browser cannot read it as local time.
* `America/New_York` is loaded lazily. When tzdata is missing (a base-image
  change) the calendar raises `CalendarUnavailable` at the call, not at
  import, so a quote consumer can degrade to the stored close instead of
  failing to import.

Years after `CALENDAR_VERIFIED_THROUGH` are still answered by rule, with one
WARNING per year per process: a quote label must never 500, and the warning
is the prompt to re-verify against nyse.com/markets/hours-calendars. There
is no wall-clock time bomb for CI.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger(__name__)

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

# Re-verify the rule output (and the tables below) against the exchange's
# published calendar each year, then bump this.
CALENDAR_VERIFIED_THROUGH = 2027

# Unscheduled full closures: not derivable by rule. Keep the source beside
# each entry when one is added.
SPECIAL_CLOSURES: dict[date, str] = {
    date(2012, 10, 29): "Hurricane Sandy",
    date(2012, 10, 30): "Hurricane Sandy",
    date(2018, 12, 5): "National Day of Mourning (G.H.W. Bush)",
    date(2025, 1, 9): "National Day of Mourning (Carter)",
}
# Unscheduled early closes. None are known; the table exists so one can be
# added without touching the rules.
SPECIAL_EARLY_CLOSES: dict[date, time] = {}

# Bound on the day-walk in `next_open` / `last_session`. The longest run of
# non-trading days on record is a handful; this is a guard against a rule
# bug looping forever, not a real limit.
_MAX_WALK_DAYS = 31

_WARN_LOCK = threading.Lock()
_WARNED_YEARS: set[int] = set()


class CalendarUnavailable(RuntimeError):
    """The exchange time zone could not be loaded (tzdata missing)."""


@lru_cache(maxsize=1)
def _et() -> ZoneInfo:
    try:
        return ZoneInfo("America/New_York")
    except (ZoneInfoNotFoundError, OSError, ValueError) as exc:
        raise CalendarUnavailable(
            "America/New_York could not be loaded; is tzdata installed?"
        ) from exc


def _require_naive(value: datetime, name: str) -> datetime:
    if value.tzinfo is not None:
        raise ValueError(f"{name} must be naive UTC, got an aware datetime")
    return value


def _aware(naive_utc: datetime) -> datetime:
    return naive_utc.replace(tzinfo=UTC)


def et_date(naive_utc: datetime) -> date:
    """The America/New_York calendar date of a naive-UTC instant."""
    _require_naive(naive_utc, "et_date")
    return _aware(naive_utc).astimezone(_et()).date()


def _warn_if_unverified(year: int) -> None:
    if year <= CALENDAR_VERIFIED_THROUGH:
        return
    with _WARN_LOCK:
        if year in _WARNED_YEARS:
            return
        _WARNED_YEARS.add(year)
    log.warning(
        "NYSE calendar computed by rule for %d; unscheduled closures unknown "
        "(verified through %d)", year, CALENDAR_VERIFIED_THROUGH,
    )


def easter(year: int) -> date:
    """Western (Gregorian) Easter Sunday: the anonymous Meeus/Butcher algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    """Saturday → the Friday before, Sunday → the Monday after."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def nyse_holidays(year: int) -> dict[date, str]:
    """Scheduled full-day closures for `year`, by rule (special closures excluded)."""
    _warn_if_unverified(year)
    out: dict[date, str] = {}
    new_year = date(year, 1, 1)
    # NYSE Rule 7.2: a Saturday New Year's Day is not observed (Dec 31 of
    # the prior year trades); a Sunday one moves to Monday.
    if new_year.weekday() == 6:
        out[new_year + timedelta(days=1)] = "New Year's Day"
    elif new_year.weekday() < 5:
        out[new_year] = "New Year's Day"
    out[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    out[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    out[easter(year) - timedelta(days=2)] = "Good Friday"
    out[_last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        out[_observed(date(year, 6, 19))] = "Juneteenth"
    out[_observed(date(year, 7, 4))] = "Independence Day"
    out[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    out[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    out[_observed(date(year, 12, 25))] = "Christmas Day"
    return out


def closure_reason(day: date) -> str | None:
    """Why `day` has no session: a holiday or special-closure name, "weekend",
    or None when it is a trading day."""
    if day in SPECIAL_CLOSURES:
        return SPECIAL_CLOSURES[day]
    name = nyse_holidays(day.year).get(day)
    if name:
        return name
    if day.weekday() >= 5:
        return "weekend"
    return None


def is_trading_day(day: date) -> bool:
    return closure_reason(day) is None


def early_close(day: date) -> time | None:
    """13:00 ET on a half day, None on a full day or a non-trading day."""
    if not is_trading_day(day):
        return None
    if day in SPECIAL_EARLY_CLOSES:
        return SPECIAL_EARLY_CLOSES[day]
    # Jul 3 is a half day only Mon-Thu: a Friday Jul 3 is itself the
    # observed Independence Day (is_trading_day already said no).
    if day.month == 7 and day.day == 3 and day.weekday() < 4:
        return EARLY_CLOSE
    if day == _nth_weekday(day.year, 11, 3, 4) + timedelta(days=1):
        return EARLY_CLOSE
    if day.month == 12 and day.day == 24:
        return EARLY_CLOSE
    return None


def session_bounds(day: date) -> tuple[datetime, datetime] | None:
    """(open, close) of `day`'s regular session as aware UTC, or None."""
    if not is_trading_day(day):
        return None
    et = _et()
    close_t = early_close(day) or REGULAR_CLOSE
    open_at = datetime.combine(day, REGULAR_OPEN, tzinfo=et).astimezone(UTC)
    close_at = datetime.combine(day, close_t, tzinfo=et).astimezone(UTC)
    return open_at, close_at


def next_open(after_utc: datetime) -> datetime:
    """The first session open strictly after `after_utc` (naive UTC in, aware UTC out)."""
    _require_naive(after_utc, "next_open")
    after = _aware(after_utc)
    start = et_date(after_utc)
    for offset in range(_MAX_WALK_DAYS):
        bounds = session_bounds(start + timedelta(days=offset))
        if bounds is not None and bounds[0] > after:
            return bounds[0]
    raise CalendarUnavailable(f"no NYSE session within {_MAX_WALK_DAYS} days of {after_utc}")


def last_session(on_or_before_utc: datetime) -> tuple[date, datetime, datetime]:
    """The most recent session that has OPENED at `on_or_before_utc`: today's
    once 09:30 ET has passed, else the previous trading day's."""
    _require_naive(on_or_before_utc, "last_session")
    at = _aware(on_or_before_utc)
    start = et_date(on_or_before_utc)
    for offset in range(_MAX_WALK_DAYS):
        day = start - timedelta(days=offset)
        bounds = session_bounds(day)
        if bounds is not None and bounds[0] <= at:
            return day, bounds[0], bounds[1]
    raise CalendarUnavailable(f"no NYSE session within {_MAX_WALK_DAYS} days before {on_or_before_utc}")


@dataclass(frozen=True)
class MarketState:
    is_open: bool                 # within [open, close) of a trading day
    session_date: date            # most recent session that has opened (today when open)
    session_open_utc: datetime
    session_close_utc: datetime
    early_close: bool
    next_open_utc: datetime
    reason: str                   # "open" | "pre_open" | "after_close" | "weekend" | "holiday:<name>"


def market_state(now_utc: datetime) -> MarketState:
    _require_naive(now_utc, "market_state")
    now = _aware(now_utc)
    today = et_date(now_utc)
    session_day, open_at, close_at = last_session(now_utc)
    is_open = open_at <= now < close_at
    if is_open:
        reason = "open"
    else:
        closed_for = closure_reason(today)
        if closed_for == "weekend":
            reason = "weekend"
        elif closed_for is not None:
            reason = f"holiday:{closed_for}"
        elif session_day != today:
            reason = "pre_open"
        else:
            reason = "after_close"
    return MarketState(
        is_open=is_open,
        session_date=session_day,
        session_open_utc=open_at,
        session_close_utc=close_at,
        early_close=early_close(session_day) is not None,
        next_open_utc=next_open(now_utc),
        reason=reason,
    )
