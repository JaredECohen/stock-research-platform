"""W5b — the NYSE regular-session calendar behind the 15-minute quote policy.

Pure-function tests: fixed dates, no clock, no DB. The expected dates are
the exchange's published 2025-2027 calendars; the half-day and DST cases
are the ones a rule-based calendar gets wrong first.
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.finance import market_calendar as cal

REPO = Path(__file__).resolve().parents[3]


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_2026_holidays_match_nyse():
    assert set(cal.nyse_holidays(2026)) == {
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
        date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
        date(2026, 11, 26), date(2026, 12, 25),
    }


def test_2027_observed_shifts():
    holidays = cal.nyse_holidays(2027)
    assert date(2027, 6, 18) in holidays      # Juneteenth falls on a Saturday
    assert date(2027, 7, 5) in holidays       # Independence Day on a Sunday
    assert date(2027, 12, 24) in holidays     # Christmas on a Saturday
    assert date(2027, 3, 26) in holidays      # Good Friday (Easter 2027-03-28)
    assert holidays[date(2027, 3, 26)] == "Good Friday"


def test_new_year_on_saturday_is_not_observed():
    # Jan 1 2028 is a Saturday: NYSE Rule 7.2 keeps Dec 31 2027 open.
    assert cal.is_trading_day(date(2027, 12, 31))
    assert date(2027, 12, 31) not in cal.nyse_holidays(2027)
    assert not any(d.year == 2028 and d.month == 1 and d.day <= 2 for d in cal.nyse_holidays(2028))


def test_early_closes():
    assert cal.early_close(date(2026, 11, 27)) == time(13, 0)
    assert cal.early_close(date(2026, 12, 24)) == time(13, 0)
    assert cal.early_close(date(2025, 7, 3)) == time(13, 0)
    # Jul 3 2026 is a Friday and the observed holiday, so Jul 2 is a full day.
    assert cal.early_close(date(2026, 7, 2)) is None
    assert cal.is_trading_day(date(2026, 7, 2))
    year_2027 = (date(2027, 1, 1) + timedelta(days=i) for i in range(365))
    halves_2027 = [d for d in year_2027 if cal.early_close(d) is not None]
    assert halves_2027 == [date(2027, 11, 26)]


def test_special_closure_2025_01_09():
    assert not cal.is_trading_day(date(2025, 1, 9))
    assert cal.closure_reason(date(2025, 1, 9)) == "National Day of Mourning (Carter)"
    assert cal.session_bounds(date(2025, 1, 9)) is None


def test_session_bounds_across_dst():
    # 2026-03-08 is the spring-forward Sunday: EST before, EDT after.
    assert cal.session_bounds(date(2026, 3, 6))[0] == _utc(2026, 3, 6, 14, 30)
    assert cal.session_bounds(date(2026, 3, 9))[0] == _utc(2026, 3, 9, 13, 30)
    # 2026-11-01 is the fall-back Sunday.
    assert cal.session_bounds(date(2026, 10, 30))[1] == _utc(2026, 10, 30, 20, 0)
    assert cal.session_bounds(date(2026, 11, 2))[0] == _utc(2026, 11, 2, 14, 30)
    # A half day closes at 13:00 ET.
    assert cal.session_bounds(date(2026, 11, 27))[1] == _utc(2026, 11, 27, 18, 0)


def test_next_open_skips_weekend_and_good_friday():
    # Thu 2026-04-02 17:00 ET (21:00 UTC) -> Fri is Good Friday -> Mon 09:30 ET.
    assert cal.next_open(datetime(2026, 4, 2, 21, 0)) == _utc(2026, 4, 6, 13, 30)


def test_last_session_and_market_state_reasons():
    # Tuesday 10:45 ET: open.
    state = cal.market_state(datetime(2026, 9, 22, 14, 45))
    assert state.is_open and state.reason == "open"
    assert state.session_date == date(2026, 9, 22)
    # Tuesday 08:00 ET: pre-open, the last session is Monday's.
    state = cal.market_state(datetime(2026, 9, 22, 12, 0))
    assert not state.is_open and state.reason == "pre_open"
    assert state.session_date == date(2026, 9, 21)
    assert state.next_open_utc == _utc(2026, 9, 22, 13, 30)
    # Tuesday 17:00 ET: after the close.
    assert cal.market_state(datetime(2026, 9, 22, 21, 0)).reason == "after_close"
    # Labor Day and a Saturday.
    labor = cal.market_state(datetime(2026, 9, 7, 15, 0))
    assert labor.reason == "holiday:Labor Day" and labor.session_date == date(2026, 9, 4)
    assert cal.market_state(datetime(2026, 9, 26, 15, 0)).reason == "weekend"
    # The day after Thanksgiving is flagged as a half day while it is the session.
    assert cal.market_state(datetime(2026, 11, 27, 15, 0)).early_close is True


def test_aware_datetime_raises():
    with pytest.raises(ValueError):
        cal.next_open(_utc(2026, 9, 22, 14, 0))
    with pytest.raises(ValueError):
        cal.market_state(_utc(2026, 9, 22, 14, 0))


def test_unverified_year_warns_once_and_answers(caplog, monkeypatch):
    monkeypatch.setattr(cal, "_WARNED_YEARS", set())
    with caplog.at_level(logging.WARNING, logger=cal.__name__):
        first = cal.nyse_holidays(2030)
        cal.nyse_holidays(2030)
        assert cal.is_trading_day(date(2030, 7, 5))
    assert date(2030, 7, 4) in first
    warnings = [r for r in caplog.records if "computed by rule for 2030" in r.getMessage()]
    assert len(warnings) == 1


def test_zoneinfo_available():
    """tzdata is present wherever the suite runs, which is the property the
    image relies on (the Dockerfile checks the same thing at build)."""
    assert ZoneInfo("America/New_York").utcoffset(datetime(2026, 7, 1)).total_seconds() == -4 * 3600


def test_calendar_unavailable_is_a_typed_error(monkeypatch):
    """A missing tzdata surfaces as `CalendarUnavailable` at the call, so
    quote consumers can catch it and fall back to the stored close."""
    def boom(_key):
        raise cal.ZoneInfoNotFoundError("no tzdata")

    cal._et.cache_clear()
    monkeypatch.setattr(cal, "ZoneInfo", boom)
    try:
        with pytest.raises(cal.CalendarUnavailable):
            cal.market_state(datetime(2026, 9, 22, 14, 0))
    finally:
        monkeypatch.undo()
        cal._et.cache_clear()
    assert cal.market_state(datetime(2026, 9, 22, 14, 0)).is_open


def test_tzdata_is_an_unconditional_dependency_and_checked_at_build():
    """The lock used to carry tzdata only on win32; `python:3.12-slim`
    happens to ship zone files, but nothing guaranteed it. Declared
    unconditionally, locked, and checked in the image build."""
    pyproject = (REPO / "backend" / "pyproject.toml").read_text()
    assert re.search(r'^\s*"tzdata>=[0-9.]+",', pyproject, re.MULTILINE), "pyproject lacks an unconditional tzdata"
    lock = (REPO / "backend" / "requirements.txt").read_text()
    tz_lines = [ln for ln in lock.splitlines() if ln.startswith("tzdata==")]
    assert tz_lines and ";" not in tz_lines[0], f"tzdata is not locked unconditionally: {tz_lines}"
    dockerfile = (REPO / "Dockerfile").read_text()
    assert "zoneinfo.ZoneInfo('America/New_York')" in dockerfile
