"""FIX-005: fundamentals load as often as they change (W5a S6).

Owner decision 2026-09-24: "fundamental data should load as often as it
changes, i.e. quarterly results queried quarterly." These tests drive the
filing-driven refresh with a frozen, advanced clock and a fake named
provider chain; nothing reaches the network and nothing asserts wall time.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Company, FinancialPeriod, FundamentalRefreshState
from app.services import fundamental_history_service as fhs
from app.services import fundamental_refresh as fr
from app.services import market_data_backfill as backfill
from app.services import provider_cache

T = "TEST"
START = datetime(2026, 8, 1, 10, 0)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.ticks = 0.0

    def night(self, day: date) -> None:
        """03:15 UTC on `day`, when `history_backfill` drains."""
        self.now = datetime(day.year, day.month, day.day, 3, 15)


@pytest.fixture
def env(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    for module in (fhs, fr, backfill, provider_cache):
        monkeypatch.setattr(module, "SessionLocal", factory)
    clock = Clock(START)
    monkeypatch.setattr(fr, "_now", lambda: clock.now)
    monkeypatch.setattr(fhs, "_now", lambda: clock.now)
    monkeypatch.setattr(fhs, "_today", lambda: clock.now.date())
    monkeypatch.setattr(fr, "_sleep", lambda s: None)
    monkeypatch.setattr(fr, "_monotonic", lambda: clock.ticks)
    invalidated: list[tuple] = []
    import app.cache.snapshots as snapshots
    monkeypatch.setattr(snapshots, "invalidate", lambda subject, kind=None, **k: invalidated.append((subject, kind)) or 1)
    with factory() as db:
        db.add(Company(ticker=T, company_name="Test Co", sector="Test", industry="Test", universe_tier="auto_analysis"))
        db.commit()
    yield SimpleNamespace(factory=factory, clock=clock, invalidated=invalidated, engine=engine)
    engine.dispose()


def quarters(through: date, *, first: date = date(2022, 3, 30)) -> list[tuple[str, date]]:
    out = []
    for y in range(first.year, through.year + 1):
        for q in (1, 2, 3, 4):
            end = date(y, q * 3, 30)
            if first <= end <= through:
                out.append((f"{y}Q{q}", end))
    return out


def statements(*, through: date = date(2026, 3, 30), annual_through: int = 2025, value: float = 100,
               first: date = date(2022, 3, 30), extra: list[tuple[str, date]] = ()) -> dict:
    periods = [(f"FY{y}", date(y, 12, 31)) for y in range(first.year, annual_through + 1)]
    periods += quarters(through, first=first) + list(extra)
    return {s: [{"period": p, "period_end": d.isoformat(), "filing_date": (d + timedelta(days=30)).isoformat(),
                 "currency": "USD", primary: value} for p, d in periods] for s, primary in fhs.PRIMARY.items()}


def chain(monkeypatch, *providers):
    """Fake financials chain; each entry is (name, payload | callable | Exception)."""
    calls: list[tuple[str, str]] = []
    built = []
    for name, rows in providers:
        def fetch(symbol, start, name=name, rows=rows):
            calls.append((name, symbol))
            data = rows() if callable(rows) else rows
            if isinstance(data, Exception):
                raise data
            return data
        built.append(SimpleNamespace(name=name, get_financial_history=fetch))
    monkeypatch.setattr(fhs, "get_data_service", lambda: SimpleNamespace(_live_chain=lambda cap: built))
    return calls


def ten_q(end: date, filed: date, acc: str = "0000000001-26-000100", form: str = "10-Q") -> dict:
    return {"type": form, "period_end": end.isoformat(), "filing_date": filed.isoformat(), "accession_number": acc}


def state(env, ticker: str = T) -> dict:
    with env.factory() as db:
        row = db.execute(select(FundamentalRefreshState.__table__)
                         .where(FundamentalRefreshState.ticker == ticker)).mappings().first()
        return dict(row) if row else {}


def seed(env, monkeypatch, **kwargs) -> None:
    chain(monkeypatch, ("fmp", statements(**kwargs)))
    report = fhs.backfill_fundamentals(T, date(2024, 8, 1), force_refresh=True)
    assert report["committed"], report["issues"]


def nights(env, first: date, count: int) -> list[dict]:
    out = []
    for n in range(count):
        env.clock.night(first + timedelta(days=n))
        out.append(fr.drain())
    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_due_times_round_down_to_the_drain_slot():
    assert fr.drain_slot(datetime(2026, 8, 2, 3, 20)) == datetime(2026, 8, 2, 3, 0)
    assert fr.drain_slot(datetime(2026, 8, 2, 4, 0)) == datetime(2026, 8, 2, 3, 0)
    assert fr.drain_slot(datetime(2026, 8, 2, 4, 1)) == datetime(2026, 8, 2, 4, 1)
    assert fr.drain_slot(datetime(2026, 8, 2, 2, 59)) == datetime(2026, 8, 2, 2, 59)


def test_observe_records_filed_periods_and_schedules_only_missing(env, monkeypatch):
    """State-based detection: a deploy against a fully imported database
    schedules nothing; only a filed period that is not stored schedules."""
    seed(env, monkeypatch, through=date(2026, 6, 30))
    assert fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 30))])]) == []
    current = state(env)
    assert current["status"] == "idle" and current["filed_quarter_end"] == date(2026, 6, 30)
    assert current["quarter_observed_at"] == START and current["filed_quarter_form"] == "10-Q"

    with env.factory() as db:  # a company whose newest stored quarter is Q1
        db.add(Company(ticker="LAGS", company_name="Lags", sector="Test", industry="Test", universe_tier="auto_analysis"))
        db.add(FinancialPeriod(ticker="LAGS", period="2026Q1", statement="income", line_item="revenue", value=1,
                               period_end=date(2026, 3, 30), fiscal_year=2026, fiscal_quarter=1, source="fmp",
                               currency="USD"))
        db.commit()
    assert fr.observe_many([("LAGS", [ten_q(date(2026, 6, 30), date(2026, 7, 30))])]) == []
    lagging = state(env, "LAGS")
    assert (lagging["status"], lagging["trigger"], lagging["attempts"]) == ("pending", "filing", 0)
    assert lagging["due_at"] == lagging["first_due_at"] == START + fr.PUBLICATION_LAG


def test_observe_ignores_body_only_forms_and_is_idempotent(env, monkeypatch):
    seed(env, monkeypatch)
    assert fr.observe_many([(T, [{"type": "8-K", "period_end": "2026-07-29", "filing_date": "2026-07-30",
                                  "accession_number": "0000000001-26-000099"}])]) == []
    assert state(env) == {}
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 30))])])
    first = state(env)
    env.clock.now += timedelta(hours=2)
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 30))])])
    again = state(env)
    assert again["due_at"] == first["due_at"] and again["row_version"] == first["row_version"]


def test_amendment_schedules_one_reverification_but_not_on_first_observation(env, monkeypatch):
    seed(env, monkeypatch, through=date(2026, 6, 30))
    filed = ten_q(date(2026, 6, 30), date(2026, 7, 30))
    amend = {"type": "10-Q/A", "period_end": "2026-06-30", "filing_date": "2026-08-01",
             "accession_number": "0000000001-26-000200"}
    fr.observe_many([(T, [amend, filed])])
    assert state(env)["status"] == "idle" and state(env)["last_amendment_accession"] == amend["accession_number"]
    newer = {**amend, "accession_number": "0000000001-26-000300", "filing_date": "2026-08-03"}
    fr.observe_many([(T, [newer, amend, filed])])
    pending = state(env)
    assert (pending["status"], pending["trigger"]) == ("pending", "amendment")
    assert pending["due_at"] == fr.drain_slot(START + fhs.PUBLICATION_GRACE) and pending["first_due_at"] is None
    calls = chain(monkeypatch, ("fmp", statements(through=date(2026, 6, 30))))
    env.clock.night(date(2026, 8, 5))
    assert fr.drain()["refreshed"] == 1 and len(calls) == 1
    assert state(env)["status"] == "idle"  # single shot: an amendment has no retries
    env.clock.night(date(2026, 8, 6))
    assert fr.drain()["refreshed"] == 0 and len(calls) == 1


def test_25nse_on_active_filer_changes_neither_schedule_nor_coverage(env, monkeypatch):
    """Critique: a 25-NSE can delist one class of notes of an issuer that
    keeps filing. It is evidence only; only the registry ends reporting."""
    seed(env, monkeypatch, through=date(2026, 6, 30))
    filed = ten_q(date(2026, 6, 30), date(2026, 7, 30))
    fr.observe_many([(T, [filed])])
    before_state = state(env)
    before = fhs.fundamental_coverage(T, date(2024, 8, 1))
    notice = {"type": "25-NSE", "period_end": None, "filing_date": "2026-08-01",
              "accession_number": "0000000001-26-000400"}
    fr.observe_many([(T, [notice, filed])])
    after_state = state(env)
    assert after_state["last_deregistration_form"] == "25-NSE"
    assert after_state["last_deregistration_on"] == date(2026, 8, 1)
    for key in ("status", "trigger", "due_at", "attempts", "filed_quarter_end", "filed_annual_end"):
        assert after_state[key] == before_state[key]
    after = fhs.fundamental_coverage(T, date(2024, 8, 1))
    assert after == before and after["reporting_ended_on"] is None and after["success"]
    # A later, missing 10-Q still schedules the issuer as usual.
    fr.observe_many([(T, [ten_q(date(2026, 9, 30), date(2026, 10, 30), acc="0000000001-26-000500"), notice])])
    assert state(env)["status"] == "pending" and state(env)["trigger"] == "filing"


def test_calendar_request_upgraded_by_new_10q(env, monkeypatch):
    seed(env, monkeypatch)
    fr.request(T, "calendar", due_at=START + timedelta(days=3))
    assert state(env)["trigger"] == "calendar" and state(env)["first_due_at"] is None
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 30))])])
    upgraded = state(env)
    assert (upgraded["status"], upgraded["trigger"], upgraded["attempts"]) == ("pending", "filing", 0)
    assert upgraded["first_due_at"] == START + fr.PUBLICATION_LAG == upgraded["due_at"]


# ---------------------------------------------------------------------------
# Drain: lag, retries, secondary, abandonment
# ---------------------------------------------------------------------------

def test_drain_waits_for_publication_lag(env, monkeypatch):
    seed(env, monkeypatch)
    calls = chain(monkeypatch, ("fmp", statements(through=date(2026, 6, 30))))
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 8, 1))])])
    env.clock.now = START + timedelta(hours=5)
    assert fr.drain()["refreshed"] == 0 and calls == []
    env.clock.now = START + timedelta(hours=6)
    result = fr.drain()
    assert result["refreshed"] == result["satisfied"] == 1 and calls == [("fmp", T)]
    assert state(env)["status"] == "idle" and state(env)["last_result"]["satisfied"]


def available(env) -> dict:
    with env.factory() as db:
        return {r.id: (r.available_at, r.available_at_source, r.value) for r in
                db.execute(select(FinancialPeriod).where(FinancialPeriod.ticker == T)).scalars()}


def test_retry_backoff_until_fmp_publishes_then_invalidates_caches(env, monkeypatch):
    seed(env, monkeypatch)
    before = available(env)
    provider_cache.put("financials", T, {"income": [{"period": "FY2025"}]})
    lagging = statements()
    published = statements(through=date(2026, 6, 30))
    current = {"payload": lagging}
    chain(monkeypatch, ("fmp", lambda: current["payload"]))
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 31))])])

    env.clock.night(date(2026, 8, 2))
    first = fr.drain()
    assert first["refreshed"] == 1 and first["pending"] == 1 and first["errors"] == []
    assert first["missing"] == ["TEST:quarterly:2026-06-30(10-Q filed 2026-07-31, attempt 1, next 2026-08-02T15:15)"]
    row = state(env)
    assert (row["attempts"], row["due_at"]) == (1, datetime(2026, 8, 2, 15, 15))
    env.clock.night(date(2026, 8, 3))
    fr.drain()
    # +24h from 03:15 rounds down to the 03:00 drain slot instead of missing a night.
    assert (state(env)["attempts"], state(env)["due_at"]) == (2, datetime(2026, 8, 4, 3, 0))
    assert env.invalidated == [] and provider_cache.get("financials", T) is not None

    env.clock.now = START + timedelta(hours=73)
    blocked = fhs.fundamental_coverage(T, date(2024, 8, 1))
    assert not blocked["success"]
    assert {i["kind"] for i in blocked["issues"] if i["kind"] == "expected_period_missing"} == {"expected_period_missing"}

    current["payload"] = published
    env.clock.night(date(2026, 8, 5))
    done = fr.drain()
    assert done["satisfied"] == 1 and done["pending"] == 0
    assert (state(env)["status"], state(env)["attempts"], state(env)["due_at"]) == ("idle", 0, None)
    assert provider_cache.get("financials", T) is None
    assert env.invalidated == [(T, "company_cold")]
    assert state(env)["last_result"]["cache_invalidation"]["financials"]["rows_removed"] == 1
    after = available(env)
    assert all(after[i] == before[i] for i in before), "a refresh re-dated a stored observation"
    assert fhs.fundamental_coverage(T, date(2024, 8, 1))["success"]


def test_secondary_fills_filed_period_only_after_retry_window(env, monkeypatch):
    """Nights 1, 2, 3, 5 ask FMP only; night 9 (the fifth attempt) may let
    the fallback fill exactly the filed period FMP still lacks."""
    seed(env, monkeypatch)
    calls = chain(monkeypatch, ("fmp", statements()),
                  ("alpha_vantage", statements(through=date(2026, 6, 30), value=100)))
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 31))])])
    results = nights(env, date(2026, 8, 2), 9)
    ran = [n + 1 for n, r in enumerate(results) if r["refreshed"]]
    assert ran == [1, 2, 3, 5, 9]
    assert [c for c in calls if c[0] == "alpha_vantage"] == [("alpha_vantage", T)]
    assert calls[-1] == ("alpha_vantage", T) and len(calls) == 6
    assert state(env)["status"] == "idle" and results[-1]["satisfied"] == 1
    with env.factory() as db:
        filled = db.execute(select(FinancialPeriod.source).where(
            FinancialPeriod.ticker == T, FinancialPeriod.period == "2026Q2")).scalars().all()
        in_fmp_periods = db.execute(select(func.count()).select_from(FinancialPeriod).where(
            FinancialPeriod.ticker == T, FinancialPeriod.source == "alpha_vantage",
            FinancialPeriod.period != "2026Q2")).scalar_one()
    assert set(filled) == {"alpha_vantage"} and in_fmp_periods == 0


def test_filing_retries_abandon_after_45_days_and_stay_named(env, monkeypatch):
    seed(env, monkeypatch)
    calls = chain(monkeypatch, ("fmp", statements()))
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 31))])])
    results = nights(env, date(2026, 8, 2), 60)
    ran = [n + 1 for n, r in enumerate(results) if r["refreshed"]]
    assert ran == [1, 2, 3, 5, 9, 16, 23, 30, 37, 44, 51]
    assert len(calls) == len(ran)
    assert results[8]["stuck"] == [T]  # night 9: stuck once
    assert all(T not in r["stuck"] for r in results[9:])
    assert all(r["still_missing"] == [f"{T}(since 2026-08-10)"] for r in results[9:])
    final = state(env)
    assert final["status"] == "idle" and final["due_at"] is None and final["last_result"]["abandoned"]
    coverage = fhs.fundamental_coverage(T, date(2024, 8, 1))
    bucket = coverage["coverage"]["income"]["quarterly"]
    # Still blocking (by now the 140-day period guard fires as well).
    assert not coverage["success"] and not bucket["period_current"] and not bucket["fresh"]


def test_drain_is_bounded_ordered_and_names_over_cap(env, monkeypatch):
    chain(monkeypatch, ("fmp", statements()))
    order: list[str] = []
    monkeypatch.setattr(fr, "_run", lambda t, s: order.append(t) or {"success": True, "attempts": [], "coverage": {}})
    triggers = ["sweep", "calendar", "admin", "first_contact", "filing"]
    expected = []
    for i in range(35):
        ticker = f"Q{i:02d}"
        fr.request(ticker, triggers[i % 5], due_at=START - timedelta(hours=i))
        expected.append((fr.TRIGGER_PRIORITY[triggers[i % 5]], START - timedelta(hours=i), ticker))
    expected = [t for *_, t in sorted(expected)]
    result = fr.drain()
    assert result["refreshed"] == 30 and order == expected[:30]
    assert result["over_cap"] == expected[30:]
    # The wall-clock budget names what it did not reach, too.
    order.clear()
    ticks = iter(range(0, 10_000, 400))
    monkeypatch.setattr(fr, "_monotonic", lambda: next(ticks))
    budget = fr.drain()
    assert budget["refreshed"] == 2 and budget["over_budget"] and budget["over_cap"] == []


def test_no_refetch_for_30_days_without_a_filing(env, monkeypatch):
    """The 7-day re-verification contract is gone: an unchanged issuer is
    not re-fetched and its coverage stays successful."""
    seed(env, monkeypatch, through=date(2026, 6, 30))
    calls = chain(monkeypatch, ("fmp", statements(through=date(2026, 6, 30))))
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 30))])])
    for n in range(30):
        env.clock.night(date(2026, 8, 2) + timedelta(days=n))
        result = fr.nightly()
        assert result["refreshed"] == 0 and result["scheduled"] == {"calendar": [], "sweep": [], "first_import": []}
    assert calls == []
    assert fhs.fundamental_coverage(T, date(2024, 8, 31))["success"]


def test_calendar_fallback_for_foreign_filer_rechecks_at_most_weekly(env, monkeypatch):
    seed(env, monkeypatch, through=date(2025, 6, 30), annual_through=2024)
    with env.factory() as db:
        db.add(FundamentalRefreshState(ticker=T, filed_annual_end=date(2024, 12, 31), filed_annual_form="20-F",
                                       annual_observed_at=START))
        db.commit()
    calls = chain(monkeypatch, ("fmp", statements(through=date(2025, 6, 30), annual_through=2024)))
    per_night = []
    for n in range(8):
        env.clock.night(date(2026, 6, 1) + timedelta(days=n))
        result = fr.nightly()
        per_night.append((len(calls), result["scheduled"]["calendar"]))
        assert result["errors"] == []
    assert [c for c, _ in per_night] == [1, 1, 1, 1, 1, 1, 1, 2]
    assert per_night[0][1] == [T] and per_night[7][1] == [T]
    assert state(env)["last_calendar_check_at"] == datetime(2026, 6, 8, 3, 15)


def test_out_of_tier_company_scheduled_by_calendar(env, monkeypatch):
    """Critique: scheduling covers every fundamentals_required company, not
    only the polled `auto_analysis` tier."""
    seed(env, monkeypatch, through=date(2025, 6, 30), annual_through=2024)
    with env.factory() as db:
        db.get(Company, T).universe_tier = "analyzed_on_demand"
        db.commit()
    calls = chain(monkeypatch, ("fmp", statements(through=date(2025, 6, 30), annual_through=2024)))
    env.clock.night(date(2026, 6, 1))
    result = fr.nightly()
    assert result["scheduled"]["calendar"] == [T] and result["refreshed"] == 1 and calls == [("fmp", T)]


def test_sweep_only_after_120_days_without_confirmation(env, monkeypatch):
    seed(env, monkeypatch, through=date(2026, 6, 30))
    calls = chain(monkeypatch, ("fmp", statements(through=date(2026, 6, 30))))
    with env.factory() as db:
        for row in db.execute(select(FinancialPeriod)).scalars():
            row.fetched_at = datetime(2026, 4, 14, 3, 15)
        db.commit()
    env.clock.night(date(2026, 8, 11))  # 119 days
    assert fr.nightly()["scheduled"]["sweep"] == [] and calls == []
    env.clock.night(date(2026, 8, 12))  # 120 days
    swept = fr.nightly()
    assert swept["scheduled"]["sweep"] == [T] and swept["refreshed"] == 1 and len(calls) == 1
    env.clock.night(date(2026, 8, 13))  # the refresh re-confirmed every line
    assert fr.nightly()["scheduled"]["sweep"] == [] and len(calls) == 1


def test_reporting_ended_skips_scheduling_and_judges_coverage_to_end_date(env, monkeypatch):
    monkeypatch.setitem(fhs.KNOWN_REPORTING_ENDED, T, (date(2025, 7, 17), "test evidence"))
    env.clock.now = datetime(2026, 9, 13, 12)
    seed(env, monkeypatch, through=date(2025, 3, 30), annual_through=2024)
    calls = chain(monkeypatch, ("fmp", statements(through=date(2025, 3, 30), annual_through=2024)))
    fr.observe_many([(T, [ten_q(date(2025, 6, 30), date(2025, 7, 30))])])
    assert state(env)["status"] == "idle"
    env.clock.night(date(2026, 9, 14))
    result = fr.nightly()
    assert result["scheduled"] == {"calendar": [], "sweep": [], "first_import": []} and calls == []
    coverage = fhs.fundamental_coverage(T, date(2024, 9, 14))
    assert coverage["success"], coverage["issues"]
    assert coverage["reporting_ended_on"] == coverage["coverage_end"] == "2025-07-17"


def test_reporting_ended_coverage_after_3_years(env, monkeypatch):
    """Critique: the window's start is clamped too, so an acquired issuer
    does not turn blocking again once the requested start passes its last
    period (ANSS around mid-2027)."""
    monkeypatch.setitem(fhs.KNOWN_REPORTING_ENDED, T, (date(2025, 7, 17), "test evidence"))
    env.clock.now = datetime(2025, 8, 1, 12)
    seed(env, monkeypatch, through=date(2025, 3, 30), annual_through=2024)
    env.clock.now = datetime(2028, 7, 17, 12)
    coverage = fhs.fundamental_coverage(T, date(2026, 7, 17))
    assert coverage["success"], coverage["issues"]
    assert coverage["coverage_start"] < "2025-07-17" and coverage["coverage_end"] == "2025-07-17"
    monkeypatch.delitem(fhs.KNOWN_REPORTING_ENDED, T)
    assert not fhs.fundamental_coverage(T, date(2026, 7, 17))["success"]


# ---------------------------------------------------------------------------
# Lease, failures, demo, pull-through, admin, CAS
# ---------------------------------------------------------------------------

def test_lease_prevents_concurrent_refresh(env, monkeypatch):
    seed(env, monkeypatch)
    calls = chain(monkeypatch, ("fmp", statements(through=date(2026, 6, 30))))
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 31))])])
    env.clock.night(date(2026, 8, 2))
    assert fr.claim(T) and not fr.claim(T)
    result = fr.drain()
    assert result["leased"] == [T] and result["refreshed"] == 0 and calls == []
    assert fr.refresh_if_due(T) is None and calls == []
    env.clock.now += fr.LEASE + timedelta(minutes=1)  # an abandoned lease expires
    assert fr.drain()["refreshed"] == 1 and len(calls) == 1
    assert state(env)["lease_until"] is None


def test_provider_failure_keeps_values_and_backs_off(env, monkeypatch):
    seed(env, monkeypatch)
    with env.factory() as db:
        before = {r.id: (r.value, r.fetched_at, r.available_at) for r in db.execute(select(FinancialPeriod)).scalars()}
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 31))])])
    chain(monkeypatch, ("fmp", RuntimeError("provider down")))
    env.clock.night(date(2026, 8, 2))
    failed = fr.drain()
    assert failed["pending"] == 1 and failed["missing"][0].startswith("TEST:quarterly:2026-06-30")
    assert (state(env)["attempts"], state(env)["due_at"]) == (1, datetime(2026, 8, 2, 15, 15))
    assert state(env)["last_result"]["issue_kinds"]["provider_error"] == 1
    denied = {"income": [], "balance": [], "cash": [], "_history_issues": [
        {"kind": "provider_entitlement_denied", "status": 402, "endpoint": "/income-statement",
         "statement": "income", "cadence": "quarterly"}]}
    chain(monkeypatch, ("fmp", denied))
    env.clock.night(date(2026, 8, 3))
    refused = fr.drain()
    assert refused["entitlement_denied"] == ["TEST:/income-statement:quarterly:402"]
    monkeypatch.setattr(fr, "_start_for", lambda t: (_ for _ in ()).throw(RuntimeError("db gone")))
    env.clock.night(date(2026, 8, 4))
    crashed = fr.drain()
    assert crashed["errors"] == ["TEST:RuntimeError"] and state(env)["attempts"] == 3
    assert state(env)["status"] == "pending" and state(env)["due_at"] == datetime(2026, 8, 6, 3, 0)
    with env.factory() as db:
        after = {r.id: (r.value, r.fetched_at, r.available_at) for r in db.execute(select(FinancialPeriod)).scalars()}
    assert after == before


def test_demo_chain_skips_refresh_without_db_writes(env):
    """CI: the financials chain is the demo provider, so nothing schedules,
    nothing drains and nothing is written."""
    assert not fr.history_capable()
    result = fr.nightly()
    assert result["skipped_reason"] == "no named history provider in the financials chain"
    assert result["refreshed"] == 0 and fr.refresh_if_due(T) is None
    with env.factory() as db:
        assert db.execute(select(func.count()).select_from(FundamentalRefreshState)).scalar_one() == 0


def test_refresh_if_due_runs_only_when_pending_and_due(env, monkeypatch):
    seed(env, monkeypatch)
    calls = chain(monkeypatch, ("fmp", statements(through=date(2026, 6, 30))))
    assert fr.refresh_if_due(T) is None  # no state
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 31))])])
    assert fr.refresh_if_due(T) is None and calls == []  # pending, inside the publication lag
    env.clock.now = START + fr.PUBLICATION_LAG
    pulled = fr.refresh_if_due(T)
    assert pulled["satisfied"] and pulled["trigger"] == "filing" and len(calls) == 1
    assert fr.refresh_if_due(T) is None and len(calls) == 1  # idle now


def test_admin_run_on_lagging_ticker_stays_pending(env, monkeypatch):
    """Critique: status is derived from stored state after every run, so an
    admin sync that still lacks the filed period cannot clear the retry."""
    seed(env, monkeypatch)
    chain(monkeypatch, ("fmp", statements()))
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 31))])])
    nights(env, date(2026, 8, 2), 2)
    before = state(env)
    assert (before["status"], before["attempts"]) == ("pending", 2)
    env.clock.now = datetime(2026, 8, 3, 12)
    report = backfill.sync_ticker(T, force_refresh=True, scope="fundamentals")
    assert report["fundamentals"]["committed"] and report["refresh_state"]["status"] == "pending"
    after = state(env)
    for key in ("status", "trigger", "attempts", "first_due_at", "due_at"):
        assert after[key] == before[key], key
    assert after["last_result"]["trigger"] == "admin" and after["lease_until"] is None
    # With the lease held elsewhere the admin stage reports running and does nothing.
    assert fr.claim(T)
    held = backfill.sync_ticker(T, force_refresh=True, scope="fundamentals")
    assert held["status"] == "running" and not held["success"] and "fundamentals" not in held


def test_observe_during_drain_keeps_filing(env, monkeypatch):
    """CAS: a newer 10-Q observed while the drain records its result must
    survive. The drain's write conflicts, re-reads and merges."""
    seed(env, monkeypatch)
    chain(monkeypatch, ("fmp", statements(through=date(2026, 6, 30))))
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 31))])])
    env.clock.now = datetime(2026, 11, 2, 3, 15)
    original = fr._stored_newest
    fired = []

    def racing(db, tickers):
        if not fired and tickers == [T]:
            fired.append(True)
            fr.observe_many([(T, [ten_q(date(2026, 9, 30), date(2026, 10, 30), acc="0000000001-26-000900")])],
                            now=env.clock.now + timedelta(minutes=1))
        return original(db, tickers)

    monkeypatch.setattr(fr, "_stored_newest", racing)
    result = fr.drain()
    assert fired and result["refreshed"] == 1
    final = state(env)
    assert final["filed_quarter_end"] == date(2026, 9, 30)
    assert (final["status"], final["trigger"], final["attempts"]) == ("pending", "filing", 0)
    assert final["first_due_at"] == fr.drain_slot(env.clock.now + timedelta(minutes=1) + fr.PUBLICATION_LAG)
    assert final["row_version"] == 3  # observe, racing observe, merged result


def test_legacy_cache_backed_pass_never_advances_fetched_at(env, monkeypatch):
    """The prior planner's guard: an aged `financials` cache row with no
    provider behind it, run through the legacy daily path, leaves every
    durable fetch time (and so every freshness judgement) unchanged."""
    from app.services import history_service
    from app.services.data_service import get_data_service

    seed(env, monkeypatch)
    monkeypatch.setattr(history_service, "SessionLocal", env.factory)
    with env.factory() as db:
        before = {r.id: r.fetched_at for r in db.execute(select(FinancialPeriod)).scalars()}
    ds = get_data_service()
    monkeypatch.setattr(ds, "get_financial_statements", lambda t: pytest.fail("provider-owned ticker read"))
    monkeypatch.setattr(ds, "get_filings", lambda *a, **k: [])
    monkeypatch.setattr(ds, "get_earnings_transcripts", lambda *a, **k: [])
    env.clock.now += timedelta(days=30)
    result = history_service.backfill_ticker(T)
    assert result["financial_periods"] == 0 and result["fundamentals"] == "durable"
    with env.factory() as db:
        assert {r.id: r.fetched_at for r in db.execute(select(FinancialPeriod)).scalars()} == before


def test_explorer_names_a_filed_fiscal_year_that_is_not_stored(env, monkeypatch):
    from app.services import fundamentals_series_service as fss

    seed(env, monkeypatch)
    with env.factory() as db:
        db.add(FundamentalRefreshState(ticker=T, filed_annual_end=date(2026, 12, 31), filed_annual_form="10-K",
                                       filed_annual_on=date(2027, 2, 20), annual_observed_at=datetime(2027, 2, 20, 12)))
        db.commit()
    monkeypatch.setattr(fss, "_utcnow", lambda: datetime(2027, 2, 22, 12))  # inside the grace
    with env.factory() as db:
        early = fss.build_series([T], ["revenue"], db=db).series[0].provenance
    assert "was filed" not in (early.stale_reason or "")
    monkeypatch.setattr(fss, "_utcnow", lambda: datetime(2027, 2, 24, 12))
    with env.factory() as db:
        late = fss.build_series([T], ["revenue"], db=db).series[0].provenance
    assert late.stale and "FY ending 2026-12-31 was filed (10-K, 2027-02-20) and is not stored yet" in late.stale_reason
