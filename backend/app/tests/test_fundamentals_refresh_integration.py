"""FIX-005 end to end: EDGAR poller observes -> nightly drain -> coverage.

The pieces are unit-tested in `test_fundamental_refresh.py`; this drives the
real `edgar_poller.run_once` and `history_backfill.run_once` against one
database with a fake named provider chain, the way the worker runs them.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from app.monitoring import edgar_poller, history_backfill
from app.tests.test_fundamental_refresh import (  # noqa: F401
    START,
    T,
    chain,
    env,
    seed,
    state,
    statements,
    ten_q,
)


def test_poller_to_drain_to_coverage_integration(env, monkeypatch):  # noqa: F811
    seed(env, monkeypatch)  # FMP history through 2026Q1
    index = [ten_q(date(2026, 6, 30), date(2026, 7, 31)),
             {"type": "8-K", "period_end": "2026-07-30", "filing_date": "2026-07-31",
              "accession_number": "0000000001-26-000101"}]
    seen: dict[str, set] = {}
    monkeypatch.setattr(edgar_poller, "get_filings_index", lambda t: list(index))
    monkeypatch.setattr(edgar_poller, "_seen_accessions", lambda t: set(seen.get(t, set())))
    monkeypatch.setattr(edgar_poller, "_save_seen_accessions", lambda t, accs: seen.__setitem__(t, set(accs)))
    runs: list[tuple[str, bool, str]] = []
    monkeypatch.setattr(edgar_poller, "record_run", lambda name, **k: runs.append((name, k["success"], k["note"])))
    monkeypatch.setattr(edgar_poller, "record_progress", lambda *a, **k: None)
    monkeypatch.setattr(history_backfill, "record_run", lambda name, **k: runs.append((name, k["success"], k["note"])))
    monkeypatch.setattr(history_backfill, "_tier1_tickers", lambda: [])
    published = {"payload": statements()}
    calls = chain(monkeypatch, ("fmp", lambda: published["payload"]))

    # 1. The 30-minute poll sees the 10-Q (first run: no filing event).
    assert edgar_poller.run_once([T]) == []
    assert runs[-1][:2] == ("edgar_poller", True) and "refresh-state" not in runs[-1][2]
    assert (state(env)["status"], state(env)["trigger"]) == ("pending", "filing")

    # 2. Night 1: FMP has not published yet. Named, not a failure.
    env.clock.night(date(2026, 8, 2))
    history_backfill.run_once(day=0)
    name, success, note = runs[-1]
    assert name == "history_backfill" and success and "fund_refreshed=1 fund_pending=1" in note
    assert "fundamentals missing filed periods: TEST:quarterly:2026-06-30" in note

    # 3. After the publication grace the missing filed period blocks coverage.
    from app.services import fundamental_history_service as fhs
    env.clock.now = START + timedelta(hours=73)
    blocked = fhs.fundamental_coverage(T, date(2024, 8, 1))
    assert not blocked["success"]
    assert any(i["kind"] == "expected_period_missing" for i in blocked["issues"])

    # 4. FMP publishes; the next due night stores it and coverage is current.
    published["payload"] = statements(through=date(2026, 6, 30))
    env.clock.night(date(2026, 8, 5))
    history_backfill.run_once(day=3)
    assert runs[-1][1] and "fund_refreshed=1 fund_pending=0" in runs[-1][2]
    coverage = fhs.fundamental_coverage(T, date(2024, 8, 1))
    quarterly = coverage["coverage"]["income"]["quarterly"]
    assert coverage["success"] and quarterly["newest"] == "2026-06-30" and quarterly["period_current"]
    assert state(env)["status"] == "idle" and len(calls) == 2  # the lagging night, the published night

    # 5. The next poll of the same index schedules nothing new.
    env.clock.now = datetime(2026, 8, 5, 12)
    edgar_poller.run_once([T])
    assert state(env)["status"] == "idle"
