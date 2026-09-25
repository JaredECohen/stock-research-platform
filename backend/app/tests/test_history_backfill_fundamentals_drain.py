"""FIX-005: the nightly history loop drains filing-driven fundamentals refreshes.

`history_backfill.run_once` now ends with `fundamental_refresh.nightly()`
under its own cap. What must hold: every name the drain skipped, deferred or
failed on is in the note; a single-ticker admin run does not drain; a stuck
ticker fails the loop exactly once and is named afterwards; and the
cold-read notes (which own the word "deferred") are unchanged.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.monitoring import history_backfill
from app.services import fundamental_refresh as fr
from app.tests.test_fundamental_refresh import T, chain, env, seed, state, statements, ten_q  # noqa: F401


@pytest.fixture()
def loop(monkeypatch):
    """The loop around the drain, with no tier tickers and captured runs."""
    runs: list[dict] = []
    monkeypatch.setattr(history_backfill, "_tier1_tickers", lambda: [])
    monkeypatch.setattr(history_backfill, "record_run",
                        lambda *a, **k: runs.append({"success": k.get("success"), "note": k.get("note", "")}))
    return runs


def canned(**overrides) -> dict:
    result = fr._empty_result()
    result.update(overrides)
    return result


def test_note_names_every_drain_outcome_and_keeps_cold_read_language(monkeypatch, loop):
    monkeypatch.setattr(fr, "nightly", lambda: canned(
        refreshed=30, pending=4, over_cap=["AAA", "BBB"], over_budget=["CCC"], leased=["DDD"],
        missing=["EEE:quarterly:2026-06-30(10-Q filed 2026-07-30, attempt 2, next 2026-08-03T03:00)"],
        still_missing=["FFF(since 2026-08-10)"], rows_quarantined=3, repair_ids=["GGG:r-1"],
        entitlement_denied=["HHH:/income-statement:quarterly:402"]))
    totals = history_backfill.run_once(day=0)
    note = loop[-1]["note"]
    assert "fund_refreshed=30 fund_pending=4" in note
    for name in ("AAA, BBB", "CCC", "DDD", "EEE:quarterly:2026-06-30(10-Q filed 2026-07-30, attempt 2",
                 "FFF(since 2026-08-10)", "fundamentals quarantined=3: GGG:r-1",
                 "fmp entitlement denied: HHH:/income-statement:quarterly:402"):
        assert name in note, note
    assert "deferred" not in note
    # Lagging, over-cap and still-missing names alone do not fail the loop.
    assert loop[-1]["success"] is True
    assert (totals["fund_refreshed"], totals["fund_pending"], totals["fund_over_cap"]) == (30, 4, 2)


@pytest.mark.parametrize("failure", [{"errors": ["ZZZ:RuntimeError"]}, {"stuck": ["ZZZ"]}])
def test_drain_errors_and_newly_stuck_fail_the_loop(monkeypatch, loop, failure):
    monkeypatch.setattr(fr, "nightly", lambda: canned(**failure))
    history_backfill.run_once(day=0)
    assert loop[-1]["success"] is False and "ZZZ" in loop[-1]["note"]


def test_single_ticker_admin_run_skips_the_drain(monkeypatch, loop):
    monkeypatch.setattr(fr, "nightly", lambda: pytest.fail("a single-ticker run drained fundamentals"))
    monkeypatch.setattr(history_backfill, "backfill_ticker",
                        lambda t, **k: {"financial_periods": 0, "filings": 0, "transcripts": 0})
    totals = history_backfill.run_once("ABC")
    assert "fund_refreshed" not in totals and "fund_" not in loop[-1]["note"]


def test_demo_mode_names_the_skip_without_failing(loop):
    history_backfill.run_once(day=0)
    assert loop[-1]["success"] is True
    assert "fundamentals refresh skipped: no named history provider in the financials chain" in loop[-1]["note"]


def test_stuck_fails_once_then_named(env, monkeypatch, loop):  # noqa: F811
    """Critique: the loop fails on the night a ticker becomes stuck (its
    filed period still missing after the night-9 secondary attempt) and on
    no later night; afterwards the ticker is named as still missing."""
    seed(env, monkeypatch)
    chain(monkeypatch, ("fmp", statements()))
    fr.observe_many([(T, [ten_q(date(2026, 6, 30), date(2026, 7, 31))])])
    for n in range(20):
        env.clock.night(date(2026, 8, 2) + timedelta(days=n))
        history_backfill.run_once(day=n)
    verdicts = [r["success"] for r in loop]
    assert verdicts == [True] * 8 + [False] + [True] * 11
    assert f"fundamentals stuck (filed period not published after retries): {T}" in loop[8]["note"]
    assert all(f"fundamentals still missing: {T}(since 2026-08-10)" in r["note"] for r in loop[9:])
    assert state(env)["stuck_reported_at"] is not None
    # A lagging (not yet stuck) night names the ticker without failing.
    assert "fundamentals missing filed periods: TEST:quarterly:2026-06-30" in loop[0]["note"]
