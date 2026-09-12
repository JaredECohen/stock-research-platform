"""The transcript poller defers its overflow instead of losing it.

Same bound, same failure mode, same reasoning as
`test_filing_poller_detection_and_burst_cap.py` — see that file for the full
story. The short version: `transcripts` was one of the capabilities
`provider_cache` cached forever, so the first pass after that fix surfaces
every period that published in the meantime across the whole curated universe,
and `on_transcript_event` runs `_persist_raw_data_only` (filings, transcripts,
embeddings) before its gate. Unbounded, that is one burst on a worker that has
been OOM-killed twice.

The cap is only safe because it defers. `run_once` used to end every iteration
with `_save_seen_periods(t, periods | seen)` regardless of what happened
above, so a cap bolted on without moving that line would mark a skipped
ticker's periods as seen and the event would never fire for anyone.

The cap's *value* is a tuning decision (30, against a daily cadence). Its
behaviour is a contract, so the tests below shrink the cap and assert the
behaviour, rather than building thirty-odd fixtures to pin a number that is
meant to be adjustable.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.cache import cache_get
from app.monitoring import transcripts_poller


@pytest.fixture()
def fake_transcripts(monkeypatch):
    periods: dict[str, list[dict]] = {}
    monkeypatch.setattr(
        transcripts_poller, "get_transcripts", lambda t: list(periods.get(t, [])),
    )
    return periods


@pytest.fixture()
def notes(monkeypatch) -> list[str]:
    captured: list[str] = []
    monkeypatch.setattr(
        transcripts_poller, "record_run",
        lambda *a, **k: captured.append(k.get("note", "")),
    )
    return captured


@pytest.fixture()
def small_cap(monkeypatch):
    monkeypatch.setattr(transcripts_poller, "MAX_EVENT_TICKERS_PER_PASS", 2)
    return 2


def _tickers(n: int) -> list[str]:
    stamp = str(time.perf_counter_ns())[-6:]
    return [f"ZZTR{stamp}{i:02d}" for i in range(n)]


def test_the_cap_defers_and_a_later_pass_picks_the_deferred_ticker_up(
    fake_transcripts, notes, small_cap,
):
    tickers = _tickers(small_cap + 2)
    for t in tickers:
        fake_transcripts[t] = [{"period": "2025Q1"}]

    with patch("app.services.update_orchestrator.on_transcript_event") as handler:
        handler.return_value = {"kind": "skipped"}
        assert transcripts_poller.run_once(tickers) == []  # first-run init

        for t in tickers:
            fake_transcripts[t].append({"period": "2025Q2"})
        first = transcripts_poller.run_once(tickers)
        first_handled = [c.args[0] for c in handler.call_args_list]

    assert len(first) == small_cap
    deferred = tickers[small_cap:]
    assert not set(deferred) & set(first_handled)

    for t in deferred:
        snap = cache_get(t, "transcripts_seen_periods")
        assert snap is not None
        assert "2025Q2" not in snap.payload["periods"], (
            f"{t} was deferred but its new period was recorded as seen — the "
            "transcript event can now never fire"
        )

    with patch("app.services.update_orchestrator.on_transcript_event") as handler:
        handler.return_value = {"kind": "skipped"}
        second = transcripts_poller.run_once(tickers)
        second_handled = [c.args[0] for c in handler.call_args_list]

    assert sorted(e["ticker"] for e in second) == sorted(deferred)
    assert sorted(second_handled) == sorted(deferred)
    assert {e["period"] for e in second} == {"2025Q2"}


def test_the_note_reports_the_deferred_count_and_names(
    fake_transcripts, notes, small_cap,
):
    tickers = _tickers(small_cap + 1)
    for t in tickers:
        fake_transcripts[t] = [{"period": "2025Q1"}]
    with patch("app.services.update_orchestrator.on_transcript_event") as handler:
        handler.return_value = {"kind": "skipped"}
        transcripts_poller.run_once(tickers)
        notes.clear()
        for t in tickers:
            fake_transcripts[t].append({"period": "2025Q2"})
        transcripts_poller.run_once(tickers)

    (note,) = notes
    assert "deferred 1" in note
    assert tickers[-1] in note, note


def test_a_ticker_with_several_new_periods_counts_once_against_the_cap(
    fake_transcripts, notes, small_cap,
):
    """The cap counts tickers, because the cost is per-ticker.

    `on_transcript_event` re-reads and re-persists the whole ticker for each
    period, but the events for one ticker arrive together; splitting a
    ticker's periods across passes would re-do that work every time.
    """
    tickers = _tickers(small_cap)
    for t in tickers:
        fake_transcripts[t] = [{"period": "2025Q1"}]
    with patch("app.services.update_orchestrator.on_transcript_event") as handler:
        handler.return_value = {"kind": "skipped"}
        transcripts_poller.run_once(tickers)
        notes.clear()
        for t in tickers:
            fake_transcripts[t].extend([{"period": "2025Q2"}, {"period": "2025Q3"}])
        events = transcripts_poller.run_once(tickers)

    assert len(events) == 2 * small_cap, "both new periods per ticker fire"
    (note,) = notes
    assert "deferred" not in note


def test_the_seen_period_set_is_bounded():
    seen = {f"20{y:02d}Q{q}" for y in range(0, 26) for q in range(1, 5)}
    current = {"2025Q1", "2025Q2", "2025Q3", "2025Q4"}

    kept = transcripts_poller._bounded_seen(current, seen)

    assert len(kept) == transcripts_poller.MAX_SEEN_PERIODS
    assert current <= kept, "a period still being returned must never be pruned"
    assert kept == set(sorted(seen | current, reverse=True)[
        : transcripts_poller.MAX_SEEN_PERIODS
    ])
