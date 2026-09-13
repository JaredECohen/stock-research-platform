"""The filing poller: cheap detection, and a catch-up burst that is bounded.

Two properties, both learned the hard way.

**Detection.** `edgar_poller` compares the accession numbers EDGAR returns
against a per-ticker `edgar_seen_accessions` set. That comparison was against a
frozen list for the life of the deployment, because `filings` had no TTL entry
and `provider_cache` read a missing entry as "never expires". The fix is not
just a TTL: a 30-minute poll through `data_service.get_filings` would download
the body of every form for every ticker — up to ten multi-megabyte documents
each, ~1,700 per pass — on a worker Render has already OOM-killed twice. So
detection reads a separate `filings_index` capability with `fetch_text=False`,
and the expensive bodies are refetched only for the one ticker that changed.

**The burst.** `on_filing_event` calls `_persist_raw_data_only` *before* its
auto-regen gate, and that path runs `filing_memory.post_pass`, which uses the
LLM. On the first healthy pass ~166 tickers each surface a month of accumulated
filings at once, so the number of events per pass has to be capped.

The cap is where the subtle bug lives, and it is the thing most of this file
is about. `run_once` ended every iteration with `if accessions:
_save_seen_accessions(t, accessions | seen)` — unconditionally, for every
ticker it looked at. Capping the events without moving that line would record a
deferred ticker's new accessions as *seen* while never processing them: the
next pass computes an empty diff, and the filing is lost permanently. A cap
that silently drops events is worse than no cap, so the test that matters most
here is the one that runs a second pass and asserts the deferred ticker's event
actually fires.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.cache import cache_get
from app.database import SessionLocal
from app.models import Company
from app.monitoring import edgar_poller
from app.services import provider_cache as pc
from app.services.data_service import get_data_service


def _suffix() -> str:
    return str(time.perf_counter_ns())[-6:]


def _accession(n: int) -> str:
    return f"0000001234-26-{n:06d}"


# ---------------------------------------------------------------------------
# Detection, through data_service and the provider
# ---------------------------------------------------------------------------

class RecordingProvider:
    """Stands in for `SECEdgarProvider`, recording how it was asked."""

    name = "recording"

    def __init__(self) -> None:
        self.filings: list[dict] = []
        self.calls: list[dict] = []

    def get_filings(self, ticker, *, cik=None, fetch_text=True):
        self.calls.append({"ticker": ticker, "cik": cik, "fetch_text": fetch_text})
        return list(self.filings)


@pytest.fixture()
def edgar_ticker():
    """A company row with a CIK, so `_lookup_cik` resolves without a network."""
    ticker = f"ZZEDG{_suffix()}"[:12]
    with SessionLocal() as db:
        db.merge(Company(
            ticker=ticker, company_name="Edgar Test Co",
            sector="Test", industry="Test", universe_tier="data_only",
            cik="0000001234",
        ))
        db.commit()
    yield ticker
    with SessionLocal() as db:
        db.query(Company).filter(Company.ticker == ticker).delete()
        db.commit()
    pc.invalidate("filings", ticker.upper())
    pc.invalidate("filings_index", ticker.upper())


@pytest.fixture()
def provider():
    """Swap the session's DemoProvider for one that records its arguments."""
    ds = get_data_service()
    previous = ds._test_provider
    recording = RecordingProvider()
    ds.register_test_provider(recording)
    try:
        yield recording
    finally:
        ds.register_test_provider(previous)


def test_the_poll_asks_the_provider_to_skip_document_bodies(provider, edgar_ticker):
    """The whole point of the index capability, asserted at the provider.

    `fetch_text=True` here is a ~1,700-document pass across the curated
    universe. There is no cheaper way to observe the difference than to look
    at the argument, because the demo provider has no bodies to fetch.
    """
    provider.filings = [{"type": "10-K", "accession_number": _accession(1)}]
    edgar_poller.run_once([edgar_ticker])

    assert provider.calls, "the poller never reached the provider"
    assert all(c["fetch_text"] is False for c in provider.calls), (
        f"the poller pulled document bodies: {provider.calls}"
    )


def test_a_genuinely_new_accession_reaches_on_filing_event(provider, edgar_ticker):
    """End to end: provider → data_service → poller → orchestrator.

    This is the path that had never once completed in production.
    """
    provider.filings = [{"type": "10-K", "accession_number": _accession(1)}]

    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped", "ticker": edgar_ticker}
        # First pass initialises the bookkeeping; nothing is "new" yet.
        assert edgar_poller.run_once([edgar_ticker]) == []
        handler.assert_not_called()

        # A new 8-K appears — NVDA filed four times while the frozen cache
        # meant none of them were noticed.
        provider.filings.append({"type": "8-K", "accession_number": _accession(2)})
        events = edgar_poller.run_once([edgar_ticker])

    assert [e["ticker"] for e in events] == [edgar_ticker]
    assert events[0]["new_accessions"] == [_accession(2)]
    handler.assert_called_once_with(edgar_ticker)


def test_the_new_accession_drops_the_stale_filing_bodies(provider, edgar_ticker):
    """Better detection is worthless if the text stays frozen.

    Downstream readers (`_persist_raw_data_only`, the filings analyst) go
    through `get_filings`, which is cached for a week. Without invalidation
    the poller would notice the new 10-Q and every reader would keep being
    served the body cached before it existed.
    """
    provider.filings = [{"type": "10-K", "accession_number": _accession(1)}]
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped", "ticker": edgar_ticker}
        edgar_poller.run_once([edgar_ticker])

        # A cached body, as a production worker would have.
        pc.put("filings", edgar_ticker.upper(), [{"accession_number": _accession(1),
                                                  "raw_text": "old body"}])
        assert pc.get("filings", edgar_ticker.upper()) is not None

        provider.filings.append({"type": "8-K", "accession_number": _accession(2)})
        edgar_poller.run_once([edgar_ticker])

    assert pc.get("filings", edgar_ticker.upper()) is None, (
        "the filing bodies for a ticker with a new accession must be refetched, "
        "not served from the row written before the filing existed"
    )


def test_no_new_accession_leaves_the_cached_bodies_alone(provider, edgar_ticker):
    """The invalidation is targeted: a quiet pass must not blow the cache away
    and re-download every document on the next read."""
    provider.filings = [{"type": "10-K", "accession_number": _accession(1)}]
    edgar_poller.run_once([edgar_ticker])
    pc.put("filings", edgar_ticker.upper(), [{"accession_number": _accession(1)}])

    edgar_poller.run_once([edgar_ticker])

    assert pc.get("filings", edgar_ticker.upper()) is not None


# ---------------------------------------------------------------------------
# The burst cap
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_index(monkeypatch):
    """Drive the poller from a dict of ticker → filings, with no DB provider.

    Returned object is mutable, so a test can add an accession between passes
    the way EDGAR would.
    """
    filings: dict[str, list[dict]] = {}
    monkeypatch.setattr(
        edgar_poller, "get_filings_index", lambda t: list(filings.get(t, [])),
    )
    # Both invalidations are exercised by the tests above; here they are noise.
    monkeypatch.setattr(edgar_poller, "invalidate", lambda *a, **k: None)
    monkeypatch.setattr(edgar_poller, "invalidate_filings_text", lambda t: 0)
    return filings


@pytest.fixture()
def runs(monkeypatch) -> list[dict]:
    """Every `record_run` the poller makes — the verdict as well as the note.

    Capturing only the note is how a progress record's `success` flag went
    unasserted while it was hardcoded to True. `record_run` keeps one row
    per loop and overwrites both fields, so the flag is half of what
    cron-health shows and belongs in the capture.
    """
    captured: list[dict] = []
    monkeypatch.setattr(
        edgar_poller, "record_run",
        lambda *a, **k: captured.append(
            {"success": k.get("success", True), "note": k.get("note", "")}
        ),
    )
    return captured


def _notes(runs: list[dict]) -> list[str]:
    return [r["note"] for r in runs]


def _prime(fake_index: dict, tickers: list[str]) -> None:
    """Give every ticker one filing and run a pass, so `seen` is non-empty.

    The poller deliberately treats an empty `seen` as first-run initialisation
    and fires nothing, so every cap test has to get past that first.
    """
    for t in tickers:
        fake_index[t] = [{"type": "10-K", "accession_number": _accession(1)}]
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        assert edgar_poller.run_once(tickers) == []


def test_the_cap_defers_the_overflow_and_a_later_pass_picks_it_up(fake_index, runs):
    """The one that matters. A deferred ticker's event must still fire.

    Deferral is only acceptable because it is a delay. If `seen` were updated
    for a ticker whose event was skipped — which is what the original
    unconditional `_save_seen_accessions` line would have done — the next pass
    would see no diff and the filing would be lost with no trace anywhere.
    """
    overflow = 3
    count = edgar_poller.MAX_FILING_EVENTS_PER_PASS + overflow
    tickers = [f"ZZBC{_suffix()}{i:02d}" for i in range(count)]
    _prime(fake_index, tickers)

    # Every ticker now has a new 8-K, all at once — the catch-up shape.
    for t in tickers:
        fake_index[t].append({"type": "8-K", "accession_number": _accession(2)})

    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        first = edgar_poller.run_once(tickers)
        first_handled = [c.args[0] for c in handler.call_args_list]

    assert len(first) == edgar_poller.MAX_FILING_EVENTS_PER_PASS, (
        "the pass handed more events to the orchestrator than the cap allows"
    )
    assert len(first_handled) == edgar_poller.MAX_FILING_EVENTS_PER_PASS
    deferred = tickers[edgar_poller.MAX_FILING_EVENTS_PER_PASS:]
    assert not set(deferred) & set(first_handled)

    # The deferred tickers must NOT have been recorded as seen.
    for t in deferred:
        snap = cache_get(t, "edgar_seen_accessions")
        assert snap is not None
        assert _accession(2) not in snap.payload["accessions"], (
            f"{t} was deferred but its new accession was recorded as seen — its "
            "event can now never fire, which is the failure this cap must not "
            "introduce"
        )

    # Second pass: nothing new for the ones already handled, and the deferred
    # ones fire.
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        second = edgar_poller.run_once(tickers)
        second_handled = [c.args[0] for c in handler.call_args_list]

    assert sorted(e["ticker"] for e in second) == sorted(deferred)
    assert sorted(second_handled) == sorted(deferred)
    assert all(e["new_accessions"] == [_accession(2)] for e in second)

    # And now everyone is caught up: a third pass is quiet.
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        assert edgar_poller.run_once(tickers) == []


def test_the_note_reports_the_deferred_count_and_names(fake_index, runs):
    """No silent caps: cron-health has to show what is waiting, by name."""
    overflow = 2
    count = edgar_poller.MAX_FILING_EVENTS_PER_PASS + overflow
    tickers = [f"ZZNT{_suffix()}{i:02d}" for i in range(count)]
    _prime(fake_index, tickers)
    runs.clear()
    for t in tickers:
        fake_index[t].append({"type": "8-K", "accession_number": _accession(2)})

    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        edgar_poller.run_once(tickers)

    (note,) = _notes(runs)
    assert f"deferred {overflow}" in note, note
    for t in tickers[edgar_poller.MAX_FILING_EVENTS_PER_PASS:]:
        assert t in note, f"{t} was deferred but not named in the note: {note}"


def test_a_long_deferral_list_is_elided_rather_than_dropped(fake_index, runs):
    """The first pass after the fix defers ~150 names. The note says so."""
    count = edgar_poller.MAX_FILING_EVENTS_PER_PASS + 9
    tickers = [f"ZZEL{_suffix()}{i:02d}" for i in range(count)]
    _prime(fake_index, tickers)
    runs.clear()
    for t in tickers:
        fake_index[t].append({"type": "8-K", "accession_number": _accession(2)})

    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        edgar_poller.run_once(tickers)

    (note,) = _notes(runs)
    assert "deferred 9" in note
    assert "+4 more" in note, f"truncation must be visible, not silent: {note}"


def test_an_uncapped_pass_still_reports_no_deferrals(fake_index, runs):
    """Steady state: a handful of 8-Ks, no cap language in the note."""
    tickers = [f"ZZOK{_suffix()}{i:02d}" for i in range(3)]
    _prime(fake_index, tickers)
    runs.clear()
    for t in tickers:
        fake_index[t].append({"type": "8-K", "accession_number": _accession(2)})

    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        events = edgar_poller.run_once(tickers)

    assert len(events) == 3
    (note,) = _notes(runs)
    assert "deferred" not in note
    assert note.startswith("3 new filings")


# ---------------------------------------------------------------------------
# Bookkeeping growth
# ---------------------------------------------------------------------------

def test_the_seen_set_is_bounded():
    seen = {_accession(n) for n in range(200)}
    current = {_accession(n) for n in range(195, 205)}

    kept = edgar_poller._bounded_seen(current, seen)

    assert len(kept) == edgar_poller.MAX_SEEN_ACCESSIONS, (
        "the union used to grow forever — one row per ticker, never pruned"
    )
    assert current <= kept, (
        "a filing still in the provider's window was pruned; the next pass "
        "would read it as new and re-fire the event"
    )
    # What is kept beyond the current window is the newest history, not an
    # arbitrary slice — the drop has to be deterministic to be reasoned about.
    expected_history = sorted(seen - current, reverse=True)[
        : edgar_poller.MAX_SEEN_ACCESSIONS - len(current)
    ]
    assert kept == current | set(expected_history)


def test_a_small_seen_set_is_left_alone():
    current = {_accession(1), _accession(2)}
    seen = {_accession(1)}
    assert edgar_poller._bounded_seen(current, seen) == current | seen


# ---------------------------------------------------------------------------
# The wall-clock budget
# ---------------------------------------------------------------------------
#
# Third bound, same discipline. Measured in production on 2026-09-12: the
# last COMPLETED pass was logged at 20:52:34Z and it was still running 95
# minutes later — it fired AAPL at 21:29Z and AMZN at 22:05Z, 36 minutes
# apart and in alphabetical order, so it was working the whole time. The loop
# is registered `interval, minutes=30` with APScheduler's default
# `max_instances=1`, so that pass silently ate three ticks, and `record_run`
# fires only at the END of `run_once`, so `/api/admin/cron-health` showed one
# stale timestamp for the entire hour and a half. An operator could not tell
# a slow pass from a wedged loop.
#
# A budget alone would be worse than none: stopping at the same place every
# pass polls the head of the alphabet forever and never reads the tail again.
# So the pass rotates — and the test that matters, exactly as for the event
# cap, is the one that runs consecutive passes and asserts every ticker's
# event actually fires.

class _Clock:
    """A wall clock that only moves when the poller polls something."""

    def __init__(self) -> None:
        self.now = 0.0
        self.cost = 0.0

    def monotonic(self) -> float:
        return self.now

    def spend(self) -> None:
        self.now += self.cost


@pytest.fixture()
def clock(monkeypatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(edgar_poller, "time", c)
    return c


@pytest.fixture()
def timed_index(fake_index, clock):
    """`fake_index`, with each provider read costing `clock.cost` seconds."""
    real = edgar_poller.get_filings_index

    def charged(ticker):
        clock.spend()
        return real(ticker)

    edgar_poller.get_filings_index = charged
    try:
        yield fake_index
    finally:
        edgar_poller.get_filings_index = real


@pytest.fixture()
def scheduled_universe(monkeypatch):
    """Drive `run_once()` with no arguments — the scheduled path.

    Rotation and the pass cursor are deliberately scoped to that path: an
    admin re-run naming one ticker must not move the scheduled pass's place
    in the universe.
    """
    universe: list[str] = []
    monkeypatch.setattr(
        edgar_poller, "curated_poll_universe", lambda: (list(universe), 0),
    )
    edgar_poller._save_resume_from(None)
    return universe


def test_the_budget_stops_the_pass_and_consecutive_passes_cover_everyone(
    timed_index, runs, clock, scheduled_universe, monkeypatch,
):
    """The one that matters, and the twin of the burst-cap test above.

    A ticker the budget never reached is not deferred by a decision — it is
    simply never looked at — so the failure mode is quieter: its
    bookkeeping is untouched (good), but if the next pass starts from the
    same end of the universe it is never looked at again either. Asserted
    end to end: run passes until the universe is covered, and require every
    ticker's filing event to have fired exactly once.
    """
    monkeypatch.setattr(edgar_poller, "MAX_PASS_SECONDS", 180)
    tickers = [f"ZZBG{_suffix()}{i:02d}" for i in range(10)]
    scheduled_universe.extend(tickers)

    clock.cost = 0.0          # priming is free
    _prime(timed_index, tickers)
    for t in tickers:
        timed_index[t].append({"type": "8-K", "accession_number": _accession(2)})

    clock.cost = 60.0         # three tickers per 180-second budget
    handled: list[str] = []
    fired: list[str] = []
    for _ in range(6):
        with patch("app.services.update_orchestrator.on_filing_event") as handler:
            handler.return_value = {"kind": "skipped"}
            events = edgar_poller.run_once()
            handled.extend(c.args[0] for c in handler.call_args_list)
        fired.extend(e["ticker"] for e in events)
        assert len(events) <= 3, "the pass ignored its wall-clock budget"

    assert sorted(fired) == sorted(tickers), (
        "a ticker the budget skipped never had its filing event fire; a pass "
        "that always restarts at the same end of the universe starves the "
        "tail exactly as thoroughly as losing the event would"
    )
    assert sorted(handled) == sorted(tickers)
    assert len(set(fired)) == len(fired), "a ticker's event fired twice"


def test_an_unvisited_ticker_keeps_its_bookkeeping(
    timed_index, runs, clock, scheduled_universe, monkeypatch,
):
    """Same rule as the event cap: never record what was not processed."""
    monkeypatch.setattr(edgar_poller, "MAX_PASS_SECONDS", 60)
    tickers = [f"ZZBK{_suffix()}{i:02d}" for i in range(6)]
    scheduled_universe.extend(tickers)

    clock.cost = 0.0
    _prime(timed_index, tickers)
    for t in tickers:
        timed_index[t].append({"type": "8-K", "accession_number": _accession(2)})

    clock.cost = 60.0
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        visited = [e["ticker"] for e in edgar_poller.run_once()]

    assert visited, "the budget stopped the pass before it did any work"
    for t in set(tickers) - set(visited):
        snap = cache_get(t, "edgar_seen_accessions")
        assert snap is not None
        assert _accession(2) not in snap.payload["accessions"], (
            f"{t} was never visited but its new accession was recorded as "
            "seen — its event can now never fire"
        )


def test_the_note_says_how_far_the_pass_got(
    timed_index, runs, clock, scheduled_universe, monkeypatch,
):
    """No silent caps — cron-health has to show what is still waiting."""
    monkeypatch.setattr(edgar_poller, "MAX_PASS_SECONDS", 180)
    tickers = [f"ZZNB{_suffix()}{i:02d}" for i in range(8)]
    scheduled_universe.extend(tickers)

    clock.cost = 0.0
    _prime(timed_index, tickers)
    runs.clear()
    clock.cost = 60.0
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        edgar_poller.run_once()

    note = _notes(runs)[-1]
    assert "polled 3/8" in note, note
    assert "5 unvisited" in note, note
    assert str(edgar_poller.MAX_PASS_SECONDS) in note, note
    for t in tickers[3:]:
        assert t in note or "+" in note


def test_a_pass_that_finishes_reports_no_budget_language(
    timed_index, runs, clock, scheduled_universe, monkeypatch,
):
    monkeypatch.setattr(edgar_poller, "MAX_PASS_SECONDS", 1800)
    tickers = [f"ZZFN{_suffix()}{i:02d}" for i in range(4)]
    scheduled_universe.extend(tickers)

    clock.cost = 0.0
    _prime(timed_index, tickers)
    runs.clear()
    clock.cost = 60.0
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        edgar_poller.run_once()

    note = _notes(runs)[-1]
    assert "unvisited" not in note, note
    assert "polled 4/4" in note, note
    # And the cursor is back at the head, so the next pass is a full sweep.
    assert edgar_poller._resume_from() is None


def test_a_long_pass_reports_progress_before_it_finishes(
    timed_index, runs, clock, scheduled_universe, monkeypatch,
):
    """The other half of the production symptom.

    `record_run` firing only at the end is what made a 95-minute pass
    indistinguishable from a loop that died 95 minutes ago: cron-health had
    nothing but the *previous* pass's timestamp to report for the whole
    window.
    """
    monkeypatch.setattr(edgar_poller, "MAX_PASS_SECONDS", 100_000)
    monkeypatch.setattr(edgar_poller, "PROGRESS_INTERVAL_SECONDS", 120)
    tickers = [f"ZZPR{_suffix()}{i:02d}" for i in range(8)]
    scheduled_universe.extend(tickers)

    clock.cost = 0.0
    _prime(timed_index, tickers)
    runs.clear()
    clock.cost = 60.0
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        edgar_poller.run_once()

    notes = _notes(runs)
    progress = [n for n in notes if n.startswith("in progress")]
    assert progress, (
        f"a pass spanning {8 * 60}s of wall clock reported nothing until it "
        f"finished: {notes}"
    )
    assert "polled" in progress[0] and "elapsed" in progress[0]
    assert not notes[-1].startswith("in progress"), (
        "the completion note must be the last thing recorded"
    )


def test_a_progress_record_never_claims_a_success_the_pass_has_not_earned(
    timed_index, runs, clock, scheduled_universe, monkeypatch,
):
    """The verdict half of the progress ping — the half cron-health alarms on.

    `record_run` keeps ONE row per loop and overwrites `success` along with
    the note, so a progress ping that hardcodes True is false telemetry of
    exactly the kind this branch exists to remove: it erases the previous
    pass's failure two minutes into a pass that has established nothing,
    and it leaves a *fresh* `success=true` behind for a pass the OOM killer
    stopped mid-run — the one case an operator most needs to see. A ping
    reports the verdict the pass has reached so far, and names what failed.
    """
    monkeypatch.setattr(edgar_poller, "MAX_PASS_SECONDS", 100_000)
    monkeypatch.setattr(edgar_poller, "PROGRESS_INTERVAL_SECONDS", 120)
    tickers = [f"ZZGV{_suffix()}{i:02d}" for i in range(8)]
    scheduled_universe.extend(tickers)

    clock.cost = 0.0
    _prime(timed_index, tickers)
    for t in tickers:
        timed_index[t].append({"type": "8-K", "accession_number": _accession(2)})
    runs.clear()
    clock.cost = 60.0
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        # What a DB hiccup on the worker produces: the auto-regen gate
        # crashed rather than deciding to skip.
        handler.return_value = {"kind": "gate_error"}
        edgar_poller.run_once()

    progress = [r for r in runs if r["note"].startswith("in progress")]
    assert progress, f"no progress record to check: {_notes(runs)}"
    assert all(r["success"] is False for r in progress), (
        "a pass that has already seen gate errors reported success mid-flight, "
        f"overwriting the standing verdict: {progress}"
    )
    assert all("gate errors" in r["note"] for r in progress), (
        f"the progress note must name what is failing, not only how far the "
        f"pass got: {progress}"
    )
    # The completion record still has the last word, and agrees.
    assert runs[-1]["success"] is False
    assert not runs[-1]["note"].startswith("in progress")


def test_a_clean_pass_still_reports_progress_as_a_success(
    timed_index, runs, clock, scheduled_universe, monkeypatch,
):
    """The other side of it: the verdict tracks the pass, it is not pinned False."""
    monkeypatch.setattr(edgar_poller, "MAX_PASS_SECONDS", 100_000)
    monkeypatch.setattr(edgar_poller, "PROGRESS_INTERVAL_SECONDS", 120)
    tickers = [f"ZZGC{_suffix()}{i:02d}" for i in range(8)]
    scheduled_universe.extend(tickers)

    clock.cost = 0.0
    _prime(timed_index, tickers)
    runs.clear()
    clock.cost = 60.0
    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        edgar_poller.run_once()

    progress = [r for r in runs if r["note"].startswith("in progress")]
    assert progress and all(r["success"] is True for r in progress)
    assert all("gate errors" not in r["note"] for r in progress)


def test_an_explicit_ticker_list_does_not_move_the_scheduled_cursor(
    fake_index, runs,
):
    """An admin re-run for one name must not reposition the nightly sweep."""
    edgar_poller._save_resume_from("ZZANCHOR")
    tickers = [f"ZZAD{_suffix()}{i:02d}" for i in range(3)]
    _prime(fake_index, tickers)

    with patch("app.services.update_orchestrator.on_filing_event") as handler:
        handler.return_value = {"kind": "skipped"}
        edgar_poller.run_once(tickers)

    assert edgar_poller._resume_from() == "ZZANCHOR"


def test_the_rotation_falls_back_to_the_head_for_a_departed_ticker():
    tickers = ["A", "B", "C"]
    assert edgar_poller._rotated(tickers, "B") == ["B", "C", "A"]
    assert edgar_poller._rotated(tickers, "GONE") == tickers
    assert edgar_poller._rotated(tickers, None) == tickers
