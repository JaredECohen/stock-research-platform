"""The nightly sweep must not become the thing the poller cap was written
to prevent.

`history_backfill.run_once` fans `backfill_ticker` out over every
`auto_analysis` ticker — ~166 of them, in one job, with no cap. That was
free for the life of the deployment because `filings` and `transcripts`
never expired: after the first night the loop made zero provider calls, and
the note read a constant `filings=1660` to prove it.

Giving those two capabilities real TTLs is what arms it. `get_filings`
fetches the document body of every form it returns, up to ten per ticker at
a few MB each, and every filing it newly inserts runs
`filing_memory.post_pass` — embeddings plus an LLM diff. `transcripts` is
four AlphaVantage requests, and its 12-hour TTL is *shorter* than this
loop's 24-hour cadence, so the loop is expired for transcripts on every
single run, forever. The seven-day `filings` TTL is worse than it sounds:
the pollers refresh the whole universe within hours of each other, so all
166 rows expire inside one window and the herd arrives at a single 03:15
job.

That is the same arithmetic used to justify `edgar_poller
.MAX_FILING_EVENTS_PER_PASS`. Capping the poller at 15 events per pass does
not bound the first-pass burst if the nightly loop can do all 166 at 03:15
— it just changes which door the burst comes through.

Two properties, both behavioural:

- A ticker whose rows are cached is reconciled without consulting a
  provider, however old those rows are. Freshness is the pollers' job, and
  theirs is capped.
- A ticker that is genuinely cold still gets read, but no more than
  `MAX_COLD_TICKERS_PER_PASS` of them per night, the rest are deferred by
  name in the run note, and the rotation means a deferred ticker is not the
  same ticker every night.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from app.database import SessionLocal
from app.models import Company, EarningsTranscript, FilingDoc, FinancialPeriod
from app.monitoring import history_backfill
from app.services import history_service
from app.services import provider_cache as pc
from app.services.data_service import get_data_service

CAP = history_backfill.MAX_COLD_TICKERS_PER_PASS


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class FakeClock:
    """Controllable `provider_cache._now`, so row ages are exact."""

    def __init__(self) -> None:
        self.current = datetime.utcnow().replace(microsecond=0)

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current = self.current + timedelta(seconds=seconds)


@pytest.fixture()
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(pc, "_now", fake.now)
    return fake


@pytest.fixture(autouse=True)
def _quiet_indexing(monkeypatch):
    """Skip the post-insert vector indexing.

    Inserting a filing fires `filing_memory.post_pass` (chunk embedding plus
    an LLM diff). That cost is precisely what this file is about bounding,
    which is also why it must not actually run here.
    """
    from app.services import filing_memory
    monkeypatch.setattr(filing_memory, "post_pass", lambda row: None)
    monkeypatch.setattr(filing_memory, "index_transcript", lambda row: None)


@pytest.fixture()
def live_cache_ticker(monkeypatch):
    """A ticker whose reads go through `provider_cache` to a call counter.

    `DataService._cached` short-circuits straight to the fetcher whenever a
    test provider is registered, and conftest registers `DemoProvider` for
    the whole session — so a test about caching has to take that off first.
    `_try_chain` is the seam below the cache and above the network, so
    counting there counts exactly the calls this budget exists to limit.
    """
    ticker = f"ZZBF{int(time.perf_counter_ns() % 100000)}"
    ds = get_data_service()
    previous = ds._test_provider
    ds.register_test_provider(None)

    with SessionLocal() as db:
        db.merge(Company(
            ticker=ticker, company_name="Backfill Budget Co",
            sector="Test", industry="Test", universe_tier="auto_analysis",
            cik="0000009876",
        ))
        db.commit()

    calls: dict[str, int] = {}

    def fake_chain(capability, fn_name, *args, **kwargs):
        calls[capability] = calls.get(capability, 0) + 1
        if capability == "filings":
            return [{
                "accession_number": "0000009876-26-000001",
                "type": "10-K", "filing_date": "2026-02-01",
                "raw_text": "body text", "url": "https://example.invalid/f",
            }]
        if capability == "transcripts":
            return [{"period": "2026Q1", "date": "2026-02-10",
                     "prepared_remarks": "hello"}]
        return {}

    monkeypatch.setattr(ds, "_try_chain", fake_chain)
    monkeypatch.setattr(history_backfill, "_tier1_tickers", lambda: [ticker])

    try:
        yield ticker, calls
    finally:
        ds.register_test_provider(previous)
        for capability in ("filings", "transcripts", "financials"):
            pc.invalidate(capability, ticker)
        with SessionLocal() as db:
            db.query(FilingDoc).filter(FilingDoc.ticker == ticker).delete()
            db.query(EarningsTranscript).filter(EarningsTranscript.ticker == ticker).delete()
            db.query(FinancialPeriod).filter(FinancialPeriod.ticker == ticker).delete()
            db.query(Company).filter(Company.ticker == ticker).delete()
            db.commit()


# ---------------------------------------------------------------------------
# The sweep reconciles; it does not refetch
# ---------------------------------------------------------------------------

def test_an_expired_filings_row_does_not_make_the_nightly_sweep_refetch_bodies(
    clock, live_cache_ticker,
):
    """The finding, reproduced at the loop that would have paid for it.

    Under the naive version this assertion is 1, not 0 — and 1 here is 166
    in production, each up to ten document bodies with an LLM post-pass per
    filing, in a single nightly job.
    """
    ticker, calls = live_cache_ticker
    ds = get_data_service()

    ds.get_filings(ticker)
    assert calls["filings"] == 1, "priming read should reach the provider"

    # Well past the seven-day TTL — the state the whole curated universe is
    # in on the same night, because the pollers warm it in one window.
    clock.advance(pc.TTL_BY_CAPABILITY["filings"] + 86400)
    calls.clear()

    history_backfill.run_once()

    assert calls.get("filings", 0) == 0, (
        "the nightly sweep re-read filing bodies because a TTL rolled over; "
        "body freshness belongs to edgar_poller's targeted invalidation, "
        "which is capped per pass"
    )


def test_an_expired_transcripts_row_does_not_make_the_nightly_sweep_refetch(
    clock, live_cache_ticker,
):
    """Transcripts are the worse half: 12-hour TTL, 24-hour cadence.

    There is no age at which this loop finds a transcript row fresh, so
    without the fix it is ~664 AlphaVantage calls every night, for ever,
    three hours before `transcripts_poller` makes the same calls again.
    """
    ticker, calls = live_cache_ticker
    ds = get_data_service()

    ds.get_earnings_transcripts(ticker)
    assert calls["transcripts"] == 1

    clock.advance(pc.TTL_BY_CAPABILITY["transcripts"] + 3600)
    calls.clear()

    history_backfill.run_once()

    assert calls.get("transcripts", 0) == 0


def test_a_cold_ticker_is_still_read(clock, live_cache_ticker):
    """The sweep is bounded, not switched off.

    A ticker with no cached row at all — a fresh deployment, or a name just
    promoted into the tier — has nothing to reconcile, so it does reach the
    provider. That is the cost the per-pass budget bounds.
    """
    ticker, calls = live_cache_ticker
    assert calls == {}

    history_backfill.run_once()

    assert calls.get("filings", 0) == 1
    assert calls.get("transcripts", 0) == 1


def test_an_explicit_single_ticker_reads_fresh(clock, live_cache_ticker):
    """`run_once("NVDA")` and `/api/admin/backfill?ticker=` are deliberate
    human acts on one name. They are neither deferred nor served a stale
    body — only the universe-wide sweep trades freshness for cost."""
    ticker, calls = live_cache_ticker
    ds = get_data_service()
    ds.get_filings(ticker)
    clock.advance(pc.TTL_BY_CAPABILITY["filings"] + 86400)
    calls.clear()

    history_backfill.run_once(ticker)

    assert calls.get("filings", 0) == 1


# ---------------------------------------------------------------------------
# The cold-read budget
# ---------------------------------------------------------------------------

@pytest.fixture()
def counted_backfill(monkeypatch):
    """Record which tickers a pass actually backfilled."""
    processed: list[str] = []

    def fake_backfill(ticker, *, db=None, prefer_cached=False):
        processed.append(ticker)
        return {"financial_periods": 0, "filings": 0, "transcripts": 0}

    monkeypatch.setattr(history_backfill, "backfill_ticker", fake_backfill)
    return processed


@pytest.fixture()
def captured_note(monkeypatch):
    notes: list[str] = []
    monkeypatch.setattr(
        history_backfill, "record_run",
        lambda *a, **k: notes.append(k.get("note", "")),
    )
    return notes


def _all_cold(monkeypatch, cold: set[str] | None = None) -> None:
    monkeypatch.setattr(
        history_backfill, "backfill_hits_provider",
        (lambda t: True) if cold is None else (lambda t: t in cold),
    )


def test_a_pass_reads_at_most_the_cap_cold_and_defers_the_rest_by_name(
    monkeypatch, counted_backfill, captured_note,
):
    """The production safety property. 166 cold tickers is ~1,660 document
    bodies and as many LLM post-passes in one job, on a 512 MiB worker."""
    universe = [f"T{i:03d}" for i in range(CAP * 3)]
    monkeypatch.setattr(history_backfill, "_tier1_tickers", lambda: list(universe))
    _all_cold(monkeypatch)

    res = history_backfill.run_once(day=0)

    assert len(counted_backfill) == CAP
    assert res["cold_reads"] == CAP
    assert res["deferred"] == len(universe) - CAP
    assert res["tickers_processed"] == CAP

    note = captured_note[-1]
    assert f"deferred {len(universe) - CAP}" in note, note
    # Standing rule in this repo: no silent caps. A bare count leaves nobody
    # able to tell which names are waiting.
    deferred_names = [t for t in universe if t not in counted_backfill]
    assert deferred_names[0] in note, note


def test_a_warm_ticker_is_never_deferred(monkeypatch, counted_backfill, captured_note):
    """The budget is on provider calls, not on tickers.

    Reconciling a cached row against the history tables is a local database
    comparison. Capping that would slow the sweep down for no saving and
    leave warm tickers un-reconciled behind a queue of cold ones.
    """
    cold = {f"C{i:03d}" for i in range(CAP * 2)}
    warm = [f"W{i:03d}" for i in range(5)]
    universe = sorted(cold) + warm
    monkeypatch.setattr(history_backfill, "_tier1_tickers", lambda: list(universe))
    _all_cold(monkeypatch, cold)

    res = history_backfill.run_once(day=0)

    assert set(warm) <= set(counted_backfill), "a free read was deferred"
    assert res["cold_reads"] == CAP
    assert res["deferred"] == len(cold) - CAP


def test_the_pass_rotates_so_the_tail_is_not_starved(monkeypatch, counted_backfill):
    """A cap that always spends its budget on the same head of the list
    drains nothing, and the note would show steady progress while a fixed
    set of tickers stayed permanently cold.

    Fails against an unrotated implementation, which returns the identical
    set of tickers every night.
    """
    universe = [f"T{i:03d}" for i in range(CAP * 3)]
    monkeypatch.setattr(history_backfill, "_tier1_tickers", lambda: list(universe))
    _all_cold(monkeypatch)

    nights = []
    for day in range(3):
        counted_backfill.clear()
        history_backfill.run_once(day=day)
        nights.append(set(counted_backfill))

    assert nights[0] != nights[1], "consecutive nights processed the same tickers"
    assert set().union(*nights) == set(universe), (
        "three nights at a cap of one third of the universe must cover it"
    )


def test_backfill_hits_provider_is_false_once_the_rows_exist(clock, live_cache_ticker):
    """The predicate the budget is spent against, at the cache.

    Asserted behaviourally rather than by reading the table, because the
    question is "will this cost a provider call", and only the presence of
    a row answers it.
    """
    ticker, _calls = live_cache_ticker
    ds = get_data_service()

    assert history_service.backfill_hits_provider(ticker) is True

    ds.get_filings(ticker)
    ds.get_earnings_transcripts(ticker)

    assert history_service.backfill_hits_provider(ticker) is False
    # Age is irrelevant: a `prefer_cached` read serves the row whatever its
    # age, so it still costs nothing.
    clock.advance(365 * 86400)
    assert history_service.backfill_hits_provider(ticker) is False
