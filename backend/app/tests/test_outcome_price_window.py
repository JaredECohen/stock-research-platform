"""The price window an outcome is scored against must span the memo.

Two production defects sit behind these tests, both reproduced before the
fix and both scoring the learning loop wrong:

1. The fetched window was sized as ``horizon_days + 30`` bars *ending
   today*, so it slid forward every night while the memo's generation date
   stayed put. A memo older than that many bars fell out of the window and
   scored ``price_window_incomplete`` forever — the nightly
   ``outcome_loop  ok=False … unavailable=58``.

2. Worse, a *partially* covering window scored silently wrong numbers. The
   baseline fell back to "first close on or after the memo date", and when
   the window began after the memo every row satisfied that, so the
   earliest row in the window — a price from weeks later — was used as the
   memo's baseline. A bullish call that lost 20% was persisted as +14.29%
   with ``thesis_held=True`` (the ``TSTWSHIFT`` fixture below: the old
   60-bar window opened on 2026-06-22 at 140.0, and 160.0 on the target
   date scores +14.29% instead of the true -20%).

Every price stub here honours ``days`` the way the real providers do — FMP
passes it as ``limit=``, Polygon and Tiingo slice ``[-days:]`` — so the
window arithmetic under test is the production one, not a convenience.
Dates are fixed rather than relative to "now" so a memo never lands on a
weekend by accident.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date as _date
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from app.database import SessionLocal
from app.models import MemoOutcome, MemoSnapshot
from app.services import outcome_service

# A Friday. Every date in this module is anchored to it, and the fake
# provider's series always ends here, exactly as a real fetch ends today.
ANCHOR = _date(2026, 9, 11)
BENCH = "SPY"


# ---------------------------------------------------------------------------
# A provider that behaves like the real ones
# ---------------------------------------------------------------------------

def _weekdays(start: _date, end: _date) -> list[_date]:
    out: list[_date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _tape(
    close_fn: Callable[[_date], float], *,
    start: _date, end: _date = ANCHOR,
) -> list[dict[str, Any]]:
    """A daily bar series, oldest first — the shape `get_price_series` returns."""
    return [
        {"date": d.isoformat(), "close": float(close_fn(d))}
        for d in _weekdays(start, end)
    ]


class _BarProvider:
    """`days` is a count of trading bars ending today, not calendar days.

    That is the contract every configured provider actually implements, and
    it is the reason a horizon-sized window cannot reach an old memo.
    """

    def __init__(self, tapes: dict[str, list[dict[str, Any]]]) -> None:
        self.tapes = {k.upper(): v for k, v in tapes.items()}
        self.requests: list[tuple[str, int]] = []

    def __call__(self, ticker: str, days: int = 252) -> list[dict[str, Any]]:
        self.requests.append((ticker.upper(), days))
        rows = self.tapes.get(ticker.upper())
        return list(rows[-days:]) if rows else []

    def days_for(self, ticker: str) -> set[int]:
        return {d for t, d in self.requests if t == ticker.upper()}


def _stub(provider: _BarProvider):
    from app.services import market_data_service
    return patch.object(
        market_data_service, "get_price_series", side_effect=provider,
    )


def _flat(value: float) -> Callable[[_date], float]:
    return lambda _d: value


def _seed(
    ticker: str, *, memo_date: _date, rating: str = "Bullish",
) -> MemoSnapshot:
    """A live memo generated on `memo_date` (no `as_of_date` → not a backtest)."""
    with SessionLocal() as db:
        outcome_service._ensure_table(db)
        from app.services.memo_store import _ensure_table as _ensure_memo
        _ensure_memo(db)
        db.query(MemoOutcome).filter(MemoOutcome.ticker == ticker).delete()
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker == ticker).delete()
        snap = MemoSnapshot(
            ticker=ticker, version=1, parent_version=None, trigger="first_run",
            memo_json={
                "ticker": ticker, "rating_label": rating,
                "confidence_score": 70.0, "sector": "Technology",
                "generation_mode": "live",  # W6: eligibility is fail-closed
            },
            revision_log=[],
            generated_at=datetime.combine(memo_date, datetime.min.time()).replace(hour=13),
            as_of_date=None,
        )
        db.add(snap)
        db.commit()
        db.refresh(snap)
        db.expunge(snap)
        return snap


def _evaluate(snap: MemoSnapshot, horizon: int) -> tuple[MemoOutcome | None, str]:
    with SessionLocal() as db:
        outcome_service._ensure_table(db)
        out, status = outcome_service._evaluate_one(
            snap, horizon, today=ANCHOR, benchmark=BENCH, db=db,
        )
        if out is not None:
            db.commit()
            db.refresh(out)
            db.expunge(out)
        return out, status


# ---------------------------------------------------------------------------
# Window sizing
# ---------------------------------------------------------------------------

def test_window_is_sized_off_memo_age_not_horizon():
    """The fetch ends today, so what it must span is memo date → today.

    The horizon never binds: a pair is only due once `target_date` has
    passed, so the memo is always at least `horizon_days` old by then.
    """
    rungs = outcome_service.PRICE_WINDOW_RUNGS
    assert list(rungs) == sorted(rungs), "rungs must be ascending"

    # A memo evaluated the night its 30d horizon comes due needs very
    # little history — but still lands on the first rung, so all four of a
    # snapshot's horizons share one cache key.
    fresh = outcome_service._window_days_for_memo(ANCHOR - timedelta(days=30), ANCHOR)
    assert fresh == rungs[0]

    # Four months old: the old sizing asked for 60 bars, roughly 84
    # calendar days, and missed the memo by a month.
    four_months = outcome_service._window_days_for_memo(
        ANCHOR - timedelta(days=120), ANCHOR,
    )
    assert four_months >= 120 + outcome_service.MEMO_WINDOW_BUFFER_DAYS

    # Monotone in memo age, and every answer covers its own span.
    for age in (1, 30, 100, 200, 364, 500, 700, 790):
        got = outcome_service._window_days_for_memo(
            ANCHOR - timedelta(days=age), ANCHOR,
        )
        assert got in rungs
        assert got >= age + outcome_service.MEMO_WINDOW_BUFFER_DAYS

    # Bounded: past the longest rung we refuse rather than fetch an
    # unbounded series.
    assert outcome_service._window_days_for_memo(
        ANCHOR - timedelta(days=5000), ANCHOR,
    ) is None


def test_all_horizons_of_one_memo_share_a_single_price_fetch():
    """Cache-key discipline: `days` is part of `"{TICKER}:{days}"`.

    Sizing per (memo, horizon, today) would rotate the key every night and
    multiply provider calls by the number of distinct memo dates. Sizing
    off memo age alone collapses a snapshot's four horizons onto one key —
    strictly fewer fetches than the four the old `horizon_days + 30` made.
    """
    memo_date = ANCHOR - timedelta(days=200)
    snap = _seed("TSTWCACHE", memo_date=memo_date)
    provider = _BarProvider({
        "TSTWCACHE": _tape(_flat(100.0), start=ANCHOR - timedelta(days=900)),
        BENCH: _tape(_flat(400.0), start=ANCHOR - timedelta(days=900)),
    })
    with _stub(provider):
        for horizon in outcome_service.DEFAULT_HORIZONS:
            _evaluate(snap, horizon)

    assert len(provider.days_for("TSTWCACHE")) == 1
    # The benchmark rides the same rung, so SPY stays one key too and both
    # legs of alpha are measured over the same span.
    assert provider.days_for(BENCH) == provider.days_for("TSTWCACHE")


def test_all_four_horizons_reuse_the_actual_persistent_provider_cache(monkeypatch):
    """Drive the service/cache path too: equal requested sizes alone do not
    prove that SQL cache lookups actually suppress provider calls.
    """
    from sqlalchemy import select

    from app.models import ProviderCache
    from app.services import market_data_service
    from app.services.data_service import DataService

    ticker, benchmark = "TSTWDBKEY", "TSTWDBBENCH"
    memo_date = ANCHOR - timedelta(days=500)
    snap = _seed(ticker, memo_date=memo_date)
    provider = _BarProvider({
        ticker: _tape(_flat(100.0), start=ANCHOR - timedelta(days=1300)),
        benchmark: _tape(_flat(400.0), start=ANCHOR - timedelta(days=1300)),
    })
    ds = DataService()
    # No registered test provider: retain the production persistent-cache
    # path, replacing only its outbound provider boundary.
    monkeypatch.setattr(ds, "_try_chain", lambda _cap, _fn, t, d: provider(t, d))
    monkeypatch.setattr(market_data_service, "get_data_service", lambda: ds)
    for horizon in outcome_service.DEFAULT_HORIZONS:
        with SessionLocal() as db:
            out, status = outcome_service._evaluate_one(
                snap, horizon, today=ANCHOR, benchmark=benchmark, db=db,
            )
            assert status == "written" and out is not None
            db.commit()
    assert provider.requests == [(ticker, 800), (benchmark, 800)]
    with SessionLocal() as db:
        keys = db.execute(select(ProviderCache.key).where(
            ProviderCache.capability == "prices",
            ProviderCache.key.in_([f"{ticker}:800", f"{benchmark}:800"]),
        )).scalars().all()
    assert set(keys) == {f"{ticker}:800", f"{benchmark}:800"}


@pytest.mark.parametrize("missing", [None, "baseline", "target"])
def test_alpha_uses_exact_actual_ticker_dates_or_is_unavailable(missing):
    memo_date = ANCHOR - timedelta(days=100)
    target_date = memo_date + timedelta(days=30)
    baseline = (memo_date + timedelta(days=2)).isoformat()
    target = (target_date - timedelta(days=1)).isoformat()
    ticker = f"TSTWALIGN{missing or 'OK'}".upper()
    snap = _seed(ticker, memo_date=memo_date)
    bench_rows = [
        {"date": memo_date.isoformat(), "close": 200},
        {"date": baseline, "close": 250},
        {"date": target, "close": 275},
        {"date": target_date.isoformat(), "close": 200},
    ]
    if missing:
        absent = baseline if missing == "baseline" else target
        bench_rows = [row for row in bench_rows if row["date"] != absent]
    provider = _BarProvider({
        ticker: [{"date": baseline, "close": 100}, {"date": target, "close": 110}],
        BENCH: bench_rows,
    })
    with _stub(provider):
        out, status = _evaluate(snap, 30)
    assert status == "written"
    assert out.forward_return == pytest.approx(0.1)
    if missing:
        assert out.alpha is None and out.benchmark_return is None
        assert "alpha_unavailable=benchmark_missing_exact_dates" in out.note
        assert f"benchmark_required_baseline={baseline}" in out.note
        assert f"benchmark_required_target={target}" in out.note
    else:
        assert out.benchmark_return == pytest.approx(0.1)
        assert out.alpha == pytest.approx(0.0)
        assert f"benchmark_baseline={baseline}" in out.note
        assert f"benchmark_target={target}" in out.note


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0, -1, "invalid"])
def test_invalid_target_price_is_never_scored(bad):
    memo_date = ANCHOR - timedelta(days=100)
    target = (memo_date + timedelta(days=30)).isoformat()
    snap = _seed("TSTWINVALID", memo_date=memo_date)
    provider = _BarProvider({
        snap.ticker: [{"date": memo_date.isoformat(), "close": 100}, {"date": target, "close": bad}],
        BENCH: [{"date": memo_date.isoformat(), "close": 200}, {"date": target, "close": 210}],
    })
    with _stub(provider):
        out, status = _evaluate(snap, 30)
    assert out is None and status == "price_window_incomplete"
    assert outcome_service.get_outcomes_for_snapshot(snap.id) == []


def test_dated_close_rejects_invalid_date_and_invalid_primary_even_with_adjusted_close():
    assert outcome_service._dated_close({"date": "not-a-date", "close": 100}) is None
    assert outcome_service._dated_close({"date": "2026-01-02", "close": 0, "adjusted_close": 100}) is None
    assert outcome_service._dated_close({"date": "2026-01-02", "close": float("nan"), "adjusted_close": 100}) is None
    assert outcome_service._dated_close({"date": "2026-01-02", "adjusted_close": 100}) == ("2026-01-02", 100)


# ---------------------------------------------------------------------------
# Claim A — an old memo must still be scoreable
# ---------------------------------------------------------------------------

def test_four_month_old_memo_scores_against_closes_near_its_own_dates():
    """The production symptom: nine of seventeen memo-carrying tickers had
    memos ~4 months old, and every 30d pair reported
    `price_window_incomplete` night after night.

    With the old sizing this pair fetched 60 bars (~84 calendar days),
    a window beginning three weeks *after* the target date, so no close on
    or before the target existed and nothing was ever written.
    """
    memo_date = ANCHOR - timedelta(days=120)   # 2026-05-14, a Thursday
    target_date = memo_date + timedelta(days=30)
    # The target itself falls on a weekend, so the step happens a week
    # early: whichever trading day stands in for the target carries 120.
    settled = target_date - timedelta(days=7)

    def close(d: _date) -> float:
        if d <= memo_date:
            return 100.0
        if d < settled:
            return 90.0        # a dip the memo-date baseline must not sample
        return 120.0

    snap = _seed("TSTWOLD", memo_date=memo_date, rating="Bullish")
    provider = _BarProvider({
        "TSTWOLD": _tape(close, start=ANCHOR - timedelta(days=700)),
        BENCH: _tape(_flat(400.0), start=ANCHOR - timedelta(days=700)),
    })
    with _stub(provider):
        out, status = _evaluate(snap, 30)

    assert status == "written"
    assert out is not None
    # Baseline is the memo day's own close, target is the target day's.
    assert out.price_at_memo == 100.0
    assert out.forward_return == pytest.approx(0.20)
    assert out.thesis_held is True
    assert f"baseline={memo_date.isoformat()}" in out.note
    assert "target=" in out.note
    assert "price_window=" in out.note
    assert "benchmark_baseline=" in out.note
    assert "benchmark_target=" in out.note
    # Flat benchmark → alpha is the whole return, and it is present at all.
    assert out.benchmark_return == pytest.approx(0.0)
    assert out.alpha == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Claim B — a shifted baseline must never be scored
# ---------------------------------------------------------------------------

def test_shifted_window_no_longer_scores_a_wrong_baseline():
    """The silent one. This memo is inside the band where the old window
    reached the target date but not the memo date.

    Old behaviour, reproduced end-to-end: the window began ~16 days after
    the memo, mid-drawdown, every row satisfied "on or after", and a
    bullish call that lost 20% was written as a gain with
    `thesis_held=True`. Nothing in the row or its note said so.
    """
    memo_date = ANCHOR - timedelta(days=100)   # 2026-06-03, a Wednesday
    target_date = memo_date + timedelta(days=30)
    settled = target_date - timedelta(days=7)

    def close(d: _date) -> float:
        if d <= memo_date:
            return 200.0
        if d < settled:
            return 140.0       # what the old code mistook for the baseline
        return 160.0

    snap = _seed("TSTWSHIFT", memo_date=memo_date, rating="Bullish")
    provider = _BarProvider({
        "TSTWSHIFT": _tape(close, start=ANCHOR - timedelta(days=700)),
        BENCH: _tape(_flat(400.0), start=ANCHOR - timedelta(days=700)),
    })
    with _stub(provider):
        out, status = _evaluate(snap, 30)

    assert status == "written"
    assert out is not None
    assert out.price_at_memo == 200.0
    assert out.forward_return == pytest.approx(-0.20)
    # The verdict, not just the number: a bullish call that lost 20% is broken.
    assert out.thesis_held is False
    assert out.alpha == pytest.approx(-0.20)

    # Sanity on the fixture itself: the old 60-bar window really did begin
    # after the memo and really would have produced the wrong sign.
    stale = provider.tapes["TSTWSHIFT"][-(30 + 30):]
    assert stale[0]["date"] > memo_date.isoformat()
    assert outcome_service._close_on_or_after(stale, memo_date.isoformat()) == 140.0


def test_history_that_starts_after_the_memo_is_refused_not_guessed():
    """Same hole, reached a different way: a window wide enough to cover
    the memo, but a provider whose answer simply doesn't go back that far.

    A refusal to score is correct here; a confidently wrong alpha is not.
    The refusal is an *outage* status, not a permanent one — see
    `test_short_provider_response_is_an_outage_not_a_permanent_refusal`.
    """
    memo_date = ANCHOR - timedelta(days=100)
    history_starts = memo_date + timedelta(days=40)

    snap = _seed("TSTWSHORT", memo_date=memo_date, rating="Bullish")
    provider = _BarProvider({
        "TSTWSHORT": _tape(_flat(150.0), start=history_starts),
        BENCH: _tape(_flat(400.0), start=ANCHOR - timedelta(days=700)),
    })
    with _stub(provider):
        out, status = _evaluate(snap, 30)

    assert out is None
    assert status == "price_history_too_short"
    assert outcome_service.get_outcomes_for_snapshot(snap.id) == []


class _TruncatingProvider(_BarProvider):
    """Answers every request with the last `cap` bars, whatever was asked.

    This is not a hypothetical. `_live_chain("prices")` is
    `[fmp, tiingo, polygon]` and `_try_chain` takes the first truthy
    result, so when FMP misses, Tiingo answers. Its history adapter used
    to omit `startDate`, then slice `[-days:]` from a latest-only response.
    That adapter is fixed separately; provider failures can still yield
    partial history, and a slice cannot lengthen what came back.
    """

    def __init__(self, tapes, *, cap: int) -> None:
        super().__init__(tapes)
        self.cap = cap

    def __call__(self, ticker: str, days: int = 252) -> list[dict[str, Any]]:
        rows = super().__call__(ticker, days)
        return rows[-self.cap:]


def test_short_provider_response_is_an_outage_not_a_permanent_refusal():
    """A partial price outage must still turn the loop red.

    The permanent bucket exists so an unfixable memo cannot hold the loop
    red forever. Putting a truncated *response* in it inverts that: the
    loop goes green while writing nothing, which is the exact failure this
    whole module was written to remove. The provider here holds full
    history and simply hands back five bars — tomorrow it may hand back
    all of them, so nothing about this pair is permanent.
    """
    memo_date = ANCHOR - timedelta(days=40)   # 30d horizon, due 10 days ago
    snap = _seed("TSTWTRUNC", memo_date=memo_date, rating="Bullish")
    tapes = {
        "TSTWTRUNC": _tape(_flat(100.0), start=ANCHOR - timedelta(days=700)),
        BENCH: _tape(_flat(400.0), start=ANCHOR - timedelta(days=700)),
    }

    with _stub(_TruncatingProvider(tapes, cap=5)):
        out, status = _evaluate(snap, 30)
    assert out is None
    assert status == "price_history_too_short"

    with _stub(_TruncatingProvider(tapes, cap=5)):
        res = outcome_service.evaluate_all_due(horizons=[30], today=ANCHOR)
    assert res["written"] == 0
    assert res["price_history_too_short"] >= 1
    # Counted as an outage, so the loop's own condition reports failure.
    # (`evaluate_all_due` scans every snapshot in the DB, so only `>=`
    # assertions survive whatever other tests have seeded.)
    assert res["data_unavailable"] >= res["price_history_too_short"]
    assert (res["errors"] == 0 and res["data_unavailable"] == 0) is False

    # The contrast that matters, taken per-pair so no other test's
    # snapshots can colour it: the *same* truncating provider, and a memo
    # older than the remote ladder now needs the durable archive. Missing
    # archived history is repairable by the explicit backfill.
    ancient = _seed("TSTWTRUNCOLD", memo_date=ANCHOR - timedelta(days=1000))
    truncating = _TruncatingProvider(tapes, cap=5)
    with _stub(truncating):
        out, status = _evaluate(ancient, 30)
    assert out is None
    assert status == "ticker_prices_unavailable"
    assert truncating.requests == []   # settled before any provider call

    # Same pair, same code, provider restored: it scores. Nothing about
    # the memo was ever permanent.
    with _stub(_BarProvider(tapes)):
        out, status = _evaluate(snap, 30)
    assert status == "written"
    assert out is not None and out.price_at_memo == 100.0


def test_complete_daily_tapes_cover_the_memo_under_bar_and_calendar_sizing():
    """Complete tapes cover the memo under either interpretation of days.

    `window_days` is a calendar-day count and providers read it as bars,
    so a full-length response spans ~40% more calendar days than asked
    for; read as calendar days it spans exactly that many. Both are at
    least `memo age + MEMO_WINDOW_BUFFER_DAYS`, so both reach past the
    memo. Actual response dates still decide coverage: gaps, duplicate
    dates and invalid prices can invalidate a nominally full response.
    """
    for age in (1, 30, 100, 200, 364, 500, 700, 790):
        memo_date = ANCHOR - timedelta(days=age)
        window = outcome_service._window_days_for_memo(memo_date, ANCHOR)
        assert window is not None

        # Reading 1: `days` bars ending today.
        as_bars = _tape(_flat(100.0), start=ANCHOR - timedelta(days=1200))[-window:]
        assert len(as_bars) == window
        assert outcome_service._baseline_close(as_bars, memo_date) is not None

        # Reading 2: `days` calendar days ending today.
        as_days = _tape(_flat(100.0), start=ANCHOR - timedelta(days=window - 1))
        assert outcome_service._baseline_close(as_days, memo_date) is not None


def test_baseline_tolerance_is_a_bounded_number_of_days():
    """The tolerance has to absorb weekends and holidays without absorbing
    drift. Seven days clears every US market closure on record — the
    longest left seven calendar days between consecutive sessions.
    """
    memo_date = _date(2026, 6, 3)
    tol = outcome_service.PRICE_DATE_TOLERANCE_DAYS

    edge = [{"date": (memo_date + timedelta(days=tol)).isoformat(), "close": 11.0}]
    assert outcome_service._baseline_close(edge, memo_date) == (
        edge[0]["date"], 11.0,
    )

    just_past = [
        {"date": (memo_date + timedelta(days=tol + 1)).isoformat(), "close": 11.0},
    ]
    assert outcome_service._baseline_close(just_past, memo_date) is None

    # A close just *before* the memo is fine too — that is the price the
    # memo was most likely written against.
    before = [{"date": (memo_date - timedelta(days=1)).isoformat(), "close": 9.0}]
    assert outcome_service._baseline_close(before, memo_date) == (
        before[0]["date"], 9.0,
    )


def test_series_that_stops_before_the_target_is_not_scored_as_the_target():
    """The mirror image: a halted or delisted ticker whose last bar sits
    two months before the target date. Taking it as the target close turns
    a 90-day return into a 27-day one and writes it as a 90-day result.
    """
    memo_date = ANCHOR - timedelta(days=200)
    last_bar = memo_date + timedelta(days=30)

    snap = _seed("TSTWHALT", memo_date=memo_date, rating="Bullish")
    provider = _BarProvider({
        "TSTWHALT": _tape(_flat(50.0), start=memo_date - timedelta(days=60),
                          end=last_bar),
        BENCH: _tape(_flat(400.0), start=ANCHOR - timedelta(days=700)),
    })
    with _stub(provider):
        out, status = _evaluate(snap, 90)

    assert out is None
    assert status == "price_window_incomplete"


# ---------------------------------------------------------------------------
# Old memos can be repaired with durable history beyond the remote ladder
# ---------------------------------------------------------------------------

def test_memo_older_than_the_longest_window_requires_archive_without_remote_fetch():
    memo_date = ANCHOR - timedelta(days=1000)
    snap = _seed("TSTWANCIENT", memo_date=memo_date, rating="Bullish")
    provider = _BarProvider({
        "TSTWANCIENT": _tape(_flat(100.0), start=ANCHOR - timedelta(days=1200)),
        BENCH: _tape(_flat(400.0), start=ANCHOR - timedelta(days=1200)),
    })
    with _stub(provider):
        out, status = _evaluate(snap, 30)

    assert out is None
    assert status == "ticker_prices_unavailable"
    # This path reads the archive; the explicit backfill supplies missing dates.
    assert provider.requests == []


def test_missing_old_archived_history_is_counted_as_repairable_outage(caplog):
    memo_date = ANCHOR - timedelta(days=1000)
    snap = _seed("TSTWCOUNT", memo_date=memo_date, rating="Bullish")
    provider = _BarProvider({
        "TSTWCOUNT": _tape(_flat(100.0), start=ANCHOR - timedelta(days=1200)),
        BENCH: _tape(_flat(400.0), start=ANCHOR - timedelta(days=1200)),
    })
    with _stub(provider):
        res = outcome_service.evaluate_all_due(horizons=[30], today=ANCHOR)

    assert res["ticker_prices_unavailable"] >= 1
    assert res["data_unavailable"] >= 1
    assert res["memo_predates_price_window"] == 0
    assert res["unevaluable_pairs"] == []
    assert res["due"] >= res["data_unavailable"]
    assert outcome_service.get_outcomes_for_snapshot(snap.id) == []


def test_loop_stays_green_when_the_only_shortfall_is_permanent(monkeypatch):
    """`success = errors == 0 and data_unavailable == 0` is right only if
    `data_unavailable` excludes the unfixable. A loop that reports failure
    every night for a reason nobody can act on trains everyone to ignore
    it — this repo has been bitten by that twice.
    """
    from app.monitoring import outcome_loop

    result = {
        "evaluated": 40, "due": 4, "written": 2, "already_recorded": 0,
        "data_unavailable": 0, "ticker_prices_unavailable": 0,
        "price_history_too_short": 0,
        "price_window_incomplete": 0, "unevaluable": 2,
        "memo_predates_price_window": 2, "reflections": 0, "errors": 0,
    }
    calls: list[tuple[tuple, dict[str, Any]]] = []
    monkeypatch.setattr(outcome_loop, "evaluate_all_due", lambda: result)
    monkeypatch.setattr(
        outcome_loop, "record_run",
        lambda *a, **k: calls.append((a, k)),
    )

    outcome_loop.run_once()
    (_, kwargs), = calls
    assert kwargs["success"] is True
    assert "unevaluable=2" in kwargs["note"]


def test_loop_note_names_each_shortfall_separately(monkeypatch):
    """An operator has to tell "the provider has no data for this ticker"
    from "the window has a gap" from "no history reaches this memo"."""
    from app.monitoring import outcome_loop

    result = {
        "evaluated": 40, "due": 6, "written": 0, "already_recorded": 0,
        "data_unavailable": 5, "ticker_prices_unavailable": 3,
        "price_history_too_short": 1,
        "price_window_incomplete": 1, "unevaluable": 2,
        "memo_predates_price_window": 2, "reflections": 0, "errors": 0,
        "unevaluable_pairs": ["OLD:snap=1:30d", "OLD:snap=1:90d"],
    }
    calls: list[tuple[tuple, dict[str, Any]]] = []
    monkeypatch.setattr(outcome_loop, "evaluate_all_due", lambda: result)
    monkeypatch.setattr(
        outcome_loop, "record_run",
        lambda *a, **k: calls.append((a, k)),
    )

    outcome_loop.run_once()
    (_, kwargs), = calls
    note = kwargs["note"]
    assert "no_prices=3" in note
    assert "short_history=1" in note
    assert "window_gap=1" in note
    assert "unevaluable=2" in note
    assert "unevaluable_pairs=OLD:snap=1:30d,OLD:snap=1:90d" in note
    # A real outage still turns the loop red.
    assert kwargs["success"] is False
