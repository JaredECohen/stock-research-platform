"""`app.scripts.backfill_month_end_prices` — the one-time deep month-end
price backfill that lifts `scorecard_evaluation` out of
`insufficient_data`.

Offline by construction: every provider read goes through a stub swapped
into `data_service.get_data_service`, and the stub COUNTS its calls so the
budget, the dry run and the resume rules are asserted on the thing that
actually costs money rather than on the report's own bookkeeping.

Synthetic tickers (`BMEA`, `BMEB`, …) keep these rows away from the demo
names other suites seed.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from app.config import settings
from app.database import SessionLocal
from app.models import Company, PriceMonthEnd, ScorecardRun, ScorecardScore
from app.scripts import backfill_month_end_prices as bmep
from app.services import data_service as ds_mod
from app.services import scorecard_evaluation, scorecard_pit, scorecard_queue, scorecard_service

A, B, C, SHORT = "BMEA", "BMEB", "BMEC", "BMES"
# Five extra names so a month can form five quintile legs with min_leg 1.
PANEL = ("BMEP1", "BMEP2", "BMEP3", "BMEP4", "BMEP5", "BMEP6")
ALL = (A, B, C, SHORT, *PANEL)

# The store's newest complete month in these tests. Fixed, not "today", so
# the arithmetic below is the same in every month of the year.
ANCHOR = date(2026, 8, 31)


def _prev(d: date, back: int = 1) -> date:
    return scorecard_evaluation._prev_month_end(d, back)


def _month_ends(n: int, *, end: date = ANCHOR) -> list[date]:
    """The `n` most recent month ends up to and including `end`, oldest first."""
    out: list[date] = []
    cur = end
    for _ in range(n):
        out.append(cur)
        cur = _prev(cur)
    return sorted(out)


def _clear() -> None:
    with SessionLocal() as db:
        scorecard_pit._ensure_tables(db)
        scorecard_evaluation._ensure_tables(db)
        bmep._ensure_attempts_table(db)
        run_ids = [r[0] for r in db.query(ScorecardRun.id).filter(
            ScorecardRun.requested_by == "test-bmep").all()]
        db.query(ScorecardScore).filter(ScorecardScore.ticker.in_(ALL)).delete(synchronize_session=False)
        if run_ids:
            db.query(ScorecardRun).filter(ScorecardRun.id.in_(run_ids)).delete(synchronize_session=False)
        db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker.in_(ALL)).delete(synchronize_session=False)
        db.query(Company).filter(Company.ticker.in_(ALL)).delete(synchronize_session=False)
        db.execute(bmep.price_backfill_attempts.delete().where(
            bmep.price_backfill_attempts.c.ticker.in_(ALL)))
        db.commit()


@pytest.fixture(autouse=True)
def _clean():
    _clear()
    yield
    _clear()


class _StubProvider:
    """Stands in for the whole provider chain.

    `months[ticker]` is how deep that name's history goes; a ticker absent
    from the map returns None (every provider failed), which
    `sync_price_month_ends` raises as `PriceSeriesUnavailable`. The
    returned series honours `days` the way FMP's `limit` does, so a short
    window really does return a short history.
    """

    TRADING_DAYS_PER_MONTH = 21

    def __init__(self, months: dict[str, int]):
        self.months = dict(months)
        self.calls: list[tuple[str, int]] = []

    def get_price_history(self, ticker: str, days: int = 252):
        ticker = ticker.upper()
        self.calls.append((ticker, days))
        depth = self.months.get(ticker)
        if depth is None:
            return None
        rows = [
            {"date": m.isoformat(), "close": 100.0 + i, "adjusted_close": 100.0 + i}
            for i, m in enumerate(_month_ends(depth))
        ]
        limit = max(1, days // self.TRADING_DAYS_PER_MONTH)
        return rows[-limit:]

    def mode(self) -> str:
        return "stub"

    @property
    def tickers_called(self) -> list[str]:
        return [t for t, _ in self.calls]


@pytest.fixture
def stub(monkeypatch):
    def _install(months: dict[str, int]) -> _StubProvider:
        provider = _StubProvider(months)
        monkeypatch.setattr(ds_mod, "get_data_service", lambda: provider)
        return provider
    return _install


def _seed_scored_months(
    ticker: str, months: list[date], *, status: str | None = None,
    market_cap: float | None = None, roa: float | None = None, score: float = 0.1,
) -> None:
    """One succeeded month-end scorecard run per (ticker, month), with one
    score row — the other half of a panel observation."""
    status = status or scorecard_queue.STATUS_SUCCEEDED
    with SessionLocal() as db:
        scorecard_evaluation._ensure_tables(db)
        for m in months:
            run = ScorecardRun(
                run_id=f"test-bmep-{ticker}-{m.isoformat()}", version_key=scorecard_service.VERSION_KEY,
                as_of=m, run_kind=scorecard_queue.KIND_MONTH_END, status=status,
                requested_by="test-bmep", enqueued_at=datetime(2026, 9, 1, 12, 0, 0),
            )
            db.add(run)
            db.flush()
            raw: dict = {}
            if market_cap is not None:
                raw["_context"] = {"market_cap": market_cap}
            if roa is not None:
                raw["roa"] = roa
            db.add(ScorecardScore(
                run_id=run.id, version_key=scorecard_service.VERSION_KEY, as_of=m, ticker=ticker,
                overall_z=score, coverage=0.9, is_month_end=True, feature_raw=raw,
            ))
        db.commit()


def _seed_universe_month(
    tickers: tuple[str, ...], months: list[date], *, with_controls: bool,
) -> None:
    """ONE succeeded run per month covering every ticker — the shape
    production writes, and the shape `build_panel` needs (it keeps the
    latest succeeded run per month and discards the rest)."""
    with SessionLocal() as db:
        scorecard_evaluation._ensure_tables(db)
        for m in months:
            run = ScorecardRun(
                run_id=f"test-bmep-universe-{m.isoformat()}", version_key=scorecard_service.VERSION_KEY,
                as_of=m, run_kind=scorecard_queue.KIND_MONTH_END,
                status=scorecard_queue.STATUS_SUCCEEDED, requested_by="test-bmep",
                enqueued_at=datetime(2026, 9, 1, 12, 0, 0),
            )
            db.add(run)
            db.flush()
            for i, t in enumerate(tickers):
                raw: dict = {}
                if with_controls:
                    raw = {"_context": {"market_cap": 1e9 * (i + 1)}, "roa": 0.05 + 0.01 * i}
                db.add(ScorecardScore(
                    run_id=run.id, version_key=scorecard_service.VERSION_KEY, as_of=m, ticker=t,
                    sector="Technology", overall_z=(i - 2.5) / 2.0, coverage=0.9,
                    is_month_end=True, feature_raw=raw,
                ))
        db.commit()


def _seed_companies(tickers: tuple[str, ...], *, beta: float | None) -> None:
    with SessionLocal() as db:
        for t in tickers:
            db.add(Company(
                ticker=t, company_name=t, sector="Technology", industry="Software", beta=beta,
            ))
        db.commit()


@pytest.fixture
def thin_legs(monkeypatch):
    """Five quintile legs of one name each, so a six-name panel can produce
    eligible months. Everything else about the evaluation is untouched."""
    monkeypatch.setattr(settings, "scorecard_min_leg_n", 1)
    return settings


# ---------------------------------------------------------------------------
# The price-store arithmetic — pure, no DB, no provider
# ---------------------------------------------------------------------------

def test_a_contiguous_run_of_L_month_ends_yields_L_minus_13_evaluable_months():
    """`build_panel` needs M-12, M-1, M and M+1 stored for month M, so the
    first 12 and the last month of any run can never be observations. If
    this drifts, the script would promise the evaluation months it does
    not have."""
    assert bmep.LEAD_MONTHS == 13
    for depth in (0, 1, 13, 14, 15, 37, 40, 60):
        stored = _month_ends(depth)
        assert len(bmep.evaluable_months(stored)) == max(0, depth - 13), depth
    # 24 price-evaluable months — the script's per-name target — needs 37.
    assert len(bmep.evaluable_months(_month_ends(37))) == bmep.DEFAULT_DEPTH_TARGET_MONTHS
    assert len(bmep.evaluable_months(_month_ends(36))) == bmep.DEFAULT_DEPTH_TARGET_MONTHS - 1


def test_a_hole_in_the_store_invalidates_every_month_that_needed_it():
    """One missing month end kills four observations (itself, the month
    before, the month after, and the month 12 later), and is reported as a
    gap rather than absorbed into the total."""
    stored = _month_ends(40)
    missing = stored[20]
    holed = [m for m in stored if m != missing]
    assert len(bmep.evaluable_months(stored)) == 27
    assert len(bmep.evaluable_months(holed)) == 23

    cov = bmep.build_coverage([A], {A: set(holed)})
    row = cov.tickers[0]
    assert row.months_stored == 39
    assert row.gap_months == 1
    assert row.price_evaluable == 23
    assert row.depth_target_met is False
    assert "39 month ends stored (1 missing inside the span)" in (row.reason or "")
    assert "23 price-evaluable months; 24 is this script's per-name target" in (row.reason or "")
    assert "37 contiguous month ends reach it" in (row.reason or "")


def test_an_empty_store_reports_a_reason_not_a_bare_zero():
    cov = bmep.build_coverage([A], {A: set()})
    row = cov.tickers[0]
    assert (row.months_stored, row.price_evaluable) == (0, 0)
    assert row.first_month is None and row.last_month is None
    assert row.reason == "no month ends stored for this ticker"
    d = cov.to_dict()
    assert d["months_stored_min"] == 0  # one ticker, zero months: measured, not unknown
    empty = bmep.build_coverage([], {}).to_dict()
    assert empty["months_stored_min"] is None
    assert empty["months_stored_reason"] == "no tickers in scope"


def test_the_price_side_totals_are_labelled_as_a_ceiling_on_the_panel_wide_floors():
    """`min_obs` is a TOTAL across names and `min_months` is a count of
    DISTINCT month ends across the whole panel (see
    `test_the_evaluation_floors_are_panel_wide_not_per_name`). The price
    numbers are the reachable upper bounds on both, never a per-name
    claim."""
    price = {t: set(_month_ends(40)) for t in (A, B)}
    cov = bmep.build_coverage([A, B], price, min_obs=54, min_months=27)
    # 27 evaluable months per name, but the SAME 27 calendar months.
    assert cov.price_evaluable == 54 and cov.price_evaluable_months == 27
    assert cov.price_could_meet_min_obs is True
    assert cov.price_could_meet_min_months is True
    assert bmep.build_coverage([A, B], price, min_obs=55).price_could_meet_min_obs is False
    assert bmep.build_coverage([A, B], price, min_months=28).price_could_meet_min_months is False
    # One name alone reaches the same 27 distinct months: months are shared.
    assert bmep.build_coverage([A], {A: price[A]}).price_evaluable_months == 27


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_plan_skips_finished_names_and_names_what_the_budget_cut():
    price = {A: set(_month_ends(40)), B: set(_month_ends(5)), C: set(), SHORT: set()}
    names = [A, B, C, SHORT]
    cov = bmep.build_coverage(names, price)
    plan = bmep.plan_backfill(names, cov, max_tickers=1)
    assert plan.satisfied == (A,)          # already at the target: no paid call
    assert plan.fetch == (B,)              # the budget's one slot
    assert plan.over_budget == (C, SHORT)  # counted and named, never dropped
    assert plan.exhausted == ()
    assert plan.to_dict()["n_over_budget"] == 2
    assert plan.needs_work == 3            # one selected + two cut

    # `--include-satisfied` puts the finished name back in the queue.
    greedy = bmep.plan_backfill(names, cov, max_tickers=4, include_satisfied=True)
    assert greedy.fetch == (A, B, C, SHORT) and greedy.satisfied == ()
    with pytest.raises(ValueError):
        bmep.plan_backfill(names, cov, max_tickers=-1)


def test_an_attempt_at_this_depth_retires_a_name_that_still_falls_short():
    """The resume rule that makes the loop terminate. Without it a name
    that CANNOT reach the target is re-selected every run, eats the budget
    forever and blocks the rest of the universe."""
    names = [SHORT, B]
    cov = bmep.build_coverage(names, {SHORT: set(_month_ends(10)), B: set()})
    attempt = bmep.Attempt(
        ticker=SHORT, attempted_at=datetime(2026, 9, 1), requested_days=1260,
        months_returned=10, outcome=bmep.OUTCOME_FETCHED, reason="10 complete month ends returned",
    )
    plan = bmep.plan_backfill(names, cov, max_tickers=1, days=1260, attempts={SHORT: attempt})
    assert plan.exhausted == (SHORT,)
    assert plan.fetch == (B,), "the slot goes to a name that can still use it"
    assert plan.needs_work == 1
    reason = plan.exhausted_reasons[SHORT]
    assert "asked on 2026-09-01 at 1260 trading days and got 10 complete months" in reason
    assert "--retry-exhausted" in reason

    # A DEEPER request is a new question, so the name comes back.
    deeper = bmep.plan_backfill(names, cov, max_tickers=2, days=1500, attempts={SHORT: attempt})
    assert deeper.exhausted == () and deeper.fetch == (SHORT, B)
    # …and so does an explicit retry at the same depth.
    forced = bmep.plan_backfill(names, cov, max_tickers=2, days=1260,
                                attempts={SHORT: attempt}, retry_exhausted=True)
    assert forced.exhausted == () and forced.fetch == (SHORT, B)


def test_an_outage_is_retired_too_and_says_so_rather_than_reading_as_zero_months():
    cov = bmep.build_coverage([B], {B: set()})
    attempt = bmep.Attempt(
        ticker=B, attempted_at=datetime(2026, 9, 1), requested_days=1260, months_returned=None,
        outcome=bmep.OUTCOME_UNAVAILABLE, reason="price series unavailable: chain returned nothing",
    )
    plan = bmep.plan_backfill([B], cov, max_tickers=5, days=1260, attempts={B: attempt})
    assert plan.exhausted == (B,) and plan.fetch == ()
    reason = plan.exhausted_reasons[B]
    assert "got no series at all" in reason, "None must not print as 0 months"
    assert "unavailable" in reason and "--retry-exhausted" in reason


# ---------------------------------------------------------------------------
# End to end, against the stub provider
# ---------------------------------------------------------------------------

def test_dry_run_fetches_nothing_and_reports_what_it_would_do(stub):
    provider = stub({A: 40, B: 40})
    report = bmep.run(tickers=[A, B], max_tickers=1, dry_run=True)
    assert provider.calls == [], "a dry run must not reach a paid provider"
    with SessionLocal() as db:
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker.in_(ALL)).count() == 0
        assert bmep.load_attempts(db, [A, B]) == {}, "a dry run records no attempt"
    assert report["budget"]["fetch"] == [A]
    assert report["budget"]["over_budget"] == [B]
    assert report["budget"]["needs_work"] == 2
    # No zero standing in for "not measured".
    assert report["fetched"] is None and "dry run" in report["fetched_reason"]
    assert report["coverage_after"] is None and "unchanged" in report["coverage_after_reason"]
    assert report["evaluation_after"] is None and "unchanged" in report["evaluation_after_reason"]
    assert report["delta"] is None and "nothing changed" in report["delta_reason"]
    text = bmep.format_report(report)
    assert "DRY RUN (nothing fetched, nothing written)" in text
    assert "would fetch           BMEA" in text
    assert "names still to fetch  2" in text


def test_the_budget_stops_the_run_and_the_next_run_picks_up_where_it_left_off(stub):
    provider = stub({A: 40, B: 40, C: 40})
    first = bmep.run(tickers=[A, B, C], max_tickers=1)
    assert provider.tickers_called == [A]
    assert first["budget"]["n_over_budget"] == 2 and first["budget"]["over_budget"] == [B, C]
    assert first["budget"]["needs_work"] == 3
    assert first["coverage_after"]["names_at_depth_target"] == 1

    second = bmep.run(tickers=[A, B, C], max_tickers=1)
    assert provider.tickers_called == [A, B], "the finished name must not be re-fetched"
    assert second["budget"]["satisfied"] == [A]
    assert second["budget"]["needs_work"] == 2
    assert second["coverage_after"]["names_at_depth_target"] == 2
    assert "skipped for budget    BMEC" in bmep.format_report(second)


def test_a_name_the_provider_cannot_fill_does_not_eat_the_budget_on_every_run(stub):
    """The realistic resume case, and the one a suite of only-clearing
    names never reaches: `BMES` has ten months of history, so it can NEVER
    reach the 24-month target however often it is bought. It must be asked
    exactly once and then retired with the reason, or the operator's
    documented loop ("keep re-running the SAME command") pays forever and
    never advances to `BMEB` or `BMEC`.
    """
    provider = stub({SHORT: 10, B: 40, C: 40})
    names = [SHORT, B, C]          # the run sorts them: BMEB, BMEC, BMES
    reports = [bmep.run(tickers=names, max_tickers=1) for _ in range(4)]

    assert provider.tickers_called == [B, C, SHORT], (
        "one paid call per name: the short name must not be re-bought, and the "
        "budget must reach the names that can use it"
    )
    # Every name is resolved, so the documented stop signal is reachable.
    assert [r["budget"]["needs_work"] for r in reports] == [3, 2, 1, 0]
    last = reports[-1]
    assert last["budget"]["exhausted"] == [SHORT]
    assert last["budget"]["satisfied"] == [B, C]
    assert last["fetched"]["months"] == 0 and last["fetched"]["written"] == 0
    reason = last["budget"]["exhausted_reasons"][SHORT]
    assert "10 complete months" in reason and "1260 trading days" in reason
    text = bmep.format_report(last)
    assert "names still to fetch  0" in text
    assert f"EXHAUSTED {SHORT}:" in text

    with SessionLocal() as db:
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == B).count() == 40
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == C).count() == 40


def test_a_dead_provider_chain_is_retired_after_one_run_not_bought_forever(stub):
    """`PriceSeriesUnavailable` leaves the name un-satisfied just like a
    short history does, so it stalls the loop the same way unless the
    attempt is recorded."""
    provider = stub({B: 40})           # SHORT is absent: the chain returns None
    names = [SHORT, B]                 # sorted: BMEB, then BMES
    for _ in range(3):
        report = bmep.run(tickers=names, max_tickers=1)
    assert provider.tickers_called == [B, SHORT]
    assert report["budget"]["exhausted"] == [SHORT]
    assert report["budget"]["needs_work"] == 0
    with SessionLocal() as db:
        attempt = bmep.load_attempts(db, names)[SHORT]
    assert attempt.outcome == bmep.OUTCOME_UNAVAILABLE
    assert attempt.months_returned is None, "an outage is not a zero-month ticker"

    # The documented remedy re-opens it once the provider is back.
    provider.months[SHORT] = 40
    recovered = bmep.run(tickers=names, max_tickers=5, retry_exhausted=True)
    assert provider.tickers_called == [B, SHORT, SHORT]
    assert recovered["coverage_after"]["names_at_depth_target"] == 2


def test_rerunning_writes_nothing_and_duplicates_nothing(stub):
    """Idempotency at both levels: the whole ticker is skipped once it
    reaches the target, and forcing the fetch anyway rewrites no row."""
    provider = stub({A: 40})
    first = bmep.run(tickers=[A], max_tickers=5)
    assert first["fetched"]["written"] == 40
    with SessionLocal() as db:
        rows = db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == A).count()
    assert rows == 40

    again = bmep.run(tickers=[A], max_tickers=5)
    assert again["budget"]["n_fetch"] == 0 and again["budget"]["satisfied"] == [A]
    assert again["budget"]["needs_work"] == 0
    assert provider.calls == [(A, bmep.DEFAULT_DAYS)]
    assert again["delta"]["months_stored"] == 0
    assert again["delta"]["price_evaluable"] == 0
    assert again["delta"]["names_at_depth_target"] == 0

    forced = bmep.run(tickers=[A], max_tickers=5, include_satisfied=True)
    assert forced["fetched"]["written"] == 0, "an unchanged month end must not be rewritten"
    assert forced["fetched"]["months"] == 40
    with SessionLocal() as db:
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == A).count() == 40


def test_a_short_history_ticker_is_reported_with_the_reason_not_silently_counted(stub):
    provider = stub({A: 40, SHORT: 10})
    report = bmep.run(tickers=[A, SHORT], max_tickers=5)
    assert sorted(provider.tickers_called) == [A, SHORT]
    partial = {p["ticker"]: p for p in report["partial"]}
    assert list(partial) == [SHORT], "only the name that still falls short is flagged"
    row = partial[SHORT]
    assert row["months_stored"] == 10 and row["price_evaluable"] == 0 and row["gap_months"] == 0
    assert "10 month ends stored (0 missing inside the span) yield 0 price-evaluable months" in row["reason"]
    assert "PARTIAL BMES:" in bmep.format_report(report)
    # It is counted in the store, not in the names at the target.
    assert report["coverage_after"]["months_stored"] == 50
    assert report["coverage_after"]["names_at_depth_target"] == 1


def test_a_provider_outage_is_an_error_with_a_reason_not_a_zero_month_ticker(stub):
    provider = stub({A: 40})            # B is absent: the chain returns None
    report = bmep.run(tickers=[A, B], max_tickers=5)
    assert provider.tickers_called == [A, B]
    assert report["fetched"]["errors"] == 1 and report["fetched"]["unavailable"] == 1
    failed = report["fetched"]["failed"]
    assert [f["ticker"] for f in failed] == [B]
    assert "price series unavailable" in failed[0]["reason"]
    with SessionLocal() as db:
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == B).count() == 0
    assert "FAILED BMEB:" in bmep.format_report(report)


def test_the_printed_price_coverage_matches_the_stored_rows(stub):
    stub({A: 40, B: 40})
    report = bmep.run(tickers=[A, B], max_tickers=5)
    before, after = report["coverage_before"], report["coverage_after"]

    with SessionLocal() as db:
        stored = {
            t: {m for (m,) in db.query(PriceMonthEnd.month_end).filter(PriceMonthEnd.ticker == t).all()}
            for t in (A, B)
        }
    assert after["months_stored"] == sum(len(v) for v in stored.values()) == 80
    per = {row["ticker"]: row for row in after["tickers"]}
    assert per[A]["months_stored"] == 40 and per[B]["months_stored"] == 40
    assert per[A]["price_evaluable"] == 27 and per[B]["price_evaluable"] == 27
    # 54 observations, but only 27 distinct months — the two names share them.
    assert after["price_evaluable"] == 54 and after["price_evaluable_months"] == 27
    assert after["names_at_depth_target"] == 2

    assert report["delta"]["months_stored"] == after["months_stored"] - before["months_stored"]
    assert report["delta"]["price_evaluable"] == after["price_evaluable"] - before["price_evaluable"]
    assert report["delta"]["price_evaluable_months"] == 27
    assert report["delta"]["names_at_depth_target"] == 2
    text = bmep.format_report(report)
    assert "+80 month ends" in text and "+2 names at the per-name target" in text


# ---------------------------------------------------------------------------
# The headline number is the evaluation's own
# ---------------------------------------------------------------------------

def test_a_full_price_store_can_still_yield_zero_usable_observations(stub, thin_legs):
    """The number this script exists to print must come from
    `scorecard_evaluation` itself, not from intersecting prices with score
    rows. A universe at FULL price depth, with a succeeded month-end score
    row for every evaluable month, yields ZERO usable observations when
    the LASSO's controls are missing — and the read-out has to say so with
    the reason instead of reporting the price-side ceiling as the answer.
    """
    stub({t: 40 for t in PANEL})
    months = sorted(bmep.evaluable_months(_month_ends(40)))
    _seed_universe_month(PANEL, months, with_controls=False)   # no market_cap, no roa
    _seed_companies(PANEL, beta=None)                          # no beta either

    report = bmep.run(tickers=list(PANEL), max_tickers=len(PANEL))
    after = report["coverage_after"]
    assert after["names_at_depth_target"] == len(PANEL), "price depth is full"
    assert after["price_evaluable"] == 27 * len(PANEL)

    ev = report["evaluation_after"]
    assert ev is not None and report["evaluation_after_reason"] is None
    assert ev["panel_rows"] == 27 * len(PANEL), "the rows exist…"
    assert ev["n_obs"] == 0 and ev["n_months"] == 0, "…and not one of them is usable"
    assert ev["verdict"] == "insufficient_data"
    assert ev["meets_min_obs"] is False and ev["meets_min_months"] is False
    assert ev["n_skipped_rows"] == 27 * len(PANEL)
    assert ev["n_skipped_by_reason"] == {"missing_control:log_mktcap": 27 * len(PANEL)}

    text = bmep.format_report(report)
    assert "EVALUATION READ-OUT (scorecard_evaluation's OWN panel — this is the gate):" in text
    assert "usable observations   0 of 2000 required (not met)" in text
    assert "dropped because       missing_control:log_mktcap=162" in text


def test_the_read_out_moves_when_the_missing_control_arrives(stub, thin_legs):
    """The companion to the test above: the same prices and the same score
    months, now with the controls the LASSO needs, produce real
    observations. Proves the read-out tracks the panel rather than
    printing a constant."""
    stub({t: 40 for t in PANEL})
    months = sorted(bmep.evaluable_months(_month_ends(40)))
    _seed_universe_month(PANEL, months, with_controls=True)
    _seed_companies(PANEL, beta=1.1)

    report = bmep.run(tickers=list(PANEL), max_tickers=len(PANEL))
    ev = report["evaluation_after"]
    assert ev["n_obs"] == 27 * len(PANEL)
    assert ev["n_months"] == 27
    assert ev["n_skipped_by_reason"] == {}
    # Still insufficient — 162 observations, 2000 required — but now the
    # reason is the sample size, not a missing column.
    assert ev["verdict"] == "insufficient_data"
    assert ev["meets_min_months"] is True and ev["meets_min_obs"] is False
    assert any("2000 required" in r for r in ev["reasons"])
    assert report["delta"]["evaluation_n_obs"] == 27 * len(PANEL)
    assert report["delta"]["evaluation_n_months"] == 27


def test_the_evaluation_floors_are_panel_wide_not_per_name():
    """`double_selection` counts `len(unique(month_ids))` across the WHOLE
    panel and `min_obs` is a total, so neither floor is a per-name month
    count. A hundred names present in twenty months each, staggered so the
    union spans thirty-four months, draws a verdict — while a per-name
    reading of the same panel would call every single name short.
    """
    import numpy as np

    from app.finance import double_selection_lasso as dsl

    rng = np.random.default_rng(0)
    y, d, X, ids = [], [], [], []
    for name in range(100):
        start = name % 15
        for k in range(20):
            y.append(rng.normal())
            d.append(rng.normal())
            X.append([rng.normal()])
            ids.append(f"m{start + k:02d}")
    res = dsl.double_selection(np.array(y), np.array(d), np.array(X), np.array(ids),
                               control_names=["x"])
    assert len(set(ids)) == 34
    assert res.n_months == 34 and res.n_obs == 2000
    assert res.verdict != dsl.VERDICT_INSUFFICIENT

    # The script therefore never claims a per-name month count IS the gate.
    cov = bmep.build_coverage(["X1"], {"X1": set(_month_ends(20))})
    assert not hasattr(cov, "panel_names_clearing")
    assert not hasattr(cov.tickers[0], "clears_minimum")
    assert "is this script's per-name target" in (cov.to_dict()["tickers"][0]["reason"] or "")


def test_a_failed_scoring_run_contributes_no_panel_rows(stub, thin_legs):
    """`build_panel` counts SUCCEEDED month-end runs only, so a failed run
    must leave the read-out empty rather than counted."""
    stub({A: 40})
    _seed_scored_months(A, sorted(bmep.evaluable_months(_month_ends(40))),
                        status=scorecard_queue.STATUS_FAILED)
    report = bmep.run(tickers=[A], max_tickers=5)
    assert report["coverage_after"]["names_at_depth_target"] == 1
    ev = report["evaluation_after"]
    assert ev["panel_rows"] == 0 and ev["n_obs"] == 0


def test_a_broken_read_out_is_named_not_rendered_as_zero(stub, monkeypatch):
    """House rule: a missing value carries its reason. A read-out that
    blows up must not print as `n_obs = 0`, which would read as a measured
    empty panel."""
    stub({A: 40})

    def _boom(*_a, **_kw):
        raise RuntimeError("panel exploded")

    monkeypatch.setattr(scorecard_evaluation, "build_panel", _boom)
    report = bmep.run(tickers=[A], max_tickers=5)
    assert report["evaluation_after"] is None
    assert "RuntimeError" in report["evaluation_after_reason"]
    assert report["delta"]["evaluation_n_obs"] is None
    assert "unknown" in report["delta"]["evaluation_reason"]
    text = bmep.format_report(report)
    assert "EVALUATION READ-OUT: not measured (RuntimeError" in text


def test_the_read_out_creates_no_run_row_and_writes_nothing(stub):
    """It is a measurement, not an evaluation: `run_evaluation` would
    create a `scorecard_runs` row and persist three results."""
    stub({A: 40})
    with SessionLocal() as db:
        before = db.query(ScorecardRun).count()
    bmep.read_evaluation(scorecard_service.VERSION_KEY)
    with SessionLocal() as db:
        assert db.query(ScorecardRun).count() == before


# ---------------------------------------------------------------------------
# The attempt ledger
# ---------------------------------------------------------------------------

def test_every_fetch_records_its_attempt_and_the_record_survives_the_process(stub):
    provider = stub({A: 40, SHORT: 10})
    bmep.run(tickers=[A, SHORT], max_tickers=5)
    assert sorted(provider.tickers_called) == [A, SHORT]
    with SessionLocal() as db:
        attempts = bmep.load_attempts(db, [A, SHORT, B])
    assert sorted(attempts) == [A, SHORT]
    assert attempts[A].months_returned == 40 and attempts[A].outcome == bmep.OUTCOME_FETCHED
    assert attempts[SHORT].months_returned == 10
    assert attempts[A].requested_days == bmep.DEFAULT_DAYS
    assert attempts[A].covers(bmep.DEFAULT_DAYS) and not attempts[A].covers(bmep.DEFAULT_DAYS + 1)
    assert B not in attempts, "a name that was never asked has no row"


def test_a_second_attempt_replaces_the_first_rather_than_accumulating(stub):
    stub({SHORT: 10})
    bmep.run(tickers=[SHORT], max_tickers=5)
    bmep.run(tickers=[SHORT], max_tickers=5, retry_exhausted=True, days=400)
    with SessionLocal() as db:
        rows = db.execute(bmep.price_backfill_attempts.select().where(
            bmep.price_backfill_attempts.c.ticker == SHORT)).all()
    assert len(rows) == 1
    assert rows[0].requested_days == 400


def test_the_ledger_table_is_not_application_schema():
    """Operator bookkeeping, deliberately off `Base.metadata`: it must not
    appear in `init_db`, the schema-drift reconciler or the frozen table
    name set."""
    from app.database import Base

    assert "price_backfill_attempts" not in Base.metadata.tables
    assert bmep.price_backfill_attempts.metadata is not Base.metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_the_requested_window_is_what_reaches_the_provider(stub):
    """`--days` is a parameter, not a cap: it must arrive at
    `get_price_history` so the deep fetch gets its OWN provider-cache entry
    (`ticker:days`) and leaves the app's 252-day series alone."""
    provider = stub({A: 40})
    bmep.run(tickers=[A], max_tickers=5, days=420)
    assert provider.calls == [(A, 420)]
    with SessionLocal() as db:
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == A).count() == 20


def test_cli_dry_run_emits_valid_json_and_exits_zero(stub, capsys):
    stub({A: 40})
    rc = bmep.main(["--tickers", f"{A.lower()},{B.lower()}", "--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["universe"] == {"n_tickers": 2, "source": "explicit --tickers"}
    assert payload["days"] == bmep.DEFAULT_DAYS
    assert payload["budget"]["max_tickers"] == bmep.DEFAULT_MAX_TICKERS
    # The documented stop signal is in the JSON under the name the
    # docstring tells the operator to watch.
    assert payload["budget"]["needs_work"] == 2


def test_cli_exits_nonzero_when_a_ticker_failed(stub, capsys):
    stub({})                 # every name is an outage
    rc = bmep.main(["--tickers", A, "--max-tickers", "1", "--json"])
    capsys.readouterr()
    assert rc == 1


def test_cli_rejects_a_negative_budget(stub):
    stub({A: 40})
    with pytest.raises(SystemExit) as exc:
        bmep.main(["--tickers", A, "--max-tickers", "-1"])
    assert exc.value.code == 2


def test_the_documented_stop_signal_exists_under_the_documented_name():
    """The module docstring tells the operator to re-run until
    `budget.needs_work` is 0. A docstring naming a field the report does
    not have leaves the loop with no stop condition at all."""
    doc = bmep.__doc__ or ""
    assert "needs_work" in doc
    plan = bmep.plan_backfill([], bmep.build_coverage([], {}), max_tickers=5)
    assert "needs_work" in plan.to_dict()
    assert plan.to_dict()["needs_work"] == 0


# ---------------------------------------------------------------------------
# It stays an operator tool
# ---------------------------------------------------------------------------

def test_nothing_in_the_app_imports_the_backfill():
    """An operator tool with a paid provider budget must never be wired
    into a loop, a route or a service — scheduling it would spend money on
    a timer. Only `app/scripts` and `app/tests` may name it."""
    app_dir = Path(bmep.__file__).resolve().parents[1]
    offenders = [
        str(p.relative_to(app_dir)) for p in sorted(app_dir.rglob("*.py"))
        if p.parts[-2:-1] != ("scripts",) and "tests" not in p.parts
        and "backfill_month_end_prices" in p.read_text()
    ]
    assert offenders == []


def test_the_documented_entry_point_parses():
    """`python -m app.scripts.backfill_month_end_prices --help` is the
    documented command; a broken flag definition would only show up here."""
    with pytest.raises(SystemExit) as exc:
        bmep.main(["--help"])
    assert exc.value.code == 0
