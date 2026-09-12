"""`app.scripts.backfill_month_end_prices` — the one-time deep month-end
price backfill that lifts `scorecard_evaluation` out of
`insufficient_data`.

Offline by construction: every provider read goes through a stub swapped
into `data_service.get_data_service`, and the stub COUNTS its calls so the
budget, the dry run and the skip-what-is-done resume rule are asserted on
the thing that actually costs money rather than on the report's own
bookkeeping.

Synthetic tickers (`BMEA`, `BMEB`, …) keep these rows away from the demo
names other suites seed.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from app.database import SessionLocal
from app.models import PriceMonthEnd, ScorecardRun, ScorecardScore
from app.scripts import backfill_month_end_prices as bmep
from app.services import data_service as ds_mod
from app.services import scorecard_evaluation, scorecard_pit, scorecard_queue, scorecard_service

A, B, C, SHORT = "BMEA", "BMEB", "BMEC", "BMES"
ALL = (A, B, C, SHORT)

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
        run_ids = [r[0] for r in db.query(ScorecardRun.id).filter(
            ScorecardRun.requested_by == "test-bmep").all()]
        db.query(ScorecardScore).filter(ScorecardScore.ticker.in_(ALL)).delete(synchronize_session=False)
        if run_ids:
            db.query(ScorecardRun).filter(ScorecardRun.id.in_(run_ids)).delete(synchronize_session=False)
        db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker.in_(ALL)).delete(synchronize_session=False)
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


def _seed_scored_months(ticker: str, months: list[date], *, status: str | None = None) -> None:
    """One succeeded month-end scorecard run per month, with one score row
    — the other half of a panel observation."""
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
            db.add(ScorecardScore(
                run_id=run.id, version_key=scorecard_service.VERSION_KEY, as_of=m, ticker=ticker,
                overall_z=0.1, coverage=0.9, is_month_end=True,
            ))
        db.commit()


# ---------------------------------------------------------------------------
# The coverage arithmetic — pure, no DB, no provider
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
    # 24 evaluable months — the evaluation's floor — needs exactly 37.
    assert len(bmep.evaluable_months(_month_ends(37))) == bmep.MIN_MONTHS
    assert len(bmep.evaluable_months(_month_ends(36))) == bmep.MIN_MONTHS - 1


def test_a_hole_in_the_store_invalidates_every_month_that_needed_it():
    """One missing month end kills four observations (itself, the month
    before, the month after, and the month 12 later), and is reported as a
    gap rather than absorbed into the total."""
    stored = _month_ends(40)
    missing = stored[20]
    holed = [m for m in stored if m != missing]
    assert len(bmep.evaluable_months(stored)) == 27
    assert len(bmep.evaluable_months(holed)) == 23

    cov = bmep.build_coverage([A], {A: set(holed)}, {A: set()})
    row = cov.tickers[0]
    assert row.months_stored == 39
    assert row.gap_months == 1
    assert row.price_evaluable == 23
    assert row.clears_minimum is False
    assert "39 month ends stored (1 missing inside the span)" in (row.reason or "")
    assert "23 evaluable months; 24 required" in (row.reason or "")
    assert "37 contiguous month ends clear it" in (row.reason or "")


def test_an_empty_store_reports_a_reason_not_a_bare_zero():
    cov = bmep.build_coverage([A], {A: set()}, {A: set()})
    row = cov.tickers[0]
    assert (row.months_stored, row.price_evaluable, row.panel_observations) == (0, 0, 0)
    assert row.first_month is None and row.last_month is None
    assert row.reason == "no month ends stored for this ticker"
    d = cov.to_dict()
    assert d["months_stored_min"] == 0  # one ticker, zero months: measured, not unknown
    empty = bmep.build_coverage([], {}, {}).to_dict()
    assert empty["months_stored_min"] is None
    assert empty["months_stored_reason"] == "no tickers in scope"


def test_panel_counts_only_months_that_have_both_a_price_and_a_score():
    """Prices alone draw no verdict. The panel number must be the
    intersection with the month-end score rows, and the price number the
    ceiling — conflating them would overstate the backfill."""
    stored = set(_month_ends(40))
    evaluable = bmep.evaluable_months(stored)
    scored = set(sorted(evaluable)[:10])
    cov = bmep.build_coverage([A], {A: stored}, {A: scored})
    row = cov.tickers[0]
    assert row.price_evaluable == 27 and row.clears_minimum is True
    assert row.panel_observations == 10 and row.panel_clears_minimum is False
    assert cov.price_names_clearing == 1 and cov.panel_names_clearing == 0
    assert cov.price_evaluable == 27 and cov.panel_observations == 10
    assert cov.price_meets_min_obs is False and cov.panel_meets_min_obs is False
    assert cov.scored_months == 10 and cov.scored_rows == 10
    # A scored month with no usable price window is NOT an observation.
    cov2 = bmep.build_coverage([A], {A: stored}, {A: {sorted(stored)[0]}})
    assert cov2.panel_observations == 0


def test_min_obs_is_a_total_across_names_not_a_per_name_floor():
    price = {t: set(_month_ends(40)) for t in (A, B)}
    scored = {t: bmep.evaluable_months(price[t]) for t in (A, B)}
    cov = bmep.build_coverage([A, B], price, scored, min_obs=54)
    assert cov.panel_observations == 54 and cov.panel_meets_min_obs is True
    assert bmep.build_coverage([A, B], price, scored, min_obs=55).panel_meets_min_obs is False


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_plan_skips_finished_names_and_names_what_the_budget_cut():
    price = {A: set(_month_ends(40)), B: set(_month_ends(5)), C: set(), SHORT: set()}
    cov = bmep.build_coverage(list(ALL), price, {t: set() for t in ALL})
    plan = bmep.plan_backfill(list(ALL), cov, max_tickers=1)
    assert plan.satisfied == (A,)          # already clears: no paid call
    assert plan.fetch == (B,)              # the budget's one slot
    assert plan.over_budget == (C, SHORT)  # counted and named, never dropped
    assert plan.to_dict()["n_over_budget"] == 2

    # `--include-satisfied` puts the finished name back in the queue.
    greedy = bmep.plan_backfill(list(ALL), cov, max_tickers=4, include_satisfied=True)
    assert greedy.fetch == (A, B, C, SHORT) and greedy.satisfied == ()
    with pytest.raises(ValueError):
        bmep.plan_backfill(list(ALL), cov, max_tickers=-1)


# ---------------------------------------------------------------------------
# End to end, against the stub provider
# ---------------------------------------------------------------------------

def test_dry_run_fetches_nothing_and_reports_what_it_would_do(stub):
    provider = stub({A: 40, B: 40})
    report = bmep.run(tickers=[A, B], max_tickers=1, dry_run=True)
    assert provider.calls == [], "a dry run must not reach a paid provider"
    with SessionLocal() as db:
        assert db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker.in_(ALL)).count() == 0
    assert report["budget"]["fetch"] == [A]
    assert report["budget"]["over_budget"] == [B]
    # No zero standing in for "not measured".
    assert report["fetched"] is None and "dry run" in report["fetched_reason"]
    assert report["coverage_after"] is None and "unchanged" in report["coverage_after_reason"]
    assert report["delta"] is None and "nothing changed" in report["delta_reason"]
    text = bmep.format_report(report)
    assert "DRY RUN (nothing fetched, nothing written)" in text
    assert "would fetch           BMEA" in text
    assert "NAMES CLEARING THE EVALUATION MINIMUMS: 0 of 2" in text


def test_the_budget_stops_the_run_and_the_next_run_picks_up_where_it_left_off(stub):
    provider = stub({A: 40, B: 40, C: 40})
    first = bmep.run(tickers=[A, B, C], max_tickers=1)
    assert provider.tickers_called == [A]
    assert first["budget"]["n_over_budget"] == 2 and first["budget"]["over_budget"] == [B, C]
    assert first["coverage_after"]["price_names_clearing"] == 1

    second = bmep.run(tickers=[A, B, C], max_tickers=1)
    assert provider.tickers_called == [A, B], "the finished name must not be re-fetched"
    assert second["budget"]["satisfied"] == [A]
    assert second["coverage_after"]["price_names_clearing"] == 2
    assert "skipped for budget    BMEC" in bmep.format_report(second)


def test_rerunning_writes_nothing_and_duplicates_nothing(stub):
    """Idempotency at both levels: the whole ticker is skipped once it
    clears, and forcing the fetch anyway rewrites no row."""
    provider = stub({A: 40})
    first = bmep.run(tickers=[A], max_tickers=5)
    assert first["fetched"]["written"] == 40
    with SessionLocal() as db:
        rows = db.query(PriceMonthEnd).filter(PriceMonthEnd.ticker == A).count()
    assert rows == 40

    again = bmep.run(tickers=[A], max_tickers=5)
    assert again["budget"]["n_fetch"] == 0 and again["budget"]["satisfied"] == [A]
    assert provider.calls == [(A, bmep.DEFAULT_DAYS)]
    assert again["delta"] == {
        "months_stored": 0, "price_evaluable": 0, "price_names_clearing": 0,
        "panel_observations": 0, "panel_names_clearing": 0,
    }

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
    assert "10 month ends stored (0 missing inside the span) yield 0 evaluable months; 24 required" in row["reason"]
    assert "PARTIAL BMES:" in bmep.format_report(report)
    # It is counted in the store, not in the names that clear.
    assert report["coverage_after"]["months_stored"] == 50
    assert report["coverage_after"]["price_names_clearing"] == 1


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


def test_the_printed_coverage_arithmetic_matches_the_stored_rows(stub):
    """The headline number — names clearing the evaluation's minimums — is
    the point of the exercise, so it is checked against the database and
    against an independently computed delta rather than against itself."""
    stub({A: 40, B: 40})
    _seed_scored_months(A, sorted(bmep.evaluable_months(_month_ends(40))))
    _seed_scored_months(B, sorted(bmep.evaluable_months(_month_ends(40)))[:5])

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
    # A has a score row for all 27 evaluable months; B for only 5.
    assert per[A]["panel_observations"] == 27 and per[A]["panel_clears_minimum"] is True
    assert per[B]["panel_observations"] == 5 and per[B]["panel_clears_minimum"] is False
    assert after["price_names_clearing"] == 2 and after["panel_names_clearing"] == 1
    assert after["scored_rows"] == 32 and after["scored_month_ends"] == 27

    assert report["delta"] == {
        "months_stored": after["months_stored"] - before["months_stored"],
        "price_evaluable": after["price_evaluable"] - before["price_evaluable"],
        "price_names_clearing": after["price_names_clearing"] - before["price_names_clearing"],
        "panel_observations": after["panel_observations"] - before["panel_observations"],
        "panel_names_clearing": after["panel_names_clearing"] - before["panel_names_clearing"],
    }
    text = bmep.format_report(report)
    assert "NAMES CLEARING THE EVALUATION MINIMUMS: 1 of 2 (price depth alone would allow 2)" in text
    assert "+80 month ends" in text and "+1 names clearing in the panel" in text


def test_a_failed_scoring_run_contributes_no_panel_months(stub):
    """`build_panel` counts SUCCEEDED month-end runs only; the coverage
    report must apply the same filter or it reports observations the
    evaluation will never see."""
    stub({A: 40})
    _seed_scored_months(A, sorted(bmep.evaluable_months(_month_ends(40))), status=scorecard_queue.STATUS_FAILED)
    report = bmep.run(tickers=[A], max_tickers=5)
    after = report["coverage_after"]
    assert after["price_names_clearing"] == 1
    assert after["panel_observations"] == 0 and after["scored_rows"] == 0


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
