"""FEAT-003 slice 3 — industry-group analytics.

What can rot here: a missing price series scored as a zero return, a
horizon the price window cannot reach reported as a number, a group
below the sample floor producing statistics, a provider fetch per
constituent with no budget, a benchmark quietly absent, and a recompute
over identical inputs producing a different row. Each has a test.

Every read is injected through ``Loaders`` so the arithmetic is checked
on hand-built series; the persistence tests use the real table under the
active taxonomy and clean up after themselves. No test asserts a group
count or a real company's group.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import delete

from app.database import SessionLocal
from app.models import IndustryStatSnapshot
from app.services import gics_registry as reg
from app.services import industry_analytics as ia
from app.tests.gating_helpers import seed_demo_universe

AS_OF = datetime(2026, 9, 4, 21, 0)  # a Friday close
CUTOFF = AS_OF.date()


@pytest.fixture(scope="module", autouse=True)
def _taxonomy():
    seed_demo_universe()
    info = reg.ensure_taxonomy(activate=True)
    assert info is not None
    yield info
    with SessionLocal() as db:
        db.execute(delete(IndustryStatSnapshot).where(
            IndustryStatSnapshot.taxonomy_version_id == info.id,
            IndustryStatSnapshot.period_key.in_(["2026-W36", "2026-W35"]),
            IndustryStatSnapshot.industry_group_code.in_(list(_two_active_groups(info))),
        ))
        db.commit()
    reg.activate_version(info.version_key)


def _two_active_groups(info: reg.VersionInfo) -> tuple[str, str]:
    """Two active groups in different sectors — codes come from the
    registry, never from a literal."""
    groups = reg.industry_groups(version=info)
    first = groups[0]
    other = next(g for g in groups if g.sector_code != first.sector_code)
    return first.code, other.code


@pytest.fixture(scope="module")
def codes(_taxonomy) -> tuple[str, str]:
    return _two_active_groups(_taxonomy)


def _step_series(p_before: float, p_after: float, *, step_days_ago: int = 29, length: int = 120,
                 cutoff: date = CUTOFF) -> list[dict[str, Any]]:
    """Daily rows: ``p_before`` up to and including the 1M anchor (30 days
    before the cutoff), ``p_after`` from ``step_days_ago`` days before it —
    so the 1M return is exactly ``p_after / p_before − 1`` and the 1W
    return is exactly zero."""
    rows = []
    for i in range(length, -1, -1):
        d = cutoff - timedelta(days=i)
        rows.append({"date": d.isoformat(), "close": p_before if i > step_days_ago else p_after})
    return rows


def _loaders(
    *, groups: dict[str, list[str]], prices: dict[str, list[dict[str, Any]]],
    caps: dict[str, float] | None = None, metrics: dict[str, dict[str, Any]] | None = None,
    fetch: dict[str, list[dict[str, Any]] | None] | None = None,
    factor: list[dict[str, Any]] | None = None, prior: dict[str, Any] | None = None,
    spy: list[str] | None = None, inactive: set[str] | None = None,
) -> ia.Loaders:
    caps = caps or {}
    metrics = metrics or {}
    fetch = fetch or {}
    spy = spy if spy is not None else []
    dead = inactive or set()

    def fetch_prices(ticker: str):
        spy.append(ticker)
        return fetch.get(ticker)

    return ia.Loaders(
        constituents_by_group=lambda version: groups,
        companies=lambda tickers: {
            t: {"company_name": f"{t} Co", "sector": "x", "market_cap": caps.get(t),
                "shares_outstanding": None, "last_price": None, "is_active": t not in dead}
            for t in tickers
        },
        metrics=lambda tickers: {t: metrics[t] for t in tickers if t in metrics},
        cached_prices=lambda tickers: {t: prices[t] for t in tickers if t in prices},
        cached_price_tickers=lambda tickers: {t for t in tickers if t in prices},
        fetch_prices=fetch_prices,
        market_factor=lambda: factor,
        prior_stats=lambda code, version, key: prior,
    )


# --- horizons -----------------------------------------------------------------


def test_period_key_is_the_iso_week_of_the_as_of():
    assert ia.period_key_for(AS_OF) == "2026-W36"
    assert ia.period_key_for(date(2026, 1, 2)) == "2026-W01"


def test_horizon_beyond_the_price_window_is_null_with_a_reason():
    rows = _step_series(100.0, 110.0, length=100)
    rets = ia.horizon_returns(ia.close_series(rows), CUTOFF)
    assert rets["1m"]["value"] == pytest.approx(0.10)
    assert rets["1w"]["value"] == pytest.approx(0.0)
    assert rets["qtd"]["value"] is not None
    for h in ("ytd", "1y"):
        assert rets[h]["value"] is None
        assert rets[h]["reason"] == ia.REASON_HISTORY_WINDOW


def test_close_series_prefers_adjusted_close_and_drops_bad_rows():
    series = ia.close_series([
        {"date": "2026-09-01", "close": 10, "adjusted_close": 9.5},
        {"date": "bad", "close": 1},
        {"date": "2026-09-02", "close": None},
        {"date": "2026-09-03", "close": -1},
    ])
    assert series == [(date(2026, 9, 1), 9.5)]


# --- weighting -----------------------------------------------------------------


def test_equal_weight_and_market_cap_weight_on_three_tickers(codes):
    code, _ = codes
    prices = {
        "AAA": _step_series(100.0, 110.0),   # +10%
        "BBB": _step_series(50.0, 50.0),     #  0%
        "CCC": _step_series(20.0, 18.0),     # −10%
    }
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC"]}, prices=prices,
                  caps={"AAA": 1.0, "BBB": 1.0, "CCC": 8.0})
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=0)
    r1m = row.payload["returns"]["1m"]
    assert row.payload["status"] == "ok"
    assert r1m["n"] == 3 and r1m["n_mcw"] == 3
    assert r1m["equal_weight"] == pytest.approx(0.0, abs=1e-6)
    assert r1m["median"] == pytest.approx(0.0, abs=1e-6)
    assert r1m["market_cap_weight"] == pytest.approx((0.10 * 1 + 0.0 * 1 - 0.10 * 8) / 10, abs=1e-6)
    assert row.per_ticker["CCC"]["weight_mcw"] == pytest.approx(0.8)
    assert row.payload["breadth"]["1m"]["pct_positive"] == pytest.approx(1 / 3)
    assert row.payload["dispersion"]["n"] == 3 and row.payload["dispersion"]["range"] == pytest.approx(0.2)
    # Three names split one/one: listing all three as leaders AND CCC as a
    # laggard (the old behaviour) said the same company was both.
    assert [x["ticker"] for x in row.payload["leaders"]] == ["AAA"]
    assert [x["ticker"] for x in row.payload["laggards"]] == ["CCC"]
    assert row.method["weighting"] == ["equal", "market_cap"]


def test_market_cap_weight_without_caps_is_null_with_reason_not_equal_weight(codes):
    code, _ = codes
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC"]},
                  prices={t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")})
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=0)
    r1m = row.payload["returns"]["1m"]
    assert r1m["equal_weight"] == pytest.approx(0.10)
    assert r1m["market_cap_weight"] is None and r1m["n_mcw"] == 0
    assert r1m["market_cap_weight_reason"] == "no_market_caps"


# --- missing data ----------------------------------------------------------------


def test_missing_prices_exclude_the_ticker_with_a_reason_never_a_zero(codes):
    code, _ = codes
    spy: list[str] = []
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC", "DDD"]},
                  prices={t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")}, spy=spy)
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=10)
    assert spy == ["DDD"]  # the provider was asked once, within budget, and had nothing
    assert row.sample["n_constituents"] == 4 and row.sample["n_with_prices"] == 3
    assert row.sample["excluded"] == [{"ticker": "DDD", "reason": ia.REASON_NO_PRICES}]
    assert row.payload["returns"]["1m"]["n"] == 3
    assert row.payload["returns"]["1m"]["equal_weight"] == pytest.approx(0.10)  # not dragged toward zero
    assert row.per_ticker["DDD"]["returns"] == {} and row.per_ticker["DDD"]["exclusion"] == ia.REASON_NO_PRICES
    assert "DDD" not in {x["ticker"] for x in row.payload["leaders"] + row.payload["laggards"]}


def test_stale_series_is_excluded_not_stretched_to_the_as_of(codes):
    code, _ = codes
    stale = _step_series(10.0, 11.0, cutoff=CUTOFF - timedelta(days=ia.STALE_PRICE_DAYS + 5))
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC", "OLD"]},
                  prices={**{t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")}, "OLD": stale})
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=0)
    assert {"ticker": "OLD", "reason": ia.REASON_STALE_PRICES} in row.sample["excluded"]
    assert row.sample["n_with_prices"] == 3


def test_below_the_sample_floor_is_a_labelled_state_not_numbers(codes):
    code, _ = codes
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC"]},
                  prices={"AAA": _step_series(10.0, 11.0), "BBB": _step_series(10.0, 12.0)})
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=10, min_sample=3)
    assert row.payload["status"] == ia.REASON_INSUFFICIENT
    assert row.payload["insufficient_sample"] == {
        "n_with_prices": 2, "min_sample": 3, "reasons": [ia.REASON_NO_PRICES],
    }
    for h in ia.HORIZONS:
        entry = row.payload["returns"][h]
        assert entry["equal_weight"] is None and entry["market_cap_weight"] is None
        assert entry["reason"] == ia.REASON_INSUFFICIENT
    assert row.payload["leaders"] == [] and row.payload["laggards"] == []
    assert row.payload["breadth"]["1m"]["pct_positive"] is None
    # The per-ticker rows are still kept — the closes are evidence for the
    # next period even when this one cannot aggregate them.
    assert row.per_ticker["AAA"]["weekly_closes"]


# --- benchmarks -------------------------------------------------------------------


def test_missing_market_factor_is_a_degraded_entry_and_other_benchmarks_still_report(codes):
    code, other = codes
    prices = {
        "AAA": _step_series(10.0, 11.0), "BBB": _step_series(10.0, 12.0), "CCC": _step_series(10.0, 13.0),
        "ZZZ": _step_series(10.0, 9.0),
    }
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC"], other: ["ZZZ"]}, prices=prices, factor=None)
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=0)
    assert f"benchmark:{ia.MARKET_FACTOR_ID}:missing" in row.payload["degraded"]
    universe = row.payload["benchmarks"][ia.BENCHMARK_UNIVERSE]["1m"]
    assert universe["n"] == 4 and universe["value"] == pytest.approx((0.1 + 0.2 + 0.3 - 0.1) / 4, abs=1e-6)
    sector = row.payload["benchmarks"][ia.BENCHMARK_SECTOR]["1m"]
    assert sector["n"] == 3 and sector["value"] == pytest.approx(0.2, abs=1e-6)
    rel = row.payload["benchmark_relative"]
    assert rel[ia.BENCHMARK_UNIVERSE]["1m"]["value"] == pytest.approx(0.2 - 0.125, abs=1e-6)
    assert rel[ia.BENCHMARK_SECTOR]["1m"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert rel[ia.MARKET_FACTOR_ID]["1m"]["value"] is None and rel[ia.MARKET_FACTOR_ID]["1m"]["reason"] == "missing"
    by_id = {b["id"]: b for b in row.method["benchmarks"]}
    assert by_id[ia.MARKET_FACTOR_ID]["available"] is False and by_id[ia.BENCHMARK_UNIVERSE]["available"] is True
    assert "EXCESS return" in by_id[ia.MARKET_FACTOR_ID]["definition"]


def test_market_factor_cumulates_daily_excess_returns_over_the_window(codes):
    code, _ = codes
    points = [{"date": (CUTOFF - timedelta(days=i)).isoformat(), "value": 0.01} for i in range(120, -1, -1)]
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC"]},
                  prices={t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")}, factor=points)
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=0)
    factor = row.payload["benchmarks"][ia.MARKET_FACTOR_ID]
    assert factor["1w"]["n_days"] == 7 and factor["1w"]["value"] == pytest.approx(1.01 ** 7 - 1, abs=1e-6)
    assert factor["1m"]["n_days"] == 30
    assert factor["1y"]["value"] is None and factor["1y"]["reason"] == ia.REASON_HISTORY_WINDOW
    assert row.payload["degraded"] == []
    assert row.payload["benchmark_relative"][ia.MARKET_FACTOR_ID]["1w"]["value"] == pytest.approx(0.0 - (1.01 ** 7 - 1), abs=1e-6)


def test_the_benchmark_cohort_applies_the_same_exclusions_the_group_applies(codes):
    """A stale series and a delisted company are thrown out of the group's
    numbers. If they stay in the universe/sector cohort, the group is
    measured against the very rows it ruled unusable and
    ``benchmark_relative`` — and the regime label the PM reads off it —
    reports leadership that is not there."""
    code, _ = codes
    stale = _step_series(20.0, 10.0, cutoff=CUTOFF - timedelta(days=ia.STALE_PRICE_DAYS + 5))
    ld = _loaders(
        groups={code: ["AAA", "BBB", "CCC", "DEAD", "OLD"]},
        prices={**{t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")},
                "DEAD": _step_series(20.0, 10.0), "OLD": stale},
        inactive={"DEAD"},
    )
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=0)
    assert row.sample["excluded"] == [
        {"ticker": "DEAD", "reason": ia.REASON_INACTIVE}, {"ticker": "OLD", "reason": ia.REASON_STALE_PRICES},
    ]
    assert row.payload["returns"]["1m"]["equal_weight"] == pytest.approx(0.10)
    # The cohort is the same three tickers, so like-for-like relative is 0.
    for bid in (ia.BENCHMARK_UNIVERSE, ia.BENCHMARK_SECTOR):
        bench = row.payload["benchmarks"][bid]["1m"]
        assert bench["n"] == 3 and bench["value"] == pytest.approx(0.10)
        assert row.payload["benchmark_relative"][bid]["1m"]["value"] == pytest.approx(0.0, abs=1e-9)
    by_id = {b["id"]: b for b in row.method["benchmarks"]}
    assert by_id[ia.BENCHMARK_UNIVERSE]["excluded_by_reason"] == {
        ia.REASON_INACTIVE: 1, ia.REASON_STALE_PRICES: 1,
    }
    cohort = row.sample["benchmark_cohort"]
    assert cohort == {
        "n_universe": 5, "n_eligible": 3,
        "excluded_by_reason": {ia.REASON_INACTIVE: 1, ia.REASON_STALE_PRICES: 1},
        "group_members_outside_cohort": [],
    }
    assert "fixed before any group is computed" in row.method["benchmark_cohort_basis"]


def test_a_groups_row_does_not_depend_on_which_group_was_drained_first(codes):
    """Two groups share one context (the pattern the weekly drainer uses).
    ``ensure_prices`` grows ``ctx.prices`` per group, so a cohort read off
    ``ctx.prices`` would give the first group a 3-ticker universe and the
    second a 6-ticker one — same inputs, two different benchmarks, two
    different ``inputs_hash`` values, and a regime label that flips with
    drain order."""
    code, other = codes
    cached = {t: _step_series(10.0, 11.0) for t in ("A1", "A2", "A3")}      # +10% 1M, in the cache
    fetched = {t: _step_series(20.0, 10.0) for t in ("B1", "B2", "B3")}     # −50% 1M, fetched on demand
    groups = {code: sorted(cached), other: sorted(fetched)}

    def run(order: list[str]) -> dict[str, Any]:
        ld = _loaders(groups=groups, prices=cached, fetch=fetched)
        ctx = ia.load_context(AS_OF, version=None, max_fetch=10, loaders=ld)
        rows = {c: ia.compute_group_stats(c, as_of=AS_OF, context=ctx, persist=False) for c in order}
        return {c: rows[c] for c in sorted(order)}

    forward = run([code, other])
    backward = run([other, code])
    for c in (code, other):
        assert forward[c].inputs_hash == backward[c].inputs_hash
        assert forward[c].payload == backward[c].payload
    # And the cohort is the baseline three, in both orders — the on-demand
    # fetches count in their own group but are not retro-fitted into it.
    universe = forward[code].payload["benchmarks"][ia.BENCHMARK_UNIVERSE]["1m"]
    assert universe["n"] == 3 and universe["value"] == pytest.approx(0.10)
    assert forward[code].payload["benchmark_relative"][ia.BENCHMARK_UNIVERSE]["1m"]["value"] == pytest.approx(0.0, abs=1e-9)
    outside = forward[other].sample["benchmark_cohort"]["group_members_outside_cohort"]
    assert outside == [{"ticker": t, "reason": ia.REASON_NO_PRICES} for t in ("B1", "B2", "B3")]


# --- fetch budget -------------------------------------------------------------------


def test_fetch_budget_is_respected_and_skipped_tickers_are_reported(codes):
    code, _ = codes
    spy: list[str] = []
    fetch = {t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC", "DDD")}
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC", "DDD"]}, prices={}, fetch=fetch, spy=spy)
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=2)
    assert spy == ["AAA", "BBB"]
    assert row.sample["fetches_this_run"] == 2 and row.sample["fetch_budget"] == 2
    assert row.sample["excluded"] == [
        {"ticker": "CCC", "reason": ia.REASON_FETCH_BUDGET}, {"ticker": "DDD", "reason": ia.REASON_FETCH_BUDGET},
    ]
    assert row.sample["price_sources"] == {"fetch": 2}


def test_cached_series_mean_zero_provider_fetches(codes):
    code, _ = codes
    spy: list[str] = []
    prices = {t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")}
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC"]}, prices=prices, fetch=prices, spy=spy)
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=50)
    assert spy == [] and row.sample["price_sources"] == {"cache": 3}


def test_stored_weekly_closes_place_a_ticker_when_the_budget_is_gone(codes):
    code, _ = codes
    stored = ia.weekly_closes(ia.close_series(_step_series(10.0, 15.0)))
    prior = {"period_key": "2026-W35", "payload": {"fundamentals": {"revenue_growth_yoy": {"median": 0.05}}},
             "per_ticker": {"DDD": {"weekly_closes": stored}}}
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC", "DDD"]},
                  prices={t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")}, prior=prior)
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=0)
    assert row.per_ticker["DDD"]["price_source"] == "stored_weekly_closes"
    assert row.per_ticker["DDD"]["returns"]["1m"] == pytest.approx(0.5)
    assert row.sample["n_with_prices"] == 4 and row.sample["excluded"] == []


# --- fundamentals ----------------------------------------------------------------------


def test_valuation_medians_exclude_non_positive_multiples_and_momentum_needs_a_prior(codes):
    code, _ = codes
    metrics = {
        "AAA": {"ev_ebitda": 10.0, "pe_ttm": -5.0, "revenue_growth_yoy": 0.10, "last_updated": "2026-09-01T00:00:00"},
        "BBB": {"ev_ebitda": 20.0, "pe_ttm": 15.0, "revenue_growth_yoy": 0.20, "last_updated": "2026-09-01T00:00:00"},
        "CCC": {"ev_ebitda": None, "pe_ttm": 25.0, "revenue_growth_yoy": None, "last_updated": "2026-09-01T00:00:00"},
    }
    base = dict(groups={code: ["AAA", "BBB", "CCC"]},
                prices={t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")}, metrics=metrics)
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=_loaders(**base), persist=False, max_fetch=0)
    val = row.payload["valuation"]
    assert val["ev_ebitda"]["median"] == pytest.approx(15.0) and val["ev_ebitda"]["n"] == 2
    assert val["pe_ttm"]["n"] == 2 and val["pe_ttm"]["n_excluded_nonpositive"] == 1
    assert val["ev_revenue"]["median"] is None and val["ev_revenue"]["reason"] == "no_metric_values"
    assert row.payload["fundamentals"]["revenue_growth_yoy"]["median"] == pytest.approx(0.15)
    assert row.payload["fundamental_momentum"]["value"] is None
    assert row.payload["fundamental_momentum"]["reason"] == "no_prior_period"
    prior = {"period_key": "2026-W35", "payload": {"fundamentals": {"revenue_growth_yoy": {"median": 0.10}}}, "per_ticker": {}}
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=_loaders(**base, prior=prior), persist=False, max_fetch=0)
    assert row.payload["fundamental_momentum"]["value"] == pytest.approx(0.05)
    assert row.payload["fundamental_momentum"]["prior_period_key"] == "2026-W35"
    assert "proxy" in row.payload["fundamental_momentum"]["basis"]


# --- breadth window ----------------------------------------------------------------------


def _weekday_series(cutoff: date = CUTOFF, *, length: int = 200) -> list[dict[str, Any]]:
    """Weekday-only bars: 100 until 55 calendar days before the cutoff, 80
    for the next 30, 82 for the last 25. Over the last 50 TRADING sessions
    the mean sits above 82; over the last 50 CALENDAR days (≈36 bars) it
    sits below. The two windows disagree about the same series, which is
    the whole point."""
    rows = []
    for i in range(length, -1, -1):
        d = cutoff - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        close = 100.0 if i > 55 else (80.0 if i > 25 else 82.0)
        rows.append({"date": d.isoformat(), "close": close})
    return rows


def test_above_50d_mean_counts_trading_sessions_not_calendar_days():
    """The field is named for a 50-day moving average and a report will
    quote it as one. A 50-calendar-day window is about 35 bars and answers
    a different question — here it flips the flag from below to above."""
    series = ia.close_series(_weekday_series())
    sessions = [px for _, px in series][-ia.MEAN_WINDOW_SESSIONS:]
    assert len(sessions) == ia.MEAN_WINDOW_SESSIONS
    assert sessions[-1] < sum(sessions) / len(sessions)  # genuinely below its 50-session mean
    flag, why = ia.above_mean_window(series, CUTOFF)
    assert flag is False and why is None
    # The window this replaced: fewer bars, and the opposite answer.
    calendar = [px for d, px in series if CUTOFF - timedelta(days=50) < d <= CUTOFF]
    assert len(calendar) < ia.MEAN_WINDOW_SESSIONS
    assert calendar[-1] > sum(calendar) / len(calendar)


def test_a_series_that_cannot_support_the_50_session_mean_is_null_with_a_reason():
    short = ia.close_series(_step_series(10.0, 11.0, length=40))
    assert ia.above_mean_window(short, CUTOFF) == (None, ia.REASON_MEAN_WINDOW_SHORT)
    # 60 weekly closes are 60 bars but reach back more than a year; calling
    # their mean a "50-day mean" would be a factual error.
    weekly = ia.close_series([
        {"date": (CUTOFF - timedelta(days=7 * i)).isoformat(), "close": 10.0 + i} for i in range(59, -1, -1)
    ])
    assert ia.above_mean_window(weekly, CUTOFF) == (None, ia.REASON_MEAN_WINDOW_SPARSE)


def test_breadth_reports_the_window_it_used_and_names_who_it_could_not_place(codes):
    code, _ = codes
    prices = {t: _weekday_series() for t in ("AAA", "BBB", "CCC")}
    prices["SHORT"] = _step_series(10.0, 11.0, length=30)  # too few bars for a 50-session mean
    ld = _loaders(groups={code: sorted(prices)}, prices=prices)
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=0)
    above = row.payload["breadth"]["above_50d_mean"]
    assert above["share"] == 0.0 and above["n"] == 3
    assert above["window_sessions"] == ia.MEAN_WINDOW_SESSIONS
    assert above["excluded_by_reason"] == {ia.REASON_MEAN_WINDOW_SHORT: 1}
    assert row.per_ticker["SHORT"]["above_50d_mean"] is None
    assert row.per_ticker["SHORT"]["above_50d_mean_reason"] == ia.REASON_MEAN_WINDOW_SHORT
    window = row.method["breadth_mean_window"]
    assert window["sessions"] == 50 and "not calendar days" in window["basis"]


# --- determinism and persistence ------------------------------------------------------------


def test_same_inputs_give_the_same_hash_payload_and_row(_taxonomy, codes):
    code, _ = codes
    prices = {"AAA": _step_series(10.0, 11.0), "BBB": _step_series(10.0, 12.0), "CCC": _step_series(10.0, 13.0)}
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC"]}, prices=prices, caps={"AAA": 1, "BBB": 2, "CCC": 3})
    first = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, max_fetch=0)
    second = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, max_fetch=0)
    assert first.id is not None and second.id == first.id
    assert second.inputs_hash == first.inputs_hash
    assert second.payload == first.payload and second.per_ticker == first.per_ticker
    assert ia.latest_stats(code, version=_taxonomy)["id"] == first.id
    assert ia.stats_for_period(code, "2026-W36", version=_taxonomy)["inputs_hash"] == first.inputs_hash
    assert code in ia.period_stats("2026-W36", version=_taxonomy)

    changed = dict(prices)
    changed["CCC"] = _step_series(10.0, 14.0)
    third = ia.compute_group_stats(
        code, as_of=AS_OF, max_fetch=0,
        loaders=_loaders(groups={code: ["AAA", "BBB", "CCC"]}, prices=changed, caps={"AAA": 1, "BBB": 2, "CCC": 3}),
    )
    assert third.id != first.id and third.inputs_hash != first.inputs_hash
    assert "compute_ms" not in third.payload  # nothing time-dependent lives in the payload


def test_a_revision_the_endpoints_hide_still_changes_the_hash(_taxonomy, codes):
    """A provider revision that leaves the last close and every horizon
    anchor untouched still moves the payload (the 50-day breadth flag and
    the weekly closes a later period reuses). If the hash misses it,
    ``_persist`` hands back the stored row and the recompute is silently
    discarded — a stale row wearing a fresh timestamp.
    """
    code, _ = codes
    base = {t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")}
    revised = {t: [dict(r) for r in rows] for t, rows in base.items()}
    mid = (CUTOFF - timedelta(days=20)).isoformat()
    for row in revised["AAA"]:
        if row["date"] == mid:
            row["close"] = 33.0  # a back-fill inside the 50-day window
    first = ia.compute_group_stats(code, as_of=AS_OF, period_key="2026-W35", max_fetch=0,
                                   loaders=_loaders(groups={code: sorted(base)}, prices=base))
    second = ia.compute_group_stats(code, as_of=AS_OF, period_key="2026-W35", max_fetch=0,
                                    loaders=_loaders(groups={code: sorted(revised)}, prices=revised))
    assert first.per_ticker["AAA"]["last_close"] == second.per_ticker["AAA"]["last_close"]
    assert first.per_ticker["AAA"]["returns"] == second.per_ticker["AAA"]["returns"]
    assert second.inputs_hash != first.inputs_hash and second.id != first.id
    assert second.payload["breadth"]["above_50d_mean"] != first.payload["breadth"]["above_50d_mean"]


def test_metric_values_are_part_of_the_identity_even_when_the_timestamp_is_not(_taxonomy, codes):
    """Valuation medians come from ``screener_metrics``; a refreshed value
    with no ``last_updated`` must not collide with the stored row."""
    code, _ = codes
    prices = {t: _step_series(10.0, 11.0) for t in ("AAA", "BBB", "CCC")}
    cheap = {t: {"ev_ebitda": 10.0, "last_updated": None} for t in prices}
    rich = {**cheap, "AAA": {"ev_ebitda": 30.0, "last_updated": None}, "BBB": {"ev_ebitda": 30.0, "last_updated": None}}
    first = ia.compute_group_stats(code, as_of=AS_OF, period_key="2026-W35", max_fetch=0,
                                   loaders=_loaders(groups={code: sorted(prices)}, prices=prices, metrics=cheap))
    second = ia.compute_group_stats(code, as_of=AS_OF, period_key="2026-W35", max_fetch=0,
                                    loaders=_loaders(groups={code: sorted(prices)}, prices=prices, metrics=rich))
    assert second.inputs_hash != first.inputs_hash
    assert second.payload["valuation"]["ev_ebitda"]["median"] != first.payload["valuation"]["ev_ebitda"]["median"]


def test_unknown_group_code_raises_before_any_read():
    with pytest.raises(reg.UnknownNode):
        ia.compute_group_stats("0000", as_of=AS_OF, loaders=_loaders(groups={}, prices={}), persist=False, max_fetch=0)


# --- warm-up ------------------------------------------------------------------------------


def test_warm_up_fetches_lowest_coverage_groups_first_within_the_budget(_taxonomy, codes):
    code, other = codes
    low, high = sorted([code, other])
    spy: list[str] = []
    series = _step_series(10.0, 11.0)
    cached = {"H1": series, "H2": series}  # `high` has 2 of 3 cached; `low` has none
    fetch = {t: series for t in ("L1", "L2", "L3", "H3")}
    ld = _loaders(groups={high: ["H1", "H2", "H3"], low: ["L1", "L2", "L3"]}, prices=cached, fetch=fetch, spy=spy)
    out = ia.warm_up_prices(budget=2, version=_taxonomy, loaders=ld)
    assert spy == ["L1", "L2"]
    assert out["fetched"] == 2 and out["failed"] == [] and out["remaining_missing"] == 2
    assert out["groups_touched"] == [low]
    assert out["coverage_before"] == {high: pytest.approx(2 / 3, abs=1e-3), low: 0.0}
    assert out["coverage_after"][low] == pytest.approx(2 / 3, abs=1e-3)
    assert out["coverage_after"][high] == pytest.approx(2 / 3, abs=1e-3)


def test_warm_up_reports_failed_fetches_and_stops_at_the_budget(_taxonomy, codes):
    code, _ = codes
    spy: list[str] = []
    ld = _loaders(groups={code: ["A1", "A2", "A3"]}, prices={}, fetch={"A2": _step_series(10.0, 11.0)}, spy=spy)
    out = ia.warm_up_prices(budget=10, version=_taxonomy, loaders=ld)
    assert spy == ["A1", "A2", "A3"]
    assert out["fetched"] == 1 and out["failed"] == ["A1", "A3"] and out["remaining_missing"] == 2


# ---------------------------------------------------------------------------
# Review low findings
# ---------------------------------------------------------------------------

def test_leaders_and_laggards_never_name_the_same_company(codes):
    """Below 2 x LEADER_COUNT priced names the two sections used to be the
    top and bottom of one short ranking, so they overlapped."""
    code, _ = codes
    prices = {t: _step_series(100.0, 100.0 + i) for i, t in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"])}
    ld = _loaders(groups={code: ["AAA", "BBB", "CCC", "DDD", "EEE"]}, prices=prices)
    row = ia.compute_group_stats(code, as_of=AS_OF, loaders=ld, persist=False, max_fetch=0)
    leaders = {x["ticker"] for x in row.payload["leaders"]}
    laggards = {x["ticker"] for x in row.payload["laggards"]}
    assert leaders and laggards
    assert leaders & laggards == set(), "a company cannot lead and lag its own group"


def test_weekly_closes_never_run_past_the_as_of():
    """A point-in-time row must not carry closes dated after its as-of —
    a later period reuses these, so the future would leak backwards."""
    from datetime import date as _date

    from app.services.industry_analytics import weekly_closes
    series = [(_date(2026, 8, 24), 10.0), (_date(2026, 9, 4), 11.0), (_date(2026, 9, 11), 12.0)]
    cutoff = _date(2026, 9, 4)
    rows = weekly_closes(series, cutoff=cutoff)
    assert rows, "expected the weeks on or before the cutoff"
    assert all(r[0] <= cutoff.isoformat() for r in rows), rows
    assert not any(r[1] == 12.0 for r in rows), "the post-cutoff close must be dropped"
    # Without a cutoff the helper is unchanged.
    assert any(r[1] == 12.0 for r in weekly_closes(series))
