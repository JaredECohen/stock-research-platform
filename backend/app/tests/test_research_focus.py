"""The monitoring loops must spend their budget on researched tickers.

Both `news_loop` and `social_loop` used to pick their work with

    tickers = list(ds.list_tickers())[:10]   # demo universe sample

and `DataService.list_tickers()` has no ORDER BY. *Which* ten that returned
is not a fact the source determines — it is whatever the engine handed
back. What can be checked, and what was checked by reading
`app/data/sp500.json` and `app/seed_universe.py` rather than by observing
production, is the list being sliced: `tickers` holds 170 symbols in two
alphabetically sorted runs (the SP100 block AAPL..XOM, then the curated
extensions ADI..ZTS), and `seed_universe._seed` inserts them in file order,
so a table filled by that seeder has AAPL, ABBV, ABT, ACN, ADBE, AIG, AMD,
AMGN, AMT, AMZN at its head. Of the 172 companies in the universe only 17
have ever had a memo generated and only 10 carry the `auto_update_memo`
pin; that file head overlaps the 17 by four (AAPL, ABBV, ADBE, AMZN) and
the pins by two (AAPL, AMZN). NVDA has 230 memo versions and MSFT 108, and
neither was being monitored.

The failure mode is quiet in the worst way: the loop reports success, the
cache fills with fresh news, cron-health is green, and the news is about
companies nobody is researching. Nothing 500s. Nothing is logged. The only
symptom is a memo that never mentions the thing that moved the stock.

So these tests assert the selection *policy*, not the plumbing: the rank
order is a pure function of the inputs, a live regen request always gets a
slot, no ticker is ever permanently excluded at any caller cadence, a
ticker the loop could not act on does not burn a slot but comes straight
back when it can, an over-budget run names every ticker it dropped or held
back, and a sick database degrades to the ten biggest companies instead of
to an alphabetical accident. The last tests are the regression guard
proper: they re-read both loop modules and fail if the arbitrary slice, or
a scheduler interval the rotation does not know about, ever comes back.
"""
from __future__ import annotations

import inspect
import io
import re
import tokenize
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Company, MemoSnapshot, RegenJob
from app.monitoring import news_loop, social_loop
from app.monitoring.research_focus import (
    FocusSelection,
    BAND_DEGRADED,
    BAND_MEMO_FRESH,
    BAND_MEMO_STALE,
    BAND_PINNED,
    BAND_REGEN,
    LIVE_RESEARCH_RESERVE,
    ROTATING_SLOTS,
    WITHHELD_PINNED_NO_MEMO,
    WITHHELD_REGEN_NO_MEMO,
    select_focus,
)

# A fixed clock. Everything below is expressed relative to it, so no test
# depends on the day it runs.
NOW = datetime(2026, 9, 12, 12, 0, 0)

# The curated pin list in `app/data/sp500.json`, in file order (market cap
# descending) — what the degraded path is expected to fall back to.
STATIC_PINS = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK.B", "TSLA", "AVGO", "LLY",
)

# The five pins that carry a memo in production, and the five that do not.
PINS_WITH_MEMO = ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN")
PINS_WITHOUT_MEMO = ("META", "BRK.B", "TSLA", "AVGO", "LLY")

# Every ticker that has ever had a memo generated (17 of 172).
MEMO_CARRIERS = (
    "NVDA", "MSFT", "ADBE", "ABBV", "GOOGL", "AAPL", "XOM", "V", "COST",
    "TSM", "BAC", "MELI", "MO", "BK", "AMZN", "GOOG", "CAT",
)

# Stand-in for a real selection where a test only cares about the kwargs
# `run_once` passed in, not about what came back.
_EMPTY = FocusSelection()


def _capture_kwargs(monkeypatch, module, sink: dict) -> None:
    """Record the kwargs `module.run_once` hands to `select_focus`."""
    def fake_select_focus(**kwargs):
        sink.update(kwargs)
        return _EMPTY

    monkeypatch.setattr(module, "select_focus", fake_select_focus)
    monkeypatch.setattr(module, "record_run", lambda *a, **k: None)


@pytest.fixture()
def db():
    """A private in-memory database holding only the three tables we rank on.

    Deliberately NOT the session-wide sqlite file: the selector reads every
    company and every memo, so a shared database would make each assertion
    depend on whatever the rest of the suite happened to have seeded. This
    is exactly why `select_focus` takes an injectable `db`.
    """
    engine = create_engine("sqlite://", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(
        engine,
        tables=[Company.__table__, MemoSnapshot.__table__, RegenJob.__table__],
    )
    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as session:
        yield session
    engine.dispose()


def add_company(db, ticker: str, *, pinned: bool = False) -> None:
    if db.get(Company, ticker) is not None:
        return
    db.add(Company(
        ticker=ticker, company_name=f"{ticker} Inc.",
        sector="Technology", industry="Software",
        auto_update_memo=pinned,
    ))
    db.flush()


def add_memo(db, ticker: str, *, age_days: float, version: int = 1, backtest: bool = False) -> None:
    db.add(MemoSnapshot(
        ticker=ticker, version=version,
        generated_at=NOW - timedelta(days=age_days),
        as_of_date=(NOW - timedelta(days=400)) if backtest else None,
    ))
    db.flush()


def add_regen(db, ticker: str, *, age_days: float, create_company: bool = False) -> None:
    if create_company:
        add_company(db, ticker)
    db.add(RegenJob(
        ticker=ticker, run_id=f"run-{ticker}-{age_days}", status="succeeded",
        enqueued_at=NOW - timedelta(days=age_days),
    ))
    db.flush()


# ---------------------------------------------------------------------------
# Band 0 — the pins rank first, and none of them starves
# ---------------------------------------------------------------------------

def test_pins_outrank_recent_research_when_the_budget_can_hold_everyone(db):
    """`auto_update_memo` is a promise, and half the pins have no memo yet.

    Five of the ten production pins (META, BRK.B, TSLA, AVGO, LLY) have
    never had a memo generated. A selector that ranked on memo evidence
    alone would drop exactly the names the owner curated by hand, so with
    room for everyone the pins still come first.
    """
    for ticker in STATIC_PINS:
        add_company(db, ticker, pinned=True)
    # Plenty of competition with real, recent research behind it.
    competitors = ("COST", "TSM", "BAC", "XOM", "V")
    for ticker in competitors:
        add_company(db, ticker)
        add_memo(db, ticker, age_days=1)

    sel = select_focus(budget=15, now=NOW, db=db)

    assert set(sel.tickers) == set(STATIC_PINS) | set(competitors)
    assert set(sel.tickers[:10]) == set(STATIC_PINS)
    assert {sel.reasons[t] for t in STATIC_PINS} == {BAND_PINNED}
    assert sel.dropped == ()
    assert sel.degraded is False


def test_no_pin_is_starved_when_the_pin_list_is_bigger_than_the_budget(db):
    """Adding an 11th pin must not permanently un-monitor an existing one.

    `POST /api/admin/.../set_auto_update_memo` lets an operator pin any
    number of tickers. Truncating the pin list in a fixed order — which is
    what a rank-order cut with no rotation does — means the pin that sorts
    last is dropped on every run, forever, chosen by nothing more than its
    name. Pin `budget + 1` tickers and every one of them must run at least
    once inside `budget + 1` consecutive runs.
    """
    budget = 10
    pins = [*STATIC_PINS, "ORCL"]
    for ticker in pins:
        add_company(db, ticker, pinned=True)

    covered: set[str] = set()
    for hour in range(budget + 1):
        sel = select_focus(budget=budget, now=NOW + timedelta(hours=hour), db=db)
        assert len(sel.tickers) == budget
        # Whatever is not selected is named, every single run.
        assert set(sel.tickers) | set(sel.dropped) == set(pins)
        covered.update(sel.tickers)

    assert covered == set(pins), (
        "these pins are starved — adding one pin silently un-monitored them: "
        f"{sorted(set(pins) - covered)}"
    )


def test_pins_are_ranked_by_research_activity_not_by_name(db):
    """The degenerate case: pin everything that has ever had a memo.

    22 pins at a budget of 10 cut alphabetically gives AAPL, ABBV, ADBE,
    AMZN, AVGO, BAC, BK, BRK.B, CAT, COST — dropping NVDA (230 memo
    versions) and MSFT (108) and reproducing exactly the alphabetical head
    this module exists to remove. Ranking band 0 by research activity puts
    the busiest names in the guaranteed prefix instead.
    """
    pins = list(dict.fromkeys([*STATIC_PINS, *MEMO_CARRIERS]))
    for ticker in pins:
        add_company(db, ticker, pinned=True)
    # NVDA and MSFT are the two most actively researched names.
    for age, ticker in enumerate(("NVDA", "MSFT", "ADBE", "ABBV", "GOOGL")):
        add_memo(db, ticker, age_days=1 + age)

    sel = select_focus(budget=10, now=NOW, db=db)

    assert sel.tickers[:2] == ("NVDA", "MSFT"), (
        f"band 0 fell back to an alphabetical cut: {sel.tickers}"
    )
    # And still nobody starves.
    covered: set[str] = set()
    for hour in range(len(pins)):
        covered.update(select_focus(budget=10, now=NOW + timedelta(hours=hour), db=db).tickers)
    assert covered == set(pins)


# ---------------------------------------------------------------------------
# Band 1 — "being researched right now"
# ---------------------------------------------------------------------------

def test_a_recent_regen_request_outranks_a_ticker_whose_memo_is_stale(db):
    """A manual regen is the truest signal, and it reaches outside the universe.

    MELI and SHOP are `analyzed_on_demand` — they are not in the curated
    S&P list and they are not pinned, so the only evidence that anyone
    cares about them is that a user asked for a memo.
    """
    add_company(db, "MELI")
    add_regen(db, "MELI", age_days=0.5)
    add_company(db, "OLD")
    add_memo(db, "OLD", age_days=200)

    sel = select_focus(budget=1, now=NOW, db=db)

    assert sel.tickers == ("MELI",)
    assert sel.reasons["MELI"] == BAND_REGEN
    assert sel.dropped == ("OLD",)


def test_the_production_shape_still_reaches_the_ticker_a_user_just_asked_about(db):
    """The trace that proves the reserve works, in the exact shipped shape.

    Ten pins, a budget of ten: with a plain rank-order cut the pins fill
    the run every hour and band 1 is unreachable, so the user reading MELI
    right now gets no news coverage, ever, and the rotation is dead code.
    Here MELI is covered on every run, five pins the news loop could not
    have acted on are withheld rather than burning a slot, and the rotation
    actually turns.
    """
    for ticker in STATIC_PINS:
        add_company(db, ticker, pinned=True)
    for ticker in MEMO_CARRIERS:
        add_company(db, ticker)
        add_memo(db, ticker, age_days=45)
    add_regen(db, "MELI", age_days=5 / (24 * 60))       # enqueued 5 minutes ago

    runs = [
        select_focus(budget=10, now=NOW + timedelta(hours=h), db=db, require_memo=True)
        for h in range(24)
    ]

    for sel in runs:
        assert "MELI" in sel.tickers, f"the on-demand ticker was dropped: {sel.note()}"
        assert sel.reasons["MELI"] == BAND_REGEN
        assert len(sel.tickers) == 10
    # The rotation branch is reachable, not dead code.
    assert runs[0].rotation_slots == ROTATING_SLOTS
    assert runs[0].rotation_pool_size > 0
    assert len({r.tickers for r in runs}) > 1, "the selection is as static as the slice it replaced"
    # The five memo-less pins are held back, by name, with their own label.
    assert set(runs[0].withheld) == set(PINS_WITHOUT_MEMO)
    assert set(runs[0].withheld.values()) == {WITHHELD_PINNED_NO_MEMO}
    # And the slots they freed went to names with research behind them.
    assert set(PINS_WITH_MEMO) <= set(runs[0].tickers)


def test_a_regen_row_for_a_company_that_no_longer_exists_is_ignored(db):
    """`regen_jobs` outlives the company row it names; a loop cannot research
    a ticker we hold no profile for."""
    add_regen(db, "GHOST", age_days=0.5)      # no company row
    add_company(db, "REAL")
    add_memo(db, "REAL", age_days=200)

    sel = select_focus(budget=5, now=NOW, db=db)

    assert "GHOST" not in sel.tickers
    assert "GHOST" not in sel.dropped
    assert sel.tickers == ("REAL",)


def test_a_regen_older_than_the_window_falls_back_to_its_memo_band(db):
    add_company(db, "AAA")
    add_regen(db, "AAA", age_days=90)         # outside the 30d window
    add_memo(db, "AAA", age_days=2)

    sel = select_focus(budget=5, now=NOW, db=db)

    assert sel.reasons["AAA"] == BAND_MEMO_FRESH


def test_the_live_research_reserve_survives_a_full_pin_list(db):
    """Two concurrent pieces of research both get in, not just the first."""
    for ticker in STATIC_PINS:
        add_company(db, ticker, pinned=True)
        add_memo(db, ticker, age_days=3)
    for ticker in ("MELI", "SHOP"):
        add_company(db, ticker)
        add_memo(db, ticker, age_days=300)
        add_regen(db, ticker, age_days=0.01)

    sel = select_focus(budget=10, now=NOW, db=db)

    assert {"MELI", "SHOP"} <= set(sel.tickers)
    assert LIVE_RESEARCH_RESERVE == 2


# ---------------------------------------------------------------------------
# Actionability — a slot the loop could not have used is a slot wasted
# ---------------------------------------------------------------------------

def test_a_pin_with_no_memo_does_not_take_a_news_slot(db):
    """`on_news_alert`'s second guard returns `no_prior_memo` and stops.

    So a news slot spent on a ticker with no memo on file is a guaranteed
    no-op — 5 of the 10 pins, which is half the loop's hourly Gemini spend.
    Those slots go to names the loop can actually act on.
    """
    for ticker in STATIC_PINS:
        add_company(db, ticker, pinned=True)
    for ticker in PINS_WITH_MEMO:
        add_memo(db, ticker, age_days=3)
    for ticker in ("COST", "TSM", "BAC", "XOM", "V"):
        add_company(db, ticker)
        add_memo(db, ticker, age_days=2)

    sel = select_focus(budget=10, now=NOW, db=db, require_memo=True)

    assert set(sel.tickers).isdisjoint(PINS_WITHOUT_MEMO)
    assert set(sel.withheld) == set(PINS_WITHOUT_MEMO)
    assert set(sel.withheld.values()) == {WITHHELD_PINNED_NO_MEMO}
    # The freed slots were actually spent, not simply lost.
    assert len(sel.tickers) == 10
    assert {"COST", "TSM", "BAC", "XOM", "V"} <= set(sel.tickers)


def test_a_withheld_pin_re_enters_selection_the_moment_its_first_memo_lands(db):
    """Nothing about withholding is sticky — verified, not asserted in prose."""
    for ticker in STATIC_PINS:
        add_company(db, ticker, pinned=True)
    for ticker in PINS_WITH_MEMO:
        add_memo(db, ticker, age_days=5)

    before = select_focus(budget=10, now=NOW, db=db, require_memo=True)
    assert "META" not in before.tickers
    assert before.withheld["META"] == WITHHELD_PINNED_NO_MEMO

    # A filing lands and the regen worker writes META's first memo.
    add_memo(db, "META", age_days=0.01)

    after = select_focus(budget=10, now=NOW, db=db, require_memo=True)

    assert "META" in after.tickers
    assert after.reasons["META"] == BAND_PINNED
    assert "META" not in after.withheld
    # Newest research, so it ranks at the top of band 0.
    assert after.tickers[0] == "META"


def test_a_regen_whose_first_memo_has_not_landed_yet_is_withheld_under_its_own_label(db):
    """Transient, and labelled apart from the pins so the log reads correctly."""
    add_company(db, "SHOP")
    add_regen(db, "SHOP", age_days=0.01)      # requested; memo still generating
    add_company(db, "MELI")
    add_memo(db, "MELI", age_days=1)
    add_regen(db, "MELI", age_days=0.02)

    sel = select_focus(budget=10, now=NOW, db=db, require_memo=True)

    assert sel.tickers == ("MELI",)
    assert sel.withheld == {"SHOP": WITHHELD_REGEN_NO_MEMO}


def test_the_memo_bands_can_never_be_withheld(db):
    """Bands 2 and 3 are *defined* by carrying a memo, so `require_memo`
    cannot touch them. Guards against a future refactor that filters the
    ranking as a whole and quietly empties it."""
    add_company(db, "FRESH")
    add_memo(db, "FRESH", age_days=1)
    add_company(db, "STALE")
    add_memo(db, "STALE", age_days=300)

    sel = select_focus(budget=10, now=NOW, db=db, require_memo=True)

    assert set(sel.tickers) == {"FRESH", "STALE"}
    assert sel.withheld == {}


def test_withholding_is_off_by_default_so_the_social_loop_keeps_its_pins(db):
    """`social_agent.run` needs no prior memo — its scalar is an input to the
    first one — so the daily loop must not inherit the news loop's gate."""
    for ticker in STATIC_PINS:
        add_company(db, ticker, pinned=True)
    for ticker in PINS_WITH_MEMO:
        add_memo(db, ticker, age_days=3)

    sel = select_focus(budget=10, now=NOW, db=db)

    assert set(sel.tickers) == set(STATIC_PINS)
    assert sel.withheld == {}


def test_every_withheld_ticker_is_named_in_the_note(db):
    """A cap the reader cannot see is the behaviour this change removes."""
    for ticker in STATIC_PINS:
        add_company(db, ticker, pinned=True)
    for ticker in PINS_WITH_MEMO:
        add_memo(db, ticker, age_days=3)

    note = select_focus(budget=10, now=NOW, db=db, require_memo=True).note()

    assert f"withheld {len(PINS_WITHOUT_MEMO)}" in note
    assert f"{WITHHELD_PINNED_NO_MEMO}={len(PINS_WITHOUT_MEMO)}" in note
    for ticker in PINS_WITHOUT_MEMO:
        assert ticker in note, f"{ticker} was held back silently: {note}"


# ---------------------------------------------------------------------------
# Bands 2 / 3 and total ordering
# ---------------------------------------------------------------------------

def test_ordering_is_fully_deterministic_for_a_fixed_now(db):
    """Rank order across all four bands, with a ticker-ascending tiebreak."""
    add_company(db, "PIN2", pinned=True)
    add_company(db, "PIN1", pinned=True)
    add_company(db, "REG_OLDER")
    add_regen(db, "REG_OLDER", age_days=5)
    add_company(db, "REG_NEWER")
    add_regen(db, "REG_NEWER", age_days=1)
    add_company(db, "FRESH_B")
    add_memo(db, "FRESH_B", age_days=3)
    add_company(db, "FRESH_A")
    add_memo(db, "FRESH_A", age_days=3)       # exact tie -> ticker ascending
    add_company(db, "STALE")
    add_memo(db, "STALE", age_days=120)
    add_company(db, "NOSIGNAL")               # no memo, no regen, not pinned

    sel = select_focus(budget=10, now=NOW, db=db)

    assert sel.tickers == (
        "PIN1", "PIN2",                        # band 0, no activity -> ticker asc
        "REG_NEWER", "REG_OLDER",              # band 1, most recent first
        "FRESH_A", "FRESH_B",                  # band 2, tie broken by ticker
        "STALE",                               # band 3
    )
    # A ticker with no research signal at all is simply not selected — the
    # whole point of dropping the arbitrary slice.
    assert "NOSIGNAL" not in sel.tickers
    assert "NOSIGNAL" not in sel.dropped
    assert select_focus(budget=10, now=NOW, db=db).tickers == sel.tickers


def test_a_ticker_appears_in_exactly_one_band(db):
    add_company(db, "ALL", pinned=True)
    add_regen(db, "ALL", age_days=1)
    add_memo(db, "ALL", age_days=1)

    sel = select_focus(budget=10, now=NOW, db=db)

    assert sel.tickers == ("ALL",)
    assert sel.reasons == {"ALL": BAND_PINNED}


def test_backtest_snapshots_do_not_count_as_research(db):
    """`as_of_date` means "memo reproduced as of an earlier date" — not
    someone watching the ticker today. `memo_store.latest_memo` excludes
    them by default and so does the ranking."""
    add_company(db, "BACKTEST")
    add_memo(db, "BACKTEST", age_days=1, backtest=True)

    sel = select_focus(budget=10, now=NOW, db=db)

    assert sel.tickers == ()


def test_a_backtest_only_ticker_is_not_actionable_either(db):
    """The `as_of_date IS NULL` filter is the same predicate
    `memo_store.latest_memo` applies, so `require_memo` and
    `on_news_alert` agree about who has a memo."""
    add_company(db, "BACKTEST", pinned=True)
    add_memo(db, "BACKTEST", age_days=1, backtest=True)

    sel = select_focus(budget=10, now=NOW, db=db, require_memo=True)

    assert sel.tickers == ()
    assert sel.withheld == {"BACKTEST": WITHHELD_PINNED_NO_MEMO}


# ---------------------------------------------------------------------------
# The budget is never silently enforced
# ---------------------------------------------------------------------------

def test_budget_is_respected_and_every_dropped_ticker_is_named(db):
    for i in range(6):
        add_company(db, f"PIN{i}", pinned=True)
    for i in range(4):
        add_company(db, f"REG{i}")
        add_regen(db, f"REG{i}", age_days=1 + i)
    for i in range(3):
        add_company(db, f"FRESH{i}")
        add_memo(db, f"FRESH{i}", age_days=2 + i)
    add_company(db, "STALE0")
    add_memo(db, "STALE0", age_days=300)

    sel = select_focus(budget=8, now=NOW, db=db)

    assert len(sel.tickers) == 8
    # 14 qualify, 8 fit.
    assert len(sel.dropped) == 6
    assert set(sel.tickers) | set(sel.dropped) == set(sel.reasons) | set(sel.dropped)
    # The reserve went to the two most recent regen requests, and the
    # guaranteed prefix to the top of the ranking.
    assert {"REG0", "REG1"} <= set(sel.tickers)
    assert {"PIN0", "PIN1", "PIN2", "PIN3"} <= set(sel.tickers)

    note = sel.note()
    assert note.startswith("focus 8 (")
    assert f"dropped {len(sel.dropped)}" in note
    for ticker in sel.dropped:
        assert ticker in note, f"{ticker} was dropped silently: {note}"
    assert f"{BAND_PINNED}=" in note


def test_dropped_keeps_rank_order_so_the_worst_omission_reads_first(db):
    add_company(db, "PIN", pinned=True)
    add_company(db, "REG")
    add_regen(db, "REG", age_days=0.5)
    for i in range(4):
        add_company(db, f"STALE{i}")
        add_memo(db, f"STALE{i}", age_days=100 + i)

    sel = select_focus(budget=2, now=NOW, db=db)

    assert sel.dropped[0] == "PIN"            # band 0 sorts ahead of band 3
    assert set(sel.tickers) | set(sel.dropped) == {"PIN", "REG", *(f"STALE{i}" for i in range(4))}


def test_note_is_quiet_when_nothing_was_dropped(db):
    add_company(db, "AAPL", pinned=True)

    note = select_focus(budget=10, now=NOW, db=db).note()

    assert note == f"focus 1 ({BAND_PINNED}=1)"
    assert "dropped" not in note
    assert "withheld" not in note


# ---------------------------------------------------------------------------
# Anti-starvation, at every caller cadence
# ---------------------------------------------------------------------------

# `news_loop` runs hourly and `social_loop` daily, and they call the same
# selector. The rotation offset comes off the wall clock, so a daily caller
# advances it 24 places per run unless it says so: reachable offsets then
# collapse to `len(pool) / gcd(24, len(pool))`. The pool below is 12 long,
# where gcd(24, 12) == 12 — the worst case, 2 of 12 names covered and the
# other 10 starved forever. Both cadences must cover everything.
_CADENCES = [
    pytest.param(1, timedelta(hours=1), id="hourly-news_loop"),
    pytest.param(24, timedelta(days=1), id="daily-social_loop"),
]


@pytest.mark.parametrize("period_hours,step", _CADENCES)
def test_rotation_eventually_covers_every_ranked_ticker(db, period_hours, step):
    """The property the rotation exists for, asserted directly at both cadences.

    A deterministic ranking with a fixed budget runs the same head every
    run forever: everything below the cut is never monitored, not once. So
    the leftover budget walks the rest one step per RUN — not per hour.
    Over as many consecutive runs as the pool is long, the union of what
    ran must be the whole list, not "most of it".
    """
    add_company(db, "PIN", pinned=True)
    tail = [f"STALE{i:02d}" for i in range(13)]
    for i, ticker in enumerate(tail):
        add_company(db, ticker)
        add_memo(db, ticker, age_days=100 + i)

    budget = 4
    covered: set[str] = set()
    pool_size = None
    for run in range(len(tail) + 1):
        sel = select_focus(
            budget=budget, now=NOW + step * run, db=db,
            rotation_period_hours=period_hours,
        )
        assert sel.tickers[0] == "PIN"          # the ranked prefix is stable
        assert len(sel.tickers) == budget
        pool_size = sel.rotation_pool_size
        covered.update(sel.tickers)
        # Nothing is dropped quietly on the way round.
        assert set(sel.tickers) | set(sel.dropped) == {"PIN", *tail}

    assert pool_size == 12, "the aliasing worst case is not being exercised"
    assert covered == {"PIN", *tail}, (
        "these tickers are starved — the rotation never reaches them at "
        f"{period_hours}h cadence: {sorted({'PIN', *tail} - covered)}"
    )


def test_a_daily_caller_that_claims_to_be_hourly_is_the_bug_this_guards(db):
    """The regression itself, pinned: 12 pool members, 365 daily runs.

    Kept as a separate test so the next reader can see what the parameter
    is for. Advancing the wall clock a day at a time while telling the
    selector it runs hourly steps the offset by 24, and `gcd(24, 12) == 12`
    leaves exactly one reachable offset.
    """
    add_company(db, "PIN", pinned=True)
    tail = [f"STALE{i:02d}" for i in range(13)]
    for i, ticker in enumerate(tail):
        add_company(db, ticker)
        add_memo(db, ticker, age_days=100 + i)

    def coverage(period_hours: int) -> set[str]:
        out: set[str] = set()
        for day in range(365):
            out.update(select_focus(
                budget=4, now=NOW + timedelta(days=day), db=db,
                rotation_period_hours=period_hours,
            ).tickers)
        return out

    assert len(coverage(1)) < len(tail)        # the defect, reproduced
    assert coverage(24) == {"PIN", *tail}      # and what fixes it


def test_the_rotation_window_is_wider_than_one_slot_so_the_throttle_cannot_starve_it(db):
    """`news_loop._THROTTLE_SECONDS` and the loop interval are both 1 hour.

    A run that fires a few seconds early finds `elapsed < 3600` and skips
    the ticker. If a tail name were selected for exactly one run it could
    be throttled out on every appearance and never actually run. It is
    selected for `ROTATING_SLOTS` consecutive runs instead, so a missed
    hour is recoverable.
    """
    add_company(db, "PIN", pinned=True)
    tail = [f"STALE{i:02d}" for i in range(9)]
    for i, ticker in enumerate(tail):
        add_company(db, ticker)
        add_memo(db, ticker, age_days=100 + i)

    runs = [
        set(select_focus(budget=4, now=NOW + timedelta(hours=h), db=db).tickers)
        for h in range(len(tail) * 2)
    ]

    for ticker in tail:
        # Doubled so a run that wraps past the end of the cycle is not
        # counted as two short ones.
        consecutive = max(_true_runs(ticker in run for run in runs + runs))
        assert consecutive >= ROTATING_SLOTS, (
            f"{ticker} is only ever selected for one run at a time"
        )


def _true_runs(flags) -> list[int]:
    """Lengths of each consecutive run of True in a bool sequence."""
    import itertools
    return [len(list(g)) for k, g in itertools.groupby(flags) if k]


def test_an_empty_pool_does_not_divide_by_zero(db):
    add_company(db, "PIN", pinned=True)

    sel = select_focus(budget=10, now=NOW, db=db)

    assert sel.tickers == ("PIN",)
    assert sel.rotation_pool_size == 0
    assert sel.rotation_slots == 0
    assert sel.rotation_offset == 0


def test_a_nonsense_rotation_period_does_not_crash(db):
    """A zero period would be a ZeroDivisionError inside a scheduler thread."""
    add_company(db, "PIN", pinned=True)
    for i in range(4):
        add_company(db, f"STALE{i}")
        add_memo(db, f"STALE{i}", age_days=100 + i)

    for period in (0, -5):
        sel = select_focus(budget=3, now=NOW, db=db, rotation_period_hours=period)
        assert len(sel.tickers) == 3


def test_a_zero_budget_selects_nothing_and_reports_what_it_skipped(db):
    add_company(db, "PIN", pinned=True)

    sel = select_focus(budget=0, now=NOW, db=db)

    assert sel.tickers == ()
    assert sel.dropped == ("PIN",)
    assert "PIN" in sel.note()


# ---------------------------------------------------------------------------
# Degraded path
# ---------------------------------------------------------------------------

class _ExplodingSession:
    """A session that fails the way a sick Postgres does: on first query."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("could not connect to server: Connection refused")


def test_a_database_failure_degrades_to_the_static_pin_list():
    sel = select_focus(budget=10, now=NOW, db=_ExplodingSession())

    assert sel.degraded is True
    assert sel.tickers == STATIC_PINS
    assert set(sel.reasons.values()) == {BAND_DEGRADED}
    assert "degraded" in sel.note()


def test_the_degraded_path_ignores_require_memo():
    """Memo presence is a fact the database holds, and the database is what
    just failed — guessing would drop pins for a reason we cannot check."""
    sel = select_focus(budget=10, now=NOW, db=_ExplodingSession(), require_memo=True)

    assert sel.tickers == STATIC_PINS
    assert sel.withheld == {}


def test_the_degraded_path_still_respects_the_budget_and_keeps_the_biggest_names():
    sel = select_focus(budget=3, now=NOW, db=_ExplodingSession())

    assert sel.tickers == STATIC_PINS[:3]
    assert sel.dropped == STATIC_PINS[3:]
    for ticker in sel.dropped:
        assert ticker in sel.note()


def test_the_fallback_has_its_own_fallback(monkeypatch):
    """Both the DB and sp500.json unreadable: an empty selection, not a crash.

    A monitoring loop must not die because ticker selection had a bad day;
    a run that covers nothing is a legible outcome, an exception escaping
    into the scheduler is not.
    """
    import app.seed_universe as seed_universe

    def boom(*args, **kwargs):
        raise OSError("no such file or directory: sp500.json")

    monkeypatch.setattr(seed_universe, "load_universe_file", boom)

    sel = select_focus(budget=10, now=NOW, db=_ExplodingSession())

    assert sel.tickers == ()
    assert sel.degraded is True
    assert "degraded" in sel.note()


# ---------------------------------------------------------------------------
# The regression guard
# ---------------------------------------------------------------------------

_ARBITRARY_SLICE = re.compile(r"list_tickers\s*\(\s*\)\s*\)?\s*\[\s*:\s*\d+\s*\]")


def _code_only(module) -> str:
    """Module source with comments and string literals stripped out.

    The guard has to read code, not prose. Both loops now carry comments
    naming the old `list_tickers()[:10]` slice and explaining why it was
    wrong, and a guard that matched those would fire on its own
    documentation — which would train the next person to delete the
    explanation rather than the defect.
    """
    tokens = []
    for tok in tokenize.generate_tokens(io.StringIO(inspect.getsource(module)).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            tokens.append(tok.string)
    return " ".join(tokens)


@pytest.mark.parametrize("module", [news_loop, social_loop], ids=lambda m: m.__name__)
def test_neither_loop_picks_tickers_off_an_arbitrary_universe_slice(module):
    """Asserted against the source, because the defect is invisible at runtime.

    `list_tickers()` returns rows in whatever order the database hands them
    back. Any `[:N]` of that is an accident of insertion order that will
    keep passing every behavioural test ever written — the loop runs, the
    cache fills, cron-health goes green. Only the source says which ten.
    """
    code = _code_only(module)

    assert not _ARBITRARY_SLICE.search(code), (
        f"{module.__name__} is back to slicing an unordered ticker list; use "
        "research_focus.select_focus so the budget goes to researched names"
    )
    assert "list_tickers" not in code, (
        f"{module.__name__} reads the raw universe list again — there is no "
        "ordering on it to make that meaningful"
    )
    assert "select_focus" in code, f"{module.__name__} no longer ranks its tickers"


def test_each_loop_declares_its_budget_as_a_named_constant_with_the_cost_math():
    assert news_loop.NEWS_FOCUS_BUDGET == 10
    assert social_loop.SOCIAL_FOCUS_BUDGET == 10
    for module, name in ((news_loop, "NEWS_FOCUS_BUDGET"), (social_loop, "SOCIAL_FOCUS_BUDGET")):
        source = inspect.getsource(module)
        head = source.split(f"{name} =")[0]
        assert "Gemini" in head and "day" in head, (
            f"{module.__name__}'s budget constant has no cost math above it; the "
            "number is meaningless to whoever considers raising it"
        )


class _FakeScheduler:
    def __init__(self):
        self.jobs: list[dict] = []

    def add_job(self, func, trigger, **kwargs):
        self.jobs.append({"trigger": trigger, **kwargs})


@pytest.mark.parametrize(
    "module,expected_hours",
    [(news_loop, 1), (social_loop, 24)],
    ids=lambda v: getattr(v, "__name__", v),
)
def test_each_loop_gives_the_rotation_the_same_interval_it_gives_the_scheduler(
    module, expected_hours, monkeypatch,
):
    """The rotation advances one position per RUN, and the only way it can
    know what a run is worth is the interval the caller passes. If these two
    ever disagree the tail starves silently — `social_loop` at `days=1` with
    an hourly rotation covered 2 of 12 names and nobody would have noticed.
    """
    scheduler = _FakeScheduler()
    module.register(scheduler)
    assert scheduler.jobs == [{
        "trigger": "interval", "hours": expected_hours,
        "id": module.__name__.rsplit(".", 1)[-1], "replace_existing": True,
    }]

    seen: dict = {}
    _capture_kwargs(monkeypatch, module, seen)
    module.run_once()

    assert seen.get("rotation_period_hours") == expected_hours, (
        f"{module.__name__} runs every {expected_hours}h but tells the rotation "
        f"{seen.get('rotation_period_hours')!r}"
    )


def test_only_the_news_loop_requires_an_actionable_ticker(monkeypatch):
    """News alerts are dead without a prior memo; sentiment scalars are not.

    `social_agent.run` is also the tool `sdk_runtime.run_social_agent` calls
    during memo generation, against a 24h cache — so the daily pass warms
    the first memo of exactly the pins that have none yet.
    """
    news_kwargs: dict = {}
    social_kwargs: dict = {}
    for module, sink in ((news_loop, news_kwargs), (social_loop, social_kwargs)):
        _capture_kwargs(monkeypatch, module, sink)
        module.run_once()

    assert news_kwargs["require_memo"] is True
    assert social_kwargs.get("require_memo", False) is False


def test_an_explicit_ticker_argument_bypasses_selection_entirely(monkeypatch):
    """Callers that already know what they want must not pay for a query."""
    for module in (news_loop, social_loop):
        called: list[int] = []
        monkeypatch.setattr(
            module, "select_focus",
            lambda _seen=called, **kwargs: _seen.append(1),
        )
        monkeypatch.setattr(module, "record_run", lambda *a, **k: None)
        if module is news_loop:
            monkeypatch.setattr(module.news_agent, "run", lambda t, **k: [])
            monkeypatch.setattr(module, "_last_run_for", lambda t: None)
            monkeypatch.setattr(module, "_record_run_for", lambda t: None)
        else:
            monkeypatch.setattr(module.social_agent, "run", lambda t, **k: {})
        module.run_once(["NVDA"])
        assert called == []


def test_the_loop_note_carries_the_selection_summary(monkeypatch):
    """Dropped and withheld tickers have to reach `/api/admin/cron-health`,
    not just a log line nobody reads."""
    selection = FocusSelection(
        tickers=("NVDA",), reasons={"NVDA": BAND_PINNED},
        dropped=("MELI", "SHOP"),
        withheld={"TSLA": WITHHELD_PINNED_NO_MEMO},
    )
    recorded: list[dict] = []
    monkeypatch.setattr(news_loop, "select_focus", lambda **kwargs: selection)
    monkeypatch.setattr(news_loop, "record_run", lambda *a, **k: recorded.append(k))
    monkeypatch.setattr(news_loop, "_last_run_for", lambda t: None)
    monkeypatch.setattr(news_loop, "_record_run_for", lambda t: None)
    monkeypatch.setattr(news_loop.news_agent, "run", lambda t, **k: [])

    news_loop.run_once()

    note = recorded[0]["note"]
    assert recorded and "MELI" in note and "SHOP" in note and "TSLA" in note
    assert note.startswith("0 material events; focus 1")


def test_stale_band_reason_is_labelled(db):
    add_company(db, "STALE")
    add_memo(db, "STALE", age_days=300)

    sel = select_focus(budget=5, now=NOW, db=db)

    assert sel.reasons == {"STALE": BAND_MEMO_STALE}
