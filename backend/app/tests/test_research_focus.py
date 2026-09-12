"""The monitoring loops must spend their budget on researched tickers.

Both `news_loop` and `social_loop` used to pick their work with

    tickers = list(ds.list_tickers())[:10]   # demo universe sample

and `DataService.list_tickers()` has no ORDER BY, so that was the first ten
rows the seeder inserted — the alphabetical head of the S&P 500. Every hour,
forever, the news loop spent 10 Gemini calls on A / AAPL / ABBV / ABNB /
ACGL / ACN / ADBE / ADI / ADM / ADP. Of the 172 companies in the universe
only 17 have ever had a memo generated and only 10 carry the
`auto_update_memo` pin; the alphabetical head overlapped that set by two.
NVDA has 230 memo versions and MSFT 108, and neither was being monitored.

The failure mode is quiet in the worst way: the loop reports success, the
cache fills with fresh news, cron-health is green, and the news is about
companies nobody is researching. Nothing 500s. Nothing is logged. The only
symptom is a memo that never mentions the thing that moved the stock.

So these tests assert the selection *policy*, not the plumbing: pins always
run, a live regen request outranks a stale memo, the order is a pure
function of the injected clock, an over-budget run names every ticker it
dropped, the rotation actually reaches every stale name rather than
re-running the same head forever, and a sick database degrades to the ten
biggest companies instead of to an alphabetical accident. The last test is
the regression guard proper: it re-reads both loop modules and fails if the
arbitrary slice ever comes back.
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
    BAND_DEGRADED,
    BAND_MEMO_FRESH,
    BAND_MEMO_STALE,
    BAND_PINNED,
    BAND_REGEN,
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
# Band 0 — the pins are the contract
# ---------------------------------------------------------------------------

def test_every_pinned_ticker_is_selected_even_with_no_memo(db):
    """`auto_update_memo` is a promise, and half the pins have no memo yet.

    Five of the ten production pins (META, BRK.B, TSLA, AVGO, LLY) have
    never had a memo generated. A selector that ranked on memo evidence
    alone would drop exactly the names the owner curated by hand.
    """
    for ticker in STATIC_PINS:
        add_company(db, ticker, pinned=True)
    # Plenty of competition with real, recent research behind it.
    for ticker in ("COST", "TSM", "BAC", "XOM", "V"):
        add_company(db, ticker)
        add_memo(db, ticker, age_days=1)

    sel = select_focus(budget=10, now=NOW, db=db)

    assert set(sel.tickers) == set(STATIC_PINS)
    assert set(sel.reasons.values()) == {BAND_PINNED}
    assert sel.degraded is False


def test_pins_never_rotate_out_when_the_head_is_over_budget(db):
    """Bands 0-2 are truncated in rank order, never rotated.

    Rotating the head would make a pin miss a turn, which is the one thing
    `auto_update_memo` exists to prevent. Advance the clock a full day an
    hour at a time and the pinned head must not move.
    """
    for ticker in STATIC_PINS:
        add_company(db, ticker, pinned=True)
    for ticker in ("COST", "TSM", "BAC"):
        add_company(db, ticker)
        add_memo(db, ticker, age_days=1)

    runs = [
        select_focus(budget=10, now=NOW + timedelta(hours=h), db=db).tickers
        for h in range(24)
    ]

    assert all(run == runs[0] for run in runs)
    assert set(runs[0]) == set(STATIC_PINS)


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
        "PIN1", "PIN2",                        # band 0, ticker ascending
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
    # Head is 13 long (6 + 4 + 3); 5 of it plus the whole tail are dropped.
    assert len(sel.dropped) == 6
    assert set(sel.dropped) == {"REG3", "FRESH0", "FRESH1", "FRESH2", "STALE0"} | {"REG2"}
    # Rank order is preserved in `dropped`, so the most important omission
    # is the first thing the reader sees.
    assert sel.dropped[0] == "REG2"

    note = sel.note()
    assert note.startswith("focus 8 (")
    assert f"dropped {len(sel.dropped)}" in note
    for ticker in sel.dropped:
        assert ticker in note, f"{ticker} was dropped silently: {note}"
    # And the selected ones are accounted for by band.
    assert f"{BAND_PINNED}=6" in note
    assert f"{BAND_REGEN}=2" in note


def test_note_is_quiet_when_nothing_was_dropped(db):
    add_company(db, "AAPL", pinned=True)

    note = select_focus(budget=10, now=NOW, db=db).note()

    assert note == f"focus 1 ({BAND_PINNED}=1)"
    assert "dropped" not in note


# ---------------------------------------------------------------------------
# Anti-starvation
# ---------------------------------------------------------------------------

def test_rotation_eventually_covers_every_stale_ticker(db):
    """The property the rotation exists for, asserted directly.

    A deterministic ranking with a fixed budget runs the same head every
    hour forever: the tail below the cut is never monitored, not once. So
    the leftover budget walks the tail one step per hour. Over as many
    consecutive hours as the tail is long, the union of what ran must be
    the entire tail — not "most of it".
    """
    add_company(db, "PIN", pinned=True)
    tail = [f"STALE{i:02d}" for i in range(17)]
    for i, ticker in enumerate(tail):
        add_company(db, ticker)
        add_memo(db, ticker, age_days=100 + i)

    budget = 4                                  # 1 pin + 3 rotating tail slots
    covered: set[str] = set()
    for hour in range(len(tail)):
        sel = select_focus(budget=budget, now=NOW + timedelta(hours=hour), db=db)
        assert sel.tickers[0] == "PIN"          # the head never rotates
        assert len(sel.tickers) == budget
        assert sel.tail_size == len(tail)
        covered.update(sel.tickers[1:])
        # Nothing is dropped quietly on the way round.
        assert set(sel.tickers[1:]) | set(sel.dropped) == set(tail)

    assert covered == set(tail), (
        "these band-3 tickers are starved — the rotation never reaches them: "
        f"{sorted(set(tail) - covered)}"
    )


def _true_runs(flags) -> list[int]:
    """Lengths of each consecutive run of True in a bool sequence."""
    import itertools
    return [len(list(g)) for k, g in itertools.groupby(flags) if k]


def test_the_rotation_window_is_wider_than_one_slot_so_the_throttle_cannot_starve_it(db):
    """`news_loop._THROTTLE_SECONDS` and the loop interval are both 1 hour.

    A run that fires a few seconds early finds `elapsed < 3600` and skips
    the ticker. If a tail name were selected for exactly one hour it could
    be throttled out on every appearance and never actually run. It is
    selected for `budget - len(head)` consecutive hours instead, so a
    missed hour is recoverable.
    """
    add_company(db, "PIN", pinned=True)
    tail = [f"STALE{i:02d}" for i in range(9)]
    for i, ticker in enumerate(tail):
        add_company(db, ticker)
        add_memo(db, ticker, age_days=100 + i)

    budget = 4
    runs = [
        set(select_focus(budget=budget, now=NOW + timedelta(hours=h), db=db).tickers)
        for h in range(len(tail))
    ]

    for ticker in tail:
        # Doubled so a run that wraps past the end of the cycle is not
        # counted as two short ones.
        consecutive = max(_true_runs(ticker in run for run in runs + runs))
        assert consecutive >= 2, f"{ticker} is only ever selected for one hour at a time"


def test_an_empty_tail_does_not_divide_by_zero(db):
    add_company(db, "PIN", pinned=True)

    sel = select_focus(budget=10, now=NOW, db=db)

    assert sel.tickers == ("PIN",)
    assert sel.tail_size == 0
    assert sel.rotation_offset == 0


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
    """Dropped tickers have to reach `/api/admin/cron-health`, not just a log."""
    from app.monitoring.research_focus import FocusSelection
    selection = FocusSelection(
        tickers=("NVDA",), reasons={"NVDA": BAND_PINNED},
        dropped=("MELI", "SHOP"),
    )
    recorded: list[dict] = []
    monkeypatch.setattr(news_loop, "select_focus", lambda **kwargs: selection)
    monkeypatch.setattr(news_loop, "record_run", lambda *a, **k: recorded.append(k))
    monkeypatch.setattr(news_loop, "_last_run_for", lambda t: None)
    monkeypatch.setattr(news_loop, "_record_run_for", lambda t: None)
    monkeypatch.setattr(news_loop.news_agent, "run", lambda t, **k: [])

    news_loop.run_once()

    assert recorded and "MELI" in recorded[0]["note"] and "SHOP" in recorded[0]["note"]
    assert recorded[0]["note"].startswith("0 material events; focus 1")


def test_stale_band_reason_is_labelled(db):
    add_company(db, "STALE")
    add_memo(db, "STALE", age_days=300)

    sel = select_focus(budget=5, now=NOW, db=db)

    assert sel.reasons == {"STALE": BAND_MEMO_STALE}
