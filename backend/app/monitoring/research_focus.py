"""Pick the tickers the monitoring loops should spend their budget on.

`news_loop` and `social_loop` used to choose their work like this::

    tickers = list(ds.list_tickers())[:10]   # demo universe sample

`DataService.list_tickers()` is `db.query(Company.ticker).all()` with no
ORDER BY, so *which* ten that returned is not something the source
determines: it is whatever the engine hands back, and it can change when
rows are rewritten or the query plan does. What can be checked is the
shape of the list being sliced, and that is what makes the slice a bug
rather than a choice. Both facts below were re-derived by reading the
files, not observed against production:

  * `app/data/sp500.json` `tickers` holds 170 symbols in two
    alphabetically sorted runs — the SP100 block (AAPL..XOM) followed by
    the curated-extensions block (ADI..ZTS) — and `seed_universe._seed`
    iterates it in file order, so a table filled by that seeder has
    AAPL, ABBV, ABT, ACN, ADBE, AIG, AMD, AMGN, AMT, AMZN at its head.
  * Of the 172 companies in the universe only 17 have ever had a memo
    generated. That file head overlaps the 17 by four (AAPL, ABBV, ADBE,
    AMZN) and overlaps the ten curated pins by two (AAPL, AMZN).

So the loop's whole budget went to names chosen by insertion order, while
NVDA (230 memo versions), MSFT (108) and every ticker a user had just
asked us to research got no coverage at all.

Rank by evidence that a ticker is actually being researched, and spend the
budget from the top. A ticker with no research signal at all is simply not
selected — there is no fallback to an arbitrary slice, because an
arbitrary slice is the bug.

Bands, highest priority first:

  0 `pinned`          `Company.auto_update_memo` — the explicitly curated
                      auto-regen set (top 10 by market cap, from
                      sp500.json).
  1 `regen_requested` a `regen_jobs` row enqueued inside the window. This
                      is the truest "being researched right now" signal:
                      one row per regen request, including the manual,
                      user-triggered ones, so it catches on-demand names
                      (MELI, SHOP) that are outside the curated universe.
  2 `memo_fresh`      newest memo snapshot inside the window — the same
                      signal `update_orchestrator.should_auto_regen` uses
                      to decide a ticker is worth re-analysing.
  3 `memo_stale`      has a memo, but older than the window.

A ticker lands in exactly one band (its highest). Within a band, most
recent first, ties broken by ticker ascending, so the order is a total
function of the inputs. Band 0 is ranked the same way — by the most recent
of a pin's regen request and its newest memo, pins nobody has touched last
— because when the budget cannot hold every pin the cut has to fall
somewhere defensible instead of landing alphabetically, which is the bug
this module exists to remove.

Which memo table: the brief for this change named `StockMemo`
(`stock_memos`), but nothing has written to that table since `memo_store`
replaced it — it is the legacy pre-versioning store and it is empty in
production. The live memos, and the version counts the ranking is meant to
reflect, are `MemoSnapshot` (`memo_snapshots`), which is also what
`should_auto_regen` reads through `memo_store.latest_memo`. Ranking on the
empty table would have selected nobody. Backtest snapshots (`as_of_date`
set) are excluded, matching `latest_memo`'s default: a reproduced-as-of
memo is not someone watching the ticker today.

Actionability (`require_memo`). Ranking says who deserves a slot; this says
who can *use* one. The news loop's alerts feed exactly one action path,
`update_orchestrator.on_news_alert`, whose second guard is::

    snap = memo_store.latest_memo(ticker)
    if snap is None:
        return {"patched": False, "ticker": ticker, "reason": "no_prior_memo"}

so a news alert on a ticker with no memo on file provably cannot do
anything. Five of the ten pins (AVGO, BRK.B, LLY, META, TSLA — the pin
list minus the five that carry memos) are in exactly that state, which is
half the news loop's hourly spend buying a guaranteed no-op while the
ticker a user asked about five minutes ago is dropped. With
`require_memo=True` such a ticker does not take a slot; it is reported
under its own label (`pinned_no_memo` / `regen_requested_no_memo`) in
`withheld`, so cron-health explains the decision rather than hiding it.
Bands 2 and 3 are defined by having a memo, so they can never be withheld.
Nothing is sticky: the moment a filing gives that pin its first memo it is
eligible again on the next run, with no intervention — asserted in
`test_research_focus.py`, not assumed.

This is scoped per caller and is NOT set for the social loop.
`social_agent.run` writes a sentiment scalar that neither reads nor needs a
prior memo, and `sdk_runtime.run_social_agent` calls the same function
during memo generation against a 24h cache — so the daily pre-warm is an
*input* to a pin's first memo. Withholding it from a memo-less pin would
remove the very warm-up the first memo benefits from, to save one Gemini
call a day.

Spending the budget. Ranking alone is not enough. In production the pin
list is exactly ten names and the budget is exactly ten, so "take the top
`budget` in rank order" runs the identical ten every hour forever: band 1
is unreachable, the tail is never touched, and the selection is as static
as the slice it replaced. So a run that cannot afford everything splits
its budget three ways:

  reserve     `LIVE_RESEARCH_RESERVE` slots go to band 1 ahead of anything
              else, so a ticker someone asked us to research is covered on
              every run even when the pins alone would fill the budget.
              Unused reserve returns to the pool — it costs nothing on the
              (common) runs where no unpinned ticker is being researched.
  guaranteed  the next names in rank order, covered on every run.
  rotating    `ROTATING_SLOTS` slots walk everything below the guaranteed
              prefix, advancing exactly one position per run, so the whole
              tail gets covered over a cycle instead of never. The window
              is two slots wide deliberately: `news_loop` throttles each
              ticker for exactly one hour and fires hourly, so a one-slot
              window could be throttled out on every single appearance.

The consequence is that a pin can sit out a run — that is the price of a
budget the same size as the pin list, and it is paid by the pins with the
least research activity (see the band-0 ranking above) rather than by the
ticker a user is reading right now. Nothing is ever permanently excluded:
every name below the guaranteed prefix comes round within one rotation
cycle, a property the tests assert exhaustively rather than sample.

`rotation_period_hours` is how the rotation learns the caller's cadence,
and it is not optional. The offset is derived from the wall clock, so
`hours_since_epoch % len(pool)` advances by one position per *hour* — which
is right for `news_loop` (hourly) and badly wrong for `social_loop`
(daily), where consecutive runs are 24 hours apart and the offset jumps 24
places at a time. Reachable offsets then collapse to
`len(pool) / gcd(24, len(pool))`: at a pool of 12 that is 2 of 12 names,
and the other 10 get zero social coverage ever. Dividing by the caller's
period first makes every run advance exactly one position at any cadence.
Each loop passes the same constant it gives the scheduler, so the two
cannot drift apart.

The offset is derived from the injected `now`, so the whole selection is a
pure function of (database, clock, budget, period).

Degraded path: every DB read is wrapped. On failure we fall back to the
static pin list in `app/data/sp500.json`, in file (market-cap) order, and
say so in `note()`. It ignores `require_memo` and the ranking for the same
reason: both are answers the database holds, and the database is what just
failed. Monitoring the ten biggest names while it is unhappy beats
monitoring an arbitrary alphabetical slice, and beats monitoring nothing.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..services.update_orchestrator import AUTO_REGEN_RECENCY_DAYS

log = logging.getLogger(__name__)

__all__ = [
    "AUTO_REGEN_RECENCY_DAYS",
    "BAND_DEGRADED",
    "BAND_MEMO_FRESH",
    "BAND_MEMO_STALE",
    "BAND_PINNED",
    "BAND_REGEN",
    "LIVE_RESEARCH_RESERVE",
    "ROTATING_SLOTS",
    "WITHHELD_PINNED_NO_MEMO",
    "WITHHELD_REGEN_NO_MEMO",
    "FocusSelection",
    "select_focus",
]

# Reason labels. Also the band names `note()` counts by, so they are part
# of what shows up in `/api/admin/cron-health`.
BAND_PINNED = "pinned"
BAND_REGEN = "regen_requested"
BAND_MEMO_FRESH = "memo_fresh"
BAND_MEMO_STALE = "memo_stale"
BAND_DEGRADED = "degraded_static_pin"

_BAND_ORDER: tuple[str, ...] = (
    BAND_PINNED, BAND_REGEN, BAND_MEMO_FRESH, BAND_MEMO_STALE, BAND_DEGRADED,
)

# Why a ranked ticker was held back under `require_memo` — it qualified on
# research signal but the loop could not have acted on what it found. One
# label per band so the reader can tell "a curated pin has no memo yet"
# (expected, five of ten in production) from "someone is researching a
# ticker whose first memo has not landed" (transient, minutes).
WITHHELD_PINNED_NO_MEMO = "pinned_no_memo"
WITHHELD_REGEN_NO_MEMO = "regen_requested_no_memo"

_WITHHELD_LABEL: dict[str, str] = {
    BAND_PINNED: WITHHELD_PINNED_NO_MEMO,
    BAND_REGEN: WITHHELD_REGEN_NO_MEMO,
}

_WITHHELD_ORDER: tuple[str, ...] = (
    WITHHELD_PINNED_NO_MEMO, WITHHELD_REGEN_NO_MEMO,
)

# Slots of an over-budget run held for band 1 (`regen_requested`) before
# the ranked prefix gets a look in. Without this, ten pins and a budget of
# ten mean a user's on-demand ticker is never monitored — the pins fill the
# run every hour, forever. Two rather than one so a second concurrent piece
# of research is not queued behind the first for a whole rotation cycle.
# Unused slots are given straight back to the ranked prefix.
LIVE_RESEARCH_RESERVE = 2

# Slots of an over-budget run that rotate through everything below the
# guaranteed prefix, advancing one position per run. Two is the minimum
# that survives `news_loop`'s one-hour per-ticker throttle: at one slot a
# ticker is selected for exactly one run per cycle, and a run that fires a
# second early finds the throttle still warm and skips it — for every
# appearance, forever. Raising either constant does not raise spend; it
# only moves slots between "covered every run" and "covered in turn".
ROTATING_SLOTS = 2

_EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class FocusSelection:
    """What the loop should run, why, and what it could not afford.

    `reasons` maps every selected ticker to its band label. `dropped` is
    every ticker that qualified but did not fit the budget, still in rank
    order, so the note names the most important omission first. `withheld`
    maps a ticker the budget never got to consider — it ranked, but
    `require_memo` says the loop could not have acted on it — to the label
    saying which band lost it. The two are kept apart because they are
    different questions: `dropped` means "wait your turn", `withheld` means
    "there is nothing here to do yet".
    """

    tickers: tuple[str, ...] = ()
    reasons: dict[str, str] = field(default_factory=dict)
    dropped: tuple[str, ...] = ()
    withheld: dict[str, str] = field(default_factory=dict)
    degraded: bool = False
    # Where in the rotation pool this run's window started, how big the
    # pool was, and how many slots walked it. Diagnostics only — but they
    # are the numbers you need to explain why a given ticker did or did not
    # run this hour, and to tell "waiting its turn" from "starved".
    rotation_offset: int = 0
    rotation_pool_size: int = 0
    rotation_slots: int = 0

    def note(self) -> str:
        """One-line summary for `record_run`, hence for cron-health.

        Every dropped and every withheld ticker is named, deliberately and
        without a cap: a cap is exactly the silent behaviour this change
        exists to remove. The list is bounded by the size of the universe
        (~172).
        """
        counts = Counter(self.reasons.values())
        breakdown = ", ".join(
            f"{band}={counts[band]}" for band in _BAND_ORDER if counts.get(band)
        )
        out = f"focus {len(self.tickers)}"
        if breakdown:
            out += f" ({breakdown})"
        if self.degraded:
            out += " [degraded: static pins]"
        if self.rotation_slots and self.rotation_pool_size:
            # Says *why* a named drop is only a drop for this run.
            out += (
                f"; rotating {self.rotation_slots} of {self.rotation_pool_size}"
                f" @offset {self.rotation_offset}"
            )
        if self.withheld:
            held = Counter(self.withheld.values())
            labels = ", ".join(
                f"{label}={held[label]}" for label in _WITHHELD_ORDER if held.get(label)
            )
            out += (
                f"; withheld {len(self.withheld)} with no memo to patch"
                f" ({labels}): " + ", ".join(sorted(self.withheld))
            )
        if self.dropped:
            out += f"; over budget, dropped {len(self.dropped)}: " + ", ".join(self.dropped)
        return out


def _hours_since_epoch(now: datetime) -> int:
    """Whole hours since the Unix epoch, for the rotation.

    Naive datetimes are read as UTC, which is what the whole codebase
    stores (`datetime.utcnow()`). Computed as a timedelta rather than via
    `.timestamp()` because `.timestamp()` on a naive value interprets it
    as *local* time, which is not monotonic across a DST transition and
    would make the rotation jump backwards twice a year.
    """
    if now.tzinfo is not None:
        now = now.astimezone(UTC).replace(tzinfo=None)
    return int((now - _EPOCH).total_seconds() // 3600)


def _rotation_step(now: datetime, period_hours: int) -> int:
    """How many positions the rotation has advanced by `now`.

    One per run at the caller's cadence, whatever that cadence is. The
    hourly clock divided by the caller's period, rather than the clock
    itself, is the whole fix for the daily loop: at `period_hours=24`
    consecutive runs are 24 hours apart, so the raw hour count advances 24
    places and only `len(pool) / gcd(24, len(pool))` positions are ever
    reachable — 2 of 12, for instance, starving the other 10 forever.
    """
    return _hours_since_epoch(now) // max(1, int(period_hours))


def _by_recency_then_ticker(row: tuple[str, datetime]) -> tuple[timedelta, str]:
    """Sort key: most recent first, ties broken by ticker ascending."""
    return (_EPOCH - row[1], row[0])


def _gather(
    db: Session, *, cutoff: datetime,
) -> tuple[list[str], list[str], list[str], list[str], set[str]]:
    """Return the four bands in final rank order, plus who has a live memo."""
    from ..models import Company, MemoSnapshot, RegenJob

    known: set[str] = set()
    pinned: list[str] = []
    for ticker, auto in db.execute(
        select(Company.ticker, Company.auto_update_memo)
    ).all():
        if not ticker:
            continue
        ticker = ticker.upper()
        known.add(ticker)
        if auto:
            pinned.append(ticker)

    # A `regen_jobs` row can outlive the company row it names (on-demand
    # tickers get pruned), and a loop cannot research a company we have no
    # profile for — so everything below is filtered against `known`.
    regen_rows = [
        (t.upper(), ts or _EPOCH)
        for t, ts in db.execute(
            select(RegenJob.ticker, func.max(RegenJob.enqueued_at))
            .where(RegenJob.enqueued_at >= cutoff)
            .group_by(RegenJob.ticker)
        ).all()
        if t and t.upper() in known
    ]

    # `as_of_date IS NULL` mirrors `memo_store.latest_memo`'s default
    # exactly, which is what makes `has_memo` below the same predicate
    # `on_news_alert` will apply when the alert arrives.
    memo_rows = [
        (t.upper(), ts or _EPOCH)
        for t, ts in db.execute(
            select(MemoSnapshot.ticker, func.max(MemoSnapshot.generated_at))
            .where(MemoSnapshot.as_of_date.is_(None))
            .group_by(MemoSnapshot.ticker)
        ).all()
        if t and t.upper() in known
    ]

    regen_rows.sort(key=_by_recency_then_ticker)
    memo_rows.sort(key=_by_recency_then_ticker)
    has_memo = {t for t, _ts in memo_rows}

    # Last time anyone did anything to this ticker, for the band-0 rank.
    activity: dict[str, datetime] = {}
    for ticker, ts in regen_rows + memo_rows:
        if ts > activity.get(ticker, _EPOCH):
            activity[ticker] = ts

    claimed = set(pinned)
    band_regen: list[str] = []
    for ticker, _ts in regen_rows:
        if ticker not in claimed:
            claimed.add(ticker)
            band_regen.append(ticker)

    band_fresh: list[str] = []
    band_stale: list[str] = []
    for ticker, ts in memo_rows:
        if ticker in claimed:
            continue
        claimed.add(ticker)
        (band_fresh if ts >= cutoff else band_stale).append(ticker)

    # Pins ranked by research activity, untouched pins last (ticker
    # ascending). When the budget cannot hold the whole pin list, this is
    # what decides which pins are covered every run and which take turns.
    pinned.sort(key=lambda t: _by_recency_then_ticker((t, activity.get(t, _EPOCH))))

    return pinned, band_regen, band_fresh, band_stale, has_memo


def _split_budget(budget: int, live_available: int) -> tuple[int, int, int]:
    """Split an over-budget run into (reserved, guaranteed, rotating) slots.

    Order of claim matters: the live-research reserve is taken first (it is
    the only thing that can outrank a full pin list), the rotation next (it
    is the only thing that stops the tail starving), and the guaranteed
    prefix gets the remainder. At a budget too small to hold all three the
    higher-priority claims win, which is why `select_focus(budget=1)` with
    one regen request returns that one ticker and nothing else.
    """
    reserved = min(LIVE_RESEARCH_RESERVE, live_available, budget)
    rotating = min(ROTATING_SLOTS, budget - reserved)
    return reserved, budget - reserved - rotating, rotating


def _select(
    db: Session, *,
    budget: int,
    window_days: int,
    now: datetime,
    require_memo: bool,
    rotation_period_hours: int,
) -> FocusSelection:
    cutoff = now - timedelta(days=window_days)
    pinned, regen, fresh, stale, has_memo = _gather(db, cutoff=cutoff)

    withheld: dict[str, str] = {}
    if require_memo:
        # Bands 2 and 3 are *defined* by carrying a memo, so only 0 and 1
        # can lose anyone here. Held-back tickers leave the ranking
        # entirely rather than sinking to the bottom of it: a slot they
        # cannot act on is a slot wasted wherever they sit.
        for band, members in ((BAND_PINNED, pinned), (BAND_REGEN, regen)):
            for ticker in members:
                if ticker not in has_memo:
                    withheld[ticker] = _WITHHELD_LABEL[band]
        pinned = [t for t in pinned if t not in withheld]
        regen = [t for t in regen if t not in withheld]

    ranked = pinned + regen + fresh + stale
    bands = {t: BAND_PINNED for t in pinned}
    bands.update({t: BAND_REGEN for t in regen})
    bands.update({t: BAND_MEMO_FRESH for t in fresh})
    bands.update({t: BAND_MEMO_STALE for t in stale})

    if len(ranked) <= budget:
        # Everything that qualifies fits; nothing to ration and nothing to
        # rotate. This is the whole story for a small or quiet universe.
        return FocusSelection(
            tickers=tuple(ranked),
            reasons={t: bands[t] for t in ranked},
            withheld=withheld,
        )

    n_reserved, n_guaranteed, n_rotating = _split_budget(budget, len(regen))
    reserved = regen[:n_reserved]
    rest = [t for t in ranked if t not in set(reserved)]
    guaranteed = rest[:n_guaranteed]
    pool = rest[n_guaranteed:]

    offset = 0
    picked: list[str] = []
    if pool and n_rotating:
        offset = _rotation_step(now, rotation_period_hours) % len(pool)
        picked = (pool[offset:] + pool[:offset])[:n_rotating]
    else:
        n_rotating = 0

    chosen = set(reserved) | set(guaranteed) | set(picked)
    return FocusSelection(
        tickers=tuple(t for t in ranked if t in chosen),
        reasons={t: bands[t] for t in ranked if t in chosen},
        dropped=tuple(t for t in ranked if t not in chosen),
        withheld=withheld,
        degraded=False,
        rotation_offset=offset,
        rotation_pool_size=len(pool),
        rotation_slots=n_rotating,
    )


def _static_pins(budget: int) -> FocusSelection:
    """Fallback when the DB work raised: the curated top-10 from sp500.json.

    Kept in file order rather than sorted, because that file lists the pins
    by market cap — if the budget is smaller than the list, the biggest
    names are the ones worth keeping. This is deliberately a *different*
    policy from the healthy path's activity ranking, and not an oversight:
    research activity and memo presence both live in the database, and the
    database is precisely what just failed.
    """
    try:
        from ..seed_universe import load_universe_file
        pins = list(dict.fromkeys(t.upper() for t in load_universe_file().auto_update if t))
    except Exception:
        log.warning("static pin list unavailable; monitoring loop has nothing to run", exc_info=True)
        return FocusSelection(degraded=True)

    selected = pins[:budget]
    return FocusSelection(
        tickers=tuple(selected),
        reasons={t: BAND_DEGRADED for t in selected},
        dropped=tuple(pins[budget:]),
        degraded=True,
    )


def select_focus(
    *,
    budget: int,
    window_days: int = AUTO_REGEN_RECENCY_DAYS,
    now: datetime | None = None,
    db: Session | None = None,
    require_memo: bool = False,
    rotation_period_hours: int = 1,
) -> FocusSelection:
    """Choose up to `budget` tickers worth spending a monitoring run on.

    `require_memo` restricts selection to tickers the caller could actually
    act on — see the module docstring; the news loop sets it, the social
    loop does not. `rotation_period_hours` is the caller's run interval, so
    the rotation advances exactly one position per run rather than one per
    wall-clock hour; pass the same constant the scheduler gets.

    `now` and `db` are injectable so callers (and tests) can pin the clock
    and the database. Never raises and never touches the network: on any
    DB failure it degrades to the static pin list and says so.
    """
    now = now or datetime.utcnow()
    budget = max(0, int(budget))
    rotation_period_hours = max(1, int(rotation_period_hours))

    try:
        if db is not None:
            selection = _select(
                db, budget=budget, window_days=window_days, now=now,
                require_memo=require_memo,
                rotation_period_hours=rotation_period_hours,
            )
        else:
            from ..database import SessionLocal
            with SessionLocal() as session:
                selection = _select(
                    session, budget=budget, window_days=window_days, now=now,
                    require_memo=require_memo,
                    rotation_period_hours=rotation_period_hours,
                )
    except Exception as exc:
        log.warning(
            "research focus selection failed (%s); falling back to the static pin list",
            exc, exc_info=True,
        )
        selection = _static_pins(budget)

    if selection.withheld:
        # Not a budget decision, so it gets its own line: these tickers
        # ranked high enough to run and were held back because the loop
        # had no memo to patch. Five of the ten pins are in this state,
        # which is the single biggest thing this log explains.
        log.info(
            "research focus: withheld %d ranked ticker(s) with no memo to patch: %s",
            len(selection.withheld),
            ", ".join(f"{t} ({r})" for t, r in sorted(selection.withheld.items())),
        )
    if selection.dropped:
        # Constraint: a cap is never silent. The note carries this to
        # cron-health; this line carries it to the worker's logs. The
        # rotation numbers come along so the reader can tell a ticker that
        # is waiting its turn from one that is starved.
        log.info(
            "research focus: budget %d, rotating %d of %d @offset %d, "
            "dropped %d qualifying ticker(s): %s",
            budget, selection.rotation_slots,
            selection.rotation_pool_size, selection.rotation_offset,
            len(selection.dropped), ", ".join(selection.dropped),
        )
    return selection
